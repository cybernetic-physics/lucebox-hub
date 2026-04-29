"""Training-loop test harness.

Three test modes:

  step1-grad     — run forward+backward once via the kernel-bwd path,
                   capture the resulting flat gradients, run the SAME
                   forward+backward via HF+PEFT autograd starting from
                   the same LoRA init, compare grads element-wise.
                   Localizes which projection has bad gradients.

  drift-trace    — run N optimizer steps via each path on the same
                   data, dump per-step (loss, lora-norm, grad-norm) to
                   pinpoint the step where drift starts.

  per-stage      — instrument run_layer_walking_bwd via a debug hook to
                   dump dh after every layer's bwd; compare against HF
                   autograd dh extracted via .register_full_backward_hook
                   on each HF layer.

Run any one with:
  CUDA_VISIBLE_DEVICES=1 .venv-3090/bin/python \\
      experiments/grad_harness.py <mode>
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, "/home/freiza/lucebox-hub/models/qwen35_0p8b/trainer")
sys.path.insert(0, "/home/freiza/lucebox-hub/models/qwen35_0p8b")

import torch  # noqa: E402

from rl_trainer import LoraMegakernelTrainer  # noqa: E402


def _datum(prompt_ids, target_ids):
    return {
        "model_input": {"chunks": [{"type": "input", "tokens": list(prompt_ids)}]},
        "loss_fn_inputs": {"target_tokens": list(target_ids)},
    }


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _lora_param_iter(peft_model):
    """Yield (name, param) for every PEFT LoRA A/B param. Keeps order so
    different runs (but same model) produce comparable lists."""
    for n, p in peft_model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            yield n, p


def _flatten_lora_grads(peft_model):
    """Return one fp32 vector concatenating every LoRA param's .grad."""
    chunks = []
    for _, p in _lora_param_iter(peft_model):
        if p.grad is None:
            chunks.append(torch.zeros(p.numel(), dtype=torch.float32, device=p.device))
        else:
            chunks.append(p.grad.detach().to(torch.float32).flatten())
    return torch.cat(chunks) if chunks else torch.zeros(0, device="cuda")


def _flatten_lora_params(peft_model):
    chunks = []
    for _, p in _lora_param_iter(peft_model):
        chunks.append(p.detach().to(torch.float32).flatten())
    return torch.cat(chunks) if chunks else torch.zeros(0, device="cuda")


def _grad_per_param(peft_model):
    """{name: grad_norm} for diagnostics."""
    out: dict[str, float] = {}
    for n, p in _lora_param_iter(peft_model):
        out[n] = float(p.grad.norm()) if p.grad is not None else 0.0
    return out


# -------------------------------------------------------------------------
# Mode 1: step-1 gradient comparison
# -------------------------------------------------------------------------


def _run_step1_grads(trainer, model_id, data, *, kernel_bwd: bool):
    """Run forward_backward but skip optim_step so we can read .grad.

    Returns (loss_mean, flat_grads_vector, per_param_grad_norms).
    """
    if kernel_bwd:
        os.environ["MEGAKERNEL_USE_KERNEL_BWD"] = "1"
    else:
        os.environ.pop("MEGAKERNEL_USE_KERNEL_BWD", None)

    out = trainer.forward_backward(model_id=model_id, data=data,
                                    loss_fn="cross_entropy")
    loss = out["metrics"]["loss:mean"]
    s = trainer._sessions[model_id]
    grads = _flatten_lora_grads(s.hf_model)
    norms = _grad_per_param(s.hf_model)
    return loss, grads, norms


def mode_step1_grad(args):
    print("=== mode: step1-grad ===")
    trainer = LoraMegakernelTrainer(verbose_loader=False)
    trainer.register_model(
        model_id="t",
        base_model="Qwen/Qwen3.5-0.8B",
        lora_rank=args.rank,
        train_mlp=True, train_attn=True, train_unembed=False,
        user_metadata=None,
    )

    if args.batch == 1:
        data = [_datum(list(range(10, 30)),  list(range(100, 110)))]
    else:
        data = [
            _datum(list(range(10, 30)),  list(range(100, 110))),
            _datum(list(range(50, 90)),  list(range(200, 220))),
        ]

    # Snapshot the LoRA params for verification.
    s = trainer._sessions["t"]
    init_params = _flatten_lora_params(s.hf_model).clone()

    # Path B FIRST (kernel-bwd). Why first: if path A were run first, its
    # autograd graph could leave hooks / inplace-version counters that
    # confuse the kernel path. Running B fresh isolates the kernel's
    # output.
    print(f"\n[B] kernel-bwd (batch={args.batch}):")
    loss_b, grad_b, norms_b = _run_step1_grads(trainer, "t", data, kernel_bwd=True)
    print(f"    loss = {loss_b:.6f}    |grad| = {float(grad_b.norm()):.4e}")
    print(f"    has NaN: {bool(torch.isnan(grad_b).any())}    "
          f"has Inf: {bool(torch.isinf(grad_b).any())}")

    # Verify params are unchanged.
    diff_params = (init_params - _flatten_lora_params(s.hf_model)).abs().max().item()
    print(f"    params unchanged: max|Δ| = {diff_params:.2e}")

    # Reset .grad.
    s.hf_model.zero_grad(set_to_none=True)

    # Path A: HF+PEFT autograd.
    print(f"\n[A] HF+PEFT autograd (batch={args.batch}):")
    loss_a, grad_a, norms_a = _run_step1_grads(trainer, "t", data, kernel_bwd=False)
    print(f"    loss = {loss_a:.6f}    |grad| = {float(grad_a.norm()):.4e}")

    # Compare grad vectors.
    if grad_a.numel() != grad_b.numel():
        print(f"    ERROR: grad vector size differs ({grad_a.numel()} vs {grad_b.numel()})")
        return

    cos = float(torch.dot(grad_a, grad_b)
                / (grad_a.norm() * grad_b.norm() + 1e-12))
    rel_l2 = float((grad_a - grad_b).norm() / (grad_a.norm() + 1e-12))
    max_diff = float((grad_a - grad_b).abs().max())

    print(f"\nFLAT-GRAD COMPARISON (HF vs kernel-bwd, step 1):")
    print(f"    cos similarity   = {cos:.6f}")
    print(f"    relative L2 diff = {rel_l2*100:.3f}%")
    print(f"    max element diff = {max_diff:.4e}")

    # Per-param diagnostics: which projections diverge most.
    print(f"\nPER-PARAM grad-norm comparison (HF vs kernel):")
    print(f"    {'name':<60}  {'HF |g|':>10}  {'K |g|':>10}  {'rel%':>7}")
    rows = []
    for name in norms_a:
        ga = norms_a[name]
        gb = norms_b.get(name, 0.0)
        rel = abs(ga - gb) / (ga + 1e-12) * 100
        rows.append((rel, name, ga, gb))
    rows.sort(reverse=True)
    for rel, name, ga, gb in rows[:20]:
        nm = name if len(name) <= 60 else "…" + name[-58:]
        print(f"    {nm:<60}  {ga:>10.3e}  {gb:>10.3e}  {rel:>6.1f}%")


# -------------------------------------------------------------------------
# Mode 2: drift trace
# -------------------------------------------------------------------------


def mode_drift_trace(args):
    print("=== mode: drift-trace ===")

    def trace_path(label, kernel_bwd):
        print(f"\n[{label}]")
        trainer = LoraMegakernelTrainer(verbose_loader=False)
        trainer.register_model(
            model_id="t",
            base_model="Qwen/Qwen3.5-0.8B",
            lora_rank=args.rank,
            train_mlp=True, train_attn=True, train_unembed=False,
            user_metadata=None,
        )
        if kernel_bwd:
            os.environ["MEGAKERNEL_USE_KERNEL_BWD"] = "1"
        else:
            os.environ.pop("MEGAKERNEL_USE_KERNEL_BWD", None)

        if args.batch == 1:
            data = [_datum(list(range(10, 30)),  list(range(100, 110)))]
        else:
            data = [
                _datum(list(range(10, 30)),  list(range(100, 110))),
                _datum(list(range(50, 90)),  list(range(200, 220))),
            ]

        s = trainer._sessions["t"]
        for step in range(args.steps):
            out = trainer.forward_backward(model_id="t", data=data, loss_fn="cross_entropy")
            grad_norm = float(_flatten_lora_grads(s.hf_model).norm())
            param_norm = float(_flatten_lora_params(s.hf_model).norm())
            trainer.optim_step(model_id="t",
                                adam_params={"lr": args.lr, "betas": (0.9, 0.999),
                                             "eps": 1e-8, "wd": 0.01})
            param_norm_after = float(_flatten_lora_params(s.hf_model).norm())
            print(f"  step {step+1}  loss={out['metrics']['loss:mean']:.6f}  "
                  f"|grad|={grad_norm:.4e}  |Δparam|={(param_norm_after - param_norm):+.4e}  "
                  f"|param|={param_norm_after:.4e}")
        del trainer

    trace_path("HF+PEFT", kernel_bwd=False)
    trace_path("kernel-bwd", kernel_bwd=True)


# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# Mode 3: stability — repeat step1 grad many times, report stable vs NaN
# -------------------------------------------------------------------------


def mode_stability(args):
    print(f"=== mode: stability  (N={args.iters} iterations) ===")
    trainer = LoraMegakernelTrainer(verbose_loader=False)
    trainer.register_model(
        model_id="t",
        base_model="Qwen/Qwen3.5-0.8B",
        lora_rank=args.rank,
        train_mlp=True, train_attn=True, train_unembed=False,
        user_metadata=None,
    )
    if args.batch == 1:
        data = [_datum(list(range(10, 30)),  list(range(100, 110)))]
    else:
        data = [
            _datum(list(range(10, 30)),  list(range(100, 110))),
            _datum(list(range(50, 90)),  list(range(200, 220))),
        ]

    s = trainer._sessions["t"]

    # Establish HF reference grad (deterministic; one run suffices).
    os.environ.pop("MEGAKERNEL_USE_KERNEL_BWD", None)
    s.hf_model.zero_grad(set_to_none=True)
    out_a = trainer.forward_backward(model_id="t", data=data, loss_fn="cross_entropy")
    grad_hf = _flatten_lora_grads(s.hf_model).clone()
    print(f"HF reference: loss = {out_a['metrics']['loss:mean']:.6f}    "
          f"|grad| = {float(grad_hf.norm()):.4e}")

    # Repeat kernel-bwd N times, comparing each to HF.
    os.environ["MEGAKERNEL_USE_KERNEL_BWD"] = "1"
    nan_count = 0
    bad_count = 0      # cos < 0.9 (defined as failed)
    good_count = 0     # cos >= 0.9
    cos_history = []
    for i in range(args.iters):
        s.hf_model.zero_grad(set_to_none=True)
        out_b = trainer.forward_backward(model_id="t", data=data,
                                          loss_fn="cross_entropy")
        grad_k = _flatten_lora_grads(s.hf_model)
        if not torch.isfinite(grad_k).all():
            nan_count += 1
            cos_history.append(float("nan"))
            print(f"  iter {i+1:3d}: NaN")
            continue
        cos = float(torch.dot(grad_hf, grad_k)
                    / (grad_hf.norm() * grad_k.norm() + 1e-12))
        cos_history.append(cos)
        knorm = float(grad_k.norm())
        ratio = knorm / float(grad_hf.norm())
        if cos < 0.9 or ratio > 5.0 or ratio < 0.2:
            bad_count += 1
            print(f"  iter {i+1:3d}: cos={cos:.4f}  |k_grad|={knorm:.3e}  "
                  f"ratio={ratio:.2f}x  *** FAIL ***")
        else:
            good_count += 1
            if i < 5 or i % 10 == 0:
                print(f"  iter {i+1:3d}: cos={cos:.4f}  |k_grad|={knorm:.3e}  "
                      f"ratio={ratio:.2f}x")

    print()
    print(f"Stability summary over {args.iters} iterations:")
    print(f"  good (cos>=0.9 and 0.2 <= ratio <= 5.0): {good_count} "
          f"({100*good_count/args.iters:.1f}%)")
    print(f"  bad gradient (large/wrong direction)   : {bad_count} "
          f"({100*bad_count/args.iters:.1f}%)")
    print(f"  NaN gradient                           : {nan_count} "
          f"({100*nan_count/args.iters:.1f}%)")
    finite_cos = [c for c in cos_history if c == c]  # filter NaN
    if finite_cos:
        finite_cos.sort()
        print(f"  cos similarity range (good runs only): "
              f"min={min(finite_cos):.4f}  median={finite_cos[len(finite_cos)//2]:.4f}  "
              f"max={max(finite_cos):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["step1-grad", "drift-trace", "stability"])
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    if args.mode == "step1-grad":
        mode_step1_grad(args)
    elif args.mode == "drift-trace":
        mode_drift_trace(args)
    elif args.mode == "stability":
        mode_stability(args)


if __name__ == "__main__":
    main()
