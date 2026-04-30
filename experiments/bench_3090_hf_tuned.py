"""Tuned HF + PyTorch baseline for Qwen3.5-0.8B on RTX 3090.

Two side-by-side measurements:
  (A) full HF rollout: AutoModel.generate at the canonical shape sweep
  (B) full HF training step: forward + autograd + AdamW
plus the "+compile" variants of each (torch.compile mode='reduce-overhead').

Compares to our megakernel numbers in
docs/results/qwen35_0p8b_3090.md so we have a defensible
"X times faster than tuned PyTorch" claim.

Run from repo root:
  CUDA_VISIBLE_DEVICES=1 .venv-3090/bin/python \\
      experiments/bench_3090_hf_tuned.py
"""
from __future__ import annotations

import argparse
import time
import torch


def _setup():
    """Apply the production-tuning knobs: TF32 matmul, bf16 default,
    cuDNN benchmark for stable shapes, no determinism overhead."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    # PEFT >= 0.19 unconditionally probes torchao via a strict version
    # check during LoRA Linear dispatch; on this box torchao is 0.9.0
    # so the check trips. Stub it (matches what rl_trainer.py does).
    def _no_torchao(*_a, **_kw): return False
    try:
        import peft.import_utils as _peft_iu
        _peft_iu.is_torchao_available = _no_torchao
    except Exception:
        pass
    try:
        import peft.tuners.lora.torchao as _peft_lora_torchao
        _peft_lora_torchao.is_torchao_available = _no_torchao
    except Exception:
        pass


def _time(fn, runs, warm):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / runs


def bench_rollout(prompt_lens, gen_tokens, runs, warm, use_compile):
    """HF AutoModel.generate at each prompt length."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print()
    print(f"== Rollout (compile={use_compile}) ==")
    print(f"{'P':>6} {'gen ms':>9} {'tok/s':>8}")
    print("-" * 28)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).to("cuda").eval()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")

    if use_compile:
        # generate uses model.forward many times; compile that.
        model.forward = torch.compile(model.forward, mode="reduce-overhead",
                                       dynamic=True)

    for P in prompt_lens:
        prompt = list(range(2, 2 + P))
        ids = torch.tensor(prompt, device="cuda").unsqueeze(0)

        def gen():
            with torch.no_grad():
                model.generate(
                    ids, max_new_tokens=gen_tokens, do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
        try:
            ms = _time(gen, runs=runs, warm=warm)
        except Exception as e:
            print(f"{P:>6} | error: {repr(e)[:70]}")
            continue
        tps = gen_tokens / (ms / 1000.0)
        print(f"{P:>6} {ms:>9.1f} {tps:>8.0f}")
    del model
    torch.cuda.empty_cache()


def bench_train_step(prompt_lens, target_len, runs, warm, use_compile):
    """HF + PEFT LoRA forward + backward + fused AdamW step."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    print()
    print(f"== Training step (compile={use_compile}) ==")
    print(f"{'P':>6} {'T':>4} {'step ms':>9} {'tok/s':>8}")
    print("-" * 35)
    base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).to("cuda").train()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    cfg = LoraConfig(
        r=8, lora_alpha=8, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, cfg).to("cuda", dtype=torch.bfloat16)
    model.train()
    if use_compile:
        model.forward = torch.compile(model.forward, mode="reduce-overhead",
                                       dynamic=True)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4, fused=True,
    )

    for P in prompt_lens:
        T = target_len
        prompt = list(range(10, 10 + P))
        target = list(range(100, 100 + T))
        full = torch.tensor(prompt + target, device="cuda").unsqueeze(0)
        target_ids = torch.tensor(target, device="cuda")

        def step():
            optim.zero_grad(set_to_none=True)
            out = model(input_ids=full, use_cache=False)
            logits = out.logits.float()
            predict = logits[0, P - 1: P - 1 + T]
            logp = torch.nn.functional.log_softmax(predict, dim=-1)
            loss = -logp.gather(1, target_ids.unsqueeze(1)).mean()
            loss.backward()
            optim.step()
        try:
            ms = _time(step, runs=runs, warm=warm)
        except torch.cuda.OutOfMemoryError:
            print(f"{P:>6} {T:>4} | OOM")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"{P:>6} {T:>4} | err: {repr(e)[:50]}")
            continue
        tps = (P + T) / (ms / 1000.0)
        print(f"{P:>6} {T:>4} {ms:>9.1f} {tps:>8.0f}")
    del model, base
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-lens", type=int, nargs="+",
                    default=[128, 512, 2048, 8192])
    ap.add_argument("--gen-tokens", type=int, default=32)
    ap.add_argument("--target-len", type=int, default=32)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--rollout-only", action="store_true")
    ap.add_argument("--train-only", action="store_true")
    ap.set_defaults(compile=False)
    args = ap.parse_args()

    _setup()

    if not args.train_only:
        bench_rollout(args.prompt_lens, args.gen_tokens,
                       args.runs, args.warm, use_compile=args.compile)

    if not args.rollout_only:
        bench_train_step(args.prompt_lens, args.target_len,
                          args.runs, args.warm, use_compile=args.compile)


if __name__ == "__main__":
    main()
