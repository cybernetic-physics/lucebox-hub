"""Probe whether the kernel-bwd forward_backward path is CUDA-graph-capturable.

Calls `_forward_backward_kernel_path` (the inner method that does the actual
kernel work) directly, bypassing the public `forward_backward` wrapper that
calls .item() / .cpu() for Python-side metrics. Those metrics calls force
CPU syncs that break graph capture; they belong OUTSIDE the captured region.

Reports specifically WHAT fails (sync-inside-capture, dynamic shape,
allocator pool issue, etc.) so the next step is concrete.
"""
from __future__ import annotations

import os
import sys
import torch

os.environ["MEGAKERNEL_USE_KERNEL_BWD"] = "1"

sys.path.insert(0, "/home/freiza/lucebox-hub/models/qwen35_0p8b/trainer")
sys.path.insert(0, "/home/freiza/lucebox-hub/models/qwen35_0p8b")


def main():
    from rl_trainer import LoraMegakernelTrainer
    import lora_pack
    from lora_megakernel_step import kernel_loss_autograd, load_base_model
    from lora_layer_bwd_skel import run_layer_walking_bwd

    P, T = 256, 32
    trainer = LoraMegakernelTrainer(verbose_loader=False)
    trainer.register_model(
        model_id="t",
        base_model="Qwen/Qwen3.5-0.8B",
        lora_rank=8,
        train_mlp=True, train_attn=True, train_unembed=False,
        user_metadata=None,
    )
    data = [{
        "model_input": {"chunks": [{"type": "input", "tokens": list(range(10, 10 + P))}]},
        "loss_fn_inputs": {"target_tokens": list(range(100, 100 + T))},
    }]
    adam = {"lr": 1e-4, "betas": (0.9, 0.999), "eps": 1e-8, "wd": 0.01}

    s = trainer._sessions["t"]
    handle = trainer._kernel_base_handle if hasattr(trainer, "_kernel_base_handle") \
        else load_base_model(trainer.BASE_MODEL, verbose=False)
    if not hasattr(trainer, "_kernel_base_handle"):
        trainer._kernel_base_handle = handle

    prompt = torch.tensor(list(range(10, 10 + P)), dtype=torch.int32, device="cuda")
    targets = torch.tensor(list(range(100, 100 + T)), dtype=torch.int32, device="cuda")

    def kernel_step():
        """Inline forward_backward kernel work — mirrors
        rl_trainer._forward_backward_kernel_path but bypasses the .item() /
        .cpu() metrics wrapping that's hostile to graph capture."""
        lora_flat = lora_pack.pack_peft_to_flat(s.hf_model, s.lora_rank)
        out = kernel_loss_autograd(
            handle=handle, prompt_tokens=prompt, target_tokens=targets,
            lora_flat=lora_flat, lora_rank=s.lora_rank,
            lora_scaling=s.lora_scaling, hf_model=s.hf_model,
        )
        flat_grads = run_layer_walking_bwd(
            grad_h_pre_norm=out["grad_h_pre_norm"],
            saves=out["saves"], lora_flat=lora_flat,
            final_norm_weight=handle.final_norm_weight,
            hf_model=s.hf_model, lora_rank=s.lora_rank,
            lora_scaling=s.lora_scaling,
            fa_k_cache=out["scratch"]["fa_k_cache"],
            fa_v_cache=out["scratch"]["fa_v_cache"],
        )
        lora_pack.scatter_flat_grads_to_peft(s.hf_model, flat_grads, accumulate=True)
        s.hf_optimizer.step()
        s.hf_optimizer.zero_grad(set_to_none=True)

    print("Warmup (3 iters)...")
    for _ in range(3):
        kernel_step()
    torch.cuda.synchronize()
    print("  OK")

    # Capture-stream warmup (required by torch.cuda.graph).
    print("Attempting graph capture of kernel forward_backward + optim_step...")
    cap_s = torch.cuda.Stream()
    cap_s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(cap_s):
        for _ in range(2):
            kernel_step()
    torch.cuda.current_stream().wait_stream(cap_s)
    torch.cuda.synchronize()
    print("  Capture-stream warmup OK")

    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            kernel_step()
        print("  Graph capture SUCCEEDED")
    except Exception as e:
        msg = repr(e)
        print(f"  Graph capture FAILED: {msg[:300]}")
        # Common diagnostic: look for known blockers
        if "synchronize" in msg.lower():
            print("  → cuda.synchronize() called inside capture (rl_trainer.py:508)")
        if "stream" in msg.lower():
            print("  → stream-related issue; may need to align fla/Triton kernels to capture stream")
        if "allocate" in msg.lower() or "alloc" in msg.lower():
            print("  → allocator issue; may need static-buffer refactor")
        return 1

    # Try replay
    print("Replaying graph...")
    try:
        g.replay()
        torch.cuda.synchronize()
        print("  Replay OK")
    except Exception as e:
        print(f"  Replay FAILED: {repr(e)[:200]}")
        return 1

    # Correctness: replay graph + reset state, run eager step, compare
    # one LoRA param's value drift after one extra step.
    p_sample = next(p for p in s.hf_model.parameters() if p.requires_grad)
    p_before = p_sample.detach().clone()
    g.replay()
    torch.cuda.synchronize()
    p_after_graph = p_sample.detach().clone()
    # Restore params to before-state (load original) and run eager step
    p_sample.data.copy_(p_before)
    # Also reset optim state by zeroing grads (params copied back; optim
    # state will diverge slightly, but the step delta should match).
    s.hf_optimizer.zero_grad(set_to_none=True)
    kernel_step()
    torch.cuda.synchronize()
    p_after_eager = p_sample.detach().clone()
    diff = (p_after_graph - p_after_eager).abs()
    rel = diff.max().item() / max(p_after_eager.abs().max().item(), 1e-8)
    print(f"  Param drift graph vs eager: max|Δ|={diff.max().item():.2e}  "
          f"rel-to-max={rel:.3%}")
    if rel > 0.05:
        print(f"  WARN: graph and eager diverge >5% — may indicate stale buffer")

    # Time the replay vs eager
    import time
    runs = 5
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        g.replay()
    torch.cuda.synchronize()
    graph_ms = (time.perf_counter() - t0) * 1000.0 / runs

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        kernel_step()
    torch.cuda.synchronize()
    eager_ms = (time.perf_counter() - t0) * 1000.0 / runs

    print()
    print(f"  Eager:  {eager_ms:.1f} ms/step")
    print(f"  Graph:  {graph_ms:.1f} ms/step")
    print(f"  Speedup: {eager_ms / graph_ms:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
