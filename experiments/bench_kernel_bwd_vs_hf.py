"""Time forward_backward + optim_step end-to-end on the same batch via:
   (A) HF + PEFT autograd (default)
   (B) kernel-bwd path  (MEGAKERNEL_USE_KERNEL_BWD=1)
across shapes the trainer is likely to see in production.

Single-item heterogeneous batches force the kernel branch (HF can't
batch a 1-sample call any harder than the kernel path can). For
multi-item homogeneous batches HF batches all items into one forward,
so the two paths solve different problems — we measure both.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import torch

sys.path.insert(0, "/home/freiza/lucebox-hub/models/qwen35_0p8b/trainer")
sys.path.insert(0, "/home/freiza/lucebox-hub/models/qwen35_0p8b")


def _datum(prompt, target):
    return {
        "model_input": {"chunks": [{"type": "input", "tokens": list(prompt)}]},
        "loss_fn_inputs": {"target_tokens": list(target)},
    }


def time_step(trainer, model_id, data, runs=3, warm=1):
    for _ in range(warm):
        trainer.forward_backward(model_id=model_id, data=data, loss_fn="cross_entropy")
        trainer.optim_step(model_id=model_id, adam_params={
            "lr": 1e-4, "betas": (0.9, 0.999), "eps": 1e-8, "wd": 0.01})
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        trainer.forward_backward(model_id=model_id, data=data, loss_fn="cross_entropy")
        trainer.optim_step(model_id=model_id, adam_params={
            "lr": 1e-4, "betas": (0.9, 0.999), "eps": 1e-8, "wd": 0.01})
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--prompt-lens", type=int, nargs="+", default=[64, 256, 1024])
    ap.add_argument("--target-len", type=int, default=32)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    print(f"{'P':>5} {'T':>4} | {'HF+PEFT (ms)':>13} {'kernel-bwd (ms)':>16} "
          f"{'speedup':>8}")
    print("-" * 60)

    for P in args.prompt_lens:
        T = args.target_len

        # Single-item batch -> kernel-bwd branch is taken.
        from rl_trainer import LoraMegakernelTrainer
        trainer_a = LoraMegakernelTrainer(verbose_loader=False)
        trainer_a.register_model(
            model_id="t",
            base_model="Qwen/Qwen3.5-0.8B",
            lora_rank=args.rank,
            train_mlp=True, train_attn=True, train_unembed=False,
            user_metadata=None,
        )
        data = [_datum(list(range(10, 10 + P)), list(range(100, 100 + T)))]

        os.environ.pop("MEGAKERNEL_USE_KERNEL_BWD", None)
        ms_hf = time_step(trainer_a, "t", data, runs=args.runs)

        # Reset to keep state comparable.
        trainer_a.unload_model(model_id="t")
        trainer_a.register_model(
            model_id="t",
            base_model="Qwen/Qwen3.5-0.8B",
            lora_rank=args.rank,
            train_mlp=True, train_attn=True, train_unembed=False,
            user_metadata=None,
        )
        os.environ["MEGAKERNEL_USE_KERNEL_BWD"] = "1"
        ms_k = time_step(trainer_a, "t", data, runs=args.runs)

        sp = ms_hf / ms_k if ms_k > 0 else float('nan')
        print(f"{P:>5} {T:>4} | {ms_hf:>13.1f} {ms_k:>16.1f} {sp:>7.2f}x")
        del trainer_a
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
