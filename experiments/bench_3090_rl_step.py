"""Bench full RL step (rollout + train) on RTX 3090.

For each S, measure:
  - sample N tokens (LoraMegakernelTrainer.sample) — wall ms
  - forward_backward + optim_step — wall ms
  - HF baseline: same model.generate + autograd loss + AdamW

Usage (from repo root):
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=models/qwen35_0p8b \
        .venv-3090/bin/python experiments/bench_3090_rl_step.py \
        --prompt-lens 128 512 2048 8192 --gen 32 --runs 2
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, "/home/freiza/lucebox-hub/models/qwen35_0p8b")
sys.path.insert(0, "/home/freiza/lucebox-hub/models/qwen35_0p8b/trainer")

import torch  # noqa: E402

from rl_trainer import LoraMegakernelTrainer  # noqa: E402


def time_fn(fn, runs=2, warm=1):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / runs


def make_datum(prompt, target):
    return {
        "model_input": {"chunks": [{"type": "input", "tokens": list(prompt)}]},
        "loss_fn_inputs": {"target_tokens": list(target)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-lens", type=int, nargs="+",
                    default=[128, 512, 2048, 8192])
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--warm", type=int, default=1)
    args = ap.parse_args()

    print("Loading trainer...")
    trainer = LoraMegakernelTrainer(verbose_loader=False)
    trainer.register_model(
        model_id="bench",
        base_model="Qwen/Qwen3.5-0.8B",
        lora_rank=8,
        train_mlp=True,
        train_attn=True,
        train_unembed=False,
        user_metadata=None,
    )

    print()
    print(f"{'P':>6} | {'sample ms':>10} {'fwd+bwd ms':>12} {'optim ms':>9} {'step total':>11}")
    print("-" * 60)

    for P in args.prompt_lens:
        prompt = list(range(2, 2 + P))
        target = list(range(100, 100 + 32))

        try:
            sample_ms = time_fn(
                lambda: trainer.sample(
                    prompt_tokens=prompt, max_tokens=args.gen,
                    num_samples=1, prompt_logprobs=False,
                    topk_prompt_logprobs=0, model_id="bench"),
                runs=args.runs, warm=args.warm)
        except Exception as e:
            print(f"{P:>6} | sample failed: {repr(e)[:80]}")
            continue

        try:
            data = [make_datum(prompt, target)]
            def fwd_bwd():
                trainer.forward_backward(model_id="bench", data=data,
                                          loss_fn="cross_entropy")
            fb_ms = time_fn(fwd_bwd, runs=args.runs, warm=args.warm)
        except Exception as e:
            fb_ms = float("nan")
            print(f"{P:>6} | fwd_bwd failed: {repr(e)[:80]}")

        try:
            def optim():
                trainer.optim_step(model_id="bench",
                                    adam_params={"lr": 1e-4, "betas": (0.9, 0.999), "eps": 1e-8, "wd": 0.01})
            opt_ms = time_fn(optim, runs=args.runs, warm=args.warm)
        except Exception as e:
            opt_ms = float("nan")
            print(f"{P:>6} | optim failed: {repr(e)[:80]}")

        total = sample_ms + fb_ms + opt_ms
        print(f"{P:>6} | {sample_ms:>10.1f} {fb_ms:>12.1f} {opt_ms:>9.1f} {total:>11.1f}")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
