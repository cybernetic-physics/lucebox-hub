"""SGLang baseline for Qwen3.5-0.8B rollout on RTX 3090.

Same shape sweep as bench_3090_rollout.py — measures wall ms for
prefill + N gen tokens. Adds the third reference column to the perf
table in docs/results/qwen35_0p8b_3090.md.

Usage:
  CUDA_VISIBLE_DEVICES=1 .venv-3090/bin/python \\
      experiments/bench_3090_sglang.py \\
      --prompt-lens 128 512 2048 8192 16384 32768 --gen 32 --runs 3
"""
from __future__ import annotations

import argparse
import time

import torch  # imported but not used directly; ensures CUDA is available


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-lens", type=int, nargs="+",
                    default=[128, 512, 2048, 8192, 16384, 32768])
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warm", type=int, default=1)
    args = ap.parse_args()

    print("Loading SGLang engine for Qwen/Qwen3.5-0.8B (this can take ~30s)...")
    from sglang import Engine
    eng = Engine(
        model_path="Qwen/Qwen3.5-0.8B",
        tp_size=1,
        mem_fraction_static=0.7,
        disable_cuda_graph=True,   # SGLang's mamba/DN backend
        disable_radix_cache=True,  # ensure each call hits a fresh KV cache
    )

    sampling_params = {
        "temperature": 0.0,        # greedy, deterministic
        "max_new_tokens": args.gen,
    }

    print()
    print(f"{'P':>6} | {'sglang ms':>10} {'tok/s':>8}")
    print("-" * 30)

    for P in args.prompt_lens:
        # SGLang's Engine.generate accepts token IDs via input_ids on the
        # chat-style API and string text via the regular API. We use the
        # input_ids path so we control the prompt length precisely.
        prompt_ids = list(range(2, 2 + P))

        # Warm up.
        for _ in range(args.warm):
            try:
                eng.generate(input_ids=[prompt_ids], sampling_params=sampling_params)
            except TypeError:
                # Older SGLang signature — fall back to string prompts.
                eng.generate([str(prompt_ids[:1])], sampling_params=sampling_params)

        # Time.
        try:
            t0 = time.perf_counter()
            for _ in range(args.runs):
                out = eng.generate(input_ids=[prompt_ids],
                                    sampling_params=sampling_params)
            ms = (time.perf_counter() - t0) * 1000.0 / args.runs
        except Exception as e:
            print(f"{P:>6} | error: {repr(e)[:80]}")
            continue

        tps = args.gen / (ms / 1000.0) if ms > 0 else float('nan')
        print(f"{P:>6} | {ms:>10.1f} {tps:>8.0f}")

    eng.shutdown()


if __name__ == "__main__":
    main()
