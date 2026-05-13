"""Tiny ncu target: one prefill + one decode step on the GB10 decoder.

Used as `ncu --set full -o profile python -m experiments.ncu_target ...`
"""
from __future__ import annotations

import argparse
import sys
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="bf16", choices=("bf16", "nvfp4"))
    ap.add_argument("--prompt-tokens", type=int, default=520)
    ap.add_argument("--steps", type=int, default=4)
    args = ap.parse_args()

    sys.path.insert(0, "models/qwen35_0p8b")
    import qwen35_megakernel_bf16_C  # noqa: F401
    from model import Decoder

    d = Decoder(model_name="Qwen/Qwen3.5-0.8B", backend=args.backend,
                verbose=False)
    prompt_ids = list(range(2, 2 + args.prompt_tokens))

    if args.backend == "bf16":
        d.reset()
        first = d.prefill(prompt_ids)
    else:
        d.reset()
        for t in prompt_ids[:-1]:
            d.step(int(t))
        first = d.step(int(prompt_ids[-1]))

    torch.cuda.synchronize()
    cur = int(first)
    for _ in range(args.steps):
        cur = int(d.step(cur))
    torch.cuda.synchronize()
    print(f"OK backend={args.backend} last_token={cur}")


if __name__ == "__main__":
    main()
