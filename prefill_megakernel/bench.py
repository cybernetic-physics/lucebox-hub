"""Walltime comparison: single-dispatch megakernel vs existing multi-kernel prefill.

Both run the full Qwen 3.5-0.8B forward pass over S tokens with random
bf16 weights. The existing prefill issues ~20 launches per layer × 24
layers ≈ 480 dispatches per forward; the megakernel issues 1.
"""

from __future__ import annotations

import sys
import time

import torch

sys.path.insert(0, "/root/lucebox-hub-b200-train/megakernel")
sys.path.insert(0, "/root/lucebox-hub-b200-train/prefill_megakernel")

import qwen35_megakernel_bf16_C  # noqa: F401
from model import VOCAB, PrefillMegakernel
from smoke import random_weights
from test_vs_existing_prefill import run_existing_prefill


def bench(S, n_warmup=3, n_iter=10):
    torch.manual_seed(0)
    w = random_weights()
    tokens = torch.randint(0, VOCAB, (S,), dtype=torch.int32, device="cuda")

    mega = PrefillMegakernel(w)

    for _ in range(n_warmup):
        run_existing_prefill(w, tokens)
        mega.forward(tokens)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        run_existing_prefill(w, tokens)
    torch.cuda.synchronize()
    ex_ms = (time.perf_counter() - t0) * 1000.0 / n_iter

    t0 = time.perf_counter()
    for _ in range(n_iter):
        mega.forward(tokens)
    torch.cuda.synchronize()
    mk_ms = (time.perf_counter() - t0) * 1000.0 / n_iter

    return ex_ms, mk_ms


def main():
    print(f"{'S':>5}  {'existing (ms)':>15}  {'megakernel (ms)':>17}  {'speedup':>8}")
    for S in [16, 32, 64, 128, 256]:
        ex, mk = bench(S)
        print(f"{S:>5}  {ex:>15.2f}  {mk:>17.2f}  {ex/mk:>7.2f}x")


if __name__ == "__main__":
    main()
