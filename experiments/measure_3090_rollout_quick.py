"""Quick measurement of current 3090-train HEAD rollout at S=32K.

Refreshes the "Ours" number in docs/results/qwen35_0p8b_3090.md.
"""
from __future__ import annotations

import sys
import time
import torch

sys.path.insert(0, "/home/freiza/lucebox-hub/models/qwen35_0p8b")
import qwen35_megakernel_bf16_C  # noqa: F401
from model import Decoder


def main():
    decoder = Decoder(verbose=False)
    for P in [128, 512, 2048, 8192, 16384, 32768]:
        prompt = list(range(2, 2 + P))
        # Warm
        decoder.reset()
        decoder.prefill(prompt)
        for _ in range(31):
            decoder.step(0)
        torch.cuda.synchronize()

        runs = []
        for _ in range(3):
            decoder.reset()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            tok = decoder.prefill(prompt)
            for _ in range(31):
                tok = decoder.step(int(tok))
            torch.cuda.synchronize()
            runs.append((time.perf_counter() - t0) * 1000)
        ms = min(runs)
        print(f"P={P:>5} | Ours rollout (best-of-3): {ms:>7.1f} ms")


if __name__ == "__main__":
    main()
