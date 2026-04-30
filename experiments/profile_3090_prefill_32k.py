"""Per-kernel profile of S=32K prefill on the current 3090-train HEAD.

Goal: identify the actual hotspot now that dn_chunked_3090 is wired in.
The doc's "743 ms in pf_deltanet_recurrence_vsplit_prepped" line is from
before commit dba22e3 — at HEAD, S>=128 routes to dn_chunked_3090 on
sm_86 by default, so the V-split recurrence is no longer in the hot path.

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=models/qwen35_0p8b \\
      .venv-3090/bin/python experiments/profile_3090_prefill_32k.py
"""
from __future__ import annotations

import sys
import time
import torch

sys.path.insert(0, "/home/freiza/lucebox-hub/models/qwen35_0p8b")
import qwen35_megakernel_bf16_C  # noqa: F401
from model import Decoder


def main():
    P = 32768
    decoder = Decoder(verbose=False)
    prompt = list(range(2, 2 + P))

    # Warmup
    decoder.reset()
    decoder.prefill(prompt)
    torch.cuda.synchronize()

    # Wall-clock baseline
    decoder.reset()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    decoder.prefill(prompt)
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1000
    print(f"S={P} prefill wall: {wall_ms:.1f} ms")

    # Per-kernel breakdown
    decoder.reset()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        decoder.prefill(prompt)
        torch.cuda.synchronize()

    print()
    print("Top CUDA kernels by self-time:")
    print("-" * 88)
    events = prof.key_averages()
    # Filter: only events with cuda_time_total > 0
    cuda_events = [e for e in events if e.self_device_time_total > 0]
    cuda_events.sort(key=lambda e: e.self_device_time_total, reverse=True)
    total_us = sum(e.self_device_time_total for e in cuda_events)
    for e in cuda_events[:20]:
        ms = e.self_device_time_total / 1000.0
        share = 100.0 * e.self_device_time_total / max(total_us, 1)
        name = (e.key[:65] + "..") if len(e.key) > 67 else e.key
        print(f"{ms:>8.2f} ms  {share:>5.1f}%  ({e.count:>4}×)  {name}")
    print("-" * 88)
    print(f"sum self_device_time = {total_us/1000:.1f} ms")


if __name__ == "__main__":
    main()
