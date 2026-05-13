"""GB10 kernel-cost breakdown without ncu.

We vary the parameters that change one substructure at a time, so the
slope tells us where time is going inside the persistent decode
megakernel and the prefill paths.

Probes:
  A. Decode step at increasing context lengths -> separates attention
     vs MLP/projection cost. Attention is O(L), KV reads are O(L),
     everything else is O(1) per step.
  B. Per-op time on a fixed-size prompt:
       - prefill_bf16 (eager, cuBLAS+custom kernels)
       - prefill_bf16_mega (single persistent kernel)
       - decode bf16  (one step)
       - decode nvfp4 (one step)
       - quantize_nvfp4_out (one matrix, illustrative)
  C. Long-context decode rate: tg at L = 0 / 1024 / 8192 / 32768 to
     confirm scaling.
"""
from __future__ import annotations

import argparse
import sys
import time

import torch


def cuda_time(fn, runs=20, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(runs):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / runs


def probe_decode_vs_context(backend: str, model_name: str):
    from model import Decoder
    print(f"\n=== A. Decode step vs context length (backend={backend}) ===")
    d = Decoder(model_name=model_name, backend=backend, verbose=False)
    for L in (0, 256, 1024, 4096, 16384, 32768):
        d.reset()
        if L > 0:
            # Prime the KV cache.
            ids = list(range(2, 2 + L))
            if backend == "bf16":
                first = d.prefill(ids)
            else:
                for t in ids[:-1]:
                    d.step(int(t))
                first = d.step(int(ids[-1]))
        else:
            first = d.step(2)
        cur = int(first)
        ms = cuda_time(lambda: setattr_then_step(d, cur), runs=20, warm=3)
        print(f"  L={L:>6}: {ms*1000:7.1f} us/step  ({1000/ms:6.1f} tok/s)")
    del d
    torch.cuda.empty_cache()


def setattr_then_step(d, cur):
    """One step call. We don't update `cur` because we just want timing,
    not generation correctness."""
    d.step(int(cur))


def probe_per_op(model_name: str, prompt_tokens: int = 512):
    print(f"\n=== B. Per-op cost (P={prompt_tokens}) ===")
    sys.path.insert(0, "models/qwen35_0p8b")
    from model import Decoder, MAX_SEQ_LEN  # noqa: F401

    # BF16 path
    d_bf16 = Decoder(model_name=model_name, backend="bf16", verbose=False)
    pids = list(range(2, 2 + prompt_tokens))
    d_bf16.reset()
    # prefill_bf16 (eager).
    ms_pf = cuda_time(lambda: d_bf16.prefill(pids), runs=5, warm=2)
    d_bf16.reset()
    _ = d_bf16.prefill(pids)
    ms_step_bf16 = cuda_time(lambda: d_bf16.step(2), runs=50, warm=10)
    print(f"  prefill_bf16 (eager) P={prompt_tokens}: {ms_pf:8.2f} ms "
          f"({prompt_tokens/ms_pf*1000:8.0f} t/s)")
    print(f"  decode_bf16 step (after P={prompt_tokens}): "
          f"{ms_step_bf16*1000:6.1f} us  ({1000/ms_step_bf16:6.1f} t/s)")
    del d_bf16
    torch.cuda.empty_cache()

    # NVFP4 path
    d_fp4 = Decoder(model_name=model_name, backend="nvfp4", verbose=False)
    d_fp4.reset()
    # NVFP4 has no fused prefill; we measure step-loop prefill time too.
    t0 = time.perf_counter()
    for t in pids[:-1]:
        d_fp4.step(int(t))
    _ = d_fp4.step(int(pids[-1]))
    torch.cuda.synchronize()
    ms_pf_fp4 = (time.perf_counter() - t0) * 1000
    ms_step_fp4 = cuda_time(lambda: d_fp4.step(2), runs=50, warm=10)
    print(f"  prefill_step_loop_nvfp4 P={prompt_tokens}: {ms_pf_fp4:8.2f} ms "
          f"({prompt_tokens/ms_pf_fp4*1000:8.0f} t/s)")
    print(f"  decode_nvfp4 step (after P={prompt_tokens}): "
          f"{ms_step_fp4*1000:6.1f} us  ({1000/ms_step_fp4:6.1f} t/s)")
    print(f"  nvfp4 vs bf16 step speedup: "
          f"{ms_step_bf16 / ms_step_fp4:.2f}x")
    del d_fp4
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--skip-context-sweep", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, "models/qwen35_0p8b")
    import qwen35_megakernel_bf16_C  # noqa: F401

    probe_per_op(args.model_name, args.prompt_tokens)
    if not args.skip_context_sweep:
        probe_decode_vs_context("bf16", args.model_name)
        probe_decode_vs_context("nvfp4", args.model_name)


if __name__ == "__main__":
    main()
