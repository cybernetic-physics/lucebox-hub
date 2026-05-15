# Optimization frontier — Qwen3.6-27B on GB10

Snapshot after the session-2 round of work. **Decode is HBM-bound at
~88% of theoretical peak** on the BF16 path; further single-token
gains require quantization, not kernel tuning.

## Where wall-time goes per decode step (185 ms total)

| component | time | % |
|---|---:|---:|
| HBM read of 48 GB of BF16 weights (273 GB/s peak) | ~175 ms | 95% |
| FA scan compute (16 layers) | ~3 ms | 1.6% |
| DN scan compute (48 layers, V/QK split, recurrence) | ~3 ms | 1.6% |
| RMSnorm + RoPE + activation funcs | ~2 ms | 1.1% |
| LM head argmax (BF16 kernel, S7a) | ~10 ms | 5.4% |
| grid.sync overhead (~450 calls × ~1-5 µs) | ~1-2 ms | 1% |
| **measured total** | **185 ms** | |

The 88% HBM efficiency is **already optimal for one-token-at-a-time
BF16 inference**. The only paths to faster decode are:

## What's actually left to do (in real ROI order)

### 1. NVFP4 weights live — 3.5× decode speedup (~12 tok/s)
- All wireup is committed (S1a-d). Kernel paths, layer functions,
  192 B `LayerWeights` union with NVFP4 variants, model_id=3
  dispatch, runtime `backend="nvfp4"` flag, optimal-MSE quantizer
  imported from 0.8B, disk cache (`$HF_HOME/qwen3x_nvfp4_27b_cache.pt`).
- **Blocker**: first quantization takes 30+ minutes (per-projection
  optimal-MSE scale search). Subsequent runs hit the cache in seconds.
- Action: someone runs `test/test_s1e_nvfp4_vs_hf.py` once to populate
  the cache + verify correctness vs HF.

### 2. S2 parallel-S prefill — eliminates HF dependency for prefill
- We have `prefill_via_hf` (~50× speedup, requires HF model in memory).
- A native parallel-S prefill matches HF speed without the 50 GB
  HF-model alloc.
- Plan: `megakernel/PREFILL_S2_PLAN.md` (port from 0.8B's 1100-line
  `prefill_megakernel.cu`).
- Effort: 3-5 days of dense WMMA + chunked-DN-scan work.

### 3. Real MTP head — 3× decode via speculative
- HF discards `mtp.*` keys at load time. Need to bypass HF and load
  safetensors manually for the MTP head — a few hours of work.
- Probe script in place at `test/test_s5_mtp_probe.py`.

### 4. NVFP4 KV cache — enables long context
- Doesn't help decode latency much (KV reads are small vs weights).
- Drops KV memory 3.5× — critical at S ≥ 32k.
- Plan: `megakernel/KV_S3_PLAN.md`. Effort: 2-3 days, reuses
  0.8B helpers.

### 5. cuBLASLt FP4 LM head (S7b)
- Drops the 10 ms LM head step to ~3 ms. Useful but 4% of total.
- Needs cuBLASLt FP4 hookup at the 27B vocab × hidden shape.

## What we tried this session that didn't pay

- **num_blocks tuning**: GB10 has 48 SMs and we're already at 1-2
  blocks per SM. Tested LM head at 32/48/64/96 → all within 1%.
- **Async decode chain (device-side token passing)**: would save
  per-token host syncs (~50 µs each). On 185 ms steps, that's
  <0.05%. Engineering cost not worth it.
- **CUDA Graph capture**: requires the device-side token passing
  above. Same <1% argument.
- **Final RMSnorm fusion into LM head**: saves one kernel launch
  + 40 KB HBM round-trip. ~10 µs / 185 ms = 0.005%.
- **Inside-the-decode-kernel multi-row warps**: shmem-bound region
  is tiny; not a bottleneck.
- **Larger HBM transactions**: already 16-byte uint4 loads via
  `ld.global.L1::no_allocate.v4.b32`.

The 12% gap from the HBM floor (175 ms theoretical, 185 ms measured)
covers FA/DN scan compute, grid sync overhead, and LM head — none of
which are individually big enough to chase below 5% returns.
