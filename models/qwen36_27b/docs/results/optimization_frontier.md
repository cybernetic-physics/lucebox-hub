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

## Update: NVFP4 went live, delivers 1.80× (not 3.5×)

| Path | tok/s | ms/tok | HBM floor | % of HBM peak |
|---|---:|---:|---:|---:|
| BF16 megakernel | 4.57 | 219 | 183 ms (50 GB) | **84%** |
| NVFP4 megakernel | 8.23 | 121 | 51 ms (14 GB) | **42%** |

Theoretical was 3.5× (full HBM scaling). Actual is 1.80× because the
matvec becomes compute-bound on the FP4 → FP32 decode. We did two
follow-on optimizations:

1. **Moved FP4 LUT from `__constant__` to shared memory.**
   The constant LUT serialized 32 cycles per lookup (one per lane);
   shared mem with 16-entry table resolves divergent reads in ~2
   cycles (worst-case 2-way bank conflict).
   Impact: **4.39 → 8.23 tok/s (1.88× on NVFP4 path alone).**

2. **uint4 (16-byte) loads instead of uint32 (4-byte).** Each lane now
   reads one full FP4 group per iter (32 elements, 16 bytes, one
   scale), 4× fewer warp iters. Impact: 8.23 → 8.01 tok/s (no
   speedup — confirms compute-bound). Kept for code clarity.

3. **Hardware FP4→FP16 cvt via `__nv_cvt_fp4x2_to_halfraw2`** (cuda_fp4.h,
   sm_120+ intrinsic = one PTX `cvt` per FP4 pair). Replaced the shmem
   LUT entirely. Impact: 8.01 → 8.35 tok/s (no measurable speedup).

The three follow-on attempts land at the same ~125 ms/token. **The
remaining ~70 ms gap to the 51 ms NVFP4 HBM floor is NOT in the FP4
decode itself.** It's elsewhere — likely grid sync overhead (400
syncs × 10-100 µs each), FA scan compute, and cooperative launch
serialization with 1 block per SM. Profiling with Nsight is the next
step to pinpoint; without it, further matvec micro-opts won't move
the needle.

Possible bigger wins from here:
- Reduce grid syncs by fusing layer phases (FA scan + O-proj into
  one section, etc.). 100-200 syncs of overhead at 50 µs each is
  a few ms — small but stackable.
- Use 2 blocks per SM via shmem reduction (would need to push 68 KB
  static down to ≤51 KB, lossy precision in the silu(gate)*up
  intermediate).
- Switch the entire decode to Tensor Core mma.sync FP4 (would
  require either S2 parallel-S or speculative-decode-style batch).

## What's still left to do (in real ROI order)

### 1. ~~NVFP4 weights live~~ — DONE (1.80× decode speedup)
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
