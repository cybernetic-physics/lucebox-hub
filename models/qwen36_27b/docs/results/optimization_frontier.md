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
   LUT entirely. Impact: 8.01 → 8.35 tok/s.

4. **`__hfma2` half2 SIMD multiply-add** in the dot product. Pre-convert
   activations bf16 → half once into registers, accumulate in half2
   (1 hfma2 = 2 half mul+adds = 1 PTX instruction), reduce to fp32 only
   at the end. Impact: 8.35 → 8.36 tok/s.

**Four independent inner-loop optimizations all land at ~8.3 tok/s.**
Suspicion: the bottleneck wasn't the matvec at all. **Built a
breakdown bench (cuda events around decode_qwen3x vs lm_head_argmax)
to find out.** Result:

  decode_qwen3x:  63.79 ms  (53%)
  lm_head + sync: **56.48 ms  (47%)**

The LM head fast-path kernel was gated on `MODEL_ID in (0, 1)`. NVFP4
uses MODEL_ID=3 → it was silently falling back to the Python
`hidden.float() @ lm_head.float().t()` matmul (~50 ms/token of needless
fp32 cast + cuBLAS). One-line fix to remap `3 → 1` for the kernel
dispatch (the kernel only cares about HIDDEN/VOCAB shape, not the
backend).

5. **LM head fast-path NVFP4 dispatch fix.** Impact:
   **8.36 → 13.40 tok/s (1.60× alone, 2.95× over the original S1d).**

### Current state — NVFP4 decode 2.97× BF16

| Path | tok/s | ms/tok | % HBM peak |
|---|---:|---:|---:|
| BF16 megakernel | 4.52 | 221 | 83% |
| **NVFP4 megakernel** | **13.18** | **76** | **65%** |

S7b (FP4 LM head) was wired and measured: NVFP4 + FP4 LM head =
13.18 tok/s, basically the same as NVFP4 + BF16 LM head (13.40). The
FP4 LM head ends up compute-bound at ~10 ms instead of HBM-bound at
~3 ms, so it doesn't reduce LM-head time vs the BF16 LM head's 10 ms
HBM-bound execution. Net: neutral. Kept the wireup since it costs no
extra memory (lm_head goes 2.5 GB → 0.7 GB, saving ~2 GB RAM) but
don't expect speed gains.

NVFP4 is now at 68% of its 51 ms HBM peak (3.5× theoretical). The
remaining 24 ms gap is plausibly grid sync overhead (~400/token)
and FA/DN scan compute. Each is small enough that further matvec or
single-kernel micro-opts won't move the needle much; the real lever
is now S2 (parallel-S prefill via Tensor Cores) or grid sync count
reduction at the layer-fusion level.

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
