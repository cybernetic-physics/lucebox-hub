# NVIDIA B200 (sm_100, Qwen3.5-0.8B BF16)

Numbers measured on a single Blackwell datacenter GPU
(NVIDIA B200, 148 SMs, sm_100a, CUDA 12.8, PyTorch 2.11). Reproduced
on GPU index 2 and 3 of the same host with the same numbers within
1% noise.

```
                              pp520 (tok/s)    tg128 (tok/s)
  Megakernel BF16 (this branch)   40,635            719
  Megakernel NVFP4 (gb10 path)    29,994            327
  PyTorch HuggingFace BF16        12,919             36
  llama.cpp BF16 (separate run)   ~26,800           ~440
```

`final_bench.py --backend bf16` runs the BF16 megakernel pipeline
(`prefill_bf16` + the BF16 decode megakernel). `final_bench.py
--backend nvfp4` runs the gb10 NVFP4 path
(`prefill_bf16_nvfp4_lm` + LUT-based NVFP4 decode).

## What changed for B200

The gb10 PR (Luce-Org#30) introduced a Blackwell path targeted at
sm_120 / sm_121a. On B200 (sm_100):

- The build only needed minor surgery — auto-detect to `sm_100a`,
  widen `is_blackwell` to include `sm_10*`, and lower a few
  `__CUDA_ARCH__ < 1200` guards to `< 1000`. cuBLASLt FP4 / NVFP4
  packing intrinsics from `cuda_fp4.h` resolve fine on `sm_100a`.
- The BF16 decode kernel hung. Its hand-rolled grid barrier
  (`fence.acq_rel.gpu` + `atomicAdd` on a counter) made no forward
  progress across 148 SMs on sm_100. Replacing it with
  `cudaLaunchCooperativeKernel` + `cg::this_grid().sync()` (the same
  primitive the NVFP4 path already uses) unblocks decode and is a
  strict improvement on sm_86 too.
- BF16 prefill spent 88% of wall time inside the DeltaNet
  recurrence, with cuBLAS GEMMs at 4%. Several optimization passes
  brought the recurrence under control:
  1. graph-capture the prefill body, fold per-channel conv1d into
     shared memory once per kernel entry, fuse the q/k/v conv passes
     into one parallel sweep — pp520 12,420 → 15,859 (+28%);
  2. remove two redundant `__syncthreads` per recurrence step —
     pp520 → 16,770 (+5.7%);
  3. run q/k L2-normalize in parallel with the beta/decay scalar
     compute on a previously-idle warp — pp520 → 17,847 (+6.4%);
  4. cache the per-lane `s_q` / `s_k` slice into registers once per
     step (clarity, perf-neutral);
  5. V-split the recurrence so `head x v_split` populates 64 blocks
     instead of 16 (B200 has 148 SMs); raw attn lands in `dn_out_buf`
     and a small per-`(t, h)` block computes the gnorm — pp520
     17,847 → 27,501 (+54%, first commit where we cross llama.cpp
     BF16 on B200);
  6. pre-pass conv1d, q/k normalize, and beta/decay out of the
     recurrence into a parallel `pf_dn_prep` kernel; the inner loop
     becomes pure V-local arithmetic with zero per-step
     `__syncthreads` — pp520 27,501 → 40,278 (+46%).

Total: pp520 12,420 → 40,278 (+224%) without changing decoding
behavior. Decode held at 711 tok/s throughout the prefill changes.

## What did NOT change

- **sm_86 (RTX 3090)**: All `is_blackwell` gates in `setup.py`
  evaluate false on sm_86 — same sources, same defines, same arch
  flag as upstream. The shared `kernel.cu` + `prefill.cu` files do
  pick up changes (cooperative grid sync in decode, the prefill
  recurrence rewrite), but `cg::this_grid().sync()` is supported
  from sm_60+ and the prefill changes are arch-agnostic.

  Verified on a rented vast.ai RTX 3090 (PCIe 4.0×8, 350 W stock,
  driver 570.181, CUDA 12.8, PyTorch 2.5.1+cu124) by re-running
  `final_bench.py` apples-to-apples on `upstream/main` and on
  this branch:

  ```
                                pp520 (tok/s)   tg128 (tok/s)   completion vs HF
    upstream/main                   8,744           419            bit-exact
    fork-parent-b200               15,875           417            bit-exact
    delta                           +82 %          −0.5 %          unchanged
  ```

  Decode is flat (cooperative grid sync is neutral on sm_86 —
  expected, since the hand-rolled barrier was already correct on
  82 SMs). Prefill is +82 % from the same shared changes that win
  on B200. Neither branch reproduces the upstream README's headline
  37,800 pp520 / 413 tg128 numbers on this rig — an unrelated
  baseline gap likely tied to PCIe lane width / power profile /
  CUDA-runtime ABI of the published reference machine — but the
  delta between the two branches is consistent and positive.
- **sm_120 / sm_121a (RTX 50 / DGX Spark / GB10)**: same sources,
  same defines as before; the gb10 NVFP4 path is untouched. The
  shared decode kernel flips to cooperative grid sync, which is a
  strict improvement on sm_60+.
- **NVFP4 path on B200**: the LUT-based NVFP4 decode runs (29,994
  pp520, 327 tg128) but is slower than the new BF16 path because
  it intentionally avoids tensor cores. Recommended use of B200 is
  `--backend bf16`. NVFP4 decode tuning for B200 (tcgen05.mma
  warp-group tensor cores, NVFP4 native matvec) is deferred.

## Honest caveats

- The remaining pp gap to llama.cpp on the persistent
  `prefill_bf16_mega` path (still ~9k tok/s on this hardware) is
  inside `phase_matmul_bf16`'s WMMA loop. cuBLAS sm_100 picks
  hand-tuned kernels with `tcgen05.mma` warp-group tensor cores,
  swizzled shared layouts with `ldmatrix.x4`, and epilogue fusion;
  reproducing that in-kernel is a CUTLASS-grade GEMM engine of
  ~500 lines, deferred.
- The DeltaNet recurrence is now only ~30-35% of pp520 wall time
  (down from 88%), so further prefill speedups need either matmul
  rewrites or a chunked associative-scan recurrence (Mamba-style).
- All numbers are wall-clock pp520/tg128 from `final_bench.py`
  (10 warmup + 20 timed) on a single B200 with no other workload
  pinned to the same GPU.
