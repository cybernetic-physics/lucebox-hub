# Qwen3.5-0.8B on NVIDIA GB10 (DGX Spark) — gb10-train branch

Port of the 3090-train megakernel to GB10 (Grace Blackwell, sm_121a,
compute cap 12.1, 48 SMs, aarch64). Built with multi-arch nvcc gencode
so the same `.so` ships SASS for both sm_86 (3090) and sm_121a (GB10).

## Environment

| Item | Value |
|------|-------|
| GPU | NVIDIA GB10 (sm_121a, cap 12.1, 48 SMs, integrated Grace Blackwell) |
| Driver | 580.126.09 (`nvidia-driver-580-open`) |
| CUDA | 13.2.78 (`/usr/local/cuda`) |
| PyTorch | 2.11.0+cu130 |
| Build flags | `-gencode arch=compute_121a,code=sm_121a -O3 --use_fast_math -DBLOCK_SIZE=512 -DLM_BLOCK_SIZE=256` |
| Model | `Qwen/Qwen3.5-0.8B` (HF cache: `/home/sparkz/rl/.hf_cache`) |
| Branch | `gb10-train` (off `3090-train`) |

GB10 reports `power.draw=14W` at idle; the integrated SoC exposes no
`power.limit` rail, so the 3090's 220 W DVFS sweet-spot table doesn't
translate.

## Build (multi-arch)

```bash
cd models/qwen35_0p8b
# Single arch (current host):
MEGAKERNEL_CUDA_ARCH=sm_121a MAX_JOBS=8 python setup.py build_ext --inplace

# Cross-arch fat binary (3090 + GB10 in one .so):
MEGAKERNEL_CUDA_ARCHS=sm_86,sm_121a MAX_JOBS=8 python setup.py build_ext --inplace
```

`setup.py` accepts:

| Env | Default | What it does |
|-----|---------|--------------|
| `MEGAKERNEL_CUDA_ARCHS` | (auto-detect) | Comma-separated list, emits one `-gencode` per arch. |
| `MEGAKERNEL_CUDA_ARCH`  | (auto-detect) | Single-arch shorthand. Legacy 3090-train build env still works. |
| `MEGAKERNEL_BLOCK_SIZE` | `512`         | Threads per cooperative block. **Don't lower on GB10** — 256 is ~25 % slower because warp count shrinks. |
| `MEGAKERNEL_LM_BLOCK_SIZE` | `256`      | Threads per LM-head block. |

Runtime env (read by the launcher in `kernel.cu` / `kernel_gb10_nvfp4.cu`):

| Env | What it does |
|-----|--------------|
| `MEGAKERNEL_DECODE_BLOCKS` | Override decode grid size. **Forcing > `cudaOccupancyMaxActiveBlocksPerMultiprocessor * SM_count` corrupts the cooperative kernel** — see "What didn't work" below. |
| `MEGAKERNEL_LM_BLOCKS` | Override LM-head grid size. Non-cooperative kernel; safe to oversubscribe. |
| `MEGAKERNEL_BACKEND` | Force `bf16` or `nvfp4`. Overrides Decoder(`backend="auto"`). |

## Backend modes

Three backends are now available; `Decoder(backend="auto")` returns
`bf16`. Override with `--backend` or `MEGAKERNEL_BACKEND=`.

| Backend | What runs | Greedy top-1 vs HF | pp520 / tg128 (tok/s) | When to use |
|---------|-----------|:------------------:|----------------------:|-------------|
| `bf16` | BF16 megakernel decode + BF16 LM head. | **100 %** (32/32) | 7,185 / 69 | Correctness-critical (RLHF rollouts, evals). |
| **`bf16_fp4lm`** | BF16 megakernel decode + cuBLASLt FP4 block-scaled LM head (`CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`). | **100 %** (32/32) | 7,384 / 65 | Same quality as bf16, ~125 MB lower LM-head footprint. Step-time is +970 us because BF16 LM head still runs first; will be net-positive once a `decode_bf16_no_lm` variant exists. |
| `nvfp4` | Full FP4 layer projections + cuBLASLt FP4 LM head. | 3/32 (intrinsic FP4 drift across 24 layers) | 9,187 / 88 | Ablations and microbenches only — output is coherent but diverges from HF. |

`bf16_fp4lm` is the "NVFP4 with no quality loss" path: the cuBLASLt FP4
LM head is the only FP4 component, and the LM-head quantization alone
preserves HF's argmax (verified in `experiments/diag_nvfp4.py` —
FP4-roundtripped LM head reports rank 0 for HF's top token).

Background — the previous default `nvfp4` mode loses parity because the
24 layer projections each compound ~19 % per-group-32 FP4 rel err.
That's intrinsic to per-group-32 scalar FP4, not a kernel bug. The
fix path is `mma.kind::mxf4` / `tcgen05.mma` tensor-core layer
projections (open work below).

## Backend default

`Decoder(backend="auto")` returns **`bf16` on GB10**, not NVFP4. Why:

- NVFP4 decode on GB10 currently uses **software FP4 dot-products**
  (`dot8_nvfp4_bf16` does a LUT lookup + scalar FMA). No
  `mma.kind::mxf4` / `tcgen05.mma` yet, so it's only ~8 % faster than
  BF16 decode on the same kernel layout.
- That 8 % comes at the cost of **9.4 % greedy top-1 agreement** with HF
  (BF16 megakernel matches HF 32/32 = 100 %). Output stays coherent but
  the argmax flips at step 0 because 24 layers × ~8 per-group-32 FP4
  projections compound to ~19 % per-projection rel err.
- The FP4 LM head on its own is **not** the problem — applying the
  FP4-roundtripped LM head to HF's last hidden gives `rank 0` for the
  same argmax token (verified in `experiments/diag_nvfp4.py`).

NVFP4 stays available via `--backend nvfp4` / `MEGAKERNEL_BACKEND=nvfp4`
for users who want to experiment.

## Correctness vs HF eager

| Backend     | Greedy top-1 vs HF (32 steps, `"The capital of France is"`) | First divergence |
|-------------|------------------------------------------------------------:|------------------|
| BF16        | **32/32 (100 %)**                                          | none             |
| NVFP4 decode| 3/32 (9.4 %)                                                | step 0           |

## `final_bench.py` — pp520 / tg128

| Backend | Prefill mode | pp520 (tok/s) | tg128 (tok/s) | greedy top-1 |
|---------|--------------|--------------:|--------------:|:-------------:|
| **BF16 (auto default)** | eager (`prefill_bf16`) | **13,009** | **130** | 100 % |
| NVFP4 (`--backend nvfp4`) | eager | 14,589 | 152 | 9.4 % |
| BF16 | mega (`prefill_bf16_mega`) | 5,614 | 130 | 100 % |

Note: `prefill_bf16_mega` regresses 2.3× on GB10 vs eager prefill —
the WMMA tile shapes are tuned for sm_86's 100 KB L1. Eager prefill
(cuBLAS + per-layer kernels) is the right path on Blackwell.

## Rollout shape sweep (prefill + 32-gen)

BF16 default; canonical 3090-doc shape table. Best-of-2 runs per shape,
one warm pass.

| Prompt | HF (ms) | Ours (ms) | vs HF | Ours pp (tok/s) | Ours tg32 (tok/s) |
|-------:|--------:|----------:|------:|----------------:|------------------:|
|    128 |   602.6 |     252.3 | 2.39× |          6,617 |             133.2 |
|    512 |   571.3 |     277.1 | 2.06× |         12,523 |             131.5 |
|   2048 |   861.7 |     384.3 | 2.24× |         14,865 |             125.8 |
|   8192 | 2,042.1 |     802.7 | 2.54× |         15,417 |             114.2 |
|  16384 | 4,447.4 |   1,508.4 | 2.95× |         13,958 |              92.7 |
|  32768 | 8,923.6 |   3,319.2 | 2.69× |         11,432 |              70.7 |

Reproduce:

```bash
source /home/sparkz/rl/.venv/bin/activate
export HF_HOME=/home/sparkz/rl/.hf_cache
PYTHONPATH=models/qwen35_0p8b python experiments/bench_gb10_rollout.py \
    --json /tmp/gb10_rollout.json
```

## Cross-arch reference

| Target | pp520 (tok/s) | tg128 (tok/s) | Source |
|--------|--------------:|--------------:|--------|
| **GB10 (sm_121a)** | **13,009** BF16 / 14,589 NVFP4 | **130** BF16 / 152 NVFP4 | this doc |
| RTX 3090 (sm_86, @220W) | 37,800 | 413 | `qwen35_0p8b_3090.md` |
| B200 (sm_100) | 40,278 | 711 | `qwen35_0p8b_b200.md` |

GB10's gap to 3090 and B200 isn't memory bandwidth (GB10 has unified
LPDDR5X at ~512 GB/s, plenty for this workload) — it's that **every
kernel here was hand-tuned for sm_86**, and most of the wins come from
arch-specific instructions that the kernels don't yet emit on sm_121a.

## What I tried that didn't work

1. **Oversubscribe the decode grid (`MEGAKERNEL_DECODE_BLOCKS=96+`).**
   `experiments/sweep_block_count_gb10.py` reported 3-4× speedup on
   timing, but **correctness broke** — outputs degenerated to "!!!!".
   `decode_kernel` is launched via `cudaLaunchCooperativeKernel`, and
   `cudaOccupancyMaxActiveBlocksPerMultiprocessor` is the **hard
   ceiling**, not a heuristic. Forcing more blocks succeeds the launch
   but breaks `cg::this_grid().sync()` semantics. Reverted.

2. **`MEGAKERNEL_BLOCK_SIZE=256`** — compiles, correct output, but
   ~25 % slower than 512 (`tg128: 130 → 104`). Smaller blocks mean
   fewer warps available for hiding latency in the matvec hot loop.

3. **`prefill_bf16_mega`** — works but 2.3× slower than eager prefill
   on GB10 because WMMA tile shapes target sm_86 L2. Don't use on GB10.

## Open work for a real GB10 speedup

These are the high-value items, in order. Each needs CUDA work
(estimated multi-day for a clean implementation):

1. **Hardware FP4 path for sm_120+.** Replace `dot8_nvfp4_bf16`'s LUT
   + scalar FMA with `mma.kind::mxf4` / `tcgen05.mma`. Activation
   layout in shared memory needs to match the new MMA. Would also need
   FP4 activation quantization on hot paths (currently activations
   stay BF16). Realistic target: 2-3× decode throughput, restored
   correctness.

2. **Cluster-launched decode.** Replace `cg::this_grid().sync()` with
   thread-block-cluster barriers (4-block clusters share SMEM, sync
   cheaper than grid sync). This is the architectural change that
   would let us oversubscribe blocks safely.

3. **Retune `prefill_bf16_mega` block/tile for sm_121a.** WMMA still
   works on Blackwell but the existing 16×16×16 shape and L2 chunking
   are sm_86-shaped. WGMMA (`wgmma.mma_async`) would be the modern
   replacement.

4. **DeltaNet chunked forward retune (`V_SPLITS`, `C`).** Currently
   `V_SPLITS=4, C=32` in `dn_chunked_3090.cu` is tuned for 3090's 82
   SMs and 100 KB L1. GB10 has 48 SMs and a different SMEM budget.

5. **Profile.** `ncu --set full` needs `RmProfilingAdminOnly=0` or
   root. The host has `RmProfilingAdminOnly: 1` so I couldn't pull
   counters. With ncu, the items above can be prioritized by measured
   stall vs assumed stall.

## Diagnostic + perf tooling shipped on this branch

| Script | What |
|--------|------|
| `experiments/bench_gb10_rollout.py` | Shape sweep (prefill + 32-gen) vs HF. |
| `experiments/correctness_gb10.py`   | Greedy top-1 agreement vs HF, per backend. |
| `experiments/diag_nvfp4.py`         | FP4 quant round-trip check + LM-head argmax isolation. |
| `experiments/sweep_block_count_gb10.py` | Sweep `MEGAKERNEL_DECODE_BLOCKS` / `MEGAKERNEL_LM_BLOCKS`. **Note: any number > cudaOccupancy ceiling produces wrong output despite faster timing.** |
| `experiments/breakdown_gb10.py`     | Decode step time vs context length, per-op cost. |
| `experiments/ncu_target.py`         | Tiny ncu target (needs root on this host). |
