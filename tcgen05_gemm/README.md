# tcgen05 bf16 GEMM (WIP — not correct yet)

Raw PTX exploration of the Blackwell `tcgen05.mma` path that cuBLAS uses
to beat the WMMA-based `prefill_megakernel`. This directory is a
**standalone sandbox** for getting a single `[M=128, N=256, K=16]` MMA
correct before integrating into the megakernel.

## Status

| piece | state |
|---|---|
| compile on sm_100a | ✅ clean |
| `tcgen05.alloc` / `tcgen05.dealloc` | ✅ runs, no crash |
| `mbarrier.init` + `mbarrier.try_wait` | ✅ compiles and runs |
| `tcgen05.mma.cta_group::1.kind::f16` | ✅ accepts, runs |
| `tcgen05.commit` + mbarrier arrival | ✅ runs |
| `tcgen05.ld.sync.aligned.32x32b.x8.b32` readback | ✅ runs |
| **numerical correctness vs torch** | ❌ **max\|Δ\|=0.29 at K=16** |

The output is in the right range (not all zeros, not all garbage) but
does not match `A @ Bᵀ`. The failure mode is structural — almost
certainly the 64-bit shared-memory descriptor encoding. The
[gau-nernst walkthrough](https://gau-nernst.github.io/tcgen05/) uses
128B swizzle (`2ULL << 61`), leading-dim bit 46 set, specific
LBO/SBO byte offsets per matrix-box layout; my current `make_desc_a` /
`make_desc_b` helpers don't match that exactly.

## What needs to happen next

1. **Descriptor bit layout** — match the exact PTX-spec field positions
   for `kind::f16` (bits 0–13 base, 16–29 LBO, 32–45 SBO, 46 leading-dim,
   52–60 base-offset, 61–63 swizzle).
2. **128B swizzling on loads** — the A-matrix box for M=128 K=16 bf16
   is 4 rows × 128 bytes per 128B chunk; the cp.async load pattern
   needs to XOR the column index with a swizzle pattern so the MMA reads
   values from the right addresses. The alternative is TMA
   (`cp.async.bulk.tensor.2d`) which swizzles automatically given a
   tensor map set up on the host — probably the cleaner path.
3. **tcgen05.ld layout** — the `.32x32b.x8` shape loads 32 rows per
   warp; per-lane element ordering inside the 8 registers has a
   specific mapping to `(row, col)` that needs to match the MMA output
   layout for 128×256 f32.
4. **Then scale K** — add a K-loop with `enable-input-d=1` to accumulate
   across K-chunks.
5. **Then scale M, N** — tile across multiple MMA shapes to cover
   arbitrary output sizes.
6. **Then integrate into `prefill_megakernel/kernel.cu`** as a drop-in
   replacement for `gemm_wmma_core`.

## Files

- `kernel.cu` — the kernel + host launcher (compiles cleanly; wrong output)
- `torch_bindings.cpp` — `tcgen05_gemm_C.tcgen05_gemm_one_tile(A, B, C)`
- `setup.py` — `CUDAExtension` targeting sm_100a
- `test.py` — torch-reference diff

## Build & test

```bash
cd tcgen05_gemm
python3 setup.py build_ext --inplace
python3 test.py
```

Expected output right now:

```
max |Δ|: 2.89e-01  mean |Δ|: 4.27e-02
CORRECTNESS FAIL
```

## Why this is the right next step anyway

The WMMA-path prefill megakernel ships at 0.74× of cuBLAS at S=256.
That gap is fundamentally `mma.sync.m16n16k16` vs `tcgen05.mma.m128nXk16` —
the Blackwell-native instruction runs ~2× faster per-tile because it
uses the TMEM accumulator (not registers) and async issue semantics.
No amount of WMMA tuning will close it. This sandbox is the first step
toward swapping in the Blackwell path.
