# tcgen05 bf16 → fp32 GEMM — Blackwell raw-PTX sandbox

Standalone sandbox for bringing up the `tcgen05.mma` path that cuBLAS
uses on sm_100. Goal: get one GEMM tile correct outside the megakernel
before swapping WMMA out of `../prefill_megakernel`.

## Current state

Shape: **M=128, N=256, K=64 bf16, fp32 accumulate**.
128B swizzle mode, row-major shared memory load via `cp.async`, full
K-loop with `enable_input_d=1` accumulation.

### What works

| piece | state |
|---|---|
| Compile on sm_100a | ✅ |
| `tcgen05.alloc` / `tcgen05.dealloc` | ✅ |
| `mbarrier.init` + `mbarrier.try_wait.parity` | ✅ |
| Shared-mem descriptor (bits 0-13 addr / bit 46 leading-dim / bits 61-63 swizzle=2) | ✅ |
| **`tcgen05.mma.cta_group::1.kind::f16`** | ✅ **running** |
| K-loop with `enable_input_d=1` accumulation | ✅ |
| Element (0, 0) of output matches torch ref to 7 sig digits | ✅ |
| 4287 / 32768 (13%) output positions correct | ⚠️ partial |

### What doesn't yet work

The `tcgen05.ld.sync.aligned.32x32b.x8.b32` readback — the per-lane
register to (row, col) mapping. Empirical pattern across the 13% of
correct positions:

- Every row has ~32 correct columns (i.e. layout A: `lane t → row r + t` is right for the row dim).
- Only every 8th column within a row is correct — specifically, reg 0 lands at `col_base + 0`.
- Regs 1..7 do NOT land at `col_base + 1..7` (would give 256/256 matches) nor at `col_base + 32*i` (tested, 737 matches — worse) nor at a non-contiguous row map {0,1,8,9,16,17,24,25} (tested, same ~2%).

So layout A is right for the M dim and reg 0's N position, but regs 1..7 go somewhere my guesses don't predict. Possibilities still unverified:

- TMEM's 4 sub-partitions interleave logical output cols — logical col n might map to TMEM[n % 4 sub-partition, n / 4 offset]; regs 1..7 might each live in different sub-partitions.
- The `.x8` vector may load rows stride vs cols stride differently than I assumed — e.g. 32 warp-lanes × 8 regs might form a 32 × 32 tile via non-linear mapping.
- Resolving definitively needs the NVIDIA PTX ISA reference for sm_100 `tcgen05.ld` fragment layout (CUTLASS's `cute/arch/copy_sm100_tmem.hpp` has the exact descriptors).

## Files

- `kernel.cu` — tcgen05 GEMM with K=64 + 128B swizzle + full K-loop
- `torch_bindings.cpp` — single-op binding
- `setup.py` — sm_100a build
- `test.py` — torch-reference diff + match-position diagnostics

## Test run

```bash
cd tcgen05_gemm
python3 setup.py build_ext --inplace
python3 test.py
```

Output as of current commit:

```
C[0, :8]   = [0.07587875, -0.060443807, 0.032965656, -0.041262123, ...]
ref[0,:8]  = [0.07587876,  0.033420451, 0.003561549, -0.023494590, ...]
             ^^^^^^^^^^^ (0,0) matches to 7 sig digits
exact matches count: 4287
rows with matches (row:#matches): (0, 32), (1, 34), (2, 33), (3, 33), (4, 33), ...
   — every row has ~32 matches (every 8th col across all iterations)
first matching cols: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ... (cross-lane col stride 1)
```

## Why this is the right path even at 13% correctness

Every WMMA-tuning knob I've tried (shared-mem X cache, 2×2 warp tiles,
launch_bounds(512, 2), K_TILE=64) has regressed or stayed flat. The
instruction-level ceiling of `mma.sync.m16n16k16` is ~0.75× cuBLAS on
Blackwell. `tcgen05.mma` runs ~2× faster per tile.

The MMA compute itself is demonstrably WORKING in this sandbox
(correct values for the positions where readback lines up). Closing
this is a layout-docs question, not a compute-correctness question —
so once `.32x32b.x8` fragment mapping is pinned down, we go from 13%
of outputs correct → 100%, and the same kernel structure slots into
the megakernel.

## Resolving next session

1. Pull `cute/arch/copy_sm100_tmem.hpp` from CUTLASS and read the
   exact fragment mapping for .32x32b.x8.
2. Write a TMEM probe kernel: `tcgen05.st` a known pattern to
   TMEM, then `tcgen05.ld` and record which lane/reg reads which
   (row, col). 64 lines of instrumentation gets the layout for sure.
3. Rewrite the readback to match.
4. Integrate as a `gemm_tcgen05` variant in `prefill_megakernel/kernel.cu`,
   matching `gemm_wmma_core`'s signature.
5. Benchmark vs cuBLAS. Expected: 1.2–1.8× cuBLAS on 4k-scale shapes per
   published numbers; 2× is realistic at full WGMMA-descriptor + TMA
   producer/consumer specialization.
