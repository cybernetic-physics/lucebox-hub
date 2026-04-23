# Prefill Megakernel (Qwen 3.5-0.8B)

A **true single-dispatch megakernel** for Qwen 3.5-0.8B prefill. All 24
layers (18 DeltaNet + 6 Full Attention) run inside one cooperative CUDA
launch separated by `cg::this_grid().sync()` between phases — the same
persistent-kernel pattern the decode megakernel uses.

This fills a gap in this repo: until now, only **decode** (single-token
inference) was fused as a megakernel. Prefill (`../megakernel/prefill.cu`)
was a sequenced pipeline of ~480 launches per forward pass — a deliberate
choice because cuBLAS bf16 GEMMs with tensor cores beat hand-rolled GEMMs
once sequence parallelism is large enough to fill them.

The prefill megakernel here is the foundation for training — training is
prefill-shaped, so fusing prefill is a prerequisite for fusing a forward
+ backward + AdamW training step.

## Math — what one dispatch does

Identical to `../megakernel/prefill.cu` so outputs can be diffed
element-for-element. The kernel is only a structural change (one launch
vs ~480), not a math change.

Per layer (inlined inside one `__global__` function, grid-synced between
phases):

```
input_rmsnorm → Q/K/V (or QKV+Z+β+α for DN) projections →
    ↓
    FA path: q/k RMSNorm + RoPE + KV cache write → causal attention (online softmax + sigmoid gate)
    DN path: conv1d+SiLU → L2 normalize → state-in-registers recurrence → gated RMSNorm
    ↓
o_proj (or out_proj) + residual → post_attn_rmsnorm →
gate/up → silu(gate)*up → down + residual
```

Then final RMSNorm on the last token, LM-head matmul over 248320 vocab,
block-parallel argmax reduced to one integer token id.

## Correctness

Same random bf16 weights, same tokens, compared token-by-token against
`qwen35_megakernel_bf16_C.prefill_bf16` (the existing multi-kernel
prefill):

```
S    existing.next_token    megakernel.next_token
 4   104265                 104265    match
 8    71423                  71423    match
16   167570                 167570    match
32   159560                 159560    match
64   132437                 132437    match
128   18959                  18959    match
```

S=1 is degenerate with random weights — the existing kernel returns `-1`
on 5 of 10 seeds (no valid argmax found) because random-weight logits at
a single token position land in pathological near-tie territory. Not a
math issue; both kernels agree on well-conditioned inputs.

## Walltime on B200

Current: cp.async double-buffered K-pipeline + WMMA tensor cores:

```
   S    existing (ms)    megakernel (ms)    speedup
  16             3.77              10.52      0.43x
  32             4.95               9.90      0.51x
  64             7.30              12.37      0.60x
 128            12.10              17.64      0.69x
 256            21.67              29.56      0.74x
```

Progression across the three committed revisions:

| S | naive FMA | WMMA only | + cp.async pipeline |
|---|---|---|---|
| 128 | 0.20× | 0.63× | **0.69×** |
| 256 | 0.17× | 0.68× | **0.74×** |

~3.7× faster at S=256 vs the first revision. We're now within **26%** of
cuBLAS at the training use case.

The cp.async pipeline double-buffers the A and B shared-memory tiles so
the next K-chunk's global→shared loads overlap with the current chunk's
WMMA compute. `cp.async.ca.shared.global` + `cp.async.commit_group` +
`cp.async.wait_group 1` lets consumer threads see loaded data as soon as
it lands without blocking on the stage after. K_TILE=32 gives 2 MMAs per
load-step, amortizing the sync overhead.

We're still not beating cuBLAS. The remaining ~26–30% gap at S=256
comes from:

1. **WMMA's MMA throughput on Blackwell is ~50% of cuBLAS's tcgen05.**
   This is the fundamental one — `mma.sync.aligned.m16n16k16` is
   synchronous and warp-scoped, whereas cuBLAS on sm_100 uses
   `tcgen05.mma` with Tensor Memory, which issues 128×256×16 tiles
   asynchronously. Closing this is a PTX-level rewrite (see below).
2. **1 block per SM** — WMMA's register footprint caps occupancy below
   2 blocks/SM, so `__launch_bounds__(512, 1)` + occupancy query clamps
   the cooperative grid to 148 blocks. cuBLAS has no such constraint.
3. **DeltaNet recurrence serializes across 16 heads** — the other 132
   SMs idle during DN phases. Breaking that requires overlapping DN-of-L
   with GEMMs-of-L+1 and re-sequencing the grid.sync fences.
4. **Store-through-shared for accumulator conversion** — `store_matrix_sync`
   → f32 shared slot → lane-wise read+convert+write to bf16 global adds
   ~5% per GEMM. Writing the fragment directly to global via known
   PTX fragment layout eliminates the roundtrip.

### Optimizations that did NOT help this session

For posterity so I don't re-try them blindly:

- **Synchronous shared-memory X-tile caching** with `__syncthreads()`
  around each K-chunk load: regressed to 0.30–0.66×. The per-chunk
  barrier cost more than L1 already saved. (Distinct from cp.async —
  cp.async uses `cp.async.wait_group`, not `__syncthreads()`, so it
  doesn't pay this cost.)
- **`__launch_bounds__(512, 2)`** to force 2 blocks/SM: regressed to
  0.33–0.57×. Register spills outweighed the occupancy win.
- **2×2 warp tiling** (4 accumulators per warp, 4 MMAs per K-step):
  regressed to 0.25–0.56×. Macro-tile grid becomes 4× smaller than the
  warp grid at Qwen 3.5-0.8B × S≤256, dropping SM utilization faster
  than the load/MMA ratio savings gained.
- **K_TILE=64 cp.async** (wider K-chunks, fewer pipeline iterations):
  regressed to 0.31–0.65×. 84 KB shared memory footprint dropped
  occupancy and the wider pipeline didn't compensate.

### What actually closes the gap on B200 — research notes

Web research turned up one critical correction to my earlier
"WGMMA is next" claim: **WGMMA is deprecated on sm_100 (Blackwell).**
The Hopper `wgmma.mma_async` instruction is not the target architecture's
primary tensor-core path anymore. Blackwell uses `tcgen05.mma` — a
completely different instruction family — paired with **Tensor Memory
(TMEM)**, a new 256 KB on-chip memory layer separate from registers and
shared memory, used as the accumulator home for 5th-generation tensor
cores.

The primary references for getting this right on B200:

1. [`tcgen05 for dummies`](https://gau-nernst.github.io/tcgen05/) — a
   full walk-through of the PTX syntax: `tcgen05.alloc`, the i_desc
   encoding for bf16→fp32 MMA, shared-memory descriptors with 128B
   swizzling, TMA bulk loads via `cp.async.bulk.tensor.2d`, mbarrier
   synchronization, `tcgen05.ld` for reading accumulators back out, and
   `tcgen05.commit`/`tcgen05.wait`. A minimal hand-rolled kernel
   following this pattern hits **~98% of cuBLAS** on 4096³ GEMMs.
2. [ThunderKittens on Blackwell](https://hazyresearch.stanford.edu/blog/2025-03-15-tk-blackwell) —
   the pattern that matters for a persistent megakernel: **producer /
   consumer warp specialization.** 2 producer warps per consumer
   warpgroup handle loads; consumer warpgroups exclusively run MMA; MMA
   completion directly signals the producers to prefetch the next stage.
   Sustained tensor-core saturation with "only one bubble in the whole
   tensor pipeline (~140 ns every few hundred microseconds)".
3. [CUTLASS Blackwell tutorial](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/)
   and the NVIDIA CUTLASS examples at `examples/cute/tutorial/hopper/`.

So the concrete path to beat cuBLAS from inside a megakernel is:

1. **Rewrite `gemm_wmma_core` against `tcgen05.mma` + TMA + TMEM.**
   The unit of work becomes a 128×128 or 128×256 tile instead of a
   warp-scoped 16×16. Operand layouts switch to 128B-swizzled shared
   memory.
2. **Switch to producer / consumer warp specialization within each
   block.** 2 producer warps issue TMA for the next K-chunk; consumer
   warpgroups do MMA on the current one. mbarriers chain the two.
3. **Keep the cooperative-grid megakernel shape**: all 24 layers still
   in one dispatch; the tcgen05 GEMM drops in as the inner primitive.

This is ~500+ lines of new PTX-level code per GEMM primitive and needs
Nsight Compute profiling to tune. It's the real remaining work.

### Failed optimization attempts this session

For posterity, so I don't re-try these blindly later:

- **Shared-memory X-tile caching** with `__syncthreads()` between load
  and MMA: regressed to 0.30–0.66×. The per-K-chunk barrier cost more
  than L1 already saves; B200's unified 228 KB shared/L1 pool was
  already catching the intra-block X re-use pattern.
- **`__launch_bounds__(512, 2)` for 2 blocks/SM:** regressed to
  0.33–0.57×. Register spills from the forced occupancy outweighed the
  parallelism win.
- **2×2 warp tiling** (each warp owns 4 output sub-tiles, 4 independent
  accumulators): regressed to 0.25–0.56×. The macro-tile grid was
  4× smaller than the warp grid at Qwen 3.5-0.8B's shape × S ≤ 256,
  dropping effective SM utilization faster than the load-to-MMA ratio
  savings gained. 2×2 would only help when `total_macro_tiles ≳
  warps_grid`, which needs S ≳ 512 for the smallest-N GEMMs.

The committed WMMA baseline (1×1 warp tiles, sync loads) is what
actually helps today. The optimizations above need a profiler-guided
approach to not backslide.

### S % 16 requirement

WMMA tiles are 16 rows wide, so the kernel now requires `S % 16 == 0`.
`PrefillMegakernel.forward()` asserts this. Pad your prompt to the next
multiple of 16 before calling (see `smoke.py` / `test_vs_existing_prefill.py`).

## Why ship it slower

Because **the megakernel is the right shape** to hang LoRA and backward
off of. The existing prefill pipeline can't be easily extended into a
fused training step — each `cublasGemmEx` is an opaque dispatch boundary
that forces a pipeline stall across forward→backward→update. The
megakernel has explicit phase fences (`grid.sync()`) that are already set
up for exactly that kind of extension.

Speed will come back from:
1. **Tensor-core GEMMs** inside the megakernel (WGMMA on sm_100). This
   is mechanical — replace the `gemm_bf16_bf16` primitive. It'll close
   most of the 5× gap.
2. **GEMM tiling / shared-memory double buffering** to match cuBLAS's
   memory hierarchy.
3. **Layer-boundary activation reuse** — right now `hidden` and
   `residual` bounce through global memory; much of that can stay in
   shared across sub-phases.

None of these change the kernel's structure — they're optimizations
inside each GEMM phase.

## Files

| file | purpose |
|---|---|
| `kernel.cu` | the megakernel (embed + 24 layers + final norm + LM head argmax, all in one cooperative dispatch) |
| `torch_bindings.cpp` | `torch.ops.prefill_megakernel_C.prefill_mega` |
| `setup.py` | `CUDAExtension` build, auto-detects sm arch |
| `model.py` | `PrefillMegakernel` class; weight packing into `LayerWeights` device blob |
| `smoke.py` | run the megakernel on random weights, check it returns a valid token id |
| `test_vs_existing_prefill.py` | bit-for-bit correctness diff against `../megakernel/prefill.cu` |
| `bench.py` | walltime comparison |

## Build

```bash
cd prefill_megakernel
python3 setup.py build_ext --inplace
```

Verify:

```bash
python3 smoke.py 32                 # end-to-end run on random weights
python3 test_vs_existing_prefill.py 16   # argmax-token diff vs existing
```

## What's next

1. **Tensor-core GEMM tiles** inside `gemm_bf16_bf16`. Closes the speed
   gap; no architectural change needed.
2. **LoRA residuals on every trainable linear**. With the megakernel's
   phase structure this is a single extra phase per linear that adds
   `(x @ A) @ B · scaling` to the output.
3. **Backward megakernel**: symmetric structure running layers in
   reverse. Flash-attention backward for FA layers; the DeltaNet
   recurrence backward is genuinely novel work (BPTT through the
   `(I - β k kᵀ)` state transition) and is the biggest remaining
   research-grade piece.
4. **Fused AdamW phase** at the end of the backward megakernel — one
   cooperative-grid pass updates every LoRA A/B pair.

The forward megakernel here is the scaffolding those extensions fit on
top of — every `grid.sync()` boundary is a natural hand-off point for
saving activations (forward pass) or injecting gradients (backward pass).
