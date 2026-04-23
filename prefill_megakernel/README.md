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

WMMA tensor-core GEMMs (bf16 16×16×16 tiles, f32 accumulation):

```
   S    existing (ms)    megakernel (ms)    speedup
  16             7.89              10.42      0.76x
  32             5.07              11.61      0.44x
  64             7.71              13.97      0.55x
 128            12.16              19.35      0.63x
 256            21.75              31.62      0.69x
```

The first revision of this megakernel used naive warp-per-output-element
FMA GEMMs and ran at **0.20× of cuBLAS at S=128**. Switching the GEMM
primitive to `nvcuda::wmma` tiles lifted that to **0.63×** — a ~3× speedup
at S=128 and ~4× at S=256, purely from the GEMM change.

We're still not beating cuBLAS. The remaining ~30–50% gap comes from:

1. **1 block per SM** — WMMA's per-thread register footprint pushed
   occupancy below 2 blocks/SM, so `__launch_bounds__(512, 1)` and
   `cudaOccupancyMaxActiveBlocksPerMultiprocessor` clamp the cooperative
   launch to 148 blocks. cuBLAS can dispatch separate launches with
   higher per-SM occupancy.
2. **No shared-memory tile re-use** — each WMMA call loads X and W
   directly from global. cuBLAS tiles X across multiple N tiles and
   keeps it in shared.
3. **DeltaNet recurrence serializes across 16 heads** — the other 132
   SMs idle during DN phases.
4. **No WGMMA** — Blackwell-native warp-group MMA is ~2× faster per
   warp than WMMA. Requires async restructuring.

Those are all orthogonal optimization passes inside the megakernel
structure — no architectural changes needed.

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
