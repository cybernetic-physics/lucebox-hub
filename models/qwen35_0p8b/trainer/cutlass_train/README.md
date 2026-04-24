# CUTLASS training pipeline for Qwen3.5-0.8B (B200 / sm_100)

Goal: a full LoRA training stack built on CUTLASS 3.x Blackwell kernels,
replacing the HF+PEFT delegation in `LoraMegakernelTrainer`.

## Scope honestly

A complete CUTLASS training pipeline is **4-6 engineer-weeks** of expert
kernel work. This directory is the scaffold + working demo; the full
pipeline is staged behind a feature flag on `LoraMegakernelTrainer`.

## What CUTLASS gives us out of the box

- `cutlass::gemm::*` — bf16 WGMMA-based GEMMs on Blackwell (sm_100).
  Covers every linear projection (q/k/v/o/gate/up/down/adapter A/B).
- `examples/77_blackwell_fmha/` — warp-specialized causal FMHA
  **forward + backward** with bf16 Q/K/V, fp32 accum. Production-
  grade, SOL on B200. This replaces our hand-rolled causal attention
  for both inference prefill and training backward.
- Epilogue fusion — bias-add, activation, conversion; keeps the
  RMSnorm/residual/LoRA-add writes fused onto the GEMM output.

## What CUTLASS does NOT cover (we must write)

- **DeltaNet recurrence (fwd + BPTT bwd).** The gated Delta-Net used
  by Qwen3.5's 18 linear-attention layers is not in CUTLASS. Two
  options:
  1. Port the `fla` library's chunked Delta-Net Triton kernels to
     native CUDA C++ (chunk-parallel with bf16 WGMMA inside each chunk
     matmul; prefix-scan across chunks).
  2. Keep the current CUDA serial recurrence (with our q/k prefetch)
     for forward, and derive the BPTT by reversing the loop with
     saved intra-chunk states.
  Either is a few days of work after the FA stack is in place.
- **RMSnorm fwd/bwd** — already have these in our `kernel.cu`. Keep.
- **SwiGLU fwd/bwd** — already have. Keep.
- **LoRA-linear bwd** — already have. Keep.
- **Fused AdamW over flat LoRA params** — already have. Keep.
- **CE + LM-head bwd** — already have. Keep.

## Architecture

```
LoraMegakernelTrainer
  .forward_backward(data)
    If cutlass_backend=True:
      → our forward (prefill_bf16_train_step, saves activations)
      → per-layer backward chain (this directory):
           • CE+LM head bwd   (kernel.cu)
           • For each layer L, in reverse order:
               - MLP: down.bwd → silu_mul.bwd → gate/up.bwd  (kernel.cu)
               - post-attn rmsnorm.bwd                        (kernel.cu)
               - If FA layer:
                   - o_proj.bwd
                   - CUTLASS FMHA backward               (*** this dir ***)
                   - q_proj/k_proj/v_proj.bwd
               - If DN layer:
                   - out_proj.bwd
                   - gnorm.bwd, silu_mul(z).bwd
                   - CUSTOM DN BPTT                     (*** this dir ***)
                   - qkv/z/b/a proj.bwd
               - input rmsnorm.bwd
           • LoRA A/B grad accumulation per projection  (kernel.cu)
      → fused AdamW over flat LoRA param buffer        (kernel.cu)
      → done
```

Public trainer API unchanged; only the internals swap HF+PEFT for
our kernel stack.

## Build

CUTLASS is header-only. `CMakeLists.txt` wires the FMHA example
(77_blackwell_fmha) plus our DN and glue kernels into a single .so
loadable from Python.

```bash
cd models/qwen35_0p8b/trainer/cutlass_train
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=100 -DCUTLASS_DIR=/root/cutlass
cmake --build build --target cutlass_train_C -j
```

The resulting `cutlass_train_C.so` is imported by `rl_trainer.py`
and gated on `use_cutlass_backend=True`.

## Files in this directory

- `fmha_bwd.cu` — launcher around CUTLASS's sm100 FMHA backward kernel.
  Binds to a torch op `torch.ops.cutlass_train_C.fa_bwd`. **TODO: full
  wiring of GQA + causal; for now compiles a fixed-shape sanity instance.**
- `gemm_bf16_sm100.cu` — CUTLASS bf16 GEMM launcher used by the
  per-layer backward for cheap matmuls. Mirrors our existing
  `cublas_bf16_gemm` but routes through CUTLASS (WGMMA, tensor cores).
  **STUB.**
- `deltanet_bwd.cu` — chunked DN backward kernel using bf16 WGMMA
  for the chunk matmuls. **STUB.**
- `CMakeLists.txt` — build wiring.

## Status

| kernel | status | notes |
|:---|:-:|:---|
| CUTLASS bf16 GEMM launcher | scaffolded, builds | matmul sanity works |
| CUTLASS FMHA backward (sm100) | scaffolded | bindings + test pending |
| DeltaNet BPTT (chunked) | documented | impl pending |
| Trainer integration (flag) | not started | rl_trainer.py unchanged |

Shipping this dir as-is commits the architecture decision + build
infrastructure; subsequent PRs fill in kernels until the full stack
lights up and the 3× Phase 5 claim gate opens.
