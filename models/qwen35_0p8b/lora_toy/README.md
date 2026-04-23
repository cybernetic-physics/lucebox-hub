# LoRA Training Megakernel

One CUDA dispatch per training step: embedding lookup → base-logits matmul
→ LoRA residual → log-softmax / cross-entropy → backward to LoRA grads →
AdamW update on `lora_a` and `lora_b`. No CPU round-trips, no autograd.

Designed as a drop-in replacement for the torch-based step in
`tinker_backend/runtimes/lora_trainer.py` (the `TinyLoRAModel` +
`torch.optim.AdamW` pair that rl/backend uses today).

## Math — what one dispatch does

Per training step with sequence of `T` tokens:

```
hidden[t, :]     = embedding[context_tokens[t], :]           # frozen
lora_h[t, :]     = hidden[t, :] @ lora_a                      # [T, R]
logits[t, :]     = hidden[t, :] @ output_weight.T             # frozen
                 + lora_h[t, :] @ lora_b                      # LoRA residual
log_probs        = log_softmax(logits)
selected[t]      = log_probs[t, target_tokens[t]]
loss             = -(selected * token_weights).mean()

grad_logits[t,v] = (w[t]/T) * (softmax(logits)[t,v] - onehot(target[t]))
grad_lora_b      = lora_h.T @ grad_logits                     # [R, V]
grad_lora_h      = grad_logits @ lora_b.T                     # [T, R]
grad_lora_a      = hidden.T  @ grad_lora_h                    # [H, R]

# Fused AdamW:
m = β1·m + (1-β1)·g;  v = β2·v + (1-β2)·g²
param -= lr · (m / (√v + eps) + wd·param)   (bias-corrected)
```

All of this runs inside a single cooperative-grid `__global__` kernel.
Phases are separated by `cg::this_grid().sync()` — no kernel launches
between the forward and the optimizer step. bf16 weights, fp32 accumulation,
fp32 optimizer state.

## Files

| File | Purpose |
|---|---|
| `kernel.cu` | Seven-phase CUDA megakernel + host launch wrapper |
| `torch_bindings.cpp` | `lora_train_step` torch op registration |
| `setup.py` | `CUDAExtension` build (auto-detects sm arch) |
| `model.py` | `LoRATrainStepKernel` — owns params + Adam state on GPU |
| `test_correctness.py` | vs pure-torch reference across three shapes |
| `bench.py` | walltime vs torch |

## Build

```
cd lora_megakernel
python3 setup.py build_ext --inplace
```

## Correctness

`test_correctness.py` runs 3 steps at three configurations and diffs
selected log-probs, loss, and post-step LoRA parameters against a pure-torch
implementation (`torch.nn.functional.log_softmax` + `torch.optim.AdamW`).

```
=== V=256  H=64   R=8   T=8  ===
  step 2: loss Δ=0.00e+00  selected Δ=4.77e-07  lora_a Δ=6.28e-04  lora_b Δ=2.55e-05
=== V=1024 H=128  R=16  T=16 ===
  step 2: loss Δ=9.54e-07  selected Δ=4.77e-07  lora_a Δ=1.56e-03  lora_b Δ=3.07e-05
=== V=4096 H=256  R=16  T=32 ===
  step 2: loss Δ=9.54e-07  selected Δ=9.54e-07  lora_a Δ=6.91e-04  lora_b Δ=3.01e-04
CORRECTNESS OK
```

Losses match the torch reference to within fp32 rounding. LoRA parameter
residuals after three AdamW steps stay within bf16 storage rounding.

## Walltime on B200

```
     V      H     R     T    torch (ms)   megakernel (ms)   speedup
   256     64     8     8         2.316             0.086    26.81x
  1024    128    16    16         2.998             0.470     6.38x
  4096    256    16    32         2.937             1.911     1.54x
  8192    512    16    32         1.780             4.108     0.43x
```

The current kernel is **tuned for the launch-overhead regime** — which is
exactly where rl/backend's `TinyLoRAModel` lives today (V=256, H=64, R=8).
Torch's step issues ~10 separate CUDA launches (embedding, two matmuls,
`log_softmax`, `gather`, `mean`, `backward`, two per-param AdamW kernels),
and on a B200 each launch costs ~100–200 μs. Fusing into one cooperative
launch collapses all of that to a single dispatch.

At larger shapes (V≥8k, H≥512), cuBLAS's tensor-core GEMMs beat the plain
FMA loops in this kernel — the logits matmul `hidden @ output_weight.T`
starts dominating and cuBLAS wins on raw FLOPs. Closing that gap needs
`wmma::mma_sync`/WGMMA tiles in the Phase-C GEMM; it's mechanical work, not
architectural, and orthogonal to the megakernel structure.

## Use it

```python
from lora_megakernel.model import LoRATrainStepKernel, AdamConfig

kernel = LoRATrainStepKernel.from_base_model(
    base_model="meta-llama/Llama-3.2-1B",
    lora_rank=8, vocab_size=256, hidden_size=64,
    max_seq_len=128,
)

out = kernel.train_step(
    context_tokens=torch.tensor([...], dtype=torch.int32, device="cuda"),
    target_tokens=torch.tensor([...], dtype=torch.int32, device="cuda"),
    adam=AdamConfig(lr=1e-3),
)
print(out["loss"], out["selected_log_probs"])
```

## Scope

This kernel implements the exact training math that the rl/backend
`TinyLoRAModel` runs — LoRA over a single output projection, frozen
embedding and base unembedding. That is a full training step for the
model rl/backend currently ships, but it is not a general LoRA adapter
over a full transformer stack. A megakernel for full-transformer LoRA
fine-tuning would need to fuse backward through every
attention/MLP/DeltaNet layer of the megakernel inference path that lives
in `../megakernel/` — months of work, not a session.
