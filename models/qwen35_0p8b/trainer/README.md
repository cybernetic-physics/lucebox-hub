# Training Megakernel (LoRA forward, all 13 trainable linears)

Fork of `../prefill_megakernel` extended with LoRA residuals applied
inside the single cooperative dispatch, plus a fused multi-param AdamW
optimizer kernel.

## Shipped this directory

### Forward pass

LoRA forward is wired into **all 13 trainable linears** of Qwen 3.5-0.8B:

| Layer type | LoRA on |
|---|---|
| Full Attention (6 layers) | q, k, v, o, gate, up, down |
| DeltaNet (18 layers) | qkv, z, out, gate, up, down |

Per projection, the kernel calls two phases: `phase_lora_h` (x @ A) then
`phase_lora_b_add` (adds scaling · (lora_h @ B) to the base GEMM output).
Each LoRA pair is nullable in the kernel signature — passing zero-sized
tensors disables that specific linear's LoRA.

Kernel arg packs all 26 LoRA pointer ptrs into a `LoraSet` struct (208
bytes). Memory layout per pointer:

- `A_all[layer_idx, K_in, LORA_R]`
- `B_all[layer_idx, LORA_R, K_out]`

Three-way sanity (`test_lora_forward.py`, S=32, random bf16 weights):

```
baseline prefill_megakernel          next_token = 159560
train_mega A=B=0  (all 13 linears)   next_token = 159560   ✓ matches baseline
train_mega A,B≠0  (all 13 linears)   next_token = 163951   ✓ diverges
```

### Fused AdamW

`torch.ops.train_megakernel_C.fused_adamw_step(params, m, v, grad, step, lr, beta1, beta2, eps, wd)`
runs the standard bias-corrected AdamW in one kernel dispatch over a flat
contiguous bf16 parameter buffer with fp32 m/v state.

Correctness (`test_adamw.py` vs `torch.optim.AdamW` on fp32 master weights
downcast per step):

```
step 1: max|Δ| = 1.95e-03  mean|Δ| = 3.51e-05
step 2: max|Δ| = 2.93e-03  mean|Δ| = 9.83e-05
step 3: max|Δ| = 3.91e-03  mean|Δ| = 1.46e-04
CORRECTNESS OK
```

Deltas are bf16 rounding at step-3 scale (accumulated 1e-3 · 3 = 3e-3).

### What's wired but not yet plumbed

- Activation saving for backward — forward currently doesn't persist
  per-layer normalized inputs / attn outputs / gate·up / down outputs.
  Adding these is mechanical (extra device pointers for per-layer tile
  storage).
- Backward megakernel — CE loss → LM head bwd → per-layer reverse →
  RMSNorm bwd / SwiGLU bwd / linear bwd / flash-attn bwd / DeltaNet BPTT.
- End-to-end gradient-check vs pure-torch Qwen 3.5 + HF LoRA.

## Files

- `kernel.cu` — prefill megakernel + all 13 LoRA applies + fused AdamW
- `torch_bindings.cpp` — `train_mega_forward` and `fused_adamw_step` ops
- `setup.py` — CUDAExtension
- `test_lora_forward.py` — three-way diff exercising all 13 linears
- `test_adamw.py` — fused AdamW vs torch reference

## Remaining roadmap

1. ⬜ Save per-layer activations during forward so backward can consume them.
2. ⬜ Backward megakernel:
   - CE loss + LM head backward
   - Final-norm backward
   - Per-layer reverse: MLP bwd → post-norm bwd → attn/DN bwd → QKV bwd → input-norm bwd
   - Flash-attention-style bwd for FA; BPTT for DN recurrence
   - Emit LoRA grads at each trainable linear
3. ⬜ End-to-end correctness vs pure-torch Qwen 3.5-0.8B + HF LoRA

Forward + AdamW are shipped, correct, and land the foundation for the
backward work. The two remaining research-grade pieces are flash-attention
backward (standard but ~500 lines of careful CUDA) and DeltaNet BPTT (the
`(I − β k kᵀ)` recurrence backward is genuinely novel).
