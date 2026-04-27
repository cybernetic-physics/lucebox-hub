# Slice B.3b status

The custom training-step backward (Tier 0.2 of the optimization
roadmap). End goal: replace HF+PEFT autograd in
`LoraMegakernelTrainer.forward_backward` with our own per-layer kernel
walk for ~3× faster training step (114 ms → ~30-50 ms, projected),
which pulls combined RL step toward ~18× over HF+fla.

## Validated foundations (shipped)

The path from loss to layer-output-residual gradient is solid:

| piece                         | commit  | validation                       |
|-------------------------------|---------|----------------------------------|
| pack_peft_to_flat / unpack    | ea28c8a | round-trip bit-exact (26 tensors) |
| Kernel saves (4 → 6 slabs)    | 576d092 | rms_norm consistency 24/24 layers |
| kernel_loss_autograd          | 17346fb | d(h_pre_norm) cos=0.99974 vs HF+PEFT |
| layer_mlp_bwd                 | e22cf35 | 7 grads, cos=0.99999 vs autograd  |
| layer_attn_bwd_fa             | 9a5ee12 | 9 grads, cos=1.00000 vs autograd  |
| **per_layer_bwd_fa**          | eafd3f6 | **15 grads, cos>0.99996 end-to-end** |

For FA layers, the per-layer reverse walk is **correctness-complete**.
Given a layer's saved activations + LoRA tensors + base weights + the
upstream gradient `dh_out`, `per_layer_bwd_fa` produces the gradient
flowing out (`dh_in`) plus all 7 LoRA pair gradients (q, k, v, o,
gate, up, down — A and B each). Validated bit-tight against torch
autograd through the recomputed forward.

## Remaining pieces

### per_layer_bwd_dn — DN-layer reverse walk

**Status: not yet implemented.**

The DN-layer forward pipeline is more complex than FA:

```
hidden_in
  → input rmsnorm                         → normalized_in
  → in_proj_qkv (linear + LoRA)           → qkv_raw   [S, DN_CONV_CH = 6144]
  → in_proj_z   (linear + LoRA)           → z         [S, DN_V_SIZE  = 2048]
  → depthwise conv1d (kernel=4, causal)   on qkv_raw
  → SiLU                                  → qkv_post
  → pf_dn_prep:                           splits qkv_post into:
      q_post, k_post, v_post                      [S, DN_HEADS, DN_KEY/VAL]
      L2-normalize q, k                            (built into fla path)
      beta = sigmoid(beta_proj_logits)             scalar gate per (pos, head)
      decay = exp(decay_proj_logits + dt_bias) ... per (pos, head)
  → fla.chunk_gated_delta_rule(q, k, v, beta, decay, initial_state)
                                          → y_pre   [S, DN_HEADS, DN_VAL]
  → DN gnorm (per-head RMSnorm with shared dn_norm weight)
  → sigmoid(z) * y_normed                 → attn_out_pre_o  [S, DN_V_SIZE]
  → out_proj (linear + LoRA)              → attn_out
  → residual                              → h_post_attn = hidden_in + attn_out
  → MLP block (same as FA)
```

To implement the bwd:

- **Recommended (correctness path, like layer_attn_bwd_fa did)**:
  recompute the DN forward in torch ops + fla's chunk_gated_delta_rule
  under autograd, run backward, read grads. fla's chunk_gated_delta_rule
  is an `autograd.Function` so this works.
- **Faster path (later)**: hand-rolled chain rule using our recurrent
  `dn_bwd` kernel (or the chunked CUDA bwd port — Tier 1.6).

Estimated effort: 1-2 days (mostly DN forward reproduction + fla calling
conventions + per-projection LoRA bwd plumbing).

### run_layer_walking_bwd — top-level driver

**Status: not yet implemented.**

```python
def run_layer_walking_bwd(
    grad_h_pre_norm: torch.Tensor,       # from kernel_loss_autograd
    saves: dict,                          # 6 activation slabs
    lora_flat: list[torch.Tensor],        # 26 packed bf16 tensors
    base_weights_handle: BaseModelHandle,
    lora_rank: int, lora_scaling: float,
) -> dict[str, torch.Tensor]:             # returns 26 flat-tensor grads
```

For L = NUM_LAYERS-1 down to 0:
- look up LAYER_TYPE[L]
- slice saves at L, slice lora_flat per type at fa_idx or dn_idx
- per_layer_bwd_fa or per_layer_bwd_dn
- accumulate the 12-13 returned LoRA grads into the flat 26-tensor
  gradient buffer at the appropriate (kernel_idx, ...) slice

### LoRA grad scatter / gather to flat fp32 buffer

**Status: not yet implemented.**

For fused AdamW (`launch_fused_adamw`, 124× faster than torch.optim.AdamW),
need a single contiguous flat fp32 buffer:

- **forward path**: pack_peft_to_flat already gives 26 separate tensors.
  Easy to flatten if needed.
- **backward path**: each `per_layer_bwd_*` returns per-projection grads
  for a single layer. Scatter into the 26-tensor flat buffer at the
  layer's slice (kernel_idx).
- **AdamW**: takes flat bf16 params, fp32 grad, fp32 m/v as four
  contiguous buffers; updates in place.

### Trainer integration

**Status: not yet implemented.**

In `LoraMegakernelTrainer.forward_backward`:

```python
out = kernel_loss_autograd(handle, prompt, targets, lora_flat, ...)
loss = out["loss"]
grad_h_pre_norm = out["grad_h_pre_norm"]
saves = out["saves"]
flat_grads = run_layer_walking_bwd(grad_h_pre_norm, saves,
                                    lora_flat, handle, ...)
# flat_grads: 26-element list of fp32 grad tensors
unpack_flat_to_peft(peft_model, ...)  # if optimizer is torch.optim
# OR: stream grads into flat fp32 grad buffer + run fused_adamw_step
```

## Per-piece effort estimate (engineer-days)

| piece                   | days |
|-------------------------|------|
| per_layer_bwd_dn        |  1-2 |
| run_layer_walking_bwd   |  0.5 |
| LoRA grad scatter       |  0.5 |
| Trainer integration     |  0.5-1 |
| Convergence test + bench|  0.5 |
| **Total remaining**     |  **3-5 days** |

Plus optional follow-ups for ~2-3× more speed:
- Replace torch-autograd interior in `layer_attn_bwd_fa` with hand-rolled
  cuDNN FA-bwd + manual QKnorm/RoPE chain rules: ~1 day
- Replace torch-autograd interior in `layer_dn_bwd_fa` with hand-rolled
  recurrent dn_bwd + manual conv1d-bwd + custom gnorm bwd: ~2-3 days
- Wire fused_adamw on the flat buffer: ~0.5 day
- CUDA Graph wrap of full training step once kernels-only: ~1 day
