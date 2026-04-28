# Slice B.3b status

The custom training-step backward (Tier 0.2 of the optimization
roadmap). End goal: replace HF+PEFT autograd in
`LoraMegakernelTrainer.forward_backward` with our own per-layer kernel
walk for ~3× faster training step (114 ms → ~30-50 ms, projected),
which pulls combined RL step toward ~18× over HF+fla.

## Wall-time bench (`bench_kernel_bwd.py`, hybrid routing 2026-04-28)

```
path                                   ms/step (3-run avg)
-----------------------------------------------------------
HF+PEFT autograd (default)              ~540 ms
Kernel-driven Slice B.3b                ~485 ms
-----------------------------------------------------------
Kernel path 1.08× – 1.15× FASTER on heterogeneous shapes.
```

### Cross-shape sweep (4 items per step, hybrid routing active)

```
                     HOMOGENEOUS (all 4 items same P, T)
S=12   HF=127.6  Kernel=126.9  speedup=1.005x   ✓
S=48   HF=127.9  Kernel=127.7  speedup=1.002x   ✓
S=192  HF=133.8  Kernel=128.1  speedup=1.044x   ✓
S=384  HF=128.8  Kernel=129.2  speedup=0.997x   ≈ (within noise)
S=640  HF=149.5  Kernel=130.3  speedup=1.147x   ✓

                     HETEROGENEOUS (mixed P, T)
mixed (8,4)/(32,16)/(16,8)/(24,12):
  HF=493.8  Kernel=472.0  speedup=1.046x        ✓
```

The kernel path defers to HF+PEFT's `_batched_logprobs` when items
share shapes (hybrid routing in `forward_backward`). PEFT packs all
items into a single B=N forward there; our kernel processes one item
at a time today, so HF wins decisively at homogeneous + small-S. Hand
HF the batched-shapes case; we win the heterogeneous case where
HF can't batch either.

To beat HF on homogeneous shapes too, the kernel needs batched-input
support (process [B, S] in one call instead of B sequential single-
sequence calls). Multi-day kernel refactor — separate workstream.

Progression across this session:

| commit              | kernel path | vs HF+PEFT       |
|---------------------|-------------|------------------|
| pre-FA-handrolled   |  674.5 ms   | 1.33× SLOWER     |
| FA bwd handrolled   |  663.1 ms   | 1.21× slower     |
| DN bwd handrolled   |  558.8 ms   | 1.09× FASTER     |
| **+npa reuse**      |  **~485 ms** | **~1.11× FASTER** |

The hand-rolled DN bwd (`dn_attn_handrolled.py`) replaces the autograd-
through-HF.linear_attn path with a Python forward that calls fla's
`chunk_gated_delta_rule_fwd / _bwd` directly (skipping autograd's
2.1× wrapping overhead) and chains through manual silu / conv1d /
sigmoid / softplus / RMSNormGated / linear bwds. 1.48× faster per
DN call vs the autograd path; cos > 0.9998 vs HF reference.

The hand-rolled FA bwd (`layer_attn_bwd_fa_handrolled`) similarly
replaces autograd-through-recomputed-SDPA with direct cuDNN FA-2 bwd
+ manual reverse-RoPE / reverse-QKnorm / reverse-gate / kernel-driven
LoRA bwd / kernel-driven RMSnorm bwd.

### DN bwd headroom (resolved 2026-04-27)

Direct fla calls vs autograd: 2.82 ms/call (autograd) vs 1.35 ms/call
(direct fwd+bwd) vs 0.74 ms/call (direct bwd only). The autograd
overhead through `chunk_gated_delta_rule.apply` was the ~150 ms culprit.

`dn_attn_handrolled.py` reproduces HF's linear_attn forward in Python
(matching cos > 0.9999 against HF's autograd path), saves intermediates,
and runs a hand-rolled bwd that calls `chunk_gated_delta_rule_bwd`
directly. Per-call timing: 5.5 ms (HF autograd) → 3.7 ms (ours) =
**1.48×**.

Forward chain:
  input rmsnorm → [in_proj_qkv | in_proj_z | in_proj_b | in_proj_a]
  → causal-conv1d → silu → split(q,k,v) → reshape per head
  → l2norm_fwd(q), l2norm_fwd(k)
  → sigmoid(b) = beta;  -A_log.exp() * softplus(a + dt_bias) = g_log
  → chunk_gated_delta_rule_fwd(q_l2, k_l2, v, g_log, beta) = y_pre
  → RMSNorm(y_pre, dn_norm_W) * silu(z) = attn_pre_o
  → out_proj  = attn_out

Backward chain (manual chain rule, no autograd):
  out_proj^T · d_attn_out → d_attn_pre_o
  reverse RMSNormGated (rmsnorm bwd + silu(z) bwd) → d_y_pre, d_z
  chunk_gated_delta_rule_bwd → dq, dk, dv, dbeta, dg
  l2norm_bwd → pre-l2norm dq, dk
  reshape + concat → dqkv_post_silu
  silu_bwd → d(qkv_pre_silu)
  conv_transpose1d → d(qkv_raw)  (depthwise causal conv bwd)
  in_proj_qkv^T · d(qkv_raw) → d_npa_from_qkv  (no LoRA on DN attn)
  in_proj_z^T · d_z          → d_npa_from_z
  sigmoid_bwd · in_proj_b^T  → d_npa_from_b
  softplus_bwd · in_proj_a^T → d_npa_from_a
  Σ → d_npa
  input rmsnorm bwd → d_h_in

**Why slower:** the autograd-based interior in `per_layer_bwd_dn`
recomputes the HF GatedDeltaNet forward inside the bwd to get the
autograd graph, so the kernel path does ~2× DN forward work (one in
our kernel, one in autograd). HF+PEFT does 1× forward + bwd through
the cached autograd graph.

The decomposition of the 167 ms slowdown:
  - 18 DN layers × ~10 ms HF-GatedDeltaNet recompute   ≈ 180 ms
  - 6 FA layers × ~5 ms QKV+SDPA recompute              ≈ 30 ms
  - Saved by no-full-model-autograd-graph build         ≈ -40 ms
  - Net                                                  ≈ +170 ms

The MLP bwd path (kernel-driven, no recompute) is actually faster
than HF+PEFT per layer; the win is being eaten by attention recompute.

**To make the kernel path faster than HF+PEFT** (projected: ~3× faster):

1. **Hand-rolled DN bwd kernel** consuming kernel-saved DN
   intermediates. Two flavours:
   - Recurrent: our existing `dn_bwd` kernel; needs `state_history`
     [N_DN, S+1, DN_HEADS, Dk, Dv] fp32 saved during forward
     (~18 GB at S=128 — workable but heavy).
   - Chunked: requires the chunked CUDA bwd port (Tier 1.6 — Python
     reference at `dn_chunked_bwd_proto.py`, CUDA pending) plus the
     existing `state_chunks` save from `dn_chunked_fwd`
     (~50 MB at S=128 — clean).
   Effort: 3-5 days for the recurrent path with `state_history`
   wired into prefill_bf16_train_step; or wait for chunked bwd port.

2. **Hand-rolled FA bwd via cuDNN** using saved Q, K, V, O, LSE
   from forward (kernel mod to save these 5 extra slabs per FA
   layer). cuDNN FA-2 bwd is what `fa_bwd_flash.py` already wraps —
   just need to feed it the saved tensors. Effort: ~1-2 days.

3. **Fused AdamW on the flat fp32 grad buffer** that
   `run_layer_walking_bwd` already produces. Replaces
   `torch.optim.AdamW` (~1.6 ms/step) with `fused_adamw_step`
   (124× faster per the doc). Effort: 0.5 day.

4. **CUDA Graph wrap** of the full kernel-only training step (after
   1-3 are in place — graph capture won't work while autograd is
   in the loop). Effort: ~1 day.

With all four shipped: training step projected ~30-50 ms (vs today's
507 ms HF+PEFT default), pulling combined RL step toward ~18× over
HF+fla. Today's commit is the validated gradient chain that those
optimizations have to match.

## ✅ Correctness-complete (commit 3745132, 2026-04-27)

End-to-end kernel-driven training step converges:

```
MEGAKERNEL_USE_KERNEL_BWD=1 test_kernel_bwd_e2e.py:
    step 1  loss=1.1860
    step 2  loss=0.3403
    step 3  loss=0.0700
    step 4  loss=0.0087
    step 5  loss=0.0014
[3] loss decreased 1.1860 → 0.0014 via kernel-driven bwd ✓
[4] sample after training: ', in a world where the sun was always shining brightly...'
```

vs. HF+PEFT autograd reference (`test_rl_trainer_e2e.py`):
`1.1835 → 0.0016 in 5 steps`.

Same convergence shape, same final loss tier. The kernel path
correctly threads gradients through 24 layers in reverse and
populates LoRA params' .grad for the standard torch.optim.AdamW
to consume. Production code path stays HF+PEFT (default); the
kernel path is opt-in via env var.

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
