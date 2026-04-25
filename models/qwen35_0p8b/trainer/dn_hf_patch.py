"""HF Qwen3-Next DeltaNet bwd acceleration: monkey-patch the
`chunk_gated_delta_rule` callable on each Qwen3NextGatedDeltaNet layer with
our CUDA kernel via `dn_autograd.deltanet_recurrence`.

HF's `chunk_gated_delta_rule(query, key, value, g, beta, ...)` semantics:
  - shapes: q,k,v in [B, S, H, D], beta/g in [B, S, H]
  - g is log-decay (so per-step decay = exp(g))
  - if use_qk_l2norm_in_kernel: q,k get l2-normalized along D
  - q is scaled by 1/sqrt(Dk)
Returns (core_attn_out [B, S, H, Dv], last_state [B, H, Dk, Dv] or None).

Our CUDA kernel takes [S, H, D], decay (not log-decay), no l2norm/scale.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from dn_autograd import deltanet_recurrence, deltanet_chunked_inference


def _l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def cuda_chunk_gated_delta_rule(
    query, key, value, g, beta,
    chunk_size=64,                 # ignored — we do exact recurrence
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    """Drop-in replacement for HF's chunk_gated_delta_rule with our CUDA
    recurrence kernel. Works for B>=1 by looping over batch (HF's training
    typically runs B=1 per device anyway)."""
    assert query.dim() == 4, "expected [B, S, H, D]"
    B, S, H, Dk = query.shape
    Dv = value.shape[-1]
    initial_dtype = query.dtype

    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key   = _l2norm(key,   dim=-1, eps=1e-6)
    scale = 1.0 / (Dk ** 0.5)
    query = query * scale

    decay = g.exp().contiguous()

    if query.dtype != torch.bfloat16: query = query.to(torch.bfloat16)
    if key.dtype   != torch.bfloat16: key   = key.to(torch.bfloat16)
    if value.dtype != torch.bfloat16: value = value.to(torch.bfloat16)
    if beta.dtype  != torch.float32:  beta  = beta.to(torch.float32)
    if decay.dtype != torch.float32:  decay = decay.to(torch.float32)

    # Inference-only fast path: when no input requires grad, use the chunked
    # tensor-core forward (3.4x over recurrent fwd at S=512). The chunked
    # path takes log-decay g, not decay = exp(g), so we need g (= log(decay))
    # which equals the original `g` argument before exp.
    inference_fast = (
        not torch.is_grad_enabled() or not any(
            t is not None and t.requires_grad
            for t in (query, key, value, beta, g)
        )
    )

    outputs = []
    final_states = []
    for b in range(B):
        q_b = query[b].contiguous()
        k_b = key  [b].contiguous()
        v_b = value[b].contiguous()
        beta_b  = beta [b].contiguous()
        decay_b = decay[b].contiguous()
        if initial_state is None:
            s0 = torch.zeros(H, Dk, Dv, dtype=torch.float32, device=query.device)
        else:
            s0 = initial_state[b].to(torch.float32).contiguous()
        if inference_fast:
            g_b = g[b].contiguous().to(torch.float32)
            y, sN = deltanet_chunked_inference(q_b, k_b, v_b, beta_b, g_b, s0)
        else:
            y, sN = deltanet_recurrence(q_b, k_b, v_b, beta_b, decay_b, s0)
        outputs.append(y)
        final_states.append(sN)
    out = torch.stack(outputs, dim=0).to(initial_dtype)
    last_state = torch.stack(final_states, dim=0) if output_final_state else None
    return out, last_state


def patch_hf_qwen3_deltanet(model) -> int:
    """Walk a HF Qwen3-Next model and monkey-patch every
    Qwen3NextGatedDeltaNet's chunk_gated_delta_rule with our fast version.

    Returns the number of layers patched."""
    n = 0
    for module in model.modules():
        cn = module.__class__.__name__
        if cn in ("Qwen3NextGatedDeltaNet", "Qwen3_5GatedDeltaNet"):
            module.chunk_gated_delta_rule = cuda_chunk_gated_delta_rule
            n += 1
    return n
