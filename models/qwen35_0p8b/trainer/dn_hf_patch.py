"""HF Qwen3-Next DeltaNet acceleration: monkey-patch the
`chunk_gated_delta_rule` callable to our fast path.

Hybrid routing (this is the honest version after benching against fla):

  - INFERENCE, short S (default <=512): our chunked CUDA kernel
    (1.25-1.28x faster than fla at S<=512, validated by bench_vs_fla.py)
  - INFERENCE, long S: fall through to fla (faster at S>=1024)
  - TRAINING (autograd needed): fall through to fla (its chunked Triton
    bwd is much faster than our scalar recurrent CUDA bwd at S>=256)

When fla isn't available, we use our recurrent CUDA path everywhere as
the fallback (which still beats HF's torch fp32 fallback by 5-7x).

HF's `chunk_gated_delta_rule(query, key, value, g, beta, ...)` semantics:
  - shapes: q,k,v in [B, S, H, D], beta/g in [B, S, H]
  - g is log-decay (so per-step decay = exp(g))
  - if use_qk_l2norm_in_kernel: q,k get l2-normalized along D
  - q is scaled by 1/sqrt(Dk)
Returns (core_attn_out [B, S, H, Dv], last_state [B, H, Dk, Dv] or None).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from dn_autograd import deltanet_recurrence, deltanet_chunked_inference

# Try to import fla once at module-load time. If available, the hybrid
# routing uses it for paths where it dominates (training, long-S inference).
try:
    # Apply the torch.compile shim first (fla 0.5.0 + torch 2.11 issue).
    import _fla_torch_compile_shim  # noqa: F401
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _fla_chunk
    _HAS_FLA = True
except Exception:
    _fla_chunk = None
    _HAS_FLA = False


# At what S do we switch from our chunked fwd to fla for inference?
# Bench data: ours wins up to S=512 (1.27x at S=512); fla wins at S=1024.
_OURS_INFER_MAX_S = 512


def _l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def cuda_chunk_gated_delta_rule(
    query, key, value, g, beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    """Hybrid drop-in replacement for HF's chunk_gated_delta_rule.

    Routes to the fastest available path based on shape + mode (see
    module docstring). Bit-exact to HF's reference (modulo bf16 noise)
    on whichever path is taken."""
    assert query.dim() == 4, "expected [B, S, H, D]"
    B, S, H, Dk = query.shape
    Dv = value.shape[-1]
    initial_dtype = query.dtype

    # Hybrid routing: pick the path FIRST, then preprocess only for our path.
    needs_grad = torch.is_grad_enabled() and any(
        t is not None and t.requires_grad
        for t in (query, key, value, beta, g)
    )
    use_fla = _HAS_FLA and (needs_grad or S > _OURS_INFER_MAX_S)
    if use_fla:
        # fla wants RAW q/k/v (no l2norm/scale yet) — it does its own.
        # And fla wants bf16 q/k/v, fp32 g/beta.
        q_in = query if query.dtype == torch.bfloat16 else query.to(torch.bfloat16)
        k_in = key   if key.dtype   == torch.bfloat16 else key.to(torch.bfloat16)
        v_in = value if value.dtype == torch.bfloat16 else value.to(torch.bfloat16)
        g_in = g     if g.dtype     == torch.float32  else g.to(torch.float32)
        b_in = beta  if beta.dtype  == torch.float32  else beta.to(torch.float32)
        out_y, out_state = _fla_chunk(
            q_in.contiguous(), k_in.contiguous(), v_in.contiguous(),
            g=g_in.contiguous(), beta=b_in.contiguous(),
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        return out_y.to(initial_dtype), out_state

    # Our path: preprocess (l2norm, scale, exp(g)) then dispatch per batch.
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
    inference = not needs_grad

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
        if inference:
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
