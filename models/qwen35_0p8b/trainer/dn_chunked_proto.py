"""Chunked DeltaNet forward — Python prototype.

Mirrors HF's torch_chunk_gated_delta_rule but stays in bf16 instead of
casting to fp32. The matmuls dispatch to cuBLAS bf16-input + fp32-accum
tensor cores on B200.

Inputs match our autograd wrapper (per-head views the kernel will use):
  q, k     : [S, H, Dk] bf16
  v        : [S, H, Dv] bf16
  beta     : [S, H]     fp32
  g        : [S, H]     fp32 (log-decay)
  state_init: [H, Dk, Dv] fp32

Outputs:
  y          : [S, H, Dv] bf16
  state_out  : [H, Dk, Dv] fp32

This is the algorithmic reference we'll port to CUDA tensor cores.
Validated against the recurrent CUDA kernel for cos > 0.999 numerically.
"""
from __future__ import annotations

import torch


def chunked_fwd(q, k, v, beta, g, state_init, *, chunk_size=64):
    """Chunked forward. q,k,v are [S, H, D]; beta,g are [S, H];
    state_init is [H, Dk, Dv]. Returns y[S, H, Dv] bf16, state_out[H, Dk, Dv] fp32."""
    S, H, Dk = q.shape
    Dv = v.shape[-1]
    assert k.shape == (S, H, Dk) and v.shape == (S, H, Dv)
    assert beta.shape == (S, H) and g.shape == (S, H)

    # Pad S up to multiple of chunk_size.
    pad = (-S) % chunk_size
    if pad:
        q = torch.cat([q, torch.zeros(pad, H, Dk, dtype=q.dtype, device=q.device)], dim=0)
        k = torch.cat([k, torch.zeros(pad, H, Dk, dtype=k.dtype, device=k.device)], dim=0)
        v = torch.cat([v, torch.zeros(pad, H, Dv, dtype=v.dtype, device=v.device)], dim=0)
        beta = torch.cat([beta, torch.zeros(pad, H, dtype=beta.dtype, device=beta.device)], dim=0)
        g    = torch.cat([g,    torch.zeros(pad, H, dtype=g.dtype, device=g.device)], dim=0)
    Sp = S + pad
    n_chunks = Sp // chunk_size
    C = chunk_size

    # Reshape to per-chunk: [n_chunks, C, H, D]
    q_c    = q.view(n_chunks, C, H, Dk)
    k_c    = k.view(n_chunks, C, H, Dk)
    v_c    = v.view(n_chunks, C, H, Dv)
    beta_c = beta.view(n_chunks, C, H)
    g_c    = g   .view(n_chunks, C, H)

    # Cumulative g within each chunk: g_cs[c, t, h] = sum_{i<=t} g_c[c, i, h]
    g_cs   = g_c.cumsum(dim=1)
    exp_cs = g_cs.exp()                                         # [n_chunks, C, H]

    state = state_init.clone().to(torch.float32)                # [H, Dk, Dv]
    y_chunks = []

    # decay_mask[i, j] = exp(g_cs[i] - g_cs[j]) if i >= j else 0
    # Build per-chunk: [n_chunks, H, C, C]
    diff = g_cs.unsqueeze(2) - g_cs.unsqueeze(1)                # [n_chunks, C, C, H]
    diff = diff.permute(0, 3, 1, 2).contiguous()                # [n_chunks, H, C, C]
    decay_mask = diff.exp()
    tril = torch.tril(torch.ones(C, C, dtype=torch.bool, device=q.device))
    decay_mask = decay_mask * tril.unsqueeze(0).unsqueeze(0)     # zero above diag
    # For -(k_beta @ k.T) we additionally zero the DIAGONAL (HF zeros diag-and-above
    # via masked_fill(triu(diag=0), 0)). For q @ k.T (attn_in) the diagonal is
    # KEPT — that intra-chunk lookback uses the diagonal step.
    diag_mask = torch.eye(C, dtype=torch.bool, device=q.device)
    decay_strict = decay_mask * (~diag_mask).unsqueeze(0).unsqueeze(0)

    for c in range(n_chunks):
        # Per-chunk views (squeeze chunk dim).
        q_b    = q_c[c]      # [C, H, Dk] bf16
        k_b    = k_c[c]      # [C, H, Dk]
        v_b    = v_c[c]      # [C, H, Dv]
        beta_b = beta_c[c]   # [C, H]
        gcs_b  = g_cs[c]     # [C, H]
        ecs_b  = exp_cs[c]   # [C, H]
        dmask        = decay_mask[c]    # [H, C, C] keeps diag (for q@k.T)
        dmask_strict = decay_strict[c]  # [H, C, C] zeros diag (for k_beta@k.T)

        # Move H to leading axis for batched matmul over heads.
        q_h    = q_b.permute(1, 0, 2).contiguous().to(torch.bfloat16)   # [H, C, Dk]
        k_h    = k_b.permute(1, 0, 2).contiguous().to(torch.bfloat16)   # [H, C, Dk]
        v_h    = v_b.permute(1, 0, 2).contiguous().to(torch.bfloat16)   # [H, C, Dv]
        beta_h = beta_b.permute(1, 0).contiguous()                      # [H, C]
        ecs_h  = ecs_b.permute(1, 0).contiguous()                       # [H, C]

        k_beta = (k_h.float() * beta_h.unsqueeze(-1)).to(torch.bfloat16)  # [H, C, Dk]
        v_beta = (v_h.float() * beta_h.unsqueeze(-1)).to(torch.bfloat16)  # [H, C, Dv]

        # attn = -(k_beta @ k.T) * decay_mask, with diag-and-above zeroed.
        attn = -(k_beta.float() @ k_h.float().transpose(-1, -2))         # [H, C, C]
        attn = attn * dmask_strict                                       # zero diag and above
        # Construct T from attn via the standard sequential update.
        for i in range(1, C):
            row = attn[..., i, :i].clone()
            sub = attn[..., :i, :i].clone()
            attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        # T = attn + I
        eye = torch.eye(C, dtype=attn.dtype, device=attn.device)
        T = attn + eye                                                   # [H, C, C]
        T_bf = T.to(torch.bfloat16)

        # v_new = T @ v_beta
        v_new = (T_bf.float() @ v_beta.float()).to(torch.bfloat16)       # [H, C, Dv]
        # k_cd = T @ (k_beta * exp_g_cs)
        k_beta_scaled = (k_beta.float() * ecs_h.unsqueeze(-1)).to(torch.bfloat16)
        k_cd = (T_bf.float() @ k_beta_scaled.float()).to(torch.bfloat16) # [H, C, Dk]

        # v_prime = k_cd @ state
        state_bf = state.to(torch.bfloat16)                              # [H, Dk, Dv]
        v_prime = (k_cd.float() @ state_bf.float()).to(torch.bfloat16)   # [H, C, Dv]
        v_new = (v_new.float() - v_prime.float()).to(torch.bfloat16)

        # attn_in = q @ k.T * decay_mask  (zero above diag)
        attn_in = q_h.float() @ k_h.float().transpose(-1, -2)
        attn_in = attn_in * dmask
        attn_in_bf = attn_in.to(torch.bfloat16)

        # attn_int = (q * exp_g_cs) @ state
        q_scaled = (q_h.float() * ecs_h.unsqueeze(-1)).to(torch.bfloat16)
        attn_int = (q_scaled.float() @ state_bf.float()).to(torch.bfloat16)

        # y_chunk = attn_int + attn_in @ v_new
        y_chunk = (attn_int.float() + attn_in_bf.float() @ v_new.float()).to(torch.bfloat16)

        # Store y for this chunk in [C, H, Dv] layout.
        y_chunk_chw = y_chunk.permute(1, 0, 2).contiguous()              # [C, H, Dv]
        y_chunks.append(y_chunk_chw)

        # Update state:
        #   state = state * exp(g_chunk_total) + (k * exp(g_total - g_cs))^T @ v_new
        g_total_h = gcs_b[-1]                                            # [H]
        exp_total = g_total_h.exp().view(H, 1, 1)
        state_decayed = state * exp_total
        # k_decay[h, t, d] = k[h, t, d] * exp(g_total - g_cs[h, t])
        scale = (g_total_h.unsqueeze(0) - gcs_b).exp().permute(1, 0).contiguous()  # [H, C]
        k_decay = (k_h.float() * scale.unsqueeze(-1)).to(torch.bfloat16) # [H, C, Dk]
        # state += k_decay.T @ v_new   ([H, Dk, C] @ [H, C, Dv] -> [H, Dk, Dv])
        state = state_decayed + (k_decay.transpose(-1, -2).float() @ v_new.float())

    y = torch.cat(y_chunks, dim=0)[:S]                                   # [S, H, Dv]
    return y, state
