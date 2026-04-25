"""Chunked DeltaNet BACKWARD — Python algorithm reference (analytical).

Implements the backward of `dn_chunked_proto.chunked_fwd` using analytical
gradient formulas (no torch.autograd). This is the algorithmic reference
that the CUDA chunked backward kernel will port from.

Strategy: walk chunks IN REVERSE. Per chunk:
  1. Load state_in from state_chunks (saved by forward).
  2. Replay the forward to reproduce intermediates: T, v_new, attn_in,
     k_cd, attn_int, decay_mask, exp_g_cs, k_beta, v_beta.
  3. Apply backward formulas to compute per-step gradients dq, dk, dv,
     dbeta, dg, plus dstate_in to pass to the previous chunk.

The clever bit for T's backward: instead of differentiating through the
sequential row-update inner loop, use the matrix-inverse formula
  T = (I - tril(attn0))^{-1}   =>   d(I - tril(attn0)) = -T.T @ dT @ T.T
and then mask the result to the strict lower triangle to get d_attn0
restricted to the [i > j] sparsity pattern.

Validated against torch.autograd through `dn_chunked_proto.chunked_fwd`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch


class ChunkedBwdResult(NamedTuple):
    dq:    torch.Tensor   # [S, H, Dk] bf16
    dk:    torch.Tensor   # [S, H, Dk] bf16
    dv:    torch.Tensor   # [S, H, Dv] bf16
    dbeta: torch.Tensor   # [S, H]     fp32
    dg:    torch.Tensor   # [S, H]     fp32 (log-decay grad)
    dstate_init: torch.Tensor  # [H, Dk, Dv] fp32


def chunked_fwd_with_state_chunks(q, k, v, beta, g, state_init, *, chunk_size=64):
    """Same as dn_chunked_proto.chunked_fwd but ALSO returns
    state_chunks[H, n_chunks+1, Dk, Dv] (entry 0 is state_init, entry c+1
    is state at end of chunk c). Mirrors the CUDA fwd kernel's save."""
    from dn_chunked_proto import chunked_fwd as _orig_fwd
    # The Python proto doesn't currently emit state_chunks; reimplement so
    # we can capture them. Same algorithm step-for-step.
    S, H, Dk = q.shape
    Dv = v.shape[-1]
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
    state = state_init.clone().to(torch.float32)
    state_chunks = torch.zeros(H, n_chunks + 1, Dk, Dv,
                               dtype=torch.float32, device=q.device)
    state_chunks[:, 0] = state
    y_chunks = []
    g_c = g.view(n_chunks, C, H)
    g_cs_full = g_c.cumsum(dim=1)
    for c in range(n_chunks):
        q_h = q[c*C:(c+1)*C].permute(1, 0, 2).contiguous().to(torch.bfloat16)
        k_h = k[c*C:(c+1)*C].permute(1, 0, 2).contiguous().to(torch.bfloat16)
        v_h = v[c*C:(c+1)*C].permute(1, 0, 2).contiguous().to(torch.bfloat16)
        beta_h = beta[c*C:(c+1)*C].permute(1, 0).contiguous()
        gcs_h  = g_cs_full[c].permute(1, 0).contiguous()
        ecs_h  = gcs_h.exp()

        diff = gcs_h.unsqueeze(-1) - gcs_h.unsqueeze(-2)
        decay_mask = diff.exp()
        tril = torch.tril(torch.ones(C, C, dtype=torch.bool, device=q.device))
        decay_mask = decay_mask * tril.unsqueeze(0)
        diag = torch.eye(C, dtype=torch.bool, device=q.device)
        decay_strict = decay_mask * (~diag).unsqueeze(0)

        k_beta = (k_h.float() * beta_h.unsqueeze(-1)).to(torch.bfloat16)
        v_beta = (v_h.float() * beta_h.unsqueeze(-1)).to(torch.bfloat16)

        attn = -(k_beta.float() @ k_h.float().transpose(-1, -2))
        attn = attn * decay_strict
        for i in range(1, C):
            row = attn[..., i, :i].clone()
            sub = attn[..., :i, :i].clone()
            attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        eye = torch.eye(C, dtype=attn.dtype, device=attn.device)
        T = attn + eye
        T_bf = T.to(torch.bfloat16)
        v_new_inner = (T_bf.float() @ v_beta.float()).to(torch.bfloat16)
        k_beta_scaled = (k_beta.float() * ecs_h.unsqueeze(-1)).to(torch.bfloat16)
        k_cd = (T_bf.float() @ k_beta_scaled.float()).to(torch.bfloat16)
        state_bf = state.to(torch.bfloat16)
        v_prime = (k_cd.float() @ state_bf.float()).to(torch.bfloat16)
        v_new = (v_new_inner.float() - v_prime.float()).to(torch.bfloat16)
        attn_in = q_h.float() @ k_h.float().transpose(-1, -2)
        attn_in = attn_in * decay_mask
        q_scaled = (q_h.float() * ecs_h.unsqueeze(-1)).to(torch.bfloat16)
        attn_int = (q_scaled.float() @ state_bf.float()).to(torch.bfloat16)
        y_chunk = (attn_int.float() + attn_in.to(torch.bfloat16).float() @ v_new.float()).to(torch.bfloat16)
        y_chunks.append(y_chunk.permute(1, 0, 2).contiguous())

        g_total = gcs_h[..., -1]
        scale = (g_total.unsqueeze(-1) - gcs_h).exp()
        k_decay = (k_h.float() * scale.unsqueeze(-1)).to(torch.bfloat16)
        state = state * g_total.exp().view(H, 1, 1) + (
            k_decay.transpose(-1, -2).float() @ v_new.float()
        )
        state_chunks[:, c + 1] = state

    y = torch.cat(y_chunks, dim=0)[:S]
    return y, state, state_chunks


def chunked_bwd(q, k, v, beta, g, state_init, dy,
                state_chunks, *, chunk_size=64):
    """Analytical backward for chunked_fwd_with_state_chunks. Returns
    ChunkedBwdResult with all per-step grads. dy has the same shape as y
    ([S, H, Dv] bf16)."""
    S, H, Dk = q.shape
    Dv = v.shape[-1]
    pad = (-S) % chunk_size
    Sp = S + pad
    if pad:
        q  = torch.cat([q,  torch.zeros(pad, H, Dk, dtype=q.dtype,  device=q.device)],  dim=0)
        k  = torch.cat([k,  torch.zeros(pad, H, Dk, dtype=k.dtype,  device=k.device)],  dim=0)
        v  = torch.cat([v,  torch.zeros(pad, H, Dv, dtype=v.dtype,  device=v.device)],  dim=0)
        beta = torch.cat([beta, torch.zeros(pad, H, dtype=beta.dtype, device=beta.device)], dim=0)
        g    = torch.cat([g,    torch.zeros(pad, H, dtype=g.dtype,    device=g.device)],    dim=0)
        dy   = torch.cat([dy,   torch.zeros(pad, H, Dv, dtype=dy.dtype, device=dy.device)], dim=0)
    n_chunks = Sp // chunk_size
    C = chunk_size

    g_c    = g.view(n_chunks, C, H)
    g_cs_full = g_c.cumsum(dim=1)

    # Output grads (per-step, in [Sp, H, D] padded layout).
    dq    = torch.zeros(Sp, H, Dk, dtype=torch.float32, device=q.device)
    dk    = torch.zeros(Sp, H, Dk, dtype=torch.float32, device=q.device)
    dv    = torch.zeros(Sp, H, Dv, dtype=torch.float32, device=q.device)
    dbeta = torch.zeros(Sp, H,     dtype=torch.float32, device=q.device)
    dg    = torch.zeros(Sp, H,     dtype=torch.float32, device=q.device)

    # State gradient flowing back from the next chunk (final state had no
    # downstream consumer in the basic loss = sum(y * dy).fwd, so start at 0).
    d_state_next = torch.zeros(H, Dk, Dv, dtype=torch.float32, device=q.device)

    diag_bool = torch.eye(C, dtype=torch.bool, device=q.device)

    for c in reversed(range(n_chunks)):
        # ----- Load state_in for this chunk (= state at end of chunk c-1, or state_init if c==0) -----
        state_in = state_chunks[:, c].to(torch.float32)             # [H, Dk, Dv]

        # ----- Replay forward intermediates -----
        q_h = q[c*C:(c+1)*C].permute(1, 0, 2).contiguous().to(torch.bfloat16)   # [H, C, Dk]
        k_h = k[c*C:(c+1)*C].permute(1, 0, 2).contiguous().to(torch.bfloat16)   # [H, C, Dk]
        v_h = v[c*C:(c+1)*C].permute(1, 0, 2).contiguous().to(torch.bfloat16)   # [H, C, Dv]
        beta_h = beta[c*C:(c+1)*C].permute(1, 0).contiguous()                   # [H, C]
        gcs_h  = g_cs_full[c].permute(1, 0).contiguous()                        # [H, C]
        ecs_h  = gcs_h.exp()

        diff = gcs_h.unsqueeze(-1) - gcs_h.unsqueeze(-2)
        decay_mask = diff.exp()
        tril = torch.tril(torch.ones(C, C, dtype=torch.bool, device=q.device))
        decay_mask = decay_mask * tril.unsqueeze(0)
        decay_strict = decay_mask * (~diag_bool).unsqueeze(0)

        k_beta = (k_h.float() * beta_h.unsqueeze(-1))           # [H, C, Dk]
        v_beta = (v_h.float() * beta_h.unsqueeze(-1))           # [H, C, Dv]

        attn = -(k_beta @ k_h.float().transpose(-1, -2)) * decay_strict
        for i in range(1, C):
            row = attn[..., i, :i].clone()
            sub = attn[..., :i, :i].clone()
            attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        eye = torch.eye(C, dtype=attn.dtype, device=attn.device)
        T = attn + eye                                          # [H, C, C]
        v_new_inner = T @ v_beta                                # [H, C, Dv]
        k_beta_scaled = k_beta * ecs_h.unsqueeze(-1)            # [H, C, Dk]
        k_cd = T @ k_beta_scaled                                # [H, C, Dk]
        v_prime = k_cd @ state_in                               # [H, C, Dv]
        v_new = v_new_inner - v_prime                           # [H, C, Dv]
        attn_in_pre = q_h.float() @ k_h.float().transpose(-1, -2)
        attn_in = attn_in_pre * decay_mask
        q_scaled = q_h.float() * ecs_h.unsqueeze(-1)            # [H, C, Dk]
        attn_int = q_scaled @ state_in                          # [H, C, Dv]

        # ----- Backward -----
        d_y_chunk = dy[c*C:(c+1)*C].permute(1, 0, 2).contiguous().float()   # [H, C, Dv]

        # (B) y_chunk = attn_int + attn_in @ v_new
        d_attn_int = d_y_chunk
        d_attn_in  = d_y_chunk @ v_new.transpose(-1, -2)          # [H, C, C]
        d_v_new    = attn_in.transpose(-1, -2) @ d_y_chunk        # [H, C, Dv]

        # (A) state_out = state_in * exp(g_total) + k_decay.T @ v_new
        g_total   = gcs_h[..., -1]                                # [H]
        exp_total = g_total.exp()                                 # [H]
        scale     = (g_total.unsqueeze(-1) - gcs_h).exp()         # [H, C]
        k_decay   = k_h.float() * scale.unsqueeze(-1)             # [H, C, Dk]

        d_state_in = d_state_next * exp_total.view(H, 1, 1)       # [H, Dk, Dv]
        # d g_total via state * exp(g_total).
        state_in_decayed = state_in * exp_total.view(H, 1, 1)
        d_g_total_acc = (d_state_next * state_in_decayed).sum(dim=(-1, -2))   # [H]

        # k_decay.T @ v_new => d_v_new += k_decay @ d_state_next; d_k_decay = v_new @ d_state_next.T
        d_v_new = d_v_new + k_decay @ d_state_next                # [H, C, Dv]
        d_k_decay = v_new @ d_state_next.transpose(-1, -2)        # [H, C, Dk]
        # k_decay = k * exp(g_total - g_cs)
        d_k_via_decay = d_k_decay * scale.unsqueeze(-1)
        # d g_total += sum_d d_k_decay[h, j, d] * k_decay[h, j, d]   (sum over j and d)
        prod = (d_k_decay * k_decay).sum(dim=-1)                  # [H, C]
        d_g_total_acc = d_g_total_acc + prod.sum(dim=-1)          # [H]
        # d g_cs[h, j] -= prod[h, j]
        d_g_cs = -prod                                            # [H, C]

        # (D) v_new = v_new_inner - v_prime
        d_v_new_inner = d_v_new
        d_v_prime = -d_v_new

        # (E) v_prime = k_cd @ state_in
        d_k_cd = d_v_prime @ state_in.transpose(-1, -2)           # [H, C, Dk]
        d_state_in = d_state_in + k_cd.transpose(-1, -2) @ d_v_prime  # [H, Dk, Dv]

        # (F) v_new_inner = T @ v_beta
        d_T = d_v_new_inner @ v_beta.transpose(-1, -2)            # [H, C, C]
        d_v_beta = T.transpose(-1, -2) @ d_v_new_inner            # [H, C, Dv]

        # (G) k_cd = T @ k_beta_scaled
        d_T = d_T + d_k_cd @ k_beta_scaled.transpose(-1, -2)      # [H, C, C]
        d_k_beta_scaled = T.transpose(-1, -2) @ d_k_cd            # [H, C, Dk]
        # k_beta_scaled = k_beta * exp_g_cs   =>
        d_k_beta_via_kcd = d_k_beta_scaled * ecs_h.unsqueeze(-1)
        # d exp_g_cs[h, j] = sum_d d_k_beta_scaled[h, j, d] * k_beta[h, j, d]
        d_exp_g_cs = (d_k_beta_scaled * k_beta).sum(dim=-1)       # [H, C]

        # (I) attn_int = q_scaled @ state_in
        d_q_scaled = d_attn_int @ state_in.transpose(-1, -2)      # [H, C, Dk]
        d_state_in = d_state_in + q_scaled.transpose(-1, -2) @ d_attn_int

        d_q_via_attn_int = d_q_scaled * ecs_h.unsqueeze(-1)
        d_exp_g_cs = d_exp_g_cs + (d_q_scaled * q_h.float()).sum(dim=-1)  # [H, C]

        # (J) attn_in = attn_in_pre * decay_mask
        d_attn_in_pre = d_attn_in * decay_mask
        d_decay_mask  = d_attn_in * attn_in_pre

        # attn_in_pre = q @ k.T
        d_q_via_attn_in = d_attn_in_pre @ k_h.float()              # [H, C, Dk]
        d_k_via_attn_in = d_attn_in_pre.transpose(-1, -2) @ q_h.float()    # [H, C, Dk]

        # Combine d_q
        d_q = d_q_via_attn_int + d_q_via_attn_in                  # [H, C, Dk]

        # (L) d_T propagates to d_attn0 via matrix-inverse formula.
        # T = (I - tril(attn0))^{-1}. Let A = I - tril(attn0). Then T = A^{-1}.
        # dA = -A^{-T} dT A^{-T} = -T.T @ dT @ T.T.
        # d_attn0 = -dA, masked to strict lower triangle (i > j).
        dA = -T.transpose(-1, -2) @ d_T @ T.transpose(-1, -2)     # [H, C, C]
        d_attn0 = -dA
        # Mask to strict lower (i > j); diagonal and above are zero.
        strict_mask = torch.tril(torch.ones(C, C, dtype=d_attn0.dtype,
                                            device=d_attn0.device), diagonal=-1)
        d_attn0 = d_attn0 * strict_mask.unsqueeze(0)

        # attn0 = -(k_beta @ k.T) * decay_strict
        d_kbeta_kT = -d_attn0 * decay_strict                      # [H, C, C]
        d_decay_strict = -d_attn0 * (k_beta @ k_h.float().transpose(-1, -2))
        # k_beta @ k.T:
        d_k_beta_via_attn0 = d_kbeta_kT @ k_h.float()             # [H, C, Dk]
        d_k_via_attn0      = d_kbeta_kT.transpose(-1, -2) @ k_beta # [H, C, Dk]

        # Combine d_k
        d_k_total = d_k_via_attn_in + d_k_via_attn0 + d_k_via_decay
        # Combine d_k_beta from kcd path + attn0 path:
        d_k_beta = d_k_beta_via_kcd + d_k_beta_via_attn0

        # (M) k_beta = beta * k, v_beta = beta * v
        d_beta_via_kbeta = (d_k_beta * k_h.float()).sum(dim=-1)   # [H, C]
        d_k_via_kbeta    = d_k_beta * beta_h.unsqueeze(-1)
        d_beta_via_vbeta = (d_v_beta * v_h.float()).sum(dim=-1)   # [H, C]
        d_v_via_vbeta    = d_v_beta * beta_h.unsqueeze(-1)

        d_k_total = d_k_total + d_k_via_kbeta
        d_v_total = d_v_via_vbeta
        d_beta = d_beta_via_kbeta + d_beta_via_vbeta

        # (N) decay/decay_strict -> g_cs gradient.
        # decay_mask[h, i, j] = exp(g_cs[h, i] - g_cs[h, j]) for i >= j else 0
        # decay_strict same, with diagonal i == j zero.
        # Apply both: decay_mask contributes from attn_in path; decay_strict from attn0.
        # d g_cs[h, i] += sum_{j <= i} d_decay_mask[h, i, j] * decay_mask[h, i, j]
        # d g_cs[h, j] -= sum_{i >= j} d_decay_mask[h, i, j] * decay_mask[h, i, j]
        # (and similarly for decay_strict)
        # Combine into a single d_decay_eff; both masks are zero outside their
        # supports, so summing the products is safe.
        d_decay_eff = d_decay_mask * decay_mask + d_decay_strict * decay_strict
        d_g_cs = d_g_cs + d_decay_eff.sum(dim=-1)                 # [H, C]  (sum over j)
        d_g_cs = d_g_cs - d_decay_eff.sum(dim=-2)                 # subtract sum over i

        # (O) exp_g_cs = exp(g_cs)
        d_g_cs = d_g_cs + d_exp_g_cs * ecs_h

        # (P) g_total = g_cs[C-1]
        # d_g_total_acc was accumulated into; route into d_g_cs[C-1].
        d_g_cs[..., -1] = d_g_cs[..., -1] + d_g_total_acc

        # (Q) g_cs = cumsum(g) within chunk -> d_g[t] = sum_{tau >= t} d_g_cs[tau]
        # equivalent: d_g = flip(cumsum(flip(d_g_cs)))
        d_g_chunk = torch.flip(torch.cumsum(torch.flip(d_g_cs, dims=[-1]), dim=-1), dims=[-1])

        # ----- Write per-step grads back into [Sp, H, D] layout -----
        dq[c*C:(c+1)*C]    = d_q.permute(1, 0, 2)
        dk[c*C:(c+1)*C]    = d_k_total.permute(1, 0, 2)
        dv[c*C:(c+1)*C]    = d_v_total.permute(1, 0, 2)
        dbeta[c*C:(c+1)*C] = d_beta.permute(1, 0)
        dg[c*C:(c+1)*C]    = d_g_chunk.permute(1, 0)

        # Pass d_state_in to the previous chunk.
        d_state_next = d_state_in

    return ChunkedBwdResult(
        dq=dq[:S].to(torch.bfloat16),
        dk=dk[:S].to(torch.bfloat16),
        dv=dv[:S].to(torch.bfloat16),
        dbeta=dbeta[:S],
        dg=dg[:S],
        dstate_init=d_state_next,
    )
