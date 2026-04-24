"""DeltaNet backward — chunked BPTT reference.

The DeltaNet recurrence from Qwen3.5 (per head):
  state_t = decay_t * state_{t-1} + k_t ⊗ delta_t
  delta_t = beta_t * (v_t - state_{t-1} . k_t)   # [Dv]-vector
  out_t   = q_t . state_t                         # [Dv]-vector

where state is [Dk, Dv], q,k in [Dk], v in [Dv], beta,decay scalars per
step.

Backward (BPTT, given dy_t) computed here symbolically via torch's
autograd on the unrolled recurrence. Correct at any S + chunk size.
Slow at long S because the loop is Python-level; the kernel version
should be a ~C-length inner chunk with bf16 WGMMA for the chunk
matmuls, and a prefix-scan outer loop — the standard "chunked
gated-delta-net" pattern.

This module is the correctness reference. It's also what
LoraMegakernelTrainer uses for DN backward today (slow path); when the
CUDA kernel lands, swap the implementation, keep the API.
"""
from __future__ import annotations

from typing import NamedTuple

import torch


class DNBackwardResult(NamedTuple):
    dq:          torch.Tensor   # [S, H, Dk] bf16
    dk:          torch.Tensor   # [S, H, Dk] bf16
    dv:          torch.Tensor   # [S, H, Dv] bf16
    dbeta:       torch.Tensor   # [S, H]     fp32
    ddecay:      torch.Tensor   # [S, H]     fp32
    dstate_init: torch.Tensor   # [H, Dk, Dv] fp32


def dn_forward_recurrence(
    q: torch.Tensor,            # [S, H, Dk] bf16 post conv/silu/norm
    k: torch.Tensor,            # [S, H, Dk]
    v: torch.Tensor,            # [S, H, Dv]
    beta: torch.Tensor,         # [S, H] fp32 sigmoid'd
    decay: torch.Tensor,        # [S, H] fp32
    state_init: torch.Tensor,   # [H, Dk, Dv] fp32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the recurrence; return (y, final_state). Reference only.

    The real forward is already in our prefill.cu (pf_deltanet_recurrence_
    vsplit_prepped); this function is used for testing and as the
    autograd-backed backward reference.
    """
    S, H, Dk = q.shape
    Dv = v.shape[2]
    q32 = q.to(torch.float32)
    k32 = k.to(torch.float32)
    v32 = v.to(torch.float32)

    state = state_init.clone()              # [H, Dk, Dv]
    outs: list[torch.Tensor] = []
    for t in range(S):
        qt, kt, vt = q32[t], k32[t], v32[t]
        bt = beta[t].unsqueeze(-1)          # [H, 1]
        dt = decay[t].view(H, 1, 1)         # [H, 1, 1]
        # state . k -> [H, Dv]
        sk = torch.einsum("hdv,hd->hv", state, kt)
        delta = bt * (vt - sk)              # [H, Dv]
        # state update: decay*state + k ⊗ delta
        state = dt * state + torch.einsum("hd,hv->hdv", kt, delta)
        outs.append(torch.einsum("hdv,hd->hv", state, qt))
    y = torch.stack(outs, dim=0)            # [S, H, Dv]
    return y.to(torch.bfloat16), state


def dn_backward_autograd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    beta: torch.Tensor, decay: torch.Tensor, state_init: torch.Tensor,
    dy: torch.Tensor,
) -> DNBackwardResult:
    """Backward via torch autograd on an unrolled forward — correctness
    reference. O(S) Python dispatch, slow for long S but mathematically
    exact.

    dy is [S, H, Dv] bf16 (grad of recurrence output).
    """
    q32 = q.to(torch.float32).detach().requires_grad_(True)
    k32 = k.to(torch.float32).detach().requires_grad_(True)
    v32 = v.to(torch.float32).detach().requires_grad_(True)
    beta32  = beta.to(torch.float32).detach().requires_grad_(True)
    decay32 = decay.to(torch.float32).detach().requires_grad_(True)
    s0      = state_init.to(torch.float32).detach().requires_grad_(True)

    S, H, Dk = q.shape
    state = s0
    outs: list[torch.Tensor] = []
    for t in range(S):
        qt, kt, vt = q32[t], k32[t], v32[t]
        bt = beta32[t].unsqueeze(-1)
        dt = decay32[t].view(H, 1, 1)
        sk = torch.einsum("hdv,hd->hv", state, kt)
        delta = bt * (vt - sk)
        state = dt * state + torch.einsum("hd,hv->hdv", kt, delta)
        outs.append(torch.einsum("hdv,hd->hv", state, qt))
    y = torch.stack(outs, dim=0)
    y.backward(dy.to(torch.float32))

    return DNBackwardResult(
        dq          = q32.grad.to(torch.bfloat16),
        dk          = k32.grad.to(torch.bfloat16),
        dv          = v32.grad.to(torch.bfloat16),
        dbeta       = beta32.grad,
        ddecay      = decay32.grad,
        dstate_init = s0.grad,
    )


def dn_backward_chunked(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    beta: torch.Tensor, decay: torch.Tensor, state_init: torch.Tensor,
    dy: torch.Tensor,
    chunk_size: int = 64,
) -> DNBackwardResult:
    """Chunked BPTT. Reverse-walk chunks of `chunk_size` tokens; within
    each chunk do the per-step adjoint recurrence in torch (serial but
    compact), between chunks propagate dstate.

    The chunked form maps cleanly to bf16 WGMMA on sm_100 — inside each
    chunk the k ⊗ delta outer products become tall-skinny matmuls that
    tensor-core throughput dominates. This reference matches the
    autograd reference and is the port target for a CUDA kernel.

    For now we run in torch fp32 — gives us a correct and predictable
    slow path that the trainer can use while the CUDA kernel lands.
    """
    S, H, Dk = q.shape
    Dv = v.shape[2]
    dev = q.device

    q32, k32, v32 = q.to(torch.float32), k.to(torch.float32), v.to(torch.float32)
    beta32, decay32 = beta.to(torch.float32), decay.to(torch.float32)
    s0 = state_init.to(torch.float32).clone()

    # Forward sweep: save state at every chunk boundary so backward can
    # start from the right checkpoint.
    chunk_states = [s0]
    state = s0.clone()
    for c_start in range(0, S, chunk_size):
        c_end = min(c_start + chunk_size, S)
        for t in range(c_start, c_end):
            qt, kt, vt = q32[t], k32[t], v32[t]
            bt = beta32[t].unsqueeze(-1)
            dt = decay32[t].view(H, 1, 1)
            sk = torch.einsum("hdv,hd->hv", state, kt)
            delta = bt * (vt - sk)
            state = dt * state + torch.einsum("hd,hv->hdv", kt, delta)
        chunk_states.append(state.clone())

    dq    = torch.zeros_like(q32)
    dk    = torch.zeros_like(k32)
    dv    = torch.zeros_like(v32)
    dbeta = torch.zeros_like(beta32)
    ddecay = torch.zeros_like(decay32)

    # Backward sweep: for each chunk, replay forward from its saved
    # starting state with requires_grad=True on per-step tensors, call
    # backward on that chunk's dy slice, accumulate.
    # dstate flows backward between chunks (from c+1 into end-of-c).
    dstate_right = torch.zeros_like(state_init, dtype=torch.float32)

    n_chunks = (S + chunk_size - 1) // chunk_size
    for c in range(n_chunks - 1, -1, -1):
        c_start = c * chunk_size
        c_end = min(c_start + chunk_size, S)
        cs = c_end - c_start

        q_c = q32[c_start:c_end].detach().requires_grad_(True)
        k_c = k32[c_start:c_end].detach().requires_grad_(True)
        v_c = v32[c_start:c_end].detach().requires_grad_(True)
        b_c = beta32[c_start:c_end].detach().requires_grad_(True)
        d_c = decay32[c_start:c_end].detach().requires_grad_(True)
        s_in = chunk_states[c].detach().requires_grad_(True)

        state_c = s_in
        outs_c: list[torch.Tensor] = []
        for t in range(cs):
            qt, kt, vt = q_c[t], k_c[t], v_c[t]
            bt = b_c[t].unsqueeze(-1)
            dt = d_c[t].view(H, 1, 1)
            sk = torch.einsum("hdv,hd->hv", state_c, kt)
            delta = bt * (vt - sk)
            state_c = dt * state_c + torch.einsum("hd,hv->hdv", kt, delta)
            outs_c.append(torch.einsum("hdv,hd->hv", state_c, qt))
        y_c = torch.stack(outs_c, dim=0)

        # Loss contributions: dy slice + the final-state grad coming
        # from the next chunk (or zero if this is the last chunk).
        loss = (y_c * dy[c_start:c_end].to(torch.float32)).sum() + \
               (state_c * dstate_right).sum()
        loss.backward()

        dq[c_start:c_end]    = q_c.grad
        dk[c_start:c_end]    = k_c.grad
        dv[c_start:c_end]    = v_c.grad
        dbeta[c_start:c_end]  = b_c.grad
        ddecay[c_start:c_end] = d_c.grad
        dstate_right = s_in.grad

    return DNBackwardResult(
        dq=dq.to(torch.bfloat16),
        dk=dk.to(torch.bfloat16),
        dv=dv.to(torch.bfloat16),
        dbeta=dbeta,
        ddecay=ddecay,
        dstate_init=dstate_right,
    )


__all__ = [
    "DNBackwardResult", "dn_forward_recurrence",
    "dn_backward_autograd", "dn_backward_chunked",
]
