"""DeltaNet recurrence as a torch.autograd.Function backed by our CUDA kernels.

Plugs the bit-exact CUDA dn_fwd_save / dn_bwd kernels into PyTorch autograd
so a model that uses DeltaNet recurrence can train with HF/PEFT while the
DeltaNet layer's bwd takes the fast path (17x vs autograd reference).

Inputs are the *post-preprocessing* DN signals:
  q, k : [S, H, Dk] bf16
  v    : [S, H, Dv] bf16
  beta : [S, H]     fp32 (sigmoid'd)
  decay: [S, H]     fp32 (sigmoid'd)
  state_init: [H, Dk, Dv] fp32

Outputs:
  y         : [S, H, Dv] bf16
  state_out : [H, Dk, Dv] fp32 (final recurrent state — useful for prefill)

The function does not differentiate w.r.t. state_out; only y.grad is consumed.

When called outside an autograd context (i.e. inference / no_grad), uses the
faster chunked tensor-core forward kernel (`dn_chunked_fwd`, ~3.4× over the
recurrent forward). In training mode, uses the recurrent fwd which saves the
state_history needed by the recurrent bwd.
"""
from __future__ import annotations

import torch

import train_megakernel_C  # noqa: F401


class DeltaNetRecurrence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, beta, decay, state_init):
        assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16
        assert beta.dtype == torch.float32 and decay.dtype == torch.float32
        assert state_init.dtype == torch.float32
        S, H, Dk = q.shape
        Dv = v.shape[-1]
        dev = q.device

        y = torch.empty(S, H, Dv, dtype=torch.bfloat16, device=dev)
        state_out = torch.empty(H, Dk, Dv, dtype=torch.float32, device=dev)
        delta_save = torch.empty(S, H, Dv, dtype=torch.bfloat16, device=dev)
        state_history = torch.empty(H, S + 1, Dk, Dv, dtype=torch.float32, device=dev)

        torch.ops.train_megakernel_C.dn_fwd_save(
            q.contiguous(), k.contiguous(), v.contiguous(),
            beta.contiguous(), decay.contiguous(), state_init.contiguous(),
            y, state_out, delta_save, state_history,
        )
        ctx.save_for_backward(q, k, v, beta, decay, state_init, delta_save, state_history)
        return y, state_out

    @staticmethod
    def backward(ctx, grad_y, grad_state_out):
        # grad_state_out is ignored — final state is just bookkeeping for
        # prefill, not differentiated through during training.
        q, k, v, beta, decay, state_init, delta_save, state_history = ctx.saved_tensors

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dbeta = torch.empty_like(beta)
        ddecay = torch.empty_like(decay)
        dstate_init = torch.empty_like(state_init)

        torch.ops.train_megakernel_C.dn_bwd(
            q, k, v, beta, decay, state_init, delta_save, grad_y.contiguous(),
            state_history,
            dq, dk, dv, dbeta, ddecay, dstate_init,
        )
        return dq, dk, dv, dbeta, ddecay, dstate_init


def deltanet_recurrence(q, k, v, beta, decay, state_init):
    """Functional wrapper: returns (y, state_out) with autograd."""
    return DeltaNetRecurrence.apply(q, k, v, beta, decay, state_init)


def deltanet_chunked_inference(q, k, v, beta, g, state_init):
    """Inference-only fast path: bf16 tensor-core chunked forward.

    Takes log-decay g (not exp(g)) — matches HF's convention. For training
    use deltanet_recurrence (autograd path) instead.
    """
    assert not torch.is_grad_enabled() or not (
        q.requires_grad or k.requires_grad or v.requires_grad or
        beta.requires_grad or g.requires_grad or state_init.requires_grad
    ), "deltanet_chunked_inference is no-grad; use deltanet_recurrence for training"
    S, H, Dk = q.shape
    Dv = v.shape[-1]
    y = torch.empty(S, H, Dv, dtype=torch.bfloat16, device=q.device)
    state_out = torch.empty(H, Dk, Dv, dtype=torch.float32, device=q.device)
    state_chunks = torch.empty(0, dtype=torch.float32, device=q.device)
    torch.ops.train_megakernel_C.dn_chunked_fwd(
        q.contiguous(), k.contiguous(), v.contiguous(),
        beta.contiguous(), g.contiguous(), state_init.contiguous(),
        y, state_out, state_chunks,
    )
    return y, state_out
