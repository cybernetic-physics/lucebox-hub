"""cuDNN FlashAttention-2 backward helper for Qwen3.5-0.8B FA layers.

Thin wrapper around torch's low-level flash attention ops that exposes
both the forward (returning O + LSE) and the backward (given dO and the
saved O + LSE). This is what HF uses internally and is state-of-the-art
on Blackwell — cuDNN FA-2 on tcgen05 MMA.

We use this as the Phase-2 FA backward in LoraMegakernelTrainer until
a self-consistent CUTLASS Sm100FmhaBwd pair is wired in (the current
CUTLASS path is in models/qwen35_0p8b/trainer/cutlass_train/ and runs
but needs CUTLASS's matching forward kernel for full correctness).

All tensors are bf16 on cuda, [B, H, S, D] layout (torch-native).
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch


class FAForwardResult(NamedTuple):
    O: torch.Tensor          # [B, Hq, S, D]  bf16
    LSE: torch.Tensor        # [B, Hq, S]     fp32
    philox_seed: torch.Tensor
    philox_offset: torch.Tensor
    max_Q: int
    max_K: int


def fa_forward_flash(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    is_causal: bool = True,
    scale: float | None = None,
) -> FAForwardResult:
    """Run cuDNN FA-2 forward. GQA handled by expanding K/V along the Q
    head dim — cuDNN's flash kernel doesn't natively support GQA through
    broadcasting today, so we expand and let the Hq reduction happen in
    `fa_backward_flash` when computing dK/dV.

    Returns O + LSE so the backward can be invoked later without rerun-
    ning forward.
    """
    B, Hq, S, D = Q.shape
    Hk = K.shape[1]
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    if Hq != Hk:
        K = K.repeat_interleave(Hq // Hk, dim=1)
        V = V.repeat_interleave(Hq // Hk, dim=1)
    r = torch.ops.aten._scaled_dot_product_flash_attention(
        Q, K, V, 0.0, is_causal, False, scale=scale)
    # r = (output, lse, None, None, max_q, max_k, philox_seed, philox_offset, debug_mask)
    return FAForwardResult(
        O=r[0], LSE=r[1],
        philox_seed=r[6], philox_offset=r[7],
        max_Q=int(r[4]), max_K=int(r[5]))


def fa_backward_flash(
    dO: torch.Tensor,
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    O: torch.Tensor, LSE: torch.Tensor,
    *,
    is_causal: bool = True,
    scale: float | None = None,
    philox_seed: torch.Tensor | None = None,
    philox_offset: torch.Tensor | None = None,
    num_kv_heads: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """cuDNN FA-2 backward. Inputs are in [B, Hq, S, D] with Q/K/V/O/dO
    all shaped as Hq heads (GQA has already been expanded by the forward).
    Returns (dQ, dK, dV) — if `num_kv_heads` is given, dK/dV are reduced
    back to Hk heads by summing across the H_R = Hq/num_kv_heads copies.
    """
    B, Hq, S, D = Q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    if philox_seed is None:
        philox_seed = torch.zeros(2, dtype=torch.uint64, device=Q.device)
    if philox_offset is None:
        philox_offset = torch.zeros((), dtype=torch.uint64, device=Q.device)

    r = torch.ops.aten._scaled_dot_product_flash_attention_backward(
        dO, Q, K, V, O, LSE,
        None, None,   # cumulative_seq_q / cumulative_seq_k (varlen; unused)
        S, S,          # max_q, max_k
        0.0,           # dropout
        is_causal,
        philox_seed, philox_offset,
        scale=scale,
    )
    dQ, dK, dV = r[0], r[1], r[2]
    if num_kv_heads is not None and num_kv_heads != Hq:
        H_R = Hq // num_kv_heads
        dK = dK.view(B, num_kv_heads, H_R, S, D).sum(dim=2)
        dV = dV.view(B, num_kv_heads, H_R, S, D).sum(dim=2)
    return dQ, dK, dV


__all__ = ["FAForwardResult", "fa_forward_flash", "fa_backward_flash"]
