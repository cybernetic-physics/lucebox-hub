"""Layer-walking backward skeleton for Tier 0.2 Slice B.3b.

This file is a STUB / SCAFFOLD. The forward and the autograd entry
point (kernel_loss_autograd in lora_megakernel_step.py) are validated
against HF+PEFT. What's left is the per-layer reverse walk that
consumes d(h_pre_norm) + the saved activations and produces:
  - dh for prior layer (= dh_pre_norm for layer L-1)
  - dA, dB for each LoRA projection on layer L

The kernel-level building blocks all exist and are individually
tested:

  bwd_rmsnorm        x, w, dy → dx                           (test_bwd_head)
  bwd_swiglu         gate, up, dy → dgate, dup               (test_bwd_mlp)
  bwd_lora_linear    x, A, B, gy → dx, dA, dB                (test_bwd_mlp)
  fa_bwd_flash       Q, K, V, O, LSE, dO, num_kv_heads
                       → dQ, dK, dV                           (Phase 2 shipped)
  dn_bwd / dn_chunked_bwd (TODO — chunked CUDA port pending)
                       q, k, v, beta, decay, state, dy → dq, dk, dv, dbeta, ddecay, dstate
  bwd_ce_lm_head     final_normed, lm_head_w, target → grad_final_normed, loss

The work is to thread these together. This file provides:
  - per_layer_bwd_fa(...)   stub for FA layer reverse walk
  - per_layer_bwd_dn(...)   stub for DN layer reverse walk
  - lora_grad_scatter(...)  stub for per-layer LoRA grads → flat fp32 buffer
  - run_layer_walking_bwd(...) top-level driver that iterates layers in
                                reverse calling the right per-layer stub
"""
from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b")
sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")

import torch  # noqa: E402

from lora_megakernel_step import HIDDEN, INTER, NUM_LAYERS, LAYER_TYPE  # noqa: E402


# ---------------------------------------------------------------------------
# Per-projection LoRA bwd helper
# ---------------------------------------------------------------------------

def lora_linear_bwd(
    x: torch.Tensor,        # [S, K_in]   bf16  — input to forward
    A: torch.Tensor,        # [K_in, R]   bf16
    B: torch.Tensor,        # [R, K_out]  bf16
    base_W: torch.Tensor,   # [K_out, K_in] bf16 — frozen base weight
    grad_y: torch.Tensor,   # [S, K_out]  fp32  — upstream gradient
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (grad_x, grad_A, grad_B) for `y = x @ base_W.T + scaling * (x @ A) @ B`.

    Combines our `bwd_lora_linear` kernel (LoRA contribution to grad_x +
    grad_A + grad_B) with one cuBLAS GEMM for the base path's grad_x.
    """
    S, K_in = x.shape
    R = A.shape[1]
    K_out = B.shape[1]
    grad_x = torch.empty(S, K_in, dtype=torch.float32, device="cuda")
    grad_A = torch.empty(K_in, R, dtype=torch.float32, device="cuda")
    grad_B = torch.empty(R, K_out, dtype=torch.float32, device="cuda")
    ws_lora_h = torch.empty(S * R, dtype=torch.float32, device="cuda")
    ws_grad_lora_h = torch.empty(S * R, dtype=torch.float32, device="cuda")

    torch.ops.train_megakernel_C.bwd_lora_linear(
        x.contiguous(), A.contiguous(), B.contiguous(), grad_y.contiguous(),
        grad_x, grad_A, grad_B,
        ws_lora_h, ws_grad_lora_h,
        S, K_in, K_out, R, scaling,
    )
    # Add base-path grad_x: d(y) @ base_W   →  [S, K_in] fp32
    grad_x_base = grad_y.to(torch.bfloat16) @ base_W
    grad_x = grad_x + grad_x_base.float()
    return grad_x, grad_A, grad_B


# ---------------------------------------------------------------------------
# RMSNorm bwd via our kernel
# ---------------------------------------------------------------------------

def rmsnorm_bwd(
    x: torch.Tensor,        # [S, H]      bf16  — pre-norm input
    w: torch.Tensor,        # [H]         bf16  — gain (1+w convention)
    dy: torch.Tensor,       # [S, H]      fp32  — upstream
    eps: float = 1e-6,
) -> torch.Tensor:
    """Returns dx for Qwen3-Next RMSNorm: y = x * rstd * (1 + w)."""
    S, H = x.shape
    dx = torch.empty(S, H, dtype=torch.float32, device="cuda")
    torch.ops.train_megakernel_C.bwd_rmsnorm(
        x.contiguous(), w.contiguous(), dy.contiguous(), dx, S, H, eps,
    )
    return dx


# ---------------------------------------------------------------------------
# SwiGLU bwd via our kernel
# ---------------------------------------------------------------------------

def swiglu_bwd(
    gate: torch.Tensor,     # [S, INTER]  bf16
    up: torch.Tensor,       # [S, INTER]  bf16
    dy: torch.Tensor,       # [S, INTER]  fp32  — d(silu(gate) * up)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (dgate, dup)."""
    N = gate.numel()
    dgate = torch.empty_like(gate, dtype=torch.float32)
    dup = torch.empty_like(up, dtype=torch.float32)
    torch.ops.train_megakernel_C.bwd_swiglu(
        gate.contiguous(), up.contiguous(), dy.contiguous(),
        dgate, dup, N,
    )
    return dgate, dup


# ---------------------------------------------------------------------------
# FA-layer reverse walk (stub)
# ---------------------------------------------------------------------------

def per_layer_bwd_fa(
    L: int,                          # absolute layer index in [0, NUM_LAYERS)
    fa_idx: int,                     # FA-only layer index in [0, N_FA)
    dh_out: torch.Tensor,            # [S, H]  fp32 — gradient flowing into this layer's output
    saves: dict,                     # the 6 activation slabs from prefill_bf16_train_step
    lora_flat: list[torch.Tensor],   # 26 packed bf16 tensors
    base_weights_handle: Any,        # base model weights for layer L
    lora_rank: int,
    lora_scaling: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reverse walk of one FA layer. TODO: implement.

    Returns:
        dh_in: [S, H] fp32 — gradient flowing OUT of this layer into the
               previous layer's residual stream.
        layer_lora_grads: dict of per-projection LoRA grads
                          (keys: fa_q_A, fa_q_B, ..., fa_down_B; one each).
    """
    raise NotImplementedError(
        "per_layer_bwd_fa is the core Slice B.3b work. Steps:\n"
        "  1. dmlp_out = dh_out                                  (residual)\n"
        "  2. dh_post_attn = dh_out                              (residual)\n"
        "  3. dmlp_inter, dA_down, dB_down ←\n"
        "       lora_linear_bwd(saves['mlp_inter'][L], fa_down_A[fa_idx],\n"
        "                       fa_down_B[fa_idx], down_W, dmlp_out, scaling)\n"
        "  4. RECOMPUTE gate, up: 2 cuBLAS GEMMs from saves['normalized_post_attn'][L]\n"
        "       gate = saves['normalized_post_attn'][L] @ gate_W.T + LoRA gate path\n"
        "       up   = same with up_W and fa_up_A/B\n"
        "  5. dgate, dup = swiglu_bwd(gate, up, dmlp_inter)\n"
        "  6. dnorm_post_attn = LoRA-path dx (gate) + base @ gate_W +\n"
        "                       LoRA-path dx (up)   + base @ up_W\n"
        "       Plus dA/dB grads for fa_gate, fa_up.\n"
        "  7. dh_post_attn += rmsnorm_bwd(saves['h_post_attn'][L], post_attn_norm_w,\n"
        "                                  dnorm_post_attn)\n"
        "  8. dattn_out = dh_post_attn        (residual)\n"
        "  9. dh_in = dh_post_attn            (residual; will accumulate further)\n"
        " 10. d_attn_out_pre_o, dA_o, dB_o ←\n"
        "       lora_linear_bwd(saves['attn_out_pre_o'][L], fa_o_A[fa_idx],\n"
        "                       fa_o_B[fa_idx], o_W, dattn_out, scaling)\n"
        " 11. RECOMPUTE Q, K, V from saves['normalized_in'][L] + RoPE/QKnorm\n"
        "       (or save them in a separate kernel mod — TODO)\n"
        " 12. dQ, dK, dV = fa_bwd_flash(Q, K, V, O, LSE, d_attn_out_pre_o, ...)\n"
        "       (need to also compute O + LSE — either save in fwd or recompute.)\n"
        " 13. dnorm_in = sum of:\n"
        "       lora_linear_bwd over q, k, v with respective dQ_pre_norm, etc.\n"
        " 14. dh_in += rmsnorm_bwd(saves['hidden_in'][L], input_norm_w, dnorm_in)\n"
        " 15. Return dh_in + the 7 (A,B) grad pairs.\n\n"
        "The trickiest items: 11/12 (need to either save Q/K/V/O/LSE during\n"
        "forward or recompute them), 6 (correctly accumulating dA/dB grads\n"
        "for two parallel LoRA paths sharing the same x).\n"
    )


# ---------------------------------------------------------------------------
# DN-layer reverse walk (stub)
# ---------------------------------------------------------------------------

def per_layer_bwd_dn(
    L: int,
    dn_idx: int,                     # DN-only layer index in [0, N_DN)
    dh_out: torch.Tensor,
    saves: dict,
    lora_flat: list[torch.Tensor],
    base_weights_handle: Any,
    lora_rank: int,
    lora_scaling: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reverse walk of one DN layer. TODO: implement. Mirrors the FA path but:
      - The attention bwd is dn_bwd (recurrent CUDA, requires state_history)
        OR dn_chunked_bwd (chunked CUDA — Tier 1.6, pending).
      - The Q/K/V projections are 2 LoRA'd projections (in_proj_qkv, in_proj_z)
        plus a non-LoRA conv1d path. Need conv1d bwd via at::conv_depthwise2d_bwd.
    """
    raise NotImplementedError(
        "per_layer_bwd_dn — same skeleton as per_layer_bwd_fa with DN-specific\n"
        "attention and qkv handling. The state_history requirement is the\n"
        "main blocker; either:\n"
        "  (a) Use the recurrent dn_bwd kernel (bit-exact, but requires\n"
        "      ~1 MB × S × N_DN of state_history HBM — 18 GB at S=128).\n"
        "  (b) Run dn_fwd_with_delta_save once before bwd to materialize\n"
        "      state_history (extra forward pass — costs ~30% wall).\n"
        "  (c) Wait for chunked CUDA bwd port (Tier 1.6) which only needs\n"
        "      state_chunks (~50 MB at S=128).\n"
    )


# ---------------------------------------------------------------------------
# Top-level driver (stub)
# ---------------------------------------------------------------------------

def run_layer_walking_bwd(
    grad_h_pre_norm: torch.Tensor,
    saves: dict,
    lora_flat: list[torch.Tensor],
    base_weights_handle: Any,
    lora_rank: int,
    lora_scaling: float,
) -> dict[str, torch.Tensor]:
    """Iterate layers in reverse, accumulating per-projection LoRA grads.

    Returns a dict mapping projection name (e.g. 'fa_q_A') to its full
    [n_layers_of_type, K_in_or_R, R_or_K_out] gradient tensor.
    """
    raise NotImplementedError(
        "Iterate L from NUM_LAYERS-1 down to 0:\n"
        "  - look up LAYER_TYPE[L]\n"
        "  - if FA: dh_in, layer_grads = per_layer_bwd_fa(L, fa_idx, dh_out, ...)\n"
        "  - if DN: dh_in, layer_grads = per_layer_bwd_dn(L, dn_idx, dh_out, ...)\n"
        "  - accumulate layer_grads into the flat 26-tensor grad buffer at the\n"
        "    correct (kernel_idx, ...) slice.\n"
        "  - dh_out for next iter = dh_in.\n"
    )


__all__ = [
    "lora_linear_bwd", "rmsnorm_bwd", "swiglu_bwd",
    "per_layer_bwd_fa", "per_layer_bwd_dn", "run_layer_walking_bwd",
]
