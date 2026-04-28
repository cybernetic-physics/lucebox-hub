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

import importlib.util  # noqa: E402

import torch  # noqa: E402

# Import constants directly from outer model.py to avoid pulling in
# lora_megakernel_step (which transitively imports the inference
# extension). This file's helpers are pure-trainer-extension.
_spec = importlib.util.spec_from_file_location(
    "qwen_outer_model_consts",
    "/root/lucebox-hub-b200-train/models/qwen35_0p8b/model.py")
_outer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_outer)
HIDDEN = _outer.HIDDEN_SIZE
INTER = _outer.INTERMEDIATE_SIZE
NUM_LAYERS = _outer.NUM_LAYERS
LAYER_TYPE = _outer.LAYER_TYPE


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
    # zero-init: the kernel accumulates into grad_A / grad_B (matches the
    # test_bwd_mlp.py call convention, where torch.zeros is used).
    grad_x = torch.zeros(S, K_in, dtype=torch.float32, device="cuda")
    grad_A = torch.zeros(K_in, R, dtype=torch.float32, device="cuda")
    grad_B = torch.zeros(R, K_out, dtype=torch.float32, device="cuda")
    ws_lora_h = torch.zeros(S, R, dtype=torch.float32, device="cuda")
    ws_grad_lora_h = torch.zeros(S, R, dtype=torch.float32, device="cuda")

    torch.ops.train_megakernel_C.bwd_lora_linear(
        x.contiguous(), A.contiguous(), B.contiguous(), grad_y.contiguous(),
        grad_x, grad_A, grad_B,
        ws_lora_h, ws_grad_lora_h,
        S, K_in, K_out, R, scaling,
    )
    # Add base-path grad_x: d(y) @ base_W   →  [S, K_in] fp32.
    # Keep grad_y in fp32 here — casting to bf16 loses signal on small
    # gradients (std ~1e-2) which the gate/up paths typically have. The
    # extra precision is essentially free since cuBLAS does fp32 GEMM
    # natively; only the weight tensor is cast to fp32.
    grad_x_base = grad_y @ base_W.float()
    grad_x = grad_x + grad_x_base
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
# MLP-block reverse walk (FA-style: gate/up/down + post-attn RMSnorm)
# ---------------------------------------------------------------------------

def layer_mlp_bwd(
    *,
    h_post_attn: torch.Tensor,        # [S, HIDDEN] bf16 — pre-post-attn-rmsnorm residual
    normalized_post_attn: torch.Tensor,  # [S, HIDDEN] bf16 — post-rmsnorm input to gate/up
    mlp_inter: torch.Tensor,           # [S, INTER] bf16 — silu(gate)*up output, input to down
    dh_out: torch.Tensor,              # [S, HIDDEN] fp32 — d(layer output residual stream)
    # Frozen base weights for this layer:
    post_attn_norm_w: torch.Tensor,    # [HIDDEN] bf16
    gate_W: torch.Tensor,              # [INTER, HIDDEN] bf16
    up_W: torch.Tensor,                # [INTER, HIDDEN] bf16
    down_W: torch.Tensor,              # [HIDDEN, INTER] bf16
    # LoRA tensors for this layer's MLP projections (single layer slice):
    gate_A: torch.Tensor, gate_B: torch.Tensor,    # [HIDDEN, R], [R, INTER]
    up_A: torch.Tensor,   up_B: torch.Tensor,
    down_A: torch.Tensor, down_B: torch.Tensor,    # [INTER, R], [R, HIDDEN]
    lora_scaling: float,
    rms_eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Reverse walk of the MLP block of one layer.

    Forward (for reference):
        h_post_attn  → rms_norm → normalized_post_attn
        gate = gate_W·normalized_post_attn  (+ LoRA)
        up   = up_W  ·normalized_post_attn  (+ LoRA)
        mlp_inter = silu(gate) * up
        mlp_out = down_W·mlp_inter  (+ LoRA)
        h_out   = h_post_attn + mlp_out

    Backward returns:
        dh_post_attn : [S, HIDDEN] fp32 — gradient flowing back to the
                       attention output (= dh_residual + dh_through_rmsnorm).
        grad_gate_A, grad_gate_B   : LoRA gate grads
        grad_up_A,   grad_up_B     : LoRA up grads
        grad_down_A, grad_down_B   : LoRA down grads

    Uses bwd_lora_linear, bwd_swiglu, bwd_rmsnorm. The FA o_proj /
    attention bwd / qkv bwd / input rmsnorm bwd are NOT included
    here — those are the "rest" of the layer walk (steps 8-15 in the
    skeleton's docstring).
    """
    S, H = h_post_attn.shape
    INTER_dim = mlp_inter.shape[1]

    # --- Step 1: residual split -----------------------------------------
    # h_out = h_post_attn + mlp_out, so:
    #   dmlp_out = dh_out
    #   dh_post_attn (residual contribution) = dh_out
    dmlp_out = dh_out
    dh_post_attn = dh_out.clone()

    # --- Step 2: down LoRA bwd ------------------------------------------
    # mlp_out = down_W·mlp_inter + scaling·(mlp_inter·down_A)·down_B
    # → dmlp_inter, dA_down, dB_down
    dmlp_inter, grad_down_A, grad_down_B = lora_linear_bwd(
        x=mlp_inter, A=down_A, B=down_B, base_W=down_W,
        grad_y=dmlp_out, scaling=lora_scaling)

    # --- Step 3: SwiGLU bwd ---------------------------------------------
    # mlp_inter = silu(gate) * up. We didn't save gate/up — recompute them.
    # gate = normalized_post_attn @ gate_W.T + scaling * (npa @ gate_A) @ gate_B
    # Same pattern for up.
    npa = normalized_post_attn
    gate = npa.float() @ gate_W.float().t()
    gate = gate + lora_scaling * ((npa.float() @ gate_A.float()) @ gate_B.float())
    up = npa.float() @ up_W.float().t()
    up = up + lora_scaling * ((npa.float() @ up_A.float()) @ up_B.float())
    gate_bf = gate.to(torch.bfloat16).contiguous()
    up_bf = up.to(torch.bfloat16).contiguous()

    dgate, dup = swiglu_bwd(gate_bf, up_bf, dmlp_inter)

    # --- Step 4: gate, up LoRA bwd → accumulate into dnorm_post_attn ----
    dnpa_from_gate, grad_gate_A, grad_gate_B = lora_linear_bwd(
        x=npa, A=gate_A, B=gate_B, base_W=gate_W,
        grad_y=dgate, scaling=lora_scaling)
    dnpa_from_up, grad_up_A, grad_up_B = lora_linear_bwd(
        x=npa, A=up_A, B=up_B, base_W=up_W,
        grad_y=dup, scaling=lora_scaling)
    dnpa = dnpa_from_gate + dnpa_from_up

    # --- Step 5: post-attn RMSnorm bwd ----------------------------------
    # rms_norm(h_post_attn, post_attn_norm_w) = normalized_post_attn
    # bwd_rmsnorm gives d(h_post_attn) given d(normalized_post_attn).
    dh_through_norm = rmsnorm_bwd(
        x=h_post_attn, w=post_attn_norm_w, dy=dnpa, eps=rms_eps)
    dh_post_attn = dh_post_attn + dh_through_norm

    return {
        "dh_post_attn": dh_post_attn,
        "grad_gate_A": grad_gate_A, "grad_gate_B": grad_gate_B,
        "grad_up_A":   grad_up_A,   "grad_up_B":   grad_up_B,
        "grad_down_A": grad_down_A, "grad_down_B": grad_down_B,
    }


# ---------------------------------------------------------------------------
# FA attention block reverse walk (qkv + RoPE + QKnorm + FA + o_proj +
# input rmsnorm). First-pass implementation uses torch autograd through
# the recomputed attention sub-graph — slow but provably correct. The
# layer_mlp_bwd above is the fast template; future work replaces the
# autograd interior here with hand-rolled cuDNN FA-bwd + manual chain
# rule for QKnorm/RoPE.
# ---------------------------------------------------------------------------

# Constants (mirrors prefill.cu).
_FA_HEAD_DIM = 256
_FA_Q_HEADS  = 8
_FA_KV_HEADS = 2
_FA_GQA      = _FA_Q_HEADS // _FA_KV_HEADS
_FA_Q_SIZE       = _FA_Q_HEADS  * _FA_HEAD_DIM
_FA_QPROJ_SIZE   = _FA_Q_SIZE * 2     # Q + Gate per head
_FA_KV_SIZE      = _FA_KV_HEADS * _FA_HEAD_DIM
_FA_ROT_DIM      = 64
_FA_ROPE_THETA   = 1e7


def _qwen_rms(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Qwen3-Next RMSnorm: y = x * rsqrt(mean(x²) + eps) * (1 + w).
    Normalizes over the last axis. Used for both the layer-input RMSnorm
    (last-axis = HIDDEN) and the per-head QK-norm (last-axis = HEAD_DIM).
    """
    x_f = x.float()
    rstd = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_f * rstd * (1.0 + w.float())).to(x.dtype)


def _qwen_rope_cs(S: int, device, dtype=torch.float32):
    """Pre-compute Qwen3-Next RoPE cos/sin tables.

    Returns (cos, sin) each of shape [S, ROT_DIM/2] in `dtype`.
    """
    half = _FA_ROT_DIM // 2
    pos = torch.arange(S, device=device, dtype=dtype)
    i = torch.arange(half, device=device, dtype=dtype)
    fe = 2.0 * i / _FA_ROT_DIM
    inv_theta = 1.0 / (_FA_ROPE_THETA ** fe)            # [half]
    angles = pos.unsqueeze(1) * inv_theta.unsqueeze(0)  # [S, half]
    return torch.cos(angles), torch.sin(angles)


def _qwen_rope(x: torch.Tensor) -> torch.Tensor:
    """Apply Qwen3-Next RoPE to ``x`` of shape [S, num_heads, FA_HEAD_DIM].
    Only the first FA_ROT_DIM (=64) dimensions of each head get rotated.
    The pairing is "halves": for i ∈ [0, ROT_DIM/2), the partner is i+ROT_DIM/2.
    """
    S, H, D = x.shape
    assert D == _FA_HEAD_DIM, f"expected head_dim={_FA_HEAD_DIM}, got {D}"
    half = _FA_ROT_DIM // 2
    cos, sin = _qwen_rope_cs(S, x.device, dtype=torch.float32)
    x_f = x.float()
    rot = x_f[..., :_FA_ROT_DIM]      # [S, H, ROT_DIM]
    pass_ = x_f[..., _FA_ROT_DIM:]    # [S, H, D-ROT_DIM]  no rotation
    a = rot[..., :half]               # [S, H, half]
    b = rot[..., half:]               # [S, H, half]
    cos_b = cos.unsqueeze(1)          # [S, 1, half]
    sin_b = sin.unsqueeze(1)
    a_new = a * cos_b - b * sin_b
    b_new = b * cos_b + a * sin_b
    rot_new = torch.cat([a_new, b_new], dim=-1)
    out = torch.cat([rot_new, pass_], dim=-1)
    return out.to(x.dtype)


def _qwen_rope_bwd(dy: torch.Tensor) -> torch.Tensor:
    """Backward of _qwen_rope. Given d/dy(out)·dy, return dx.

    Forward (per pair):  a_new = a·cos − b·sin;  b_new = b·cos + a·sin
    Backward:            da    = da_new·cos + db_new·sin
                         db    = -da_new·sin + db_new·cos
    The pass-through tail copies straight back.

    Returns fp32 tensor of the same [S, H, D] shape as `dy`.
    """
    S, H, D = dy.shape
    assert D == _FA_HEAD_DIM
    half = _FA_ROT_DIM // 2
    cos, sin = _qwen_rope_cs(S, dy.device, dtype=torch.float32)
    dy_f = dy.float()
    rot = dy_f[..., :_FA_ROT_DIM]
    pass_ = dy_f[..., _FA_ROT_DIM:]
    da_new = rot[..., :half]
    db_new = rot[..., half:]
    cos_b = cos.unsqueeze(1)
    sin_b = sin.unsqueeze(1)
    da = da_new * cos_b + db_new * sin_b
    db = -da_new * sin_b + db_new * cos_b
    rot_back = torch.cat([da, db], dim=-1)
    return torch.cat([rot_back, pass_], dim=-1)


def layer_attn_bwd_fa(
    *,
    hidden_in: torch.Tensor,            # [S, HIDDEN]   bf16 — input to this layer
    h_post_attn: torch.Tensor,          # [S, HIDDEN]   bf16 — post-residual saved
    dh_post_attn: torch.Tensor,         # [S, HIDDEN]   fp32 — gradient flowing in
                                         #   from the MLP block bwd
    # Frozen base weights for this layer:
    input_norm_w: torch.Tensor,         # [HIDDEN]      bf16
    q_W: torch.Tensor,                  # [FA_QPROJ, HIDDEN]      bf16 (Q+Gate)
    k_W: torch.Tensor,                  # [FA_KV_SIZE, HIDDEN]    bf16
    v_W: torch.Tensor,                  # [FA_KV_SIZE, HIDDEN]    bf16
    q_nw: torch.Tensor,                 # [FA_HEAD_DIM]           bf16 (QK-norm gain)
    k_nw: torch.Tensor,                 # [FA_HEAD_DIM]           bf16
    o_W: torch.Tensor,                  # [HIDDEN, FA_Q_SIZE]     bf16
    # LoRA tensors (single-layer slices):
    q_A: torch.Tensor, q_B: torch.Tensor,
    k_A: torch.Tensor, k_B: torch.Tensor,
    v_A: torch.Tensor, v_B: torch.Tensor,
    o_A: torch.Tensor, o_B: torch.Tensor,
    lora_scaling: float,
    rms_eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Reverse walk of the attention block of one FA layer.

    Forward (for reference):
        normalized_in = rms_norm(hidden_in, input_norm_w)
        q_raw = q_W·normalized_in + LoRA   →  [S, FA_Q_HEADS, 2, FA_HEAD_DIM]
        k_raw = k_W·normalized_in + LoRA   →  [S, FA_KV_HEADS, FA_HEAD_DIM]
        v     = v_W·normalized_in + LoRA   →  [S, FA_KV_HEADS, FA_HEAD_DIM]
        Q_h   = q_raw[:, :, 0, :]          (drop gate)
        Gate  = q_raw[:, :, 1, :]
        Q     = rope(qknorm(Q_h, q_nw))
        K     = rope(qknorm(k_raw, k_nw))
        attn_out_unfold = FA(Q, K, v, causal=True)   ([S, FA_Q_HEADS, FA_HEAD_DIM])
        attn_out_pre_o  = sigmoid(Gate) · attn_out_unfold
        attn_out        = o_W·attn_out_pre_o + LoRA
        h_post_attn     = hidden_in + attn_out

    Returns dh_in (gradient out of this layer; goes to layer L-1's
    dh_post_attn) and per-projection LoRA grads.

    Implementation: torch autograd through a recomputed sub-graph.
    Cost is roughly 2× a forward attn pass (one for the recompute,
    one inside autograd). Faster paths replace the inner autograd
    with hand-rolled cuDNN FA-bwd + manual QKnorm/RoPE chain rules.
    """
    # Detach the fixed inputs (hidden_in, h_post_attn) to leaves, but
    # keep them requires_grad on the bwd-target axis (hidden_in for
    # dh_in_through_norm; h_post_attn doesn't need grad — it's just
    # used to compute the residual).
    h_in = hidden_in.detach().clone().requires_grad_(True)

    qA = q_A.detach().clone().requires_grad_(True)
    qB = q_B.detach().clone().requires_grad_(True)
    kA = k_A.detach().clone().requires_grad_(True)
    kB = k_B.detach().clone().requires_grad_(True)
    vA = v_A.detach().clone().requires_grad_(True)
    vB = v_B.detach().clone().requires_grad_(True)
    oA = o_A.detach().clone().requires_grad_(True)
    oB = o_B.detach().clone().requires_grad_(True)

    S = h_in.shape[0]

    # Forward (recompute under autograd).
    npa = _qwen_rms(h_in, input_norm_w, eps=rms_eps)

    q_raw = npa @ q_W.t() + lora_scaling * (npa @ qA) @ qB         # [S, FA_QPROJ_SIZE]
    k_raw = npa @ k_W.t() + lora_scaling * (npa @ kA) @ kB         # [S, FA_KV_SIZE]
    v_raw = npa @ v_W.t() + lora_scaling * (npa @ vA) @ vB         # [S, FA_KV_SIZE]

    q_packed = q_raw.view(S, _FA_Q_HEADS, 2, _FA_HEAD_DIM)
    Q_h = q_packed[:, :, 0, :]                                      # [S, FA_Q_HEADS, FA_HEAD_DIM]
    Gate = q_packed[:, :, 1, :]                                     # same
    K_h = k_raw.view(S, _FA_KV_HEADS, _FA_HEAD_DIM)
    V   = v_raw.view(S, _FA_KV_HEADS, _FA_HEAD_DIM)

    Q_normed = _qwen_rms(Q_h, q_nw, eps=rms_eps)
    K_normed = _qwen_rms(K_h, k_nw, eps=rms_eps)

    Q = _qwen_rope(Q_normed)
    K = _qwen_rope(K_normed)

    # Expand K, V to FA_Q_HEADS for SDPA (GQA expansion).
    K_e = K.repeat_interleave(_FA_GQA, dim=1)                       # [S, FA_Q_HEADS, D]
    V_e = V.repeat_interleave(_FA_GQA, dim=1)

    # SDPA expects [B, H, S, D].
    Q_b = Q.permute(1, 0, 2).unsqueeze(0).contiguous()              # [1, FA_Q_HEADS, S, D]
    K_b = K_e.permute(1, 0, 2).unsqueeze(0).contiguous()
    V_b = V_e.permute(1, 0, 2).unsqueeze(0).contiguous()

    attn_b = torch.nn.functional.scaled_dot_product_attention(
        Q_b, K_b, V_b, is_causal=True)                              # [1, H, S, D]
    attn_unfold = attn_b.squeeze(0).permute(1, 0, 2)                # [S, FA_Q_HEADS, D]

    attn_pre_o = torch.sigmoid(Gate.float()).to(attn_unfold.dtype) * attn_unfold
    attn_pre_o_flat = attn_pre_o.reshape(S, _FA_Q_SIZE)

    attn_out = attn_pre_o_flat @ o_W.t() + lora_scaling * (attn_pre_o_flat @ oA) @ oB

    # The forward gives h_post_attn_recomp = h_in + attn_out, and the
    # caller passed dh_post_attn as the upstream gradient. Backward:
    h_post_attn_recomp = h_in + attn_out
    h_post_attn_recomp.backward(dh_post_attn)

    return {
        "dh_in":      h_in.grad.detach(),    # [S, HIDDEN] fp32; consumed by layer L-1
        "grad_q_A":   qA.grad.detach(),
        "grad_q_B":   qB.grad.detach(),
        "grad_k_A":   kA.grad.detach(),
        "grad_k_B":   kB.grad.detach(),
        "grad_v_A":   vA.grad.detach(),
        "grad_v_B":   vB.grad.detach(),
        "grad_o_A":   oA.grad.detach(),
        "grad_o_B":   oB.grad.detach(),
    }


# ---------------------------------------------------------------------------
# Hand-rolled FA attention bwd (consumes saved Q/O/LSE + cache K/V)
#
# The autograd-based `layer_attn_bwd_fa` above recomputes the entire
# attention forward sub-graph under torch autograd to get gradients.
# That's correct but expensive — at S=512 it costs ~30ms per FA layer.
#
# This function eliminates the recompute by reading:
#     fa_q_save  : post-RoPE/QKnorm Q (gate stripped)        [S, Hq, D] bf16
#     fa_o_save  : FA output before sigmoid-gate is applied  [S, Hq, D] bf16
#     fa_lse_save: cuDNN's log-sum-exp                       [Hq, S]    fp32
#     k_cache_layer_S : post-RoPE/QKnorm K, sliced to S      [Hk, S, D] bf16
#     v_cache_layer_S : V (no RoPE)                          [Hk, S, D] bf16
#
# Backward chain:
#   dh_post_attn → split: dh_in_residual + d_attn_out
#   o_proj LoRA bwd                       (bwd_lora_linear)
#   reverse-gate (sigmoid)                (elementwise)
#   fa_bwd_flash(dO, Q, K, V, O, LSE)     (cuDNN FA-2 bwd)
#   reverse-RoPE (Q & K)                  (elementwise)
#   reverse-QKnorm (Q & K)                (bwd_rmsnorm × 2)
#   q/k/v projection LoRA bwd × 3         (bwd_lora_linear × 3)
#   sum dnpa contributions
#   input rmsnorm bwd                     (bwd_rmsnorm)
#   add to dh_in_residual
# ---------------------------------------------------------------------------

import math

# Lazy import — fa_bwd_flash imports torch ops at module load.
def _fa_bwd_flash():
    from fa_bwd_flash import fa_backward_flash
    return fa_backward_flash


def layer_attn_bwd_fa_handrolled(
    *,
    hidden_in: torch.Tensor,            # [S, HIDDEN]   bf16 — input to this layer
    normalized_in: torch.Tensor,        # [S, HIDDEN]   bf16 — output of input rmsnorm
                                          #   (= input to qkv projections)
    attn_out_pre_o: torch.Tensor,       # [S, FA_Q_SIZE]bf16 — input to o_proj
    fa_q_save: torch.Tensor,            # [S, Hq, D]    bf16 — post-RoPE/QKnorm Q
    fa_o_save: torch.Tensor,            # [S, Hq, D]    bf16 — FA output (no gate)
    fa_lse_save: torch.Tensor,          # [Hq, S]       fp32
    k_cache_layer_S: torch.Tensor,      # [Hk, S, D]    bf16 — post-RoPE/QKnorm K
    v_cache_layer_S: torch.Tensor,      # [Hk, S, D]    bf16 — V (no RoPE)
    dh_post_attn: torch.Tensor,         # [S, HIDDEN]   fp32 — gradient flowing in
    # Frozen base weights for this layer:
    input_norm_w: torch.Tensor,         # [HIDDEN]      bf16
    q_W: torch.Tensor,                  # [FA_QPROJ, HIDDEN]      bf16 (Q+Gate)
    k_W: torch.Tensor,                  # [FA_KV_SIZE, HIDDEN]    bf16
    v_W: torch.Tensor,                  # [FA_KV_SIZE, HIDDEN]    bf16
    q_nw: torch.Tensor,                 # [FA_HEAD_DIM]           bf16 (QK-norm gain)
    k_nw: torch.Tensor,                 # [FA_HEAD_DIM]           bf16
    o_W: torch.Tensor,                  # [HIDDEN, FA_Q_SIZE]     bf16
    # LoRA tensors (single-layer slices):
    q_A: torch.Tensor, q_B: torch.Tensor,
    k_A: torch.Tensor, k_B: torch.Tensor,
    v_A: torch.Tensor, v_B: torch.Tensor,
    o_A: torch.Tensor, o_B: torch.Tensor,
    lora_scaling: float,
    rms_eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Reverse walk of one FA layer's attention block — hand-rolled
    (no torch autograd through the FA sub-graph).

    Same return shape as :func:`layer_attn_bwd_fa`: the consumer is
    interchangeable with the autograd-based version.

    Side-effect-free: doesn't touch any of the input tensors.
    """
    fa_backward_flash = _fa_bwd_flash()

    S = hidden_in.shape[0]
    Hq, D = _FA_Q_HEADS, _FA_HEAD_DIM
    Hk = _FA_KV_HEADS
    H_R = _FA_GQA

    # --- Step 1: residual split -----------------------------------------
    # h_post_attn = h_in + attn_out, so:
    #   dh_in_residual = dh_post_attn (one piece — to merge with norm path)
    #   d_attn_out = dh_post_attn
    dh_in_residual = dh_post_attn
    d_attn_out = dh_post_attn

    # --- Step 2: o_proj LoRA bwd ---------------------------------------
    # attn_out = attn_pre_o @ o_W.T + scaling * (attn_pre_o @ oA) @ oB
    # → d_attn_pre_o, grad_o_A, grad_o_B
    d_attn_pre_o_flat, grad_o_A, grad_o_B = lora_linear_bwd(
        x=attn_out_pre_o, A=o_A, B=o_B, base_W=o_W,
        grad_y=d_attn_out, scaling=lora_scaling)
    d_attn_pre_o = d_attn_pre_o_flat.view(S, Hq, D)            # [S, Hq, D] fp32

    # --- Step 3: reverse-gate (sigmoid) --------------------------------
    # Forward (elementwise): attn_pre_o = sigmoid(Gate) * attn_unfold
    #   d_attn_unfold = sigmoid(Gate)         · d_attn_pre_o
    #   d_Gate         = sigmoid'(Gate) · attn_unfold · d_attn_pre_o
    #                  = sig·(1-sig) · attn_unfold · d_attn_pre_o
    #
    # Recompute Gate. q_raw = npa @ q_W.T + scaling·(npa @ qA)·qB; Gate
    # is the second [S, Hq, D] half of q_raw.view(S, Hq, 2, D).
    npa = normalized_in
    q_raw_recomp = (npa @ q_W.t()
                    + lora_scaling * (npa @ q_A) @ q_B)        # [S, FA_QPROJ_SIZE] bf16
    q_packed_recomp = q_raw_recomp.view(S, Hq, 2, D)
    Q_h_recomp = q_packed_recomp[:, :, 0, :].contiguous()      # [S, Hq, D] bf16 (pre-norm Q)
    Gate_recomp = q_packed_recomp[:, :, 1, :]                  # [S, Hq, D] bf16

    sig = torch.sigmoid(Gate_recomp.float())                   # [S, Hq, D] fp32
    attn_unfold = fa_o_save.float()                            # [S, Hq, D] fp32
    d_attn_unfold = sig * d_attn_pre_o                         # [S, Hq, D] fp32
    d_Gate = (sig * (1.0 - sig)) * attn_unfold * d_attn_pre_o  # [S, Hq, D] fp32

    # --- Step 4: fa_bwd_flash ------------------------------------------
    # cuDNN FA-2 bwd consumes [B=1, Hq, S, D] for Q/K/V/O/dO and [1,Hq,S]
    # for LSE. K/V are expanded to Hq via repeat_interleave.
    Q_bhsd = fa_q_save.permute(1, 0, 2).unsqueeze(0).contiguous()   # [1, Hq, S, D] bf16
    O_bhsd = fa_o_save.permute(1, 0, 2).unsqueeze(0).contiguous()   # [1, Hq, S, D] bf16
    dO_bhsd = d_attn_unfold.to(torch.bfloat16).permute(1, 0, 2) \
        .unsqueeze(0).contiguous()                                  # [1, Hq, S, D] bf16
    K_bhsd = k_cache_layer_S.unsqueeze(0).contiguous()              # [1, Hk, S, D] bf16
    V_bhsd = v_cache_layer_S.unsqueeze(0).contiguous()
    K_e = K_bhsd.repeat_interleave(H_R, dim=1)                      # [1, Hq, S, D]
    V_e = V_bhsd.repeat_interleave(H_R, dim=1)
    LSE_bhs = fa_lse_save.unsqueeze(0).contiguous()                 # [1, Hq, S]   fp32

    scale = 1.0 / math.sqrt(D)
    dQ_bhsd, dK_bhsd, dV_bhsd = fa_backward_flash(
        dO_bhsd, Q_bhsd, K_e, V_e, O_bhsd, LSE_bhs,
        is_causal=True, scale=scale,
        num_kv_heads=Hk,                                          # reduce dK/dV to Hk heads
    )
    # Permute back to [S, H, D].
    dQ_post = dQ_bhsd.squeeze(0).permute(1, 0, 2).contiguous()    # [S, Hq, D] bf16
    dK_post = dK_bhsd.squeeze(0).permute(1, 0, 2).contiguous()    # [S, Hk, D] bf16
    dV_h    = dV_bhsd.squeeze(0).permute(1, 0, 2).contiguous()    # [S, Hk, D] bf16

    # --- Step 5: reverse RoPE for Q and K ------------------------------
    dQ_normed = _qwen_rope_bwd(dQ_post)                           # [S, Hq, D] fp32
    dK_normed = _qwen_rope_bwd(dK_post)                           # [S, Hk, D] fp32

    # --- Step 6: reverse QKnorm for Q and K ----------------------------
    # rms_norm normalizes over the last axis (D). Our bwd_rmsnorm kernel
    # accepts [S_flat, H_flat] with w[H_flat]; treat (S, head) as a flat
    # batch dim and D as the norm axis.
    Q_h_flat = Q_h_recomp.view(S * Hq, D).contiguous()
    dQ_normed_flat = dQ_normed.contiguous().view(S * Hq, D)
    dQ_h_flat = rmsnorm_bwd(Q_h_flat, q_nw, dQ_normed_flat, eps=rms_eps)  # [S*Hq, D] fp32
    dQ_h = dQ_h_flat.view(S, Hq, D)

    # K_h pre-norm — recompute from k_raw analogously to Q.
    k_raw_recomp = (npa @ k_W.t()
                    + lora_scaling * (npa @ k_A) @ k_B)            # [S, FA_KV_SIZE] bf16
    K_h_recomp = k_raw_recomp.view(S, Hk, D).contiguous()
    K_h_flat = K_h_recomp.view(S * Hk, D)
    dK_normed_flat = dK_normed.contiguous().view(S * Hk, D)
    dK_h_flat = rmsnorm_bwd(K_h_flat, k_nw, dK_normed_flat, eps=rms_eps)
    dK_h = dK_h_flat.view(S, Hk, D)

    # --- Step 7: pack dq_raw -------------------------------------------
    # q_packed[:, :, 0, :] = Q_h ; q_packed[:, :, 1, :] = Gate
    dq_packed = torch.empty(S, Hq, 2, D, dtype=torch.float32, device="cuda")
    dq_packed[:, :, 0, :] = dQ_h
    dq_packed[:, :, 1, :] = d_Gate
    dq_raw = dq_packed.reshape(S, _FA_QPROJ_SIZE).contiguous()    # [S, FA_QPROJ_SIZE] fp32

    dk_raw = dK_h.reshape(S, _FA_KV_SIZE).contiguous()
    dv_raw = dV_h.reshape(S, _FA_KV_SIZE).to(torch.float32).contiguous()

    # --- Step 8: q/k/v LoRA bwd ----------------------------------------
    dnpa_q, grad_q_A, grad_q_B = lora_linear_bwd(
        x=npa, A=q_A, B=q_B, base_W=q_W,
        grad_y=dq_raw, scaling=lora_scaling)
    dnpa_k, grad_k_A, grad_k_B = lora_linear_bwd(
        x=npa, A=k_A, B=k_B, base_W=k_W,
        grad_y=dk_raw, scaling=lora_scaling)
    dnpa_v, grad_v_A, grad_v_B = lora_linear_bwd(
        x=npa, A=v_A, B=v_B, base_W=v_W,
        grad_y=dv_raw, scaling=lora_scaling)
    dnpa = dnpa_q + dnpa_k + dnpa_v                                # [S, HIDDEN] fp32

    # --- Step 9: input rmsnorm bwd -------------------------------------
    dh_in_through_norm = rmsnorm_bwd(
        x=hidden_in, w=input_norm_w, dy=dnpa, eps=rms_eps)         # [S, HIDDEN] fp32

    dh_in = dh_in_residual + dh_in_through_norm                    # [S, HIDDEN] fp32

    return {
        "dh_in":      dh_in,
        "grad_q_A":   grad_q_A,   "grad_q_B":   grad_q_B,
        "grad_k_A":   grad_k_A,   "grad_k_B":   grad_k_B,
        "grad_v_A":   grad_v_A,   "grad_v_B":   grad_v_B,
        "grad_o_A":   grad_o_A,   "grad_o_B":   grad_o_B,
    }


# ---------------------------------------------------------------------------
# FA-layer reverse walk: compose mlp_bwd + attn_bwd
# ---------------------------------------------------------------------------

def per_layer_bwd_fa(
    *,
    fa_idx: int,                     # FA-only layer index in [0, N_FA)
    dh_out: torch.Tensor,            # [S, HIDDEN] fp32 — gradient flowing into the
                                      #   layer's output (= layer L+1's dh_in)
    # Saved activations (single-layer slices already taken):
    hidden_in: torch.Tensor,         # [S, HIDDEN] bf16
    normalized_in: torch.Tensor,     # [S, HIDDEN] bf16 — output of input rmsnorm
    normalized_post_attn: torch.Tensor,  # [S, HIDDEN] bf16
    mlp_inter: torch.Tensor,         # [S, INTER]  bf16
    h_post_attn: torch.Tensor,       # [S, HIDDEN] bf16
    attn_out_pre_o: torch.Tensor,    # [S, FA_Q_SIZE] bf16 — input to o_proj
    fa_q_save: torch.Tensor,         # [S, Hq, D]  bf16 — post-RoPE/QKnorm Q
    fa_o_save: torch.Tensor,         # [S, Hq, D]  bf16 — FA output (no gate)
    fa_lse_save: torch.Tensor,       # [Hq, S]     fp32
    k_cache_layer_S: torch.Tensor,   # [Hk, S, D]  bf16 — post-RoPE K from cache
    v_cache_layer_S: torch.Tensor,   # [Hk, S, D]  bf16 — V from cache
    # Frozen base weights for this layer:
    input_norm_w: torch.Tensor,      # [HIDDEN]
    q_W: torch.Tensor, k_W: torch.Tensor, v_W: torch.Tensor,
    q_nw: torch.Tensor, k_nw: torch.Tensor,
    o_W: torch.Tensor,
    post_attn_norm_w: torch.Tensor,
    gate_W: torch.Tensor, up_W: torch.Tensor, down_W: torch.Tensor,
    # Single-layer LoRA tensor slices (already indexed by fa_idx):
    fa_q_A: torch.Tensor,    fa_q_B: torch.Tensor,
    fa_k_A: torch.Tensor,    fa_k_B: torch.Tensor,
    fa_v_A: torch.Tensor,    fa_v_B: torch.Tensor,
    fa_o_A: torch.Tensor,    fa_o_B: torch.Tensor,
    fa_gate_A: torch.Tensor, fa_gate_B: torch.Tensor,
    fa_up_A: torch.Tensor,   fa_up_B: torch.Tensor,
    fa_down_A: torch.Tensor, fa_down_B: torch.Tensor,
    lora_scaling: float,
    rms_eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Compose layer_mlp_bwd and layer_attn_bwd_fa_handrolled for one FA layer.

    Returns:
        dh_in              : [S, HIDDEN] fp32 — flows out of this layer to L-1
        grad_*_A, grad_*_B : 7 LoRA pairs (q, k, v, o, gate, up, down)
    """
    # MLP block first (it sits at the layer's output side).
    mlp_out = layer_mlp_bwd(
        h_post_attn=h_post_attn,
        normalized_post_attn=normalized_post_attn,
        mlp_inter=mlp_inter,
        dh_out=dh_out,
        post_attn_norm_w=post_attn_norm_w,
        gate_W=gate_W, up_W=up_W, down_W=down_W,
        gate_A=fa_gate_A, gate_B=fa_gate_B,
        up_A=fa_up_A, up_B=fa_up_B,
        down_A=fa_down_A, down_B=fa_down_B,
        lora_scaling=lora_scaling, rms_eps=rms_eps,
    )
    # mlp_out["dh_post_attn"] is the gradient flowing into the attention
    # output residual stream — this is the upstream for the attention bwd.

    attn_out = layer_attn_bwd_fa_handrolled(
        hidden_in=hidden_in,
        normalized_in=normalized_in,
        attn_out_pre_o=attn_out_pre_o,
        fa_q_save=fa_q_save,
        fa_o_save=fa_o_save,
        fa_lse_save=fa_lse_save,
        k_cache_layer_S=k_cache_layer_S,
        v_cache_layer_S=v_cache_layer_S,
        dh_post_attn=mlp_out["dh_post_attn"],
        input_norm_w=input_norm_w,
        q_W=q_W, k_W=k_W, v_W=v_W,
        q_nw=q_nw, k_nw=k_nw, o_W=o_W,
        q_A=fa_q_A, q_B=fa_q_B,
        k_A=fa_k_A, k_B=fa_k_B,
        v_A=fa_v_A, v_B=fa_v_B,
        o_A=fa_o_A, o_B=fa_o_B,
        lora_scaling=lora_scaling, rms_eps=rms_eps,
    )

    return {
        "dh_in":      attn_out["dh_in"],
        "grad_q_A":   attn_out["grad_q_A"],   "grad_q_B":   attn_out["grad_q_B"],
        "grad_k_A":   attn_out["grad_k_A"],   "grad_k_B":   attn_out["grad_k_B"],
        "grad_v_A":   attn_out["grad_v_A"],   "grad_v_B":   attn_out["grad_v_B"],
        "grad_o_A":   attn_out["grad_o_A"],   "grad_o_B":   attn_out["grad_o_B"],
        "grad_gate_A": mlp_out["grad_gate_A"], "grad_gate_B": mlp_out["grad_gate_B"],
        "grad_up_A":   mlp_out["grad_up_A"],   "grad_up_B":   mlp_out["grad_up_B"],
        "grad_down_A": mlp_out["grad_down_A"], "grad_down_B": mlp_out["grad_down_B"],
    }


# ---------------------------------------------------------------------------
# DN-layer reverse walk
# ---------------------------------------------------------------------------
#
# DN-attention has no LoRA in the trainer's default config (PEFT targets
# q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj — none match
# the DN attention modules in_proj_qkvz / in_proj_ba / out_proj). So
# per_layer_bwd_dn only needs to:
#   - compute dh_in (gradient flowing back to layer L-1)
#   - compute MLP-side LoRA grads (dn_gate, dn_up, dn_down) — same kernel
#     path as FA, just different LoRA tensor identities
#
# Approach: run autograd through the HF GatedDeltaNet module directly.
# That module already uses fla.chunk_gated_delta_rule under the hood
# (the same kernel our hybrid router calls during forward). The
# autograd path is correct and ~2× a forward DN pass (recompute +
# bwd). Future optimization: replace with our recurrent dn_bwd kernel
# or the chunked CUDA bwd port (Tier 1.6).

def per_layer_bwd_dn(
    *,
    dn_idx: int,                          # DN-only layer index in [0, N_DN)
    dh_out: torch.Tensor,                 # [S, HIDDEN] fp32 — gradient into layer's output
    # Saved activations (single-layer slices already taken):
    hidden_in: torch.Tensor,              # [S, HIDDEN] bf16
    normalized_in: torch.Tensor,          # [S, HIDDEN] bf16 — kernel's saved npa
    normalized_post_attn: torch.Tensor,   # [S, HIDDEN] bf16
    mlp_inter: torch.Tensor,              # [S, INTER]  bf16
    h_post_attn: torch.Tensor,            # [S, HIDDEN] bf16
    # HF layer module — used to read the DN attention base weights.
    hf_layer,
    # Optional: pre-computed DN attention saves dict (from
    # `lora_megakernel_step.precompute_dn_saves`). When provided, the
    # bwd skips `dn_attn_forward` entirely and uses these saves directly.
    dn_saves: dict | None = None,
    # Frozen MLP base weights:
    post_attn_norm_w: torch.Tensor,       # [HIDDEN]
    gate_W: torch.Tensor, up_W: torch.Tensor, down_W: torch.Tensor,
    # Single-layer LoRA slices (DN attention LoRA tensors expected to
    # be zero / not present; only MLP LoRAs matter):
    dn_gate_A: torch.Tensor, dn_gate_B: torch.Tensor,
    dn_up_A: torch.Tensor,   dn_up_B: torch.Tensor,
    dn_down_A: torch.Tensor, dn_down_B: torch.Tensor,
    lora_scaling: float,
    rms_eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Reverse walk of one DN layer.

    Returns:
        dh_in: [S, HIDDEN] fp32 — gradient flowing OUT of this layer
        grad_dn_{gate,up,down}_{A,B}: 6 LoRA grads for the MLP block

    DN attention block uses :func:`dn_attn_handrolled` which calls fla's
    chunk_gated_delta_rule_fwd / _bwd directly (skipping autograd's
    wrapping overhead). 1.48× faster than autograd-through-HF.
    """
    # Lazy import — avoids loading fla at module-load time for callers
    # that only use the FA reverse walk.
    from dn_attn_handrolled import dn_attn_forward, dn_attn_backward

    # MLP block bwd via our kernel (identical to FA's MLP path).
    mlp_out = layer_mlp_bwd(
        h_post_attn=h_post_attn,
        normalized_post_attn=normalized_post_attn,
        mlp_inter=mlp_inter,
        dh_out=dh_out,
        post_attn_norm_w=post_attn_norm_w,
        gate_W=gate_W, up_W=up_W, down_W=down_W,
        gate_A=dn_gate_A, gate_B=dn_gate_B,
        up_A=dn_up_A,     up_B=dn_up_B,
        down_A=dn_down_A, down_B=dn_down_B,
        lora_scaling=lora_scaling, rms_eps=rms_eps,
    )

    # DN attention block: hand-rolled forward + manual fla-bwd.
    # The forward includes input_layernorm; bwd returns dh_in directly.
    if dn_saves is not None:
        # Forward already done at fwd-time via precompute_dn_saves.
        saves = dn_saves
    else:
        input_norm = hf_layer.input_layernorm
        dn = hf_layer.linear_attn
        rms_eps_in = getattr(input_norm, "eps",
                              getattr(input_norm, "variance_epsilon", rms_eps))
        rms_eps_dn = getattr(dn.norm, "eps",
                              getattr(dn.norm, "variance_epsilon", rms_eps))
        h_in_b = hidden_in.unsqueeze(0).contiguous()                       # [1, S, HIDDEN]
        npa_b  = normalized_in.unsqueeze(0).contiguous()                   # [1, S, HIDDEN]
        _attn_out, saves = dn_attn_forward(
            h_in_b,
            input_norm_w=input_norm.weight,
            in_proj_qkv_W=dn.in_proj_qkv.weight,
            in_proj_z_W=dn.in_proj_z.weight,
            in_proj_b_W=dn.in_proj_b.weight,
            in_proj_a_W=dn.in_proj_a.weight,
            conv1d_W=dn.conv1d.weight,
            A_log=dn.A_log, dt_bias=dn.dt_bias,
            dn_norm_W=dn.norm.weight,
            out_proj_W=dn.out_proj.weight,
            rms_eps=rms_eps_in, layer_norm_eps=rms_eps_dn,
            npa_precomputed=npa_b,
        )
    # Upstream is dh_post_attn (mlp + attention contribution to the
    # residual stream's outgoing gradient). dn_attn_backward returns
    # gradient on the residual-stream input via the input-norm path;
    # we add the residual-skip term separately.
    d_attn_out_b = mlp_out["dh_post_attn"].unsqueeze(0)                 # [1, S, HIDDEN] fp32
    dh_in_through_attn = dn_attn_backward(d_attn_out_b, saves)[0]       # [S, HIDDEN] fp32
    # Residual: h_post_attn = h_in + attn_out, so d(h_in)_via_residual = d(h_post_attn).
    dh_in = dh_in_through_attn + mlp_out["dh_post_attn"]

    return {
        "dh_in":         dh_in,
        "grad_dn_gate_A": mlp_out["grad_gate_A"], "grad_dn_gate_B": mlp_out["grad_gate_B"],
        "grad_dn_up_A":   mlp_out["grad_up_A"],   "grad_dn_up_B":   mlp_out["grad_up_B"],
        "grad_dn_down_A": mlp_out["grad_down_A"], "grad_dn_down_B": mlp_out["grad_down_B"],
    }


# ---------------------------------------------------------------------------
# Top-level driver: iterate 24 layers in reverse, dispatch per-layer bwd
# ---------------------------------------------------------------------------

# Canonical kernel order (matches lora_pack._KERNEL_ORDER and the
# torch_bindings.cpp SET(idx, name) table). Each entry is a (name,
# A_shape_per_layer, B_shape_per_layer) tuple where the per-layer
# shapes are looked up in the flat tensors via the head_idx (fa_idx
# or dn_idx).
N_FA_TOTAL = 6
N_DN_TOTAL = 18


def _allocate_flat_grads(lora_flat: list[torch.Tensor]) -> list[torch.Tensor]:
    """Returns a list of 26 fp32 zero-tensors with the same shape as
    each LoRA tensor in lora_flat. These accumulate per-layer grads."""
    return [torch.zeros_like(t, dtype=torch.float32) for t in lora_flat]


# Indices into lora_flat (matching lora_pack._KERNEL_ORDER):
#   FA: 0..6 are (q, k, v, o, gate, up, down) × A then B → tensor pairs
#       at indices 0/1, 2/3, 4/5, 6/7, 8/9, 10/11, 12/13
#   DN: indices 14/15 (qkv), 16/17 (z), 18/19 (out),
#               20/21 (gate), 22/23 (up), 24/25 (down)
_FA_FLAT_IDX = {
    "q":    (0, 1),
    "k":    (2, 3),
    "v":    (4, 5),
    "o":    (6, 7),
    "gate": (8, 9),
    "up":   (10, 11),
    "down": (12, 13),
}
_DN_FLAT_IDX = {
    "qkv":  (14, 15),
    "z":    (16, 17),
    "out":  (18, 19),
    "gate": (20, 21),
    "up":   (22, 23),
    "down": (24, 25),
}


def run_layer_walking_bwd(
    *,
    grad_h_pre_norm: torch.Tensor,    # [S, HIDDEN] fp32 — entry from kernel_loss_autograd
    saves: dict,                       # 6+ activation slabs, each [NUM_LAYERS, S, *]
                                        #   + (optional) FA-only saves
                                        #   fa_q_save / fa_o_save / fa_lse_save
    lora_flat: list[torch.Tensor],     # 26 packed bf16 LoRA tensors
    final_norm_weight: torch.Tensor,   # [HIDDEN] bf16
    hf_model,                          # PEFT-wrapped HF model, used for DN bwd
                                        #   + base-weight access on a per-layer basis
    lora_rank: int,
    lora_scaling: float,
    fa_k_cache: torch.Tensor | None = None,  # [N_FA, Hk, max_seq, D] bf16 — kernel scratch
    fa_v_cache: torch.Tensor | None = None,
    rms_eps: float = 1e-6,
) -> list[torch.Tensor]:
    """Walk layers in reverse, dispatching to per_layer_bwd_fa or
    per_layer_bwd_dn per LAYER_TYPE. Accumulates per-layer LoRA grads
    into a flat 26-tensor list of fp32 grad tensors (matching the
    lora_flat layout) which the caller passes to fused AdamW.

    Computes d(h_pre_norm) → backward through final RMSnorm → dh_out for
    the last layer → walk → dh_out for first layer's input. The first
    layer's dh_in flows back to the embedding (which has no LoRA, so
    discarded).
    """
    S = grad_h_pre_norm.shape[0]

    # --- final RMSnorm bwd: d(h_pre_norm) → d(h_out[NUM_LAYERS-1]) ---
    # h_out[NUM_LAYERS-1] is the input to the final RMSnorm, h_pre_norm
    # in our terminology. We need to backward through:
    #   final_normed = rms_norm(h_pre_norm, final_norm_weight)
    # to get d(h_pre_norm) — but grad_h_pre_norm IS d(h_pre_norm).
    # Wait: kernel_loss_autograd returns grad of loss wrt h_pre_norm
    # (the pre-final-norm hidden), which is exactly the output residual
    # of the last layer. So no extra bwd-rmsnorm here; dh_out for layer
    # NUM_LAYERS-1 is grad_h_pre_norm directly.

    flat_grads = _allocate_flat_grads(lora_flat)
    # bwd_lora_linear and bwd_rmsnorm both require fp32 dy. Autograd's
    # grad on a bf16 leaf comes back as bf16, so cast once at entry and
    # at every per-layer boundary (see end of loop).
    dh = grad_h_pre_norm.to(torch.float32).contiguous()

    fa_idx = N_FA_TOTAL - 1
    dn_idx = N_DN_TOTAL - 1
    # Iterate from last layer to first.
    for L in range(NUM_LAYERS - 1, -1, -1):
        layer_type = LAYER_TYPE[L]
        hf_layer = hf_model.base_model.model.model.layers[L]

        # Pull per-layer activation slices.
        hidden_in_L            = saves["hidden_in"][L]               # [S, HIDDEN] bf16
        normalized_in_L        = saves["normalized_in"][L]           # [S, HIDDEN] bf16
        normalized_post_attn_L = saves["normalized_post_attn"][L]    # [S, HIDDEN] bf16
        mlp_inter_L            = saves["mlp_inter"][L]               # [S, INTER]  bf16
        h_post_attn_L          = saves["h_post_attn"][L]             # [S, HIDDEN] bf16

        # Pull per-layer base norm + MLP weights via the HF module.
        post_attn_norm_w = hf_layer.post_attention_layernorm.weight
        gate_W = hf_layer.mlp.gate_proj.base_layer.weight if hasattr(hf_layer.mlp.gate_proj, "base_layer") else hf_layer.mlp.gate_proj.weight
        up_W   = hf_layer.mlp.up_proj.base_layer.weight   if hasattr(hf_layer.mlp.up_proj, "base_layer")   else hf_layer.mlp.up_proj.weight
        down_W = hf_layer.mlp.down_proj.base_layer.weight if hasattr(hf_layer.mlp.down_proj, "base_layer") else hf_layer.mlp.down_proj.weight

        if layer_type == 1:
            # Full attention layer.
            input_norm_w = hf_layer.input_layernorm.weight
            sa = hf_layer.self_attn
            q_W = sa.q_proj.base_layer.weight if hasattr(sa.q_proj, "base_layer") else sa.q_proj.weight
            k_W = sa.k_proj.base_layer.weight if hasattr(sa.k_proj, "base_layer") else sa.k_proj.weight
            v_W = sa.v_proj.base_layer.weight if hasattr(sa.v_proj, "base_layer") else sa.v_proj.weight
            o_W = sa.o_proj.base_layer.weight if hasattr(sa.o_proj, "base_layer") else sa.o_proj.weight
            q_nw = sa.q_norm.weight
            k_nw = sa.k_norm.weight

            # FA-side LoRA slices for this layer.
            slices = {
                "fa_q_A":    lora_flat[_FA_FLAT_IDX["q"][0]][fa_idx],
                "fa_q_B":    lora_flat[_FA_FLAT_IDX["q"][1]][fa_idx],
                "fa_k_A":    lora_flat[_FA_FLAT_IDX["k"][0]][fa_idx],
                "fa_k_B":    lora_flat[_FA_FLAT_IDX["k"][1]][fa_idx],
                "fa_v_A":    lora_flat[_FA_FLAT_IDX["v"][0]][fa_idx],
                "fa_v_B":    lora_flat[_FA_FLAT_IDX["v"][1]][fa_idx],
                "fa_o_A":    lora_flat[_FA_FLAT_IDX["o"][0]][fa_idx],
                "fa_o_B":    lora_flat[_FA_FLAT_IDX["o"][1]][fa_idx],
                "fa_gate_A": lora_flat[_FA_FLAT_IDX["gate"][0]][fa_idx],
                "fa_gate_B": lora_flat[_FA_FLAT_IDX["gate"][1]][fa_idx],
                "fa_up_A":   lora_flat[_FA_FLAT_IDX["up"][0]][fa_idx],
                "fa_up_B":   lora_flat[_FA_FLAT_IDX["up"][1]][fa_idx],
                "fa_down_A": lora_flat[_FA_FLAT_IDX["down"][0]][fa_idx],
                "fa_down_B": lora_flat[_FA_FLAT_IDX["down"][1]][fa_idx],
            }

            # Hand-rolled FA bwd needs:
            #   - per-layer attn_out_pre_o save (input to o_proj)
            #   - per-layer fa_q_save / fa_o_save / fa_lse_save (FA fwd saves)
            #   - K/V cache slices for THIS FA layer's positions [0, S)
            attn_out_pre_o_L = saves["attn_out_pre_o"][L][:, :_FA_Q_SIZE]
            fa_q_L   = saves["fa_q_save"][fa_idx]
            fa_o_L   = saves["fa_o_save"][fa_idx]
            fa_lse_L = saves["fa_lse_save"][fa_idx]
            S_L = hidden_in_L.shape[0]
            k_cache_L = fa_k_cache[fa_idx, :, :S_L, :].contiguous()  # [Hk, S, D] bf16
            v_cache_L = fa_v_cache[fa_idx, :, :S_L, :].contiguous()

            out = per_layer_bwd_fa(
                fa_idx=fa_idx, dh_out=dh,
                hidden_in=hidden_in_L,
                normalized_in=normalized_in_L,
                normalized_post_attn=normalized_post_attn_L,
                mlp_inter=mlp_inter_L,
                h_post_attn=h_post_attn_L,
                attn_out_pre_o=attn_out_pre_o_L,
                fa_q_save=fa_q_L,
                fa_o_save=fa_o_L,
                fa_lse_save=fa_lse_L,
                k_cache_layer_S=k_cache_L,
                v_cache_layer_S=v_cache_L,
                input_norm_w=input_norm_w,
                q_W=q_W, k_W=k_W, v_W=v_W,
                q_nw=q_nw, k_nw=k_nw, o_W=o_W,
                post_attn_norm_w=post_attn_norm_w,
                gate_W=gate_W, up_W=up_W, down_W=down_W,
                **slices,
                lora_scaling=lora_scaling, rms_eps=rms_eps,
            )

            # Scatter grads into flat_grads at fa_idx.
            for proj, (a_i, b_i) in _FA_FLAT_IDX.items():
                flat_grads[a_i][fa_idx] += out[f"grad_{proj}_A"].to(flat_grads[a_i].dtype)
                flat_grads[b_i][fa_idx] += out[f"grad_{proj}_B"].to(flat_grads[b_i].dtype)
            fa_idx -= 1
        else:
            # DN layer — uses the HF GatedDeltaNet under autograd for
            # the attention block; MLP path identical to FA.
            slices = {
                "dn_gate_A": lora_flat[_DN_FLAT_IDX["gate"][0]][dn_idx],
                "dn_gate_B": lora_flat[_DN_FLAT_IDX["gate"][1]][dn_idx],
                "dn_up_A":   lora_flat[_DN_FLAT_IDX["up"][0]][dn_idx],
                "dn_up_B":   lora_flat[_DN_FLAT_IDX["up"][1]][dn_idx],
                "dn_down_A": lora_flat[_DN_FLAT_IDX["down"][0]][dn_idx],
                "dn_down_B": lora_flat[_DN_FLAT_IDX["down"][1]][dn_idx],
            }
            out = per_layer_bwd_dn(
                dn_idx=dn_idx, dh_out=dh,
                hidden_in=hidden_in_L,
                normalized_in=normalized_in_L,
                normalized_post_attn=normalized_post_attn_L,
                mlp_inter=mlp_inter_L,
                h_post_attn=h_post_attn_L,
                hf_layer=hf_layer,
                dn_saves=saves.get("dn_attn_saves", [None]*NUM_LAYERS)[L]
                          if "dn_attn_saves" in saves else None,
                post_attn_norm_w=post_attn_norm_w,
                gate_W=gate_W, up_W=up_W, down_W=down_W,
                **slices,
                lora_scaling=lora_scaling, rms_eps=rms_eps,
            )

            # Scatter MLP grads (DN attention LoRA grads aren't computed
            # because the trainer's default config doesn't target them).
            for proj in ("gate", "up", "down"):
                a_i, b_i = _DN_FLAT_IDX[proj]
                flat_grads[a_i][dn_idx] += out[f"grad_dn_{proj}_A"].to(flat_grads[a_i].dtype)
                flat_grads[b_i][dn_idx] += out[f"grad_dn_{proj}_B"].to(flat_grads[b_i].dtype)
            dn_idx -= 1

        # The bwd_lora_linear kernel requires grad_y in fp32. Autograd-
        # generated grads on bf16 leaf tensors are bf16; cast back at the
        # layer boundary so the next layer's MLP-down LoRA bwd is happy.
        dh = out["dh_in"].to(torch.float32).contiguous()

    return flat_grads


__all__ = [
    "lora_linear_bwd", "rmsnorm_bwd", "swiglu_bwd",
    "layer_mlp_bwd", "layer_attn_bwd_fa",
    "per_layer_bwd_fa", "per_layer_bwd_dn", "run_layer_walking_bwd",
]
