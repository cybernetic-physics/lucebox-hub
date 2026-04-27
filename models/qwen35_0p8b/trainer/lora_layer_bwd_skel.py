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


def _qwen_rope(x: torch.Tensor) -> torch.Tensor:
    """Apply Qwen3-Next RoPE to ``x`` of shape [S, num_heads, FA_HEAD_DIM].
    Only the first FA_ROT_DIM (=64) dimensions of each head get rotated.
    The pairing is "halves": for i ∈ [0, ROT_DIM/2), the partner is i+ROT_DIM/2.
    """
    S, H, D = x.shape
    assert D == _FA_HEAD_DIM, f"expected head_dim={_FA_HEAD_DIM}, got {D}"
    half = _FA_ROT_DIM // 2
    pos = torch.arange(S, device=x.device, dtype=torch.float32)
    # freq[i] for i ∈ [0, half) is 1/theta^(2*i/ROT_DIM).
    i = torch.arange(half, device=x.device, dtype=torch.float32)
    fe = 2.0 * i / _FA_ROT_DIM
    inv_theta = 1.0 / (_FA_ROPE_THETA ** fe)            # [half]
    angles = pos.unsqueeze(1) * inv_theta.unsqueeze(0)  # [S, half]
    cos = torch.cos(angles)                              # [S, half]
    sin = torch.sin(angles)
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
# FA-layer reverse walk: compose mlp_bwd + attn_bwd
# ---------------------------------------------------------------------------

def per_layer_bwd_fa(
    *,
    fa_idx: int,                     # FA-only layer index in [0, N_FA)
    dh_out: torch.Tensor,            # [S, HIDDEN] fp32 — gradient flowing into the
                                      #   layer's output (= layer L+1's dh_in)
    # Saved activations (single-layer slices already taken):
    hidden_in: torch.Tensor,         # [S, HIDDEN] bf16
    normalized_post_attn: torch.Tensor,  # [S, HIDDEN] bf16
    mlp_inter: torch.Tensor,         # [S, INTER]  bf16
    h_post_attn: torch.Tensor,       # [S, HIDDEN] bf16
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
    """Compose layer_mlp_bwd and layer_attn_bwd_fa for one FA layer.

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

    attn_out = layer_attn_bwd_fa(
        hidden_in=hidden_in,
        h_post_attn=h_post_attn,
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
    "layer_mlp_bwd", "layer_attn_bwd_fa",
    "per_layer_bwd_fa", "per_layer_bwd_dn", "run_layer_walking_bwd",
]
