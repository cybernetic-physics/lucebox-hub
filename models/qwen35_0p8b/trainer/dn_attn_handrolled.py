"""Hand-rolled DN attention forward+backward via direct fla calls.

Matches Qwen3.5 GatedDeltaNet semantics (in_proj_qkv → causal-conv1d →
silu → split → l2norm-Q/K → chunk_gated_delta_rule → RMSNormGated with
silu(z) gate → out_proj). The DN-attention modules in our trainer's
default config have NO LoRA — only the MLP block has LoRA on
gate/up/down. So this function returns dh_in only, no LoRA grads.

Why this exists: per_layer_bwd_dn currently runs HF's linear_attn under
torch autograd to compute dh_in. That recompiles the full DN forward
+ bwd via fla's autograd.Function wrapper each call, ~5.8 ms/call.
Direct fla calls (chunk_gated_delta_rule_fwd / _bwd, l2norm_fwd / _bwd)
are 2.1× faster than going through autograd. Combined with hand-rolled
chain rules through silu/conv1d/sigmoid/softplus, this trims ~50% of
per-layer DN-bwd time.

Validated against torch autograd through HF.linear_attn at cos > 0.999.
"""
from __future__ import annotations

import math
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")

# fla import (with the existing torch.compile shim).
import _fla_torch_compile_shim  # noqa: F401
from fla.modules.l2norm import l2norm_fwd, l2norm_bwd
from fla.ops.gated_delta_rule.chunk import (
    chunk_gated_delta_rule_fwd, chunk_gated_delta_rule_bwd,
)


# Constants matching Qwen3.5-0.8B's GatedDeltaNet config.
_DN_NUM_HEADS = 16        # both num_v_heads and num_k_heads (no GQA in DN)
_DN_HEAD_DIM  = 128
_DN_KEY_DIM   = _DN_NUM_HEADS * _DN_HEAD_DIM     # 2048
_DN_V_DIM     = _DN_NUM_HEADS * _DN_HEAD_DIM     # 2048
_DN_CONV_CH   = _DN_KEY_DIM * 2 + _DN_V_DIM      # 6144
_DN_CONV_K    = 4


def _qwen_rms(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Qwen3-Next RMSnorm: y = x * rsqrt(mean(x^2) + eps) * (1 + w)."""
    x_f = x.float()
    rstd = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_f * rstd * (1.0 + w.float())).to(x.dtype)


def dn_attn_forward(
    h_in: torch.Tensor,                     # [B, S, HIDDEN]  bf16
    *,
    input_norm_w: torch.Tensor,             # [HIDDEN]        bf16
    in_proj_qkv_W: torch.Tensor,            # [6144, HIDDEN]  bf16
    in_proj_z_W: torch.Tensor,              # [2048, HIDDEN]  bf16
    in_proj_b_W: torch.Tensor,              # [16, HIDDEN]    bf16
    in_proj_a_W: torch.Tensor,              # [16, HIDDEN]    bf16
    conv1d_W: torch.Tensor,                 # [6144, 1, 4]    bf16
    A_log: torch.Tensor,                    # [16]            fp32
    dt_bias: torch.Tensor,                  # [16]            fp32
    dn_norm_W: torch.Tensor,                # [128]           bf16
    out_proj_W: torch.Tensor,               # [HIDDEN, 2048]  bf16
    rms_eps: float = 1e-6,
    layer_norm_eps: float = 1e-6,
    npa_precomputed: torch.Tensor | None = None,  # [B, S, HIDDEN] bf16
) -> tuple[torch.Tensor, dict]:
    """Run DN attention forward in pure Python (matches HF.linear_attn
    semantically). Returns (attn_out, saves_for_bwd).

    If ``npa_precomputed`` is given, the input RMSnorm step is skipped
    (the kernel already saved this tensor as ``saves["normalized_in"][L]``,
    so passing it in saves ~0.1 ms/call of redundant compute).

    saves_for_bwd is a dict with the intermediates the manual bwd needs.
    """
    B, S, H = h_in.shape
    Hh = _DN_NUM_HEADS
    D  = _DN_HEAD_DIM
    KD = _DN_KEY_DIM           # 2048
    VD = _DN_V_DIM             # 2048
    CC = _DN_CONV_CH           # 6144
    KK = _DN_CONV_K

    # 1. Input RMSnorm (skipped when caller supplies npa_precomputed).
    if npa_precomputed is not None:
        npa = npa_precomputed
    else:
        npa = _qwen_rms(h_in, input_norm_w, eps=rms_eps)               # bf16

    # 2. Linear projections (no LoRA on DN attention).
    qkv_raw = npa @ in_proj_qkv_W.t()                                  # [B, S, 6144] bf16
    z_raw = npa @ in_proj_z_W.t()                                      # [B, S, 2048] bf16
    b_raw = npa @ in_proj_b_W.t()                                      # [B, S, 16]   bf16
    a_raw = npa @ in_proj_a_W.t()                                      # [B, S, 16]   bf16

    # 3. Causal depthwise conv1d.
    qkv_t = qkv_raw.transpose(1, 2).contiguous()                       # [B, 6144, S]
    qkv_pre_silu = F.conv1d(qkv_t, conv1d_W, groups=CC, padding=KK - 1)[..., :S]
    qkv_post_silu = F.silu(qkv_pre_silu)                                # [B, 6144, S]
    qkv = qkv_post_silu.transpose(1, 2).contiguous()                    # [B, S, 6144]

    # 4. Split + reshape.
    q_raw, k_raw, v_raw = torch.split(qkv, [KD, KD, VD], dim=-1)
    q = q_raw.reshape(B, S, Hh, D).contiguous()
    k = k_raw.reshape(B, S, Hh, D).contiguous()
    v = v_raw.reshape(B, S, Hh, D).contiguous()

    # 5. l2norm Q/K (kernel does it inside chunk_gated_delta_rule when
    #    use_qk_l2norm_in_kernel=True; we run it explicitly so we can
    #    reuse the rstd in bwd without a recompute).
    q_l2, q_rstd = l2norm_fwd(q)
    k_l2, k_rstd = l2norm_fwd(k)

    # 6. beta, g.
    beta = b_raw.float().sigmoid().contiguous()                         # [B, S, 16] fp32
    g_log = (-A_log.float().exp()
             * F.softplus(a_raw.float() + dt_bias.float())).contiguous()  # [B, S, 16] fp32

    # 7. fla chunk_gated_delta_rule_fwd (direct, no autograd).
    scale = 1.0 / math.sqrt(D)
    g_post, y_pre, A_state, _final_state, init_state, g_input = chunk_gated_delta_rule_fwd(
        q=q_l2, k=k_l2, v=v, g=g_log, beta=beta,
        scale=scale, initial_state=None, output_final_state=False,
        cu_seqlens=None, cp_context=None, chunk_indices=None,
        transpose_state_layout=False, use_gate_in_kernel=False,
        A_log=None, dt_bias=None,
    )                                                                    # [B, S, 16, 128]

    # 8. RMSNormGated: y_norm = norm_W * y_pre * rstd_y, then *silu(z).
    z_h = z_raw.reshape(B, S, Hh, D).contiguous()
    y_pre_f = y_pre.float()
    var_y = y_pre_f.pow(2).mean(dim=-1, keepdim=True)
    rstd_y = torch.rsqrt(var_y + layer_norm_eps)                         # [B, S, 16, 1] fp32
    y_normed_pre_gate = (dn_norm_W.float() * y_pre_f * rstd_y).to(y_pre.dtype)
    z_silu = F.silu(z_h.float()).to(y_pre.dtype)
    attn_pre_o_h = y_normed_pre_gate * z_silu                            # [B, S, 16, 128]
    attn_pre_o = attn_pre_o_h.reshape(B, S, VD).contiguous()             # [B, S, 2048]

    # 9. out_proj (no LoRA).
    attn_out = attn_pre_o @ out_proj_W.t()                                # [B, S, HIDDEN] bf16

    saves = dict(
        h_in=h_in, npa=npa, qkv_raw=qkv_raw, z_raw=z_raw, b_raw=b_raw, a_raw=a_raw,
        qkv_pre_silu=qkv_pre_silu, qkv_post_silu=qkv_post_silu,
        q=q, k=k, v=v, q_l2=q_l2, q_rstd=q_rstd, k_l2=k_l2, k_rstd=k_rstd,
        beta=beta, g_log=g_log, g_post=g_post, A_state=A_state, init_state=init_state,
        g_input=g_input,
        y_pre=y_pre, rstd_y=rstd_y, z_h=z_h, z_silu=z_silu,
        y_normed_pre_gate=y_normed_pre_gate, attn_pre_o=attn_pre_o,
        scale=scale,
        rms_eps=rms_eps, layer_norm_eps=layer_norm_eps,
        # frozen weight refs (kept for bwd):
        input_norm_w=input_norm_w,
        in_proj_qkv_W=in_proj_qkv_W, in_proj_z_W=in_proj_z_W,
        in_proj_b_W=in_proj_b_W, in_proj_a_W=in_proj_a_W,
        conv1d_W=conv1d_W, A_log=A_log, dt_bias=dt_bias,
        dn_norm_W=dn_norm_W, out_proj_W=out_proj_W,
    )
    return attn_out, saves


def dn_attn_backward(d_attn_out: torch.Tensor, saves: dict) -> torch.Tensor:
    """Hand-rolled bwd through the Python DN forward, using fla's
    chunk_gated_delta_rule_bwd directly. Returns dh_in [B, S, HIDDEN] fp32.
    """
    B, S, H = saves["h_in"].shape
    Hh = _DN_NUM_HEADS
    D  = _DN_HEAD_DIM
    KD = _DN_KEY_DIM
    VD = _DN_V_DIM
    CC = _DN_CONV_CH
    KK = _DN_CONV_K

    out_proj_W = saves["out_proj_W"]
    in_proj_qkv_W = saves["in_proj_qkv_W"]
    in_proj_z_W = saves["in_proj_z_W"]
    in_proj_b_W = saves["in_proj_b_W"]
    in_proj_a_W = saves["in_proj_a_W"]
    conv1d_W    = saves["conv1d_W"]
    input_norm_w = saves["input_norm_w"]
    dn_norm_W = saves["dn_norm_W"]
    A_log = saves["A_log"]
    dt_bias = saves["dt_bias"]

    # --- Step 1: out_proj bwd (frozen). Returns d(attn_pre_o). -----------
    # attn_out = attn_pre_o @ out_proj_W.T → d(attn_pre_o) = d_attn_out @ out_proj_W
    d_attn_pre_o = (d_attn_out.float() @ out_proj_W.float())             # [B, S, 2048] fp32

    # --- Step 2: gate split (sigmoid silu(z) * y_normed_pre_gate) ---------
    d_attn_pre_o_h = d_attn_pre_o.view(B, S, Hh, D)
    z_silu = saves["z_silu"]
    y_normed_pre_gate = saves["y_normed_pre_gate"]
    # Forward: attn_pre_o_h = y_normed * z_silu (elementwise)
    d_y_normed = d_attn_pre_o_h * z_silu.float()
    d_z_silu = d_attn_pre_o_h * y_normed_pre_gate.float()
    # silu bwd: dsilu/dz = sigmoid(z) * (1 + z * (1 - sigmoid(z)))
    z_h_f = saves["z_h"].float()
    sig_z = torch.sigmoid(z_h_f)
    silu_grad = sig_z * (1.0 + z_h_f * (1.0 - sig_z))
    d_z = (d_z_silu * silu_grad).contiguous()                            # [B, S, 16, 128] fp32

    # --- Step 3: RMSNormGated bwd (no gate path; pure RMSNorm with norm_W gain) ---
    # y_normed = norm_W * x * rstd, where x = y_pre and rstd = rsqrt(mean(x^2)+eps)
    # dx[i] = norm_W[i] * rstd * dy_normed[i] - rstd^3 / D * x[i] * sum_j(dy_normed[j] * norm_W[j] * x[j])
    y_pre = saves["y_pre"]
    rstd_y = saves["rstd_y"]                                              # [B, S, 16, 1] fp32
    norm_W_f = dn_norm_W.float()
    y_pre_f = y_pre.float()
    # Inner sum over D: sum_d (dy_normed[d] * norm_W[d] * x[d])
    dot = (d_y_normed * norm_W_f * y_pre_f).sum(dim=-1, keepdim=True)
    d_y_pre = norm_W_f * rstd_y * d_y_normed - (rstd_y ** 3) / float(D) * y_pre_f * dot
    d_y_pre = d_y_pre.to(y_pre.dtype).contiguous()                        # [B, S, 16, 128]

    # --- Step 4: chunk_gated_delta_rule_bwd ------------------------------
    q_l2 = saves["q_l2"]
    k_l2 = saves["k_l2"]
    v    = saves["v"]
    g_post = saves["g_post"]
    beta = saves["beta"]
    A_state = saves["A_state"]
    init_state = saves["init_state"]
    g_input = saves["g_input"]
    scale = saves["scale"]
    dq, dk, dv, dbeta, dg, _dh0, _dA_log, _ddt_bias = chunk_gated_delta_rule_bwd(
        q=q_l2, k=k_l2, v=v, g=g_post, beta=beta, A=A_state, scale=scale,
        initial_state=init_state, do=d_y_pre, dht=None,
        cu_seqlens=None, cp_context=None, chunk_indices=None,
        transpose_state_layout=False, use_gate_in_kernel=False,
        g_input=g_input, A_log=None, dt_bias=None,
    )
    # l2norm_bwd: pass POST-norm Q (= q_l2) and rstd
    dq = l2norm_bwd(q_l2, saves["q_rstd"], dq)
    dk = l2norm_bwd(k_l2, saves["k_rstd"], dk)

    # --- Step 5: reshape Q/K/V back to [B, S, KD] / [B, S, VD] flat ------
    dq_flat = dq.reshape(B, S, KD).contiguous()
    dk_flat = dk.reshape(B, S, KD).contiguous()
    dv_flat = dv.reshape(B, S, VD).contiguous()
    dqkv = torch.cat([dq_flat, dk_flat, dv_flat], dim=-1)                 # [B, S, 6144]

    # --- Step 6: silu bwd (qkv_post_silu = silu(qkv_pre_silu)) -----------
    # Need d(qkv_pre_silu) given d(qkv_post_silu) and qkv_pre_silu.
    # qkv_post_silu was computed in transposed layout [B, 6144, S]; bring
    # dqkv into the same layout for the silu/conv bwd.
    qkv_pre_silu = saves["qkv_pre_silu"]                                  # [B, 6144, S] bf16
    dqkv_t = dqkv.transpose(1, 2).contiguous()                            # [B, 6144, S] fp32
    qkv_pre_silu_f = qkv_pre_silu.float()
    sig_qkv = torch.sigmoid(qkv_pre_silu_f)
    silu_grad_qkv = sig_qkv * (1.0 + qkv_pre_silu_f * (1.0 - sig_qkv))
    d_qkv_pre_silu = dqkv_t * silu_grad_qkv                               # [B, 6144, S] fp32

    # --- Step 7: conv1d bwd (depthwise causal) ---------------------------
    # qkv_t = transpose(qkv_raw, 1, 2)              shape [B, 6144, S]
    # qkv_pre_silu = F.conv1d(qkv_t, W, groups=6144, padding=KK-1)[..., :S]
    # The conv was depthwise (groups=in_channels). Bwd via F.conv1d with
    # padding mirrored: gradient w.r.t. input.
    # Use torch's conv1d backward via F.grad.conv1d_input (or by hand).
    # Easier: construct the gradient via F.conv1d with the W flipped;
    # since groups=CC, each channel is independent.
    # The forward (with padding=KK-1, no slicing) produces [B, CC, S+KK-1];
    # then slice to [..., :S]. So d(conv_full_out) is dqkv_pre_silu padded
    # with zeros for indices [S, S+KK-1).
    full_out_S = S + KK - 1
    d_conv_full = torch.zeros(
        B, CC, full_out_S,
        dtype=d_qkv_pre_silu.dtype, device=d_qkv_pre_silu.device,
    )
    d_conv_full[..., :S] = d_qkv_pre_silu
    # For depthwise causal conv1d with kernel K, the input gradient is the
    # cross-correlation (transpose conv) of d_out with the kernel. F.conv_transpose1d
    # with groups=CC implements exactly this.
    d_qkv_t = F.conv_transpose1d(d_conv_full, conv1d_W.float(),
                                  groups=CC, padding=KK - 1)            # [B, CC, S]
    d_qkv_raw = d_qkv_t.transpose(1, 2).contiguous()                     # [B, S, 6144] fp32

    # --- Step 8: in_proj_qkv bwd (no LoRA) -------------------------------
    # qkv_raw = npa @ W.T → d(npa)_from_qkv = d(qkv_raw) @ W (fp32 matmul)
    d_npa_from_qkv = d_qkv_raw @ in_proj_qkv_W.float()                    # [B, S, HIDDEN] fp32

    # --- Step 9: in_proj_z bwd -------------------------------------------
    d_z_flat = d_z.reshape(B, S, VD).contiguous()                          # [B, S, 2048]
    d_npa_from_z = d_z_flat @ in_proj_z_W.float()                          # [B, S, HIDDEN]

    # --- Step 10: in_proj_b / in_proj_a bwd ------------------------------
    # beta = sigmoid(b_raw); db_logit = dbeta * sigmoid * (1-sigmoid)
    b_raw_f = saves["b_raw"].float()
    sig_b = torch.sigmoid(b_raw_f)
    db_logit = dbeta * sig_b * (1.0 - sig_b)                               # [B, S, 16] fp32
    d_npa_from_b = db_logit @ in_proj_b_W.float()                          # [B, S, HIDDEN]

    # g_log = -A_log.exp() * softplus(a_raw + dt_bias)
    # da = dg * (-A_log.exp()) * sigmoid(a_raw + dt_bias)   (softplus' = sigmoid)
    a_raw_f = saves["a_raw"].float()
    sig_a = torch.sigmoid(a_raw_f + dt_bias.float())
    da = dg * (-A_log.float().exp()) * sig_a                                # [B, S, 16] fp32
    d_npa_from_a = da @ in_proj_a_W.float()                                 # [B, S, HIDDEN]

    # --- Step 11: combine npa grads --------------------------------------
    d_npa = d_npa_from_qkv + d_npa_from_z + d_npa_from_b + d_npa_from_a    # [B, S, HIDDEN]

    # --- Step 12: input rmsnorm bwd: y = x * rstd * (1 + w) --------------
    # y[i,j] = x[i,j] * rstd[i] * (1 + w[j])  where rstd[i] = rsqrt(mean(x[i,:]^2) + eps)
    # dx = (1+w) * rstd * dy - rstd^3 / H * x * sum_j(dy[j] * (1+w[j]) * x[j])
    h_in = saves["h_in"]
    h_in_f = h_in.float()
    var_h = h_in_f.pow(2).mean(dim=-1, keepdim=True)
    rstd_h = torch.rsqrt(var_h + saves["rms_eps"])
    inw_f = (1.0 + input_norm_w.float())
    dot_h = (d_npa * inw_f * h_in_f).sum(dim=-1, keepdim=True)
    d_h_in = inw_f * rstd_h * d_npa - (rstd_h ** 3) / float(H) * h_in_f * dot_h

    return d_h_in.contiguous()                                              # [B, S, HIDDEN] fp32


__all__ = ["dn_attn_forward", "dn_attn_backward"]
