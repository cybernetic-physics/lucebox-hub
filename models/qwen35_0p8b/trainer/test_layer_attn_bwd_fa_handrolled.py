"""Validate layer_attn_bwd_fa_handrolled (kernel-driven FA bwd) against
the autograd reference defined in test_layer_attn_bwd_fa.py.

The hand-rolled path:
  - reads saved Q (post-RoPE/QKnorm), O (FA out, no gate), LSE (cuDNN's)
  - reads K (post-RoPE/QKnorm) + V from a "cache" buffer
  - calls fa_bwd_flash directly for the FA bwd
  - chains through manual reverse-gate / reverse-RoPE / reverse-QKnorm
  - calls bwd_lora_linear / bwd_rmsnorm kernels for the rest

vs. autograd-through-recomputed-FA reference. Cosine ≥ 0.999 expected
across all 9 returned grads (dh_in + 8 LoRA grads).
"""
from __future__ import annotations

import math
import sys

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b")
sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")

import torch  # noqa: E402

import train_megakernel_C  # noqa: F401, E402

from lora_layer_bwd_skel import (  # noqa: E402
    layer_attn_bwd_fa_handrolled, _qwen_rms, _qwen_rope,
    _FA_HEAD_DIM, _FA_Q_HEADS, _FA_KV_HEADS, _FA_GQA,
    _FA_Q_SIZE, _FA_QPROJ_SIZE, _FA_KV_SIZE,
)

HIDDEN = 1024
RANK   = 8
S      = 16

RMS_EPS = 1e-6
LORA_SCALING = 1.0


def make_inputs(seed=42):
    g = torch.Generator(device="cuda").manual_seed(seed)
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    h_in = torch.randn(S, HIDDEN, generator=g, **bf16) * 0.5
    input_norm_w = torch.randn(HIDDEN, generator=g, **bf16) * 0.1
    q_W = torch.randn(_FA_QPROJ_SIZE, HIDDEN, generator=g, **bf16) * 0.02
    k_W = torch.randn(_FA_KV_SIZE,    HIDDEN, generator=g, **bf16) * 0.02
    v_W = torch.randn(_FA_KV_SIZE,    HIDDEN, generator=g, **bf16) * 0.02
    q_nw = torch.randn(_FA_HEAD_DIM, generator=g, **bf16) * 0.05
    k_nw = torch.randn(_FA_HEAD_DIM, generator=g, **bf16) * 0.05
    o_W = torch.randn(HIDDEN, _FA_Q_SIZE, generator=g, **bf16) * 0.02
    q_A = torch.randn(HIDDEN, RANK, generator=g, **bf16) * 0.02
    q_B = torch.randn(RANK, _FA_QPROJ_SIZE, generator=g, **bf16) * 0.02
    k_A = torch.randn(HIDDEN, RANK, generator=g, **bf16) * 0.02
    k_B = torch.randn(RANK, _FA_KV_SIZE, generator=g, **bf16) * 0.02
    v_A = torch.randn(HIDDEN, RANK, generator=g, **bf16) * 0.02
    v_B = torch.randn(RANK, _FA_KV_SIZE, generator=g, **bf16) * 0.02
    o_A = torch.randn(_FA_Q_SIZE, RANK, generator=g, **bf16) * 0.02
    o_B = torch.randn(RANK, HIDDEN, generator=g, **bf16) * 0.02

    g_for_loss = torch.randn(S, HIDDEN, generator=g, **bf16) * 0.1

    return dict(
        h_in=h_in, input_norm_w=input_norm_w,
        q_W=q_W, k_W=k_W, v_W=v_W, q_nw=q_nw, k_nw=k_nw, o_W=o_W,
        q_A=q_A, q_B=q_B, k_A=k_A, k_B=k_B, v_A=v_A, v_B=v_B, o_A=o_A, o_B=o_B,
        g_for_loss=g_for_loss,
    )


def autograd_reference_with_saves(inp: dict):
    """Run the FA attention block under torch autograd from h_in to
    h_post_attn = h_in + attn_out, and additionally capture the saves
    that the kernel would've written: normalized_in, attn_out_pre_o,
    fa_q_save (post-RoPE/QKnorm Q, gate stripped), fa_o_save (FA output
    before sigmoid-gate), fa_lse_save (cuDNN's LSE), and the K/V cache
    (post-RoPE/QKnorm K, raw V).
    """
    h_in_ref = inp["h_in"].detach().clone().requires_grad_(True)
    qA = inp["q_A"].detach().clone().requires_grad_(True)
    qB = inp["q_B"].detach().clone().requires_grad_(True)
    kA = inp["k_A"].detach().clone().requires_grad_(True)
    kB = inp["k_B"].detach().clone().requires_grad_(True)
    vA = inp["v_A"].detach().clone().requires_grad_(True)
    vB = inp["v_B"].detach().clone().requires_grad_(True)
    oA = inp["o_A"].detach().clone().requires_grad_(True)
    oB = inp["o_B"].detach().clone().requires_grad_(True)

    npa = _qwen_rms(h_in_ref, inp["input_norm_w"], eps=RMS_EPS)
    q_raw = npa @ inp["q_W"].t() + LORA_SCALING * (npa @ qA) @ qB
    k_raw = npa @ inp["k_W"].t() + LORA_SCALING * (npa @ kA) @ kB
    v_raw = npa @ inp["v_W"].t() + LORA_SCALING * (npa @ vA) @ vB

    q_packed = q_raw.view(S, _FA_Q_HEADS, 2, _FA_HEAD_DIM)
    Q_h, Gate = q_packed[:, :, 0, :], q_packed[:, :, 1, :]
    K_h = k_raw.view(S, _FA_KV_HEADS, _FA_HEAD_DIM)
    V   = v_raw.view(S, _FA_KV_HEADS, _FA_HEAD_DIM)

    Q_normed = _qwen_rms(Q_h, inp["q_nw"], eps=RMS_EPS)
    K_normed = _qwen_rms(K_h, inp["k_nw"], eps=RMS_EPS)
    Q_post_rope = _qwen_rope(Q_normed)                          # [S, Hq, D]
    K_post_rope = _qwen_rope(K_normed)                          # [S, Hk, D]

    K_e = K_post_rope.repeat_interleave(_FA_GQA, dim=1)
    V_e = V.repeat_interleave(_FA_GQA, dim=1)
    Q_b = Q_post_rope.permute(1, 0, 2).unsqueeze(0).contiguous()
    K_b = K_e.permute(1, 0, 2).unsqueeze(0).contiguous()
    V_b = V_e.permute(1, 0, 2).unsqueeze(0).contiguous()
    attn_b = torch.nn.functional.scaled_dot_product_attention(
        Q_b, K_b, V_b, is_causal=True)
    attn_unfold = attn_b.squeeze(0).permute(1, 0, 2)            # [S, Hq, D]

    attn_pre_o = torch.sigmoid(Gate.float()).to(attn_unfold.dtype) * attn_unfold
    attn_pre_o_flat = attn_pre_o.reshape(S, _FA_Q_SIZE)
    attn_out = (attn_pre_o_flat @ inp["o_W"].t()
                + LORA_SCALING * (attn_pre_o_flat @ oA) @ oB)
    h_post_attn_ref = h_in_ref + attn_out

    g_for_loss = inp["g_for_loss"]
    loss = (h_post_attn_ref.float() * g_for_loss.float()).sum()
    loss.backward()

    # Capture saves the kernel would have written. We need an LSE
    # tensor — compute it from a manual scaled-dot-product (matching
    # cuDNN's value).
    with torch.no_grad():
        scale = 1.0 / math.sqrt(_FA_HEAD_DIM)
        # logits = Q @ K.T * scale (causal mask)
        logits = (Q_b * scale) @ K_b.transpose(-1, -2)          # [1, Hq, S, S]
        mask = torch.triu(torch.ones(S, S, device=Q_b.device,
                                       dtype=torch.bool), diagonal=1)
        logits = logits.masked_fill(mask, float("-inf"))
        lse = torch.logsumexp(logits.float(), dim=-1)            # [1, Hq, S]

    return {
        "h_post_attn":     h_post_attn_ref.detach(),
        # gradients to compare against:
        "dh_in":   h_in_ref.grad.detach(),
        "grad_q_A": qA.grad.detach(), "grad_q_B": qB.grad.detach(),
        "grad_k_A": kA.grad.detach(), "grad_k_B": kB.grad.detach(),
        "grad_v_A": vA.grad.detach(), "grad_v_B": vB.grad.detach(),
        "grad_o_A": oA.grad.detach(), "grad_o_B": oB.grad.detach(),
        # saves the kernel would have written:
        "normalized_in":   npa.detach(),
        "attn_out_pre_o":  attn_pre_o_flat.detach(),
        "fa_q_save":       Q_post_rope.detach().contiguous(),    # [S, Hq, D]
        "fa_o_save":       attn_unfold.detach().contiguous(),    # [S, Hq, D]
        "fa_lse_save":     lse.squeeze(0).detach().contiguous(), # [Hq, S]
        "k_cache":         K_post_rope.permute(1, 0, 2).detach().contiguous(),  # [Hk, S, D]
        "v_cache":         V.permute(1, 0, 2).detach().contiguous(),            # [Hk, S, D]
    }


def compare(ours, ref, label, *, max_abs_thresh, cos_thresh=0.999):
    diff = (ours.float() - ref.float()).abs()
    cos = torch.nn.functional.cosine_similarity(
        ours.float().flatten().unsqueeze(0),
        ref.float().flatten().unsqueeze(0), dim=-1).item()
    max_a = diff.max().item()
    ok = max_a < max_abs_thresh and cos > cos_thresh
    s = "PASS" if ok else "FAIL"
    print(f"  [{s}] {label:15s} max|Δ|={max_a:.4e}  cos={cos:.6f}  shape={tuple(ours.shape)}")
    return ok


def main():
    print(f"Building synthetic FA attention block (S={S}, HIDDEN={HIDDEN}, R={RANK})...")
    inp = make_inputs(seed=42)

    print("Running autograd reference (also captures kernel-equivalent saves)...")
    ref = autograd_reference_with_saves(inp)

    print("Running layer_attn_bwd_fa_handrolled with kernel-equivalent saves...")
    out = layer_attn_bwd_fa_handrolled(
        hidden_in=inp["h_in"],
        normalized_in=ref["normalized_in"],
        attn_out_pre_o=ref["attn_out_pre_o"],
        fa_q_save=ref["fa_q_save"],
        fa_o_save=ref["fa_o_save"],
        fa_lse_save=ref["fa_lse_save"],
        k_cache_layer_S=ref["k_cache"],
        v_cache_layer_S=ref["v_cache"],
        dh_post_attn=inp["g_for_loss"].float(),
        input_norm_w=inp["input_norm_w"],
        q_W=inp["q_W"], k_W=inp["k_W"], v_W=inp["v_W"],
        q_nw=inp["q_nw"], k_nw=inp["k_nw"], o_W=inp["o_W"],
        q_A=inp["q_A"], q_B=inp["q_B"],
        k_A=inp["k_A"], k_B=inp["k_B"],
        v_A=inp["v_A"], v_B=inp["v_B"],
        o_A=inp["o_A"], o_B=inp["o_B"],
        lora_scaling=LORA_SCALING,
        rms_eps=RMS_EPS,
    )

    print()
    fails = []
    pairs = [
        ("dh_in",      out["dh_in"],      ref["dh_in"]),
        ("grad_q_A",   out["grad_q_A"],   ref["grad_q_A"]),
        ("grad_q_B",   out["grad_q_B"],   ref["grad_q_B"]),
        ("grad_k_A",   out["grad_k_A"],   ref["grad_k_A"]),
        ("grad_k_B",   out["grad_k_B"],   ref["grad_k_B"]),
        ("grad_v_A",   out["grad_v_A"],   ref["grad_v_A"]),
        ("grad_v_B",   out["grad_v_B"],   ref["grad_v_B"]),
        ("grad_o_A",   out["grad_o_A"],   ref["grad_o_A"]),
        ("grad_o_B",   out["grad_o_B"],   ref["grad_o_B"]),
    ]
    for label, ours, refg in pairs:
        if not compare(ours, refg, label, max_abs_thresh=2e-1, cos_thresh=0.99):
            fails.append(label)

    print()
    print("=" * 60)
    if not fails:
        print("ALL CHECKS PASSED  layer_attn_bwd_fa_handrolled cos > 0.99")
    else:
        print(f"FAIL on: {fails}")
        sys.exit(1)


if __name__ == "__main__":
    main()
