"""End-to-end validation of per_layer_bwd_fa for one FA layer.

Builds a complete FA-layer block (input rmsnorm → qkv → SDPA → o_proj
→ residual → post-attn rmsnorm → gate/up → SwiGLU → down → residual)
with the same LoRA + base-weight + norm-weight structure as a real
Qwen3.5-0.8B layer, runs torch autograd as the reference, runs our
per_layer_bwd_fa, compares all 14 returned gradients (dh_in + 7 LoRA
A/B pairs).
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b")
sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")

import torch  # noqa: E402

import train_megakernel_C  # noqa: F401, E402

from lora_layer_bwd_skel import (  # noqa: E402
    per_layer_bwd_fa, _qwen_rms, _qwen_rope,
    _FA_HEAD_DIM, _FA_Q_HEADS, _FA_KV_HEADS, _FA_GQA,
    _FA_Q_SIZE, _FA_QPROJ_SIZE, _FA_KV_SIZE,
)

HIDDEN = 1024
INTER  = 3584
RANK   = 8
S      = 16

RMS_EPS = 1e-6
LORA_SCALING = 1.0


def make_inputs(seed=42):
    g = torch.Generator(device="cuda").manual_seed(seed)
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    h_in = torch.randn(S, HIDDEN, generator=g, **bf16) * 0.5
    input_norm_w     = torch.randn(HIDDEN, generator=g, **bf16) * 0.1
    post_attn_norm_w = torch.randn(HIDDEN, generator=g, **bf16) * 0.1
    q_W = torch.randn(_FA_QPROJ_SIZE, HIDDEN, generator=g, **bf16) * 0.02
    k_W = torch.randn(_FA_KV_SIZE,    HIDDEN, generator=g, **bf16) * 0.02
    v_W = torch.randn(_FA_KV_SIZE,    HIDDEN, generator=g, **bf16) * 0.02
    q_nw = torch.randn(_FA_HEAD_DIM, generator=g, **bf16) * 0.05
    k_nw = torch.randn(_FA_HEAD_DIM, generator=g, **bf16) * 0.05
    o_W = torch.randn(HIDDEN, _FA_Q_SIZE, generator=g, **bf16) * 0.02
    gate_W = torch.randn(INTER, HIDDEN, generator=g, **bf16) * 0.02
    up_W   = torch.randn(INTER, HIDDEN, generator=g, **bf16) * 0.02
    down_W = torch.randn(HIDDEN, INTER, generator=g, **bf16) * 0.02
    # LoRA tensors — single-layer (no batch / index axis).
    fa_q_A = torch.randn(HIDDEN,         RANK, generator=g, **bf16) * 0.02
    fa_q_B = torch.randn(RANK, _FA_QPROJ_SIZE, generator=g, **bf16) * 0.02
    fa_k_A = torch.randn(HIDDEN,         RANK, generator=g, **bf16) * 0.02
    fa_k_B = torch.randn(RANK, _FA_KV_SIZE,    generator=g, **bf16) * 0.02
    fa_v_A = torch.randn(HIDDEN,         RANK, generator=g, **bf16) * 0.02
    fa_v_B = torch.randn(RANK, _FA_KV_SIZE,    generator=g, **bf16) * 0.02
    fa_o_A = torch.randn(_FA_Q_SIZE,     RANK, generator=g, **bf16) * 0.02
    fa_o_B = torch.randn(RANK, HIDDEN,         generator=g, **bf16) * 0.02
    fa_gate_A = torch.randn(HIDDEN, RANK, generator=g, **bf16) * 0.02
    fa_gate_B = torch.randn(RANK, INTER, generator=g, **bf16) * 0.02
    fa_up_A = torch.randn(HIDDEN, RANK, generator=g, **bf16) * 0.02
    fa_up_B = torch.randn(RANK, INTER, generator=g, **bf16) * 0.02
    fa_down_A = torch.randn(INTER, RANK, generator=g, **bf16) * 0.02
    fa_down_B = torch.randn(RANK, HIDDEN, generator=g, **bf16) * 0.02
    g_for_loss = torch.randn(S, HIDDEN, generator=g, **bf16) * 0.1

    return dict(
        h_in=h_in, input_norm_w=input_norm_w, post_attn_norm_w=post_attn_norm_w,
        q_W=q_W, k_W=k_W, v_W=v_W, q_nw=q_nw, k_nw=k_nw, o_W=o_W,
        gate_W=gate_W, up_W=up_W, down_W=down_W,
        fa_q_A=fa_q_A, fa_q_B=fa_q_B, fa_k_A=fa_k_A, fa_k_B=fa_k_B,
        fa_v_A=fa_v_A, fa_v_B=fa_v_B, fa_o_A=fa_o_A, fa_o_B=fa_o_B,
        fa_gate_A=fa_gate_A, fa_gate_B=fa_gate_B,
        fa_up_A=fa_up_A,     fa_up_B=fa_up_B,
        fa_down_A=fa_down_A, fa_down_B=fa_down_B,
        g_for_loss=g_for_loss,
    )


def autograd_reference(inp):
    """Run the full FA layer block under autograd and return:
       - saves: normalized_post_attn, mlp_inter, h_post_attn (used by ours)
       - g for h_in and 14 LoRA tensors
       - h_out (= h_post_attn + mlp_out)
    """
    h = inp["h_in"].detach().clone().requires_grad_(True)
    qA = inp["fa_q_A"].detach().clone().requires_grad_(True)
    qB = inp["fa_q_B"].detach().clone().requires_grad_(True)
    kA = inp["fa_k_A"].detach().clone().requires_grad_(True)
    kB = inp["fa_k_B"].detach().clone().requires_grad_(True)
    vA = inp["fa_v_A"].detach().clone().requires_grad_(True)
    vB = inp["fa_v_B"].detach().clone().requires_grad_(True)
    oA = inp["fa_o_A"].detach().clone().requires_grad_(True)
    oB = inp["fa_o_B"].detach().clone().requires_grad_(True)
    gateA = inp["fa_gate_A"].detach().clone().requires_grad_(True)
    gateB = inp["fa_gate_B"].detach().clone().requires_grad_(True)
    upA = inp["fa_up_A"].detach().clone().requires_grad_(True)
    upB = inp["fa_up_B"].detach().clone().requires_grad_(True)
    downA = inp["fa_down_A"].detach().clone().requires_grad_(True)
    downB = inp["fa_down_B"].detach().clone().requires_grad_(True)

    # Attention block.
    npa_in = _qwen_rms(h, inp["input_norm_w"], eps=RMS_EPS)
    q_raw = npa_in @ inp["q_W"].t() + LORA_SCALING * (npa_in @ qA) @ qB
    k_raw = npa_in @ inp["k_W"].t() + LORA_SCALING * (npa_in @ kA) @ kB
    v_raw = npa_in @ inp["v_W"].t() + LORA_SCALING * (npa_in @ vA) @ vB
    q_packed = q_raw.view(S, _FA_Q_HEADS, 2, _FA_HEAD_DIM)
    Q_h = q_packed[:, :, 0, :]; Gate = q_packed[:, :, 1, :]
    K_h = k_raw.view(S, _FA_KV_HEADS, _FA_HEAD_DIM)
    V   = v_raw.view(S, _FA_KV_HEADS, _FA_HEAD_DIM)
    Q = _qwen_rope(_qwen_rms(Q_h, inp["q_nw"], eps=RMS_EPS))
    K = _qwen_rope(_qwen_rms(K_h, inp["k_nw"], eps=RMS_EPS))
    K_e = K.repeat_interleave(_FA_GQA, dim=1)
    V_e = V.repeat_interleave(_FA_GQA, dim=1)
    Q_b = Q.permute(1, 0, 2).unsqueeze(0).contiguous()
    K_b = K_e.permute(1, 0, 2).unsqueeze(0).contiguous()
    V_b = V_e.permute(1, 0, 2).unsqueeze(0).contiguous()
    attn_b = torch.nn.functional.scaled_dot_product_attention(Q_b, K_b, V_b, is_causal=True)
    attn_unfold = attn_b.squeeze(0).permute(1, 0, 2)
    attn_pre_o = (torch.sigmoid(Gate.float()).to(attn_unfold.dtype) * attn_unfold).reshape(S, _FA_Q_SIZE)
    attn_out = attn_pre_o @ inp["o_W"].t() + LORA_SCALING * (attn_pre_o @ oA) @ oB
    h_post_attn = h + attn_out

    # MLP block.
    npa_post = _qwen_rms(h_post_attn, inp["post_attn_norm_w"], eps=RMS_EPS)
    gate = npa_post @ inp["gate_W"].t() + LORA_SCALING * (npa_post @ gateA) @ gateB
    up   = npa_post @ inp["up_W"].t()   + LORA_SCALING * (npa_post @ upA) @ upB
    mlp_inter = torch.nn.functional.silu(gate.float()).to(gate.dtype) * up
    mlp_out = mlp_inter @ inp["down_W"].t() + LORA_SCALING * (mlp_inter @ downA) @ downB
    h_out = h_post_attn + mlp_out

    loss = (h_out.float() * inp["g_for_loss"].float()).sum()
    loss.backward()

    # Capture FA saves the kernel would have written.
    import math as _math
    with torch.no_grad():
        scale = 1.0 / _math.sqrt(_FA_HEAD_DIM)
        logits = (Q_b * scale) @ K_b.transpose(-1, -2)
        mask = torch.triu(torch.ones(S, S, device=Q_b.device, dtype=torch.bool),
                          diagonal=1)
        logits = logits.masked_fill(mask, float("-inf"))
        lse = torch.logsumexp(logits.float(), dim=-1).squeeze(0).contiguous()  # [Hq, S]

    return {
        # Saves our function will need:
        "h_post_attn":          h_post_attn.detach().to(torch.bfloat16),
        "normalized_in":        npa_in.detach().to(torch.bfloat16),
        "normalized_post_attn": npa_post.detach().to(torch.bfloat16),
        "mlp_inter":            mlp_inter.detach().to(torch.bfloat16),
        "attn_out_pre_o":       attn_pre_o.detach().to(torch.bfloat16),
        "fa_q_save":            Q.detach().contiguous(),                       # [S, Hq, D] bf16
        "fa_o_save":            attn_unfold.detach().contiguous(),              # [S, Hq, D] bf16
        "fa_lse_save":          lse,                                             # [Hq, S] fp32
        "k_cache":              K.permute(1, 0, 2).detach().contiguous(),       # [Hk, S, D] bf16
        "v_cache":              V.permute(1, 0, 2).detach().contiguous(),       # [Hk, S, D] bf16
        # Reference grads:
        "dh_in": h.grad.detach(),
        "grad_q_A": qA.grad.detach(), "grad_q_B": qB.grad.detach(),
        "grad_k_A": kA.grad.detach(), "grad_k_B": kB.grad.detach(),
        "grad_v_A": vA.grad.detach(), "grad_v_B": vB.grad.detach(),
        "grad_o_A": oA.grad.detach(), "grad_o_B": oB.grad.detach(),
        "grad_gate_A": gateA.grad.detach(), "grad_gate_B": gateB.grad.detach(),
        "grad_up_A":   upA.grad.detach(),   "grad_up_B":   upB.grad.detach(),
        "grad_down_A": downA.grad.detach(), "grad_down_B": downB.grad.detach(),
    }


def compare(ours, ref, label, *, max_abs_thresh, cos_thresh=0.999):
    diff = (ours.float() - ref.float()).abs()
    cos = torch.nn.functional.cosine_similarity(
        ours.float().flatten().unsqueeze(0),
        ref.float().flatten().unsqueeze(0), dim=-1).item()
    max_a = diff.max().item()
    ok = max_a < max_abs_thresh and cos > cos_thresh
    s = "✓" if ok else "✗"
    print(f"  {s} {label:15s} max|Δ|={max_a:.4e}  cos={cos:.6f}  shape={tuple(ours.shape)}")
    return ok


def main():
    print(f"Building synthetic full FA layer (S={S}, HIDDEN={HIDDEN}, INTER={INTER}, R={RANK})...")
    inp = make_inputs(seed=42)

    print("Running autograd reference (full layer, end-to-end)...")
    ref = autograd_reference(inp)

    print("Running our per_layer_bwd_fa...")
    out = per_layer_bwd_fa(
        fa_idx=0,
        dh_out=inp["g_for_loss"].float(),
        hidden_in=inp["h_in"],
        normalized_in=ref["normalized_in"],
        normalized_post_attn=ref["normalized_post_attn"],
        mlp_inter=ref["mlp_inter"],
        h_post_attn=ref["h_post_attn"],
        attn_out_pre_o=ref["attn_out_pre_o"],
        fa_q_save=ref["fa_q_save"],
        fa_o_save=ref["fa_o_save"],
        fa_lse_save=ref["fa_lse_save"],
        k_cache_layer_S=ref["k_cache"],
        v_cache_layer_S=ref["v_cache"],
        input_norm_w=inp["input_norm_w"],
        q_W=inp["q_W"], k_W=inp["k_W"], v_W=inp["v_W"],
        q_nw=inp["q_nw"], k_nw=inp["k_nw"], o_W=inp["o_W"],
        post_attn_norm_w=inp["post_attn_norm_w"],
        gate_W=inp["gate_W"], up_W=inp["up_W"], down_W=inp["down_W"],
        fa_q_A=inp["fa_q_A"], fa_q_B=inp["fa_q_B"],
        fa_k_A=inp["fa_k_A"], fa_k_B=inp["fa_k_B"],
        fa_v_A=inp["fa_v_A"], fa_v_B=inp["fa_v_B"],
        fa_o_A=inp["fa_o_A"], fa_o_B=inp["fa_o_B"],
        fa_gate_A=inp["fa_gate_A"], fa_gate_B=inp["fa_gate_B"],
        fa_up_A=inp["fa_up_A"],     fa_up_B=inp["fa_up_B"],
        fa_down_A=inp["fa_down_A"], fa_down_B=inp["fa_down_B"],
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
        ("grad_gate_A", out["grad_gate_A"], ref["grad_gate_A"]),
        ("grad_gate_B", out["grad_gate_B"], ref["grad_gate_B"]),
        ("grad_up_A",  out["grad_up_A"],  ref["grad_up_A"]),
        ("grad_up_B",  out["grad_up_B"],  ref["grad_up_B"]),
        ("grad_down_A", out["grad_down_A"], ref["grad_down_A"]),
        ("grad_down_B", out["grad_down_B"], ref["grad_down_B"]),
    ]
    for label, ours, refg in pairs:
        if not compare(ours, refg, label, max_abs_thresh=1e-1, cos_thresh=0.99):
            fails.append(label)

    print()
    print("=" * 60)
    if not fails:
        print("ALL CHECKS PASSED ✓  per_layer_bwd_fa matches autograd end-to-end.")
        print("Slice B.3b's FA-layer reverse walk is complete.")
    else:
        print(f"FAIL on: {fails}")
        sys.exit(1)


if __name__ == "__main__":
    main()
