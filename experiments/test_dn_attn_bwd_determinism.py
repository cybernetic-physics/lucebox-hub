"""Test fla.chunk_gated_delta_rule_bwd (used in dn_attn_handrolled) for determinism."""
from __future__ import annotations

import sys
import torch

sys.path.insert(0, "/home/freiza/lucebox-hub/models/qwen35_0p8b/trainer")
import train_megakernel_C  # noqa: F401
from dn_attn_handrolled import dn_attn_forward, dn_attn_backward


def main():
    torch.manual_seed(0)
    B, S, HIDDEN = 1, 30, 1024
    H, Dk, Dv = 16, 128, 128

    # Build minimal valid input + weights for one DN attention block.
    h_in = (torch.randn(B, S, HIDDEN, device='cuda') * 0.05).to(torch.bfloat16).contiguous()
    npa = (torch.randn(B, S, HIDDEN, device='cuda') * 0.05).to(torch.bfloat16).contiguous()
    in_proj_qkv_W = (torch.randn(2 * H * Dk + H * Dv, HIDDEN, device='cuda') * 0.02).to(torch.bfloat16)
    in_proj_z_W   = (torch.randn(H * Dv, HIDDEN, device='cuda') * 0.02).to(torch.bfloat16)
    in_proj_b_W   = (torch.randn(H, HIDDEN, device='cuda') * 0.02).to(torch.bfloat16)
    in_proj_a_W   = (torch.randn(H, HIDDEN, device='cuda') * 0.02).to(torch.bfloat16)
    conv1d_W = (torch.randn(2 * H * Dk + H * Dv, 1, 4, device='cuda') * 0.05).to(torch.bfloat16)
    A_log = (torch.randn(H, device='cuda') * 0.05).to(torch.bfloat16)
    dt_bias = (torch.randn(H, device='cuda') * 0.05).to(torch.bfloat16)
    dn_norm_W = (torch.ones(Dv, device='cuda') + 0.01 * torch.randn(Dv, device='cuda')).to(torch.bfloat16)
    out_proj_W = (torch.randn(HIDDEN, H * Dv, device='cuda') * 0.02).to(torch.bfloat16)
    input_norm_w = (torch.ones(HIDDEN, device='cuda') + 0.01 * torch.randn(HIDDEN, device='cuda')).to(torch.bfloat16)

    # Run forward once to build saves.
    _attn_out, saves = dn_attn_forward(
        h_in,
        input_norm_w=input_norm_w,
        in_proj_qkv_W=in_proj_qkv_W,
        in_proj_z_W=in_proj_z_W,
        in_proj_b_W=in_proj_b_W,
        in_proj_a_W=in_proj_a_W,
        conv1d_W=conv1d_W,
        A_log=A_log, dt_bias=dt_bias,
        dn_norm_W=dn_norm_W,
        out_proj_W=out_proj_W,
        rms_eps=1e-6, layer_norm_eps=1e-6,
        npa_precomputed=npa,
    )
    torch.cuda.synchronize()

    d_attn_out = (torch.randn(B, S, HIDDEN, device='cuda') * 0.05).to(torch.float32).contiguous()

    # Run dn_attn_backward N times and check determinism.
    runs = []
    for i in range(8):
        out = dn_attn_backward(d_attn_out, saves)
        torch.cuda.synchronize()
        # out is a tuple; first element is dh_in [S, HIDDEN]
        runs.append(out[0].clone())

    base = runs[0]
    print("dn_attn_backward determinism (8 iters, identical inputs):")
    for i, r in enumerate(runs[1:], 2):
        mx = float((base - r).abs().max())
        rel = float((base - r).norm() / (base.norm() + 1e-12))
        if mx > 0:
            print(f"  iter {i}: dh differs  max|Δ|={mx:.4e}  rel={rel*100:.4f}%")
        else:
            print(f"  iter {i}: bit-identical")


if __name__ == "__main__":
    main()
