"""Correctness test: dn_chunked_3090 vs the existing trainer chunked
forward (which is bit-exact to HF torch_chunk_gated_delta_rule).

Trainer chunked has B200-only smem layout — won't run on RTX 3090.
So we compare against fla.chunk_gated_delta_rule, which fits the same
math (matches HF reference) and runs on Ampere via Triton.
"""
from __future__ import annotations

import sys
import torch

sys.path.insert(0, "/home/freiza/lucebox-hub/models/qwen35_0p8b")
import qwen35_megakernel_bf16_C  # noqa: F401

from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # type: ignore


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def run_one(S: int, H: int = 16, Dk: int = 128, Dv: int = 128, seed: int = 0):
    torch.manual_seed(seed)
    dev = torch.device("cuda")

    # Generate inputs in fla's expected layout: q/k [B, S, H, Dk] bf16, v
    # [B, S, H, Dv] bf16, beta/g [B, S, H] fp32. Use B=1 then squeeze.
    q = (torch.randn(1, S, H, Dk, device=dev) * 0.05).to(torch.bfloat16)
    k = (torch.randn(1, S, H, Dk, device=dev) * 0.05).to(torch.bfloat16)
    v = (torch.randn(1, S, H, Dv, device=dev) * 0.05).to(torch.bfloat16)
    beta = torch.rand(1, S, H, device=dev, dtype=torch.float32)
    g = -torch.rand(1, S, H, device=dev, dtype=torch.float32) * 0.1
    state0 = torch.zeros(1, H, Dk, Dv, device=dev, dtype=torch.float32)

    # Reference: fla.
    y_ref, sN_ref = chunk_gated_delta_rule(
        q.contiguous(), k.contiguous(), v.contiguous(),
        g=g.contiguous(), beta=beta.contiguous(),
        initial_state=state0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )
    y_ref = y_ref.float()  # [1, S, H, Dv]
    sN_ref = sN_ref.float()  # [1, H, Dk, Dv]

    # Ours: drop the batch dim, build the inputs the kernel expects.
    # Our kernel matches HF's torch_chunk_gated_delta_rule (which expects
    # pre-scaled q); fla.chunk_gated_delta_rule applies the 1/sqrt(Dk)
    # scale internally. Pre-scale q for the kernel to match fla's output.
    q_scaled = (q.float() * (Dk ** -0.5)).to(torch.bfloat16)
    q2 = q_scaled.squeeze(0).contiguous()
    k2 = k.squeeze(0).contiguous()
    v2 = v.squeeze(0).contiguous()
    beta2 = beta.squeeze(0).contiguous()  # [S, H] fp32
    g2 = g.squeeze(0).contiguous()
    state0_2 = state0.squeeze(0).contiguous()  # [H, Dk, Dv]

    y_ours = torch.empty(S, H, Dv, device=dev, dtype=torch.bfloat16)
    sN_ours = torch.empty(H, Dk, Dv, device=dev, dtype=torch.float32)

    torch.ops.qwen35_megakernel_bf16_C.dn_chunked_3090(
        q2, k2, v2, beta2, g2, state0_2, y_ours, sN_ours
    )
    torch.cuda.synchronize()

    # Compare.
    y_ref_2 = y_ref.squeeze(0)
    sN_ref_2 = sN_ref.squeeze(0)

    y_cos = cos(y_ours, y_ref_2)
    sN_cos = cos(sN_ours, sN_ref_2)
    y_max = float((y_ours.float() - y_ref_2).abs().max())
    sN_max = float((sN_ours - sN_ref_2).abs().max())
    # Per-chunk slice cos: first 32 tokens, first 32 v columns of head 0.
    y_chunk0_slice = y_ours[:32, 0, :32]
    y_ref_chunk0_slice = y_ref_2[:32, 0, :32]
    chunk0_cos = cos(y_chunk0_slice, y_ref_chunk0_slice)
    print(f"S={S:6d}  y cos={y_cos:.6f}  max|Δ|={y_max:.4e}   "
          f"sN cos={sN_cos:.6f}  max|Δ|={sN_max:.4e}   "
          f"y[0:32,0,0:32] cos={chunk0_cos:.6f}")
    return y_cos, sN_cos


def main():
    print(f"{'S':>6}  {'y cos':>9}  {'y maxΔ':>10}  {'sN cos':>9}  {'sN maxΔ':>10}")
    print("-" * 55)
    for S in [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        run_one(S)


if __name__ == "__main__":
    main()
