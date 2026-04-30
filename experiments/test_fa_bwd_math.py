"""Cross-check the math fa_backward_flash against torch autograd."""
from __future__ import annotations

import math
import sys
import torch

sys.path.insert(0, "/home/freiza/lucebox-hub/models/qwen35_0p8b/trainer")

from fa_bwd_flash import fa_backward_flash, fa_forward_flash


def main():
    torch.manual_seed(0)
    B, Hq, Hk, S, D = 1, 8, 2, 30, 256
    Q = (torch.randn(B, Hq, S, D, device='cuda') * 0.1).to(torch.bfloat16).contiguous()
    K = (torch.randn(B, Hk, S, D, device='cuda') * 0.1).to(torch.bfloat16).contiguous()
    V = (torch.randn(B, Hk, S, D, device='cuda') * 0.1).to(torch.bfloat16).contiguous()
    dO = (torch.randn(B, Hq, S, D, device='cuda') * 0.05).to(torch.bfloat16).contiguous()
    scale = 1.0 / math.sqrt(D)

    # Reference: build attention via torch (autograd).
    Q_ref = Q.float().clone().requires_grad_(True)
    K_ref = K.float().clone().requires_grad_(True)
    V_ref = V.float().clone().requires_grad_(True)
    K_ref_e = K_ref.repeat_interleave(Hq // Hk, dim=1)
    V_ref_e = V_ref.repeat_interleave(Hq // Hk, dim=1)

    scores = torch.einsum("bhid,bhjd->bhij", Q_ref, K_ref_e) * scale
    causal = torch.triu(torch.full((S, S), float("-inf"), device='cuda'), diagonal=1)
    scores = scores + causal
    P = torch.softmax(scores, dim=-1)
    O_ref = torch.einsum("bhij,bhjd->bhid", P, V_ref_e)
    O_ref.backward(dO.float())

    dQ_ref = Q_ref.grad
    dK_ref = K_ref.grad
    dV_ref = V_ref.grad

    # Run fwd to get LSE.
    K_e = K.repeat_interleave(Hq // Hk, dim=1).contiguous()
    V_e = V.repeat_interleave(Hq // Hk, dim=1).contiguous()
    fwd = fa_forward_flash(Q, K_e, V_e, is_causal=True, scale=scale)

    dQ, dK, dV = fa_backward_flash(
        dO, Q, K_e, V_e, fwd.O, fwd.LSE,
        is_causal=True, scale=scale, num_kv_heads=Hk,
    )

    def cmp(name, a, b):
        a, b = a.float(), b.float()
        cos = float(torch.dot(a.flatten(), b.flatten()) /
                    (a.norm() * b.norm() + 1e-12))
        rel = float((a - b).norm() / (a.norm() + 1e-12))
        mx = float((a - b).abs().max())
        print(f"  {name:6s}  cos={cos:.6f}  rel L2={rel*100:.3f}%  max|Δ|={mx:.4e}")

    print("Math FA bwd vs torch autograd:")
    cmp("dQ", dQ.float(), dQ_ref)
    cmp("dK", dK.float(), dK_ref)
    cmp("dV", dV.float(), dV_ref)


if __name__ == "__main__":
    main()
