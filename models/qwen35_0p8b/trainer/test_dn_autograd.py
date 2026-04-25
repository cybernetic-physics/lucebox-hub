"""Test that DeltaNetRecurrence (CUDA kernel + autograd) matches the
torch reference dn_forward_recurrence under autograd, end-to-end.

We build a tiny module: y = deltanet_recurrence(q, k, v, beta, decay, s0)
and check dq/dk/dv/dbeta/ddecay/ds0 match against the same call routed
through the torch python recurrence (which builds a real autograd graph).
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")
from dn_autograd import deltanet_recurrence
from dn_bwd import dn_forward_recurrence


def diff(name, a, b):
    af, bf = a.to(torch.float32), b.to(torch.float32)
    mx = (af - bf).abs().max().item()
    cos = F.cosine_similarity(af.flatten(), bf.flatten(), dim=0).item()
    print(f"  {name:<14} max|Δ|={mx:.4e}  cos={cos:.6f}")
    return cos


def main():
    torch.manual_seed(0)
    H, Dk, Dv = 16, 128, 128

    # Correctness vs torch ref at S<=512; parity is bit-exact and the
    # ref is too slow to run above. For S in {1024, 4096, 16384} we just
    # NaN-check the autograd path end-to-end.
    for S in (32, 128, 512, 1024, 4096, 16384):
        dev = "cuda"
        q  = (torch.randn(S, H, Dk, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16).requires_grad_(True)
        k  = (torch.randn(S, H, Dk, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16).requires_grad_(True)
        v  = (torch.randn(S, H, Dv, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16).requires_grad_(True)
        beta  = torch.sigmoid(torch.randn(S, H, dtype=torch.float32, device=dev)).requires_grad_(True)
        decay = torch.sigmoid(torch.randn(S, H, dtype=torch.float32, device=dev)).requires_grad_(True)
        s0    = torch.zeros(H, Dk, Dv, dtype=torch.float32, device=dev, requires_grad=True)
        dy    = (torch.randn(S, H, Dv, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16)

        # ----- CUDA path -----
        q1, k1, v1, b1, d1, s01 = [t.detach().clone().requires_grad_(True) for t in (q, k, v, beta, decay, s0)]
        y_cuda, _ = deltanet_recurrence(q1, k1, v1, b1, d1, s01)
        (y_cuda * dy.to(y_cuda.dtype)).sum().backward()

        if S <= 512:
            # ----- torch ref path -----
            q2, k2, v2, b2, d2, s02 = [t.detach().clone().requires_grad_(True) for t in (q, k, v, beta, decay, s0)]
            y_ref, _ = dn_forward_recurrence(q2, k2, v2, b2, d2, s02)
            (y_ref * dy.to(y_ref.dtype)).sum().backward()

            print(f"--- S={S} ---")
            cy = diff("y",      y_cuda, y_ref)
            cq = diff("dq",     q1.grad,  q2.grad)
            ck = diff("dk",     k1.grad,  k2.grad)
            cv = diff("dv",     v1.grad,  v2.grad)
            cb = diff("dbeta",  b1.grad,  b2.grad)
            cd = diff("ddecay", d1.grad,  d2.grad)
            cs = diff("ds0",    s01.grad, s02.grad)
            passed = all(c > 0.99 for c in (cy, cq, ck, cv, cb, cd, cs))
            print(f"  -> {'PASS' if passed else 'FAIL'} (all cos > 0.99)")
        else:
            # NaN/Inf-check only (ref is too slow above).
            grads = [y_cuda, q1.grad, k1.grad, v1.grad, b1.grad, d1.grad, s01.grad]
            ok = all(torch.isfinite(g).all().item() for g in grads)
            print(f"--- S={S} (no-ref, finite-check) -> {'PASS' if ok else 'FAIL (NaN/Inf)'}")


if __name__ == "__main__":
    main()
