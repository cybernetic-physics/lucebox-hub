"""Validate the analytical chunked backward (dn_chunked_bwd_proto) against
torch autograd through the chunked forward prototype.

If this passes, we have a verified algorithmic reference for the CUDA
chunked backward port.
"""
from __future__ import annotations

import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")
from dn_chunked_proto import chunked_fwd
from dn_chunked_bwd_proto import chunked_fwd_with_state_chunks, chunked_bwd


def diff(name, a, b):
    af, bf = a.to(torch.float32).flatten(), b.to(torch.float32).flatten()
    mx = (af - bf).abs().max().item()
    cos = F.cosine_similarity(af, bf, dim=0).item()
    print(f"  {name:<14} max|Δ|={mx:.4e}  cos={cos:.6f}")
    return cos


def main():
    torch.manual_seed(0)
    H, Dk, Dv = 16, 128, 128
    for S in (64, 128, 512):
        dev = "cuda"
        q  = (torch.randn(S, H, Dk, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16).contiguous()
        k  = (torch.randn(S, H, Dk, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16).contiguous()
        v  = (torch.randn(S, H, Dv, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16).contiguous()
        beta = torch.sigmoid(torch.randn(S, H, dtype=torch.float32, device=dev)).contiguous()
        g    = (-torch.rand(S, H, dtype=torch.float32, device=dev) * 0.05).contiguous()
        s0   = torch.zeros(H, Dk, Dv, dtype=torch.float32, device=dev).contiguous()
        dy   = (torch.randn(S, H, Dv, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16).contiguous()

        # ----- torch autograd reference -----
        q1, k1, v1, b1, g1, s01 = [t.detach().clone().requires_grad_(True)
                                   for t in (q, k, v, beta, g, s0)]
        y_auto, _ = chunked_fwd(q1, k1, v1, b1, g1, s01)
        (y_auto * dy.to(y_auto.dtype)).sum().backward()

        # ----- analytical chunked bwd -----
        y_proto, _, state_chunks = chunked_fwd_with_state_chunks(q, k, v, beta, g, s0)
        bwd = chunked_bwd(q, k, v, beta, g, s0, dy, state_chunks)

        print(f"--- S={S} ---")
        cy  = diff("y",     y_proto, y_auto)
        cq  = diff("dq",    bwd.dq,  q1.grad)
        ck  = diff("dk",    bwd.dk,  k1.grad)
        cv  = diff("dv",    bwd.dv,  v1.grad)
        cb  = diff("dbeta", bwd.dbeta, b1.grad)
        cg  = diff("dg",    bwd.dg,  g1.grad)
        cs  = diff("ds0",   bwd.dstate_init, s01.grad)
        passed = all(c > 0.99 for c in (cy, cq, ck, cv, cb, cg, cs))
        print(f"  -> {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
