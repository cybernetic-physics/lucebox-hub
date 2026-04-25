"""Validate that torch autograd through the chunked Python prototype gives
the same gradients as the recurrent CUDA bwd. Since the recurrent kernel
is bit-exact to the torch reference, matching it confirms the chunked
forward is differentiable end-to-end with the right semantics."""
from __future__ import annotations

import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")
import train_megakernel_C  # noqa: F401
from dn_chunked_proto import chunked_fwd


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
        decay = g.exp().contiguous()
        s0   = torch.zeros(H, Dk, Dv, dtype=torch.float32, device=dev).contiguous()
        dy   = (torch.randn(S, H, Dv, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16).contiguous()

        # ----- chunked Python with autograd -----
        q1 = q.detach().clone().requires_grad_(True)
        k1 = k.detach().clone().requires_grad_(True)
        v1 = v.detach().clone().requires_grad_(True)
        b1 = beta.detach().clone().requires_grad_(True)
        g1 = g.detach().clone().requires_grad_(True)
        s01 = s0.detach().clone().requires_grad_(True)
        y_proto, _ = chunked_fwd(q1, k1, v1, b1, g1, s01)
        (y_proto * dy.to(y_proto.dtype)).sum().backward()

        # ----- recurrent CUDA fwd+bwd -----
        y_rec = torch.empty(S, H, Dv, dtype=torch.bfloat16, device=dev)
        state_rec = torch.empty(H, Dk, Dv, dtype=torch.float32, device=dev)
        delta_save = torch.empty(S, H, Dv, dtype=torch.bfloat16, device=dev)
        state_history = torch.empty(H, S + 1, Dk, Dv, dtype=torch.float32, device=dev)
        torch.ops.train_megakernel_C.dn_fwd_save(
            q, k, v, beta, decay, s0, y_rec, state_rec, delta_save, state_history)

        dq = torch.empty_like(q); dk_ = torch.empty_like(k); dv_ = torch.empty_like(v)
        dbeta = torch.empty_like(beta); ddecay = torch.empty_like(decay)
        ds0 = torch.empty_like(s0)
        torch.ops.train_megakernel_C.dn_bwd(
            q, k, v, beta, decay, s0, delta_save, dy, state_history,
            dq, dk_, dv_, dbeta, ddecay, ds0)

        # The recurrent kernel reports ddecay; the chunked path differentiates
        # w.r.t. g (log-decay). Per-step: decay = exp(g) so dg = ddecay * decay.
        # Convert ddecay -> dg for comparison.
        dg_from_decay = ddecay * decay

        print(f"--- S={S} ---")
        cy  = diff("y",     y_proto, y_rec)
        cq  = diff("dq",    q1.grad,  dq)
        ck  = diff("dk",    k1.grad,  dk_)
        cv  = diff("dv",    v1.grad,  dv_)
        cb  = diff("dbeta", b1.grad,  dbeta)
        cd  = diff("dg",    g1.grad,  dg_from_decay)
        cs  = diff("ds0",   s01.grad, ds0)
        passed = all(c > 0.99 for c in (cy, cq, ck, cv, cb, cd, cs))
        print(f"  -> {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
