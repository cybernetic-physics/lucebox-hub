"""Test our dn_bwd CUDA kernel for determinism across runs."""
from __future__ import annotations

import sys
import torch

sys.path.insert(0, "/home/freiza/lucebox-hub/models/qwen35_0p8b/trainer")
import train_megakernel_C  # noqa: F401


def main():
    torch.manual_seed(0)
    H, S, Dk, Dv = 16, 30, 128, 128
    q = (torch.randn(S, H, Dk, device='cuda') * 0.1).to(torch.bfloat16).contiguous()
    k = (torch.randn(S, H, Dk, device='cuda') * 0.1).to(torch.bfloat16).contiguous()
    v = (torch.randn(S, H, Dv, device='cuda') * 0.1).to(torch.bfloat16).contiguous()
    beta = torch.rand(S, H, device='cuda', dtype=torch.float32)
    g = -torch.rand(S, H, device='cuda', dtype=torch.float32) * 0.1
    decay = torch.exp(g)
    state_in = torch.zeros(H, Dk, Dv, device='cuda', dtype=torch.float32)
    dy = (torch.randn(S, H, Dv, device='cuda') * 0.1).to(torch.bfloat16).contiguous()

    # Run forward first to build saves.
    y = torch.empty(S, H, Dv, device='cuda', dtype=torch.bfloat16)
    state_out = torch.empty(H, Dk, Dv, device='cuda', dtype=torch.float32)
    delta_save = torch.empty(S, H, Dk, device='cuda', dtype=torch.bfloat16)
    state_history = torch.empty(H, S + 1, Dk, Dv, device='cuda', dtype=torch.float32)
    torch.ops.train_megakernel_C.dn_fwd_save(
        q, k, v, beta, decay, state_in,
        y, state_out, delta_save, state_history,
    )
    torch.cuda.synchronize()

    runs = []
    for i in range(8):
        dq = torch.empty(S, H, Dk, device='cuda', dtype=torch.float32)
        dk = torch.empty(S, H, Dk, device='cuda', dtype=torch.float32)
        dv = torch.empty(S, H, Dv, device='cuda', dtype=torch.float32)
        dbeta = torch.empty(S, H, device='cuda', dtype=torch.float32)
        ddecay = torch.empty(S, H, device='cuda', dtype=torch.float32)
        dstate_init = torch.empty(H, Dk, Dv, device='cuda', dtype=torch.float32)
        torch.ops.train_megakernel_C.dn_bwd(
            q, k, v, beta, decay, state_in, delta_save, dy, state_history,
            dq, dk, dv, dbeta, ddecay, dstate_init,
        )
        torch.cuda.synchronize()
        runs.append((dq.clone(), dk.clone(), dv.clone(),
                     dbeta.clone(), ddecay.clone(), dstate_init.clone()))

    base = runs[0]
    names = ("dq", "dk", "dv", "dbeta", "ddecay", "dstate_init")
    failed = False
    for i, r in enumerate(runs[1:], 2):
        for name, a, b in zip(names, base, r):
            mx = float((a - b).abs().max())
            if mx > 0:
                print(f"  iter {i}: {name} differs by max|Δ|={mx:.4e}")
                failed = True
                break
    if not failed:
        print(f"PASS — dn_bwd bit-deterministic across 8 iterations")
        print(f"  |dq|={float(base[0].norm()):.4e}  |dk|={float(base[1].norm()):.4e}  "
              f"|dv|={float(base[2].norm()):.4e}")


if __name__ == "__main__":
    main()
