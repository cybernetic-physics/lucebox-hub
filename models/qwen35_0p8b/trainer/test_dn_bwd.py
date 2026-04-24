"""Correctness test for dn_bwd — chunked vs autograd BPTT should match."""
from __future__ import annotations

import sys
import time

import torch

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")
from dn_bwd import (
    dn_forward_recurrence, dn_backward_autograd, dn_backward_chunked,
)


def diff(name, a, b):
    af, bf = a.to(torch.float32), b.to(torch.float32)
    mx = (af - bf).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        af.flatten(), bf.flatten(), dim=0).item()
    print(f"  {name:<16} max|Δ|={mx:.4e}  cos={cos:.6f}")
    return cos


def main():
    torch.manual_seed(0)
    # Qwen3.5-0.8B DN shape: H=16 heads, Dk=128, Dv=128.
    H, Dk, Dv = 16, 128, 128

    for S in (32, 128, 512):
        dev = "cuda"
        q = (torch.randn(S, H, Dk, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16)
        k = (torch.randn(S, H, Dk, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16)
        v = (torch.randn(S, H, Dv, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16)
        beta  = torch.sigmoid(torch.randn(S, H, dtype=torch.float32, device=dev))
        decay = torch.sigmoid(torch.randn(S, H, dtype=torch.float32, device=dev))
        state_init = torch.zeros(H, Dk, Dv, dtype=torch.float32, device=dev)
        dy = (torch.randn(S, H, Dv, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16)

        # Autograd reference.
        r_auto = dn_backward_autograd(q, k, v, beta, decay, state_init, dy)
        # Chunked.
        r_chunk = dn_backward_chunked(q, k, v, beta, decay, state_init, dy,
                                      chunk_size=32)

        print(f"--- S={S} (chunk_size=32) ---")
        cq  = diff("dq",     r_chunk.dq,     r_auto.dq)
        ck  = diff("dk",     r_chunk.dk,     r_auto.dk)
        cv  = diff("dv",     r_chunk.dv,     r_auto.dv)
        cb  = diff("dbeta",  r_chunk.dbeta,  r_auto.dbeta)
        cd  = diff("ddecay", r_chunk.ddecay, r_auto.ddecay)
        cs  = diff("dstate", r_chunk.dstate_init, r_auto.dstate_init)
        passed = all(c > 0.999 for c in (cq, ck, cv, cb, cd, cs))
        print(f"  -> {'PASS' if passed else 'FAIL'} (all cos > 0.999)")

    # Quick time comparison at S=512 (representative training ctx).
    S = 512
    dev = "cuda"
    q = (torch.randn(S, H, Dk, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16)
    k = (torch.randn(S, H, Dk, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16)
    v = (torch.randn(S, H, Dv, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16)
    beta  = torch.sigmoid(torch.randn(S, H, dtype=torch.float32, device=dev))
    decay = torch.sigmoid(torch.randn(S, H, dtype=torch.float32, device=dev))
    state_init = torch.zeros(H, Dk, Dv, dtype=torch.float32, device=dev)
    dy = (torch.randn(S, H, Dv, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16)

    for chunk in (16, 32, 64, 128):
        for _ in range(2):
            dn_backward_chunked(q, k, v, beta, decay, state_init, dy,
                                chunk_size=chunk)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(3):
            dn_backward_chunked(q, k, v, beta, decay, state_init, dy,
                                chunk_size=chunk)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 3 * 1000
        print(f"  chunk_size={chunk:>3}  S=512: {ms:.2f} ms/iter")


if __name__ == "__main__":
    main()
