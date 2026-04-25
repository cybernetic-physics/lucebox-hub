"""Correctness + speed test for the dn_fwd_save / dn_bwd CUDA kernels.

Compares against dn_backward_autograd (the torch reference in dn_bwd.py)
on the actual Qwen3.5-0.8B DeltaNet shapes (H=16, Dk=128, Dv=128).
"""
from __future__ import annotations

import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")
import train_megakernel_C  # noqa: F401
from dn_bwd import dn_forward_recurrence, dn_backward_autograd


def diff(name, a, b):
    af, bf = a.to(torch.float32), b.to(torch.float32)
    mx = (af - bf).abs().max().item()
    cos = F.cosine_similarity(af.flatten(), bf.flatten(), dim=0).item()
    print(f"  {name:<14} max|Δ|={mx:.4e}  cos={cos:.6f}")
    return cos


def main():
    torch.manual_seed(0)
    H, Dk, Dv = 16, 128, 128

    for S in (32, 128, 256, 512):
        dev = "cuda"
        q = (torch.randn(S, H, Dk, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16).contiguous()
        k = (torch.randn(S, H, Dk, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16).contiguous()
        v = (torch.randn(S, H, Dv, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16).contiguous()
        beta  = torch.sigmoid(torch.randn(S, H, dtype=torch.float32, device=dev)).contiguous()
        decay = torch.sigmoid(torch.randn(S, H, dtype=torch.float32, device=dev)).contiguous()
        state_init = torch.zeros(H, Dk, Dv, dtype=torch.float32, device=dev).contiguous()
        dy = (torch.randn(S, H, Dv, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16).contiguous()

        # ----- run our CUDA forward -----
        y_cuda = torch.empty(S, H, Dv, dtype=torch.bfloat16, device=dev)
        state_out = torch.empty(H, Dk, Dv, dtype=torch.float32, device=dev)
        delta_save = torch.empty(S, H, Dv, dtype=torch.bfloat16, device=dev)
        state_history = torch.empty(H, S + 1, Dk, Dv, dtype=torch.float32, device=dev)
        torch.ops.train_megakernel_C.dn_fwd_save(
            q, k, v, beta, decay, state_init, y_cuda, state_out, delta_save, state_history)
        torch.cuda.synchronize()

        # Reference forward.
        y_ref, _ = dn_forward_recurrence(q, k, v, beta, decay, state_init)

        # ----- run our CUDA backward -----
        dq = torch.empty_like(q)
        dk_ = torch.empty_like(k)
        dv = torch.empty_like(v)
        dbeta = torch.empty_like(beta)
        ddecay = torch.empty_like(decay)
        dstate_init = torch.empty_like(state_init)
        torch.ops.train_megakernel_C.dn_bwd(
            q, k, v, beta, decay, state_init, delta_save, dy, state_history,
            dq, dk_, dv, dbeta, ddecay, dstate_init)
        torch.cuda.synchronize()

        # Reference backward.
        ref = dn_backward_autograd(q, k, v, beta, decay, state_init, dy)

        print(f"--- S={S} ---")
        cy = diff("y (fwd)",     y_cuda, y_ref)
        cq = diff("dq",          dq,     ref.dq)
        ck = diff("dk",          dk_,    ref.dk)
        cv = diff("dv",          dv,     ref.dv)
        cb = diff("dbeta",       dbeta,  ref.dbeta)
        cd = diff("ddecay",      ddecay, ref.ddecay)
        cs = diff("dstate_init", dstate_init, ref.dstate_init)
        passed = all(c > 0.99 for c in (cy, cq, ck, cv, cb, cd, cs))
        print(f"  -> {'PASS' if passed else 'FAIL'} (all cos > 0.99)")

    # ----- speed @ S=512 -----
    S = 512
    dev = "cuda"
    q = (torch.randn(S, H, Dk, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16).contiguous()
    k = (torch.randn(S, H, Dk, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16).contiguous()
    v = (torch.randn(S, H, Dv, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16).contiguous()
    beta  = torch.sigmoid(torch.randn(S, H, dtype=torch.float32, device=dev)).contiguous()
    decay = torch.sigmoid(torch.randn(S, H, dtype=torch.float32, device=dev)).contiguous()
    state_init = torch.zeros(H, Dk, Dv, dtype=torch.float32, device=dev).contiguous()
    dy = (torch.randn(S, H, Dv, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16).contiguous()
    delta_save = torch.empty(S, H, Dv, dtype=torch.bfloat16, device=dev)
    state_out = torch.empty(H, Dk, Dv, dtype=torch.float32, device=dev)
    y_cuda = torch.empty(S, H, Dv, dtype=torch.bfloat16, device=dev)
    dq = torch.empty_like(q); dk_ = torch.empty_like(k); dv = torch.empty_like(v)
    dbeta = torch.empty_like(beta); ddecay = torch.empty_like(decay)
    dstate_init = torch.empty_like(state_init)

    def run_cuda_bwd():
        state_history = torch.empty(H, S + 1, Dk, Dv, dtype=torch.float32, device=dev)
        torch.ops.train_megakernel_C.dn_fwd_save(
            q, k, v, beta, decay, state_init, y_cuda, state_out, delta_save, state_history)
        torch.ops.train_megakernel_C.dn_bwd(
            q, k, v, beta, decay, state_init, delta_save, dy, state_history,
            dq, dk_, dv, dbeta, ddecay, dstate_init)

    for _ in range(3): run_cuda_bwd()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10): run_cuda_bwd()
    torch.cuda.synchronize()
    cuda_ms = (time.perf_counter() - t0) * 100.0
    print(f"\nSpeed @ S=512:")
    print(f"  CUDA dn fwd+bwd (one DN layer): {cuda_ms:.2f} ms/iter")
    print(f"  vs torch ref (~275 ms/iter)   = {275.0/cuda_ms:.1f}x speedup")


if __name__ == "__main__":
    main()
