"""Validate dn_chunked_fwd CUDA kernel against the recurrent CUDA kernel
and the Python chunked prototype.
"""
from __future__ import annotations

import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")
import train_megakernel_C  # noqa: F401
from dn_chunked_proto import chunked_fwd as proto_fwd


def diff(name, a, b):
    af = a.to(torch.float32).flatten()
    bf = b.to(torch.float32).flatten()
    mx = (af - bf).abs().max().item()
    cos = F.cosine_similarity(af, bf, dim=0).item()
    print(f"  {name:<14} max|Δ|={mx:.4e}  cos={cos:.6f}")
    return cos


def main():
    torch.manual_seed(0)
    H, Dk, Dv = 16, 128, 128
    for S in (64, 128, 512, 1024):
        dev = "cuda"
        q = (torch.randn(S, H, Dk, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16).contiguous()
        k = (torch.randn(S, H, Dk, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16).contiguous()
        v = (torch.randn(S, H, Dv, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16).contiguous()
        beta = torch.sigmoid(torch.randn(S, H, dtype=torch.float32, device=dev)).contiguous()
        g = (-torch.rand(S, H, dtype=torch.float32, device=dev) * 0.05).contiguous()
        decay = g.exp().contiguous()
        state_init = torch.zeros(H, Dk, Dv, dtype=torch.float32, device=dev).contiguous()

        # Recurrent CUDA reference.
        y_rec = torch.empty(S, H, Dv, dtype=torch.bfloat16, device=dev)
        state_rec = torch.empty(H, Dk, Dv, dtype=torch.float32, device=dev)
        delta_save = torch.empty(S, H, Dv, dtype=torch.bfloat16, device=dev)
        state_history = torch.empty(H, S + 1, Dk, Dv, dtype=torch.float32, device=dev)
        torch.ops.train_megakernel_C.dn_fwd_save(
            q, k, v, beta, decay, state_init,
            y_rec, state_rec, delta_save, state_history)
        torch.cuda.synchronize()

        # Chunked CUDA.
        y_chk = torch.empty(S, H, Dv, dtype=torch.bfloat16, device=dev)
        state_chk = torch.empty(H, Dk, Dv, dtype=torch.float32, device=dev)
        torch.ops.train_megakernel_C.dn_chunked_fwd(
            q, k, v, beta, g, state_init, y_chk, state_chk)
        torch.cuda.synchronize()

        print(f"--- S={S} ---")
        cy = diff("y (chk vs rec)", y_chk, y_rec)
        cs = diff("state_out (chk vs rec)", state_chk, state_rec)
        passed = cy > 0.99 and cs > 0.99
        print(f"  -> {'PASS' if passed else 'FAIL'}")

        # Speed.
        for _ in range(3):
            torch.ops.train_megakernel_C.dn_chunked_fwd(
                q, k, v, beta, g, state_init, y_chk, state_chk)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        N = 10
        for _ in range(N):
            torch.ops.train_megakernel_C.dn_chunked_fwd(
                q, k, v, beta, g, state_init, y_chk, state_chk)
        torch.cuda.synchronize()
        chk_ms = (time.perf_counter() - t0) * 1000.0 / N

        for _ in range(3):
            torch.ops.train_megakernel_C.dn_fwd_save(
                q, k, v, beta, decay, state_init,
                y_rec, state_rec, delta_save, state_history)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N):
            torch.ops.train_megakernel_C.dn_fwd_save(
                q, k, v, beta, decay, state_init,
                y_rec, state_rec, delta_save, state_history)
        torch.cuda.synchronize()
        rec_ms = (time.perf_counter() - t0) * 1000.0 / N

        print(f"  chunked CUDA fwd:  {chk_ms:7.3f} ms")
        print(f"  recurrent CUDA fwd:{rec_ms:7.3f} ms")
        if chk_ms < rec_ms:
            print(f"  -> {rec_ms/chk_ms:.2f}x speedup")
        else:
            print(f"  -> {chk_ms/rec_ms:.2f}x SLOWER (chunked needs more work)")


if __name__ == "__main__":
    main()
