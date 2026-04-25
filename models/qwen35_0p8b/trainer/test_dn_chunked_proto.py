"""Validate the chunked-DN Python prototype against the recurrent CUDA
kernel: same forward output (within bf16 noise) for the same inputs.

The recurrent kernel takes per-step decay; the chunked variant takes
log-decay g (so per-step decay = exp(g)). We bridge that.
"""
from __future__ import annotations

import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")
import train_megakernel_C  # noqa: F401
from dn_chunked_proto import chunked_fwd


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
        # Use small negative log-decay so per-step decay is close to (but below) 1.
        g = (-torch.rand(S, H, dtype=torch.float32, device=dev) * 0.05).contiguous()
        decay = g.exp().contiguous()
        state_init = torch.zeros(H, Dk, Dv, dtype=torch.float32, device=dev).contiguous()

        # ----- Recurrent CUDA forward -----
        y_cuda = torch.empty(S, H, Dv, dtype=torch.bfloat16, device=dev)
        state_out = torch.empty(H, Dk, Dv, dtype=torch.float32, device=dev)
        delta_save = torch.empty(S, H, Dv, dtype=torch.bfloat16, device=dev)
        state_history = torch.empty(H, S + 1, Dk, Dv, dtype=torch.float32, device=dev)
        torch.ops.train_megakernel_C.dn_fwd_save(
            q, k, v, beta, decay, state_init,
            y_cuda, state_out, delta_save, state_history)
        torch.cuda.synchronize()

        # ----- Chunked Python prototype -----
        y_chunked, state_chunked = chunked_fwd(q, k, v, beta, g, state_init, chunk_size=64)
        torch.cuda.synchronize()

        print(f"--- S={S} ---")
        cy = diff("y", y_chunked, y_cuda)
        cs = diff("state_out", state_chunked, state_out)
        print(f"  -> {'PASS' if cy > 0.99 and cs > 0.99 else 'FAIL'}")

        # Speed comparison: prototype (calls torch.bmm) vs CUDA recurrent.
        for _ in range(2):
            chunked_fwd(q, k, v, beta, g, state_init, chunk_size=64)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(3):
            chunked_fwd(q, k, v, beta, g, state_init, chunk_size=64)
        torch.cuda.synchronize()
        proto_ms = (time.perf_counter() - t0) * 1000.0 / 3

        for _ in range(3):
            torch.ops.train_megakernel_C.dn_fwd_save(
                q, k, v, beta, decay, state_init,
                y_cuda, state_out, delta_save, state_history)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            torch.ops.train_megakernel_C.dn_fwd_save(
                q, k, v, beta, decay, state_init,
                y_cuda, state_out, delta_save, state_history)
        torch.cuda.synchronize()
        cuda_ms = (time.perf_counter() - t0) * 1000.0 / 5

        print(f"  proto fwd:      {proto_ms:8.2f} ms")
        print(f"  recurrent fwd:  {cuda_ms:8.2f} ms")


if __name__ == "__main__":
    main()
