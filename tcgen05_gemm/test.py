"""Smoke + correctness test for the tcgen05 bf16 GEMM."""

import sys
import torch
import tcgen05_gemm_C  # noqa: F401

M, N, K = 128, 256, 16


def main():
    torch.manual_seed(0)
    A = (torch.randn(M, K) * 0.1).to(device="cuda", dtype=torch.bfloat16)
    B = (torch.randn(N, K) * 0.1).to(device="cuda", dtype=torch.bfloat16)
    C = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    torch.ops.tcgen05_gemm_C.tcgen05_gemm_one_tile(A, B, C)
    torch.cuda.synchronize()

    # Reference
    ref = (A.float() @ B.float().T)
    diff = (C - ref).abs()
    print(f"C mean: {C.mean().item():.4f}  ref mean: {ref.mean().item():.4f}")
    print(f"max |Δ|: {diff.max().item():.2e}  mean |Δ|: {diff.mean().item():.2e}")
    print(f"C[0, :8]  = {C[0, :8].tolist()}")
    print(f"ref[0,:8] = {ref[0, :8].tolist()}")

    if diff.max().item() > 1e-1:
        print("CORRECTNESS FAIL")
        sys.exit(1)
    print("CORRECTNESS OK")


if __name__ == "__main__":
    main()
