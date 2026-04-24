"""Correctness + speed test for cutlass_gemm_bf16 on sm_100.

Runs a bf16 matmul at Qwen3.5-0.8B projection shapes via
torch.ops.cutlass_train_C.cutlass_gemm_bf16 and compares to
torch.matmul on the same inputs. Reports max-abs-diff and a
quick speed comparison vs cuBLAS.
"""
from __future__ import annotations

import sys
import time

import torch

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer/cutlass_train")
import cutlass_train_C  # noqa: F401


def run_ours(A: torch.Tensor, B: torch.Tensor, alpha=1.0, beta=0.0) -> torch.Tensor:
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    torch.ops.cutlass_train_C.cutlass_gemm_bf16(A, B, C, float(alpha), float(beta))
    return C


def bench(fn, warmup=3, runs=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / runs


def main():
    torch.manual_seed(0)

    # Qwen3.5-0.8B projection shapes (at S=512, rank 16, all row-major bf16).
    # (M, N, K) = (S, Nout, Nin) for each projection.
    shapes = [
        ("FA q (HIDDEN→FA_QPROJ)",   512, 4096, 1024),
        ("FA o (FA_Q_SIZE→HIDDEN)",  512, 1024, 2048),
        ("MLP gate (H→I)",            512, 3584, 1024),
        ("MLP down (I→H)",            512, 1024, 3584),
        ("DN qkv (H→DN_CONV_CH)",    512, 6144, 1024),
    ]

    print(f"{'shape':<30} | {'M, N, K':<18} | {'max |Δ|':>10} | {'cuBLAS ms':>10} | {'CUTLASS ms':>11}")
    print("-" * 30 + "-+-" + "-" * 18 + "-+-" + "-" * 10 + "-+-" + "-" * 10 + "-+-" + "-" * 11)

    for name, M, N, K in shapes:
        A = torch.randn(M, K, device="cuda", dtype=torch.float32).to(torch.bfloat16)
        B = torch.randn(K, N, device="cuda", dtype=torch.float32).to(torch.bfloat16)

        C_ref = torch.matmul(A, B)                   # cuBLAS via torch
        C_ours = run_ours(A, B, 1.0, 0.0)

        diff = (C_ref.to(torch.float32) - C_ours.to(torch.float32)).abs().max().item()

        cu_ms = bench(lambda: torch.matmul(A, B))
        ct_ms = bench(lambda: run_ours(A, B, 1.0, 0.0))

        print(f"{name:<30} | {M},{N},{K:<14} | {diff:>10.4f} | {cu_ms:>8.3f}  | {ct_ms:>9.3f}")


if __name__ == "__main__":
    main()
