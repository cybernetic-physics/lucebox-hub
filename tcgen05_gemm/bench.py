"""Walltime: tcgen05 GEMM vs torch (cuBLAS) on the same 128×256×K problem."""

import time
import torch
import tcgen05_gemm_C  # noqa: F401

M, N, K = 1024, 4096, 1024


def bench_tcgen05(A, B, n_warmup=100, n_iter=1000):
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    for _ in range(n_warmup):
        torch.ops.tcgen05_gemm_C.tcgen05_gemm_one_tile(A, B, C)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        torch.ops.tcgen05_gemm_C.tcgen05_gemm_one_tile(A, B, C)
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) * 1e6 / n_iter
    return us


def bench_cublas(A, B, n_warmup=100, n_iter=1000):
    for _ in range(n_warmup):
        C = A.float() @ B.float().T
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        C = A.float() @ B.float().T
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) * 1e6 / n_iter
    return us


def bench_cublas_bf16_native(A, B, n_warmup=100, n_iter=1000):
    """cuBLAS natively bf16. Most apples-to-apples."""
    B_t = B.T.contiguous()  # pre-transpose for apples-to-apples (tcgen05 takes B row-major [N, K])
    for _ in range(n_warmup):
        C = torch.matmul(A, B_t)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        C = torch.matmul(A, B_t)
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) * 1e6 / n_iter
    return us


def main():
    torch.manual_seed(0)
    A = (torch.randn(M, K) * 0.1).to(device="cuda", dtype=torch.bfloat16)
    B = (torch.randn(N, K) * 0.1).to(device="cuda", dtype=torch.bfloat16)

    tc = bench_tcgen05(A, B)
    cb_bf16 = bench_cublas_bf16_native(A, B)
    cb_fp32 = bench_cublas(A, B)

    print(f"Shape M={M} N={N} K={K}")
    print(f"  tcgen05 (bf16 → fp32)      : {tc:8.2f} μs")
    print(f"  torch.matmul bf16 (cuBLAS) : {cb_bf16:8.2f} μs   ratio tcgen05/cublas = {tc/cb_bf16:.2f}x")
    print(f"  torch.matmul fp32 (cuBLAS) : {cb_fp32:8.2f} μs   ratio tcgen05/cublas = {tc/cb_fp32:.2f}x")
    if cb_bf16 > tc:
        print(f"  ✓ tcgen05 is {cb_bf16/tc:.2f}× faster than cuBLAS bf16")


if __name__ == "__main__":
    main()
