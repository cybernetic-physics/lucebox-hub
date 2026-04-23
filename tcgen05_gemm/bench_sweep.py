"""Sweep M/N/K to find where we are vs cuBLAS and identify the perf wall."""
import time
import subprocess
import torch
import tcgen05_gemm_C  # noqa: F401

# The kernel has compile-time M, N, K. Just measure at current config for now.
# Change config in kernel.cu + rebuild to sweep.

from bench import M, N, K

def bench_tcgen05(A, B, n_iter=1000):
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    for _ in range(50):
        torch.ops.tcgen05_gemm_C.tcgen05_gemm_one_tile(A, B, C)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        torch.ops.tcgen05_gemm_C.tcgen05_gemm_one_tile(A, B, C)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / n_iter


def bench_cublas(A, B, n_iter=1000):
    B_t = B.T.contiguous()
    for _ in range(50):
        C = torch.matmul(A, B_t)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        C = torch.matmul(A, B_t)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / n_iter


def main():
    torch.manual_seed(0)
    A = (torch.randn(M, K) * 0.1).to(device="cuda", dtype=torch.bfloat16)
    B = (torch.randn(N, K) * 0.1).to(device="cuda", dtype=torch.bfloat16)

    flops = 2 * M * N * K
    tc_us = bench_tcgen05(A, B)
    cb_us = bench_cublas(A, B)
    tc_tflops = flops / (tc_us * 1e-6) / 1e12
    cb_tflops = flops / (cb_us * 1e-6) / 1e12

    print(f"M={M} N={N} K={K}  FLOPs={flops/1e9:.2f} GFLOPs")
    print(f"  tcgen05:    {tc_us:7.2f} μs   {tc_tflops:7.1f} TFLOPs  ({tc_tflops/2000*100:.1f}% of B200 bf16 peak)")
    print(f"  cuBLAS:     {cb_us:7.2f} μs   {cb_tflops:7.1f} TFLOPs  ({cb_tflops/2000*100:.1f}% of peak)")
    print(f"  ratio:      {tc_us/cb_us:.2f}× slower")


if __name__ == "__main__":
    main()
