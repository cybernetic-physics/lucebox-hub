"""Shape sweep — find regimes where tcgen05 beats cuBLAS."""
import time
import torch
import tcgen05_gemm_C  # noqa: F401


def bench(M, N, K, n_iter=500):
    A = torch.randn(M, K).cuda().to(torch.bfloat16)
    B = torch.randn(N, K).cuda().to(torch.bfloat16)
    C = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    for _ in range(50):
        torch.ops.tcgen05_gemm_C.tcgen05_gemm_one_tile(A, B, C)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        torch.ops.tcgen05_gemm_C.tcgen05_gemm_one_tile(A, B, C)
    torch.cuda.synchronize()
    tc_us = (time.perf_counter() - t0) * 1e6 / n_iter

    Bt = B.T.contiguous()
    for _ in range(50):
        Cr = torch.matmul(A, Bt)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        Cr = torch.matmul(A, Bt)
    torch.cuda.synchronize()
    cb_us = (time.perf_counter() - t0) * 1e6 / n_iter
    return tc_us, cb_us


def main():
    # tcgen05 requires M % 128 == 0, N % 128 == 0, K % 64 == 0
    shapes = [
        (128, 128, 128),
        (128, 256, 128),
        (256, 256, 256),
        (512, 512, 512),
        (768, 768, 768),
        (1024, 1024, 1024),
        (1536, 1536, 1536),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        # Qwen 3.5-0.8B training-relevant shapes (nearest multiple of 128/64):
        (128, 4096, 1024),   # Q proj at S=128
        (256, 4096, 1024),   # Q proj at S=256
        (128, 3584, 1024),   # MLP gate/up at S=128
        (256, 3584, 1024),   # MLP gate/up at S=256
    ]
    print(f'{"M":>5} {"N":>6} {"K":>5}  {"tcgen (μs)":>11}  {"cuBLAS (μs)":>12}  {"ratio":>7}  winner')
    for M, N, K in shapes:
        if M % 128 != 0 or N % 128 != 0 or K % 64 != 0:
            print(f"  skip {M}x{N}x{K} (alignment)")
            continue
        tc, cb = bench(M, N, K)
        ratio = tc / cb
        winner = "tcgen05 WINS" if tc < cb else "cuBLAS"
        speedup = ""
        if tc < cb:
            speedup = f"  ({cb/tc:.2f}× faster)"
        print(f'{M:>5} {N:>6} {K:>5}  {tc:>9.2f}    {cb:>10.2f}    {ratio:>5.2f}x  {winner}{speedup}')


if __name__ == "__main__":
    main()
