"""Probe K-step 1 in isolation. A[:, 16] = 1, B[j, 16] = j/1024, else 0.
Expected C[i, j] = j/1024 if K-step 1 reads right addresses; wrong values otherwise.
"""

import sys
import torch
import tcgen05_gemm_C  # noqa: F401

M, N, K = 128, 256, 64

A = torch.zeros(M, K, dtype=torch.bfloat16, device="cuda")
A[:, 16] = 1.0
B = torch.zeros(N, K, dtype=torch.bfloat16, device="cuda")
B[:, 16] = (torch.arange(N, dtype=torch.float32, device="cuda") / 1024.0).to(torch.bfloat16)
C = torch.zeros(M, N, dtype=torch.float32, device="cuda")

torch.ops.tcgen05_gemm_C.tcgen05_gemm_one_tile(A, B, C)
torch.cuda.synchronize()

print("Testing K-step 1 (K=16..31) in isolation.")
print("Expected C[i, j] = j/1024 if K-step 1 works correctly.")
print()
row = 0
n_identity = 0
n_wrong = 0
n_zero = 0
for c in range(32):
    v = C[row, c].item()
    decoded = round(v * 1024.0)
    status = "ok" if decoded == c else ("ZERO" if abs(v) < 1e-5 else f"MISMATCH → col {decoded}")
    if decoded == c: n_identity += 1
    elif abs(v) < 1e-5: n_zero += 1
    else: n_wrong += 1
    print(f"  C[0, {c:3d}] = {v:.5f}  decoded = {decoded:4d}  {status}")

print()
print(f"identity: {n_identity}  zero: {n_zero}  mismatch: {n_wrong}")
