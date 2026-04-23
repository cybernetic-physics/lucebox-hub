"""Empirical probe: which TMEM (lane, reg) position holds which logical output col.

Construction:
  A = ones(M, K)                bf16
  B[j, k=0] = j / 1024.0,   B[j, k>0] = 0    bf16
  Expected C[i, j] = sum_k A[i,k] * B[j,k] = B[j, 0] = j / 1024

So for every (i, j), the correct value is j / 1024 — recoverable by rounding
C[i, j] * 1024 back to an integer.

If the kernel's readback layout lands TMEM col c's value at output col c
correctly, decoded(c) == c for all c. Otherwise decoded(c) tells us which
logical col TMEM col c actually holds → the permutation table.
"""

import sys
import torch
import tcgen05_gemm_C  # noqa: F401

M, N, K = 128, 256, 64

A = torch.ones(M, K, dtype=torch.bfloat16, device="cuda")
B = torch.zeros(N, K, dtype=torch.bfloat16, device="cuda")
B[:, 0] = (torch.arange(N, dtype=torch.float32, device="cuda") / 1024.0).to(torch.bfloat16)
C = torch.zeros(M, N, dtype=torch.float32, device="cuda")

torch.ops.tcgen05_gemm_C.tcgen05_gemm_one_tile(A, B, C)
torch.cuda.synchronize()

# Decode every position of row 0 (all rows should be identical for this probe)
row = 0
out = []
for c in range(N):
    v = C[row, c].item()
    decoded = round(v * 1024.0)
    out.append(decoded)

print("TMEM read position -> logical col held:")
# Show the first 32 positions (one .32x32b.x8 call spans 8 cols per lane,
# so first 8 is particularly informative).
for c in range(min(32, N)):
    marker = "  <-- identity" if out[c] == c else ""
    print(f"  C[0, {c:3d}] = {C[0, c].item():.5f}  decoded logical col = {out[c]:3d}{marker}")

print()
print("First n_off=0 call's 8 regs (cols 0..7):")
for c in range(8):
    print(f"  reg {c}: holds logical col {out[c]}")
print()
print("Second call's n_off=8, regs 0..7:")
for c in range(8):
    print(f"  reg {c}: holds logical col {out[8 + c]}")

print()
# Check sample of higher cols too
print("Spot-check TMEM positions 32, 40, 48, 56, 64, 72, ...:")
for c in [32, 40, 48, 56, 64, 72, 128, 192, 248]:
    if c < N:
        print(f"  C[0, {c:3d}] -> logical col {out[c]}")
