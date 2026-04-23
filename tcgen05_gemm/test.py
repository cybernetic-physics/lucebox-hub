"""Smoke + correctness test for the tcgen05 bf16 GEMM."""

import sys
import torch
import tcgen05_gemm_C  # noqa: F401

M, N, K = 128, 4096, 256


def main():
    torch.manual_seed(0)
    A = (torch.randn(M, K) * 0.1).to(device="cuda", dtype=torch.bfloat16)
    B = (torch.randn(N, K) * 0.1).to(device="cuda", dtype=torch.bfloat16)
    C = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    torch.ops.tcgen05_gemm_C.tcgen05_gemm_one_tile(A, B, C)
    torch.cuda.synchronize()

    # Full K reference
    ref = A.float() @ B.float().T
    diff = (C - ref).abs()
    print(f"C mean: {C.mean().item():.4f}  ref mean: {ref.mean().item():.4f}")
    print(f"max |Δ|: {diff.max().item():.2e}  mean |Δ|: {diff.mean().item():.2e}")
    print(f"C[0, :8]  = {C[0, :8].tolist()}")
    print(f"ref[0,:8] = {ref[0, :8].tolist()}")
    print()
    print("Check column 0 across rows 0..7 (lane 0's 8 regs if layout=reg-is-row):")
    print(f"C[:8, 0]  = {C[:8, 0].tolist()}")
    print(f"ref[:8,0] = {ref[:8, 0].tolist()}")
    print()
    print("Check row 0 cols 0..7 across lanes 0..7:")
    # If layout is lane=col within each 8-col stripe, then lane 0..7's reg 0 covers row 0
    # (under my current code path)
    # Diagnose: where does C[0,1] actually live in ref?
    target = C[0, 1].item()
    print(f"\nC[0, 1] = {target:.6f}")
    diffs = (ref[0, :].cpu() - target).abs()
    best_j = int(diffs.argmin())
    print(f"closest ref[0, j] is at j={best_j} with value {ref[0, best_j].item():.6f} (|Δ|={diffs[best_j].item():.2e})")
    # Same for C[0, 2..7]
    for i in range(2, 8):
        target = C[0, i].item()
        diffs = (ref[0, :].cpu() - target).abs()
        best_j = int(diffs.argmin())
        print(f"C[0, {i}] = {target:.4f} → ref[0, {best_j}] = {ref[0, best_j].item():.4f}  (|Δ|={diffs[best_j].item():.2e})")
    exact = (C - ref).abs() < 1e-3
    hits = exact.nonzero()
    print(f"exact matches count: {hits.shape[0]}")
    # Row-wise match count
    row_matches = exact.sum(dim=1)
    print(f"rows with matches (row:#matches): {[(i, row_matches[i].item()) for i in range(M) if row_matches[i] > 0][:20]}")
    # Col-wise
    col_matches = exact.sum(dim=0)
    matched_cols = [i for i in range(N) if col_matches[i] > 0]
    print(f"first 20 matching cols: {matched_cols[:20]}")
    print(f"col match counts (first 20): {[col_matches[c].item() for c in matched_cols[:20]]}")

    if diff.max().item() > 1e-1:
        print("CORRECTNESS FAIL")
        sys.exit(1)
    print("CORRECTNESS OK")


if __name__ == "__main__":
    main()
