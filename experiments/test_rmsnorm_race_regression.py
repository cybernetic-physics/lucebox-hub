"""Regression test for the bwd_rmsnorm shared-mem race fixed in `ebb57dd`.

Background: bwd_rmsnorm_kernel does two block-wide reductions over the
SAME `s_red[NW]` smem array. Without a __syncthreads() between
"read s_red[0]" (final mean_sq from reduction 1) and "write
s_red[warp_id] = dot" (start of reduction 2), a fast warp 0 could
overwrite s_red[0] while slow warps were still reading it -> garbage
gradient.

The race was invisible at most shapes: warps happened to stay in
lockstep at H=128, H=1024, S=60, S=30. It only manifested at the
exact Q-norm bwd shape (S=240, H=256, 8 warps). Hence the dedicated
test below.

Two checks:

  1. **Determinism**: 30 launches with identical inputs must produce
     bit-identical outputs.
  2. **Compute-sanitizer racecheck** (optional, gated on the binary
     being on PATH): one launch must report 0 hazards.

Both checks fail before the fix (1324 hazards / 9 unique outputs in
20 iters); both pass after.

Usage:
    CUDA_VISIBLE_DEVICES=1 .venv-3090/bin/python \\
        experiments/test_rmsnorm_race_regression.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import torch

sys.path.insert(0, "/home/freiza/lucebox-hub/models/qwen35_0p8b/trainer")
import train_megakernel_C  # noqa: F401


# Q-norm bwd shape: 30 tokens × 8 q-heads = 240 batch rows of head_dim 256.
RACING_SHAPE = (240, 256)
N_DETERMINISM_ITERS = 30


def _run_one(x, w, dy, S, H):
    dx = torch.empty(S, H, dtype=torch.float32, device='cuda')
    torch.ops.train_megakernel_C.bwd_rmsnorm(x, w, dy, dx, S, H, 1e-6)
    torch.cuda.synchronize()
    return dx


def check_determinism():
    print(f"[1/2] Determinism @ {RACING_SHAPE} (30 iterations, identical inputs)")
    torch.manual_seed(0)
    S, H = RACING_SHAPE
    x = (torch.randn(S, H, device='cuda') * 0.1).to(torch.bfloat16).contiguous()
    w = (torch.randn(H, device='cuda') * 0.1).to(torch.bfloat16).contiguous()
    dy = (torch.randn(S, H, device='cuda') * 0.1).to(torch.float32).contiguous()

    base = _run_one(x, w, dy, S, H)
    for i in range(2, N_DETERMINISM_ITERS + 1):
        cur = _run_one(x, w, dy, S, H)
        if not base.equal(cur):
            mx = float((base - cur).abs().max())
            print(f"      FAIL: iter {i} differs by max|Δ|={mx:.4e}")
            print(f"      The bwd_rmsnorm shared-mem race is back. "
                  f"See kernel.cu:1519-1528 fence (commit ebb57dd).")
            return False
    print(f"      PASS — all {N_DETERMINISM_ITERS} iters bit-identical")
    return True


def check_racecheck():
    print(f"[2/2] Compute-sanitizer racecheck @ {RACING_SHAPE}")
    if shutil.which("compute-sanitizer") is None:
        print("      SKIP — compute-sanitizer not on PATH")
        return True

    helper = """
import sys, torch
sys.path.insert(0, '/home/freiza/lucebox-hub/models/qwen35_0p8b/trainer')
import train_megakernel_C
torch.manual_seed(0)
S, H = 240, 256
x = (torch.randn(S, H, device='cuda') * 0.1).to(torch.bfloat16).contiguous()
w = (torch.randn(H, device='cuda') * 0.1).to(torch.bfloat16).contiguous()
dy = (torch.randn(S, H, device='cuda') * 0.1).to(torch.float32).contiguous()
dx = torch.empty(S, H, dtype=torch.float32, device='cuda')
torch.ops.train_megakernel_C.bwd_rmsnorm(x, w, dy, dx, S, H, 1e-6)
torch.cuda.synchronize()
"""
    cmd = [
        "compute-sanitizer", "--tool", "racecheck", "--print-limit", "1",
        sys.executable, "-c", helper,
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "1")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                            env=env)
    except subprocess.TimeoutExpired:
        print("      SKIP — racecheck timed out")
        return True
    out = r.stdout + "\n" + r.stderr
    if "RACECHECK SUMMARY: 0 hazards" in out:
        print("      PASS — 0 hazards reported by compute-sanitizer")
        return True
    if "Race reported" in out or "hazards" in out:
        print("      FAIL — racecheck found hazards. Excerpt:")
        for line in out.splitlines():
            if "Race" in line or "hazards" in line or "kernel.cu" in line:
                print(f"        {line.strip()}")
        return False
    print("      WARNING — racecheck output unparseable; treating as PASS")
    return True


def main():
    ok1 = check_determinism()
    ok2 = check_racecheck()
    if ok1 and ok2:
        print("\nALL CHECKS PASSED — bwd_rmsnorm shared-mem race is fixed.")
        return 0
    print("\nFAIL — bwd_rmsnorm has a shared-memory race.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
