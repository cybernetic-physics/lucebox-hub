"""Run each of our backward kernels N times with identical inputs and
verify outputs are bit-identical. Localizes which kernel is racy.

Tests:
  bwd_lora_linear  (5 inner kernels over shared workspaces)
  bwd_swiglu
  bwd_rmsnorm
"""
from __future__ import annotations

import sys
import torch

sys.path.insert(0, "/home/freiza/lucebox-hub/models/qwen35_0p8b/trainer")
import train_megakernel_C  # noqa: F401


def test_bwd_lora_linear():
    print("=== bwd_lora_linear (5 inner kernels, ws_lora_h + ws_grad_lora_h) ===")
    torch.manual_seed(0)
    S, K_in, K_out, R = 30, 1024, 1024, 8
    scaling = 2.0
    x = (torch.randn(S, K_in, device='cuda') * 0.1).to(torch.bfloat16).contiguous()
    A = (torch.randn(K_in, R, device='cuda') * 0.1).to(torch.bfloat16).contiguous()
    B = (torch.randn(R, K_out, device='cuda') * 0.1).to(torch.bfloat16).contiguous()
    grad_y = (torch.randn(S, K_out, device='cuda') * 0.1).to(torch.float32).contiguous()

    runs = []
    for i in range(8):
        grad_x = torch.zeros(S, K_in, dtype=torch.float32, device='cuda')
        grad_A = torch.zeros(K_in, R, dtype=torch.float32, device='cuda')
        grad_B = torch.zeros(R, K_out, dtype=torch.float32, device='cuda')
        ws_lora_h = torch.zeros(S, R, dtype=torch.float32, device='cuda')
        ws_grad_lora_h = torch.zeros(S, R, dtype=torch.float32, device='cuda')
        torch.ops.train_megakernel_C.bwd_lora_linear(
            x, A, B, grad_y,
            grad_x, grad_A, grad_B,
            ws_lora_h, ws_grad_lora_h,
            S, K_in, K_out, R, scaling,
        )
        torch.cuda.synchronize()
        runs.append((grad_x.clone(), grad_A.clone(), grad_B.clone()))
    base = runs[0]
    for i, r in enumerate(runs[1:], 2):
        for name, a, b in zip(("grad_x", "grad_A", "grad_B"), base, r):
            mx = float((a - b).abs().max())
            if mx > 0:
                print(f"  iter {i}: {name} differs by max|Δ|={mx:.4e}")
                break
        else:
            continue
        break
    else:
        print("  PASS — all 8 iters bit-identical")


def test_bwd_swiglu():
    print("=== bwd_swiglu (compares dy chain through silu * up) ===")
    torch.manual_seed(0)
    S, INTER = 30, 3584
    gate = (torch.randn(S, INTER, device='cuda') * 0.1).to(torch.bfloat16).contiguous()
    up = (torch.randn(S, INTER, device='cuda') * 0.1).to(torch.bfloat16).contiguous()
    dy = (torch.randn(S, INTER, device='cuda') * 0.1).to(torch.float32).contiguous()
    runs = []
    for i in range(8):
        dgate = torch.empty_like(gate, dtype=torch.float32)
        dup = torch.empty_like(up, dtype=torch.float32)
        torch.ops.train_megakernel_C.bwd_swiglu(gate, up, dy, dgate, dup, S * INTER)
        torch.cuda.synchronize()
        runs.append((dgate.clone(), dup.clone()))
    base = runs[0]
    for i, r in enumerate(runs[1:], 2):
        for name, a, b in zip(("dgate", "dup"), base, r):
            mx = float((a - b).abs().max())
            if mx > 0:
                print(f"  iter {i}: {name} differs by max|Δ|={mx:.4e}")
                break
        else:
            continue
        break
    else:
        print("  PASS — all 8 iters bit-identical")


def test_bwd_rmsnorm():
    print("=== bwd_rmsnorm ===")
    torch.manual_seed(0)
    S, H = 30, 1024
    x = (torch.randn(S, H, device='cuda') * 0.1).to(torch.bfloat16).contiguous()
    w = (torch.randn(H, device='cuda') * 0.1).to(torch.bfloat16).contiguous()
    dy = (torch.randn(S, H, device='cuda') * 0.1).to(torch.float32).contiguous()
    runs = []
    for i in range(8):
        dx = torch.empty(S, H, dtype=torch.float32, device='cuda')
        torch.ops.train_megakernel_C.bwd_rmsnorm(x, w, dy, dx, S, H, 1e-6)
        torch.cuda.synchronize()
        runs.append(dx.clone())
    base = runs[0]
    for i, r in enumerate(runs[1:], 2):
        mx = float((base - r).abs().max())
        if mx > 0:
            print(f"  iter {i}: dx differs by max|Δ|={mx:.4e}")
            break
    else:
        print("  PASS — all 8 iters bit-identical")


def main():
    test_bwd_lora_linear()
    test_bwd_swiglu()
    test_bwd_rmsnorm()


if __name__ == "__main__":
    main()
