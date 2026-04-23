"""Correctness: bwd_swiglu + bwd_lora_linear vs torch autograd."""

import sys
import torch
import train_megakernel_C  # noqa: F401


def test_bwd_swiglu():
    torch.manual_seed(0)
    N = 4096 * 8
    gate_fp = torch.randn(N, device="cuda") * 0.7
    up_fp = torch.randn(N, device="cuda") * 0.7
    dy_fp = torch.randn(N, device="cuda") * 0.1

    # torch ref
    g = gate_fp.detach().clone().requires_grad_(True)
    u = up_fp.detach().clone().requires_grad_(True)
    y = torch.nn.functional.silu(g) * u
    y.backward(dy_fp)
    ref_dg = g.grad
    ref_du = u.grad

    # mine (bf16 inputs, fp32 dy and outputs)
    gate_bf = gate_fp.to(torch.bfloat16).contiguous()
    up_bf = up_fp.to(torch.bfloat16).contiguous()
    dy_f = dy_fp.contiguous()
    dgate = torch.zeros(N, device="cuda", dtype=torch.float32)
    dup = torch.zeros(N, device="cuda", dtype=torch.float32)
    torch.ops.train_megakernel_C.bwd_swiglu(gate_bf, up_bf, dy_f, dgate, dup, N)
    torch.cuda.synchronize()

    # NOTE: torch ref uses fp32 inputs to silu; we use bf16 cast to fp32. Expect
    # bf16-level agreement (3-digit absolute tolerance).
    dg_diff = (dgate - ref_dg).abs()
    du_diff = (dup - ref_du).abs()
    print(f"bwd_swiglu  dgate max|Δ|={dg_diff.max().item():.2e}  dup max|Δ|={du_diff.max().item():.2e}")
    ok = dg_diff.max().item() < 5e-2 and du_diff.max().item() < 5e-2
    print("BWD_SWIGLU OK" if ok else "BWD_SWIGLU FAIL")
    return ok


def test_bwd_lora_linear():
    torch.manual_seed(1)
    S, K_in, K_out, R = 32, 1024, 4096, 16
    scaling = 2.0

    # inputs
    x_fp   = torch.randn(S, K_in, device="cuda") * 0.1
    A_fp   = torch.randn(K_in, R, device="cuda") * 0.05
    B_fp   = torch.randn(R, K_out, device="cuda") * 0.05
    gy_fp  = torch.randn(S, K_out, device="cuda") * 0.05

    # torch ref
    x = x_fp.detach().clone().requires_grad_(True)
    A = A_fp.detach().clone().requires_grad_(True)
    B = B_fp.detach().clone().requires_grad_(True)
    # y = x @ Wᵀ + scaling * (x @ A) @ B  (base W term we skip — ref without it)
    lora_h = x @ A
    y = scaling * (lora_h @ B)
    y.backward(gy_fp)
    ref_dx = x.grad
    ref_dA = A.grad
    ref_dB = B.grad

    # mine (inputs bf16)
    x_bf = x_fp.to(torch.bfloat16).contiguous()
    A_bf = A_fp.to(torch.bfloat16).contiguous()
    B_bf = B_fp.to(torch.bfloat16).contiguous()
    gy_f = gy_fp.contiguous()

    dx = torch.zeros(S, K_in, device="cuda", dtype=torch.float32)
    dA = torch.zeros(K_in, R, device="cuda", dtype=torch.float32)
    dB = torch.zeros(R, K_out, device="cuda", dtype=torch.float32)
    ws_lora_h = torch.zeros(S, R, device="cuda", dtype=torch.float32)
    ws_grad_lora_h = torch.zeros(S, R, device="cuda", dtype=torch.float32)

    torch.ops.train_megakernel_C.bwd_lora_linear(
        x_bf, A_bf, B_bf, gy_f,
        dx, dA, dB,
        ws_lora_h, ws_grad_lora_h,
        S, K_in, K_out, R, scaling,
    )
    torch.cuda.synchronize()

    dx_diff = (dx - ref_dx).abs()
    dA_diff = (dA - ref_dA).abs()
    dB_diff = (dB - ref_dB).abs()
    print(f"bwd_lora_linear:")
    print(f"  grad_x max|Δ|={dx_diff.max().item():.3e}  mean|Δ|={dx_diff.mean().item():.3e}")
    print(f"  grad_A max|Δ|={dA_diff.max().item():.3e}  mean|Δ|={dA_diff.mean().item():.3e}")
    print(f"  grad_B max|Δ|={dB_diff.max().item():.3e}  mean|Δ|={dB_diff.mean().item():.3e}")
    tol = 5e-2
    ok = (dx_diff.max().item() < tol and dA_diff.max().item() < tol and dB_diff.max().item() < tol)
    print("BWD_LORA_LINEAR OK" if ok else "BWD_LORA_LINEAR FAIL")
    return ok


def main():
    ok = True
    ok &= test_bwd_swiglu()
    ok &= test_bwd_lora_linear()
    print("ALL OK" if ok else "SOME FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
