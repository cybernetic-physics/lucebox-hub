"""Validate the cuBLAS path of lora_linear_bwd matches the SIMT path.

The cuBLAS path activates at S >= 512. Run lora_linear_bwd at S=1024
through both paths (with the threshold flipped) and compare grad_x,
grad_A, grad_B element-wise. They must agree to bf16-noise level.
"""
from __future__ import annotations

import sys
import torch

sys.path.insert(0, "/home/freiza/lucebox-hub/models/qwen35_0p8b/trainer")
import train_megakernel_C  # noqa: F401
import lora_layer_bwd_skel as bwd


def main():
    torch.manual_seed(0)
    S, K_in, K_out, R = 1024, 1024, 1024, 8

    x       = (torch.randn(S, K_in, device="cuda") * 0.1).to(torch.bfloat16).contiguous()
    A       = (torch.randn(K_in, R, device="cuda") * 0.1).to(torch.bfloat16).contiguous()
    B       = (torch.randn(R, K_out, device="cuda") * 0.1).to(torch.bfloat16).contiguous()
    base_W  = (torch.randn(K_out, K_in, device="cuda") * 0.05).to(torch.bfloat16).contiguous()
    grad_y  = (torch.randn(S, K_out, device="cuda") * 0.1).to(torch.float32).contiguous()
    scaling = 1.0

    # Path A: cuBLAS (default since S=1024 >= 512)
    gx_c, gA_c, gB_c = bwd.lora_linear_bwd(x, A, B, base_W, grad_y, scaling)

    # Path B: force SIMT by flipping the threshold
    saved = bwd._LORA_BWD_CUBLAS_MIN_S
    bwd._LORA_BWD_CUBLAS_MIN_S = 1 << 30
    try:
        gx_s, gA_s, gB_s = bwd.lora_linear_bwd(x, A, B, base_W, grad_y, scaling)
    finally:
        bwd._LORA_BWD_CUBLAS_MIN_S = saved

    def cmp(name, a, b):
        diff = (a - b).abs()
        max_abs = max(a.abs().max().item(), b.abs().max().item())
        rel_to_max = diff.max().item() / max(max_abs, 1e-8)
        print(f"  {name:>10}  shape={tuple(a.shape)}  max|Δ|={diff.max().item():.4e}  "
              f"mean|Δ|={diff.mean().item():.4e}  max|Δ|/max|val|={rel_to_max:.3%}")
        return rel_to_max

    print("cuBLAS path vs SIMT path (S=1024, K=1024, R=8):")
    r1 = cmp("grad_x", gx_c, gx_s)
    r2 = cmp("grad_A", gA_c, gA_s)
    r3 = cmp("grad_B", gB_c, gB_s)

    # Tolerance: bf16 inputs produce ~1e-3 absolute noise per K-step; over
    # K=1024 steps this can drift to ~1e-2. Compare max|Δ| to max|val| in
    # the tensor — element-wise rel diff is meaningless near zeros. 1%
    # tolerance is comfortably above the bf16 reordering floor.
    ok = max(r1, r2, r3) < 0.01
    if ok:
        print("PASS — cuBLAS path matches SIMT path within bf16 tolerance (<1% of max|val|).")
    else:
        print(f"FAIL — cuBLAS path diverges from SIMT path (worst rel-to-max = "
              f"{max(r1, r2, r3):.3%}).")
        sys.exit(1)


if __name__ == "__main__":
    main()
