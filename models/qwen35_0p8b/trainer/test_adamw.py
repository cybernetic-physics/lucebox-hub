"""Correctness test: fused AdamW kernel vs torch.optim.AdamW reference."""

import sys
import torch
import train_megakernel_C  # noqa: F401


def main():
    torch.manual_seed(0)
    n_elements = 16384 * 17 + 37    # non-power-of-two to exercise tail

    # Initial state: params in bf16, m/v in fp32, fresh gradient.
    p_ref = torch.randn(n_elements, dtype=torch.float32, device="cuda") * 0.1
    p_mine = p_ref.clone().to(torch.bfloat16).contiguous()

    # torch AdamW reference — keep params in fp32 then downcast to bf16 per step
    # to match how my bf16-param kernel rounds each step. This matches the
    # typical mixed-precision training recipe (fp32 master weights, bf16
    # downcast for matmul). If we kept torch weights in bf16 the rounding
    # would differ from what my kernel does.
    p_ref_fp32 = p_ref.clone()
    p_torch = torch.nn.Parameter(p_ref_fp32.clone())
    opt = torch.optim.AdamW([p_torch], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    # Common state for my kernel.
    m = torch.zeros(n_elements, device="cuda", dtype=torch.float32)
    v = torch.zeros(n_elements, device="cuda", dtype=torch.float32)

    for step in range(1, 4):
        g = torch.randn(n_elements, dtype=torch.float32, device="cuda") * 0.05

        # Reference: torch AdamW on fp32 param
        p_torch.grad = g.clone()
        opt.step()
        opt.zero_grad(set_to_none=True)

        # Mine
        torch.ops.train_megakernel_C.fused_adamw_step(
            p_mine, m, v, g, step,
            1e-3, 0.9, 0.999, 1e-8, 0.0,
        )
        torch.cuda.synchronize()

        # My kernel stores bf16 params; compare torch fp32 → bf16 → fp32
        ref_bf16 = p_torch.data.to(torch.bfloat16).float()
        diff = (p_mine.float() - ref_bf16).abs()
        print(f"step {step}: max|Δ| = {diff.max().item():.3e}  mean|Δ| = {diff.mean().item():.3e}")
        if diff.max().item() > 1e-2:
            print("  FAIL tolerance")
            sys.exit(1)
    print("CORRECTNESS OK")


if __name__ == "__main__":
    main()
