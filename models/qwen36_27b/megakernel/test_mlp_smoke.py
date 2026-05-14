"""Smoke test for the templated MLP kernel.

Loads the qwen3x_C extension, runs the MLP kernel for both Cfg_0p8B
(HIDDEN=1024, INTER=3584) and Cfg_27B (HIDDEN=5120, INTER=17408)
specializations, compares each against a reference PyTorch
implementation. Validates that:

  - both Cfg specializations compile and execute on GB10 sm_121a
  - the bf16 MLP path produces output close to fp32 PyTorch reference
    (within bf16 noise: ~5% relative error per element)

This is the Phase-1 proof-of-concept that the template-on-Cfg approach
works. The full decode kernel (DeltaNet with V/QK split, full
attention, layer walker) is the follow-on work.

Run:
    cd models/qwen36_27b/megakernel
    /home/sparkz/rl/.venv/bin/python3 setup.py build_ext --inplace
    /home/sparkz/rl/.venv/bin/python3 test_mlp_smoke.py
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qwen3x_C  # noqa: F401  registers torch.ops.qwen3x_C

ops = torch.ops.qwen3x_C


def reference_mlp(input_bf16: torch.Tensor, gain_bf16: torch.Tensor,
                  w_gate_bf16: torch.Tensor, w_up_bf16: torch.Tensor,
                  w_down_bf16: torch.Tensor) -> torch.Tensor:
    """Reference: RMSNorm with (1 + gain), gate/up SwiGLU, down + residual."""
    x = input_bf16.to(torch.float32)
    rstd = (x.pow(2).mean() + 1e-6).rsqrt()
    normed = (x * rstd * (1.0 + gain_bf16.to(torch.float32))).to(torch.bfloat16)
    gate = (w_gate_bf16.to(torch.float32) @ normed.to(torch.float32))
    up   = (w_up_bf16.to(torch.float32)   @ normed.to(torch.float32))
    inter = (F.silu(gate) * up).to(torch.bfloat16)
    down = (w_down_bf16.to(torch.float32) @ inter.to(torch.float32))
    out = (x + down).to(torch.bfloat16)
    return out


def run_one(name: str, hidden: int, intermediate: int, op):
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    f32  = dict(dtype=torch.float32,  device="cuda")
    g = torch.Generator(device="cuda").manual_seed(hidden + intermediate)
    input    = (torch.randn(hidden, generator=g, **bf16) * 0.5).contiguous()
    gain     = (torch.randn(hidden, generator=g, **bf16) * 0.1).contiguous()
    # Smaller weight init so values stay in bf16 dynamic range.
    w_gate   = (torch.randn(intermediate, hidden, generator=g, **bf16) * 0.02).contiguous()
    w_up     = (torch.randn(intermediate, hidden, generator=g, **bf16) * 0.02).contiguous()
    w_down   = (torch.randn(hidden, intermediate, generator=g, **bf16) * 0.02).contiguous()
    sh_norm  = torch.zeros(hidden, **bf16)
    g_gate   = torch.zeros(intermediate, **f32)
    g_up     = torch.zeros(intermediate, **f32)
    sh_inter = torch.zeros(intermediate, **bf16)
    out      = torch.zeros(hidden, **bf16)

    # Warmup + time.
    op(input, gain, w_gate, w_up, w_down, sh_norm, g_gate, g_up, sh_inter, out)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    N = 20
    for _ in range(N):
        op(input, gain, w_gate, w_up, w_down, sh_norm, g_gate, g_up, sh_inter, out)
    torch.cuda.synchronize()
    dt_us = (time.perf_counter() - t0) * 1e6 / N

    ref = reference_mlp(input, gain, w_gate, w_up, w_down)
    err = (out.to(torch.float32) - ref.to(torch.float32)).abs()
    cos = F.cosine_similarity(out.to(torch.float32).unsqueeze(0),
                              ref.to(torch.float32).unsqueeze(0), dim=-1).item()
    max_abs = err.max().item()
    scale   = ref.to(torch.float32).abs().mean().item()
    rel = max_abs / max(scale, 1e-6)
    finite = bool(torch.isfinite(out.to(torch.float32)).all().item())
    print(f"  [{name:>5}]  H={hidden:>5}  I={intermediate:>5}  "
          f"{dt_us:>7.1f} us  cos={cos:.6f}  rel_max={rel:.3f}  finite={finite}")
    return cos > 0.99 and rel < 0.10 and finite


def main():
    print(f"Device: {torch.cuda.get_device_name()}  cap={torch.cuda.get_device_capability()}")
    print()
    print("Templated MLP smoke test — Cfg_0p8B + Cfg_27B specializations:")
    ok_0p8b = run_one("0p8B", 1024, 3584,  ops.mlp_smoke_0p8b)
    ok_27b  = run_one("27B",  5120, 17408, ops.mlp_smoke_27b)
    print()
    if ok_0p8b and ok_27b:
        print("BOTH Cfg SPECIALIZATIONS PASS — template-on-Cfg approach validated")
    else:
        print(f"FAILURE: 0p8B={ok_0p8b}  27B={ok_27b}")
        sys.exit(1)


if __name__ == "__main__":
    main()
