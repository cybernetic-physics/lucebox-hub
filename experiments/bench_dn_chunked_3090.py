"""Time the 3090-tuned chunked DN forward vs fla.chunk_gated_delta_rule.

Both are inference paths; this measures only the DN forward, not the
full prefill. Use to decide the routing threshold in prefill.cu.
"""
from __future__ import annotations

import sys
import time
import torch

sys.path.insert(0, "/home/freiza/lucebox-hub/models/qwen35_0p8b")
import qwen35_megakernel_bf16_C  # noqa: F401

from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # type: ignore


def time_ours(q, k, v, beta, g, state0, runs=3, warm=1):
    H, Dk, Dv = q.size(1), q.size(2), v.size(2)
    S = q.size(0)
    y = torch.empty(S, H, Dv, device='cuda', dtype=torch.bfloat16)
    sN = torch.empty(H, Dk, Dv, device='cuda', dtype=torch.float32)
    op = torch.ops.qwen35_megakernel_bf16_C.dn_chunked_3090
    for _ in range(warm):
        op(q, k, v, beta, g, state0, y, sN)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        op(q, k, v, beta, g, state0, y, sN)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / runs


def time_fla(q4, k4, v4, beta4, g4, state0_4, runs=3, warm=1):
    for _ in range(warm):
        _ = chunk_gated_delta_rule(q4, k4, v4, g=g4, beta=beta4,
                                    initial_state=state0_4,
                                    output_final_state=True,
                                    use_qk_l2norm_in_kernel=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = chunk_gated_delta_rule(q4, k4, v4, g=g4, beta=beta4,
                                    initial_state=state0_4,
                                    output_final_state=True,
                                    use_qk_l2norm_in_kernel=False)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / runs


def main():
    H, Dk, Dv = 16, 128, 128
    print(f"{'S':>6} | {'fla ms':>10} {'ours ms':>10} {'speedup':>8}")
    print("-" * 40)
    for S in [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
        torch.manual_seed(0)
        dev = torch.device('cuda')
        q = (torch.randn(S, H, Dk, device=dev) * 0.05).to(torch.bfloat16)
        k = (torch.randn(S, H, Dk, device=dev) * 0.05).to(torch.bfloat16)
        v = (torch.randn(S, H, Dv, device=dev) * 0.05).to(torch.bfloat16)
        beta = torch.rand(S, H, device=dev, dtype=torch.float32)
        g = -torch.rand(S, H, device=dev, dtype=torch.float32) * 0.1
        state0 = torch.zeros(H, Dk, Dv, device=dev, dtype=torch.float32)

        # 4D variants for fla.
        q4 = q.unsqueeze(0).contiguous()
        k4 = k.unsqueeze(0).contiguous()
        v4 = v.unsqueeze(0).contiguous()
        beta4 = beta.unsqueeze(0).contiguous()
        g4 = g.unsqueeze(0).contiguous()
        state0_4 = state0.unsqueeze(0).contiguous()

        # Pre-scaled q for our kernel (HF/torch_chunk convention).
        q_scaled = (q.float() * (Dk ** -0.5)).to(torch.bfloat16).contiguous()

        try:
            fla_ms = time_fla(q4, k4, v4, beta4, g4, state0_4)
        except Exception as e:
            fla_ms = float('nan')
        try:
            our_ms = time_ours(q_scaled, k, v, beta, g, state0)
        except Exception as e:
            our_ms = float('nan')
        sp = fla_ms / our_ms if our_ms == our_ms and our_ms > 0 else float('nan')
        print(f"{S:>6} | {fla_ms:>10.2f} {our_ms:>10.2f} {sp:>7.2f}x")


if __name__ == "__main__":
    main()
