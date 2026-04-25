"""Head-to-head kernel benchmark: our chunked CUDA fwd vs fla's
chunk_gated_delta_rule (Triton). Same inputs, same shapes, isolated.
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")
import _fla_torch_compile_shim  # noqa: F401
import torch

import train_megakernel_C  # noqa: F401
from fla.ops.gated_delta_rule import chunk_gated_delta_rule


def time_fn(fn, runs=10, warm=3):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / runs


def main():
    torch.manual_seed(0)
    H, Dk, Dv = 16, 128, 128
    print(f"{'S':>6} | {'fla fwd ms':>11} {'ours fwd ms':>12} {'ratio':>8}")
    print("-" * 50)
    for S in (128, 256, 512, 1024, 2048, 4096):
        dev = "cuda"
        # Inputs in fla's expected layout: [B, S, H, D]
        q = (torch.randn(1, S, H, Dk, dtype=torch.float32, device=dev)*0.1).to(torch.bfloat16)
        k = (torch.randn(1, S, H, Dk, dtype=torch.float32, device=dev)*0.1).to(torch.bfloat16)
        v = (torch.randn(1, S, H, Dv, dtype=torch.float32, device=dev)*0.1).to(torch.bfloat16)
        beta = torch.sigmoid(torch.randn(1, S, H, dtype=torch.float32, device=dev))
        g = -torch.rand(1, S, H, dtype=torch.float32, device=dev)*0.05

        # Our CUDA kernel takes [S, H, D]
        q_ours = q.squeeze(0).contiguous()
        k_ours = k.squeeze(0).contiguous()
        v_ours = v.squeeze(0).contiguous()
        beta_ours = beta.squeeze(0).contiguous()
        g_ours = g.squeeze(0).contiguous()
        s0 = torch.zeros(H, Dk, Dv, dtype=torch.float32, device=dev).contiguous()
        y_ours = torch.empty(S, H, Dv, dtype=torch.bfloat16, device=dev)
        state_out = torch.empty(H, Dk, Dv, dtype=torch.float32, device=dev)
        empty = torch.empty(0, dtype=torch.float32, device=dev)

        def fla_fn():
            chunk_gated_delta_rule(q, k, v, g=g, beta=beta,
                                   use_qk_l2norm_in_kernel=True,
                                   output_final_state=False)

        def ours_fn():
            torch.ops.train_megakernel_C.dn_chunked_fwd(
                q_ours, k_ours, v_ours, beta_ours, g_ours, s0,
                y_ours, state_out, empty)

        fla_ms = time_fn(fla_fn)
        ours_ms = time_fn(ours_fn)
        ratio = fla_ms / ours_ms
        tag = "ours" if ratio > 1 else "fla"
        print(f"{S:>6} | {fla_ms:>11.3f} {ours_ms:>12.3f} {ratio:>7.2f}x  ({tag} faster)")


if __name__ == "__main__":
    main()
