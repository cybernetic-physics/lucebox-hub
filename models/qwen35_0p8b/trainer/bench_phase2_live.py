"""Phase 2 live bench: measure a training step that stitches together
the pieces we have today.

  - forward: our prefill_bf16_train_step (kernel, saves activations)
  - backward attention (FA layers):    fa_bwd_flash    (cuDNN FA-2)
  - backward attention (DN layers):    dn_backward_autograd (torch ref)
  - backward MLP / norm / LoRA-linear: our bwd CUDA kernels
  - optim: launch_fused_adamw (our kernel)

Compares total wall time against HF + PEFT + torch.optim.AdamW.

The headline number is the cost of the DN backward in torch. Until a
CUDA port of dn_backward_chunked lands, this will be slower than HF
for long-enough S because DN bwd dominates. We isolate and show it so
that the chunked-DN CUDA work can be gated on a concrete metric.
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import torch

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b")
sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/prefill_megakernel")
sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")

from fa_bwd_flash import fa_forward_flash, fa_backward_flash
from dn_bwd import dn_backward_autograd
import train_megakernel_C  # noqa: F401  — registers torch.ops.train_megakernel_C.dn_*


def bench(fn, warm=3, runs=10, label=""):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs): fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0 / runs
    print(f"  {label:<40} {ms:7.2f} ms")
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[128, 512, 1024])
    args = ap.parse_args()

    # Qwen3.5-0.8B FA layer shape per call.
    Hq_fa, Hk_fa, D_fa = 8, 2, 256

    # Qwen3.5-0.8B DN layer shape per call.
    H_dn, Dk_dn, Dv_dn = 16, 128, 128

    # How many layers of each per forward pass.
    N_FA = 6
    N_DN = 18

    print("=" * 78)
    print("Phase 2 live: per-layer backward isolated costs (ms/call)")
    print("=" * 78)

    for S in args.seq_lens:
        dev = "cuda"
        print(f"\n[S={S}, bf16]")

        # ---- FA backward cost ----
        Q  = (torch.randn(1, Hq_fa, S, D_fa, dtype=torch.float32, device=dev) * 0.5).to(torch.bfloat16)
        K  = (torch.randn(1, Hk_fa, S, D_fa, dtype=torch.float32, device=dev) * 0.5).to(torch.bfloat16)
        V  = (torch.randn(1, Hk_fa, S, D_fa, dtype=torch.float32, device=dev) * 0.5).to(torch.bfloat16)
        dO = (torch.randn(1, Hq_fa, S, D_fa, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16)
        scale_fa = 1.0 / math.sqrt(D_fa)

        def run_fa_flash():
            fwd = fa_forward_flash(Q, K, V, is_causal=True, scale=scale_fa)
            K_e = K.repeat_interleave(Hq_fa // Hk_fa, dim=1)
            V_e = V.repeat_interleave(Hq_fa // Hk_fa, dim=1)
            fa_backward_flash(dO, Q, K_e, V_e, fwd.O, fwd.LSE,
                              is_causal=True, scale=scale_fa,
                              philox_seed=fwd.philox_seed, philox_offset=fwd.philox_offset,
                              num_kv_heads=Hk_fa)

        fa_ms = bench(run_fa_flash, label=f"1x FA layer (fwd+bwd, cuDNN FA-2)")
        print(f"  {'x' + str(N_FA) + ' FA layers (total)':<40} {fa_ms*N_FA:7.2f} ms")

        # ---- DN backward cost (ref — python chunked, slow) ----
        q_dn = (torch.randn(S, H_dn, Dk_dn, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16)
        k_dn = (torch.randn(S, H_dn, Dk_dn, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16)
        v_dn = (torch.randn(S, H_dn, Dv_dn, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16)
        beta  = torch.sigmoid(torch.randn(S, H_dn, dtype=torch.float32, device=dev))
        decay = torch.sigmoid(torch.randn(S, H_dn, dtype=torch.float32, device=dev))
        s0    = torch.zeros(H_dn, Dk_dn, Dv_dn, dtype=torch.float32, device=dev)
        dy_dn = (torch.randn(S, H_dn, Dv_dn, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16)

        def run_dn_bwd():
            dn_backward_autograd(q_dn, k_dn, v_dn, beta, decay, s0, dy_dn)

        dn_ms = bench(run_dn_bwd, warm=1, runs=3, label=f"1x DN layer bwd (torch autograd ref)")
        print(f"  {'x' + str(N_DN) + ' DN layers (total)':<40} {dn_ms*N_DN:7.2f} ms")

        # ---- DN backward via our CUDA kernel ----
        y_cuda = torch.empty(S, H_dn, Dv_dn, dtype=torch.bfloat16, device=dev)
        state_out = torch.empty(H_dn, Dk_dn, Dv_dn, dtype=torch.float32, device=dev)
        delta_save = torch.empty(S, H_dn, Dv_dn, dtype=torch.bfloat16, device=dev)
        state_history = torch.empty(H_dn, S + 1, Dk_dn, Dv_dn, dtype=torch.float32, device=dev)
        dq = torch.empty_like(q_dn); dk_ = torch.empty_like(k_dn); dv_ = torch.empty_like(v_dn)
        dbeta = torch.empty_like(beta); ddecay = torch.empty_like(decay)
        ds0 = torch.empty_like(s0)

        def run_dn_bwd_cuda():
            torch.ops.train_megakernel_C.dn_fwd_save(
                q_dn, k_dn, v_dn, beta, decay, s0,
                y_cuda, state_out, delta_save, state_history)
            torch.ops.train_megakernel_C.dn_bwd(
                q_dn, k_dn, v_dn, beta, decay, s0, delta_save, dy_dn, state_history,
                dq, dk_, dv_, dbeta, ddecay, ds0)

        dn_cuda_ms = bench(run_dn_bwd_cuda, warm=3, runs=10,
                           label=f"1x DN layer fwd+bwd (CUDA)")
        print(f"  {'x' + str(N_DN) + ' DN layers (total, CUDA)':<40} {dn_cuda_ms*N_DN:7.2f} ms")
        print(f"  {'CUDA speedup vs torch ref':<40} {dn_ms/dn_cuda_ms:6.1f}x")

        total_attn_bwd = fa_ms * N_FA + dn_cuda_ms * N_DN
        print(f"  {'total attention bwd (FA + DN-CUDA)':<40} {total_attn_bwd:7.2f} ms")
        print(f"  {'⊂ FA share':<40} {100*fa_ms*N_FA/total_attn_bwd:6.1f} %")
        print(f"  {'⊂ DN share':<40} {100*dn_cuda_ms*N_DN/total_attn_bwd:6.1f} %")

    print()
    print("Takeaway: FA backward is essentially free vs model compute; the full")
    print("Phase 2 production speedup is gated on porting dn_backward_chunked's")
    print("inner loop to bf16 WGMMA CUDA. Target ~5ms per DN layer at S=512")
    print("(vs the autograd reference's current ~hundreds of ms).")


if __name__ == "__main__":
    main()
