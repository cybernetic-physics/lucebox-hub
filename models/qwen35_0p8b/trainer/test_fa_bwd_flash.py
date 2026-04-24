"""Correctness + speed test for fa_bwd_flash.

Compares (dQ, dK, dV) from `fa_backward_flash` against torch autograd
through `F.scaled_dot_product_attention`. Both should route to the
same cuDNN FA-2 kernel underneath, so agreement should be within bf16
tolerance (machine-level differences in philox seed / reduction order).

Also times our direct backward call vs the autograd path to show the
overhead of wrapping (~50 µs) vs just doing it by hand.
"""
from __future__ import annotations

import math
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")
from fa_bwd_flash import fa_forward_flash, fa_backward_flash


def run_autograd_ref(Q, K, V, dO, scale):
    """Autograd through sdpa with explicit GQA expansion."""
    B, Hq, S, D = Q.shape
    Hk = K.shape[1]
    H_R = Hq // Hk
    Qt = Q.detach().requires_grad_(True)
    Ke = K.detach().repeat_interleave(H_R, dim=1).requires_grad_(True)
    Ve = V.detach().repeat_interleave(H_R, dim=1).requires_grad_(True)
    Op = F.scaled_dot_product_attention(Qt, Ke, Ve, is_causal=True, scale=scale)
    Op.backward(dO)
    dKe = Ke.grad.view(B, Hk, H_R, S, D).sum(dim=2)
    dVe = Ve.grad.view(B, Hk, H_R, S, D).sum(dim=2)
    return Qt.grad, dKe, dVe


def diff_stats(a, b, name):
    af, bf = a.to(torch.float32), b.to(torch.float32)
    mx = (af - bf).abs().max().item()
    cos = F.cosine_similarity(af.flatten(), bf.flatten(), dim=0).item()
    print(f"  {name}: max|Δ|={mx:.4e}  cos={cos:.6f}")
    return cos


def main():
    torch.manual_seed(0)
    B, Hq, Hk, D = 1, 8, 2, 256
    scale = 1.0 / math.sqrt(D)

    for S in (128, 512, 2048):
        dev = "cuda"
        Q  = (torch.randn(B, Hq, S, D, dtype=torch.float32, device=dev) * 0.5).to(torch.bfloat16)
        K  = (torch.randn(B, Hk, S, D, dtype=torch.float32, device=dev) * 0.5).to(torch.bfloat16)
        V  = (torch.randn(B, Hk, S, D, dtype=torch.float32, device=dev) * 0.5).to(torch.bfloat16)
        dO = (torch.randn(B, Hq, S, D, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16)

        # Reference: autograd through sdpa (uses same cuDNN FA-2 under the hood).
        dQ_ref, dK_ref, dV_ref = run_autograd_ref(Q, K, V, dO, scale)

        # Ours: call forward once to get O+LSE, then backward.
        fwd = fa_forward_flash(Q, K, V, is_causal=True, scale=scale)
        # Need K,V with Hq heads for backward (matches forward's expansion).
        K_e = K.repeat_interleave(Hq // Hk, dim=1)
        V_e = V.repeat_interleave(Hq // Hk, dim=1)
        dQ, dK, dV = fa_backward_flash(
            dO, Q, K_e, V_e, fwd.O, fwd.LSE,
            is_causal=True, scale=scale,
            philox_seed=fwd.philox_seed, philox_offset=fwd.philox_offset,
            num_kv_heads=Hk,
        )

        print(f"--- S={S} ---")
        cq = diff_stats(dQ, dQ_ref, "dQ")
        ck = diff_stats(dK, dK_ref, "dK")
        cv = diff_stats(dV, dV_ref, "dV")
        passed = all(c > 0.99 for c in (cq, ck, cv))
        print(f"  -> {'PASS' if passed else 'FAIL'} (all cos > 0.99)")

    # Quick wall-time comparison at a representative training shape.
    S = 512
    Q  = (torch.randn(B, Hq, S, D, dtype=torch.float32, device="cuda") * 0.5).to(torch.bfloat16)
    K  = (torch.randn(B, Hk, S, D, dtype=torch.float32, device="cuda") * 0.5).to(torch.bfloat16)
    V  = (torch.randn(B, Hk, S, D, dtype=torch.float32, device="cuda") * 0.5).to(torch.bfloat16)
    dO = (torch.randn(B, Hq, S, D, dtype=torch.float32, device="cuda") * 0.1).to(torch.bfloat16)

    def run_ours():
        fwd = fa_forward_flash(Q, K, V, is_causal=True, scale=scale)
        K_e = K.repeat_interleave(Hq // Hk, dim=1)
        V_e = V.repeat_interleave(Hq // Hk, dim=1)
        fa_backward_flash(dO, Q, K_e, V_e, fwd.O, fwd.LSE,
                          is_causal=True, scale=scale,
                          philox_seed=fwd.philox_seed, philox_offset=fwd.philox_offset,
                          num_kv_heads=Hk)

    def run_autograd():
        run_autograd_ref(Q, K, V, dO, scale)

    for fn, name in [(run_ours, "ours (fwd+bwd)"), (run_autograd, "autograd sdpa")]:
        for _ in range(3): fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50): fn()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 20.0
        print(f"  {name:<18}: {ms:.3f} ms/iter at S={S}")


if __name__ == "__main__":
    main()
