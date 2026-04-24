"""Test cutlass_train_C.fa_bwd — sm_100 FMHA backward on bf16 GQA inputs.

Compares dQ/dK/dV to torch.autograd through F.scaled_dot_product_attention
(which routes to cuDNN FA-2 bwd). A match within bf16 tolerance means
we have a working CUTLASS FMHA backward for our Qwen3.5-0.8B FA layers.
"""
from __future__ import annotations

import math
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer/cutlass_train")
import cutlass_train_C  # noqa: F401


def run_torch_sdpa(Q, K, V, dO, scale):
    """Reference bwd via explicit K/V expansion + sum-reduce across the
    H_R group — sidesteps any GQA grad-path subtlety in torch. Inputs
    are [B, H, S, D]; dK/dV returned are [B, Hk, S, D] matching the
    CUTLASS kernel's expectation."""
    B, Hq, S, D = Q.shape
    Hk = K.shape[1]
    H_R = Hq // Hk
    Qt = Q.detach().requires_grad_(True)
    # Expand K/V to per-Q-head copies ourselves so we control the grad
    # accumulation.
    Ke = K.detach().repeat_interleave(H_R, dim=1).requires_grad_(True)
    Ve = V.detach().repeat_interleave(H_R, dim=1).requires_grad_(True)
    Op = F.scaled_dot_product_attention(
        Qt, Ke, Ve, is_causal=True, scale=scale)
    Op.backward(dO)
    # Sum per-Q-head grads back into per-Hk-head grads.
    dKe = Ke.grad.view(B, Hk, H_R, S, D).sum(dim=2)
    dVe = Ve.grad.view(B, Hk, H_R, S, D).sum(dim=2)
    return Op.detach(), Qt.grad, dKe, dVe


def flash_fwd_with_lse(Q, K, V, scale):
    """Forward via torch's flash-attn low-level op that exposes LSE.
    Returns (O, LSE_fp32) both bit-exact from the fused kernel — no
    math fallback, no GQA broadcast emulation."""
    B, Hq, S, D = Q.shape
    Hk = K.shape[1]
    Ke = K.repeat_interleave(Hq // Hk, dim=1) if Hq != Hk else K
    Ve = V.repeat_interleave(Hq // Hk, dim=1) if Hq != Hk else V
    r = torch.ops.aten._scaled_dot_product_flash_attention(
        Q, Ke, Ve, 0.0, True, False, scale=scale)  # dropout, is_causal, return_debug_mask
    O   = r[0]                              # [B, Hq, S, D] bf16
    LSE = r[1]                              # [B, Hq, S]    fp32
    return O, LSE


def main():
    torch.manual_seed(0)

    # Qwen3.5-0.8B FA: Hq=8, Hk=2, D=256. Try a small S first.
    B  = 1
    S  = 128
    Hq = 8
    Hk = 2
    D  = 256
    scale = 1.0 / math.sqrt(D)

    dev = "cuda"
    # [B, H, S, D] torch sdpa layout — matches CUTLASS example 77.
    Q  = (torch.randn(B, Hq, S, D, dtype=torch.float32, device=dev) * 0.5).to(torch.bfloat16)
    K  = (torch.randn(B, Hk, S, D, dtype=torch.float32, device=dev) * 0.5).to(torch.bfloat16)
    V  = (torch.randn(B, Hk, S, D, dtype=torch.float32, device=dev) * 0.5).to(torch.bfloat16)
    dO = (torch.randn(B, Hq, S, D, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16)

    # Reference backward (for comparing dQ/dK/dV) via autograd.
    _, dQ_ref, dK_ref, dV_ref = run_torch_sdpa(Q, K, V, dO, scale)
    # O and LSE fed INTO cutlass bwd come from the same flash kernel so
    # they're numerically self-consistent (not a math fallback).
    O_ref, LSE = flash_fwd_with_lse(Q, K, V, scale)

    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)

    # Ensure all tensors are contiguous in [B, S, H, D] layout.
    assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()
    assert O_ref.is_contiguous(), "O must be [B, S, H, D] contiguous"
    assert dO.is_contiguous()
    # LSE must be [B, Hq, S] contiguous.
    LSE = LSE.contiguous()
    print(f"Q stride: {Q.stride()}, K stride: {K.stride()}")
    print(f"LSE shape: {tuple(LSE.shape)} stride: {LSE.stride()}")

    torch.ops.cutlass_train_C.fa_bwd(
        Q, K, V, O_ref, dO, LSE, dQ, dK, dV, scale, True)
    torch.cuda.synchronize()

    def stats(a, b, name):
        af = a.to(torch.float32); bf = b.to(torch.float32)
        mx = (af - bf).abs().max().item()
        cos = F.cosine_similarity(af.flatten(), bf.flatten(), dim=0).item()
        print(f"  {name}: max|Δ|={mx:.4e}  cos={cos:.6f}")
        return cos

    print(f"Shapes: B={B}, S={S}, Hq={Hq}, Hk={Hk}, D={D}  (GQA={Hq//Hk})")
    print()
    cQ = stats(dQ, dQ_ref, "dQ")
    cK = stats(dK, dK_ref, "dK")
    cV = stats(dV, dV_ref, "dV")
    print()
    passed = all(c > 0.98 for c in (cQ, cK, cV))
    print(f"{'CUTLASS FMHA bwd PASS' if passed else 'FAIL'} ({'all cos > 0.98' if passed else 'cos below 0.98'})")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
