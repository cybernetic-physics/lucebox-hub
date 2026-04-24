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
    """Forward + autograd backward via cuDNN FA-2. Also returns O and LSE
    we'd need to save during our forward (extracted from autograd internals
    is hard; for test purposes we just return what backward computes).
    """
    Qt = Q.detach().requires_grad_(True)
    Kt = K.detach().requires_grad_(True)
    Vt = V.detach().requires_grad_(True)
    # [B, H, S, D] layout expected by sdpa.
    Qp = Qt.permute(0, 2, 1, 3).contiguous()
    Kp = Kt.permute(0, 2, 1, 3).contiguous()
    Vp = Vt.permute(0, 2, 1, 3).contiguous()
    Op = F.scaled_dot_product_attention(
        Qp, Kp, Vp, is_causal=True, enable_gqa=True, scale=scale)
    # dO in the same layout
    dOp = dO.permute(0, 2, 1, 3).contiguous()
    Op.backward(dOp)
    # Map outputs back to [B, S, H, D] layout.
    O = Op.permute(0, 2, 1, 3).contiguous().detach()
    dQ = Qt.grad
    dK = Kt.grad
    dV = Vt.grad
    return O, dQ, dK, dV


def compute_lse_reference(Q, K, scale):
    """LSE = log-sum-exp over keys for each query row (after scale + causal
    mask). fp32 for stability. Returns [B, Hq, S]."""
    B, S, Hq, D = Q.shape
    Hk = K.shape[2]
    H_R = Hq // Hk
    # Expand K to match Q's head count for simple matmul.
    K_expanded = K.repeat_interleave(H_R, dim=2)   # [B, S, Hq, D]
    Qf = Q.to(torch.float32).permute(0, 2, 1, 3)    # [B, Hq, S, D]
    Kf = K_expanded.to(torch.float32).permute(0, 2, 1, 3)  # [B, Hq, S, D]
    scores = torch.matmul(Qf, Kf.transpose(-1, -2)) * scale   # [B, Hq, S, S]
    # Causal mask
    mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=Q.device),
                      diagonal=1)
    scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    # LSE over last dim
    lse = torch.logsumexp(scores, dim=-1)           # [B, Hq, S]
    return lse


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
    Q  = (torch.randn(B, S, Hq, D, dtype=torch.float32, device=dev) * 0.5).to(torch.bfloat16)
    K  = (torch.randn(B, S, Hk, D, dtype=torch.float32, device=dev) * 0.5).to(torch.bfloat16)
    V  = (torch.randn(B, S, Hk, D, dtype=torch.float32, device=dev) * 0.5).to(torch.bfloat16)
    dO = (torch.randn(B, S, Hq, D, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16)

    O_ref, dQ_ref, dK_ref, dV_ref = run_torch_sdpa(Q, K, V, dO, scale)
    LSE = compute_lse_reference(Q, K, scale)   # [B, Hq, S] fp32

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
