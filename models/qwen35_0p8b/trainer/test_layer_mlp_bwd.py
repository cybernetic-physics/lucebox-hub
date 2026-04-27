"""Validate layer_mlp_bwd against torch autograd.

Builds a synthetic FA-style MLP block (gate + up + SwiGLU + down +
residual + post-attn RMSnorm) with the same shapes as a Qwen3.5-0.8B
layer, runs forward + autograd backward as the reference, then runs
our layer_mlp_bwd over the same activations + LoRA tensors and
compares per-tensor gradients.

This is the first end-to-end validation of Slice B.3b — kernel
helpers (bwd_lora_linear, bwd_swiglu, bwd_rmsnorm) are individually
tested in test_bwd_head/test_bwd_mlp; this file proves they thread
together correctly inside one layer.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b")
sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")

import torch  # noqa: E402

import train_megakernel_C  # noqa: F401, E402

from lora_layer_bwd_skel import layer_mlp_bwd  # noqa: E402

HIDDEN = 1024
INTER  = 3584
RANK   = 8
S      = 16

RMS_EPS = 1e-6
LORA_SCALING = 1.0


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = RMS_EPS):
    """Qwen3-Next RMSNorm: y = x * rsqrt(mean(x²) + eps) * (1 + w)."""
    x_f = x.float()
    rstd = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_f * rstd * (1.0 + w.float())).to(x.dtype)


def silu_mul(gate: torch.Tensor, up: torch.Tensor):
    return torch.nn.functional.silu(gate.float()).to(gate.dtype) * up


def make_inputs(seed=42):
    g = torch.Generator(device="cuda").manual_seed(seed)
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    h_post_attn = torch.randn(S, HIDDEN, generator=g, **bf16) * 0.5
    post_attn_norm_w = torch.randn(HIDDEN, generator=g, **bf16) * 0.1
    gate_W = torch.randn(INTER, HIDDEN, generator=g, **bf16) * 0.02
    up_W = torch.randn(INTER, HIDDEN, generator=g, **bf16) * 0.02
    down_W = torch.randn(HIDDEN, INTER, generator=g, **bf16) * 0.02
    gate_A = torch.randn(HIDDEN, RANK, generator=g, **bf16) * 0.02
    gate_B = torch.randn(RANK, INTER, generator=g, **bf16) * 0.02
    up_A = torch.randn(HIDDEN, RANK, generator=g, **bf16) * 0.02
    up_B = torch.randn(RANK, INTER, generator=g, **bf16) * 0.02
    down_A = torch.randn(INTER, RANK, generator=g, **bf16) * 0.02
    down_B = torch.randn(RANK, HIDDEN, generator=g, **bf16) * 0.02
    return dict(
        h_post_attn=h_post_attn,
        post_attn_norm_w=post_attn_norm_w,
        gate_W=gate_W, up_W=up_W, down_W=down_W,
        gate_A=gate_A, gate_B=gate_B,
        up_A=up_A, up_B=up_B,
        down_A=down_A, down_B=down_B,
    )


def autograd_reference(inputs: dict):
    """Run the MLP block under torch autograd and return:
      - mlp_inter (the silu(gate)*up output)
      - normalized_post_attn (post-rmsnorm input to gate/up)
      - h_out (= h_post_attn + mlp_out)
      - grads on all LoRA tensors and on h_post_attn (the input).
    """
    # Make leaf tensors with requires_grad on the things we want grads for:
    # h_post_attn (input), and the 6 LoRA matrices. Base W frozen.
    h = inputs["h_post_attn"].detach().clone().requires_grad_(True)
    gate_A = inputs["gate_A"].detach().clone().requires_grad_(True)
    gate_B = inputs["gate_B"].detach().clone().requires_grad_(True)
    up_A = inputs["up_A"].detach().clone().requires_grad_(True)
    up_B = inputs["up_B"].detach().clone().requires_grad_(True)
    down_A = inputs["down_A"].detach().clone().requires_grad_(True)
    down_B = inputs["down_B"].detach().clone().requires_grad_(True)

    # Forward.
    npa = rms_norm(h, inputs["post_attn_norm_w"])
    # gate = npa @ gate_W.T + scaling * (npa @ gate_A) @ gate_B
    gate = npa @ inputs["gate_W"].t() + LORA_SCALING * (npa @ gate_A) @ gate_B
    up   = npa @ inputs["up_W"].t()   + LORA_SCALING * (npa @ up_A)   @ up_B
    mlp_inter = silu_mul(gate, up)
    mlp_out = mlp_inter @ inputs["down_W"].t() + LORA_SCALING * (mlp_inter @ down_A) @ down_B
    h_out = h + mlp_out

    # Pick an arbitrary scalar loss = sum(h_out * weight) so gradients are
    # well-defined and reproducible.
    g_for_loss = torch.randn(S, HIDDEN, dtype=torch.bfloat16, device="cuda",
                              generator=torch.Generator(device="cuda").manual_seed(99))
    loss = (h_out.float() * g_for_loss.float()).sum()
    loss.backward()

    # Save mlp_inter and npa (detached) for the kernel-side bwd.
    return {
        "loss": loss.detach(),
        "mlp_inter_save": mlp_inter.detach().to(torch.bfloat16),
        "npa_save": npa.detach().to(torch.bfloat16),
        "dh_out": g_for_loss.float(),  # = d(loss)/d(h_out) since loss = sum(h_out * g)
        "ref_dh_post_attn": h.grad.detach().float(),
        "ref_grad_gate_A": gate_A.grad.detach().float(),
        "ref_grad_gate_B": gate_B.grad.detach().float(),
        "ref_grad_up_A":   up_A.grad.detach().float(),
        "ref_grad_up_B":   up_B.grad.detach().float(),
        "ref_grad_down_A": down_A.grad.detach().float(),
        "ref_grad_down_B": down_B.grad.detach().float(),
    }


def compare(ours: torch.Tensor, ref: torch.Tensor, label: str,
            *, max_abs_thresh: float = 5e-2, cos_thresh: float = 0.99):
    diff = (ours.float() - ref.float()).abs()
    cos = torch.nn.functional.cosine_similarity(
        ours.float().flatten().unsqueeze(0),
        ref.float().flatten().unsqueeze(0), dim=-1).item()
    max_a = diff.max().item()
    rel = diff / (ref.float().abs() + 1e-3)
    rel_max = rel.max().item()
    ok = max_a < max_abs_thresh and cos > cos_thresh
    status = "✓" if ok else "✗"
    print(f"  {status} {label:25s} max|Δ|={max_a:.4e}  cos={cos:.5f}  "
          f"max_rel={rel_max:.3e}  shape={tuple(ours.shape)}")
    return ok


def main():
    print("Building synthetic FA-style MLP block (S=16, HIDDEN=1024, INTER=3584, R=8)...")
    inputs = make_inputs(seed=42)

    print("\nRunning autograd reference...")
    ref = autograd_reference(inputs)

    print("Running our layer_mlp_bwd...")
    out = layer_mlp_bwd(
        h_post_attn=inputs["h_post_attn"],
        normalized_post_attn=ref["npa_save"],
        mlp_inter=ref["mlp_inter_save"],
        dh_out=ref["dh_out"],
        post_attn_norm_w=inputs["post_attn_norm_w"],
        gate_W=inputs["gate_W"], up_W=inputs["up_W"], down_W=inputs["down_W"],
        gate_A=inputs["gate_A"], gate_B=inputs["gate_B"],
        up_A=inputs["up_A"], up_B=inputs["up_B"],
        down_A=inputs["down_A"], down_B=inputs["down_B"],
        lora_scaling=LORA_SCALING,
        rms_eps=RMS_EPS,
    )

    print()
    fails = []
    for label, ours, ref_t, thresh in [
        ("dh_post_attn",  out["dh_post_attn"],  ref["ref_dh_post_attn"],  5e-2),
        ("grad_gate_A",   out["grad_gate_A"],   ref["ref_grad_gate_A"],   1e-1),
        ("grad_gate_B",   out["grad_gate_B"],   ref["ref_grad_gate_B"],   1e-1),
        ("grad_up_A",     out["grad_up_A"],     ref["ref_grad_up_A"],     1e-1),
        ("grad_up_B",     out["grad_up_B"],     ref["ref_grad_up_B"],     1e-1),
        ("grad_down_A",   out["grad_down_A"],   ref["ref_grad_down_A"],   1e-1),
        ("grad_down_B",   out["grad_down_B"],   ref["ref_grad_down_B"],   1e-1),
    ]:
        if not compare(ours, ref_t, label, max_abs_thresh=thresh):
            fails.append(label)

    print()
    print("=" * 60)
    if not fails:
        print("ALL CHECKS PASSED ✓  layer_mlp_bwd matches autograd within bf16 tolerance")
    else:
        print(f"FAIL on: {fails}")
        sys.exit(1)


if __name__ == "__main__":
    main()
