"""Validate dn_attn_handrolled (Python forward + manual fla-direct bwd)
against HF's linear_attn under torch autograd.
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")

# fla shim must come before any HF import that triggers fla.
import _fla_torch_compile_shim  # noqa: F401, E402

import torch  # noqa: E402

# Disable HF's fla-import probe so the trainer's standard shim path is used.
try:
    import transformers.utils.import_utils as _iu
    _iu.is_flash_linear_attention_available = lambda: False
except Exception:
    pass

from transformers import AutoModelForCausalLM  # noqa: E402

from dn_attn_handrolled import dn_attn_forward, dn_attn_backward  # noqa: E402
from dn_hf_patch import patch_hf_qwen3_deltanet  # noqa: E402

S = 16


def main():
    print("Loading Qwen3.5-0.8B + applying DN patch (so HF uses fla too)...")
    m = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16
    ).cuda().eval()
    n = patch_hf_qwen3_deltanet(m)
    print(f"  patched {n} layers")

    layer_idx = 1
    layer = m.model.layers[layer_idx]
    dn = layer.linear_attn
    input_norm = layer.input_layernorm

    HIDDEN = dn.hidden_size
    print(f"  HIDDEN={HIDDEN}, S={S}, layer_idx={layer_idx}")

    torch.manual_seed(42)
    h_in = (torch.randn(1, S, HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.5)
    g_for_loss = torch.randn(1, S, HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.1

    # --- Reference: autograd through HF.input_layernorm + HF.linear_attn -
    h_ref = h_in.detach().clone().requires_grad_(True)
    npa_hf = input_norm(h_ref)
    out_hf = dn(npa_hf)
    loss = (out_hf.float() * g_for_loss.float()).sum()
    loss.backward()
    dh_in_ref = h_ref.grad.detach().clone()

    # --- Ours: dn_attn_forward + dn_attn_backward -------------------------
    # Note: our forward includes the input_layernorm (matching the way
    # per_layer_bwd_dn frames the boundary: dh_in is gradient on the
    # layer's residual-stream input).
    rms_eps_in = getattr(input_norm, "eps", getattr(input_norm, "variance_epsilon", 1e-6))
    rms_eps_dn = getattr(dn.norm, "eps", getattr(dn.norm, "variance_epsilon", 1e-6))

    out_ours, saves = dn_attn_forward(
        h_in,
        input_norm_w=input_norm.weight,
        in_proj_qkv_W=dn.in_proj_qkv.weight,
        in_proj_z_W=dn.in_proj_z.weight,
        in_proj_b_W=dn.in_proj_b.weight,
        in_proj_a_W=dn.in_proj_a.weight,
        conv1d_W=dn.conv1d.weight,
        A_log=dn.A_log,
        dt_bias=dn.dt_bias,
        dn_norm_W=dn.norm.weight,
        out_proj_W=dn.out_proj.weight,
        rms_eps=rms_eps_in,
        layer_norm_eps=rms_eps_dn,
    )
    dh_in_ours = dn_attn_backward(g_for_loss.float(), saves)

    # --- Compare forward outputs -----------------------------------------
    out_diff = (out_hf.float() - out_ours.float()).abs().max().item()
    out_cos = torch.nn.functional.cosine_similarity(
        out_hf.float().flatten(), out_ours.float().flatten(), dim=0).item()
    print(f"forward out max|d|={out_diff:.4e}  cos={out_cos:.6f}")

    # --- Compare backward gradient ---------------------------------------
    dh_diff = (dh_in_ref.float() - dh_in_ours.float()).abs().max().item()
    dh_cos = torch.nn.functional.cosine_similarity(
        dh_in_ref.float().flatten(), dh_in_ours.float().flatten(), dim=0).item()
    print(f"dh_in    max|d|={dh_diff:.4e}  cos={dh_cos:.6f}")

    # --- Speed comparison ------------------------------------------------
    def hf_path():
        h = h_in.detach().clone().requires_grad_(True)
        npa = input_norm(h)
        out = dn(npa)
        out.backward(g_for_loss)
        return h.grad

    def ours_path():
        out, saves = dn_attn_forward(
            h_in,
            input_norm_w=input_norm.weight,
            in_proj_qkv_W=dn.in_proj_qkv.weight,
            in_proj_z_W=dn.in_proj_z.weight,
            in_proj_b_W=dn.in_proj_b.weight,
            in_proj_a_W=dn.in_proj_a.weight,
            conv1d_W=dn.conv1d.weight,
            A_log=dn.A_log,
            dt_bias=dn.dt_bias,
            dn_norm_W=dn.norm.weight,
            out_proj_W=dn.out_proj.weight,
            rms_eps=rms_eps_in,
            layer_norm_eps=rms_eps_dn,
        )
        return dn_attn_backward(g_for_loss.float(), saves)

    for _ in range(5): hf_path(); ours_path()
    torch.cuda.synchronize()
    N = 100
    t0 = time.perf_counter()
    for _ in range(N): hf_path()
    torch.cuda.synchronize()
    hf_ms = (time.perf_counter() - t0) * 1000.0 / N

    t0 = time.perf_counter()
    for _ in range(N): ours_path()
    torch.cuda.synchronize()
    ours_ms = (time.perf_counter() - t0) * 1000.0 / N

    print()
    print(f"HF autograd through linear_attn:  {hf_ms:.3f} ms/call")
    print(f"Ours (manual fwd + fla-bwd direct): {ours_ms:.3f} ms/call ({hf_ms/ours_ms:.2f}x)")

    print()
    print("=" * 60)
    if out_diff < 5e-2 and out_cos > 0.999 and dh_diff < 5e-2 and dh_cos > 0.999:
        print("ALL CHECKS PASSED  dn_attn_handrolled matches HF autograd")
    else:
        print(f"FAIL: out_cos={out_cos:.6f}, dh_cos={dh_cos:.6f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
