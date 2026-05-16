"""Kernel determinism smoke test.

Running the same prefill twice on identical inputs MUST produce
identical outputs (cos=1.0). Catches regressions of the conv1d
ring-buffer race fix (F2) and any future inter-block race in the
megakernel.

Uses random small weights to keep the test fast (no HF load).

Run:
    /home/sparkz/rl/.venv/bin/python3 \
        models/qwen36_27b/test/test_kernel_determinism.py
"""
from __future__ import annotations
import os, sys
import torch
import torch.nn.functional as F

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))

from weight_packer import (
    NUM_LAYERS, HIDDEN_SIZE as H, INTERMEDIATE_SIZE as I_,
    FA_HEAD_DIM, FA_Q_SIZE, FA_KV_SIZE, FA_QPROJ_SIZE,
    DN_NUM_V_HEADS, DN_HEAD_DIM, DN_V_SIZE, DN_CONV_CH, DN_CONV_KERNEL,
    LAYER_TYPE, VOCAB_SIZE, alloc_scratch, pack_layer_weights,
)
import importlib.util as _u
_spec = _u.spec_from_file_location(
    "qwen3x_C",
    "/home/sparkz/rl/lucebox-hub/models/qwen36_27b/megakernel/"
    "qwen3x_C.cpython-312-aarch64-linux-gnu.so")
_m = _u.module_from_spec(_spec); _spec.loader.exec_module(_m)


def build_random_weights():
    """Returns (embed, final_norm, blob, layer_data) — keep layer_data alive
    in the caller because pack_layer_weights stores raw .data_ptr() values
    into the blob; the underlying tensors must outlive every kernel call."""
    torch.manual_seed(0)
    def bf(*s): return (torch.randn(*s, dtype=torch.bfloat16, device="cuda") * 0.02)
    embed = bf(VOCAB_SIZE, H); final_norm = bf(H)
    fa_norm = bf(H); fa_qproj = bf(FA_QPROJ_SIZE, H); fa_k = bf(FA_KV_SIZE, H); fa_v = bf(FA_KV_SIZE, H)
    fa_qn = bf(FA_HEAD_DIM); fa_kn = bf(FA_HEAD_DIM); fa_o = bf(H, FA_Q_SIZE); fa_pan = bf(H)
    gate = bf(I_, H); up = bf(I_, H); down = bf(H, I_)
    dn_norm = bf(H); dn_qkv = bf(DN_CONV_CH, H); dn_z = bf(DN_V_SIZE, H)
    dn_b = bf(DN_NUM_V_HEADS, H); dn_a = bf(DN_NUM_V_HEADS, H)
    dn_conv = bf(DN_CONV_CH, DN_CONV_KERNEL); dn_al = bf(DN_NUM_V_HEADS); dn_dt = bf(DN_NUM_V_HEADS)
    dn_norm2 = bf(DN_HEAD_DIM); dn_out = bf(H, DN_V_SIZE); dn_pan = bf(H)
    layer_data = []
    for i in range(NUM_LAYERS):
        if LAYER_TYPE[i] == 1:
            layer_data.append({"type": 1,
                "ptrs": [fa_norm, fa_qproj, fa_k, fa_v, fa_qn, fa_kn, fa_o,
                          fa_pan, gate, up, down]})
        else:
            layer_data.append({"type": 0,
                "ptrs": [dn_norm, dn_qkv, dn_z, dn_b, dn_a, dn_conv, dn_al,
                          dn_dt, dn_norm2, dn_out, dn_pan, gate, up, down]})
    return embed, final_norm, pack_layer_weights(layer_data), layer_data


def main():
    embed, final_norm, blob, _keepalive = build_random_weights()
    ops = torch.ops.qwen3x_C

    def run_prefill(S):
        T = torch.arange(S, dtype=torch.int32, device="cuda") + 1
        sc = alloc_scratch(max_seq=128, verbose=False)
        ops.prefill_qwen3x_naive(
            1, T, embed, final_norm, blob,
            sc.fa_k_cache, sc.fa_v_cache, sc.dn_states, sc.conv_bufs,
            sc.hidden_buffer, sc.g_residual,
            sc.g_qkv_scratch, sc.g_kv_scratch, sc.g_attn_out, sc.g_mlp_inter,
            sc.g_z_scratch, sc.g_beta_scratch, sc.g_alpha_scratch,
            sc.g_normalized, sc.g_fa_partials, sc.g_rope_inv_freq,
            128, 1.0, 32.0, 1.0, 262144, False, 0, None, 0)
        torch.cuda.synchronize()
        return sc.g_normalized.cpu().float()

    failures = 0
    # S=5 is the first iteration that wraps the conv1d ring buffer
    # (CONV_K=4, so position 4 overwrites slot 0). S=14 was the
    # original repro from F2. S=32 stresses the FA cache reads.
    for S in (5, 14, 32):  # S=5 wraps conv1d ring buffer, S=14 was F2 repro
        a = run_prefill(S)
        b = run_prefill(S)
        cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item()
        diff = (a - b).abs().max().item()
        nan = bool(torch.isnan(a).any() or torch.isnan(b).any())
        flag = "  <-- FAIL" if (cos < 0.9999 or diff > 1e-3 or nan) else ""
        print(f"  S={S:>3}  cos={cos:.8f}  max_diff={diff:.6g}  nan={nan}{flag}")
        if cos < 0.9999 or diff > 1e-3 or nan:
            failures += 1

    if failures:
        print(f"\nFAIL: {failures} non-deterministic prefills")
        sys.exit(1)
    print("\nPASS  kernel is deterministic")


if __name__ == "__main__":
    main()
