"""Smoke test for the NVFP4 dispatch path in decode_qwen3x.

Allocates zero NVFP4 packed buffers for every projection across all 64
layers, packs a `layer_data` blob with types 2/3 (DN_nvfp4 / FA_nvfp4),
and launches decode_qwen3x with model_id=3 (Cfg_27B + USE_NVFP4=true).

Verifies the kernel runs to completion without crashes — does NOT check
numeric correctness (weights are zero). End-to-end correctness vs HF is
tested separately in test_correctness_nvfp4_vs_hf.py.

Run:
    /home/sparkz/rl/.venv/bin/python3 \
        models/qwen36_27b/test/test_nvfp4_dispatch.py
"""
from __future__ import annotations
import os, sys
import torch

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


GROUP = 32


def _bf(*shape): return torch.zeros(*shape, dtype=torch.bfloat16, device="cuda")
def _u8(*shape): return torch.zeros(*shape, dtype=torch.uint8, device="cuda")
def _h16(*shape): return torch.zeros(*shape, dtype=torch.float16, device="cuda")


def fp4_pack(out_dim, in_dim):
    """Return (packed_data, scales) for a fake NVFP4 weight of shape [out, in]."""
    return (_u8(out_dim, in_dim // 2).contiguous(),
            _h16(out_dim, in_dim // GROUP).contiguous())


def main():
    print("Allocating zero NVFP4 layer-shared buffers...")
    fa_norm = _bf(H); fa_qn = _bf(FA_HEAD_DIM); fa_kn = _bf(FA_HEAD_DIM); fa_pan = _bf(H)
    fa_qproj = fp4_pack(FA_QPROJ_SIZE, H)
    fa_k     = fp4_pack(FA_KV_SIZE,    H)
    fa_v     = fp4_pack(FA_KV_SIZE,    H)
    fa_o     = fp4_pack(H,             FA_Q_SIZE)
    gate = fp4_pack(I_, H); up = fp4_pack(I_, H); down = fp4_pack(H, I_)

    dn_norm = _bf(H); dn_norm2 = _bf(DN_HEAD_DIM); dn_pan = _bf(H)
    dn_conv = _bf(DN_CONV_CH, DN_CONV_KERNEL); dn_al = _bf(DN_NUM_V_HEADS); dn_dt = _bf(DN_NUM_V_HEADS)
    dn_qkv = fp4_pack(DN_CONV_CH, H)
    dn_z   = fp4_pack(DN_V_SIZE,  H)
    dn_b   = fp4_pack(DN_NUM_V_HEADS, H)
    dn_a   = fp4_pack(DN_NUM_V_HEADS, H)
    dn_out = fp4_pack(H, DN_V_SIZE)

    embed = _bf(VOCAB_SIZE, H); final_norm = _bf(H)

    layer_data = []
    for i in range(NUM_LAYERS):
        if LAYER_TYPE[i] == 1:
            layer_data.append({"type": 3,
                "ptrs": [fa_norm, fa_qproj, fa_k, fa_v, fa_qn, fa_kn, fa_o,
                          fa_pan, gate, up, down]})
        else:
            layer_data.append({"type": 2,
                "ptrs": [dn_norm, dn_qkv, dn_z, dn_b, dn_a, dn_conv, dn_al,
                          dn_dt, dn_norm2, dn_out, dn_pan, gate, up, down]})
    blob = pack_layer_weights(layer_data)
    sc = alloc_scratch(max_seq=128, verbose=False)

    print(f"  layer blob: {blob.numel()} bytes "
          f"(NUM_LAYERS * 192 = {NUM_LAYERS*192})")
    print(f"  GPU alloc: "
          f"{torch.cuda.memory_allocated()/(1024**3):.2f} GB")

    print("\nLaunching decode_qwen3x(model_id=3, NVFP4, Cfg_27B)...")
    torch.ops.qwen3x_C.decode_qwen3x(
        3, embed, final_norm, blob,
        sc.fa_k_cache, sc.fa_v_cache, sc.dn_states, sc.conv_bufs,
        sc.hidden_buffer, sc.g_residual,
        sc.g_qkv_scratch, sc.g_kv_scratch, sc.g_attn_out, sc.g_mlp_inter,
        sc.g_z_scratch, sc.g_beta_scratch, sc.g_alpha_scratch,
        sc.g_normalized, sc.g_fa_partials, sc.g_rope_inv_freq,
        220, 0, 0, 0, 128,
        1.0, 32.0, 1.0, 262144, False, 0, None)
    torch.cuda.synchronize()
    print(f"PASS  NVFP4 dispatch ran without crash; "
          f"g_normalized max = {sc.g_normalized.abs().max().item():.4g}")


if __name__ == "__main__":
    main()
