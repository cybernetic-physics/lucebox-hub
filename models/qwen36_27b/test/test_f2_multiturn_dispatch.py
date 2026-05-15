"""F2 smoke: multi-turn KV-reuse dispatch.

Allocates zero-weight buffers and runs prefill_qwen3x_naive twice:
  turn 1: tokens [220,221,222] at start_position=0
  turn 2: tokens [223,224]      at start_position=3
The second turn must not crash and must write into the KV cache at
positions 3..4 without disturbing positions 0..2.

This does NOT check numeric correctness — for that we need a test with
real weights that compares ours vs a single HF forward on the
concatenated sequence. That lives in test_correctness_multiturn.py
(TODO).

Run:
    /home/sparkz/rl/.venv/bin/python3 \
        models/qwen36_27b/test/test_f2_multiturn_dispatch.py
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


def _bf(*shape): return torch.zeros(*shape, dtype=torch.bfloat16, device="cuda")


def main():
    fa_norm = _bf(H); fa_qproj = _bf(FA_QPROJ_SIZE, H)
    fa_k = _bf(FA_KV_SIZE, H); fa_v = _bf(FA_KV_SIZE, H)
    fa_qn = _bf(FA_HEAD_DIM); fa_kn = _bf(FA_HEAD_DIM)
    fa_o = _bf(H, FA_Q_SIZE); fa_pan = _bf(H)
    gate = _bf(I_, H); up = _bf(I_, H); down = _bf(H, I_)

    dn_norm = _bf(H); dn_qkv = _bf(DN_CONV_CH, H); dn_z = _bf(DN_V_SIZE, H)
    dn_b = _bf(DN_NUM_V_HEADS, H); dn_a = _bf(DN_NUM_V_HEADS, H)
    dn_conv = _bf(DN_CONV_CH, DN_CONV_KERNEL)
    dn_al = _bf(DN_NUM_V_HEADS); dn_dt = _bf(DN_NUM_V_HEADS)
    dn_norm2 = _bf(DN_HEAD_DIM); dn_out = _bf(H, DN_V_SIZE); dn_pan = _bf(H)
    embed = _bf(VOCAB_SIZE, H); final_norm = _bf(H)

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
    blob = pack_layer_weights(layer_data)
    sc = alloc_scratch(max_seq=128, verbose=False)
    ops = torch.ops.qwen3x_C

    common = (embed, final_norm, blob,
              sc.fa_k_cache, sc.fa_v_cache, sc.dn_states, sc.conv_bufs,
              sc.hidden_buffer, sc.g_residual,
              sc.g_qkv_scratch, sc.g_kv_scratch, sc.g_attn_out, sc.g_mlp_inter,
              sc.g_z_scratch, sc.g_beta_scratch, sc.g_alpha_scratch,
              sc.g_normalized, sc.g_fa_partials, sc.g_rope_inv_freq,
              128, 1.0, 32.0, 1.0, 262144, False, 0, None)

    turn1 = torch.tensor([220, 221, 222], dtype=torch.int32, device="cuda")
    turn2 = torch.tensor([223, 224], dtype=torch.int32, device="cuda")

    ops.prefill_qwen3x_naive(1, turn1, *common, 0)
    torch.cuda.synchronize()
    print("PASS  turn 1 (start_position=0, S=3)")

    ops.prefill_qwen3x_naive(1, turn2, *common, 3)
    torch.cuda.synchronize()
    print("PASS  turn 2 (start_position=3, S=2)")

    # KV-cache slots [0..2] held turn1; [3..4] hold turn2. With zero weights
    # everything is zero anyway, but the dispatch path itself is exercised.
    print("F2 multi-turn dispatch smoke: OK")


if __name__ == "__main__":
    main()
