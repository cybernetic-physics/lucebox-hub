"""F2 bug repro (FAST: no HF load).

Allocates random small bf16 weights, runs:
  Path A: prefill_qwen3x_naive(ids[:S], start_position=0)
  Path B: prefill_qwen3x_naive(ids[:S1], start_position=0)
          prefill_qwen3x_naive(ids[S1:], start_position=S1)
on two SEPARATE Scratch buffers and compares the final
`g_normalized` vectors. They should be identical.

The full-stack F8 test on real HF weights produces NaN logits on
Path B; this faster repro stays finite but cos(A, B) ≈ 0.5, which
is exactly the same bug surfacing with smaller weight magnitude.

Run:
    /home/sparkz/rl/.venv/bin/python3 \
        models/qwen36_27b/test/test_f2_split_vs_oneshot_fast.py
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


def main():
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
    blob = pack_layer_weights(layer_data)
    ops = torch.ops.qwen3x_C

    S, S1 = 14, 7
    T1 = torch.arange(S1, dtype=torch.int32, device="cuda") + 1
    T2 = torch.arange(S - S1, dtype=torch.int32, device="cuda") + 1 + S1
    T_all = torch.cat([T1, T2])

    def _run(sc, tokens, start_position):
        ops.prefill_qwen3x_naive(
            1, tokens, embed, final_norm, blob,
            sc.fa_k_cache, sc.fa_v_cache, sc.dn_states, sc.conv_bufs,
            sc.hidden_buffer, sc.g_residual,
            sc.g_qkv_scratch, sc.g_kv_scratch, sc.g_attn_out, sc.g_mlp_inter,
            sc.g_z_scratch, sc.g_beta_scratch, sc.g_alpha_scratch,
            sc.g_normalized, sc.g_fa_partials, sc.g_rope_inv_freq,
            128, 1.0, 32.0, 1.0, 262144, False, 0, None, start_position)

    sc_a = alloc_scratch(max_seq=128, verbose=False)
    _run(sc_a, T_all, 0)
    torch.cuda.synchronize()
    gn_a = sc_a.g_normalized.cpu().float()

    sc_b = alloc_scratch(max_seq=128, verbose=False)
    _run(sc_b, T1, 0)
    _run(sc_b, T2, S1)
    torch.cuda.synchronize()
    gn_b = sc_b.g_normalized.cpu().float()

    cos = F.cosine_similarity(gn_a.unsqueeze(0), gn_b.unsqueeze(0), dim=-1).item()
    max_abs = (gn_a - gn_b).abs().max().item()
    print(f"  oneshot:  max_abs={gn_a.abs().max():.3f}  finite={bool(torch.isfinite(gn_a).all())}")
    print(f"  split:    max_abs={gn_b.abs().max():.3f}  finite={bool(torch.isfinite(gn_b).all())}")
    print(f"  cos(oneshot, split) = {cos:.4f}   diff_max_abs = {max_abs:.4f}")

    if cos > 0.99:
        print("\nPASS  split prefill matches one-shot")
    else:
        print(f"\nFAIL  split prefill diverges (expected cos>0.99, got {cos:.4f})")
        sys.exit(1)


if __name__ == "__main__":
    main()
