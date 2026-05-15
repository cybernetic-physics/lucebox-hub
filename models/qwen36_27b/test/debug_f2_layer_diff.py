"""Debug F2 bug: find first divergent layer between one-shot and split prefill.

Both paths produce a position-13 hidden state (capture at last step).
Path A: oneshot prefill(ids[:14])  -> capture layer outputs
Path B: prefill(ids[:7]) + prefill(ids[7:], start_position=7) -> capture at last step

Compare cos similarity per layer. The first layer where cos < 0.99 is where
the bug surfaces. Walking back from there finds the cause.

Run:
    /home/sparkz/rl/.venv/bin/python3 \
        models/qwen36_27b/test/debug_f2_layer_diff.py
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

    def _run(sc, tokens, start_position, layer_capture):
        ops.prefill_qwen3x_naive(
            1, tokens, embed, final_norm, blob,
            sc.fa_k_cache, sc.fa_v_cache, sc.dn_states, sc.conv_bufs,
            sc.hidden_buffer, sc.g_residual,
            sc.g_qkv_scratch, sc.g_kv_scratch, sc.g_attn_out, sc.g_mlp_inter,
            sc.g_z_scratch, sc.g_beta_scratch, sc.g_alpha_scratch,
            sc.g_normalized, sc.g_fa_partials, sc.g_rope_inv_freq,
            128, 1.0, 32.0, 1.0, 262144, False, 0, layer_capture, start_position)

    S, S1 = 14, 7
    T1 = torch.arange(S1, dtype=torch.int32, device="cuda") + 1
    T2 = torch.arange(S - S1, dtype=torch.int32, device="cuda") + 1 + S1
    T_all = torch.cat([T1, T2])

    cap_a = torch.zeros(NUM_LAYERS, H, dtype=torch.bfloat16, device="cuda")
    cap_b = torch.zeros(NUM_LAYERS, H, dtype=torch.bfloat16, device="cuda")

    sc_a = alloc_scratch(max_seq=128, verbose=False)
    _run(sc_a, T_all, 0, cap_a)
    torch.cuda.synchronize()

    sc_b = alloc_scratch(max_seq=128, verbose=False)
    _run(sc_b, T1, 0, None)
    _run(sc_b, T2, S1, cap_b)
    torch.cuda.synchronize()

    cap_a_f = cap_a.float()
    cap_b_f = cap_b.float()
    print(f"\nper-layer cos (oneshot vs split, both at position 13):")
    print(f"  {'lay':>4} {'type':>4}  {'cos':>10}  {'max_abs':>10}")
    first_divergent = None
    for i in range(NUM_LAYERS):
        c = F.cosine_similarity(cap_a_f[i].unsqueeze(0),
                                  cap_b_f[i].unsqueeze(0), dim=-1).item()
        d = (cap_a_f[i] - cap_b_f[i]).abs().max().item()
        ty = "FA" if LAYER_TYPE[i] == 1 else "DN"
        marker = ""
        if c < 0.99 and first_divergent is None:
            first_divergent = i
            marker = "  <-- first divergent"
        if i < 8 or (first_divergent and i <= first_divergent + 1) or i % 8 == 0:
            print(f"  {i:>4} {ty:>4}  {c:>10.6f}  {d:>10.4f}{marker}")
    if first_divergent is None:
        print("\nAll layers match (cos > 0.99). Bug must be in final-norm or downstream.")
    else:
        print(f"\nFirst divergent layer: {first_divergent} ({'FA' if LAYER_TYPE[first_divergent]==1 else 'DN'})")


if __name__ == "__main__":
    main()
