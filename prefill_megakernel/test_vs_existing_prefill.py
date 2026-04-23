"""Numerical correctness: our single-dispatch prefill megakernel vs the
existing multi-kernel prefill_bf16 in ../megakernel/.

Same random weights, same tokens, compare the argmax token ID. Because the
math is ported verbatim and both run in bf16 with f32 accumulation, this
should match exactly (or within a one-ULP tie-break when two logits are
near-equal).
"""

from __future__ import annotations

import struct
import sys
import torch

sys.path.insert(0, "/root/lucebox-hub-b200-train/megakernel")
sys.path.insert(0, "/root/lucebox-hub-b200-train/prefill_megakernel")

import qwen35_megakernel_bf16_C  # noqa: F401
from model import (
    NUM_LAYERS, HIDDEN, VOCAB, MAX_SEQ_LEN, LAYER_TYPE,
    FA_QPROJ_SIZE, FA_KV_SIZE, FA_HEAD_DIM, FA_KV_HEADS, FA_Q_SIZE,
    DN_CONV_CH, DN_V_SIZE, DN_HEADS, DN_VAL, DN_CONV_K, DN_KEY,
    INTER, N_FA, N_DN,
    PrefillMegakernel,
)
from smoke import random_weights


# ==== Mirrors ../megakernel/model.py _pack_layer_weights ====
def _pack_layer_weights_legacy(layer_data):
    ptr_size = 8
    max_ptrs = 14
    header = 16
    struct_size = header + max_ptrs * ptr_size
    buf = bytearray(NUM_LAYERS * struct_size)
    for i in range(NUM_LAYERS):
        ld = layer_data[i]
        off = i * struct_size
        struct.pack_into("iiii", buf, off, ld["type"], 0, 0, 0)
        for j, t in enumerate(ld["ptrs"]):
            struct.pack_into("Q", buf, off + header + j * ptr_size, t.data_ptr())
        for j in range(len(ld["ptrs"]), max_ptrs):
            struct.pack_into("Q", buf, off + header + j * ptr_size, 0)
    return torch.frombuffer(buf, dtype=torch.uint8).cuda()


def run_existing_prefill(weights, tokens):
    """Run ../megakernel/prefill.cu's prefill_bf16 on the same inputs."""
    op = torch.ops.qwen35_megakernel_bf16_C.prefill_bf16
    S = int(tokens.shape[0])
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    f32 = dict(dtype=torch.float32, device="cuda")
    i32 = dict(dtype=torch.int32, device="cuda")

    layers_packed = _pack_layer_weights_legacy(weights["layer_data"])

    fa_k_cache = torch.zeros(N_FA, FA_KV_HEADS, MAX_SEQ_LEN, FA_HEAD_DIM, **bf16)
    fa_v_cache = torch.zeros_like(fa_k_cache)
    dn_states = torch.zeros(N_DN, DN_HEADS, DN_KEY, DN_VAL, **f32)
    conv_bufs = torch.zeros(N_DN, DN_CONV_CH, DN_CONV_K, **f32)

    max_proj = max(FA_QPROJ_SIZE, DN_CONV_CH, INTER)
    max_attn = max(FA_Q_SIZE, DN_V_SIZE, FA_KV_SIZE)
    hidden = torch.empty(S, HIDDEN, **bf16)
    residual = torch.empty(S, HIDDEN, **bf16)
    normalized = torch.empty(S, HIDDEN, **bf16)
    proj_buf = torch.empty(S, max_proj, **bf16)
    proj_buf2 = torch.empty(S, max_proj, **bf16)
    attn_buf = torch.empty(S, max_attn, **bf16)
    mlp_buf = torch.empty(S, INTER, **bf16)
    dn_out_buf = torch.empty(S, max_attn, **bf16)
    beta_buf = torch.empty(S, DN_HEADS, **f32)
    alpha_buf = torch.empty(S, DN_HEADS, **f32)
    final_normed = torch.empty(HIDDEN, **bf16)
    hidden_bf16_out = torch.empty(HIDDEN, **bf16)
    out_token = torch.empty(1, **i32)
    lm_bmv = torch.empty(512, **f32)
    lm_bmi = torch.empty(512, **i32)

    op(
        out_token, tokens.to(dtype=torch.int32, device="cuda").contiguous(),
        weights["embed_weight"], layers_packed,
        weights["final_norm_weight"], weights["lm_head_weight"],
        fa_k_cache, fa_v_cache, dn_states, conv_bufs,
        hidden, residual, normalized,
        proj_buf, proj_buf2, attn_buf, mlp_buf, dn_out_buf,
        beta_buf, alpha_buf,
        final_normed, hidden_bf16_out,
        lm_bmv, lm_bmi,
    )
    return int(out_token.item())


def main():
    torch.manual_seed(42)
    S = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    w = random_weights()

    tokens = torch.randint(0, VOCAB, (S,), dtype=torch.int32, device="cuda")

    # Existing multi-kernel prefill
    existing = run_existing_prefill(w, tokens)
    print(f"existing prefill_bf16:  next_token = {existing}")

    # Fresh weights dict for the megakernel (so it doesn't share state).
    # Layer_data contains the same tensors so they share storage — fine.
    m = PrefillMegakernel(w)
    mega = m.forward(tokens)
    print(f"single-dispatch mega:   next_token = {mega}")

    match = (existing == mega)
    print(f"match: {match}")
    if not match:
        print(f"  Δ = {existing - mega}")
        sys.exit(1)


if __name__ == "__main__":
    main()
