"""Forward-path sanity check for the LoRA-extended megakernel.

Checks:
  1. With LoRA weights all zero, output == baseline prefill megakernel.
  2. With nonzero LoRA A/B weights, output differs from baseline.
"""

from __future__ import annotations

import struct
import sys
import torch

sys.path.insert(0, "/root/lucebox-hub-b200-train/prefill_megakernel")
sys.path.insert(0, "/root/lucebox-hub-b200-train/train_megakernel")

import prefill_megakernel_C  # noqa: F401
import train_megakernel_C    # noqa: F401

from model import (  # reuse prefill_megakernel's Python helpers
    NUM_LAYERS, HIDDEN, VOCAB, MAX_SEQ_LEN, LAYER_TYPE, N_FA,
    FA_QPROJ_SIZE, FA_KV_SIZE, FA_HEAD_DIM, FA_KV_HEADS, FA_Q_SIZE,
    DN_CONV_CH, DN_V_SIZE, DN_HEADS, DN_VAL, DN_CONV_K, DN_KEY, INTER, N_DN,
    PrefillMegakernel,
)
from smoke import random_weights


def _pack_layer_weights(layer_data):
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


def run_train_mega(weights, tokens, lora_a_q, lora_b_q, use_lora, lora_rank, lora_scaling):
    S = int(tokens.shape[0])
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    f32 = dict(dtype=torch.float32, device="cuda")
    i32 = dict(dtype=torch.int32, device="cuda")

    layers_packed = _pack_layer_weights(weights["layer_data"])

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
    out_token = torch.empty(1, **i32)
    lm_bmv = torch.empty(148, **f32)
    lm_bmi = torch.empty(148, **i32)
    lora_h_ws = torch.empty(S, lora_rank, **f32)

    torch.ops.train_megakernel_C.train_mega_forward(
        out_token, tokens.to(dtype=torch.int32, device="cuda").contiguous(),
        weights["embed_weight"], layers_packed,
        weights["final_norm_weight"], weights["lm_head_weight"],
        lora_a_q, lora_b_q,
        lora_rank, lora_scaling, lora_h_ws,
        fa_k_cache, fa_v_cache, dn_states, conv_bufs,
        hidden, residual, normalized,
        proj_buf, proj_buf2, attn_buf, mlp_buf, dn_out_buf,
        beta_buf, alpha_buf,
        final_normed,
        lm_bmv, lm_bmi,
        MAX_SEQ_LEN, use_lora,
    )
    return int(out_token.item())


def main():
    torch.manual_seed(42)
    S = 32
    LORA_R = 16
    w = random_weights()
    tokens = torch.randint(0, VOCAB, (S,), dtype=torch.int32, device="cuda")

    lora_a_zero = torch.zeros(N_FA, HIDDEN, LORA_R, device="cuda", dtype=torch.bfloat16)
    lora_b_zero = torch.zeros(N_FA, LORA_R, FA_QPROJ_SIZE, device="cuda", dtype=torch.bfloat16)
    # Use boosted init so LoRA residual is large enough to move the random-weight argmax.
    lora_a_nz = (torch.randn(N_FA, HIDDEN, LORA_R) * 0.5).to(device="cuda", dtype=torch.bfloat16)
    lora_b_nz = (torch.randn(N_FA, LORA_R, FA_QPROJ_SIZE) * 0.5).to(device="cuda", dtype=torch.bfloat16)

    # Baseline: run with use_lora=False (should match prefill_megakernel output)
    base = PrefillMegakernel(w).forward(tokens)

    # Path A: train_megakernel with LoRA DISABLED (skip LoRA compute)
    a = run_train_mega(w, tokens, lora_a_zero, lora_b_zero, False, LORA_R, 1.0)

    # Path B: train_megakernel with LoRA ENABLED but weights zero
    b = run_train_mega(w, tokens, lora_a_zero, lora_b_zero, True, LORA_R, 1.0)

    # Path C: train_megakernel with LoRA ENABLED + nonzero weights
    c = run_train_mega(w, tokens, lora_a_nz, lora_b_nz, True, LORA_R, 1.0)

    print(f"baseline prefill_megakernel  next_token = {base}")
    print(f"train_mega use_lora=False    next_token = {a}  (expect == baseline)")
    print(f"train_mega use_lora=True A=B=0 next_token = {b}  (expect == baseline)")
    print(f"train_mega use_lora=True A,B≠0 next_token = {c}  (expect ≠ baseline)")

    ok = True
    if a != base:
        print("  FAIL: use_lora=False diverges from baseline"); ok = False
    if b != base:
        print("  FAIL: use_lora=True with zero weights diverges from baseline"); ok = False
    if c == base:
        print("  FAIL: use_lora=True with nonzero weights matches baseline (LoRA not applied?)"); ok = False
    print("OK" if ok else "CORRECTNESS FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
