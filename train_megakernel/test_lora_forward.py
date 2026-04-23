"""Forward-path sanity check for the LoRA-extended megakernel.

Exercises all 13 trainable linears (7 FA + 6 DN) with three-way check:
  1. Baseline: LoRA disabled → same as prefill_megakernel_C.prefill_bf16.
  2. LoRA enabled but A=B=0 → same as baseline (no-op).
  3. LoRA enabled with boosted nonzero weights → diverges from baseline.
"""

from __future__ import annotations

import struct
import sys
import torch

sys.path.insert(0, "/root/lucebox-hub-b200-train/prefill_megakernel")
sys.path.insert(0, "/root/lucebox-hub-b200-train/train_megakernel")

import prefill_megakernel_C  # noqa: F401
import train_megakernel_C    # noqa: F401

from model import (
    NUM_LAYERS, HIDDEN, VOCAB, MAX_SEQ_LEN, LAYER_TYPE, N_FA, N_DN,
    FA_QPROJ_SIZE, FA_KV_SIZE, FA_HEAD_DIM, FA_KV_HEADS, FA_Q_SIZE,
    DN_CONV_CH, DN_V_SIZE, DN_HEADS, DN_VAL, DN_CONV_K, DN_KEY, INTER,
    PrefillMegakernel,
)
from smoke import random_weights

# LoRA projection specs (input dim, output dim) — must match kernel order.
FA_SHAPES = [
    ("fa_q",    HIDDEN,      FA_QPROJ_SIZE),
    ("fa_k",    HIDDEN,      FA_KV_SIZE),
    ("fa_v",    HIDDEN,      FA_KV_SIZE),
    ("fa_o",    FA_Q_SIZE,   HIDDEN),
    ("fa_gate", HIDDEN,      INTER),
    ("fa_up",   HIDDEN,      INTER),
    ("fa_down", INTER,       HIDDEN),
]
DN_SHAPES = [
    ("dn_qkv",  HIDDEN,      DN_CONV_CH),
    ("dn_z",    HIDDEN,      DN_V_SIZE),
    ("dn_out",  DN_V_SIZE,   HIDDEN),
    ("dn_gate", HIDDEN,      INTER),
    ("dn_up",   HIDDEN,      INTER),
    ("dn_down", INTER,       HIDDEN),
]


def _pack_layer_weights(layer_data):
    ptr_size = 8; max_ptrs = 14; header = 16
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


def build_lora_tensors(rank, init_scale, n_fa=N_FA, n_dn=N_DN, device="cuda"):
    """Return 26 tensors (A, B for each of 13 trainable linears)."""
    tensors = []
    gen = torch.Generator(device="cpu").manual_seed(123)
    for name, k_in, k_out in FA_SHAPES:
        A = (torch.randn(n_fa, k_in, rank, generator=gen) * init_scale).to(device=device, dtype=torch.bfloat16)
        B = (torch.randn(n_fa, rank, k_out, generator=gen) * init_scale).to(device=device, dtype=torch.bfloat16)
        tensors.extend([A, B])
    for name, k_in, k_out in DN_SHAPES:
        A = (torch.randn(n_dn, k_in, rank, generator=gen) * init_scale).to(device=device, dtype=torch.bfloat16)
        B = (torch.randn(n_dn, rank, k_out, generator=gen) * init_scale).to(device=device, dtype=torch.bfloat16)
        tensors.extend([A, B])
    return tensors


def zero_lora_tensors(rank):
    tensors = []
    for _, k_in, k_out in FA_SHAPES:
        tensors.append(torch.zeros(N_FA, k_in, rank, device="cuda", dtype=torch.bfloat16))
        tensors.append(torch.zeros(N_FA, rank, k_out, device="cuda", dtype=torch.bfloat16))
    for _, k_in, k_out in DN_SHAPES:
        tensors.append(torch.zeros(N_DN, k_in, rank, device="cuda", dtype=torch.bfloat16))
        tensors.append(torch.zeros(N_DN, rank, k_out, device="cuda", dtype=torch.bfloat16))
    return tensors


def run_train_mega(weights, tokens, lora_tensors, lora_rank, lora_scaling):
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
        lora_tensors,
        lora_rank, lora_scaling, lora_h_ws,
        fa_k_cache, fa_v_cache, dn_states, conv_bufs,
        hidden, residual, normalized,
        proj_buf, proj_buf2, attn_buf, mlp_buf, dn_out_buf,
        beta_buf, alpha_buf,
        final_normed,
        lm_bmv, lm_bmi,
        MAX_SEQ_LEN,
    )
    return int(out_token.item())


def main():
    torch.manual_seed(42)
    S = 32
    LORA_R = 16
    w = random_weights()
    tokens = torch.randint(0, VOCAB, (S,), dtype=torch.int32, device="cuda")

    base = PrefillMegakernel(w).forward(tokens)

    # Zero LoRA everywhere
    zeros = zero_lora_tensors(LORA_R)
    a = run_train_mega(w, tokens, zeros, LORA_R, 1.0)

    # Non-zero LoRA everywhere (boosted init so it moves argmax)
    nonzero = build_lora_tensors(LORA_R, init_scale=0.2)
    b = run_train_mega(w, tokens, nonzero, LORA_R, 1.0)

    print(f"baseline prefill_megakernel         next_token = {base}")
    print(f"train_mega A=B=0  (all 13 linears)  next_token = {a}  (expect == baseline)")
    print(f"train_mega A,B≠0 (all 13 linears)   next_token = {b}  (expect ≠ baseline)")

    ok = True
    if a != base:
        print("  FAIL: zero LoRA diverges from baseline"); ok = False
    if b == base:
        print("  FAIL: nonzero LoRA matches baseline (LoRA not applied?)"); ok = False
    print("OK" if ok else "CORRECTNESS FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
