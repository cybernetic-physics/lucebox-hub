"""Compare the new cuBLAS+graph LoRA forward (prefill_bf16_with_lora) to
the existing one-dispatch train megakernel. Both should produce the same
next-token on the same (weights, tokens, LoRA A/B, rank, scaling).

Also prints a simple wall-time bench so we can see the cuBLAS path's
speedup over the hand-rolled megakernel matmuls.
"""
from __future__ import annotations

import sys
import time
import torch

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b")
sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")
sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/prefill_megakernel")

import qwen35_megakernel_bf16_C  # noqa: F401
import train_megakernel_C        # noqa: F401

from model import (
    NUM_LAYERS, HIDDEN, VOCAB, MAX_SEQ_LEN, N_FA, N_DN,
    FA_QPROJ_SIZE, FA_KV_SIZE, FA_HEAD_DIM, FA_KV_HEADS, FA_Q_SIZE,
    DN_CONV_CH, DN_V_SIZE, DN_HEADS, DN_VAL, DN_CONV_K, DN_KEY, INTER,
)
from smoke import random_weights
from test_lora_forward import (
    FA_SHAPES, DN_SHAPES,
    _pack_layer_weights, build_lora_tensors, zero_lora_tensors, run_train_mega,
)


def _pack_pf_layer_weights(layer_data):
    """PFLayerWeights layout matches trainer's LayerWeights (same 16+14*8 byte
    struct), so we reuse the trainer's packer.
    """
    return _pack_layer_weights(layer_data)


def run_cublas_lora(weights, tokens, lora_tensors, lora_rank, lora_scaling):
    """Invoke the new prefill_bf16_with_lora op on the cuBLAS+graph path."""
    S = int(tokens.shape[0])
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    f32 = dict(dtype=torch.float32, device="cuda")
    i32 = dict(dtype=torch.int32, device="cuda")

    layers_packed = _pack_pf_layer_weights(weights["layer_data"])

    fa_k_cache = torch.zeros(N_FA, FA_KV_HEADS, 32768, FA_HEAD_DIM, **bf16)
    fa_v_cache = torch.zeros_like(fa_k_cache)
    dn_states = torch.zeros(N_DN, DN_HEADS, DN_KEY, DN_VAL, **f32)
    conv_bufs = torch.zeros(N_DN, DN_CONV_CH, DN_CONV_K, **f32)

    max_proj = max(FA_QPROJ_SIZE, DN_CONV_CH, INTER)
    max_attn = max(FA_Q_SIZE, DN_V_SIZE, FA_KV_SIZE)
    hidden = torch.empty(S * HIDDEN, **bf16)
    residual = torch.empty(S * HIDDEN, **bf16)
    normalized = torch.empty(S * HIDDEN, **bf16)
    proj_buf = torch.empty(S * max_proj, **bf16)
    proj_buf2 = torch.empty(S * max_proj, **bf16)
    attn_buf = torch.empty(S * max_attn, **bf16)
    mlp_buf = torch.empty(S * INTER, **bf16)
    dn_out_buf = torch.empty(S * max_attn, **bf16)
    beta_buf = torch.empty(S * DN_HEADS, **f32)
    alpha_buf = torch.empty(S * DN_HEADS, **f32)
    final_normed = torch.empty(HIDDEN, **bf16)
    hidden_bf16_out = torch.empty(HIDDEN, **bf16)
    out_token = torch.empty(1, **i32)
    lm_bmv = torch.empty(1024, **f32)
    lm_bmi = torch.empty(1024, **i32)
    lora_h_ws = torch.empty(S, lora_rank, **bf16)

    torch.ops.qwen35_megakernel_bf16_C.prefill_bf16_with_lora(
        out_token, tokens.to(dtype=torch.int32, device="cuda").contiguous(),
        weights["embed_weight"], layers_packed,
        weights["final_norm_weight"], weights["lm_head_weight"],
        fa_k_cache, fa_v_cache, dn_states, conv_bufs,
        hidden, residual, normalized,
        proj_buf, proj_buf2, attn_buf, mlp_buf, dn_out_buf,
        beta_buf, alpha_buf,
        final_normed, hidden_bf16_out,
        lm_bmv, lm_bmi,
        *lora_tensors,                      # 26 LoRA A/B tensors
        lora_rank, lora_scaling, lora_h_ws,
    )
    return int(out_token.item())


def bench_forward(fn, n_warm=3, n_runs=20):
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / n_runs


def main():
    torch.manual_seed(42)
    S = 32
    LORA_R = 16
    w = random_weights()
    tokens = torch.randint(0, VOCAB, (S,), dtype=torch.int32, device="cuda")

    # LoRA disabled everywhere: both paths should match baseline prefill.
    zeros = zero_lora_tensors(LORA_R)
    mega_zero  = run_train_mega(w, tokens, zeros, LORA_R, 1.0)
    cublas_zero = run_cublas_lora(w, tokens, zeros, LORA_R, 1.0)
    print(f"LoRA=0  trainer_mega={mega_zero}  cublas={cublas_zero}")
    assert mega_zero == cublas_zero, "zero-LoRA cuBLAS vs trainer mismatch"

    # LoRA enabled with boosted weights.
    nonzero = build_lora_tensors(LORA_R, init_scale=0.2)
    mega_nz   = run_train_mega(w, tokens, nonzero, LORA_R, 1.0)
    cublas_nz = run_cublas_lora(w, tokens, nonzero, LORA_R, 1.0)
    print(f"LoRA≠0  trainer_mega={mega_nz}  cublas={cublas_nz}")
    assert mega_nz == cublas_nz, "nonzero-LoRA cuBLAS vs trainer mismatch"
    assert mega_nz != mega_zero, "LoRA appears to be a no-op on both paths"

    print("CORRECTNESS OK")

    # Wall-time bench at S=520 (the pp520 shape) with boosted LoRA active.
    S_BENCH = 520
    tokens_bench = torch.randint(0, VOCAB, (S_BENCH,), dtype=torch.int32, device="cuda")
    lora_bench = build_lora_tensors(LORA_R, init_scale=0.02)

    mega_ms = bench_forward(lambda: run_train_mega(w, tokens_bench, lora_bench, LORA_R, 1.0))
    cublas_ms = bench_forward(lambda: run_cublas_lora(w, tokens_bench, lora_bench, LORA_R, 1.0))
    print()
    print(f"  S={S_BENCH}, rank={LORA_R}, LoRA on all 13 linears")
    print(f"  trainer megakernel : {mega_ms:7.2f} ms/step")
    print(f"  cuBLAS+graph+LoRA  : {cublas_ms:7.2f} ms/step   ({mega_ms/cublas_ms:.2f}x faster)")


if __name__ == "__main__":
    main()
