"""Training-step bench for the lucebox megakernel at one S at a time.

Runs `prefill_bf16_train_step` (BF16 weights + LoRA rank=16) for a single
seq-len, prints step ms + peak memory + per-allocation breakdown, then
exits. Spawn one process per S to release memory between runs.

Usage:
    /home/sparkz/rl/.venv/bin/python3 trainer/bench_train_seq_len.py --S 4096
    /home/sparkz/rl/.venv/bin/python3 trainer/bench_train_seq_len.py --S 32768

Memory breakdown printed:
    fa_k_cache (BF16, hardcoded max_seq=32768) ~ 384 MB (K+V combined)
    activation saves: 4 slabs of [NUM_LAYERS, S, HIDDEN/INTER] bf16
    scratch:         per-S buffers (hidden/residual/proj/mlp/dn ...)
    model weights:   ~1.6 GB BF16 (Qwen3.5-0.8B)
    LoRA params:     ~13 projs * (HIDDEN + max_dim) * rank * bf16 (small)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR  = os.path.dirname(THIS_DIR)
sys.path.insert(0, PKG_DIR)
sys.path.insert(0, THIS_DIR)

import qwen35_megakernel_bf16_C  # noqa: F401

from model import (
    NUM_LAYERS, HIDDEN_SIZE as HIDDEN, LAYER_TYPE, INTERMEDIATE_SIZE as INTER,
    FA_HEAD_DIM, FA_NUM_KV_HEADS as FA_KV_HEADS,
    FA_QPROJ_SIZE, FA_KV_SIZE, FA_Q_SIZE,
    DN_NUM_HEADS as DN_HEADS, DN_KEY_DIM as DN_KEY, DN_VALUE_DIM as DN_VAL,
    DN_CONV_CHANNELS as DN_CONV_CH, DN_CONV_KERNEL as DN_CONV_K,
    DN_V_SIZE, VOCAB_SIZE as VOCAB,
    load_weights, _pack_layer_weights,
)

N_FA = sum(1 for t in LAYER_TYPE if t == 1)
N_DN = sum(1 for t in LAYER_TYPE if t == 0)


def mb(b): return b / (1024 ** 2)


def alloc_scratch(S, lora_rank, fa_max_seq=32768):
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    f32  = dict(dtype=torch.float32, device="cuda")
    i32  = dict(dtype=torch.int32, device="cuda")
    max_proj = max(FA_QPROJ_SIZE, DN_CONV_CH, INTER)
    max_attn = max(FA_Q_SIZE, DN_V_SIZE, FA_KV_SIZE)
    return dict(
        fa_k_cache=torch.zeros(N_FA, FA_KV_HEADS, fa_max_seq, FA_HEAD_DIM, **bf16),
        fa_v_cache=torch.zeros(N_FA, FA_KV_HEADS, fa_max_seq, FA_HEAD_DIM, **bf16),
        dn_states=torch.zeros(N_DN, DN_HEADS, DN_KEY, DN_VAL, **f32),
        conv_bufs=torch.zeros(N_DN, DN_CONV_CH, DN_CONV_K, **f32),
        hidden=torch.zeros(S * HIDDEN, **bf16),
        residual=torch.zeros(S * HIDDEN, **bf16),
        normalized=torch.zeros(S * HIDDEN, **bf16),
        proj_buf=torch.zeros(S * max_proj, **bf16),
        proj_buf2=torch.zeros(S * max_proj, **bf16),
        attn_buf=torch.zeros(S * max_attn, **bf16),
        mlp_buf=torch.zeros(S * INTER, **bf16),
        dn_out_buf=torch.zeros(S * max_attn, **bf16),
        beta_buf=torch.zeros(S * DN_HEADS, **f32),
        alpha_buf=torch.zeros(S * DN_HEADS, **f32),
        final_normed=torch.zeros(HIDDEN, **bf16),
        hidden_bf16_out=torch.zeros(HIDDEN, **bf16),
        out_token=torch.zeros(1, **i32),
        lm_bmv=torch.zeros(1024, **f32),
        lm_bmi=torch.zeros(1024, **i32),
        lora_h_ws=torch.zeros(S, lora_rank, **bf16),
    )


def alloc_activation_saves(S):
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    return dict(
        hidden_in=torch.zeros(NUM_LAYERS, S, HIDDEN, **bf16),
        normalized_in=torch.zeros(NUM_LAYERS, S, HIDDEN, **bf16),
        normalized_post_attn=torch.zeros(NUM_LAYERS, S, HIDDEN, **bf16),
        mlp_inter=torch.zeros(NUM_LAYERS, S, INTER, **bf16),
    )


def build_zero_lora(rank):
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    def z(shape): return torch.zeros(shape, **bf16)
    return [
        # FA per layer: q, k, v, o, gate, up, down  (A then B for each)
        z((N_FA, HIDDEN,    rank)), z((N_FA, rank, FA_QPROJ_SIZE)),  # fa_q
        z((N_FA, HIDDEN,    rank)), z((N_FA, rank, FA_KV_SIZE)),     # fa_k
        z((N_FA, HIDDEN,    rank)), z((N_FA, rank, FA_KV_SIZE)),     # fa_v
        z((N_FA, FA_Q_SIZE, rank)), z((N_FA, rank, HIDDEN)),         # fa_o
        z((N_FA, HIDDEN,    rank)), z((N_FA, rank, INTER)),          # fa_gate
        z((N_FA, HIDDEN,    rank)), z((N_FA, rank, INTER)),          # fa_up
        z((N_FA, INTER,     rank)), z((N_FA, rank, HIDDEN)),         # fa_down
        # DN per layer
        z((N_DN, HIDDEN,    rank)), z((N_DN, rank, DN_CONV_CH)),     # dn_qkv
        z((N_DN, HIDDEN,    rank)), z((N_DN, rank, DN_V_SIZE)),      # dn_z
        z((N_DN, DN_V_SIZE, rank)), z((N_DN, rank, HIDDEN)),         # dn_out
        z((N_DN, HIDDEN,    rank)), z((N_DN, rank, INTER)),          # dn_gate
        z((N_DN, HIDDEN,    rank)), z((N_DN, rank, INTER)),          # dn_up
        z((N_DN, INTER,     rank)), z((N_DN, rank, HIDDEN)),         # dn_down
    ]


def run_train_step(weights, layers_packed, tokens, sc, saves,
                   lora_tensors, lora_rank, lora_scaling):
    empty = torch.empty(0, dtype=torch.bfloat16, device="cuda")
    empty_f32 = torch.empty(0, dtype=torch.float32, device="cuda")
    torch.ops.qwen35_megakernel_bf16_C.prefill_bf16_train_step(
        sc["out_token"], tokens.to(dtype=torch.int32, device="cuda").contiguous(),
        weights["embed_weight"], layers_packed,
        weights["final_norm_weight"], weights["lm_head_weight"],
        sc["fa_k_cache"], sc["fa_v_cache"], sc["dn_states"], sc["conv_bufs"],
        sc["hidden"], sc["residual"], sc["normalized"],
        sc["proj_buf"], sc["proj_buf2"], sc["attn_buf"], sc["mlp_buf"],
        sc["dn_out_buf"], sc["beta_buf"], sc["alpha_buf"],
        sc["final_normed"], sc["hidden_bf16_out"],
        sc["lm_bmv"], sc["lm_bmi"],
        *lora_tensors,
        lora_rank, lora_scaling, sc["lora_h_ws"],
        saves["hidden_in"], saves["normalized_in"],
        saves["normalized_post_attn"], saves["mlp_inter"],
        empty, empty,                          # B.2 saves (unused)
        empty, empty, empty_f32,               # B.3b saves (unused)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--scaling", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--fa-max-seq", type=int, default=32768,
                    help="FA KV cache row count. Must be >= S.")
    args = ap.parse_args()

    assert args.S <= args.fa_max_seq, "S exceeds FA cache size"
    torch.manual_seed(0)
    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    free_before = torch.cuda.mem_get_info()[0]
    print(f"Device: {name}  cap={cap}")
    print(f"Free CUDA mem at start: {mb(free_before):.0f} MB")
    print(f"Config: S={args.S}  rank={args.rank}  fa_max_seq={args.fa_max_seq}")

    print("\nLoading Qwen3.5-0.8B weights (BF16)...", flush=True)
    t0 = time.perf_counter()
    weights, _tok = load_weights("Qwen/Qwen3.5-0.8B", verbose=False, backend="bf16")
    layers_packed = _pack_layer_weights(weights["layer_data"])
    print(f"  ...{time.perf_counter()-t0:.1f}s")

    after_weights = torch.cuda.memory_allocated()
    print(f"  GPU mem after weights:           {mb(after_weights):>8.1f} MB")

    print("\nAllocating scratch + saves...", flush=True)
    sc = alloc_scratch(args.S, args.rank, fa_max_seq=args.fa_max_seq)
    after_sc = torch.cuda.memory_allocated()
    print(f"  scratch (FA cache + per-S):      {mb(after_sc - after_weights):>8.1f} MB")

    saves = alloc_activation_saves(args.S)
    after_saves = torch.cuda.memory_allocated()
    print(f"  activation saves (NL*S*H slabs): {mb(after_saves - after_sc):>8.1f} MB")

    lora_tensors = build_zero_lora(args.rank)
    after_lora = torch.cuda.memory_allocated()
    print(f"  lora (rank={args.rank} zeros):           {mb(after_lora - after_saves):>8.1f} MB")

    tokens = torch.randint(0, VOCAB, (args.S,), dtype=torch.int32, device="cuda")
    after_tokens = torch.cuda.memory_allocated()
    print(f"  tokens [{args.S}]:                  {mb(after_tokens - after_lora):>8.1f} MB")

    total_alloc = after_tokens
    free_after_alloc = torch.cuda.mem_get_info()[0]
    print(f"\n  TOTAL allocated:                 {mb(total_alloc):>8.1f} MB")
    print(f"  CUDA free remaining:             {mb(free_after_alloc):>8.1f} MB")

    torch.cuda.reset_peak_memory_stats()
    print(f"\nWarmup x{args.warmup} ...", flush=True)
    for _ in range(args.warmup):
        run_train_step(weights, layers_packed, tokens, sc, saves,
                       lora_tensors, args.rank, args.scaling)
    torch.cuda.synchronize()

    print(f"Timed x{args.runs} ...", flush=True)
    t0 = time.perf_counter()
    for _ in range(args.runs):
        run_train_step(weights, layers_packed, tokens, sc, saves,
                       lora_tensors, args.rank, args.scaling)
    torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1000.0 / args.runs

    peak = torch.cuda.max_memory_allocated()
    print(f"\nRESULT  S={args.S:>6}  rank={args.rank}  "
          f"step={dt_ms:>8.2f} ms  tok/s={args.S * 1000.0 / dt_ms:>9.0f}  "
          f"peak_alloc={mb(peak):>8.1f} MB")


if __name__ == "__main__":
    main()
