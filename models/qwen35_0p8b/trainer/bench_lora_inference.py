"""LoRA inference benchmark: our cuBLAS+graph prefill vs HF transformers.

Loads real Qwen3.5-0.8B weights, wraps the 13 trainable projections on
HF with random rank-R LoRA adapters, and packs the SAME A/B tensors
into our kernel's per-type-layer-major layout. Then times ms/step for a
fixed-length prefill on both sides.

Invariants we verify before benching:
- HF model on cuda, bfloat16
- LoRA A/B identical between both paths (same seed, same tensors)
- both runs warm up a few iterations before measuring

Output: ms/step for each side, and ms-ratio (our / HF — higher is
better for us ⇒ we're faster).
"""
from __future__ import annotations

import argparse
import sys
import time
import torch

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b")
sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/prefill_megakernel")
sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")

import qwen35_megakernel_bf16_C  # noqa: F401  load ops

# prefill_megakernel/model.py has the kernel-side constants that
# test_lora_forward/build_lora_tensors depend on (HIDDEN, INTER, VOCAB, ...).
from model import (
    NUM_LAYERS, HIDDEN, LAYER_TYPE, MAX_SEQ_LEN,
    FA_HEAD_DIM, FA_KV_SIZE, FA_KV_HEADS, FA_QPROJ_SIZE, FA_Q_SIZE,
    DN_CONV_CH, DN_CONV_K, DN_KEY, DN_HEADS,
    DN_VAL, DN_V_SIZE, INTER, VOCAB,
)
from test_lora_forward import build_lora_tensors, _pack_layer_weights
from lora_hf_wrap import wrap_hf_with_lora

# Separate module for the real-weights loader (outer model.py).
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "qwen_outer_model",
    "/root/lucebox-hub-b200-train/models/qwen35_0p8b/model.py",
)
_outer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_outer)
load_weights = _outer.load_weights


N_FA = sum(1 for t in LAYER_TYPE if t == 1)
N_DN = sum(1 for t in LAYER_TYPE if t == 0)


def alloc_our_scratch(S: int, lora_rank: int):
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    f32 = dict(dtype=torch.float32, device="cuda")
    i32 = dict(dtype=torch.int32, device="cuda")
    max_proj = max(FA_QPROJ_SIZE, DN_CONV_CH, INTER)
    max_attn = max(FA_Q_SIZE, DN_V_SIZE, FA_KV_SIZE)
    return dict(
        fa_k_cache=torch.zeros(N_FA, FA_KV_HEADS, 2048, FA_HEAD_DIM, **bf16),
        fa_v_cache=torch.zeros(N_FA, FA_KV_HEADS, 2048, FA_HEAD_DIM, **bf16),
        dn_states=torch.zeros(N_DN, DN_HEADS, DN_KEY, DN_VAL, **f32),
        conv_bufs=torch.zeros(N_DN, DN_CONV_CH, DN_CONV_K, **f32),
        hidden=torch.empty(S * HIDDEN, **bf16),
        residual=torch.empty(S * HIDDEN, **bf16),
        normalized=torch.empty(S * HIDDEN, **bf16),
        proj_buf=torch.empty(S * max_proj, **bf16),
        proj_buf2=torch.empty(S * max_proj, **bf16),
        attn_buf=torch.empty(S * max_attn, **bf16),
        mlp_buf=torch.empty(S * INTER, **bf16),
        dn_out_buf=torch.empty(S * max_attn, **bf16),
        beta_buf=torch.empty(S * DN_HEADS, **f32),
        alpha_buf=torch.empty(S * DN_HEADS, **f32),
        final_normed=torch.empty(HIDDEN, **bf16),
        hidden_bf16_out=torch.empty(HIDDEN, **bf16),
        out_token=torch.empty(1, **i32),
        lm_bmv=torch.empty(1024, **f32),
        lm_bmi=torch.empty(1024, **i32),
        lora_h_ws=torch.empty(S, lora_rank, **bf16),
    )


def run_our_prefill(weights, layers_packed, tokens, sc, lora_tensors,
                    lora_rank, lora_scaling):
    torch.ops.qwen35_megakernel_bf16_C.prefill_bf16_with_lora(
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
    )


def bench(fn, warmup=3, runs=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[128, 256, 512, 1024])
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--scaling", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=10)
    args = ap.parse_args()

    torch.manual_seed(0)
    LORA_R = args.rank

    # ===== Load real Qwen3.5-0.8B weights for our kernel =====
    print(f"Loading Qwen3.5-0.8B weights for our kernel (rank={LORA_R})...")
    weights, tokenizer = load_weights("Qwen/Qwen3.5-0.8B", verbose=False, backend="bf16")
    layers_packed = _pack_layer_weights(weights["layer_data"])

    # ===== Load HF model (second copy in bf16 on cuda) =====
    print("Loading HF Qwen3.5-0.8B for reference path...")
    from transformers import AutoModelForCausalLM
    hf_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16
    ).to("cuda")
    hf_model.eval()

    # Wrap 13 projections with LoRA using the SAME A/B tensors.
    torch.manual_seed(42)
    lora_tensors = build_lora_tensors(LORA_R, init_scale=0.02)
    wrap_hf_with_lora(hf_model, lora_tensors, scaling=args.scaling)
    hf_model.to("cuda", dtype=torch.bfloat16)

    print()
    print(f"Note: HF side uses torch-native DeltaNet (fla not installed).")
    print(f"      This is the default PyTorch path for Qwen3.5-0.8B.")
    print()
    print(f"{'S':>6} | {'HF PyTorch':>16} | {'ours (cuBLAS+graph)':>22} | {'speedup':>8}")
    print(f"{'-'*6}-+-{'-'*16}-+-{'-'*22}-+-{'-'*8}")

    for S in args.seq_lens:
        sc = alloc_our_scratch(S, LORA_R)
        tokens = torch.randint(0, VOCAB, (S,), dtype=torch.int32, device="cuda")
        input_ids = tokens.long().unsqueeze(0)

        @torch.no_grad()
        def hf_fwd():
            _ = hf_model(input_ids=input_ids, use_cache=False)

        def our_fwd():
            run_our_prefill(weights, layers_packed, tokens, sc,
                            lora_tensors, LORA_R, args.scaling)

        hf_ms = bench(hf_fwd, warmup=args.warmup, runs=args.runs)
        our_ms = bench(our_fwd, warmup=args.warmup, runs=args.runs)
        speedup = hf_ms / our_ms
        print(f"{S:>6} | {hf_ms:>8.2f} ms {S/(hf_ms/1000.0):>6,.0f}t/s "
              f"| {our_ms:>8.2f} ms {S/(our_ms/1000.0):>9,.0f}t/s "
              f"| {speedup:>6.1f}x")


if __name__ == "__main__":
    main()
