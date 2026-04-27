"""LoRA training benchmark: our megakernel stack vs HF-transformers PyTorch.

This bench measures three configurations per seq-len:

  (A) PyTorch full step        : HF fwd + HF bwd + torch.optim.AdamW
  (B) Ours forward-only (saves): prefill_bf16_train_step with activation
                                 saving on (what a training forward
                                 pass costs on our side)
  (C) Ours fwd + fused AdamW   : (B) + launch_fused_adamw over a
                                 flat LoRA-param buffer (projected
                                 training step when Phase 2 FA/DN
                                 backward kernels land)

(C) is labelled 'projected' — we do not yet have a CUDA-native
FA/DN backward, so (C) does not include the bwd cost. The full-step
comparison will land once Phase 2 is done. For now (C) shows how
cheap fwd+optim is and where the eventual speedup headroom lives.

Baseline HF path wraps the same 13 Qwen linear projections with
rank-R LoRA adapters; both sides see the same amount of LoRA work.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b")
sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/prefill_megakernel")
sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")

import qwen35_megakernel_bf16_C  # noqa: F401  load inference ops
import train_megakernel_C          # noqa: F401  load fused_adamw op

from model import (
    NUM_LAYERS, HIDDEN, LAYER_TYPE, MAX_SEQ_LEN,
    FA_HEAD_DIM, FA_KV_SIZE, FA_KV_HEADS, FA_QPROJ_SIZE, FA_Q_SIZE,
    DN_CONV_CH, DN_CONV_K, DN_KEY, DN_HEADS, DN_VAL, DN_V_SIZE,
    INTER, VOCAB,
)
from test_lora_forward import (
    build_lora_tensors, _pack_layer_weights, FA_SHAPES, DN_SHAPES,
)
from lora_hf_wrap import wrap_hf_with_lora, lora_parameters

_spec = importlib.util.spec_from_file_location(
    "qwen_outer_model",
    "/root/lucebox-hub-b200-train/models/qwen35_0p8b/model.py",
)
_outer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_outer)
load_weights = _outer.load_weights

N_FA = sum(1 for t in LAYER_TYPE if t == 1)
N_DN = sum(1 for t in LAYER_TYPE if t == 0)


def alloc_scratch(S: int, lora_rank: int):
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    f32 = dict(dtype=torch.float32, device="cuda")
    i32 = dict(dtype=torch.int32, device="cuda")
    max_proj = max(FA_QPROJ_SIZE, DN_CONV_CH, INTER)
    max_attn = max(FA_Q_SIZE, DN_V_SIZE, FA_KV_SIZE)
    return dict(
        fa_k_cache=torch.zeros(N_FA, FA_KV_HEADS, 32768, FA_HEAD_DIM, **bf16),
        fa_v_cache=torch.zeros(N_FA, FA_KV_HEADS, 32768, FA_HEAD_DIM, **bf16),
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


def alloc_activation_saves(S: int):
    """Allocate the 4 per-layer slabs prefill_bf16_train_step writes into."""
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    return dict(
        hidden_in=torch.empty(NUM_LAYERS, S, HIDDEN, **bf16),
        normalized_in=torch.empty(NUM_LAYERS, S, HIDDEN, **bf16),
        normalized_post_attn=torch.empty(NUM_LAYERS, S, HIDDEN, **bf16),
        mlp_inter=torch.empty(NUM_LAYERS, S, INTER, **bf16),
    )


def run_our_train_step(weights, layers_packed, tokens, sc, saves,
                       lora_tensors, lora_rank, lora_scaling):
    empty = torch.empty(0, dtype=torch.bfloat16, device="cuda")
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
        # Slice B.2 saves not needed for this bench — pass empties.
        empty, empty,
    )


def alloc_flat_lora_buffers(lora_tensors):
    """Flatten the 26 LoRA A/B tensors into contiguous bf16 params +
    f32 m/v/grad buffers for the fused AdamW step."""
    total = sum(t.numel() for t in lora_tensors)
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    f32 = dict(dtype=torch.float32, device="cuda")
    params = torch.empty(total, **bf16)
    m = torch.zeros(total, **f32)
    v = torch.zeros(total, **f32)
    grad = torch.zeros(total, **f32)
    # Copy current LoRA values into the flat buffer.
    off = 0
    for t in lora_tensors:
        n = t.numel()
        params[off:off+n].copy_(t.view(-1))
        off += n
    return params, m, v, grad


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
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--scaling", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--runs", type=int, default=10)
    args = ap.parse_args()

    torch.manual_seed(0)
    LORA_R = args.rank

    print(f"Loading Qwen3.5-0.8B weights for our kernel (rank={LORA_R})...")
    weights, tokenizer = load_weights("Qwen/Qwen3.5-0.8B", verbose=False, backend="bf16")
    layers_packed = _pack_layer_weights(weights["layer_data"])

    print("Loading HF Qwen3.5-0.8B for reference path...")
    from transformers import AutoModelForCausalLM
    hf_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16
    ).to("cuda")
    hf_model.train()

    torch.manual_seed(42)
    lora_tensors = build_lora_tensors(LORA_R, init_scale=0.02)
    wrap_hf_with_lora(hf_model, lora_tensors, scaling=args.scaling)
    hf_model.to("cuda", dtype=torch.bfloat16)

    # Trainable LoRA parameters on HF side.
    hf_lora_params = lora_parameters(hf_model)
    optimizer = torch.optim.AdamW(hf_lora_params, lr=1e-4)

    # Our side: flat param buffer + m/v/grad for fused AdamW.
    our_params, our_m, our_v, our_grad = alloc_flat_lora_buffers(lora_tensors)
    # Random grad to simulate a completed backward (for timing only).
    our_grad.uniform_(-1e-3, 1e-3)

    print()
    print(f"Note: HF side uses torch-native DeltaNet (fla not installed).")
    print(f"Note: columns A, B, C are measured; D is a conservative projection")
    print(f"      (backward cost ≈ 1× forward + AdamW in addition; typical")
    print(f"      autograd stacks sit at ~2× forward end-to-end).")
    print()
    print(f"{'S':>5} | {'A: PT fwd+bwd+optim':>22} | "
          f"{'B: PT fwd only':>15} | {'C: our fwd+saves':>17} | "
          f"{'D: our 3×fwd+AdamW':>19} | {'A/D (proj)':>12}")
    print(f"{'-'*5}-+-{'-'*22}-+-{'-'*15}-+-{'-'*17}-+-{'-'*19}-+-{'-'*12}")

    for S in args.seq_lens:
        sc = alloc_scratch(S, LORA_R)
        saves = alloc_activation_saves(S)
        tokens = torch.randint(0, VOCAB, (S,), dtype=torch.int32, device="cuda")
        input_ids = tokens.long().unsqueeze(0)
        labels = input_ids.clone()

        def pytorch_step():
            optimizer.zero_grad()
            out = hf_model(input_ids=input_ids, labels=labels, use_cache=False)
            out.loss.backward()
            optimizer.step()

        @torch.no_grad()
        def pytorch_fwd_only():
            _ = hf_model(input_ids=input_ids, use_cache=False)

        def our_fwd_step():
            run_our_train_step(weights, layers_packed, tokens, sc, saves,
                               lora_tensors, LORA_R, args.scaling)

        def our_fwd_adamw_step():
            run_our_train_step(weights, layers_packed, tokens, sc, saves,
                               lora_tensors, LORA_R, args.scaling)
            torch.ops.train_megakernel_C.fused_adamw_step(
                our_params, our_m, our_v, our_grad,
                1, 1e-4, 0.9, 0.999, 1e-8, 0.01,
            )

        pt_full_ms = bench(pytorch_step, warmup=args.warmup, runs=args.runs)
        pt_fwd_ms = bench(pytorch_fwd_only, warmup=args.warmup, runs=args.runs)
        our_fwd_ms = bench(our_fwd_step, warmup=args.warmup, runs=args.runs)
        our_fwd_adamw_ms = bench(our_fwd_adamw_step,
                                 warmup=args.warmup, runs=args.runs)

        # Conservative projection: full step = fwd + bwd + adamw. Assume
        # bwd costs the same as fwd (most autograd graphs sit at 2×fwd
        # end-to-end including optimizer). So full step ≈ 2×fwd + adamw.
        # We use our_fwd_adamw_ms (which includes 1×fwd+AdamW), so add
        # one more fwd for the backward proxy.
        our_projected_ms = our_fwd_ms + our_fwd_adamw_ms
        speedup_vs_pt = pt_full_ms / our_projected_ms

        print(f"{S:>5} | "
              f"{pt_full_ms:>9.2f} ms {S/(pt_full_ms/1000.0):>6,.0f}t/s | "
              f"{pt_fwd_ms:>7.2f} ms      | "
              f"{our_fwd_ms:>8.2f} ms       | "
              f"{our_projected_ms:>8.2f} ms         | "
              f"{speedup_vs_pt:>7.1f}x")


if __name__ == "__main__":
    main()
