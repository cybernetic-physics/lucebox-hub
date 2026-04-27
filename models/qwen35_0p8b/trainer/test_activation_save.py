"""Verify the new prefill_bf16_train_step correctly saves per-layer
activation slabs that a backward pass will consume.

Checks:
  1. next-token produced by prefill_bf16_train_step (with saves enabled)
     matches prefill_bf16_with_lora (baseline LoRA forward, no saves).
  2. saved.normalized_in[L] matches RMSnorm(saved.hidden_in[L], norm_w_L)
     to within bf16 tolerance — i.e. the checkpoint is consistent with
     the forward arithmetic.
  3. passing only some slabs (e.g. just normalized_in) works without
     clobbering anything.
"""
from __future__ import annotations

import sys
import torch

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b")
sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")
sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/prefill_megakernel")

import qwen35_megakernel_bf16_C  # noqa: F401
import train_megakernel_C         # noqa: F401

from model import (
    NUM_LAYERS, HIDDEN, VOCAB, LAYER_TYPE, N_FA, N_DN,
    FA_QPROJ_SIZE, FA_KV_SIZE, FA_HEAD_DIM, FA_KV_HEADS, FA_Q_SIZE,
    DN_CONV_CH, DN_V_SIZE, DN_HEADS, DN_VAL, DN_CONV_K, DN_KEY, INTER,
)
from smoke import random_weights
from test_lora_forward import (
    _pack_layer_weights, build_lora_tensors, zero_lora_tensors,
)


def _scratch(S, lora_rank):
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


def run_train_step(w, tokens, lora_tensors, lora_rank, lora_scaling, saves):
    """saves is a dict with keys hidden_in, normalized_in,
    normalized_post_attn, mlp_inter — any of which may be an empty
    tensor to skip that save."""
    S = int(tokens.shape[0])
    layers_packed = _pack_layer_weights(w["layer_data"])
    sc = _scratch(S, lora_rank)
    empty = torch.empty(0, dtype=torch.bfloat16, device="cuda")
    torch.ops.qwen35_megakernel_bf16_C.prefill_bf16_train_step(
        sc["out_token"], tokens.to(dtype=torch.int32, device="cuda").contiguous(),
        w["embed_weight"], layers_packed,
        w["final_norm_weight"], w["lm_head_weight"],
        sc["fa_k_cache"], sc["fa_v_cache"], sc["dn_states"], sc["conv_bufs"],
        sc["hidden"], sc["residual"], sc["normalized"],
        sc["proj_buf"], sc["proj_buf2"], sc["attn_buf"], sc["mlp_buf"],
        sc["dn_out_buf"], sc["beta_buf"], sc["alpha_buf"],
        sc["final_normed"], sc["hidden_bf16_out"],
        sc["lm_bmv"], sc["lm_bmi"],
        *lora_tensors,
        lora_rank, lora_scaling, sc["lora_h_ws"],
        saves.get("hidden_in", empty),
        saves.get("normalized_in", empty),
        saves.get("normalized_post_attn", empty),
        saves.get("mlp_inter", empty),
        saves.get("attn_out_pre_o", empty),
        saves.get("h_post_attn", empty),
    )
    return int(sc["out_token"].item())


def run_with_lora_baseline(w, tokens, lora_tensors, lora_rank, lora_scaling):
    """prefill_bf16_with_lora (no activation saves) — the reference."""
    S = int(tokens.shape[0])
    layers_packed = _pack_layer_weights(w["layer_data"])
    sc = _scratch(S, lora_rank)
    torch.ops.qwen35_megakernel_bf16_C.prefill_bf16_with_lora(
        sc["out_token"], tokens.to(dtype=torch.int32, device="cuda").contiguous(),
        w["embed_weight"], layers_packed,
        w["final_norm_weight"], w["lm_head_weight"],
        sc["fa_k_cache"], sc["fa_v_cache"], sc["dn_states"], sc["conv_bufs"],
        sc["hidden"], sc["residual"], sc["normalized"],
        sc["proj_buf"], sc["proj_buf2"], sc["attn_buf"], sc["mlp_buf"],
        sc["dn_out_buf"], sc["beta_buf"], sc["alpha_buf"],
        sc["final_normed"], sc["hidden_bf16_out"],
        sc["lm_bmv"], sc["lm_bmi"],
        *lora_tensors,
        lora_rank, lora_scaling, sc["lora_h_ws"],
    )
    return int(sc["out_token"].item())


def torch_rmsnorm(x_bf16, w_bf16, eps=1e-6):
    # bf16 × f32-compute × bf16-cast to match the kernel.
    # The kernel scales by (1.0 + w), per Qwen3.5 model convention.
    x = x_bf16.to(torch.float32)
    w = w_bf16.to(torch.float32)
    v = (x * x).mean(dim=-1, keepdim=True)
    r = torch.rsqrt(v + eps)
    return (x * r * (1.0 + w)).to(torch.bfloat16)


def layer_norm0_weight(w, li):
    """ptrs[0] is the input RMSnorm weight for layer li."""
    return w["layer_data"][li]["ptrs"][0]


def main():
    torch.manual_seed(42)
    S = 32
    LORA_R = 16
    w = random_weights()
    tokens = torch.randint(0, VOCAB, (S,), dtype=torch.int32, device="cuda")

    nonzero = build_lora_tensors(LORA_R, init_scale=0.1)

    # Baseline: no saves.
    baseline = run_with_lora_baseline(w, tokens, nonzero, LORA_R, 1.0)

    # Train step with all 4 saves enabled.
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    hidden_in_save        = torch.zeros(NUM_LAYERS, S, HIDDEN, **bf16)
    normalized_in_save    = torch.zeros(NUM_LAYERS, S, HIDDEN, **bf16)
    normalized_post_save  = torch.zeros(NUM_LAYERS, S, HIDDEN, **bf16)
    mlp_inter_save        = torch.zeros(NUM_LAYERS, S, INTER, **bf16)

    full_saves = {
        "hidden_in":            hidden_in_save,
        "normalized_in":        normalized_in_save,
        "normalized_post_attn": normalized_post_save,
        "mlp_inter":            mlp_inter_save,
    }
    tok_full = run_train_step(w, tokens, nonzero, LORA_R, 1.0, full_saves)
    assert tok_full == baseline, \
        f"save-on path diverged from baseline: {tok_full} vs {baseline}"

    # 2. Check normalized_in[L] == rmsnorm(hidden_in[L], norm_w_L)
    max_err = 0.0
    worst_li = -1
    for li in range(NUM_LAYERS):
        norm_w = layer_norm0_weight(w, li)        # bf16 [HIDDEN]
        recomputed = torch_rmsnorm(hidden_in_save[li], norm_w)
        diff = (normalized_in_save[li].to(torch.float32) -
                recomputed.to(torch.float32)).abs().max().item()
        if diff > max_err:
            max_err = diff
            worst_li = li
    print(f"normalized_in vs torch_rmsnorm max-abs-diff = {max_err:.4e} (layer {worst_li})")
    assert max_err < 5e-2, f"normalized_in activation-save mismatch too large: {max_err}"

    # 3. hidden_in[0] should equal the embed lookup on tokens.
    embed = w["embed_weight"]
    expected0 = embed[tokens.long()].to(torch.bfloat16)
    diff0 = (hidden_in_save[0].to(torch.float32) -
             expected0.to(torch.float32)).abs().max().item()
    print(f"hidden_in[0] vs embed(tokens)   max-abs-diff = {diff0:.4e}")
    assert diff0 < 1e-4, "hidden_in[0] should match embed(tokens) exactly (bf16 copy)"

    # 4. partial saves: only normalized_in enabled. Others untouched.
    empty = torch.empty(0, dtype=torch.bfloat16, device="cuda")
    partial_target = torch.zeros_like(normalized_in_save)
    sentinel = torch.full_like(mlp_inter_save, float("nan"))
    partial_saves = {
        "hidden_in":            empty,
        "normalized_in":        partial_target,
        "normalized_post_attn": empty,
        "mlp_inter":            sentinel,  # but passed empty via caller → wait, we should pass empty.
    }
    partial_saves["mlp_inter"] = empty
    tok_partial = run_train_step(w, tokens, nonzero, LORA_R, 1.0, partial_saves)
    assert tok_partial == baseline
    # The selectively-saved slab should match the all-saves run.
    diff_partial = (partial_target.to(torch.float32) -
                    normalized_in_save.to(torch.float32)).abs().max().item()
    print(f"partial-save normalized_in vs full-save max-abs-diff = {diff_partial:.4e}")
    assert diff_partial < 1e-5, "selective save-slab changed based on other slabs' presence"

    print("ACTIVATION-SAVE OK")


if __name__ == "__main__":
    main()
