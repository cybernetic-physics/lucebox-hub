"""T1 — Training memory audit for Qwen3.6-27B at various S, with
mitigation options.

The 0.8B model uses ~14 GB at S=32k for one training-step forward. At
27B the activation slabs alone are ~5× wider × ~2.5× more layers ×
same S → ~~300 GB naive. GB10 has 121 GB. Either gradient checkpoint
or NVFP4-pack activations or sequence-chunk training to fit.

Numbers below are arithmetic estimates from the 27B config, no model
load required.

Run:
    python3 training_memory_audit.py
"""
from __future__ import annotations
import argparse, json, os, sys

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from weight_packer import (
    NUM_LAYERS, HIDDEN_SIZE, INTERMEDIATE_SIZE,
    FA_NUM_Q_HEADS, FA_NUM_KV_HEADS, FA_HEAD_DIM,
    DN_NUM_V_HEADS, DN_HEAD_DIM, N_FA,
    DN_V_SIZE, FA_Q_SIZE, FA_KV_SIZE,
)


def gb(b): return b / (1024 ** 3)


def estimate(S: int, lora_rank: int = 16) -> dict:
    """Total bytes for one training-step forward at sequence length S."""
    # Weights (frozen if LoRA, full grads if not).
    weights_bf16 = 27e9 * 2  # 50 GB

    # Inference scratch (per-step, S-independent except for some).
    fa_kv_cache_bf16 = 2 * N_FA * FA_NUM_KV_HEADS * S * FA_HEAD_DIM * 2  # K + V
    dn_states_f32 = (NUM_LAYERS - N_FA) * DN_NUM_V_HEADS * DN_HEAD_DIM * DN_HEAD_DIM * 4
    per_token_scratch = (HIDDEN_SIZE + 2 * FA_Q_SIZE + INTERMEDIATE_SIZE) * 2  # rough

    # Activation saves (NUM_LAYERS slabs * S * dim).
    # 4 slabs are saved: hidden_in, normalized_in, normalized_post_attn,
    # mlp_inter. First 3 are HIDDEN; mlp_inter is INTERMEDIATE.
    act_hidden_3 = 3 * NUM_LAYERS * S * HIDDEN_SIZE * 2
    act_inter = NUM_LAYERS * S * INTERMEDIATE_SIZE * 2
    # Plus FA/DN-specific bwd saves at ~8 bf16-elements per token per layer
    act_extra = NUM_LAYERS * S * 2048 * 2  # crude bound

    inference = fa_kv_cache_bf16 + dn_states_f32 + per_token_scratch
    activations = act_hidden_3 + act_inter + act_extra

    # LoRA params + Adam state (m, v fp32 + grad fp32 + bf16 master).
    lora_params = 13 * NUM_LAYERS * 2 * lora_rank * max(HIDDEN_SIZE, INTERMEDIATE_SIZE) * 2
    lora_adam = lora_params * (4 + 4) / 2  # m + v as fp32 of params

    # Full-train grads + Adam (if training base weights).
    full_grads_fp32 = 27e9 * 4
    full_adam_fp32 = 27e9 * 8

    return dict(
        S=S,
        weights_bf16_gb       = gb(weights_bf16),
        inference_scratch_gb  = gb(inference),
        activation_saves_gb   = gb(activations),
        lora_params_gb        = gb(lora_params),
        lora_adam_gb          = gb(lora_adam),
        # Roll-ups for common configs.
        total_inference_gb    = gb(weights_bf16 + inference),
        total_lora_train_gb   = gb(weights_bf16 + inference + activations
                                    + lora_params + lora_adam),
        total_full_train_gb   = gb(weights_bf16 + inference + activations
                                    + full_grads_fp32 + full_adam_fp32),

        # Mitigations: NVFP4 weights + NVFP4 activations both ~3.5x.
        nvfp4_weights_gb      = gb(weights_bf16 / 3.5),
        nvfp4_activations_gb  = gb(activations / 3.5),
        # Gradient checkpointing keeps only one layer's activations.
        grad_ckpt_activations_gb = gb(activations / NUM_LAYERS),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-lens", type=int, nargs="+",
                    default=[1024, 4096, 8192, 16384, 32768, 65536])
    args = ap.parse_args()

    GB10_LIMIT = 121.0
    print(f"\nTraining memory budget on GB10 (unified memory limit: {GB10_LIMIT} GB)")
    print(f"Model: Qwen3.6-27B  (NUM_LAYERS={NUM_LAYERS}, "
          f"HIDDEN={HIDDEN_SIZE}, INTERMEDIATE={INTERMEDIATE_SIZE})\n")

    headers = ["S", "weights", "inference", "activations", "LoRA train", "full train",
               "+NVFP4 acts", "+grad ckpt"]
    print(f"{headers[0]:>6}  {headers[1]:>8}  {headers[2]:>10}  "
          f"{headers[3]:>11}  {headers[4]:>11}  {headers[5]:>11}  "
          f"{headers[6]:>11}  {headers[7]:>11}")
    print(f"{'':>6}  {'(GB)':>8}  {'(GB)':>10}  {'(GB)':>11}  "
          f"{'(GB)':>11}  {'(GB)':>11}  {'(GB)':>11}  {'(GB)':>11}")

    for S in args.seq_lens:
        e = estimate(S)
        # LoRA train with NVFP4 activation saves.
        nvfp4_lora = (e["weights_bf16_gb"] + e["inference_scratch_gb"]
                      + e["nvfp4_activations_gb"] + e["lora_params_gb"]
                      + e["lora_adam_gb"])
        # LoRA train with gradient checkpointing (recompute).
        grad_ckpt_lora = (e["weights_bf16_gb"] + e["inference_scratch_gb"]
                          + e["grad_ckpt_activations_gb"] + e["lora_params_gb"]
                          + e["lora_adam_gb"])
        marker_lora = "*" if e["total_lora_train_gb"] > GB10_LIMIT else " "
        marker_full = "*" if e["total_full_train_gb"] > GB10_LIMIT else " "
        marker_nvfp4 = "*" if nvfp4_lora > GB10_LIMIT else " "
        marker_ckpt = "*" if grad_ckpt_lora > GB10_LIMIT else " "
        print(f"{S:>6}  {e['weights_bf16_gb']:>8.1f}  "
              f"{e['inference_scratch_gb']:>10.2f}  "
              f"{e['activation_saves_gb']:>11.1f}  "
              f"{e['total_lora_train_gb']:>10.1f}{marker_lora}  "
              f"{e['total_full_train_gb']:>10.1f}{marker_full}  "
              f"{nvfp4_lora:>10.1f}{marker_nvfp4}  "
              f"{grad_ckpt_lora:>10.1f}{marker_ckpt}")
    print(f"\n  * = exceeds GB10 unified limit ({GB10_LIMIT} GB)")

    print("""
Mitigation strategies for S>=8192 training:

1. NVFP4 activation saves (~3.5x reduction)
   - Reuse models/qwen35_0p8b/nvfp4_kv.cuh helpers for the slabs.
   - Activation slabs already follow [NUM_LAYERS, S, dim] layout — they
     are tensor-flat and can be packed with the same kernel.
   - Estimated effort: 2-3 days.

2. Gradient checkpointing (recompute, ~NUM_LAYERS-x reduction)
   - Save only `hidden_in` per layer; recompute the rest on backward.
   - Adds ~1 FA + ~3 DN forwards per backward (matching the layer count).
   - Effort: 3-5 days (kernel-side recompute paths).

3. Sequence chunked training
   - Split S into chunks of S_chunk = 4096; train each chunk
     independently with the same params, accumulate grads.
   - Per-chunk activation memory is 8x smaller for S=32k.
   - Doesn't change kernel; just trainer-side scheduling.
   - Effort: 1-2 days.

Recommended combination for S=32768 LoRA training on GB10:
   NVFP4 weights (W ~15 GB) + NVFP4 activations + LoRA-only grads
   = ~30 GB total. Fits with headroom.
""")


if __name__ == "__main__":
    main()
