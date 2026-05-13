"""Hybrid test: BF16 prefill (KV cache pre-warmed correctly) + NVFP4 decode
with the new cuBLASLt FP4 LM head. Checks whether the cuBLASLt LM head
recovers the HF argmax when the input hidden state is the BF16-correct one.

If first-decoded token == " Paris" → cuBLASLt FP4 LM head is faithful;
the prior NVFP4 mismatch came from accumulated FP4 projection drift
across the prompt + decode steps. Then the fix is to drive the prompt
on BF16 weights and only the LM head on FP4.
"""
from __future__ import annotations

import sys

import torch


def main():
    sys.path.insert(0, "models/qwen35_0p8b")
    import qwen35_megakernel_bf16_C  # noqa: F401
    from model import (
        Decoder, MAX_SEQ_LEN, LAYER_TYPE, DN_NUM_HEADS, DN_KEY_DIM, DN_VALUE_DIM,
        DN_CONV_CHANNELS, DN_CONV_KERNEL, FA_NUM_KV_HEADS, FA_HEAD_DIM,
        _attach_nvfp4_weights, _pack_layer_weights_nvfp4, NVFP4_GROUP_SIZE,
        LM_HEAD_TENSORCORE_N, LM_HEAD_TENSORCORE_PACKED_BYTES,
        LM_HEAD_TENSORCORE_SCALE_BYTES, HIDDEN_SIZE, VOCAB_SIZE,
    )
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    prompt = "The capital of France is"
    prompt_ids = tok.encode(prompt, add_special_tokens=False)

    # 1) BF16 Decoder: prefill the prompt.
    print("Loading BF16 decoder...", flush=True)
    d = Decoder(model_name="Qwen/Qwen3.5-0.8B", backend="bf16", verbose=False)
    d.reset()
    bf16_first = int(d.prefill(prompt_ids))
    print(f"BF16 prefill -> first token: {bf16_first} "
          f"({tok.decode([bf16_first])!r})")

    # 2) Attach NVFP4 weights into the same weights dict.
    print("Attaching NVFP4 weights to the same model...", flush=True)
    _attach_nvfp4_weights(d._weights, group_size=NVFP4_GROUP_SIZE, verbose=False)
    nvfp4 = d._weights["nvfp4"]
    layer_weights_packed_nvfp4 = _pack_layer_weights_nvfp4(
        nvfp4["layer_data"], nvfp4["group_size"])
    lm_head_weight_packed = nvfp4["lm_head_weight_packed"]
    lm_head_scales = nvfp4["lm_head_scales"]

    # 3) Allocate cuBLASLt LM head scratch buffers.
    lm_hidden_bf16 = torch.empty(
        (LM_HEAD_TENSORCORE_N, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda")
    lm_hidden_packed = torch.empty(
        LM_HEAD_TENSORCORE_PACKED_BYTES, dtype=torch.uint8, device="cuda")
    lm_hidden_scales = torch.empty(
        LM_HEAD_TENSORCORE_SCALE_BYTES, dtype=torch.uint8, device="cuda")
    lm_logits_f16 = torch.empty(
        (LM_HEAD_TENSORCORE_N, VOCAB_SIZE), dtype=torch.float16, device="cuda")

    # 4) Use the BF16-prefilled state and call NVFP4 decode for the next token.
    # The KV cache is BF16 [N_FA, kv_heads, seq, head_dim] — exactly what
    # decode_nvfp4 expects. The DN state is f32 — same. Position is at len(prompt).
    print(f"Position after BF16 prefill: {d._position}", flush=True)
    print("Switching to NVFP4 decode for one step (cuBLASLt LM head)...", flush=True)
    ops = torch.ops.qwen35_megakernel_bf16_C
    ops.decode_nvfp4(
        d._out_token, bf16_first,
        d._embed_weight, layer_weights_packed_nvfp4,
        d._final_norm_weight, lm_head_weight_packed, lm_head_scales,
        lm_hidden_bf16, lm_hidden_packed, lm_hidden_scales, lm_logits_f16,
        d._fa_k_cache, d._fa_v_cache,
        d._dn_states, d._conv_bufs,
        d._hidden, d._activations, d._residual,
        d._qkv_scratch, d._kv_scratch, d._attn_out,
        d._mlp_inter, d._z_scratch, d._beta_scratch,
        d._alpha_scratch, d._normalized,
        d._barrier_counter, d._barrier_generation,
        d._block_max_vals, d._block_max_idxs,
        d._lm_sync_counter,
        d._position, MAX_SEQ_LEN, NVFP4_GROUP_SIZE,
    )
    out = int(d._out_token.item())
    print(f"NVFP4 hybrid decode -> next token: {out} ({tok.decode([out])!r})")

    # 5) Compare to HF baseline.
    from transformers import AutoModelForCausalLM
    print("\nLoading HF baseline for ground truth...", flush=True)
    hf = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16, device_map="cuda")
    hf.eval()
    with torch.inference_mode():
        # Predict the next token AFTER the prompt + bf16_first.
        input_ids = torch.tensor(
            [prompt_ids + [bf16_first]], device="cuda")
        o = hf(input_ids)
        hf_next = int(o.logits[:, -1].argmax(-1).item())
    print(f"HF eager -> next token after prefill+bf16_first: "
          f"{hf_next} ({tok.decode([hf_next])!r})")
    print(f"\nMatch: {out == hf_next}")


if __name__ == "__main__":
    main()
