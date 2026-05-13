"""Hybrid: BF16 decode trunk + cuBLASLt FP4 LM head substitution.

For each step:
  1. Run decode_bf16 (correct f32 normalized written to _normalized).
  2. Run lm_head_nvfp4_from_f32 on _normalized -> FP4 argmax.

This tests whether the FP4 LM head alone preserves greedy parity with HF
when the layer projections stay BF16. If yes, this is the practical
"NVFP4 with no quality loss" path — the FP4 LM head saves memory for
the 248320-row lm_head weight (~250 MB BF16 -> ~125 MB FP4) and lets
us upgrade to mma.kind::mxf4 layer projections later without revisiting
the LM head.
"""
from __future__ import annotations

import sys

import torch


def main():
    sys.path.insert(0, "models/qwen35_0p8b")
    import qwen35_megakernel_bf16_C  # noqa: F401
    from model import (
        Decoder, MAX_SEQ_LEN, _attach_nvfp4_weights, NVFP4_GROUP_SIZE,
        LM_HEAD_TENSORCORE_N, LM_HEAD_TENSORCORE_PACKED_BYTES,
        LM_HEAD_TENSORCORE_SCALE_BYTES, HIDDEN_SIZE, VOCAB_SIZE,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    prompt = "The capital of France is"
    prompt_ids = tok.encode(prompt, add_special_tokens=False)

    print("Loading HF baseline...", flush=True)
    hf = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16, device_map="cuda")
    hf.eval()

    hf_tokens = []
    with torch.inference_mode():
        input_ids = torch.tensor([prompt_ids], device="cuda")
        o = hf(input_ids, use_cache=True)
        past = o.past_key_values
        cur = o.logits[:, -1:].argmax(-1)
        hf_tokens.append(int(cur.item()))
        for _ in range(31):
            o = hf(cur, past_key_values=past, use_cache=True)
            past = o.past_key_values
            cur = o.logits[:, -1:].argmax(-1)
            hf_tokens.append(int(cur.item()))
    print(f"HF tokens: {hf_tokens}")
    print(f"HF text:   {tok.decode(hf_tokens, skip_special_tokens=True)!r}")
    del hf
    torch.cuda.empty_cache()

    print("\nLoading BF16 megakernel decoder...", flush=True)
    d = Decoder(model_name="Qwen/Qwen3.5-0.8B", backend="bf16", verbose=False)
    print("Attaching NVFP4 LM head weights...", flush=True)
    _attach_nvfp4_weights(d._weights, group_size=NVFP4_GROUP_SIZE, verbose=False)
    nvfp4 = d._weights["nvfp4"]
    lm_head_packed = nvfp4["lm_head_weight_packed"]
    lm_head_scales = nvfp4["lm_head_scales"]

    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    lm_hidden_bf16 = torch.empty((LM_HEAD_TENSORCORE_N, HIDDEN_SIZE), **bf16)
    lm_hidden_packed = torch.empty(LM_HEAD_TENSORCORE_PACKED_BYTES, dtype=torch.uint8, device="cuda")
    lm_hidden_scales = torch.empty(LM_HEAD_TENSORCORE_SCALE_BYTES, dtype=torch.uint8, device="cuda")
    lm_logits_f16 = torch.empty((LM_HEAD_TENSORCORE_N, VOCAB_SIZE), dtype=torch.float16, device="cuda")
    fp4_out_token = torch.empty(1, dtype=torch.int32, device="cuda")

    print("Running hybrid: BF16 decode + FP4 LM head...", flush=True)
    d.reset()
    bf16_first = int(d.prefill(prompt_ids))
    # Substitute FP4 LM head on top of the prefill's f32-normalized hidden.
    # prefill_bf16 uses `_normalized` as f32 scratch but writes to `final_normed`
    # in BF16 at the end. We can re-run from the BF16 final hidden via the
    # standalone's "bf16_top1" variant later; for now, take the BF16 prefill
    # output and use decode_bf16 to step.
    print(f"BF16 prefill first token: {bf16_first} ({tok.decode([bf16_first])!r})")

    hybrid_tokens = [bf16_first]
    cur = bf16_first
    ops = torch.ops.qwen35_megakernel_bf16_C
    for i in range(31):
        # Step BF16: this writes the next f32 normalized into d._normalized.
        d.step(int(cur))
        # The BF16 step already wrote `d._out_token` with BF16's argmax.
        # We override by calling FP4 LM head on `_normalized` (which holds
        # the f32 normed hidden of the JUST decoded token's input — actually
        # this is the f32 hidden BEFORE the LM head, i.e. the right input.
        ops.lm_head_nvfp4_from_f32(
            fp4_out_token, d._normalized,
            lm_head_packed, lm_head_scales,
            lm_hidden_bf16, lm_hidden_packed, lm_hidden_scales, lm_logits_f16,
            d._block_max_vals, d._block_max_idxs,
            NVFP4_GROUP_SIZE,
        )
        cur = int(fp4_out_token.item())
        hybrid_tokens.append(cur)
    print(f"Hybrid tokens: {hybrid_tokens}")
    print(f"Hybrid text:   {tok.decode(hybrid_tokens, skip_special_tokens=True)!r}")

    match = sum(1 for a, b in zip(hybrid_tokens, hf_tokens) if a == b)
    first_div = next((i for i, (a, b) in enumerate(zip(hybrid_tokens, hf_tokens))
                      if a != b), None)
    print(f"\nGreedy top-1 vs HF: {match}/{len(hf_tokens)} = "
          f"{100.0*match/len(hf_tokens):.1f}%")
    if first_div is not None:
        print(f"First divergence at step {first_div}: "
              f"hybrid={hybrid_tokens[first_div]} ({tok.decode([hybrid_tokens[first_div]])!r}) "
              f"HF={hf_tokens[first_div]} ({tok.decode([hf_tokens[first_div]])!r})")


if __name__ == "__main__":
    main()
