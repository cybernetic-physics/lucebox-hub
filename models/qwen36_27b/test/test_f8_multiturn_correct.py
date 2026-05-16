"""F8 — Multi-turn KV reuse correctness vs full-sequence prefill.

Verifies that prefilling a long sequence as two chunks via
`start_position` gives the same final-position logits as prefilling
the concatenated sequence in one shot. This catches bugs in the F2
KV-reuse path (start_position offset, cache write index, RoPE position).

  Path A: prefill(ids[:S1+S2])                 then read logits
  Path B: prefill(ids[:S1])
          prefill(ids[S1:S1+S2], start_position=S1)
          then read logits

Both should produce the same last-position logits (within fp32 noise).

Run:
    HF_HOME=/home/sparkz/rl/.hf_cache \
        /home/sparkz/rl/.venv/bin/python3 \
        models/qwen36_27b/test/test_f8_multiturn_correct.py
"""
from __future__ import annotations
import os, sys
import torch
import torch.nn.functional as F

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))


def main():
    print("Loading HF Qwen3.6-27B (BF16) for weight sharing...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B",
                                          trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True).eval()

    from runtime_megakernel import Qwen36MegakernelDecoder

    prompt = "Hello, my name is Bob. I work as a software engineer at"
    ids = tok(prompt, return_tensors="pt").input_ids[0].to(torch.int32).cuda()
    S = ids.numel()
    S1 = S // 2
    S2 = S - S1
    print(f"\nprompt: {prompt!r}  ({S} tokens; split {S1} + {S2})")

    # NOTE: this test uses ONE decoder and `reset()` between paths
    # because of the F10 bug (creating two decoders in the same
    # process makes the second one NaN — see TODO.md F10 and
    # test/debug_multi_decoder*.py). Single-decoder with reset()
    # exercises the same F2 logic — fa_k/v_cache, dn_state, conv_buf
    # are all cleared by reset() between paths.
    dec = Qwen36MegakernelDecoder(max_seq=256, verbose=False,
                                    hf_model=hf, tokenizer=tok)

    # Path B (split) FIRST so a passing run leaves the decoder in
    # the "interesting" state — useful if anyone is poking at it
    # interactively after the test.
    dec.prefill(ids[:S1])
    dec.prefill(ids[S1:], start_position=S1)
    torch.cuda.synchronize()
    logits_b = dec.logits_for_last().cpu()
    top1_b = int(logits_b.argmax().item())

    # Path A: one-shot prefill on fresh state.
    dec.reset()
    dec.prefill(ids); torch.cuda.synchronize()
    logits_a = dec.logits_for_last().cpu()
    top1_a = int(logits_a.argmax().item())

    cos = F.cosine_similarity(logits_a.unsqueeze(0).to(torch.float32),
                                logits_b.unsqueeze(0).to(torch.float32),
                                dim=-1).item()
    max_abs = (logits_a - logits_b).abs().max().item()

    print(f"\n  one-shot:    top-1 = {top1_a:>6}  ({tok.decode([top1_a])!r})")
    print(f"  split + KV:  top-1 = {top1_b:>6}  ({tok.decode([top1_b])!r})")
    print(f"  cos = {cos:.6f}  max_abs = {max_abs:.4f}")

    if top1_a == top1_b and cos > 0.99:
        print("\nPASS  multi-turn KV reuse produces same output as one-shot")
    else:
        print("\nFAIL  KV reuse diverges from one-shot prefill")
        sys.exit(1)


if __name__ == "__main__":
    main()
