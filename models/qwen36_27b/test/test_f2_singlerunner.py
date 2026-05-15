"""F2 — Multi-turn KV reuse on REAL HF weights, single-decoder version.

Avoids the pre-existing multi-decoder corruption bug by using ONE
Qwen36MegakernelDecoder. Runs:
  - prefill(ids[:S1])
  - prefill(ids[S1:], start_position=S1) -> capture split logits
  - reset()
  - prefill(ids[:])                       -> capture oneshot logits
  - compare.
"""
from __future__ import annotations
import os, sys
import torch
import torch.nn.functional as F

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True).eval()

    from runtime_megakernel import Qwen36MegakernelDecoder

    prompt = "Hello, my name is Bob. I work as a software engineer at"
    ids = tok(prompt, return_tensors="pt").input_ids[0].to(torch.int32).cuda()
    S = ids.numel()
    S1 = S // 2
    print(f"\nprompt: {prompt!r}  ({S} tokens; split {S1} + {S-S1})\n")

    dec = Qwen36MegakernelDecoder(max_seq=64, verbose=False, hf_model=hf, tokenizer=tok)

    # Path B (split) FIRST.
    dec.prefill(ids[:S1])
    dec.prefill(ids[S1:], start_position=S1)
    torch.cuda.synchronize()
    logits_b = dec.sc.g_normalized.float().cpu().clone()
    top1_b = int(logits_b.argmax().item())

    # Reset, then Path A (one-shot).
    dec.reset()
    dec.prefill(ids)
    torch.cuda.synchronize()
    logits_a = dec.sc.g_normalized.float().cpu().clone()
    top1_a = int(logits_a.argmax().item())

    cos = F.cosine_similarity(logits_a.unsqueeze(0), logits_b.unsqueeze(0), dim=-1).item()
    max_abs = (logits_a - logits_b).abs().max().item()

    print(f"  one-shot:    top-1 = {top1_a:>6}  ({tok.decode([top1_a])!r})")
    print(f"  split + KV:  top-1 = {top1_b:>6}  ({tok.decode([top1_b])!r})")
    print(f"  cos = {cos:.6f}  max_abs = {max_abs:.4f}")

    if top1_a == top1_b and cos > 0.99:
        print("\nPASS  F2 (single decoder)")
    else:
        print(f"\nFAIL  split diverges")
        sys.exit(1)


if __name__ == "__main__":
    main()
