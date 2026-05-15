"""Verifies prefill_via_hf produces the same downstream decode state
as our megakernel prefill, by comparing the first 8 generated tokens
on a short prompt.

If the KV / DN cache copy is correct, both decoders should generate
the same sequence (greedy).

Run:
    HF_HOME=/home/sparkz/rl/.hf_cache \
        /home/sparkz/rl/.venv/bin/python3 \
        models/qwen36_27b/test/test_prefill_via_hf.py
"""
from __future__ import annotations
import os, sys, time
import torch

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))


def main():
    print("Loading HF Qwen3.6-27B (BF16)...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True).eval()

    from runtime_megakernel import Qwen36MegakernelDecoder

    prompt = "The capital of France is"
    ids = tok(prompt, return_tensors="pt").input_ids[0].to(torch.int32).cuda()
    print(f"\nprompt: {prompt!r}  ({ids.numel()} tokens)")

    # Slow path: our prefill.
    print("\n[slow path] prefill_qwen3x_naive...")
    dec_slow = Qwen36MegakernelDecoder(max_seq=64, verbose=False,
                                         hf_model=hf, tokenizer=tok)
    t0 = time.perf_counter()
    next_id_slow = dec_slow.prefill(ids)
    torch.cuda.synchronize()
    t_slow_prefill = time.perf_counter() - t0
    print(f"  prefill: {t_slow_prefill*1000:.0f} ms  next token: {next_id_slow} "
          f"({tok.decode([next_id_slow])!r})")
    slow_tokens = [next_id_slow]
    t0 = time.perf_counter()
    for _ in range(7):
        nid = dec_slow.decode(slow_tokens[-1])
        slow_tokens.append(nid)
    torch.cuda.synchronize()
    t_slow_decode = time.perf_counter() - t0
    print(f"  decode 7 tokens: {t_slow_decode*1000:.0f} ms")
    print(f"  generated: {tok.decode(slow_tokens)!r}")

    # Fast path: prefill via HF.
    print("\n[fast path] prefill_via_hf...")
    dec_fast = Qwen36MegakernelDecoder(max_seq=64, verbose=False,
                                         hf_model=hf, tokenizer=tok)
    t0 = time.perf_counter()
    next_id_fast = dec_fast.prefill_via_hf(ids)
    torch.cuda.synchronize()
    t_fast_prefill = time.perf_counter() - t0
    print(f"  prefill: {t_fast_prefill*1000:.0f} ms  next token: {next_id_fast} "
          f"({tok.decode([next_id_fast])!r})")
    fast_tokens = [next_id_fast]
    t0 = time.perf_counter()
    for _ in range(7):
        nid = dec_fast.decode(fast_tokens[-1])
        fast_tokens.append(nid)
    torch.cuda.synchronize()
    t_fast_decode = time.perf_counter() - t0
    print(f"  decode 7 tokens: {t_fast_decode*1000:.0f} ms")
    print(f"  generated: {tok.decode(fast_tokens)!r}")

    print(f"\nspeedup ratio (prefill only): "
          f"{t_slow_prefill/t_fast_prefill:.1f}x")

    if slow_tokens == fast_tokens:
        print("PASS  HF-prefill + our-decode matches our-prefill + our-decode")
    else:
        # Show first mismatch.
        for i, (a, b) in enumerate(zip(slow_tokens, fast_tokens)):
            if a != b:
                print(f"FAIL  divergence at decode step {i}: "
                      f"slow={a} ({tok.decode([a])!r}) "
                      f"fast={b} ({tok.decode([b])!r})")
                break
        sys.exit(1)


if __name__ == "__main__":
    main()
