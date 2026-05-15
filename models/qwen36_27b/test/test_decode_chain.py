"""Verifies decode_chain (device-side token id chain) produces the same
output as the sequential decode loop, and reports the wall-clock
difference.

  - Loop: prefill, then N × (decode + .item() sync + next decode)
  - Chain: prefill, then ONE decode_chain(N) that stays on-device

Run:
    HF_HOME=/home/sparkz/rl/.hf_cache \
        /home/sparkz/rl/.venv/bin/python3 \
        models/qwen36_27b/test/test_decode_chain.py
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
    N = 32  # decode steps

    # Loop path.
    print(f"\n[loop] prefill + {N} × decode with .item() sync per token...")
    dec_loop = Qwen36MegakernelDecoder(max_seq=128, verbose=False,
                                         hf_model=hf, tokenizer=tok)
    dec_loop.prefill(ids); torch.cuda.synchronize()
    t0 = time.perf_counter()
    seq_loop = []
    nid = dec_loop._argmax_from_normalized()  # last-step argmax from prefill
    for _ in range(N):
        nid = dec_loop.decode(nid); seq_loop.append(nid)
    torch.cuda.synchronize()
    t_loop = time.perf_counter() - t0
    print(f"  loop wall: {t_loop*1000:.0f} ms  ({N/t_loop:.2f} tok/s)")

    # Chain path.
    print(f"\n[chain] prefill + decode_chain({N}) staying on device...")
    dec_chain = Qwen36MegakernelDecoder(max_seq=128, verbose=False,
                                          hf_model=hf, tokenizer=tok)
    dec_chain.prefill(ids); torch.cuda.synchronize()
    t0 = time.perf_counter()
    seq_chain = dec_chain.decode_chain(N).tolist()
    torch.cuda.synchronize()
    t_chain = time.perf_counter() - t0
    print(f"  chain wall: {t_chain*1000:.0f} ms  ({N/t_chain:.2f} tok/s)")

    print(f"\nspeedup: {t_loop/t_chain:.2f}x")
    if seq_loop == seq_chain:
        print(f"PASS  outputs match: {tok.decode(seq_chain)!r}")
    else:
        for i, (a, b) in enumerate(zip(seq_loop, seq_chain)):
            if a != b:
                print(f"FAIL  divergence at step {i}: loop={a} chain={b}")
                break
        sys.exit(1)


if __name__ == "__main__":
    main()
