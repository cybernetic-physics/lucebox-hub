"""Direct speed comparison: BF16 megakernel decode vs NVFP4 megakernel
decode, after the NVFP4 weight cache has been populated.

Requires:
    $HF_HOME/qwen3x_nvfp4_27b_cache.pt exists (run
    test/test_s1e_nvfp4_vs_hf.py first to populate it).

Both decoders share the same HF model (no extra weight allocation).

Expected: NVFP4 decode ~3.5× faster than BF16 (50 GB → 14 GB HBM read).

Run:
    HF_HOME=/home/sparkz/rl/.hf_cache \
        /home/sparkz/rl/.venv/bin/python3 \
        models/qwen36_27b/bench_nvfp4_vs_bf16.py
"""
from __future__ import annotations
import argparse, os, sys, time
import torch

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)


def _bench(dec, ids, n_decode):
    """Measure prefill + n_decode tokens. Returns dict of timings."""
    # Warmup.
    dec.reset(); dec.prefill_via_hf(ids); torch.cuda.synchronize()
    for _ in range(2):
        nid = dec._argmax_from_normalized()
        for _ in range(4): nid = dec.decode(nid)
    torch.cuda.synchronize()

    # Prefill (via HF; this is the same op for both decoders since HF
    # weights are BF16 — comparison only meaningful on decode).
    dec.reset()
    t0 = time.perf_counter()
    dec.prefill_via_hf(ids); torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t0

    # Decode.
    nid = dec._argmax_from_normalized()
    t0 = time.perf_counter()
    for _ in range(n_decode):
        nid = dec.decode(nid)
    torch.cuda.synchronize()
    t_decode = time.perf_counter() - t0

    return dict(prefill_ms=t_prefill * 1000,
                decode_ms=t_decode * 1000,
                tok_per_s=n_decode / t_decode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--decode-tokens", type=int, default=32)
    args = ap.parse_args()

    print("Loading HF Qwen3.6-27B...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B",
                                          trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True).eval()

    from runtime_megakernel import Qwen36MegakernelDecoder
    ids = tok(args.prompt, return_tensors="pt").input_ids[0].to(torch.int32).cuda()

    print("\n[BF16] building decoder...")
    bf16 = Qwen36MegakernelDecoder(max_seq=128, verbose=False,
                                     hf_model=hf, tokenizer=tok, backend="bf16")
    print("[BF16] benching...")
    r_bf16 = _bench(bf16, ids, args.decode_tokens)
    print(f"  prefill: {r_bf16['prefill_ms']:.0f} ms  "
          f"decode {args.decode_tokens} tok: {r_bf16['decode_ms']:.0f} ms  "
          f"({r_bf16['tok_per_s']:.2f} tok/s)")

    print("\n[NVFP4] building decoder (loads quantized weight cache)...")
    nvfp4 = Qwen36MegakernelDecoder(max_seq=128, verbose=False,
                                      hf_model=hf, tokenizer=tok, backend="nvfp4")
    print("[NVFP4] benching...")
    r_nvfp4 = _bench(nvfp4, ids, args.decode_tokens)
    print(f"  prefill: {r_nvfp4['prefill_ms']:.0f} ms  "
          f"decode {args.decode_tokens} tok: {r_nvfp4['decode_ms']:.0f} ms  "
          f"({r_nvfp4['tok_per_s']:.2f} tok/s)")

    print(f"\nDecode speedup NVFP4 vs BF16: "
          f"{r_bf16['decode_ms']/r_nvfp4['decode_ms']:.2f}x")


if __name__ == "__main__":
    main()
