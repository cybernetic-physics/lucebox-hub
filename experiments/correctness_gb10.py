"""Correctness check: megakernel vs HF eager on GB10 (gb10-train branch).

For each backend (bf16, nvfp4) we:
  1. Prefill the same prompt with the megakernel decoder.
  2. Drive both megakernel.step() and HF model(past_kv) greedily for N steps.
  3. Compare per-step argmax token IDs; report top-1 agreement and
     the first divergence index.

We also compare HF's prefill final-logit argmax against the megakernel's
prefill-output token, for a single-token sanity check.

Run from repo root:
  source /home/sparkz/rl/.venv/bin/activate
  export HF_HOME=/home/sparkz/rl/.hf_cache
  PYTHONPATH=models/qwen35_0p8b python experiments/correctness_gb10.py
"""
from __future__ import annotations

import argparse
import sys

import torch


def compare(prompt, gen_tokens, model_name, backends):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from model import Decoder

    tok = AutoTokenizer.from_pretrained(model_name)
    print(f"Loading HF baseline ({model_name})...", flush=True)
    hf = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda"
    )
    hf.eval()

    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    print(f"Prompt: {prompt!r} ({len(prompt_ids)} tokens)", flush=True)

    with torch.inference_mode():
        input_ids = torch.tensor([prompt_ids], device="cuda")
        out = hf(input_ids, use_cache=True)
        past = out.past_key_values
        hf_first = int(out.logits[:, -1].argmax(-1).item())
        hf_tokens = [hf_first]
        cur = torch.tensor([[hf_first]], device="cuda")
        for _ in range(gen_tokens - 1):
            o = hf(cur, past_key_values=past, use_cache=True)
            past = o.past_key_values
            cur = o.logits[:, -1:].argmax(-1)
            hf_tokens.append(int(cur.item()))

    print(f"  HF tokens: {hf_tokens[:20]}{'...' if len(hf_tokens) > 20 else ''}",
          flush=True)
    print(f"  HF text:   {tok.decode(hf_tokens, skip_special_tokens=True)!r}",
          flush=True)

    results = {}
    for backend in backends:
        print(f"\n=== Megakernel backend: {backend} ===", flush=True)
        decoder = Decoder(model_name=model_name, backend=backend, verbose=False)
        print(f"  Backend label: {decoder.backend_label}", flush=True)

        if backend in ("bf16", "bf16_fp4lm"):
            decoder.reset()
            mk_first = decoder.prefill(prompt_ids)
        else:
            decoder.reset()
            for tid in prompt_ids[:-1]:
                decoder.step(int(tid))
            mk_first = decoder.step(int(prompt_ids[-1]))

        mk_tokens = [int(mk_first)]
        cur_tok = int(mk_first)
        for _ in range(gen_tokens - 1):
            cur_tok = int(decoder.step(cur_tok))
            mk_tokens.append(cur_tok)

        match = sum(1 for a, b in zip(mk_tokens, hf_tokens) if a == b)
        first_div = next((i for i, (a, b) in enumerate(zip(mk_tokens, hf_tokens))
                          if a != b), None)
        results[backend] = {
            "mk_tokens": mk_tokens,
            "match": match,
            "first_div": first_div,
            "first_token_match": mk_first == hf_first,
        }
        print(f"  MK tokens: {mk_tokens[:20]}{'...' if len(mk_tokens) > 20 else ''}",
              flush=True)
        print(f"  MK text:   {tok.decode(mk_tokens, skip_special_tokens=True)!r}",
              flush=True)
        print(f"  Prefill first-token match: {results[backend]['first_token_match']}",
              flush=True)
        print(f"  Greedy top-1 agreement: {match}/{gen_tokens} = "
              f"{100.0*match/gen_tokens:.1f}%", flush=True)
        if first_div is None:
            print("  No divergence in the window.", flush=True)
        else:
            print(f"  First divergence at step {first_div}: "
                  f"MK={mk_tokens[first_div]} ({tok.decode([mk_tokens[first_div]])!r}) "
                  f"HF={hf_tokens[first_div]} ({tok.decode([hf_tokens[first_div]])!r})",
                  flush=True)

        del decoder
        torch.cuda.empty_cache()

    return hf_tokens, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--gen-tokens", type=int, default=32)
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--backends", nargs="+", default=["bf16", "nvfp4"])
    args = ap.parse_args()

    sys.path.insert(0, "models/qwen35_0p8b")
    import qwen35_megakernel_bf16_C  # noqa: F401

    hf_tokens, results = compare(args.prompt, args.gen_tokens,
                                 args.model_name, args.backends)

    print("\n== Summary ==", flush=True)
    print(f"{'backend':>8} | {'first match':>11} | {'top-1':>6} | {'first div':>9}")
    print("-" * 48)
    for b, r in results.items():
        fd = "—" if r["first_div"] is None else str(r["first_div"])
        print(f"{b:>8} | {str(r['first_token_match']):>11} | "
              f"{100.0*r['match']/args.gen_tokens:>5.1f}% | {fd:>9}")


if __name__ == "__main__":
    main()
