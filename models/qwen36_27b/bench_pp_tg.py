"""P2 — Prefill / decode speed bench for Qwen3.6-27B megakernel.

Measures:
  - pp{S}: wall-time to run prefill_qwen3x_naive on S tokens (host-looped)
  - tg{N}: wall-time to decode N tokens after a fixed S-token prefix
  - per-step tok/s for each

Sweeps default: S ∈ {16, 64, 256, 1024}, gen=64 tokens. Big S values cost
S * 215 ms ≈ S/4.6 seconds under the host-loop prefill, so values past
1024 are slow — that's the S2 unblock.

Usage:
    HF_HOME=/home/sparkz/rl/.hf_cache \
        /home/sparkz/rl/.venv/bin/python3 \
        models/qwen36_27b/bench_pp_tg.py
"""
from __future__ import annotations
import argparse, json, os, sys, time
import torch

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-lens", type=int, nargs="+",
                    default=[16, 64, 256, 1024])
    p.add_argument("--gen-tokens", type=int, default=64)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--measure", type=int, default=3)
    p.add_argument("--max-seq", type=int, default=2048)
    p.add_argument("--backend", default="bf16", choices=("bf16", "nvfp4"))
    p.add_argument("--prefill-mode", default="naive",
                    choices=("naive", "hf"),
                    help="'naive' = host-loop megakernel (slow at S>256), "
                          "'hf' = HF batched forward + KV copy (~50x faster)")
    p.add_argument("--json", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading HF Qwen3.6-27B (backend={args.backend})...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True).eval()

    from runtime_megakernel import Qwen36MegakernelDecoder
    dec = Qwen36MegakernelDecoder(max_seq=args.max_seq, verbose=True,
                                    hf_model=hf, tokenizer=tok,
                                    backend=args.backend)

    # Use a long stretch of wikitext as the token source so prefill input is
    # in-distribution.
    from datasets import load_dataset
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = " ".join(x for x in wt["text"] if x.strip())[:200_000]
    base_ids = tok(text, return_tensors="pt").input_ids[0].to(torch.int32).cuda()

    results = []
    for S in args.seq_lens:
        if base_ids.numel() < S:
            print(f"skip S={S}: not enough tokens"); continue
        if S > args.max_seq - args.gen_tokens:
            print(f"skip S={S}: would exceed max_seq={args.max_seq}"); continue
        ids = base_ids[:S].contiguous()

        prefill_fn = (dec.prefill_via_hf if args.prefill_mode == "hf"
                      else dec.prefill)

        # Warmup.
        for _ in range(args.warmup):
            dec.reset(); prefill_fn(ids)
            torch.cuda.synchronize()

        # Measure prefill (pp).
        pp_times = []
        for _ in range(args.measure):
            dec.reset()
            t0 = time.perf_counter()
            next_id = prefill_fn(ids)
            torch.cuda.synchronize()
            pp_times.append(time.perf_counter() - t0)

        # Measure decode (tg) from the prefilled state.
        tg_times = []
        for _ in range(args.measure):
            dec.reset(); next_id = prefill_fn(ids); torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.gen_tokens):
                next_id = dec.decode(next_id)
            torch.cuda.synchronize()
            tg_times.append(time.perf_counter() - t0)

        pp_mean = sum(pp_times) / len(pp_times)
        tg_mean = sum(tg_times) / len(tg_times)
        pp_tok_s = S / pp_mean
        tg_tok_s = args.gen_tokens / tg_mean
        print(f"S={S:>5}  pp={pp_mean*1000:>8.0f}ms ({pp_tok_s:>7.1f} tok/s)  "
              f"tg{args.gen_tokens}={tg_mean*1000:>8.0f}ms "
              f"({tg_tok_s:>6.2f} tok/s)")
        results.append({
            "S": S, "pp_ms_mean": pp_mean*1000, "pp_tok_s": pp_tok_s,
            "tg_ms_mean": tg_mean*1000, "tg_tok_s": tg_tok_s,
        })

    if args.json:
        out = {"backend": args.backend, "gen_tokens": args.gen_tokens,
                "results": results}
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
