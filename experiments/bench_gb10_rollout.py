"""GB10 rollout shape sweep for the gb10-train branch.

Mirrors the prompt-length sweep in docs/results/qwen35_0p8b_3090.md
(128 / 512 / 2048 / 8192 / 16384 / 32768) but runs on a single GB10.

Megakernel path: BF16 backend (Decoder.prefill is BF16-only; NVFP4
prefill is not implemented). HF baseline: AutoModel(input_ids) for
prefill timing, then `model.step` via past_key_values for decode.

Run from repo root:
    source /home/sparkz/rl/.venv/bin/activate
    export HF_HOME=/home/sparkz/rl/.hf_cache
    PYTHONPATH=models/qwen35_0p8b python experiments/bench_gb10_rollout.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import torch


def time_ours(prompt_lens, gen_tokens, runs, warm):
    from model import Decoder
    print("== Megakernel BF16 (prefill_bf16 + step) ==", flush=True)
    out = {}
    decoder = Decoder(backend="bf16", verbose=False)
    for P in prompt_lens:
        prompt = list(range(2, 2 + P))
        for _ in range(warm):
            decoder.reset()
            nxt = decoder.prefill(prompt)
            for _ in range(gen_tokens - 1):
                nxt = decoder.step(int(nxt))
        torch.cuda.synchronize()
        prefills, gens, totals = [], [], []
        for _ in range(runs):
            decoder.reset()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            nxt = decoder.prefill(prompt)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            for _ in range(gen_tokens - 1):
                nxt = decoder.step(int(nxt))
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            prefills.append((t1 - t0) * 1000)
            gens.append((t2 - t1) * 1000)
            totals.append((t2 - t0) * 1000)
        out[P] = {
            "prefill_ms": min(prefills),
            "gen_ms": min(gens),
            "total_ms": min(totals),
            "pp_tps": P / (min(prefills) / 1000),
            "tg_tps": (gen_tokens - 1) / (min(gens) / 1000),
        }
        print(f"P={P:>6}: prefill={out[P]['prefill_ms']:8.2f} ms "
              f"({out[P]['pp_tps']:>9.1f} t/s)  "
              f"gen32={out[P]['gen_ms']:6.2f} ms "
              f"({out[P]['tg_tps']:>6.1f} t/s)  "
              f"total={out[P]['total_ms']:7.2f} ms", flush=True)
    del decoder
    torch.cuda.empty_cache()
    return out


def time_hf(prompt_lens, gen_tokens, runs, warm, model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\n== HuggingFace BF16 eager ({model_name}) ==", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_name)
    out = {}
    with torch.inference_mode():
        for P in prompt_lens:
            input_ids = torch.arange(2, 2 + P, dtype=torch.long, device="cuda").unsqueeze(0)

            def one_run():
                o = model(input_ids, use_cache=True)
                past = o.past_key_values
                cur = o.logits[:, -1:].argmax(-1)
                for _ in range(gen_tokens - 1):
                    o = model(cur, past_key_values=past, use_cache=True)
                    past = o.past_key_values
                    cur = o.logits[:, -1:].argmax(-1)
                return past, cur

            for _ in range(warm):
                one_run()
                torch.cuda.synchronize()

            prefills, gens, totals = [], [], []
            for _ in range(runs):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                o = model(input_ids, use_cache=True)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                past = o.past_key_values
                cur = o.logits[:, -1:].argmax(-1)
                for _ in range(gen_tokens - 1):
                    o = model(cur, past_key_values=past, use_cache=True)
                    past = o.past_key_values
                    cur = o.logits[:, -1:].argmax(-1)
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                prefills.append((t1 - t0) * 1000)
                gens.append((t2 - t1) * 1000)
                totals.append((t2 - t0) * 1000)
            out[P] = {
                "prefill_ms": min(prefills),
                "gen_ms": min(gens),
                "total_ms": min(totals),
                "pp_tps": P / (min(prefills) / 1000),
                "tg_tps": (gen_tokens - 1) / (min(gens) / 1000),
            }
            print(f"P={P:>6}: prefill={out[P]['prefill_ms']:8.2f} ms "
                  f"({out[P]['pp_tps']:>9.1f} t/s)  "
                  f"gen32={out[P]['gen_ms']:6.2f} ms "
                  f"({out[P]['tg_tps']:>6.1f} t/s)  "
                  f"total={out[P]['total_ms']:7.2f} ms", flush=True)
    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-lens", type=int, nargs="+",
                    default=[128, 512, 2048, 8192, 16384, 32768])
    ap.add_argument("--gen-tokens", type=int, default=32)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--skip-hf", action="store_true")
    ap.add_argument("--skip-ours", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    sys.path.insert(0, "models/qwen35_0p8b")
    import qwen35_megakernel_bf16_C  # noqa: F401

    results = {}
    if not args.skip_ours:
        results["ours"] = time_ours(args.prompt_lens, args.gen_tokens,
                                    args.runs, args.warm)
    if not args.skip_hf:
        results["hf"] = time_hf(args.prompt_lens, args.gen_tokens,
                                args.runs, args.warm, args.model_name)

    if results.get("ours") and results.get("hf"):
        print("\n== Summary (total wall = prefill + 32-gen, ms; speedup vs HF) ==", flush=True)
        print(f"{'P':>6} | {'HF ms':>10} | {'Ours ms':>10} | {'speedup':>8}")
        print("-" * 44)
        for P in args.prompt_lens:
            ours = results["ours"][P]["total_ms"]
            hf = results["hf"][P]["total_ms"]
            print(f"{P:>6} | {hf:>10.1f} | {ours:>10.1f} | {hf/ours:>7.2f}x")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.json}", flush=True)


if __name__ == "__main__":
    main()
