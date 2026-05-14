"""Capture HF Qwen3.6-27B reference outputs as the correctness oracle.

Saves a single .pt file with fixed prompts + their HF-computed
next-token logits and last hidden states. Future passes of our
megakernel-on-27B compare against this golden file.

Run once (slow: 54 GB BF16 download on first invocation, then ~few
minutes of forward passes):

    HF_HOME=/home/sparkz/rl/.hf_cache \\
        /home/sparkz/rl/.venv/bin/python3 \\
        models/qwen36_27b/reference/capture_hf_reference.py \\
        --out models/qwen36_27b/reference/hf_golden.pt

Fixed prompts:
  natural-1   : a short factual recall ("capital of ...")
  natural-2   : a code-completion stub
  wikitext-512: 512 tokens from wikitext-2 test
  wikitext-2k : 2048 tokens from wikitext-2 test
  wikitext-4k : 4096 tokens (only captured if --max-S >= 4096)

For each prompt we save:
  tokens     : int32 [S]
  last_logits: fp32 [vocab]           argmax / softmax target
  last_hidden: bf16 [hidden]          pre-LM-head, for hidden-state
                                      cos-sim checks against ours

Why capture rather than re-run HF every time:
  - HF Qwen3.6-27B is 54 GB BF16; load takes minutes.
  - Per-S forward at 27B is slow under HF (torch-native DN fallback
    unless fla is installed). A captured golden makes regression checks
    fast (just load + compare).
  - Fixed prompts let us track drift across kernel changes.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

MODEL_NAME_DEFAULT = "Qwen/Qwen3.6-27B"


def make_prompts(tokenizer, max_S: int) -> list[dict]:
    """Build the fixed prompt set."""
    prompts: list[dict] = []

    natural_1 = (
        "The capital of France is Paris. The capital of Germany is Berlin. "
        "The capital of Italy is Rome. The capital of Spain is Madrid. "
        "The capital of Portugal is Lisbon. The capital of Japan is Tokyo. "
        "The capital of China is Beijing. The capital of Russia is Moscow. "
        "The capital of the United Kingdom is London. The capital of the "
        "United States is"
    )
    ids = tokenizer(natural_1, return_tensors="pt").input_ids[0]
    prompts.append({"name": "natural-1", "tokens": ids, "expected_continuation": " Washington"})

    natural_2 = (
        "# Python: read a file line by line and count words\n"
        "def count_words(path):\n"
        "    total = 0\n"
        "    with open(path) as f:\n"
        "        for line in f:\n"
        "            total +="
    )
    ids = tokenizer(natural_2, return_tensors="pt").input_ids[0]
    prompts.append({"name": "natural-2", "tokens": ids, "expected_continuation": " len"})

    # Wikitext-2 windows. Lazy import; the dataset is ~50 MB.
    from datasets import load_dataset
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = " ".join(x for x in wt["text"] if x.strip())[: max(50_000, max_S * 6)]
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    for S in (512, 2048, 4096, 8192, 16384, 32768):
        if S > max_S:
            break
        if ids.numel() < S:
            print(f"  skipping wikitext-{S}: only {ids.numel()} tokens in source")
            continue
        prompts.append({"name": f"wikitext-{S}", "tokens": ids[:S].clone()})
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="path to write the .pt golden")
    ap.add_argument("--model-name", default=MODEL_NAME_DEFAULT)
    ap.add_argument("--max-S", type=int, default=4096,
                    help="cap on the longest wikitext prompt (default 4096). "
                         "Set higher (e.g. 32768) for long-context goldens — "
                         "HF forward becomes slow at long S.")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true",
                    help="don't load the model, just print the prompt manifest")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer for {args.model_name}...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_name)

    prompts = make_prompts(tok, args.max_S)
    print("\nPrompt manifest:")
    for p in prompts:
        print(f"  {p['name']:>15}  S={p['tokens'].numel():>5}")
    if args.dry_run:
        print("\n--dry-run -- exiting without model load")
        return

    print(f"\nLoading {args.model_name} ({args.dtype})... (first run downloads ~54 GB)",
          flush=True)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=dtype, device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")
    if torch.cuda.is_available():
        print(f"  GPU mem allocated: {torch.cuda.memory_allocated()/(1024**3):.1f} GB")

    golden: dict[str, dict] = {}
    for p in prompts:
        S = int(p["tokens"].numel())
        print(f"\nForward {p['name']:<15} S={S} ...", flush=True)
        ids = p["tokens"].unsqueeze(0).to(args.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out_ = model(input_ids=ids, use_cache=False, output_hidden_states=True)
        fwd_s = time.perf_counter() - t0
        last_logits = out_.logits[0, -1].detach().to(torch.float32).cpu()
        # Last hidden state (pre-LM-head, post final RMSnorm).
        last_hidden = out_.hidden_states[-1][0, -1].detach().to(torch.bfloat16).cpu()
        top1 = int(last_logits.argmax().item())
        top1_tok = tok.decode([top1])
        print(f"  {fwd_s:.1f}s  top-1 token id={top1} ({top1_tok!r})")
        if "expected_continuation" in p:
            print(f"  expected: {p['expected_continuation']!r}")

        golden[p["name"]] = dict(
            tokens=p["tokens"].to(torch.int32),
            last_logits=last_logits,
            last_hidden=last_hidden,
            top1=top1,
            S=S,
        )

    meta = dict(
        model_name=args.model_name,
        dtype=args.dtype,
        prompts=[p["name"] for p in prompts],
        torch_version=torch.__version__,
        transformers_version=__import__("transformers").__version__,
    )

    payload = dict(meta=meta, golden=golden)
    print(f"\nSaving golden -> {out}", flush=True)
    torch.save(payload, out)
    print(f"  wrote {out.stat().st_size / (1024**2):.1f} MB")


if __name__ == "__main__":
    main()
