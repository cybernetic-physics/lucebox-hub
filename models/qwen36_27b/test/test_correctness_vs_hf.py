"""Correctness regression harness for the qwen36_27b runtime.

Runs three prompts through both:
  - reference: `transformers.AutoModelForCausalLM` direct call (the oracle)
  - ours:      `runtime_hf.Qwen36Runtime.logits_for()`

Today: both use HF transformers, so they should be bit-equal.
Future:  when the megakernel-backed runtime replaces runtime_hf, this
         test stays in place and catches drift on the same prompts.

Skip strategy: if Qwen3.6-27B isn't downloaded yet, the test prints a
clear message and exits non-zero so CI/operators know to pull weights.

Run (requires Qwen/Qwen3.6-27B in HF_HOME, ~54 GB):
    HF_HOME=/home/sparkz/rl/.hf_cache \\
        /home/sparkz/rl/.venv/bin/python3 \\
        models/qwen36_27b/test/test_correctness_vs_hf.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))

from runtime_hf import Qwen36Runtime, MODEL_NAME_DEFAULT, VOCAB_SIZE


def _hf_cache_has_model(name: str) -> bool:
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))
    cache = Path(hf_home) / "hub" / f"models--{name.replace('/', '--')}"
    return cache.exists() and any(cache.glob("snapshots/*/config.json"))


def compare(a: torch.Tensor, b: torch.Tensor, label: str) -> dict:
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    top1_a = int(a.argmax().item())
    top1_b = int(b.argmax().item())
    top5_a = set(a.topk(5).indices.tolist())
    top5_b = set(b.topk(5).indices.tolist())
    cos = float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item())
    max_abs = float((a - b).abs().max().item())
    kl = float((F.log_softmax(a, dim=-1).exp() * (F.log_softmax(a, dim=-1) - F.log_softmax(b, dim=-1))).sum().item())
    res = dict(
        label=label,
        top1_match=top1_a == top1_b,
        top5_overlap=len(top5_a & top5_b) / 5,
        cos=cos, kl=kl, max_abs=max_abs,
        top1_a=top1_a, top1_b=top1_b,
    )
    print(f"  {label:<32} top1={'Y' if res['top1_match'] else 'N'}  "
          f"top5_overlap={res['top5_overlap']:.2f}  cos={cos:.6f}  "
          f"kl={kl:.6f}  max_abs={max_abs:.4f}")
    return res


def main():
    if not _hf_cache_has_model(MODEL_NAME_DEFAULT):
        print(f"Qwen3.6-27B not present in HF_HOME={os.environ.get('HF_HOME', '~/.cache/huggingface')}")
        print(f"Pull it first (~54 GB BF16):")
        print(f"  HF_HOME=/home/sparkz/rl/.hf_cache \\")
        print(f"      /home/sparkz/rl/.venv/bin/python3 -c \\")
        print(f"      \"from transformers import AutoModelForCausalLM; \\")
        print(f"       AutoModelForCausalLM.from_pretrained('{MODEL_NAME_DEFAULT}', dtype='bfloat16')\"")
        sys.exit(2)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME_DEFAULT, trust_remote_code=True)

    prompts = [
        ("natural-short", "The capital of the United States is"),
        ("natural-code", "def fibonacci(n):\n    if n < 2:\n        return n\n    return"),
        ("instruction",
         tok.apply_chat_template(
             [{"role": "user", "content": "Reply with a single integer: what is 7*11?"}],
             tokenize=False, add_generation_prompt=True,
             chat_template_kwargs={"enable_thinking": False},
         )),
    ]

    print(f"Loading reference (HF transformers, BF16)...", flush=True)
    t0 = time.perf_counter()
    ref = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_DEFAULT, dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True,
    )
    ref.eval()
    print(f"  loaded in {time.perf_counter()-t0:.1f}s  "
          f"gpu={torch.cuda.memory_allocated()/(1024**3):.1f} GB")

    print(f"\nBuilding ours (runtime_hf.Qwen36Runtime)...", flush=True)
    ours = Qwen36Runtime(model_name=MODEL_NAME_DEFAULT, backend="bf16")
    print(f"  gpu after ours: {torch.cuda.memory_allocated()/(1024**3):.1f} GB")

    print(f"\nComparing on {len(prompts)} prompts:\n")
    fails: list[str] = []
    for name, prompt in prompts:
        ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            ref_out = ref(input_ids=ids, use_cache=False)
        ref_logits = ref_out.logits[0, -1].to(torch.float32)
        ours_logits = ours.logits_for(ids)
        r = compare(ref_logits, ours_logits, name)
        # Today both sides ARE HF — should be bit-equal.
        if r["max_abs"] > 1e-3:
            fails.append(f"{name}: max_abs={r['max_abs']:.4f}")
        if not r["top1_match"]:
            fails.append(f"{name}: top1 mismatch (ref={r['top1_a']} ours={r['top1_b']})")

    print()
    if fails:
        for f in fails: print(f"  FAIL: {f}")
        print(f"\n{len(fails)} FAILURE(S)")
        sys.exit(1)
    print(f"ALL {len(prompts)} PROMPTS MATCH HF REFERENCE")
    print(f"(this is the regression bar future megakernel runtimes must pass)")


if __name__ == "__main__":
    main()
