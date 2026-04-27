"""Honest benchmark: our DN kernels vs fla (the production fast path).

Re-runs the wide-shape comparison with fla properly available, so the HF
DN layer takes its real fast path (Triton chunk_gated_delta_rule) instead
of the fp32 Python fallback.
"""
from __future__ import annotations

import argparse
import copy
import sys
import time

# Shim must be imported before fla / transformers.
sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")
import _fla_torch_compile_shim  # noqa: F401
import torch

from dn_hf_patch import patch_hf_qwen3_deltanet


def time_fn(fn, runs=5, warm=2):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / runs


def runs_warm_for_S(S: int) -> tuple[int, int]:
    """Adaptive bench iteration counts so the long-S sweep finishes in a
    reasonable wall time (per-call cost grows ~linearly in S for our
    kernel and ~constant in S for fla). Long-S shapes still get >=2
    timed runs which is enough to detect cold-cache outliers."""
    if S <= 1024:
        return 5, 2
    if S <= 8192:
        return 3, 1
    return 2, 1


def make_lora_model(base, rank):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(
        r=rank, lora_alpha=rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    return get_peft_model(copy.deepcopy(base), cfg).to("cuda", dtype=torch.bfloat16)


def run(model, ids, mode):
    if mode == "train":
        model.train()
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        model.zero_grad(set_to_none=True)
    else:
        model.eval()
        with torch.no_grad():
            model(input_ids=ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-lens", type=int, nargs="+",
                    default=[128, 256, 512, 1024, 2048,
                             4096, 8192, 16384, 32768])
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--modes", type=str, nargs="+",
                    default=["train", "infer"],
                    help="which modes to bench. Training above ~8K tends "
                         "to OOM HF's autograd graph; pass `--modes infer` "
                         "to bench inference-only at very long S.")
    ap.add_argument("--max-train-s", type=int, default=8192,
                    help="skip training mode for S > this (HF+fla autograd "
                         "exhausts HBM at long context).")
    args = ap.parse_args()

    print("Loading Qwen3.5-0.8B (with fla available)...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to("cuda")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")

    # Confirm fla is detected.
    import transformers.utils.import_utils as _iu
    print(f"fla available to transformers: {_iu.is_flash_linear_attention_available()}")

    rank = args.rank
    m_base = make_lora_model(base, rank)        # uses fla fast path inside DN layer
    m_ours = make_lora_model(base, rank)
    n = patch_hf_qwen3_deltanet(m_ours)         # replaces DN layer with our kernel
    print(f"  patched {n} GatedDeltaNet layers in 'ours'")

    print()
    print(f"{'mode':<6} {'S':>6} | {'HF+fla ms':>11} {'ours ms':>11} {'ratio':>8} {'tok/s ours':>12}")
    print("-" * 64)
    for S in args.seq_lens:
        ids = torch.randint(0, tok.vocab_size, (1, S), device="cuda", dtype=torch.long)
        runs, warm = runs_warm_for_S(S)
        for mode in args.modes:
            if mode == "train" and S > args.max_train_s:
                print(f"{mode:<6} {S:>6} | (skipped — > max_train_s={args.max_train_s})")
                continue
            try:
                ms_b = time_fn(lambda: run(m_base, ids, mode), runs=runs, warm=warm)
                ms_o = time_fn(lambda: run(m_ours, ids, mode), runs=runs, warm=warm)
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                print(f"{mode:<6} {S:>6} | error: {str(e).splitlines()[0][:50]}")
                torch.cuda.empty_cache()
                continue
            ratio = ms_b / ms_o
            tag = "ours" if ratio > 1.0 else "fla "
            tok_per_sec = S / (ms_o / 1000.0)
            print(f"{mode:<6} {S:>6} | {ms_b:>11.2f} {ms_o:>11.2f} {ratio:>7.2f}x {tok_per_sec:>11.0f} ({tag})")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
