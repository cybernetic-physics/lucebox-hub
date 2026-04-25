"""Wide-shape LoRA training + inference comparison: HF baseline vs our
kernel-patched HF Qwen3.5-0.8B + PEFT LoRA.

Sweeps:
  - Sequence length: 128, 256, 512, 1024, 2048, 4096
  - LoRA rank: 8, 16, 32
  - Mode: training step (fwd+bwd), inference forward (no_grad)

Reports ms/iter for baseline vs patched, plus the speedup. Batch=1 for
clarity (the model is dominated by per-token cost).

Notes:
  * Training step exercises our autograd-wrapped recurrent CUDA bwd kernel
    on the 18 DeltaNet layers. Forward in train mode also goes through
    the recurrent path (since the chunked bwd isn't wired yet).
  * Inference (no_grad) uses our chunked tensor-core forward kernel —
    nvcuda::wmma m16n16k16 bf16 + fp32 accum.
"""
from __future__ import annotations

import argparse
import copy
import sys
import time

# Patch transformers' broken fla detection BEFORE importing transformers.
import transformers.utils.import_utils as _iu
_iu.is_flash_linear_attention_available = lambda: False

import torch

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")
from dn_hf_patch import patch_hf_qwen3_deltanet


def time_fn(fn, runs=5, warm=2):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / runs


def make_lora_model(base, rank):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(
        r=rank, lora_alpha=rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    m = get_peft_model(copy.deepcopy(base), cfg).to("cuda", dtype=torch.bfloat16)
    return m


def run_one(model, ids, mode):
    if mode == "train":
        model.train()
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        model.zero_grad(set_to_none=True)
    else:  # inference
        model.eval()
        with torch.no_grad():
            model(input_ids=ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-lens", type=int, nargs="+",
                    default=[128, 256, 512, 1024, 2048])
    ap.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    args = ap.parse_args()

    print(f"Loading {args.model}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to("cuda")
    tok = AutoTokenizer.from_pretrained(args.model)

    print()
    print("=" * 96)
    print(f"{'mode':<6} {'rank':>5} {'S':>6} | "
          f"{'baseline ms':>13} {'patched ms':>12} {'speedup':>9} | notes")
    print("=" * 96)

    rows = []
    for rank in args.ranks:
        m_baseline = make_lora_model(base, rank)
        m_patched  = make_lora_model(base, rank)
        n = patch_hf_qwen3_deltanet(m_patched)
        assert n == 18, f"expected 18 DN layers patched, got {n}"

        for S in args.seq_lens:
            ids = torch.randint(0, tok.vocab_size, (1, S), device="cuda", dtype=torch.long)

            for mode in ("train", "infer"):
                # Drop runs at long S+train where memory may be tight.
                # Try the training path; if OOM, mark and continue.
                try:
                    ms_b = time_fn(lambda: run_one(m_baseline, ids, mode))
                    ms_p = time_fn(lambda: run_one(m_patched,  ids, mode))
                    speedup = ms_b / ms_p
                    note = ""
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    msg = str(e).splitlines()[0][:60]
                    print(f"{mode:<6} {rank:>5} {S:>6} | error: {msg}")
                    torch.cuda.empty_cache()
                    continue
                print(f"{mode:<6} {rank:>5} {S:>6} | "
                      f"{ms_b:>13.2f} {ms_p:>12.2f} {speedup:>8.2f}x | {note}")
                rows.append((mode, rank, S, ms_b, ms_p, speedup))
                torch.cuda.empty_cache()

        del m_baseline, m_patched
        torch.cuda.empty_cache()

    # ------ summary ------
    print()
    print("=" * 96)
    print("SUMMARY (geomean speedup per mode)")
    print("=" * 96)
    import math
    for mode in ("train", "infer"):
        sps = [r[5] for r in rows if r[0] == mode]
        if sps:
            geo = math.exp(sum(math.log(x) for x in sps) / len(sps))
            print(f"  {mode}:  {geo:.2f}x  (min {min(sps):.2f}x, max {max(sps):.2f}x, n={len(sps)})")


if __name__ == "__main__":
    main()
