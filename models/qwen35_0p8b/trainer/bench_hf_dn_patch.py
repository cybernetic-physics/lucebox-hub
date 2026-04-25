"""End-to-end: HF Qwen3.5-0.8B + LoRA training step, with vs without our
CUDA DN backward kernel patched in.

  baseline:  HF torch chunk_gated_delta_rule   (HF default fallback)
  patched:   our CUDA recurrence + autograd wrapper

Reports fwd+bwd ms/step at multiple seq lengths, batch=1.
"""
from __future__ import annotations

import argparse
import sys
import time

# Patch transformers' broken fla detection BEFORE importing transformers.
import transformers.utils.import_utils as _iu
_iu.is_flash_linear_attention_available = lambda: False

import torch

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")
from dn_hf_patch import patch_hf_qwen3_deltanet


def time_step(model, input_ids, runs=5, warm=2):
    for _ in range(warm):
        out = model(input_ids=input_ids, labels=input_ids)
        out.loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        out = model(input_ids=input_ids, labels=input_ids)
        out.loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[128, 512])
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    args = ap.parse_args()

    print(f"Loading {args.model}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    base = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to("cuda")
    tok = AutoTokenizer.from_pretrained(args.model)

    # Wrap with LoRA (matches rl_trainer config).
    cfg = LoraConfig(
        r=16, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    import copy
    base_baseline = copy.deepcopy(base)
    base_patched  = copy.deepcopy(base)

    model_baseline = get_peft_model(base_baseline, cfg).to("cuda", dtype=torch.bfloat16)
    model_baseline.train()

    model_patched = get_peft_model(base_patched, cfg).to("cuda", dtype=torch.bfloat16)
    model_patched.train()

    n = patch_hf_qwen3_deltanet(model_patched)
    print(f"Patched {n} GatedDeltaNet layers with our CUDA kernel.")

    print("=" * 72)
    print(f"{'S':>5} | {'baseline ms/step':>20} | {'patched ms/step':>20} | speedup")
    print("=" * 72)
    for S in args.seq_lens:
        ids = torch.randint(0, tok.vocab_size, (1, S), device="cuda", dtype=torch.long)
        ms_base = time_step(model_baseline, ids)
        ms_ours = time_step(model_patched,  ids)
        print(f"{S:>5} | {ms_base:>20.2f} | {ms_ours:>20.2f} | {ms_base/ms_ours:5.2f}x")


if __name__ == "__main__":
    main()
