"""Pin down WHY second decoder NaNs."""
from __future__ import annotations
import os, sys, gc
import torch

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True).eval()

    from runtime_megakernel import Qwen36MegakernelDecoder

    prompt = "Hello"
    ids = tok(prompt, return_tensors="pt").input_ids[0].to(torch.int32).cuda()

    # 1. Create dec_a, run, DON'T keep it.
    print("=== trial 1: alone ===")
    dec = Qwen36MegakernelDecoder(max_seq=64, verbose=False, hf_model=hf, tokenizer=tok)
    dec.prefill(ids)
    torch.cuda.synchronize()
    logits = dec.sc.g_normalized.float().cpu()
    print(f"  has_nan={bool(torch.isnan(logits).any())}  max_abs={logits.abs().max().item():.4g}")
    del dec
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Create a second decoder, run.
    print("\n=== trial 2: after del + empty_cache ===")
    dec = Qwen36MegakernelDecoder(max_seq=64, verbose=False, hf_model=hf, tokenizer=tok)
    dec.prefill(ids)
    torch.cuda.synchronize()
    logits = dec.sc.g_normalized.float().cpu()
    print(f"  has_nan={bool(torch.isnan(logits).any())}  max_abs={logits.abs().max().item():.4g}")

    # 3. Create a third decoder WHILE holding the second.
    print("\n=== trial 3: alongside trial-2 decoder ===")
    dec2 = Qwen36MegakernelDecoder(max_seq=64, verbose=False, hf_model=hf, tokenizer=tok)
    dec2.prefill(ids)
    torch.cuda.synchronize()
    logits2 = dec2.sc.g_normalized.float().cpu()
    print(f"  dec2:  has_nan={bool(torch.isnan(logits2).any())}  max_abs={logits2.abs().max().item():.4g}")
    # Also re-check dec — was its state corrupted by creating dec2?
    print(f"  dec (re-checked, no new prefill):  has_nan={bool(torch.isnan(logits).any())}  "
          f"(this is the captured cpu copy)")
    # Now re-run prefill on dec to see if its scratch is corrupt.
    dec.reset()
    dec.prefill(ids)
    torch.cuda.synchronize()
    logits = dec.sc.g_normalized.float().cpu()
    print(f"  dec (re-prefilled after dec2 was created): has_nan={bool(torch.isnan(logits).any())}  "
          f"max_abs={logits.abs().max().item():.4g}")


if __name__ == "__main__":
    main()
