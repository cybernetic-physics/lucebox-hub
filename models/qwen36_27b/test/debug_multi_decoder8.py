"""Verify: keeping ALL decoders alive prevents the multi-decoder NaN.

If true, the bug is in something that gets freed when an old decoder
is GC'd before a new one is run.
"""
from __future__ import annotations
import os, sys
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
    prompt = "Hello, my name is Bob. I work as a software engineer at"
    ids = tok(prompt, return_tensors="pt").input_ids[0].to(torch.int32).cuda()

    # Case A: REASSIGN, prior decoder is GC'd.
    print("Case A: reassign `dec` per trial (old decoder GC'd)")
    for trial in range(3):
        dec = Qwen36MegakernelDecoder(max_seq=64, verbose=False, hf_model=hf, tokenizer=tok)
        dec.prefill(ids[:7]); torch.cuda.synchronize()
        nan = torch.isnan(dec.sc.g_normalized).any().item()
        print(f"  trial {trial}: nan={nan}")

    print("\nCase B: keep ALL decoders alive in a list")
    decs = []
    for trial in range(3):
        dec = Qwen36MegakernelDecoder(max_seq=64, verbose=False, hf_model=hf, tokenizer=tok)
        decs.append(dec)
        dec.prefill(ids[:7]); torch.cuda.synchronize()
        nan = torch.isnan(dec.sc.g_normalized).any().item()
        print(f"  trial {trial}: nan={nan}")


if __name__ == "__main__":
    main()
