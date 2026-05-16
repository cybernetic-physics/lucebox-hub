"""Does NVFP4 backend also exhibit the multi-decoder NaN?"""
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
    ids = tok("Hello, my name is Bob.",
              return_tensors="pt").input_ids[0].to(torch.int32).cuda()[:7]

    print("=== BF16 backend ===")
    for trial in range(2):
        dec = Qwen36MegakernelDecoder(max_seq=64, verbose=False, hf_model=hf,
                                        tokenizer=tok, backend="bf16")
        dec.prefill(ids); torch.cuda.synchronize()
        nan = torch.isnan(dec.sc.g_normalized).any().item()
        print(f"  trial {trial}: nan={nan}")

    print("\n=== NVFP4 backend ===")
    for trial in range(2):
        dec = Qwen36MegakernelDecoder(max_seq=64, verbose=False, hf_model=hf,
                                        tokenizer=tok, backend="nvfp4")
        dec.prefill(ids); torch.cuda.synchronize()
        nan = torch.isnan(dec.sc.g_normalized).any().item()
        print(f"  trial {trial}: nan={nan}")


if __name__ == "__main__":
    main()
