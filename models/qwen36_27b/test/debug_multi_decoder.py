"""Test creating multiple decoders that share the HF model."""
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
    S = ids.numel()

    for trial in range(3):
        dec = Qwen36MegakernelDecoder(max_seq=64, verbose=False, hf_model=hf, tokenizer=tok)
        dec.prefill(ids[:7])
        torch.cuda.synchronize()
        logits = dec.sc.g_normalized.float().cpu()
        has_nan = bool(torch.isnan(logits).any())
        print(f"  trial {trial}: dec.sc.g_normalized has_nan={has_nan}  "
              f"max_abs={logits.abs().max().item():.4g}")
        # also check intermediate buffers
        for n in ("fa_k_cache", "fa_v_cache", "dn_states", "conv_bufs"):
            buf = getattr(dec.sc, n)
            print(f"      {n}: nan={bool(torch.isnan(buf.float()).any())}  "
                  f"max_abs={buf.float().abs().max().item():.4g}")


if __name__ == "__main__":
    main()
