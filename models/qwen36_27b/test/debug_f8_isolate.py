"""Find the smallest S at which prefill on real HF weights produces NaN.

Runs prefill with S=1, 2, 3, ... on a fresh decoder for each S, and
reports the max_abs of g_normalized + whether any value is NaN.
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
    ids_all = tok(prompt, return_tensors="pt").input_ids[0].to(torch.int32).cuda()
    S_max = ids_all.numel()
    print(f"prompt: {prompt!r}  ({S_max} tokens)\n")

    # Reuse one decoder, but RESET it before each run (so state is fresh).
    dec = Qwen36MegakernelDecoder(max_seq=64, verbose=False, hf_model=hf, tokenizer=tok)
    for S in range(1, S_max + 1):
        dec.reset()
        dec.prefill(ids_all[:S])
        torch.cuda.synchronize()
        logits = dec.sc.g_normalized.float().cpu()
        has_nan = bool(torch.isnan(logits).any())
        max_abs = logits.abs().max().item() if not has_nan else float("nan")
        # Also inspect dn_states / conv_bufs.
        dn_nan = bool(torch.isnan(dec.sc.dn_states.float()).any())
        cb_nan = bool(torch.isnan(dec.sc.conv_bufs.float()).any())
        fa_nan = bool(torch.isnan(dec.sc.fa_k_cache.float()).any())
        flag = "  <-- FIRST NaN" if (has_nan or dn_nan or cb_nan or fa_nan) and S > 0 else ""
        print(f"  S={S:>2}  norm_max_abs={max_abs:>10.4g}  "
              f"nan(logits/dn/conv/fa)={has_nan}/{dn_nan}/{cb_nan}/{fa_nan}{flag}")
        if has_nan or dn_nan or cb_nan or fa_nan:
            return


if __name__ == "__main__":
    main()
