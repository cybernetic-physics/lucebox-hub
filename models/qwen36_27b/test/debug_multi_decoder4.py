"""Swap scratch/blob between two decoders to isolate the corrupting state."""
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
    ids = tok("Hello", return_tensors="pt").input_ids[0].to(torch.int32).cuda()

    dec = Qwen36MegakernelDecoder(max_seq=64, verbose=False, hf_model=hf, tokenizer=tok)
    dec.prefill(ids); torch.cuda.synchronize()
    print("baseline dec:", "ok" if not torch.isnan(dec.sc.g_normalized).any() else "NAN")

    # Save dec's scratch + blob.
    sc1 = dec.sc
    blob1 = dec.layer_blob

    dec2 = Qwen36MegakernelDecoder(max_seq=64, verbose=False, hf_model=hf, tokenizer=tok)
    print(f"  dec.layer_blob   ptr={blob1.data_ptr():#x}  match dec2? "
          f"{(blob1[:128] == dec2.layer_blob[:128]).all().item()}")

    # Compare full blob byte-by-byte across decoders
    diff_bytes = (blob1.cpu() != dec2.layer_blob.cpu()).sum().item()
    print(f"  blob bytes differ: {diff_bytes} / {blob1.numel()}")

    # Try dec2 with dec's scratch (swapping):
    print("\nswap-test: dec2.sc <- dec.sc, prefill")
    dec2.sc = sc1
    dec2.prefill(ids); torch.cuda.synchronize()
    print("  result:", "ok" if not torch.isnan(dec2.sc.g_normalized).any() else "NAN")
    # restore
    dec2.sc = dec2.__class__.__init__.__wrapped__ if False else None  # placeholder
    # Actually just create one more for a fair baseline
    print("\nswap-test: dec2.layer_blob <- dec.layer_blob, fresh sc:")
    dec2 = Qwen36MegakernelDecoder(max_seq=64, verbose=False, hf_model=hf, tokenizer=tok)
    dec2.layer_blob = blob1
    dec2.prefill(ids); torch.cuda.synchronize()
    print("  result:", "ok" if not torch.isnan(dec2.sc.g_normalized).any() else "NAN")


if __name__ == "__main__":
    main()
