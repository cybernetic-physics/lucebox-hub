"""Isolate which step in second decoder __init__ corrupts the first."""
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
    from weight_packer import (alloc_scratch, pack_layer_weights,
                                _unify_from_hf_model)

    ids = tok("Hello", return_tensors="pt").input_ids[0].to(torch.int32).cuda()

    # Create dec1 and run a successful prefill.
    dec = Qwen36MegakernelDecoder(max_seq=64, verbose=False, hf_model=hf, tokenizer=tok)
    dec.prefill(ids)
    torch.cuda.synchronize()
    print("dec1 baseline:", "ok" if not torch.isnan(dec.sc.g_normalized).any() else "NAN")

    # Snapshot dec1's important buffer pointers + a known-good output.
    dec_fa_ptr   = dec.sc.fa_k_cache.data_ptr()
    dec_dn_ptr   = dec.sc.dn_states.data_ptr()
    dec_blob_ptr = dec.layer_blob.data_ptr()
    weights_before = {
        n: dec.weights[n].data_ptr() if isinstance(dec.weights[n], torch.Tensor) else None
        for n in ("embed_weight", "final_norm_weight", "lm_head_weight")
    }

    # Step A only: re-call _unify_from_hf_model on the SAME hf, no other.
    w2 = _unify_from_hf_model(hf)
    print("\nafter _unify_from_hf_model:")
    print(f"  dec.layer_blob[0..16] = {dec.layer_blob[:16].tolist()}")
    dec.reset(); dec.prefill(ids); torch.cuda.synchronize()
    print("  dec re-prefill:", "ok" if not torch.isnan(dec.sc.g_normalized).any() else "NAN")
    del w2

    # Step B only: pack_layer_weights without anything else.
    blob2 = pack_layer_weights(dec.weights["layer_data"])
    print("\nafter pack_layer_weights:")
    print(f"  dec.layer_blob[0..16] = {dec.layer_blob[:16].tolist()}")
    print(f"  blob2[0..16]          = {blob2[:16].tolist()}")
    print(f"  dec.layer_blob.data_ptr()={dec_blob_ptr:#x}  (alive: {dec.layer_blob.data_ptr() == dec_blob_ptr})")
    dec.reset(); dec.prefill(ids); torch.cuda.synchronize()
    print("  dec re-prefill:", "ok" if not torch.isnan(dec.sc.g_normalized).any() else "NAN")
    del blob2

    # Step C only: alloc_scratch.
    sc2 = alloc_scratch(max_seq=64, verbose=False)
    print(f"\nafter alloc_scratch:")
    print(f"  dec.sc.fa_k_cache.data_ptr()={dec.sc.fa_k_cache.data_ptr():#x}  was={dec_fa_ptr:#x}")
    dec.reset(); dec.prefill(ids); torch.cuda.synchronize()
    print("  dec re-prefill:", "ok" if not torch.isnan(dec.sc.g_normalized).any() else "NAN")
    del sc2

    # Step ALL: full Qwen36MegakernelDecoder.
    dec2 = Qwen36MegakernelDecoder(max_seq=64, verbose=False, hf_model=hf, tokenizer=tok)
    print(f"\nafter full Qwen36MegakernelDecoder(...) (dec2 alive):")
    dec.reset(); dec.prefill(ids); torch.cuda.synchronize()
    print("  dec re-prefill:", "ok" if not torch.isnan(dec.sc.g_normalized).any() else "NAN")
    dec2.prefill(ids); torch.cuda.synchronize()
    print("  dec2 prefill:  ", "ok" if not torch.isnan(dec2.sc.g_normalized).any() else "NAN")


if __name__ == "__main__":
    main()
