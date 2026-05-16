"""Inspect dec2's layer_blob CONTENTS on GPU (not just its source bytes).
If H2D copy didn't transfer correctly, GPU bytes would differ from CPU."""
from __future__ import annotations
import os, sys, ctypes
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
    print("dec1:", "ok" if not torch.isnan(dec.sc.g_normalized).any() else "NAN")

    dec2 = Qwen36MegakernelDecoder(max_seq=64, verbose=False, hf_model=hf, tokenizer=tok)

    # READ both blobs from GPU back to CPU and compare.
    blob1_gpu = dec.layer_blob.cpu().numpy()
    blob2_gpu = dec2.layer_blob.cpu().numpy()
    print(f"\ndec1.layer_blob @ GPU={hex(dec.layer_blob.data_ptr())}")
    print(f"dec2.layer_blob @ GPU={hex(dec2.layer_blob.data_ptr())}")
    print(f"GPU contents differ by {(blob1_gpu != blob2_gpu).sum()} / {blob1_gpu.size} bytes")

    # Decode first layer's pointers — they should be HF weight addresses.
    PACK_HEADER = 8
    PACK_STRUCT = 192
    def show_layer(blob, name):
        layer_type = int.from_bytes(bytes(blob[0:4]), 'little')
        print(f"  {name}.layer[0].type={layer_type}")
        for slot in range(4):
            off = PACK_HEADER + slot * 8
            ptr = int.from_bytes(bytes(blob[off:off+8]), 'little')
            print(f"     slot {slot}: 0x{ptr:016x}")
    show_layer(blob1_gpu[:PACK_STRUCT], "blob1")
    show_layer(blob2_gpu[:PACK_STRUCT], "blob2")

    # Reference: HF weights' actual addresses for layer 0 (DN).
    sd = hf.state_dict()
    print(f"\n  HF model.layers.0.input_layernorm.weight @ "
          f"{hex(sd['model.layers.0.input_layernorm.weight'].data_ptr())}")
    print(f"  HF model.layers.0.linear_attn.in_proj_qkvz.weight @ "
          f"{hex(sd['model.layers.0.linear_attn.in_proj_qkvz.weight'].data_ptr()) if 'model.layers.0.linear_attn.in_proj_qkvz.weight' in sd else 'KEY NOT FOUND'}")

    # Now prefill dec2 and check.
    dec2.prefill(ids); torch.cuda.synchronize()
    print(f"\ndec2 prefill:", "ok" if not torch.isnan(dec2.sc.g_normalized).any() else "NAN")


if __name__ == "__main__":
    main()
