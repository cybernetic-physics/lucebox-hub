"""Test: does cuda sync between decoder creations fix it?"""
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
    ids = tok("Hello", return_tensors="pt").input_ids[0].to(torch.int32).cuda()

    dec = Qwen36MegakernelDecoder(max_seq=64, verbose=False, hf_model=hf, tokenizer=tok)
    dec.prefill(ids); torch.cuda.synchronize()
    print("dec1:", "ok" if not torch.isnan(dec.sc.g_normalized).any() else "NAN")

    # Sync HARD before creating dec2.
    torch.cuda.synchronize()
    print("  alloc cuda allocator BEFORE dec2:", torch.cuda.memory_allocated() / 1e9, "GB")

    dec2 = Qwen36MegakernelDecoder(max_seq=64, verbose=False, hf_model=hf, tokenizer=tok)
    torch.cuda.synchronize()
    print("  alloc cuda allocator AFTER dec2:", torch.cuda.memory_allocated() / 1e9, "GB")
    dec2.prefill(ids); torch.cuda.synchronize()
    print("dec2 (with extra syncs):", "ok" if not torch.isnan(dec2.sc.g_normalized).any() else "NAN")

    # Try: dec2 prefill TWICE — does the second run come back?
    dec2.reset()
    dec2.prefill(ids); torch.cuda.synchronize()
    print("dec2 (second try after reset):", "ok" if not torch.isnan(dec2.sc.g_normalized).any() else "NAN")

    # Try: explicitly call torch.ops.qwen3x_C.prefill with all the SAME args as dec2 but using dec's scratch.
    print("\nDirect kernel call: dec2 weights/blob + dec1 scratch:")
    ops = torch.ops.qwen3x_C
    ops.prefill_qwen3x_naive(
        1, ids,
        dec2.weights["embed_weight"], dec2.weights["final_norm_weight"],
        dec2.layer_blob,
        dec.sc.fa_k_cache, dec.sc.fa_v_cache,
        dec.sc.dn_states, dec.sc.conv_bufs,
        dec.sc.hidden_buffer, dec.sc.g_residual,
        dec.sc.g_qkv_scratch, dec.sc.g_kv_scratch,
        dec.sc.g_attn_out, dec.sc.g_mlp_inter,
        dec.sc.g_z_scratch, dec.sc.g_beta_scratch, dec.sc.g_alpha_scratch,
        dec.sc.g_normalized, dec.sc.g_fa_partials, dec.sc.g_rope_inv_freq,
        64, 1.0, 32.0, 1.0, 262144, False, 0, None, 0)
    torch.cuda.synchronize()
    print("  result:", "ok" if not torch.isnan(dec.sc.g_normalized).any() else "NAN")


if __name__ == "__main__":
    main()
