"""Check if dec1's scratch is corrupted AFTER dec2's first prefill."""
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
    ids = tok("Hello, my name is Bob. I work as a software engineer at",
              return_tensors="pt").input_ids[0].to(torch.int32).cuda()[:7]

    dec = Qwen36MegakernelDecoder(max_seq=64, verbose=False, hf_model=hf, tokenizer=tok)
    dec.prefill(ids); torch.cuda.synchronize()
    print("dec1 prefill 1:", "ok" if not torch.isnan(dec.sc.g_normalized).any() else "NAN")
    # Snapshot dec1 scratch.
    snap = {n: getattr(dec.sc, n).clone() for n in ("fa_k_cache", "fa_v_cache",
                                                      "dn_states", "conv_bufs",
                                                      "g_normalized", "g_rope_inv_freq")}

    # Print dec1 scratch addresses.
    addrs_dec1 = {n: getattr(dec.sc, n).data_ptr() for n in
                   ("fa_k_cache", "fa_v_cache", "dn_states", "conv_bufs",
                    "hidden_buffer", "g_residual", "g_qkv_scratch",
                    "g_normalized", "g_rope_inv_freq")}
    sizes_dec1 = {n: getattr(dec.sc, n).numel() * getattr(dec.sc, n).element_size() for n in addrs_dec1}

    # Create dec2.
    dec2 = Qwen36MegakernelDecoder(max_seq=64, verbose=False, hf_model=hf, tokenizer=tok)
    addrs_dec2 = {n: getattr(dec2.sc, n).data_ptr() for n in addrs_dec1}
    sizes_dec2 = {n: getattr(dec2.sc, n).numel() * getattr(dec2.sc, n).element_size() for n in addrs_dec2}

    print("\nAddress overlap check (dec1 vs dec2 scratch ranges):")
    for n in addrs_dec1:
        a1, a2 = addrs_dec1[n], addrs_dec2[n]
        s1, s2 = sizes_dec1[n], sizes_dec2[n]
        # check if [a1, a1+s1) intersects [a2, a2+s2)
        overlap = max(a1, a2) < min(a1 + s1, a2 + s2)
        print(f"  {n:>16}: dec1=[{a1:#x}+{s1}] dec2=[{a2:#x}+{s2}] overlap={overlap}")

    # Now run dec2.prefill (expected to NaN).
    dec2.prefill(ids); torch.cuda.synchronize()
    print(f"\ndec2 prefill: {'ok' if not torch.isnan(dec2.sc.g_normalized).any() else 'NAN'}")

    # Check if dec1's scratch is now CORRUPT.
    print("\nIs dec1's scratch still intact?")
    for n, t_old in snap.items():
        t_new = getattr(dec.sc, n)
        diff = (t_old.float() - t_new.float()).abs().max().item()
        nan = torch.isnan(t_new.float()).any().item()
        print(f"  {n:>16}: max_diff={diff:.6g}  nan_now={nan}")


if __name__ == "__main__":
    main()
