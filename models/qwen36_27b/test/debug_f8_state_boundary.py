"""Find where split-prefill diverges on REAL HF weights.

Run two paths:
  A: prefill(ids[:S])                   (one shot of 14 tokens)
  B: prefill(ids[:S1]); prefill(ids[S1:], start_position=S1)

Compare per-state-buffer at the BOUNDARY (after first half):
  - fa_k_cache, fa_v_cache  : write index [0..6]
  - dn_states               : recurrent state (cumulative)
  - conv_bufs               : ring buffer

Both buffers should be IDENTICAL after position 6 in both paths. The
first buffer to diverge points at where the bug enters.

Run:
  HF_HOME=/home/sparkz/rl/.hf_cache \
    /home/sparkz/rl/.venv/bin/python3 \
    models/qwen36_27b/test/debug_f8_state_boundary.py
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
    S = ids.numel()
    S1 = S // 2
    print(f"\nprompt: {prompt!r}  ({S} tokens; halt-at S1={S1})\n")

    # Path A: prefill ids[:S1] only (so we can snapshot after position S1-1).
    dec_a = Qwen36MegakernelDecoder(max_seq=256, verbose=False, hf_model=hf, tokenizer=tok)
    dec_a.prefill(ids[:S1])
    torch.cuda.synchronize()
    fa_k_a = dec_a.sc.fa_k_cache.clone()
    fa_v_a = dec_a.sc.fa_v_cache.clone()
    dn_s_a = dec_a.sc.dn_states.clone()
    conv_a = dec_a.sc.conv_bufs.clone()

    # Path B: same as A — just verify allocator gives same result.
    dec_b = Qwen36MegakernelDecoder(max_seq=256, verbose=False, hf_model=hf, tokenizer=tok)
    dec_b.prefill(ids[:S1])
    torch.cuda.synchronize()
    fa_k_b = dec_b.sc.fa_k_cache.clone()
    fa_v_b = dec_b.sc.fa_v_cache.clone()
    dn_s_b = dec_b.sc.dn_states.clone()
    conv_b = dec_b.sc.conv_bufs.clone()

    def diff(name, a, b):
        d = (a.float() - b.float()).abs().max().item()
        a_nan = torch.isnan(a.float()).any().item()
        b_nan = torch.isnan(b.float()).any().item()
        a_abs = a.float().abs().max().item()
        print(f"  {name:>14}  max_abs_diff={d:.6g}  a_max={a_abs:.3g}  a_nan={a_nan}  b_nan={b_nan}")

    print("Two independent runs of prefill(ids[:S1]) on same decoder/model:")
    diff("fa_k_cache", fa_k_a, fa_k_b)
    diff("fa_v_cache", fa_v_a, fa_v_b)
    diff("dn_states",  dn_s_a, dn_s_b)
    diff("conv_bufs",  conv_a, conv_b)

    # Now: full one-shot up to S to capture final state — and capture state
    # at position S1 by stopping mid-way (not possible with the current kernel
    # API, so we just rely on the per-prefill snapshot above).

    # And: extend Path B with the SECOND half via start_position.
    dec_b.prefill(ids[S1:], start_position=S1)
    torch.cuda.synchronize()
    logits_b = dec_b.sc.g_normalized.cpu()
    nb_nan = torch.isnan(logits_b).any().item()
    print(f"\nAfter second-half prefill on Path B:")
    print(f"  g_normalized: max_abs={logits_b.abs().max():.3g}  has_nan={nb_nan}")

    # Full one-shot for comparison.
    dec_c = Qwen36MegakernelDecoder(max_seq=256, verbose=False, hf_model=hf, tokenizer=tok)
    dec_c.prefill(ids)
    torch.cuda.synchronize()
    logits_c = dec_c.sc.g_normalized.cpu()
    nc_nan = torch.isnan(logits_c).any().item()
    print(f"\nOne-shot prefill on full ids:")
    print(f"  g_normalized: max_abs={logits_c.abs().max():.3g}  has_nan={nc_nan}")
    print(f"  cos(split, oneshot) = "
          f"{torch.nn.functional.cosine_similarity(logits_b.unsqueeze(0).float(), logits_c.unsqueeze(0).float(), dim=-1).item():.6f}")


if __name__ == "__main__":
    main()
