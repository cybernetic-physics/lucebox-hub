"""C1 — first megakernel forward on real Qwen3.6-27B weights.

Minimum-viable check: does Qwen36MegakernelDecoder.prefill() return
without crashing on a single token? We're not checking correctness
here, just that the kernel infrastructure can complete one step.

Captures the HF reference last-position logits for the same single
token. If our kernel returns, prints argmax + cos similarity. If our
kernel crashes, prints the CUDA / Python error so we can diagnose.

Run:
    HF_HOME=/home/sparkz/rl/.hf_cache \\
        /home/sparkz/rl/.venv/bin/python3 \\
        models/qwen36_27b/test/test_c1_first_forward.py
"""
from __future__ import annotations

import os, sys, time, traceback
import torch
import torch.nn.functional as F

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))

PROMPT_TOKEN_ID = 220  # the BPE token id for " " (space). Innocuous one-token.


def main():
    # Step 1: load HF once. Both the reference and our kernel use these weights.
    print("Loading HF Qwen3.6-27B (BF16) — both the reference path and our "
          "kernel share these tensors via state_dict views...", flush=True)
    t0 = time.perf_counter()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True,
    ).eval()
    print(f"  HF loaded in {time.perf_counter()-t0:.1f}s; "
          f"GPU alloc: {torch.cuda.memory_allocated()/(1024**3):.1f} GB",
          flush=True)

    # Step 2: HF reference for the same single token.
    ids = torch.tensor([[PROMPT_TOKEN_ID]], dtype=torch.long, device="cuda")
    with torch.no_grad():
        ref = hf(input_ids=ids, use_cache=False)
    ref_logits = ref.logits[0, -1].to(torch.float32).cpu()
    ref_top1 = int(ref_logits.argmax().item())
    ref_top1_text = tok.decode([ref_top1])
    print(f"\n[HF reference]  input_token=  {PROMPT_TOKEN_ID} ({tok.decode([PROMPT_TOKEN_ID])!r})")
    print(f"                 ref top-1 =   {ref_top1} ({ref_top1_text!r})")
    print(f"                 logit max  =  {ref_logits.max().item():.4f}")

    # Step 3: build our megakernel decoder. Reuses HF's state_dict views, so
    # no extra weight allocation — total stays at ~50 GB.
    print("\nBuilding megakernel decoder (sharing HF weight tensors)...", flush=True)
    from runtime_megakernel import Qwen36MegakernelDecoder
    try:
        dec = Qwen36MegakernelDecoder(
            max_seq=128, verbose=True, hf_model=hf, tokenizer=tok)
    except Exception as e:
        print(f"  FAIL: build crashed: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Step 4: run prefill on a single token. The big moment.
    print("\nRunning megakernel prefill on a single token...", flush=True)
    ids32 = torch.tensor([PROMPT_TOKEN_ID], dtype=torch.int32, device="cuda")
    try:
        next_id = dec.prefill(ids32)
    except Exception as e:
        print(f"  FAIL: prefill crashed: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Step 5: report what we got.
    ours_logits = dec.logits_for_last().cpu()
    ours_top1 = int(ours_logits.argmax().item())
    ours_top1_text = tok.decode([ours_top1])
    print(f"\n[megakernel]    ours top-1 =  {ours_top1} ({ours_top1_text!r})")
    print(f"                ours max   =  {ours_logits.max().item():.4f}")
    print(f"                ours min   =  {ours_logits.min().item():.4f}")
    print(f"                isfinite?  =  {bool(torch.isfinite(ours_logits).all())}")

    # Sanity comparisons.
    cos = F.cosine_similarity(
        ref_logits.unsqueeze(0).to(torch.float32),
        ours_logits.unsqueeze(0).to(torch.float32),
        dim=-1).item()
    max_abs = (ref_logits - ours_logits).abs().max().item()
    print(f"\n[compare]       cos sim   =  {cos:.6f}")
    print(f"                max abs   =  {max_abs:.4f}")
    print(f"                top-1 match: {'YES' if ref_top1 == ours_top1 else 'NO'}")

    if ref_top1 == ours_top1 and cos > 0.999:
        print("\nC1 PASS: megakernel forward runs and matches HF on this token.")
    elif torch.isfinite(ours_logits).all():
        print("\nC1 PARTIAL: kernel ran without crash but logits drift from HF.")
        print("Next: enable layer-by-layer hidden-state capture (C2) and find "
              "the first-divergent layer.")
    else:
        print("\nC1 FAIL: kernel produced non-finite logits.")
        print("Next: check NaN/Inf injection at each layer; likely DN state init "
              "or conv ring-buffer indexing.")


if __name__ == "__main__":
    main()
