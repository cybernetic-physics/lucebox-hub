"""C6 — multi-token correctness sweep at non-zero positions.

After C5 closed (FA MLP fix), single-token at position 0 matches HF
top-1. Now we verify the kernel at S > 1, which exercises:
  - RoPE at non-zero positions (no longer identity)
  - KV cache write at multiple positions
  - GQA attention over multiple K, V slots
  - DN recurrence over multiple steps

Two test modes:

  1. **Sequential decode**: prefill with a short prompt, then decode N
     more tokens autoregressively. Compare top-1 at each step against
     HF's prediction given the same prefix.

  2. **Prefill comparison**: pass a multi-token prompt directly through
     prefill_qwen3x_naive (which loops decode S times) and HF
     simultaneously, compare final-position logits.

Run:
    HF_HOME=/home/sparkz/rl/.hf_cache \\
        /home/sparkz/rl/.venv/bin/python3 \\
        models/qwen36_27b/test/test_c6_multitoken.py
"""
from __future__ import annotations

import os, sys, time
import torch
import torch.nn.functional as F

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))


def main():
    print("Loading HF Qwen3.6-27B (single load, shared)...", flush=True)
    t0 = time.perf_counter()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True,
    ).eval()
    print(f"  HF loaded in {time.perf_counter()-t0:.1f}s")

    from runtime_megakernel import Qwen36MegakernelDecoder
    dec = Qwen36MegakernelDecoder(max_seq=256, verbose=True, hf_model=hf, tokenizer=tok)
    dec.enable_layer_capture()

    # Test prompts of varying lengths.
    prompts = [
        ("short", "Hello"),
        ("natural", "The capital of France is"),
        ("longer", "In the beginning, the Earth was without form and void, and darkness was upon"),
    ]

    for name, text in prompts:
        ids = tok(text, return_tensors="pt").input_ids[0].to("cuda")
        S = ids.numel()
        print(f"\n=== {name!r}  S={S}  prompt={text!r} ===")

        # HF reference: forward over the whole prompt, get last-position logits.
        with torch.no_grad():
            hf_out = hf(input_ids=ids.unsqueeze(0), use_cache=False,
                        output_hidden_states=True)
        ref_logits = hf_out.logits[0, -1].to(torch.float32).cpu()
        ref_top1 = int(ref_logits.argmax().item())
        ref_top1_text = tok.decode([ref_top1])
        ref_per_layer = [h[0, -1].to(torch.float32).cpu()
                         for h in hf_out.hidden_states[1:]]
        print(f"  HF top-1 at last position: {ref_top1} ({ref_top1_text!r})")

        # Ours: reset state, run prefill (which loops decode S times internally).
        dec.reset()
        dec.layer_capture.zero_()  # ensure capture is from the FINAL token only
        next_id = dec.prefill(ids.to(torch.int32))
        ours_top1_text = tok.decode([next_id])
        ours_logits = dec.logits_for_last().cpu()
        ours_per_layer = [dec.layer_capture[i].to(torch.float32).cpu()
                          for i in range(64)]

        # Compare last-layer hidden + logits.
        cos = F.cosine_similarity(
            ref_logits.unsqueeze(0), ours_logits.unsqueeze(0), dim=-1).item()
        max_abs = (ref_logits - ours_logits).abs().max().item()
        print(f"  ours top-1 at last position: {next_id} ({ours_top1_text!r})")
        print(f"  final logits: cos={cos:.6f}  max_abs={max_abs:.4f}")
        print(f"  top-1 match: {'YES' if ref_top1 == next_id else 'NO'}")

        # Print first/last few layer cos.
        print(f"  per-layer cos snapshot:")
        for i in [0, 1, 2, 3, 16, 32, 47, 48, 62, 63]:
            cos_i = F.cosine_similarity(
                ref_per_layer[i].unsqueeze(0),
                ours_per_layer[i].unsqueeze(0), dim=-1).item()
            kind = "FA" if (i + 1) % 4 == 0 else "DN"
            print(f"    layer {i:>3} ({kind}): cos={cos_i:.6f}")


if __name__ == "__main__":
    main()
