"""C7 — long-context correctness sweep on wikitext.

C6 validated top-1 match at S ≤ 16. C7 verifies the kernel scales to
longer sequences without divergence. Tests at S ∈ {32, 128, 512, 2048,
4096}. Uses wikitext-2 test windows for in-distribution prompts.

For each S:
  - Take a S-token slice of wikitext.
  - Run HF forward to get last-position logits.
  - Run our prefill_qwen3x_naive (host-loops decode S times).
  - Compare top-1, top-5 overlap, cos, max_abs, KL divergence.

Acceptance bar:
  - top-1 match on natural text.
  - cos >= 0.99 across all S.
  - KL <= 0.05 nats.

Run:
    HF_HOME=/home/sparkz/rl/.hf_cache \\
        /home/sparkz/rl/.venv/bin/python3 \\
        models/qwen36_27b/test/test_c7_longcontext.py
"""
from __future__ import annotations

import os, sys, time
import torch
import torch.nn.functional as F

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))


def main():
    print("Loading HF Qwen3.6-27B (shared between paths)...", flush=True)
    t0 = time.perf_counter()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True,
    ).eval()
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    # Load wikitext-2 test split, concatenate to a single long string.
    from datasets import load_dataset
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = " ".join(x for x in wt["text"] if x.strip())[:200_000]  # ~200K chars
    all_ids = tok(text, return_tensors="pt").input_ids[0].to("cuda")
    print(f"  wikitext source: {all_ids.numel()} tokens available")

    from runtime_megakernel import Qwen36MegakernelDecoder
    dec = Qwen36MegakernelDecoder(max_seq=8192, verbose=True, hf_model=hf, tokenizer=tok)

    print(f"\n{'S':>6}  {'top1_ref':>8}  {'top1_ours':>9}  {'match':>5}  "
          f"{'cos':>9}  {'max_abs':>8}  {'kl':>9}  {'top5_ov':>7}  ours_step_ms")
    results = []
    for S in [32, 128, 512, 2048, 4096]:
        if S > 8192: continue
        if all_ids.numel() < S: continue
        ids = all_ids[:S]
        # HF reference.
        with torch.no_grad():
            t0 = time.perf_counter()
            hf_out = hf(input_ids=ids.unsqueeze(0), use_cache=False)
            ref_ms = (time.perf_counter() - t0) * 1000
        ref_logits = hf_out.logits[0, -1].to(torch.float32).cpu()
        ref_top1 = int(ref_logits.argmax().item())
        ref_top5 = set(ref_logits.topk(5).indices.tolist())

        # Ours.
        dec.reset()
        t0 = time.perf_counter()
        ours_top1 = dec.prefill(ids.to(torch.int32))
        torch.cuda.synchronize()
        ours_ms = (time.perf_counter() - t0) * 1000
        ours_logits = dec.logits_for_last().cpu()
        ours_top5 = set(ours_logits.topk(5).indices.tolist())

        cos = F.cosine_similarity(
            ref_logits.unsqueeze(0), ours_logits.unsqueeze(0), dim=-1).item()
        max_abs = (ref_logits - ours_logits).abs().max().item()
        # KL(ref || ours).
        lp_ref  = F.log_softmax(ref_logits, dim=-1)
        lp_ours = F.log_softmax(ours_logits, dim=-1)
        kl = float((lp_ref.exp() * (lp_ref - lp_ours)).sum().item())
        top5_ov = len(ref_top5 & ours_top5) / 5

        match = "YES" if ref_top1 == ours_top1 else "NO"
        print(f"{S:>6}  {ref_top1:>8}  {ours_top1:>9}  {match:>5}  "
              f"{cos:>9.6f}  {max_abs:>8.3f}  {kl:>9.4f}  {top5_ov:>7.2f}  "
              f"{ours_ms:>6.0f} (HF {ref_ms:.0f})")
        results.append((S, ref_top1 == ours_top1, cos, kl))

    print()
    passes = sum(1 for _, m, c, kl in results if m and c > 0.99 and kl < 0.05)
    print(f"PASS: {passes}/{len(results)}  (top-1 match + cos>0.99 + KL<0.05)")
    if passes == len(results):
        print("ALL CHECKS PASS")
    else:
        print("Some checks failed — investigate longer-context drift")


if __name__ == "__main__":
    main()
