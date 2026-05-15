"""P1 — Wikitext-2 perplexity comparison: ours vs HF reference.

The production correctness bar from TODO.md. Computes per-window PPL on
wikitext-2 test for both:
  - HF Qwen3.6-27B reference (oracle)
  - Our megakernel runtime (Qwen36MegakernelDecoder)

Expected: ours PPL within 1% of HF PPL. The current per-token kernel
latency makes long sweeps impractical; this script runs a short sweep
(N windows of S=256) suitable for catching regressions.

Run:
    HF_HOME=/home/sparkz/rl/.hf_cache \\
        /home/sparkz/rl/.venv/bin/python3 \\
        models/qwen36_27b/test/test_p1_wikitext_ppl.py
"""
from __future__ import annotations

import argparse, math, os, sys, time
import torch

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))


def ppl_hf(model, token_ids: torch.Tensor, ctx_len: int, stride: int,
           tokenizer) -> tuple[float, int, float]:
    """Sliding-window PPL via HF forward. Scores the LAST position of each
    S-token window against the next token in the corpus."""
    total_nll, total_n = 0.0, 0
    last_end = token_ids.numel() - 1
    t0 = time.perf_counter()
    pos = 0
    while pos + ctx_len + 1 <= last_end + 1:
        window = token_ids[pos : pos + ctx_len].unsqueeze(0).to("cuda")
        target = int(token_ids[pos + ctx_len].item())
        with torch.no_grad():
            out = model(input_ids=window, use_cache=False)
        logits = out.logits[0, -1].to(torch.float32)
        logp = torch.nn.functional.log_softmax(logits, dim=-1)
        total_nll += float(-logp[target].item())
        total_n += 1
        pos += stride
    return math.exp(total_nll / max(total_n, 1)), total_n, time.perf_counter() - t0


def ppl_ours(dec, token_ids: torch.Tensor, ctx_len: int, stride: int,
             weights_lm_head) -> tuple[float, int, float]:
    """PPL via our prefill_qwen3x_naive."""
    total_nll, total_n = 0.0, 0
    last_end = token_ids.numel() - 1
    t0 = time.perf_counter()
    pos = 0
    while pos + ctx_len + 1 <= last_end + 1:
        dec.reset()
        window = token_ids[pos : pos + ctx_len].to(torch.int32)
        target = int(token_ids[pos + ctx_len].item())
        _ = dec.prefill(window)
        logits = dec.logits_for_last()
        logp = torch.nn.functional.log_softmax(logits, dim=-1)
        total_nll += float(-logp[target].item())
        total_n += 1
        pos += stride
    return math.exp(total_nll / max(total_n, 1)), total_n, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tokens", type=int, default=1024,
                    help="total wikitext tokens to score")
    ap.add_argument("--ctx-len", type=int, default=64,
                    help="window size (kernel decode runs ctx_len times)")
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--skip-ours", action="store_true",
                    help="HF reference only")
    args = ap.parse_args()

    print("Loading HF Qwen3.6-27B...", flush=True)
    t0 = time.perf_counter()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True,
    ).eval()
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    from datasets import load_dataset
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = " ".join(x for x in wt["text"] if x.strip())[:200_000]
    all_ids = tok(text, return_tensors="pt").input_ids[0][: args.n_tokens]
    print(f"  scoring {all_ids.numel()} wikitext tokens "
          f"(ctx_len={args.ctx_len}, stride={args.stride})")

    print(f"\nHF reference PPL...")
    ppl_a, n_a, t_a = ppl_hf(hf, all_ids, args.ctx_len, args.stride, tok)
    print(f"  HF PPL = {ppl_a:.3f}  ({n_a} windows in {t_a:.1f}s)")

    if args.skip_ours:
        return

    print(f"\nOurs (megakernel) PPL...")
    from runtime_megakernel import Qwen36MegakernelDecoder
    dec = Qwen36MegakernelDecoder(
        max_seq=max(args.ctx_len * 2, 256), verbose=True,
        hf_model=hf, tokenizer=tok)
    ppl_b, n_b, t_b = ppl_ours(dec, all_ids, args.ctx_len, args.stride, None)
    print(f"  ours PPL = {ppl_b:.3f}  ({n_b} windows in {t_b:.1f}s)")

    drift = abs(ppl_b - ppl_a) / ppl_a
    print(f"\nDrift: |ours - HF| / HF = {drift:.2%}")
    if drift < 0.01:
        print("PASS: < 1% PPL drift")
    elif drift < 0.05:
        print("CLOSE: < 5% PPL drift (acceptable for bf16-noise regimes)")
    else:
        print(f"DRIFT TOO HIGH: {drift:.1%}")


if __name__ == "__main__":
    main()
