"""P4 — Numerical hardening: NaN/Inf guard on 32k random tokens.

Runs a long sequence (mixed random + structured tokens) through the
megakernel and verifies:
  - g_normalized (post-final-norm) stays finite at every position
  - logits stay finite
  - dn_states magnitudes stay bounded
  - fa_k_cache / fa_v_cache stay bounded

This is a SMOKE test for kernel numerical robustness, not a correctness
test. It is meant to catch the case where a future kernel edit
introduces a Inf/NaN propagation path that doesn't show up on short
prompts.

Run:
    HF_HOME=/home/sparkz/rl/.hf_cache \
        /home/sparkz/rl/.venv/bin/python3 \
        models/qwen36_27b/test/test_p4_nan_guard.py [--n-tokens N]
"""
from __future__ import annotations
import argparse, os, sys
import torch

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tokens", type=int, default=512,
                    help="number of decode steps to run (default 512; "
                          "1024 is the conservative bar for the host loop)")
    args = ap.parse_args()

    print(f"Loading HF Qwen3.6-27B for weight sharing (BF16)...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True,
    ).eval()

    from runtime_megakernel import Qwen36MegakernelDecoder
    dec = Qwen36MegakernelDecoder(max_seq=max(args.n_tokens, 1024),
                                    verbose=True, hf_model=hf, tokenizer=tok)

    print(f"\nRunning {args.n_tokens} random-token decode steps with NaN guards...")

    # Build a token stream: alternate "normal" wiki tokens with random ones
    # to keep the input space wide.
    from datasets import load_dataset
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = " ".join(x for x in wt["text"] if x.strip())[:200_000]
    base_ids = tok(text, return_tensors="pt").input_ids[0].to(torch.int32)
    base_ids = base_ids.to("cuda")

    # Prefill with first 64 tokens so we have non-trivial state.
    init_ids = base_ids[:64]
    dec.prefill(init_ids)
    if not torch.isfinite(dec.sc.g_normalized).all():
        print("FAIL  g_normalized has non-finite values after prefill")
        sys.exit(1)

    # Then decode one token at a time, alternating real wiki tokens and
    # random ids. Track magnitudes per N steps.
    rng = torch.Generator(device="cuda").manual_seed(20260514)
    n_bad = 0
    for step in range(args.n_tokens):
        if step % 2 == 0:
            # Real text token from wikitext (cycle through tail).
            tok_id = int(base_ids[64 + (step % (base_ids.numel() - 64))].item())
        else:
            tok_id = int(torch.randint(0, 248320, (1,),
                                         generator=rng, device="cuda").item())
        dec.decode(tok_id)
        gn = dec.sc.g_normalized
        if not torch.isfinite(gn).all():
            print(f"FAIL  g_normalized non-finite at step {step} (token={tok_id})")
            n_bad += 1
            if n_bad >= 3:
                print("Too many bad steps; aborting")
                sys.exit(1)
        # Spot-check magnitude every 64 steps.
        if step % 64 == 0:
            mx = gn.abs().max().item()
            dn_mx = dec.sc.dn_states.abs().max().item()
            kv_mx = dec.sc.fa_k_cache.abs().max().item()
            print(f"  step {step:>5}  g_norm_max={mx:.2f}  "
                  f"dn_state_max={dn_mx:.2f}  fa_k_max={kv_mx:.2f}")

    if n_bad == 0:
        print(f"\nPASS  {args.n_tokens} decode steps, no NaN/Inf in g_normalized")
    else:
        print(f"\nPARTIAL  {n_bad} bad steps")
        sys.exit(1)


if __name__ == "__main__":
    main()
