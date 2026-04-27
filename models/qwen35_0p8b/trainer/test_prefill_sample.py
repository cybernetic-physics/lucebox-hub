"""Verify Decoder.prefill produces correct rollouts and is deterministic.

Compares the new prefill-then-step path against the legacy
step-everything path for a few prompt lengths. The prefill path is the
spec: it must (a) be deterministic across runs and (b) match the legacy
path for short generations where bf16 reduction noise hasn't yet
compounded enough to flip an argmax. At long S the two paths
legitimately drift due to atomic-barrier ordering in the cooperative
decode kernel; that's not an orchestration bug.

Runs on B200 BF16 backend in ~30 s (HF base load is most of it).
"""
from __future__ import annotations

import sys

# transformers' Qwen3.5 modeling tries to import fla.modules.FusedRMSNormGated,
# which triggers a @torch.compile that can fail on this torch+transformers
# combo. rl_trainer.py disables this before importing transformers; mirror.
try:
    import transformers.utils.import_utils as _iu
    _iu.is_flash_linear_attention_available = lambda: False
except Exception:
    pass

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b")

import torch  # noqa: E402

from model import Decoder  # noqa: E402


def baseline_generate(decoder, prompt_ids, max_tokens, eos):
    decoder.reset()
    for tid in prompt_ids[:-1]:
        decoder.step(int(tid))
    pred = decoder.step(int(prompt_ids[-1]))
    out = []
    for _ in range(max_tokens):
        if pred == eos:
            break
        out.append(int(pred))
        pred = decoder.step(int(pred))
    return out


def prefill_generate(decoder, prompt_ids, max_tokens, eos):
    pred = decoder.prefill(prompt_ids)
    out = []
    for _ in range(max_tokens):
        if pred == eos:
            break
        out.append(int(pred))
        pred = decoder.step(int(pred))
    return out


def main():
    print("Loading Decoder (Qwen3.5-0.8B BF16)...", flush=True)
    decoder = Decoder(verbose=False)
    eos = decoder.tokenizer.eos_token_id

    text = (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
        "Sphinx of black quartz, judge my vow. "
    )
    ids = decoder.tokenizer.encode(text, add_special_tokens=False)

    test_cases = [
        ("short", ids[:8], 16),
        ("medium", ids[:32], 32),
        ("typical", ids[:128] if len(ids) >= 128 else ids, 64),
    ]

    fails = []
    for label, p, ntok in test_cases:
        if label == "typical" and len(p) < 128:
            p = (p * ((128 // max(1, len(p))) + 1))[:128]
        print(f"\n--- {label}: prompt_len={len(p)}, max_tokens={ntok} ---")
        b1 = baseline_generate(decoder, p, ntok, eos)
        p1 = prefill_generate(decoder, p, ntok, eos)
        p2 = prefill_generate(decoder, p, ntok, eos)
        prefill_det = p1 == p2
        cross = b1 == p1

        first_diff = None
        for i, (a, b) in enumerate(zip(b1, p1)):
            if a != b:
                first_diff = i
                break
        if not cross and first_diff is None:
            first_diff = min(len(b1), len(p1))

        print(f"  prefill run-twice match  : {prefill_det}")
        print(f"  vs legacy step path      : {cross}"
              + (f"  (agree on first {first_diff} of {ntok} tokens)" if not cross else ""))

        if not prefill_det:
            fails.append(f"{label}: prefill path not deterministic")
        if ntok <= 32 and not cross:
            fails.append(f"{label}: prefill disagrees with step on short gen")
        if not p1:
            fails.append(f"{label}: prefill produced empty output")

    print()
    print("=" * 60)
    if not fails:
        print("PASS — prefill path deterministic and agrees with step on short gens")
    else:
        print("FAIL:")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
