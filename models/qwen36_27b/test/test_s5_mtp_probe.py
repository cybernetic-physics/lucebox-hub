"""S5 probe — inspect Qwen3.6-27B safetensors for MTP / NEXTN head keys.

The mtp_speculative.py placeholder uses LM-head argmax k times. To wire
the real MTP head we need to know the exact state_dict layout. This
script loads the HF model and prints any key whose name suggests MTP /
NEXTN involvement, plus the shape of each.

Output is meant to be saved into models/qwen36_27b/docs/mtp_layout.md
so the next session can write the loader without re-running the load.

Run:
    HF_HOME=/home/sparkz/rl/.hf_cache \
        /home/sparkz/rl/.venv/bin/python3 \
        models/qwen36_27b/test/test_s5_mtp_probe.py
"""
from __future__ import annotations
import os, sys
import torch

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))


def main():
    print("Loading HF Qwen3.6-27B (BF16) for state-dict inspection...")
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True).eval()
    sd = dict(hf.state_dict())

    patterns = ("mtp", "MTP", "nextn", "NEXTN", "predict", "draft")
    matched = sorted(k for k in sd.keys() if any(p in k for p in patterns))

    if not matched:
        print("\nNo MTP-related keys found in state_dict.")
        print("All top-level keys:")
        roots = sorted(set(k.split(".")[0] for k in sd.keys()))
        for r in roots:
            print(f"  {r}")
        return

    print(f"\nFound {len(matched)} MTP-related keys:")
    for k in matched:
        t = sd[k]
        print(f"  {k:60s}  {tuple(t.shape)}  {t.dtype}")

    # Also report the model attributes — sometimes the layout is in
    # model.model.mtp_head rather than model.mtp_*.
    print("\nModel module structure (top 2 levels):")
    def walk(mod, prefix="", depth=0):
        if depth > 2: return
        for name, child in mod.named_children():
            print(f"  {'  '*depth}{prefix}{name}: {type(child).__name__}")
            walk(child, prefix=name+".", depth=depth+1)
    walk(hf)


if __name__ == "__main__":
    main()
