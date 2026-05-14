"""Verify weight_packer's HF key mapping against the actual Qwen3.6-27B
safetensors index. Runs in seconds — no model load.

If this passes, the key strings match; the actual load can then proceed.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))

from weight_packer import (
    NUM_LAYERS, LAYER_TYPE, _hf_keys_fa, _hf_keys_dn,
)


def _load_safetensors_index() -> set[str]:
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))
    cache = Path(hf_home) / "hub" / "models--Qwen--Qwen3.6-27B" / "snapshots"
    snaps = list(cache.glob("*/model.safetensors.index.json"))
    if not snaps:
        print(f"No Qwen3.6-27B safetensors index found under {cache}")
        sys.exit(2)
    with open(snaps[0]) as f:
        idx = json.load(f)
    return set(idx["weight_map"].keys())


def main():
    keys_in_safetensors = _load_safetensors_index()
    print(f"Total keys in safetensors: {len(keys_in_safetensors)}")

    missing: list[str] = []
    n_fa, n_dn = 0, 0
    for i in range(NUM_LAYERS):
        if LAYER_TYPE[i] == 1:
            n_fa += 1
            for name, key in _hf_keys_fa(i).items():
                if key not in keys_in_safetensors:
                    missing.append(f"FA layer {i} {name}: {key}")
        else:
            n_dn += 1
            for name, key in _hf_keys_dn(i).items():
                if key not in keys_in_safetensors:
                    missing.append(f"DN layer {i} {name}: {key}")
    print(f"Expected: {n_fa} FA + {n_dn} DN = {n_fa + n_dn} layers")

    # Embed + final norm + (optional) lm_head.
    for key in ["model.language_model.embed_tokens.weight",
                "model.language_model.norm.weight"]:
        if key not in keys_in_safetensors:
            missing.append(f"global: {key}")

    if missing:
        print(f"\n{len(missing)} MISSING KEYS:")
        for m in missing[:20]: print(f"  {m}")
        if len(missing) > 20: print(f"  ... and {len(missing) - 20} more")
        sys.exit(1)
    print("\nALL EXPECTED KEYS PRESENT IN SAFETENSORS")


if __name__ == "__main__":
    main()
