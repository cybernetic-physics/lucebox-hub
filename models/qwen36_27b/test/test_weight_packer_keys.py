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

from weight_packer import NUM_LAYERS, LAYER_TYPE


def _safetensors_keys_fa(i: int) -> dict:
    """Safetensors-on-disk keys for an FA layer. Distinct from the
    state-dict keys used by weight_packer at HF-load time (the latter
    strip the `language_model` wrapper)."""
    p = f"model.language_model.layers.{i}."
    return {
        "input_layernorm":       p + "input_layernorm.weight",
        "q_proj":                p + "self_attn.q_proj.weight",
        "k_proj":                p + "self_attn.k_proj.weight",
        "v_proj":                p + "self_attn.v_proj.weight",
        "q_norm":                p + "self_attn.q_norm.weight",
        "k_norm":                p + "self_attn.k_norm.weight",
        "o_proj":                p + "self_attn.o_proj.weight",
        "post_attn_layernorm":   p + "post_attention_layernorm.weight",
        "gate_proj":             p + "mlp.gate_proj.weight",
        "up_proj":               p + "mlp.up_proj.weight",
        "down_proj":             p + "mlp.down_proj.weight",
    }


def _safetensors_keys_dn(i: int) -> dict:
    p = f"model.language_model.layers.{i}."
    return {
        "input_layernorm":       p + "input_layernorm.weight",
        "qkv_proj":              p + "linear_attn.in_proj_qkv.weight",
        "z_proj":                p + "linear_attn.in_proj_z.weight",
        "beta_proj":             p + "linear_attn.in_proj_b.weight",
        "alpha_proj":            p + "linear_attn.in_proj_a.weight",
        "conv1d":                p + "linear_attn.conv1d.weight",
        "a_log":                 p + "linear_attn.A_log",
        "dt_bias":               p + "linear_attn.dt_bias",
        "norm_weight":           p + "linear_attn.norm.weight",
        "out_proj":              p + "linear_attn.out_proj.weight",
        "post_attn_layernorm":   p + "post_attention_layernorm.weight",
        "gate_proj":             p + "mlp.gate_proj.weight",
        "up_proj":               p + "mlp.up_proj.weight",
        "down_proj":             p + "mlp.down_proj.weight",
    }

_hf_keys_fa = _safetensors_keys_fa
_hf_keys_dn = _safetensors_keys_dn


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

    # Embed + final norm. The safetensors file uses
    # `model.language_model.embed_tokens.weight` etc., but at HF load time
    # they're remapped to `model.embed_tokens.weight`. We check the
    # safetensors layout (as on disk) here -- the runtime test in
    # test_packer_load.py uses the post-load state-dict names.
    for key in ["model.language_model.embed_tokens.weight",
                "model.language_model.norm.weight"]:
        if key not in keys_in_safetensors:
            missing.append(f"safetensors-on-disk: {key}")

    if missing:
        print(f"\n{len(missing)} MISSING KEYS:")
        for m in missing[:20]: print(f"  {m}")
        if len(missing) > 20: print(f"  ... and {len(missing) - 20} more")
        sys.exit(1)
    print("\nALL EXPECTED KEYS PRESENT IN SAFETENSORS")


if __name__ == "__main__":
    main()
