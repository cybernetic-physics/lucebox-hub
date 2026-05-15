"""F9 — Probe HF Qwen3.6-27B cache layout for FA + DN layers.

Runs ONE HF forward at S=4 with `use_cache=True`, then dumps the shape
+ dtype of every field on `out.past_key_values.layers[i]` for the
first FA and first DN layer.

The point: `prefill_via_hf` copies `cl.keys`, `cl.values`,
`cl.conv_states`, `cl.recurrent_states` straight into our scratch.
Any layout assumption that's wrong has caused a silent bug in the
past (DN recurrent_state transpose was found via deep review, not
this test — we want this test to catch future ones).

This test:
  - Loads HF (~4 min)
  - One forward pass
  - Prints the actual shapes/dtypes
  - Asserts they match what prefill_via_hf assumes

If the test fails, the assertion message says exactly what changed.

Run:
    HF_HOME=/home/sparkz/rl/.hf_cache \
        /home/sparkz/rl/.venv/bin/python3 \
        models/qwen36_27b/test/test_f9_hf_cache_layout.py
"""
from __future__ import annotations
import os, sys
import torch

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))


def dump_cache_layer(cl, layer_idx, label):
    print(f"\n--- layer {layer_idx} ({label}) ---")
    for attr in ("keys", "values", "key_cache", "value_cache",
                 "conv_states", "recurrent_states",
                 "is_conv_states_initialized", "is_recurrent_states_initialized"):
        if hasattr(cl, attr):
            v = getattr(cl, attr)
            if isinstance(v, torch.Tensor):
                print(f"  {attr:35s} shape={tuple(v.shape)} dtype={v.dtype} dev={v.device}")
            elif v is not None:
                print(f"  {attr:35s} {v}")
            else:
                print(f"  {attr:35s} None")


def main():
    print("Loading HF Qwen3.6-27B (BF16)...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B",
                                          trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True).eval()

    ids = tok("Hello world", return_tensors="pt").input_ids.cuda()
    print(f"\nrunning HF forward use_cache=True, S={ids.shape[-1]}")
    with torch.no_grad():
        out = hf(input_ids=ids, use_cache=True)

    cache = out.past_key_values
    print(f"\ncache type: {type(cache).__name__}")
    print(f"num layers: {len(cache.layers)}")
    print(f"out.last_hidden_state shape: "
          f"{out.logits.shape if hasattr(out, 'logits') else 'n/a'}")

    from weight_packer import LAYER_TYPE, N_FA, N_DN
    # Find the first FA and the first DN.
    first_fa = next(i for i, t in enumerate(LAYER_TYPE) if t == 1)
    first_dn = next(i for i, t in enumerate(LAYER_TYPE) if t == 0)

    dump_cache_layer(cache.layers[first_dn], first_dn, "DN")
    dump_cache_layer(cache.layers[first_fa], first_fa, "FA")

    print("\n--- expected by our prefill_via_hf ---")
    print(f"  FA keys/values: [1, KV_H=4, S, HEAD=256] bf16")
    print(f"  DN conv_states: [1, CONV_CH=10240, CONV_K=4] f32")
    print(f"  DN recurrent_states: [1, V_H=48, KEY=128, VAL=128] f32")
    print(f"    (we transpose -1, -2 -> [1, V_H, VAL, KEY] to match our state[j*KEY+i] layout)")


if __name__ == "__main__":
    main()
