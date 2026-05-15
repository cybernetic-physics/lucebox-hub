"""P3 — Memory regression check.

Walks through the megakernel runtime construction with verbose=True and
records GPU allocation at each phase. Compares against a baseline if one
exists at `models/qwen36_27b/docs/results/memory_baseline.json`.

Phases captured:
  - empty           : at script start (CUDA context only)
  - hf_loaded       : after AutoModelForCausalLM.from_pretrained()
  - weights_unified : after _unify_from_hf_model
  - layer_packed    : after pack_layer_weights
  - scratch         : after alloc_scratch(max_seq=4096)

If the *current* peak exceeds the baseline by > 1 GB, fail with a non-
zero exit code so this can wedge into CI.

Run:
    HF_HOME=/home/sparkz/rl/.hf_cache \
        /home/sparkz/rl/.venv/bin/python3 \
        models/qwen36_27b/test/test_p3_memory.py
"""
from __future__ import annotations
import json, os, sys
import torch

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))

BASELINE = os.path.join(os.path.dirname(THIS),
                         "../docs/results/memory_baseline.json")
THRESHOLD_GB = 1.0


def _gb(): return torch.cuda.memory_allocated() / (1024 ** 3)


def main():
    measurements = {}
    torch.cuda.empty_cache()
    measurements["empty"] = _gb()
    print(f"phase=empty            alloc={_gb():.2f} GB")

    print("Loading HF Qwen3.6-27B (BF16)...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True).eval()
    measurements["hf_loaded"] = _gb()
    print(f"phase=hf_loaded        alloc={_gb():.2f} GB")

    from weight_packer import (_unify_from_hf_model, pack_layer_weights,
                                alloc_scratch)
    weights = _unify_from_hf_model(hf)
    measurements["weights_unified"] = _gb()
    print(f"phase=weights_unified  alloc={_gb():.2f} GB")

    blob = pack_layer_weights(weights["layer_data"])
    measurements["layer_packed"] = _gb()
    print(f"phase=layer_packed     alloc={_gb():.2f} GB")

    sc = alloc_scratch(max_seq=4096, verbose=False)
    measurements["scratch"] = _gb()
    print(f"phase=scratch          alloc={_gb():.2f} GB")

    print()

    # Compare to baseline, or write a new one if absent.
    if os.path.exists(BASELINE):
        with open(BASELINE) as f:
            base = json.load(f)
        print("Comparison vs baseline:")
        any_regression = False
        for phase, mem in measurements.items():
            b = base.get(phase, 0.0)
            delta = mem - b
            status = "OK"
            if delta > THRESHOLD_GB:
                status = f"REGRESSION (+{delta:.2f} GB > {THRESHOLD_GB})"
                any_regression = True
            print(f"  {phase:20s}  cur={mem:.2f} GB  base={b:.2f} GB  "
                  f"delta={delta:+.2f}  {status}")
        if any_regression:
            sys.exit(1)
        print("\nALL PHASES WITHIN BASELINE")
    else:
        with open(BASELINE, "w") as f:
            json.dump(measurements, f, indent=2)
        print(f"Wrote baseline -> {BASELINE}")


if __name__ == "__main__":
    main()
