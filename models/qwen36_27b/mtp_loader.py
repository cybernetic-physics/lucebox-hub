"""S5 — manual MTP-head loader for Qwen3.6-27B.

HF discards `mtp.*` keys at load time (see modeling_qwen3_5.py:794:
`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`), so we cannot
access the MTP head via `model.state_dict()`. This module bypasses HF
and loads MTP weights directly from the safetensors shards.

Usage:
    from mtp_loader import load_mtp_head
    mtp = load_mtp_head("Qwen/Qwen3.6-27B")
    # mtp is a dict keyed by raw safetensors key (e.g. "mtp.layers.0.q_proj.weight")
    # mapped to the BF16 CUDA tensor.

After loading, the next step is wiring `mtp_speculative.MTPDecoder.
_mtp_predict` to use these tensors. That requires knowing the exact
forward pass of the Qwen3.6 NEXTN head, which is documented in the
official Qwen3.6 paper but not in HF source (since HF discards it).

Discovery: run `test/test_s5_mtp_probe.py` to see all safetensors keys
matching mtp/nextn patterns + their shapes. Update MTP_KEY_PREFIX +
the forward-pass logic below to match.
"""
from __future__ import annotations
import os
from typing import Dict

import torch


MTP_KEY_PREFIX = "mtp"  # safetensors uses this prefix per the HF ignore-list regex


def _find_safetensors_dir(model_name: str) -> str:
    """Locate the snapshot dir under HF_HOME that contains the .safetensors
    shards. Mirrors huggingface_hub's snapshot_download cache layout."""
    hf_home = os.environ.get(
        "HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    # `Qwen/Qwen3.6-27B` -> models--Qwen--Qwen3.6-27B
    fs_name = "models--" + model_name.replace("/", "--")
    snapshots_dir = os.path.join(hf_home, "hub", fs_name, "snapshots")
    if not os.path.isdir(snapshots_dir):
        raise FileNotFoundError(
            f"snapshot dir not found: {snapshots_dir}. "
            f"Pre-download with AutoModelForCausalLM.from_pretrained "
            f"first.")
    revs = sorted(os.listdir(snapshots_dir))
    if not revs:
        raise FileNotFoundError(f"no revisions under {snapshots_dir}")
    return os.path.join(snapshots_dir, revs[-1])


def load_mtp_head(model_name: str = "Qwen/Qwen3.6-27B",
                   device: str = "cuda",
                   dtype: torch.dtype = torch.bfloat16,
                   ) -> Dict[str, torch.Tensor]:
    """Load every safetensors tensor whose key starts with MTP_KEY_PREFIX.

    Returns a dict of {raw_key: tensor}. Empty dict if the checkpoint
    doesn't ship MTP weights (rare but possible).
    """
    from safetensors import safe_open

    snap = _find_safetensors_dir(model_name)
    shards = [os.path.join(snap, f) for f in os.listdir(snap)
              if f.endswith(".safetensors")]
    if not shards:
        raise FileNotFoundError(f"no .safetensors shards in {snap}")

    out = {}
    for shard in shards:
        with safe_open(shard, framework="pt", device=device) as f:
            for key in f.keys():
                if key.startswith(MTP_KEY_PREFIX):
                    t = f.get_tensor(key)
                    if t.dtype != dtype:
                        t = t.to(dtype)
                    out[key] = t
    return out


if __name__ == "__main__":
    import sys
    mtp = load_mtp_head()
    if not mtp:
        print("No MTP keys found in the checkpoint.")
        sys.exit(1)
    print(f"Loaded {len(mtp)} MTP tensors. Layout:")
    for k in sorted(mtp.keys()):
        t = mtp[k]
        print(f"  {k:60s}  {tuple(t.shape)}  {t.dtype}")
    # Total bytes.
    total = sum(t.numel() * t.element_size() for t in mtp.values())
    print(f"\ntotal: {total/(1024**2):.1f} MB")
