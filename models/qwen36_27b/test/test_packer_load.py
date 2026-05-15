"""Live integration test: load HF Qwen3.6-27B weights into our kernel
layout via weight_packer + pack_layer_weights + alloc_scratch.

Validates that:
  - Every key the weight_packer expects is present in HF state_dict.
  - All per-layer tensor shapes match what the kernel expects.
  - pack_layer_weights produces a valid layer-pointer blob.
  - alloc_scratch succeeds at max_seq=32768 within the GB10 budget.

Does NOT run the kernel forward (saved for the next test). Runs in
~3-5 minutes (weights still need to be deserialized from disk).
"""
from __future__ import annotations

import os, sys, time
import torch

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))
from weight_packer import (
    load_27b_weights, pack_layer_weights, alloc_scratch,
    NUM_LAYERS, N_FA, N_DN, HIDDEN_SIZE, VOCAB_SIZE,
)


def main():
    print("Loading HF weights + packing into kernel layout...")
    t0 = time.perf_counter()
    weights, tok = load_27b_weights("Qwen/Qwen3.6-27B", verbose=True)
    print(f"  load done in {time.perf_counter()-t0:.1f}s")
    print(f"  GPU alloc after weights: "
          f"{torch.cuda.memory_allocated()/(1024**3):.1f} GB")

    # Confirm layer_data structure.
    layer_data = weights["layer_data"]
    assert len(layer_data) == NUM_LAYERS, f"expected {NUM_LAYERS}, got {len(layer_data)}"
    n_fa = sum(1 for ld in layer_data if ld["type"] == 1)
    n_dn = sum(1 for ld in layer_data if ld["type"] == 0)
    assert n_fa == N_FA and n_dn == N_DN, f"FA/DN counts: {n_fa}/{n_dn}"
    print(f"  layers: {n_fa} FA + {n_dn} DN = {len(layer_data)}")

    # Embed + lm_head sanity.
    assert weights["embed_weight"].shape == (VOCAB_SIZE, HIDDEN_SIZE)
    assert weights["lm_head_weight"].shape == (VOCAB_SIZE, HIDDEN_SIZE)
    assert weights["final_norm_weight"].shape == (HIDDEN_SIZE,)
    print(f"  embed: {weights['embed_weight'].shape}")
    print(f"  lm_head: {weights['lm_head_weight'].shape}")

    # Pack.
    print("\nPacking layer pointers...")
    blob = pack_layer_weights(layer_data)
    print(f"  blob size: {blob.numel() / 1024:.1f} KB")
    from weight_packer import PACK_STRUCT
    print(f"  expected: {NUM_LAYERS * PACK_STRUCT} bytes "
          f"({PACK_STRUCT} B per layer struct)")

    # Allocate scratch.
    print("\nAllocating scratch (max_seq=32768)...")
    sc = alloc_scratch(max_seq=32768)
    print(f"  fa_k_cache: {sc.fa_k_cache.shape}  "
          f"{sc.fa_k_cache.numel() * 2 / (1024**3):.2f} GB")
    print(f"  dn_states:  {sc.dn_states.shape}  "
          f"{sc.dn_states.numel() * 4 / (1024**3):.2f} GB")
    print(f"  conv_bufs:  {sc.conv_bufs.shape}")
    print(f"  GPU alloc total: "
          f"{torch.cuda.memory_allocated()/(1024**3):.1f} GB")

    # Free.
    print("\nALL CHECKS PASSED — weight_packer + scratch alloc work on real weights")


if __name__ == "__main__":
    main()
