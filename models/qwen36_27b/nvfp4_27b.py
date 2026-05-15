"""NVFP4 KV cache + NVFP4 weight quantization wiring for Qwen3.6-27B.

The 0.8B megakernel already ships:

  - NVFP4 KV cache (`models/qwen35_0p8b/nvfp4_kv_test.cu`)
      head_dim=256 hardcoded; ops registered under
      `torch.ops.qwen35_megakernel_bf16_C.quantize_bf16_to_nvfp4_kv` etc.

  - Optimal-MSE NVFP4 weight quantizer (`models/qwen35_0p8b/model.py:
      _optimal_quantize_matrix_nvfp4`) — row-chunked + torch.bucketize,
      handles arbitrary shapes.

Both apply to 27B unchanged. This module is just the Python plumbing:

  - alloc_nvfp4_kv_cache_27b()  -- allocate the packed KV buffers
                                    at 27B's shapes (16 FA layers, 4 KV
                                    heads, head=256).
  - quantize_27b_weights()      -- run the optimal-MSE quantizer over
                                    every 27B projection. Produces a
                                    parallel `layer_data_nvfp4` list.
  - memory_estimate_nvfp4()     -- estimate the GB savings vs BF16.

Footprint at max_seq=32768:
  bf16 KV : 2 GB
  nvfp4 KV: ~570 MB    (3.5x)

Footprint at max_seq=262144 (native context):
  bf16 KV : 16 GB
  nvfp4 KV: ~4.6 GB    (3.5x)

27B weight footprint:
  bf16    : ~50 GB
  nvfp4   : ~14 GB     (3.6x)  -- fits on a 24 GB consumer card with KV
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Any

import torch

# Make the 0.8B-side NVFP4 ops importable.
_PKG_0P8B = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "qwen35_0p8b")
sys.path.insert(0, _PKG_0P8B)
import qwen35_megakernel_bf16_C  # noqa: F401  registers torch.ops

from weight_packer import (
    NUM_LAYERS, HIDDEN_SIZE, INTERMEDIATE_SIZE,
    FA_NUM_Q_HEADS, FA_NUM_KV_HEADS, FA_HEAD_DIM,
    FA_QPROJ_SIZE, FA_Q_SIZE, FA_KV_SIZE,
    DN_NUM_V_HEADS, DN_NUM_QK_HEADS, DN_HEAD_DIM,
    DN_QK_SIZE, DN_V_SIZE, DN_CONV_CH, DN_CONV_KERNEL,
    LAYER_TYPE, N_FA, N_DN, VOCAB_SIZE,
)

KV_GROUP_SIZE  = 16
KV_DATA_BYTES  = FA_HEAD_DIM // 2          # 128
KV_SCALE_BYTES = FA_HEAD_DIM // KV_GROUP_SIZE  # 16

# NVFP4 weight group size (matches the 0.8B optimal-MSE quantizer default).
NVFP4_GROUP_SIZE = 32


# ---------------------------------------------------------------------------
# KV cache (re-uses the 0.8B helpers — same head_dim=256)
# ---------------------------------------------------------------------------

@dataclass
class NVFP4KVCache:
    k_data:   torch.Tensor   # [N_FA, KV_H, max_seq, KV_DATA_BYTES]   uint8
    k_scales: torch.Tensor   # [N_FA, KV_H, max_seq, KV_SCALE_BYTES]  uint8 (E4M3)
    v_data:   torch.Tensor
    v_scales: torch.Tensor
    max_seq:  int


def alloc_nvfp4_kv_cache_27b(max_seq: int = 32768) -> NVFP4KVCache:
    """Allocate packed NVFP4 K + V caches for the 27B FA layers."""
    u8 = dict(dtype=torch.uint8, device="cuda")
    return NVFP4KVCache(
        k_data   = torch.zeros(N_FA, FA_NUM_KV_HEADS, max_seq, KV_DATA_BYTES,  **u8),
        k_scales = torch.zeros(N_FA, FA_NUM_KV_HEADS, max_seq, KV_SCALE_BYTES, **u8),
        v_data   = torch.zeros(N_FA, FA_NUM_KV_HEADS, max_seq, KV_DATA_BYTES,  **u8),
        v_scales = torch.zeros(N_FA, FA_NUM_KV_HEADS, max_seq, KV_SCALE_BYTES, **u8),
        max_seq  = max_seq,
    )


def quantize_kv_step(bf16_k_per_head: torch.Tensor, bf16_v_per_head: torch.Tensor,
                     cache: NVFP4KVCache, layer_idx: int, position: int):
    """Pack one step's (K, V) into the NVFP4 cache. Inputs are
    [KV_HEADS, HEAD_DIM] bf16 (post-norm, post-RoPE). The helper kernel
    expects [T, H, D] so we reshape with T=1."""
    k = bf16_k_per_head.view(1, FA_NUM_KV_HEADS, FA_HEAD_DIM).contiguous()
    v = bf16_v_per_head.view(1, FA_NUM_KV_HEADS, FA_HEAD_DIM).contiguous()
    kd = cache.k_data[layer_idx, :, position:position+1].view(1, FA_NUM_KV_HEADS, KV_DATA_BYTES).contiguous()
    ks = cache.k_scales[layer_idx, :, position:position+1].view(1, FA_NUM_KV_HEADS, KV_SCALE_BYTES).contiguous()
    vd = cache.v_data[layer_idx, :, position:position+1].view(1, FA_NUM_KV_HEADS, KV_DATA_BYTES).contiguous()
    vs = cache.v_scales[layer_idx, :, position:position+1].view(1, FA_NUM_KV_HEADS, KV_SCALE_BYTES).contiguous()
    ops = torch.ops.qwen35_megakernel_bf16_C
    ops.quantize_bf16_to_nvfp4_kv(k, kd, ks)
    ops.quantize_bf16_to_nvfp4_kv(v, vd, vs)
    # Copy results back into the strided cache slice.
    cache.k_data[layer_idx, :, position:position+1].copy_(
        kd.view(FA_NUM_KV_HEADS, 1, KV_DATA_BYTES).permute(1, 0, 2))
    cache.k_scales[layer_idx, :, position:position+1].copy_(
        ks.view(FA_NUM_KV_HEADS, 1, KV_SCALE_BYTES).permute(1, 0, 2))
    cache.v_data[layer_idx, :, position:position+1].copy_(
        vd.view(FA_NUM_KV_HEADS, 1, KV_DATA_BYTES).permute(1, 0, 2))
    cache.v_scales[layer_idx, :, position:position+1].copy_(
        vs.view(FA_NUM_KV_HEADS, 1, KV_SCALE_BYTES).permute(1, 0, 2))


# ---------------------------------------------------------------------------
# Weight quantization (re-uses the 0.8B optimal-MSE quantizer)
# ---------------------------------------------------------------------------

def _import_quantizer():
    """Pull `_optimal_quantize_matrix_nvfp4` from the 0.8B model.py without
    importing the rest of that module (which sets up Decoder etc)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "qwen35_0p8b_model_quant",
        os.path.join(_PKG_0P8B, "model.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._optimal_quantize_matrix_nvfp4


DEFAULT_NVFP4_CACHE = os.path.join(
    os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
    "qwen3x_nvfp4_27b_cache.pt")


def quantize_27b_weights(layer_data: list[dict],
                         group_size: int = NVFP4_GROUP_SIZE,
                         verbose: bool = True,
                         cache_path: str | None = DEFAULT_NVFP4_CACHE) -> list[dict]:
    """Quantize every 2D projection weight in `layer_data` to NVFP4
    (packed uint8 + fp16 scales). 1D weights (norms, biases) stay bf16.

    Returns a parallel `layer_data_nvfp4` whose `ptrs` are a mix of
    bf16 tensors (norms) and (packed, scales) tuples (projections).

    If `cache_path` is given (default: ~/.cache/huggingface/...), the
    packed tensors are loaded from disk on subsequent invocations
    instead of recomputed (~30+ min savings).
    """
    if cache_path and os.path.exists(cache_path):
        if verbose: print(f"  [nvfp4] loading cached quantized weights "
                          f"<- {cache_path}", flush=True)
        cached = torch.load(cache_path, map_location="cuda", weights_only=False)
        if cached.get("group_size") != group_size:
            if verbose: print(f"  [nvfp4] cache group_size mismatch "
                              f"({cached.get('group_size')} vs {group_size}); "
                              f"re-quantizing")
        elif len(cached.get("layers", [])) != len(layer_data):
            if verbose: print(f"  [nvfp4] cache layer count mismatch; "
                              f"re-quantizing")
        else:
            # Rebuild the output layer_data structure from cached tensors.
            out = []
            for i, ld in enumerate(layer_data):
                cl = cached["layers"][i]
                new_ptrs = []
                for j, t in enumerate(ld["ptrs"]):
                    if t.dim() == 2 and t.shape[-1] >= group_size:
                        data, scales = cl[j]
                        new_ptrs.append((data, scales))
                    else:
                        new_ptrs.append(t)
                out.append({"type": ld["type"], "ptrs": new_ptrs})
            if verbose: print(f"  [nvfp4] cache hit; {len(out)} layers loaded")
            return out

    quantize = _import_quantizer()
    out = []
    cache_layers = []
    for i, ld in enumerate(layer_data):
        new_ptrs = []
        cache_ptrs = []
        for j, t in enumerate(ld["ptrs"]):
            if t.dim() == 2 and t.shape[-1] >= group_size:
                packed = quantize(t, group_size)
                data = packed["packed"].cuda().contiguous()
                scales = packed["scales"].cuda().contiguous()
                new_ptrs.append((data, scales))
                cache_ptrs.append((data, scales))
            else:
                new_ptrs.append(t)
                cache_ptrs.append(None)
        out.append({"type": ld["type"], "ptrs": new_ptrs})
        cache_layers.append(cache_ptrs)
        if verbose and (i % 8 == 0 or i == len(layer_data) - 1):
            print(f"  quantized layer {i+1}/{len(layer_data)}")

    if cache_path:
        if verbose: print(f"  [nvfp4] writing cache -> {cache_path}", flush=True)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({"group_size": group_size, "layers": cache_layers},
                    cache_path)
    return out


def quantize_lm_head_nvfp4(lm_head_weight: torch.Tensor,
                            group_size: int = NVFP4_GROUP_SIZE,
                            cache_path: str | None = None,
                            verbose: bool = True) -> dict:
    """Quantize the LM head [VOCAB, HIDDEN] to NVFP4.
    Returns {"packed": uint8 [VOCAB, HIDDEN/2], "scales": fp16 [VOCAB, HIDDEN/G]}.
    Optionally cached separately from the layer cache."""
    if cache_path and os.path.exists(cache_path):
        if verbose: print(f"  [nvfp4-lm] loading cached <- {cache_path}", flush=True)
        cached = torch.load(cache_path, map_location="cuda", weights_only=False)
        if (cached.get("group_size") == group_size
                and cached["packed"].shape[0] == lm_head_weight.shape[0]):
            return {"packed": cached["packed"], "scales": cached["scales"]}
        if verbose: print(f"  [nvfp4-lm] cache mismatch, re-quantizing")

    quantize = _import_quantizer()
    if verbose: print(f"  [nvfp4-lm] quantizing lm_head "
                       f"{tuple(lm_head_weight.shape)} -> NVFP4 ...", flush=True)
    packed = quantize(lm_head_weight, group_size)
    out = {"packed": packed["packed"].cuda().contiguous(),
           "scales": packed["scales"].cuda().contiguous()}

    if cache_path:
        if verbose: print(f"  [nvfp4-lm] writing cache -> {cache_path}", flush=True)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({"group_size": group_size, **out}, cache_path)
    return out


DEFAULT_NVFP4_LM_CACHE = os.path.join(
    os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
    "qwen3x_nvfp4_27b_lm_head_cache.pt")


# ---------------------------------------------------------------------------
# Footprint estimator
# ---------------------------------------------------------------------------

def memory_estimate_nvfp4(max_seq: int = 32768) -> dict:
    g = lambda b: b / (1024 ** 3)
    kv = 2 * N_FA * FA_NUM_KV_HEADS * max_seq * (KV_DATA_BYTES + KV_SCALE_BYTES)
    # NVFP4 weights at group 32: 4 bits per element + 8 bits per 32 elements
    # = 4 + 8/32 = 4.25 bits = 0.53125 bytes per element.
    n_params = 0
    # FA layers: 11 projections per layer + small norms (treated as bf16).
    for i in range(NUM_LAYERS):
        if LAYER_TYPE[i] == 1:
            n_params += (FA_QPROJ_SIZE * HIDDEN_SIZE  # q (q+gate)
                       + FA_KV_SIZE * HIDDEN_SIZE     # k
                       + FA_KV_SIZE * HIDDEN_SIZE     # v
                       + HIDDEN_SIZE * FA_Q_SIZE      # o
                       + INTERMEDIATE_SIZE * HIDDEN_SIZE  # gate
                       + INTERMEDIATE_SIZE * HIDDEN_SIZE  # up
                       + HIDDEN_SIZE * INTERMEDIATE_SIZE) # down
        else:
            n_params += (DN_CONV_CH * HIDDEN_SIZE      # qkv
                       + DN_V_SIZE * HIDDEN_SIZE        # z
                       + DN_NUM_V_HEADS * HIDDEN_SIZE   # beta
                       + DN_NUM_V_HEADS * HIDDEN_SIZE   # alpha
                       + HIDDEN_SIZE * DN_V_SIZE        # out
                       + INTERMEDIATE_SIZE * HIDDEN_SIZE
                       + INTERMEDIATE_SIZE * HIDDEN_SIZE
                       + HIDDEN_SIZE * INTERMEDIATE_SIZE)
    n_params += 2 * VOCAB_SIZE * HIDDEN_SIZE  # embed + lm_head
    weights_nvfp4 = n_params * 0.53125
    return dict(
        weights_nvfp4 = g(weights_nvfp4),
        kv_cache_nvfp4 = g(kv),
        total_nvfp4    = g(weights_nvfp4 + kv),
    )


if __name__ == "__main__":
    import json
    print("Qwen3.6-27B NVFP4 footprint at max_seq=32768:")
    print(json.dumps(memory_estimate_nvfp4(32768), indent=2))
    print("\nQwen3.6-27B NVFP4 footprint at max_seq=262144 (native):")
    print(json.dumps(memory_estimate_nvfp4(262144), indent=2))
