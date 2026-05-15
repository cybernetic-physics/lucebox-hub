"""Weight packer for Qwen3.6-27B megakernel.

Mirrors `models/qwen35_0p8b/model.py:_unify_weights_from_hf` +
`_pack_layer_weights` but for the 27B dimensions.

Two key responsibilities:

  1. From an HF Qwen3.6-27B checkpoint, build the per-layer pointer
     blocks that the templated megakernel reads. The Qwen3.6 HF
     state-dict keys live under `model.layers.{i}.*`; some attribute
     names differ from Qwen3.5-0.8B (e.g. linear_attn projection
     names), so this packer maintains a small mapping table.

  2. Allocate the persistent KV / DN state / scratch buffers at the
     27B-specific sizes (HIDDEN=5120, INTER=17408, etc).

Usage:
    from weight_packer import load_27b_weights, pack_layer_weights, alloc_scratch

    weights, tokenizer = load_27b_weights("Qwen/Qwen3.6-27B")
    layer_block = pack_layer_weights(weights["layer_data"])  # device bytes
    scratch = alloc_scratch(max_seq=32768)
"""
from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Qwen3.6-27B architecture constants. Mirrors megakernel/Cfg.cuh:Cfg_27B.
# ---------------------------------------------------------------------------

NUM_LAYERS         = 64
HIDDEN_SIZE        = 5120
INTERMEDIATE_SIZE  = 17408
FA_NUM_Q_HEADS     = 24
FA_NUM_KV_HEADS    = 4
FA_HEAD_DIM        = 256
DN_NUM_V_HEADS     = 48
DN_NUM_QK_HEADS    = 16
DN_HEAD_DIM        = 128
FA_ROTARY_DIM      = 64
FA_ROPE_THETA      = 1.0e7
VOCAB_SIZE         = 248320

FA_Q_SIZE          = FA_NUM_Q_HEADS  * FA_HEAD_DIM   # 6144
FA_KV_SIZE         = FA_NUM_KV_HEADS * FA_HEAD_DIM   # 1024
FA_QPROJ_SIZE      = 2 * FA_Q_SIZE                    # 12288 (Q + gate)
DN_QK_SIZE         = DN_NUM_QK_HEADS * DN_HEAD_DIM   # 2048
DN_V_SIZE          = DN_NUM_V_HEADS  * DN_HEAD_DIM   # 6144
DN_CONV_CH         = 2 * DN_QK_SIZE + DN_V_SIZE      # 10240
DN_CONV_KERNEL     = 4

# Hybrid pattern: 3 DN + 1 FA. Layer i is FA iff (i+1) % 4 == 0.
# (Same family invariant as 0.8B; only count differs.)
LAYER_TYPE = [(1 if (i + 1) % 4 == 0 else 0) for i in range(NUM_LAYERS)]

N_FA = sum(1 for t in LAYER_TYPE if t == 1)   # 16
N_DN = sum(1 for t in LAYER_TYPE if t == 0)   # 48


# ---------------------------------------------------------------------------
# State-dict key mapping for the HF Qwen3.6 checkpoint.
# ---------------------------------------------------------------------------
# Qwen3.6 uses Qwen3 naming. The DN layer's projection names follow the
# `linear_attn.in_proj_*` convention from fla / Qwen3 series; conv1d weight
# has shape [out_channels, 1, kernel] in HF (we squeeze to [out, kernel]).

def _hf_keys_fa(i: int) -> dict:
    # NOTE: state_dict prefix is `model.layers.X.*` even though the safetensors
    # file uses `model.language_model.layers.X.*` — HF's AutoModelForCausalLM
    # instantiates Qwen3_5TextModel which flattens that wrapper. The
    # safetensors->state_dict mapping happens automatically at from_pretrained.
    p = f"model.layers.{i}."
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


def _hf_keys_dn(i: int) -> dict:
    # NOTE: state_dict prefix is `model.layers.X.*` even though the safetensors
    # file uses `model.language_model.layers.X.*` — HF's AutoModelForCausalLM
    # instantiates Qwen3_5TextModel which flattens that wrapper. The
    # safetensors->state_dict mapping happens automatically at from_pretrained.
    p = f"model.layers.{i}."
    return {
        "input_layernorm":       p + "input_layernorm.weight",
        # Qwen3.6 DN uses an in_proj_qkv that concatenates Q, K, V along the
        # output axis: shape [HIDDEN, DN_CONV_CH].
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


def _check_shape(t: torch.Tensor, expected: tuple, name: str):
    if tuple(t.shape) != expected:
        raise ValueError(f"{name}: expected shape {expected}, got {tuple(t.shape)}")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _unify_from_hf_model(model, verbose: bool = False):
    """Build the kernel-shaped weights dict from an already-loaded HF
    model. Tensors are views into model.state_dict() — no new allocation.
    The caller is responsible for keeping `model` alive."""
    state = dict(model.state_dict())
    # Strip PEFT base_layer aliasing if any.
    for k, v in list(state.items()):
        if k.endswith(".base_layer.weight"):
            short = k[: -len(".base_layer.weight")] + ".weight"
            state[short] = v

    layer_data = []
    for i in range(NUM_LAYERS):
        if LAYER_TYPE[i] == 1:
            keys = _hf_keys_fa(i)
            order = ("input_layernorm", "q_proj", "k_proj", "v_proj",
                     "q_norm", "k_norm", "o_proj", "post_attn_layernorm",
                     "gate_proj", "up_proj", "down_proj")
        else:
            keys = _hf_keys_dn(i)
            order = ("input_layernorm", "qkv_proj", "z_proj", "beta_proj",
                     "alpha_proj", "conv1d", "a_log", "dt_bias", "norm_weight",
                     "out_proj", "post_attn_layernorm", "gate_proj", "up_proj",
                     "down_proj")
        ptrs = [state[keys[k]].contiguous() for k in order]
        # conv1d in HF: [DN_CONV_CH, 1, KERNEL] -> squeeze to [DN_CONV_CH, KERNEL]
        if LAYER_TYPE[i] == 0 and ptrs[5].dim() == 3:
            ptrs[5] = ptrs[5].squeeze(1).contiguous()
        layer_data.append({"type": int(LAYER_TYPE[i]), "ptrs": ptrs})

    embed = state["model.embed_tokens.weight"].contiguous()
    fnorm = state["model.norm.weight"].contiguous()
    lm_head = state.get("lm_head.weight", embed).contiguous()
    return dict(
        embed_weight=embed, final_norm_weight=fnorm,
        lm_head_weight=lm_head, layer_data=layer_data,
        _hf_keepalive=model,
    )


def load_27b_weights(
    model_name: str = "Qwen/Qwen3.6-27B",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
    verbose: bool = True,
):
    """Load Qwen3.6-27B from HF, build the kernel-shaped weights dict.

    Returns (weights, tokenizer). `weights["layer_data"]` is a list of
    per-layer dicts; `weights["_hf_keepalive"]` retains the HF model so
    the tensor views stay valid.
    """
    if not verbose:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    if verbose:
        print(f"[weight_packer] loading {model_name} ({dtype}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, device_map=device,
        trust_remote_code=trust_remote_code,
    )
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    state = model.state_dict()
    # Strip PEFT aliasing if any.
    aliased = dict(state)
    for k, v in list(state.items()):
        if k.endswith(".base_layer.weight"):
            short = k[: -len(".base_layer.weight")] + ".weight"
            aliased[short] = v
    state = aliased

    # Diagnostic: if the expected prefix isn't found, sample available keys
    # so the user can see what HF actually wrapped.
    probe_key = _hf_keys_fa(3)["input_layernorm"]   # any FA-layer key works
    if probe_key not in state:
        sample = [k for k in state.keys() if "input_layernorm" in k][:5]
        raise KeyError(
            f"weight_packer probe failed: {probe_key!r} not in state_dict.\n"
            f"  Sample input_layernorm keys found in state_dict: {sample}\n"
            f"  This means HF wrapped the model differently than expected; "
            f"update _hf_keys_fa / _hf_keys_dn to match."
        )

    layer_data = []
    for i in range(NUM_LAYERS):
        if LAYER_TYPE[i] == 1:
            keys = _hf_keys_fa(i)
            ptrs = [state[keys[k]].contiguous() for k in (
                "input_layernorm", "q_proj", "k_proj", "v_proj",
                "q_norm", "k_norm", "o_proj", "post_attn_layernorm",
                "gate_proj", "up_proj", "down_proj")]
            # Sanity checks.
            _check_shape(ptrs[0],  (HIDDEN_SIZE,),                    f"L{i} input_norm")
            _check_shape(ptrs[1],  (FA_QPROJ_SIZE, HIDDEN_SIZE),      f"L{i} q_proj")
            _check_shape(ptrs[2],  (FA_KV_SIZE, HIDDEN_SIZE),         f"L{i} k_proj")
            _check_shape(ptrs[3],  (FA_KV_SIZE, HIDDEN_SIZE),         f"L{i} v_proj")
            _check_shape(ptrs[4],  (FA_HEAD_DIM,),                    f"L{i} q_norm")
            _check_shape(ptrs[5],  (FA_HEAD_DIM,),                    f"L{i} k_norm")
            _check_shape(ptrs[6],  (HIDDEN_SIZE, FA_Q_SIZE),          f"L{i} o_proj")
            _check_shape(ptrs[7],  (HIDDEN_SIZE,),                    f"L{i} post_attn_norm")
            _check_shape(ptrs[8],  (INTERMEDIATE_SIZE, HIDDEN_SIZE),  f"L{i} gate_proj")
            _check_shape(ptrs[9],  (INTERMEDIATE_SIZE, HIDDEN_SIZE),  f"L{i} up_proj")
            _check_shape(ptrs[10], (HIDDEN_SIZE, INTERMEDIATE_SIZE),  f"L{i} down_proj")
            layer_data.append({"type": 1, "ptrs": ptrs})
        else:
            keys = _hf_keys_dn(i)
            ptrs = [state[keys[k]].contiguous() for k in (
                "input_layernorm", "qkv_proj", "z_proj", "beta_proj", "alpha_proj",
                "conv1d", "a_log", "dt_bias", "norm_weight", "out_proj",
                "post_attn_layernorm", "gate_proj", "up_proj", "down_proj")]
            _check_shape(ptrs[0],  (HIDDEN_SIZE,),                              f"L{i} input_norm")
            _check_shape(ptrs[1],  (DN_CONV_CH, HIDDEN_SIZE),                   f"L{i} qkv_proj")
            _check_shape(ptrs[2],  (DN_V_SIZE, HIDDEN_SIZE),                    f"L{i} z_proj")
            _check_shape(ptrs[3],  (DN_NUM_V_HEADS, HIDDEN_SIZE),               f"L{i} beta_proj")
            _check_shape(ptrs[4],  (DN_NUM_V_HEADS, HIDDEN_SIZE),               f"L{i} alpha_proj")
            # conv1d in HF: [DN_CONV_CH, 1, DN_CONV_KERNEL] (groups=channels).
            if ptrs[5].dim() == 3:
                ptrs[5] = ptrs[5].squeeze(1).contiguous()
            _check_shape(ptrs[5],  (DN_CONV_CH, DN_CONV_KERNEL),                f"L{i} conv1d")
            _check_shape(ptrs[6],  (DN_NUM_V_HEADS,),                           f"L{i} a_log")
            _check_shape(ptrs[7],  (DN_NUM_V_HEADS,),                           f"L{i} dt_bias")
            _check_shape(ptrs[8],  (DN_HEAD_DIM,),                              f"L{i} norm_weight")
            _check_shape(ptrs[9],  (HIDDEN_SIZE, DN_V_SIZE),                    f"L{i} out_proj")
            _check_shape(ptrs[10], (HIDDEN_SIZE,),                              f"L{i} post_attn_norm")
            _check_shape(ptrs[11], (INTERMEDIATE_SIZE, HIDDEN_SIZE),            f"L{i} gate_proj")
            _check_shape(ptrs[12], (INTERMEDIATE_SIZE, HIDDEN_SIZE),            f"L{i} up_proj")
            _check_shape(ptrs[13], (HIDDEN_SIZE, INTERMEDIATE_SIZE),            f"L{i} down_proj")
            layer_data.append({"type": 0, "ptrs": ptrs})

    # State-dict keys for Qwen3_5TextModel (the CausalLM-wrapped variant)
    # use the standard `model.embed_tokens.weight` etc. — no `language_model`
    # middle. The safetensors file has `model.language_model.embed_tokens.weight`
    # but HF strips that wrapper at load time.
    embed = state["model.embed_tokens.weight"].contiguous()
    fnorm = state["model.norm.weight"].contiguous()
    lm_head = state.get("lm_head.weight", embed).contiguous()
    _check_shape(embed, (VOCAB_SIZE, HIDDEN_SIZE), "embed")
    _check_shape(fnorm, (HIDDEN_SIZE,), "final_norm")
    _check_shape(lm_head, (VOCAB_SIZE, HIDDEN_SIZE), "lm_head")

    if verbose:
        total_params = sum(sum(t.numel() for t in ld["ptrs"]) for ld in layer_data) \
                       + embed.numel() + fnorm.numel() + lm_head.numel()
        print(f"[weight_packer] packed {NUM_LAYERS} layers, "
              f"{total_params/1e9:.1f}B params, "
              f"~{total_params * 2 / (1024**3):.1f} GB BF16")

    weights = dict(
        embed_weight=embed,
        final_norm_weight=fnorm,
        lm_head_weight=lm_head,
        layer_data=layer_data,
        _hf_keepalive=model,
    )
    return weights, tok


# ---------------------------------------------------------------------------
# Pack layer_data into the device-side LayerWeights<Cfg> blob the kernel reads.
# ---------------------------------------------------------------------------
# Matches LayerWeights<Cfg> in kernel_decode_full.cu (192 bytes).
#   { int layer_type; int _pad; union(<= 184 B); }
#   Max ptr count:
#     DN_bf16   = 14 ptrs (112 B)
#     FA_bf16   = 11 ptrs ( 88 B)
#     FA_nvfp4  = 4 bf16 ptrs + 7 * PackedMatrixNVFP4 (2 ptrs ea) = 18 ptrs (144 B)
#     DN_nvfp4  = 5 bf16 ptrs + 8 * PackedMatrixNVFP4 (2 ptrs ea) = 21 ptrs (168 B)
# C++ struct size is 192 (forced by `char _force_size[184]` + 8 header).
PACK_HEADER  = 8
PACK_STRUCT  = 192
PACK_MAX_PTR = (PACK_STRUCT - PACK_HEADER) // 8   # 23 — caps usable slots
assert PACK_STRUCT == 192, "must match LayerWeights<Cfg> in kernel_decode_full.cu"

def pack_layer_weights(layer_data: list[dict]) -> torch.Tensor:
    """Return a uint8 CUDA tensor laid out exactly like
    `const LayerWeights<Cfg>[]` that the kernel takes.

    Each `ld["ptrs"]` entry is either:
        - a torch.Tensor on CUDA (one ptr slot, used for BF16 layers and
          for BF16 norms/scalars in NVFP4 layers), or
        - a (data: torch.Tensor, scales: torch.Tensor) tuple (two ptr
          slots, matching PackedMatrixNVFP4 = {data, scales}).
    """
    n = len(layer_data)
    buf = bytearray(n * PACK_STRUCT)
    for i, ld in enumerate(layer_data):
        off = i * PACK_STRUCT
        ctypes.c_int32.from_buffer(buf, off).value = int(ld["type"])
        ctypes.c_int32.from_buffer(buf, off + 4).value = 0
        slot = 0
        for j, t in enumerate(ld["ptrs"]):
            if isinstance(t, tuple):
                data, scales = t
                for x in (data, scales):
                    if not x.is_cuda:
                        raise ValueError(f"layer {i} ptr {j} not on CUDA")
                    if not x.is_contiguous():
                        raise ValueError(f"layer {i} ptr {j} not contiguous")
                    ctypes.c_uint64.from_buffer(
                        buf, off + PACK_HEADER + slot * 8).value = x.data_ptr()
                    slot += 1
            else:
                if not t.is_cuda:
                    raise ValueError(f"layer {i} ptr {j} not on CUDA")
                if not t.is_contiguous():
                    raise ValueError(f"layer {i} ptr {j} not contiguous")
                ctypes.c_uint64.from_buffer(
                    buf, off + PACK_HEADER + slot * 8).value = t.data_ptr()
                slot += 1
        if slot > PACK_MAX_PTR:
            raise ValueError(
                f"layer {i} packs {slot} ptr slots; PACK_MAX_PTR={PACK_MAX_PTR}")
    return torch.frombuffer(bytes(buf), dtype=torch.uint8).cuda().contiguous()


# ---------------------------------------------------------------------------
# Persistent scratch + KV cache allocator.
# ---------------------------------------------------------------------------

@dataclass
class Scratch:
    # Per-layer persistent state
    fa_k_cache: torch.Tensor   # [N_FA, FA_NUM_KV_HEADS, max_seq, FA_HEAD_DIM] bf16
    fa_v_cache: torch.Tensor
    dn_states: torch.Tensor    # [N_DN, DN_NUM_V_HEADS, DN_HEAD_DIM, DN_HEAD_DIM] f32
    conv_bufs: torch.Tensor    # [N_DN, DN_CONV_CH, DN_CONV_KERNEL] f32

    # Per-token scratch (re-used each step)
    hidden_buffer: torch.Tensor    # [HIDDEN] bf16
    g_residual: torch.Tensor       # [HIDDEN] bf16
    g_qkv_scratch: torch.Tensor    # [max(FA_QPROJ_SIZE, DN_CONV_CH)] f32
    g_kv_scratch: torch.Tensor     # [FA_KV_SIZE * 2] f32
    g_attn_out: torch.Tensor       # [max(FA_Q_SIZE, DN_V_SIZE)] f32
    g_mlp_inter: torch.Tensor      # [INTERMEDIATE_SIZE] f32
    g_z_scratch: torch.Tensor      # [DN_V_SIZE] f32
    g_beta_scratch: torch.Tensor   # [DN_NUM_V_HEADS] f32
    g_alpha_scratch: torch.Tensor  # [DN_NUM_V_HEADS] f32
    g_normalized: torch.Tensor     # [HIDDEN] f32 (post final RMSnorm)
    g_fa_partials: torch.Tensor    # split-K partials
    g_rope_inv_freq: torch.Tensor  # [FA_ROTARY_DIM/2] f32
    max_seq: int


def alloc_scratch(max_seq: int = 32768, fa_num_splits: int = 128,
                   verbose: bool = False) -> Scratch:
    """Allocate the persistent + per-token scratch buffers for the 27B
    decode path. Memory budget at max_seq=32768:
        fa_k_cache + fa_v_cache  : 2 * 16*4*32768*256 * 2 B = 1.0 GB
        dn_states                : 48 * 48*128*128 * 4 B  = 144 MB
        conv_bufs                : 48 * 10240*4 * 4 B     = 7.5 MB
        small per-step buffers   : ~few MB
    Total ~1.2 GB persistent, plus ~54 GB BF16 weights.
    """
    import time
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    f32  = dict(dtype=torch.float32,  device="cuda")
    def _t(name):
        if verbose: print(f"  [alloc_scratch] {name}", flush=True)
    _t("fa_k_cache")
    fa_k_cache = torch.zeros(N_FA, FA_NUM_KV_HEADS, max_seq, FA_HEAD_DIM, **bf16)
    _t("fa_v_cache")
    fa_v_cache = torch.zeros(N_FA, FA_NUM_KV_HEADS, max_seq, FA_HEAD_DIM, **bf16)
    _t("dn_states")
    dn_states = torch.zeros(N_DN, DN_NUM_V_HEADS, DN_HEAD_DIM, DN_HEAD_DIM, **f32)
    _t("conv_bufs")
    conv_bufs = torch.zeros(N_DN, DN_CONV_CH, DN_CONV_KERNEL, **f32)
    _t("hidden_buffer")
    hidden_buffer = torch.zeros(HIDDEN_SIZE, **bf16)
    _t("g_residual")
    g_residual = torch.zeros(HIDDEN_SIZE, **bf16)
    _t("g_qkv_scratch")
    g_qkv_scratch = torch.zeros(max(FA_QPROJ_SIZE, DN_CONV_CH), **f32)
    _t("g_kv_scratch")
    g_kv_scratch = torch.zeros(FA_KV_SIZE * 2, **f32)
    _t("g_attn_out")
    g_attn_out = torch.zeros(max(FA_Q_SIZE, DN_V_SIZE), **f32)
    _t("g_mlp_inter")
    g_mlp_inter = torch.zeros(INTERMEDIATE_SIZE, **f32)
    _t("g_z_scratch")
    g_z_scratch = torch.zeros(DN_V_SIZE, **f32)
    _t("g_beta+alpha")
    g_beta_scratch  = torch.zeros(DN_NUM_V_HEADS, **f32)
    g_alpha_scratch = torch.zeros(DN_NUM_V_HEADS, **f32)
    _t("g_normalized")
    g_normalized = torch.zeros(HIDDEN_SIZE, **f32)
    _t("g_fa_partials")
    g_fa_partials = torch.zeros(fa_num_splits * FA_NUM_Q_HEADS * (FA_HEAD_DIM + 2), **f32)
    _t("g_rope_inv_freq")
    g_rope_inv_freq = torch.zeros(FA_ROTARY_DIM // 2, **f32)
    _t("Scratch ctor")
    return Scratch(
        fa_k_cache=fa_k_cache, fa_v_cache=fa_v_cache,
        dn_states=dn_states, conv_bufs=conv_bufs,
        hidden_buffer=hidden_buffer, g_residual=g_residual,
        g_qkv_scratch=g_qkv_scratch, g_kv_scratch=g_kv_scratch,
        g_attn_out=g_attn_out, g_mlp_inter=g_mlp_inter,
        g_z_scratch=g_z_scratch,
        g_beta_scratch=g_beta_scratch, g_alpha_scratch=g_alpha_scratch,
        g_normalized=g_normalized, g_fa_partials=g_fa_partials,
        g_rope_inv_freq=g_rope_inv_freq, max_seq=max_seq,
    )

def _alloc_scratch_OLD(max_seq: int = 32768, fa_num_splits: int = 128) -> Scratch:
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    f32  = dict(dtype=torch.float32,  device="cuda")
    return Scratch(
        fa_k_cache    = torch.zeros(N_FA, FA_NUM_KV_HEADS, max_seq, FA_HEAD_DIM, **bf16),
        fa_v_cache    = torch.zeros(N_FA, FA_NUM_KV_HEADS, max_seq, FA_HEAD_DIM, **bf16),
        dn_states     = torch.zeros(N_DN, DN_NUM_V_HEADS, DN_HEAD_DIM, DN_HEAD_DIM, **f32),
        conv_bufs     = torch.zeros(N_DN, DN_CONV_CH, DN_CONV_KERNEL, **f32),
        hidden_buffer = torch.zeros(HIDDEN_SIZE, **bf16),
        g_residual    = torch.zeros(HIDDEN_SIZE, **bf16),
        g_qkv_scratch = torch.zeros(max(FA_QPROJ_SIZE, DN_CONV_CH), **f32),
        g_kv_scratch  = torch.zeros(FA_KV_SIZE * 2, **f32),
        g_attn_out    = torch.zeros(max(FA_Q_SIZE, DN_V_SIZE), **f32),
        g_mlp_inter   = torch.zeros(INTERMEDIATE_SIZE, **f32),
        g_z_scratch   = torch.zeros(DN_V_SIZE, **f32),
        g_beta_scratch  = torch.zeros(DN_NUM_V_HEADS, **f32),
        g_alpha_scratch = torch.zeros(DN_NUM_V_HEADS, **f32),
        g_normalized  = torch.zeros(HIDDEN_SIZE, **f32),
        g_fa_partials = torch.zeros(fa_num_splits * FA_NUM_Q_HEADS * (FA_HEAD_DIM + 2), **f32),
        g_rope_inv_freq = torch.zeros(FA_ROTARY_DIM // 2, **f32),
        max_seq=max_seq,
    )


def memory_estimate_gb(max_seq: int = 32768) -> dict:
    """Return a memory-footprint estimate per buffer family (GB), without
    actually allocating anything. Useful for capacity planning."""
    g = lambda b: b / (1024 ** 3)
    weights = 27e9 * 2     # 27B params at BF16
    kv = 2 * N_FA * FA_NUM_KV_HEADS * max_seq * FA_HEAD_DIM * 2
    dn = N_DN * DN_NUM_V_HEADS * DN_HEAD_DIM * DN_HEAD_DIM * 4
    conv = N_DN * DN_CONV_CH * DN_CONV_KERNEL * 4
    return dict(
        weights_bf16 = g(weights),
        kv_cache_bf16 = g(kv),
        dn_states     = g(dn),
        conv_bufs     = g(conv),
        scratch_total = g(weights + kv + dn + conv),
    )


if __name__ == "__main__":
    import json
    print("Qwen3.6-27B memory estimate at max_seq=32768:")
    print(json.dumps(memory_estimate_gb(32768), indent=2))
    print("\nQwen3.6-27B memory estimate at max_seq=262144 (native):")
    print(json.dumps(memory_estimate_gb(262144), indent=2))
