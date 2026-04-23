"""Python driver for the prefill megakernel.

Loads Qwen 3.5-0.8B bf16 weights (same format as ../megakernel/model.py)
and dispatches ONE CUDA launch per forward.
"""

from __future__ import annotations

import os
import struct
import torch

NUM_LAYERS = 24
HIDDEN = 1024
INTER = 3584
VOCAB = 248320
MAX_SEQ_LEN = 2048

FA_Q_HEADS = 8
FA_KV_HEADS = 2
FA_HEAD_DIM = 256
FA_Q_SIZE = FA_Q_HEADS * FA_HEAD_DIM           # 2048
FA_QPROJ_SIZE = FA_Q_SIZE * 2                  # 4096
FA_KV_SIZE = FA_KV_HEADS * FA_HEAD_DIM         # 512

DN_HEADS = 16
DN_KEY = 128
DN_VAL = 128
DN_V_SIZE = DN_HEADS * DN_VAL                  # 2048
DN_QK_SIZE = DN_HEADS * DN_KEY                 # 2048
DN_CONV_CH = DN_QK_SIZE * 2 + DN_V_SIZE        # 6144
DN_CONV_K = 4

LAYER_TYPE = [0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1]
N_FA = sum(1 for t in LAYER_TYPE if t == 1)
N_DN = sum(1 for t in LAYER_TYPE if t == 0)

_op = None


def _load_op():
    global _op
    if _op is None:
        import prefill_megakernel_C  # noqa: F401
        _op = torch.ops.prefill_megakernel_C.prefill_mega
    return _op


def _pack_layer_weights(layer_data):
    """Pack per-layer bf16 weight pointers into a device blob matching the
    LayerWeights struct in kernel.cu."""
    ptr_size = 8
    max_ptrs = 14
    header = 16
    struct_size = header + max_ptrs * ptr_size
    buf = bytearray(NUM_LAYERS * struct_size)
    for i in range(NUM_LAYERS):
        ld = layer_data[i]
        off = i * struct_size
        struct.pack_into("iiii", buf, off, ld["type"], 0, 0, 0)
        for j, t in enumerate(ld["ptrs"]):
            struct.pack_into("Q", buf, off + header + j * ptr_size, t.data_ptr())
        for j in range(len(ld["ptrs"]), max_ptrs):
            struct.pack_into("Q", buf, off + header + j * ptr_size, 0)
    return torch.frombuffer(buf, dtype=torch.uint8).cuda()


def build_layer_data_from_state(state):
    """Slice a HF Qwen 3.5-0.8B state dict into the per-layer ptr list."""
    out = []
    for i in range(NUM_LAYERS):
        p = f"model.layers.{i}."
        if LAYER_TYPE[i] == 1:
            out.append({"type": 1, "ptrs": [
                state[p + "input_layernorm.weight"].contiguous(),
                state[p + "self_attn.q_proj.weight"].contiguous(),
                state[p + "self_attn.k_proj.weight"].contiguous(),
                state[p + "self_attn.v_proj.weight"].contiguous(),
                state[p + "self_attn.q_norm.weight"].contiguous(),
                state[p + "self_attn.k_norm.weight"].contiguous(),
                state[p + "self_attn.o_proj.weight"].contiguous(),
                state[p + "post_attention_layernorm.weight"].contiguous(),
                state[p + "mlp.gate_proj.weight"].contiguous(),
                state[p + "mlp.up_proj.weight"].contiguous(),
                state[p + "mlp.down_proj.weight"].contiguous(),
            ]})
        else:
            out.append({"type": 0, "ptrs": [
                state[p + "input_layernorm.weight"].contiguous(),
                state[p + "linear_attn.in_proj_qkv.weight"].contiguous(),
                state[p + "linear_attn.in_proj_z.weight"].contiguous(),
                state[p + "linear_attn.in_proj_b.weight"].contiguous(),
                state[p + "linear_attn.in_proj_a.weight"].contiguous(),
                state[p + "linear_attn.conv1d.weight"].contiguous(),
                state[p + "linear_attn.A_log"].contiguous(),
                state[p + "linear_attn.dt_bias"].contiguous(),
                state[p + "linear_attn.norm.weight"].contiguous(),
                state[p + "linear_attn.out_proj.weight"].contiguous(),
                state[p + "post_attention_layernorm.weight"].contiguous(),
                state[p + "mlp.gate_proj.weight"].contiguous(),
                state[p + "mlp.up_proj.weight"].contiguous(),
                state[p + "mlp.down_proj.weight"].contiguous(),
            ]})
    return out


class PrefillMegakernel:
    """Single-dispatch prefill for Qwen 3.5-0.8B.

    Exposes a `.forward(token_ids) -> next_token_id` that runs the full
    24-layer stack plus LM head inside one cooperative CUDA launch.
    """

    def __init__(self, weights, max_seq_len: int = MAX_SEQ_LEN):
        _load_op()
        self.max_seq_len = int(max_seq_len)
        self._embed = weights["embed_weight"]
        self._final_norm = weights["final_norm_weight"]
        self._lm_head = weights["lm_head_weight"]
        self._layers_packed = _pack_layer_weights(weights["layer_data"])

        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        f32 = dict(dtype=torch.float32, device="cuda")
        i32 = dict(dtype=torch.int32, device="cuda")

        self._fa_k_cache = torch.zeros(N_FA, FA_KV_HEADS, self.max_seq_len, FA_HEAD_DIM, **bf16)
        self._fa_v_cache = torch.zeros_like(self._fa_k_cache)
        self._dn_states = torch.zeros(N_DN, DN_HEADS, DN_KEY, DN_VAL, **f32)
        self._conv_bufs = torch.zeros(N_DN, DN_CONV_CH, DN_CONV_K, **f32)
        self._max_s = 0
        self._hidden = None

        self._num_blocks = int(os.environ.get("PM_NUM_BLOCKS", 148))
        self._lm_bmv = torch.empty(self._num_blocks, **f32)
        self._lm_bmi = torch.empty(self._num_blocks, **i32)
        self._out_token = torch.empty(1, **i32)
        self._final_normed = torch.empty(HIDDEN, **bf16)

    def _alloc_for_S(self, S: int):
        if self._max_s >= S:
            return
        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        f32 = dict(dtype=torch.float32, device="cuda")

        max_proj = max(FA_QPROJ_SIZE, DN_CONV_CH, INTER)       # 6144
        max_attn = max(FA_Q_SIZE, DN_V_SIZE, FA_KV_SIZE)
        self._hidden = torch.empty(S, HIDDEN, **bf16)
        self._residual = torch.empty(S, HIDDEN, **bf16)
        self._normalized = torch.empty(S, HIDDEN, **bf16)
        self._proj_buf = torch.empty(S, max_proj, **bf16)
        self._proj_buf2 = torch.empty(S, max_proj, **bf16)
        self._attn_buf = torch.empty(S, max_attn, **bf16)
        self._mlp_buf = torch.empty(S, INTER, **bf16)
        self._dn_out_buf = torch.empty(S, max_attn, **bf16)
        self._beta_buf = torch.empty(S, DN_HEADS, **f32)
        self._alpha_buf = torch.empty(S, DN_HEADS, **f32)
        self._max_s = S

    def reset(self):
        self._fa_k_cache.zero_()
        self._fa_v_cache.zero_()
        self._dn_states.zero_()
        self._conv_bufs.zero_()

    def forward(self, token_ids: torch.Tensor) -> int:
        assert token_ids.dim() == 1
        S = int(token_ids.shape[0])
        assert S % 16 == 0, (
            f"S={S} must be a multiple of 16 (WMMA tile requirement). "
            "Pad your prompt to the next multiple of 16 before calling."
        )
        self._alloc_for_S(S)
        ids = token_ids.to(device="cuda", dtype=torch.int32).contiguous()
        _op(
            self._out_token, ids,
            self._embed, self._layers_packed,
            self._final_norm, self._lm_head,
            self._fa_k_cache, self._fa_v_cache,
            self._dn_states, self._conv_bufs,
            self._hidden, self._residual, self._normalized,
            self._proj_buf, self._proj_buf2,
            self._attn_buf, self._mlp_buf, self._dn_out_buf,
            self._beta_buf, self._alpha_buf,
            self._final_normed,
            self._lm_bmv, self._lm_bmi,
            self.max_seq_len,
        )
        return int(self._out_token.item())
