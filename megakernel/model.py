"""Weight loading and decode API for Qwen3.5-0.8B megakernel backends."""

import os
import struct
import torch

NUM_LAYERS = 24
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3584
VOCAB_SIZE = 248320
MAX_SEQ_LEN = 2048

FA_NUM_Q_HEADS = 8
FA_NUM_KV_HEADS = 2
FA_HEAD_DIM = 256
FA_Q_SIZE = FA_NUM_Q_HEADS * FA_HEAD_DIM
FA_QPROJ_SIZE = FA_Q_SIZE * 2
FA_KV_SIZE = FA_NUM_KV_HEADS * FA_HEAD_DIM

DN_NUM_HEADS = 16
DN_KEY_DIM = 128
DN_VALUE_DIM = 128
DN_QK_SIZE = DN_NUM_HEADS * DN_KEY_DIM
DN_V_SIZE = DN_NUM_HEADS * DN_VALUE_DIM
DN_CONV_CHANNELS = DN_QK_SIZE * 2 + DN_V_SIZE
DN_CONV_KERNEL = 4

LAYER_TYPE = [0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1]
NVFP4_GROUP_SIZE = 32

_decode = None
_decode_nvfp4 = None
_decode_many_nvfp4 = None
_prefill_bf16 = None
_quantize_nvfp4_out = None


def _load_op():
    global _decode, _decode_nvfp4, _decode_many_nvfp4, _prefill_bf16, _quantize_nvfp4_out
    if _decode is None:
        import qwen35_megakernel_bf16_C
        ops = torch.ops.qwen35_megakernel_bf16_C
        _decode = ops.decode
        _decode_nvfp4 = ops.decode_nvfp4
        _decode_many_nvfp4 = ops.decode_many_nvfp4
        _prefill_bf16 = ops.prefill_bf16
        _quantize_nvfp4_out = ops.quantize_nvfp4_out


def _resolve_backend(backend):
    if backend not in (None, "auto", "bf16", "nvfp4"):
        raise ValueError(f"unsupported backend: {backend}")

    forced = os.environ.get("MEGAKERNEL_BACKEND")
    if forced:
        backend = forced

    if backend in (None, "auto"):
        major, _minor = torch.cuda.get_device_capability()
        return "nvfp4" if major >= 12 else "bf16"

    return backend


def _quantize_matrix_nvfp4(weight, group_size):
    _load_op()
    if weight.dtype != torch.bfloat16:
        raise TypeError(f"expected bfloat16 weight, got {weight.dtype}")
    if weight.dim() != 2:
        raise ValueError(f"expected 2D weight, got shape {tuple(weight.shape)}")

    rows, cols = weight.shape
    if cols % 2 != 0 or cols % group_size != 0:
        raise ValueError(f"in_dim {cols} must be divisible by 2 and group_size {group_size}")

    packed = torch.empty((rows, cols // 2), dtype=torch.uint8, device=weight.device)
    scales = torch.empty((rows, cols // group_size), dtype=torch.float16, device=weight.device)
    _quantize_nvfp4_out(packed, scales, weight.contiguous(), group_size)
    return {"packed": packed, "scales": scales}


def _attach_nvfp4_weights(weights, group_size=NVFP4_GROUP_SIZE, verbose=True):
    if "nvfp4" in weights and weights["nvfp4"]["group_size"] == group_size:
        return weights

    if verbose:
        print(f"Quantizing decode hot weights to NVFP4 (group_size={group_size})...")

    layer_data_nvfp4 = []
    packed_bytes = 0
    scale_bytes = 0

    for ld in weights["layer_data"]:
        if ld["type"] == 1:
            q_proj = _quantize_matrix_nvfp4(ld["ptrs"][1], group_size)
            k_proj = _quantize_matrix_nvfp4(ld["ptrs"][2], group_size)
            v_proj = _quantize_matrix_nvfp4(ld["ptrs"][3], group_size)
            o_proj = _quantize_matrix_nvfp4(ld["ptrs"][6], group_size)
            gate_proj = _quantize_matrix_nvfp4(ld["ptrs"][8], group_size)
            up_proj = _quantize_matrix_nvfp4(ld["ptrs"][9], group_size)
            down_proj = _quantize_matrix_nvfp4(ld["ptrs"][10], group_size)
            qptrs = [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
            layer_data_nvfp4.append({
                "type": 1,
                "ptrs": [
                    ld["ptrs"][0],
                    q_proj["packed"], q_proj["scales"],
                    k_proj["packed"], k_proj["scales"],
                    v_proj["packed"], v_proj["scales"],
                    ld["ptrs"][4], ld["ptrs"][5],
                    o_proj["packed"], o_proj["scales"],
                    ld["ptrs"][7],
                    gate_proj["packed"], gate_proj["scales"],
                    up_proj["packed"], up_proj["scales"],
                    down_proj["packed"], down_proj["scales"],
                ],
                "quantized": qptrs,
            })
        else:
            qkv_proj = _quantize_matrix_nvfp4(ld["ptrs"][1], group_size)
            z_proj = _quantize_matrix_nvfp4(ld["ptrs"][2], group_size)
            out_proj = _quantize_matrix_nvfp4(ld["ptrs"][9], group_size)
            gate_proj = _quantize_matrix_nvfp4(ld["ptrs"][11], group_size)
            up_proj = _quantize_matrix_nvfp4(ld["ptrs"][12], group_size)
            down_proj = _quantize_matrix_nvfp4(ld["ptrs"][13], group_size)
            qptrs = [qkv_proj, z_proj, out_proj, gate_proj, up_proj, down_proj]
            layer_data_nvfp4.append({
                "type": 0,
                "ptrs": [
                    ld["ptrs"][0],
                    qkv_proj["packed"], qkv_proj["scales"],
                    z_proj["packed"], z_proj["scales"],
                    ld["ptrs"][3], ld["ptrs"][4], ld["ptrs"][5], ld["ptrs"][6], ld["ptrs"][7], ld["ptrs"][8],
                    out_proj["packed"], out_proj["scales"],
                    ld["ptrs"][10],
                    gate_proj["packed"], gate_proj["scales"],
                    up_proj["packed"], up_proj["scales"],
                    down_proj["packed"], down_proj["scales"],
                ],
                "quantized": qptrs,
            })

        for q in qptrs:
            packed_bytes += q["packed"].numel() * q["packed"].element_size()
            scale_bytes += q["scales"].numel() * q["scales"].element_size()

    lm_head_nvfp4 = _quantize_matrix_nvfp4(weights["lm_head_weight"], group_size)
    packed_bytes += lm_head_nvfp4["packed"].numel() * lm_head_nvfp4["packed"].element_size()
    scale_bytes += lm_head_nvfp4["scales"].numel() * lm_head_nvfp4["scales"].element_size()

    weights["nvfp4"] = {
        "group_size": group_size,
        "layer_data": layer_data_nvfp4,
        "lm_head_weight_packed": lm_head_nvfp4["packed"],
        "lm_head_scales": lm_head_nvfp4["scales"],
    }

    if verbose:
        total_mb = (packed_bytes + scale_bytes) / 1e6
        print(
            f"NVFP4 decode weights: {packed_bytes/1e6:.0f} MB packed + "
            f"{scale_bytes/1e6:.0f} MB scales ({total_mb:.0f} MB total)"
        )

    return weights


def load_weights(
    model_name="Qwen/Qwen3.5-0.8B",
    verbose=True,
    backend="bf16",
    nvfp4_group_size=NVFP4_GROUP_SIZE,
):
    """Load Qwen3.5-0.8B weights and optional GB10 NVFP4 decode weights."""
    if not verbose:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_backend = _resolve_backend(backend)

    if verbose:
        print(f"Loading {model_name} (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    state = model.state_dict()

    layer_data = []
    for i in range(NUM_LAYERS):
        p = f"model.layers.{i}."
        lt = LAYER_TYPE[i]

        if lt == 1:
            # Full Attention: 11 pointers (all bf16)
            layer_data.append({
                "type": 1,
                "ptrs": [
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
                ]
            })
        else:
            # DeltaNet: 14 pointers (all bf16)
            layer_data.append({
                "type": 0,
                "ptrs": [
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
                ]
            })

    embed_weight = state["model.embed_tokens.weight"].contiguous()
    final_norm_weight = state["model.norm.weight"].contiguous()
    lm_head = state.get("lm_head.weight", embed_weight).contiguous()

    weights = {
        "embed_weight": embed_weight,
        "final_norm_weight": final_norm_weight,
        "lm_head_weight": lm_head,
        "layer_data": layer_data,
    }

    del model
    torch.cuda.empty_cache()

    if verbose:
        total = sum(sum(t.numel() for t in ld["ptrs"]) for ld in layer_data) + lm_head.numel()
        print(f"BF16 weights: {total/1e6:.1f}M params ({total*2/1e6:.0f} MB)")

    if resolved_backend == "nvfp4":
        _attach_nvfp4_weights(weights, group_size=nvfp4_group_size, verbose=verbose)

    return weights, tokenizer


def _pack_layer_weights(layer_data):
    """Pack layer weights into device blob matching LayerWeights struct."""
    ptr_size = 8
    max_ptrs = 14
    header_size = 16
    struct_size = header_size + max_ptrs * ptr_size  # 128

    buf = bytearray(NUM_LAYERS * struct_size)
    for i in range(NUM_LAYERS):
        ld = layer_data[i]
        offset = i * struct_size
        struct.pack_into("iiii", buf, offset, ld["type"], 0, 0, 0)
        for j, tensor in enumerate(ld["ptrs"]):
            struct.pack_into("Q", buf, offset + header_size + j * ptr_size, tensor.data_ptr())
        for j in range(len(ld["ptrs"]), max_ptrs):
            struct.pack_into("Q", buf, offset + header_size + j * ptr_size, 0)

    return torch.frombuffer(buf, dtype=torch.uint8).cuda()


def _pack_layer_weights_nvfp4(layer_data, group_size):
    """Pack layer weights into device blob matching LayerWeightsNVFP4 struct."""
    ptr_size = 8
    max_ptrs = 24
    header_size = 16
    struct_size = header_size + max_ptrs * ptr_size

    buf = bytearray(NUM_LAYERS * struct_size)
    for i in range(NUM_LAYERS):
        ld = layer_data[i]
        offset = i * struct_size
        struct.pack_into("iiii", buf, offset, ld["type"], group_size, 0, 0)
        for j, tensor in enumerate(ld["ptrs"]):
            struct.pack_into("Q", buf, offset + header_size + j * ptr_size, tensor.data_ptr())
        for j in range(len(ld["ptrs"]), max_ptrs):
            struct.pack_into("Q", buf, offset + header_size + j * ptr_size, 0)

    return torch.frombuffer(buf, dtype=torch.uint8).cuda()


class Decoder:
    """Stateful decoder for Qwen3.5-0.8B megakernel backends."""

    def __init__(
        self,
        weights=None,
        tokenizer=None,
        model_name="Qwen/Qwen3.5-0.8B",
        backend="auto",
        nvfp4_group_size=NVFP4_GROUP_SIZE,
        verbose=True,
    ):
        _load_op()
        self.backend = _resolve_backend(backend)
        self.backend_label = "NVFP4 decode" if self.backend == "nvfp4" else "BF16"
        self._nvfp4_group_size = nvfp4_group_size

        if weights is None:
            weights, tokenizer = load_weights(
                model_name,
                verbose=verbose,
                backend=self.backend,
                nvfp4_group_size=nvfp4_group_size,
            )
        elif self.backend == "nvfp4":
            _attach_nvfp4_weights(weights, group_size=nvfp4_group_size, verbose=verbose)
        self.tokenizer = tokenizer
        self._position = 0
        self._weights = weights
        self._embed_weight = weights["embed_weight"]
        self._final_norm_weight = weights["final_norm_weight"]
        self._lm_head_weight = weights["lm_head_weight"]
        self._layer_weights_packed = _pack_layer_weights(weights["layer_data"])
        self._layer_weights_packed_nvfp4 = None
        self._lm_head_weight_packed = None
        self._lm_head_scales = None
        if self.backend == "nvfp4":
            nvfp4 = weights["nvfp4"]
            self._layer_weights_packed_nvfp4 = _pack_layer_weights_nvfp4(
                nvfp4["layer_data"], nvfp4["group_size"])
            self._lm_head_weight_packed = nvfp4["lm_head_weight_packed"]
            self._lm_head_scales = nvfp4["lm_head_scales"]

        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        f32 = dict(dtype=torch.float32, device="cuda")
        i32 = dict(dtype=torch.int32, device="cuda")
        u32 = dict(dtype=torch.uint32, device="cuda")

        n_fa = sum(1 for t in LAYER_TYPE if t == 1)
        self._fa_k_cache = torch.zeros(n_fa, FA_NUM_KV_HEADS, MAX_SEQ_LEN, FA_HEAD_DIM, **bf16)
        self._fa_v_cache = torch.zeros_like(self._fa_k_cache)

        n_dn = sum(1 for t in LAYER_TYPE if t == 0)
        self._dn_states = torch.zeros(n_dn, DN_NUM_HEADS, DN_KEY_DIM, DN_VALUE_DIM, **f32)
        self._conv_bufs = torch.zeros(n_dn, DN_CONV_CHANNELS, DN_CONV_KERNEL, **f32)

        self._hidden = torch.empty(HIDDEN_SIZE, **bf16)
        max_scratch = max(FA_QPROJ_SIZE, DN_CONV_CHANNELS, HIDDEN_SIZE * 8 + INTERMEDIATE_SIZE)
        self._activations = torch.empty(max_scratch, **f32)
        self._residual = torch.empty(HIDDEN_SIZE, **bf16)
        self._qkv_scratch = torch.empty(max(FA_QPROJ_SIZE, DN_CONV_CHANNELS), **f32)
        self._kv_scratch = torch.empty(FA_KV_SIZE * 2, **f32)
        self._attn_out = torch.empty(max(FA_Q_SIZE, DN_V_SIZE), **f32)
        self._mlp_inter = torch.empty(INTERMEDIATE_SIZE, **f32)
        self._z_scratch = torch.empty(DN_V_SIZE, **f32)
        self._beta_scratch = torch.empty(DN_NUM_HEADS, **f32)
        self._alpha_scratch = torch.empty(DN_NUM_HEADS, **f32)
        self._normalized = torch.empty(HIDDEN_SIZE, **f32)

        self._barrier_counter = torch.zeros(1, **u32)
        self._barrier_generation = torch.zeros(1, **u32)
        self._block_max_vals = torch.empty(1024, **f32)
        self._block_max_idxs = torch.empty(1024, **i32)
        self._lm_sync_counter = torch.zeros(1, **u32)
        self._out_token = torch.empty(1, **i32)

    def step(self, token_id: int) -> int:
        """Decode one token. Returns next token id."""
        if self.backend == "nvfp4":
            _decode_nvfp4(
                self._out_token, token_id,
                self._embed_weight, self._layer_weights_packed_nvfp4,
                self._final_norm_weight, self._lm_head_weight_packed, self._lm_head_scales,
                self._fa_k_cache, self._fa_v_cache,
                self._dn_states, self._conv_bufs,
                self._hidden, self._activations, self._residual,
                self._qkv_scratch, self._kv_scratch, self._attn_out,
                self._mlp_inter, self._z_scratch, self._beta_scratch,
                self._alpha_scratch, self._normalized,
                self._barrier_counter, self._barrier_generation,
                self._block_max_vals, self._block_max_idxs,
                self._lm_sync_counter,
                self._position, MAX_SEQ_LEN, self._nvfp4_group_size,
            )
        else:
            _decode(
                self._out_token, token_id,
                self._embed_weight, self._layer_weights_packed,
                self._final_norm_weight, self._lm_head_weight,
                self._fa_k_cache, self._fa_v_cache,
                self._dn_states, self._conv_bufs,
                self._hidden, self._activations, self._residual,
                self._qkv_scratch, self._kv_scratch, self._attn_out,
                self._mlp_inter, self._z_scratch, self._beta_scratch,
                self._alpha_scratch, self._normalized,
                self._barrier_counter, self._barrier_generation,
                self._block_max_vals, self._block_max_idxs,
                self._lm_sync_counter,
                self._position, MAX_SEQ_LEN,
            )
        self._position += 1
        return self._out_token.item()

    def step_many(self, token_id: int, num_steps: int) -> torch.Tensor:
        """Decode multiple NVFP4 steps without per-token host/device synchronization."""
        if self.backend != "nvfp4":
            raise RuntimeError("step_many is only available for the NVFP4 backend")
        if num_steps < 0:
            raise ValueError("num_steps must be non-negative")
        if num_steps == 0:
            return torch.empty(0, dtype=torch.int32, device="cuda")

        output_tokens = torch.empty(num_steps, dtype=torch.int32, device="cuda")
        _decode_many_nvfp4(
            output_tokens, self._out_token, token_id,
            self._embed_weight, self._layer_weights_packed_nvfp4,
            self._final_norm_weight, self._lm_head_weight_packed, self._lm_head_scales,
            self._fa_k_cache, self._fa_v_cache,
            self._dn_states, self._conv_bufs,
            self._hidden, self._activations, self._residual,
            self._qkv_scratch, self._kv_scratch, self._attn_out,
            self._mlp_inter, self._z_scratch, self._beta_scratch,
            self._alpha_scratch, self._normalized,
            self._barrier_counter, self._barrier_generation,
            self._block_max_vals, self._block_max_idxs,
            self._lm_sync_counter,
            self._position, MAX_SEQ_LEN, self._nvfp4_group_size,
        )
        self._position += num_steps
        return output_tokens

    def reset(self):
        self._position = 0
        self._fa_k_cache.zero_()
        self._fa_v_cache.zero_()
        self._dn_states.zero_()
        self._conv_bufs.zero_()

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        self.reset()
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        for tid in ids[:-1]:
            self.step(tid)
        out = []
        next_id = ids[-1]
        eos = self.tokenizer.eos_token_id
        for _ in range(max_tokens):
            next_id = self.step(next_id)
            if next_id == eos:
                break
            out.append(next_id)
        return self.tokenizer.decode(out, skip_special_tokens=True)
