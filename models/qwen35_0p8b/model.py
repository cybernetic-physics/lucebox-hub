"""Weight loading and decode API for Qwen3.5-0.8B megakernel backends."""

import os
import struct
import torch

NUM_LAYERS = 24
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3584
VOCAB_SIZE = 248320
MAX_SEQ_LEN = 65536  # KV cache row count for the Decoder. Sized for 64K-
                    # token rollouts. Both decode and prefill take max_seq
                    # as a runtime parameter and use it for the FA cache
                    # stride. Memory cost: ~768 MB total per Decoder
                    # (6 FA layers × 2 (K+V) × 65536 rows × 2 KV heads
                    # × 256 head_dim × 2 bytes), fine on B200's 192 GB.

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

# cuBLASLt block-scaled FP4 LM-head contract (see kernel_gb10_nvfp4.cu).
# The decoder quantizes a 16-row hidden tile to NVFP4 with UE4M3 group-16
# scales per step, then runs `cublasLtMatmul` against the FP4 LM-head
# weights for FP16 logits. These constants size the auxiliary buffers.
LM_HEAD_TENSORCORE_N = 16
NVFP4_TC_ROWS_PER_TILE = 128
NVFP4_TC_COLS_PER_TILE = 4
NVFP4_TC_BLOCK_K = 16
NVFP4_TC_K_PER_TILE = NVFP4_TC_COLS_PER_TILE * NVFP4_TC_BLOCK_K
NVFP4_LM_GROUP_SIZE = NVFP4_TC_BLOCK_K
LM_HEAD_TENSORCORE_PACKED_BYTES = LM_HEAD_TENSORCORE_N * (1024 // 2)
LM_HEAD_TENSORCORE_SCALE_BYTES = (
    ((LM_HEAD_TENSORCORE_N + NVFP4_TC_ROWS_PER_TILE - 1) // NVFP4_TC_ROWS_PER_TILE)
    * (1024 // NVFP4_TC_K_PER_TILE)
    * 512
)

_decode = None
_decode_nvfp4 = None
_decode_many_nvfp4 = None
_prefill_bf16 = None
_quantize_nvfp4_out = None
_quantize_nvfp4_lm_out = None


def _load_op():
    global _decode, _decode_nvfp4, _decode_many_nvfp4
    global _prefill_bf16, _quantize_nvfp4_out, _quantize_nvfp4_lm_out
    if _decode is None:
        import qwen35_megakernel_bf16_C  # noqa: F401
        ops = torch.ops.qwen35_megakernel_bf16_C
        _decode = ops.decode
        _prefill_bf16 = ops.prefill_bf16
        _quantize_nvfp4_out = ops.quantize_nvfp4_out
        # NVFP4 ops only exist on Blackwell builds (MEGAKERNEL_HAS_NVFP4).
        for name in ("decode_nvfp4", "decode_many_nvfp4", "quantize_nvfp4_lm_out"):
            globals()[f"_{name}"] = getattr(ops, name, None)
        global _decode_nvfp4, _decode_many_nvfp4, _quantize_nvfp4_lm_out
        _decode_nvfp4 = getattr(ops, "decode_nvfp4", None)
        _decode_many_nvfp4 = getattr(ops, "decode_many_nvfp4", None)
        _quantize_nvfp4_lm_out = getattr(ops, "quantize_nvfp4_lm_out", None)


_VALID_BACKENDS = ("auto", "bf16", "nvfp4", "bf16_fp4lm")


def _resolve_backend(backend):
    if backend not in (None,) + _VALID_BACKENDS:
        raise ValueError(
            f"unsupported backend: {backend!r}; valid: {_VALID_BACKENDS}")

    forced = os.environ.get("MEGAKERNEL_BACKEND")
    if forced:
        backend = forced

    if backend in (None, "auto"):
        # Auto-select on GB10 (sm_121a+):
        #   bf16        — fully correct (32/32 vs HF), eager BF16 LM head.
        #   bf16_fp4lm  — BF16 trunk + cuBLASLt FP4 block-scaled LM head
        #                 (CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3).
        #                 100% greedy-top1 vs HF in validation; +~7%
        #                 step time at long context vs pure bf16 because
        #                 we still pay the BF16 LM head. Saves ~125 MB
        #                 of LM-head weight RAM. Recommended once we
        #                 want the FP4 tensor-core path on this chip.
        #   nvfp4       — full FP4 layer projections + FP4 LM head.
        #                 Drifts vs HF (intrinsic to per-group-32 FP4
        #                 across 24 layers); kept for ablation.
        # See docs/results/qwen35_0p8b_gb10.md for measurements.
        return "bf16"

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

    # LM head is consumed by the cuBLASLt FP4 block-scaled matmul on GB10.
    # That contract needs UE4M3 (uint8) scales in a swizzled tile layout;
    # quantize_nvfp4_lm_out emits that format. The scalar fallback path
    # inside kernel_gb10_nvfp4.cu reads the same UE4M3 layout via
    # lm_swizzled_scale_value, so a single quantization serves both.
    _load_op()
    lm_W = weights["lm_head_weight"]
    if _quantize_nvfp4_lm_out is None:
        raise RuntimeError(
            "quantize_nvfp4_lm_out op is missing — rebuild the extension "
            "for a Blackwell arch (MEGAKERNEL_CUDA_ARCHS=sm_121a,...) so "
            "the NVFP4 LM-head path is compiled in."
        )
    lm_rows, lm_cols = lm_W.shape
    if lm_cols != 1024:
        raise ValueError(
            f"LM head expects HIDDEN_SIZE=1024 cols; got {lm_cols}. "
            "The cuBLASLt FP4 plan and the scale-swizzle layout are "
            "hard-coded for 1024."
        )
    lm_scale_bytes = (
        ((lm_rows + NVFP4_TC_ROWS_PER_TILE - 1) // NVFP4_TC_ROWS_PER_TILE)
        * (lm_cols // NVFP4_TC_K_PER_TILE)
        * 512
    )
    lm_packed = torch.empty(lm_rows, lm_cols // 2, dtype=torch.uint8, device=lm_W.device)
    lm_scales = torch.zeros(lm_scale_bytes, dtype=torch.uint8, device=lm_W.device)
    _quantize_nvfp4_lm_out(lm_packed, lm_scales, lm_W.contiguous())
    packed_bytes += lm_packed.numel()
    scale_bytes += lm_scales.numel()

    weights["nvfp4"] = {
        "group_size": group_size,
        "layer_data": layer_data_nvfp4,
        "lm_head_weight_packed": lm_packed,
        "lm_head_scales": lm_scales,
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

    # Trust the caller's `backend` here — don't re-resolve via the env
    # var, since this function is called from Decoder.__init__ which has
    # already picked the storage format ("bf16" or "nvfp4") that needs
    # loading. Re-resolving would let MEGAKERNEL_BACKEND=bf16_fp4lm sneak
    # in and skip the FP4 LM-head weight attachment.
    if verbose:
        print(f"Loading {model_name} (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    weights, _ = _unify_weights_from_hf(model)
    if verbose:
        layer_data = weights["layer_data"]
        total = sum(sum(t.numel() for t in ld["ptrs"]) for ld in layer_data) + weights["lm_head_weight"].numel()
        print(f"BF16 weights: {total/1e6:.1f}M params ({total*2/1e6:.0f} MB)")
    # Hold the HF model alive on the weights dict — the layer tensors are
    # VIEWS into model.state_dict(), so model GC would invalidate them.
    weights["_hf_model_keepalive"] = model
    if backend == "nvfp4":
        _attach_nvfp4_weights(weights, group_size=nvfp4_group_size, verbose=verbose)
    return weights, tokenizer


def _unify_weights_from_hf(model):
    """Build the kernel's flat weight pack (layer_data + embed/final/lm_head)
    from an already-loaded HF model. The returned tensors are *views into
    the HF model's parameters* — no new allocation, no extra memory cost.

    Used by :func:`load_weights` for the trainer-owned base model, AND by
    LoraMegakernelTrainer.unified_decoder_from_session(...) so the sample
    path's Decoder shares weights with the trainer's HF base instead of
    re-loading 1.5 GB.

    Handles PEFT-wrapped models: when a Linear has been replaced by a
    LoraLinear, the original frozen weight lives at base_layer.weight
    instead of weight. We map both forms to the same underlying tensor.

    Returns (weights_dict, state_dict_ref) — the second element is kept
    so callers can hold a reference if they need to keep state alive.
    """
    state = model.state_dict()
    # PEFT renames each LoRA-wrapped Linear's underlying weight from
    # `…proj.weight` -> `…proj.base_layer.weight`. Build an alias dict so
    # the lookups below work whether the model is PEFT-wrapped or not.
    aliased = dict(state)
    for k, v in list(state.items()):
        if k.endswith(".base_layer.weight"):
            short = k[: -len(".base_layer.weight")] + ".weight"
            aliased[short] = v
    state = aliased
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
    return weights, state


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


def _alloc_prefill_buffers(max_tokens: int) -> dict:
    """Allocate the scratch buffers prefill_bf16 writes into.

    Mirrors final_bench.alloc_prefill_buffers; kept here so Decoder.prefill
    can self-manage its own buffers without the bench script being in scope.
    """
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    f32 = dict(dtype=torch.float32, device="cuda")
    i32 = dict(dtype=torch.int32, device="cuda")
    mx = max(DN_CONV_CHANNELS, FA_QPROJ_SIZE, INTERMEDIATE_SIZE)
    return dict(
        hidden=torch.empty(max_tokens * HIDDEN_SIZE, **bf16),
        residual=torch.empty(max_tokens * HIDDEN_SIZE, **bf16),
        normalized=torch.empty(max_tokens * HIDDEN_SIZE, **bf16),
        proj_buf=torch.empty(max_tokens * mx, **bf16),
        proj_buf2=torch.empty(max_tokens * mx, **bf16),
        attn_buf=torch.empty(max_tokens * max(FA_Q_SIZE, FA_KV_SIZE), **bf16),
        mlp_buf=torch.empty(max_tokens * INTERMEDIATE_SIZE, **bf16),
        dn_out_buf=torch.empty(max_tokens * DN_V_SIZE, **bf16),
        beta_buf=torch.empty(max_tokens * DN_NUM_HEADS, **f32),
        alpha_buf=torch.empty(max_tokens * DN_NUM_HEADS, **f32),
        final_normed=torch.empty(HIDDEN_SIZE, **bf16),
        hidden_bf16_out=torch.empty(HIDDEN_SIZE, **bf16),
        lm_bmv=torch.empty(1024, **f32),
        lm_bmi=torch.empty(1024, **i32),
    )


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
        self.backend_label = {
            "nvfp4": "NVFP4 decode",
            "bf16_fp4lm": "BF16 + cuBLASLt FP4 LM head",
            "bf16": "BF16",
        }[self.backend]
        self._nvfp4_group_size = nvfp4_group_size
        self._needs_nvfp4_weights = self.backend in ("nvfp4", "bf16_fp4lm")

        if weights is None:
            # Use "nvfp4" so load_weights triggers _attach_nvfp4_weights.
            # bf16_fp4lm reuses the same FP4 LM-head weights but keeps the
            # BF16 layer weights for the decode trunk.
            load_backend = "nvfp4" if self._needs_nvfp4_weights else "bf16"
            weights, tokenizer = load_weights(
                model_name,
                verbose=verbose,
                backend=load_backend,
                nvfp4_group_size=nvfp4_group_size,
            )
        elif self._needs_nvfp4_weights:
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
        if self._needs_nvfp4_weights:
            nvfp4 = weights["nvfp4"]
            # The full-NVFP4 decode path needs all layer weights packed; the
            # bf16_fp4lm hybrid only needs the LM-head FP4 weights.
            if self.backend == "nvfp4":
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
        # cuBLASLt FP4 LM-head scratch buffers. Allocated on either the
        # full-NVFP4 path or the bf16+fp4lm hybrid; pure BF16 pays nothing.
        self._lm_hidden_bf16 = None
        self._lm_hidden_packed = None
        self._lm_hidden_scales = None
        self._lm_logits_f16 = None
        if self._needs_nvfp4_weights:
            self._lm_hidden_bf16 = torch.empty(
                (LM_HEAD_TENSORCORE_N, HIDDEN_SIZE), **bf16)
            self._lm_hidden_packed = torch.empty(
                LM_HEAD_TENSORCORE_PACKED_BYTES, dtype=torch.uint8, device="cuda")
            self._lm_hidden_scales = torch.empty(
                LM_HEAD_TENSORCORE_SCALE_BYTES, dtype=torch.uint8, device="cuda")
            self._lm_logits_f16 = torch.empty(
                (LM_HEAD_TENSORCORE_N, VOCAB_SIZE), dtype=torch.float16, device="cuda")

    def step(self, token_id: int) -> int:
        """Decode one token. Returns next token id."""
        if self.backend == "nvfp4":
            _decode_nvfp4(
                self._out_token, token_id,
                self._embed_weight, self._layer_weights_packed_nvfp4,
                self._final_norm_weight, self._lm_head_weight_packed, self._lm_head_scales,
                self._lm_hidden_bf16, self._lm_hidden_packed,
                self._lm_hidden_scales, self._lm_logits_f16,
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
            if self.backend == "bf16_fp4lm":
                # Override the BF16 LM head argmax with the cuBLASLt FP4
                # block-scaled LM head on the same f32 normalized hidden.
                # 32/32 greedy parity with HF on the test prompt; the
                # double-LM-head pays ~970 us/step extra. Once the BF16
                # decode kernel grows an "early-exit before LM head"
                # mode, this overhead drops to net-zero / negative.
                torch.ops.qwen35_megakernel_bf16_C.lm_head_nvfp4_from_f32(
                    self._out_token, self._normalized,
                    self._lm_head_weight_packed, self._lm_head_scales,
                    self._lm_hidden_bf16, self._lm_hidden_packed,
                    self._lm_hidden_scales, self._lm_logits_f16,
                    self._block_max_vals, self._block_max_idxs,
                    self._nvfp4_group_size,
                )
        self._position += 1
        return self._out_token.item()

    def reset(self):
        self._position = 0
        self._fa_k_cache.zero_()
        self._fa_v_cache.zero_()
        self._dn_states.zero_()
        self._conv_bufs.zero_()

    def prefill(self, prompt_ids) -> int:
        """Process a multi-token prompt via the cuBLAS+graph prefill kernel.

        Equivalent in semantics to ``reset()`` followed by
        ``[step(t) for t in prompt_ids]``, but in one dispatch instead
        of one decode kernel per token. Populates the KV cache, DN
        recurrent state, and conv ring buffers; returns the predicted
        next token id (the would-be output of ``step(prompt_ids[-1])``).

        BF16 backend only — NVFP4 prefill is not yet implemented in the
        kernel extension.
        """
        if isinstance(prompt_ids, torch.Tensor):
            ids_t_full = prompt_ids.to(dtype=torch.int32, device="cuda").contiguous()
        else:
            ids_t_full = torch.tensor(list(prompt_ids), dtype=torch.int32, device="cuda")
        prompt_len_full = ids_t_full.numel()
        if prompt_len_full == 0:
            raise ValueError("Decoder.prefill: prompt_ids must be non-empty")
        if prompt_len_full > MAX_SEQ_LEN:
            raise ValueError(
                f"Decoder.prefill: prompt_len={prompt_len_full} exceeds MAX_SEQ_LEN={MAX_SEQ_LEN}"
            )

        if self.backend == "nvfp4":
            # Full NVFP4 prefill: persistent megakernel processes the
            # entire prompt with FP4 layer projections, then runs the
            # cuBLASLt FP4 block-scaled LM head on the final hidden.
            self.reset()
            torch.ops.qwen35_megakernel_bf16_C.prefill_megakernel_nvfp4(
                self._out_token, ids_t_full,
                self._embed_weight, self._layer_weights_packed_nvfp4,
                self._final_norm_weight,
                self._lm_head_weight_packed, self._lm_head_scales,
                self._lm_hidden_bf16, self._lm_hidden_packed,
                self._lm_hidden_scales, self._lm_logits_f16,
                self._fa_k_cache, self._fa_v_cache,
                self._dn_states, self._conv_bufs,
                self._hidden, self._activations, self._residual,
                self._qkv_scratch, self._kv_scratch, self._attn_out,
                self._mlp_inter, self._z_scratch, self._beta_scratch,
                self._alpha_scratch, self._normalized,
                self._barrier_counter, self._barrier_generation,
                self._block_max_vals, self._block_max_idxs,
                self._lm_sync_counter,
                MAX_SEQ_LEN, self._nvfp4_group_size,
            )
            self._position = prompt_len_full
            return self._out_token.item()
        # bf16 / bf16_fp4lm path uses the eager BF16 prefill below.

        if isinstance(prompt_ids, torch.Tensor):
            ids_t = prompt_ids.to(dtype=torch.int32, device="cuda").contiguous()
        else:
            ids_t = torch.tensor(list(prompt_ids), dtype=torch.int32, device="cuda")
        prompt_len = ids_t.numel()
        if prompt_len == 0:
            raise ValueError("Decoder.prefill: prompt_ids must be non-empty")
        if prompt_len > MAX_SEQ_LEN:
            raise ValueError(
                f"Decoder.prefill: prompt_len={prompt_len} exceeds MAX_SEQ_LEN={MAX_SEQ_LEN}"
            )

        bufs = self._get_prefill_buffers(prompt_len)
        self.reset()
        _prefill_bf16(
            self._out_token,
            ids_t,
            self._embed_weight,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._lm_head_weight,
            self._fa_k_cache,
            self._fa_v_cache,
            self._dn_states,
            self._conv_bufs,
            bufs["hidden"],
            bufs["residual"],
            bufs["normalized"],
            bufs["proj_buf"],
            bufs["proj_buf2"],
            bufs["attn_buf"],
            bufs["mlp_buf"],
            bufs["dn_out_buf"],
            bufs["beta_buf"],
            bufs["alpha_buf"],
            bufs["final_normed"],
            bufs["hidden_bf16_out"],
            bufs["lm_bmv"],
            bufs["lm_bmi"],
        )
        self._hidden.copy_(bufs["hidden_bf16_out"])
        self._position = prompt_len
        if self.backend == "bf16_fp4lm":
            # The BF16 prefill writes `final_normed` in bf16. Convert to
            # f32 in-place into `_normalized` (sized HIDDEN_SIZE f32) and
            # run the cuBLASLt FP4 LM head, overriding `_out_token`.
            self._normalized.copy_(bufs["final_normed"].to(torch.float32))
            torch.ops.qwen35_megakernel_bf16_C.lm_head_nvfp4_from_f32(
                self._out_token, self._normalized,
                self._lm_head_weight_packed, self._lm_head_scales,
                self._lm_hidden_bf16, self._lm_hidden_packed,
                self._lm_hidden_scales, self._lm_logits_f16,
                self._block_max_vals, self._block_max_idxs,
                self._nvfp4_group_size,
            )
        return self._out_token.item()

    def _get_prefill_buffers(self, max_tokens: int) -> dict:
        """Lazy-allocate (and grow) the prefill scratch buffers."""
        cap = getattr(self, "_prefill_buf_capacity", 0)
        if cap < max_tokens:
            self._prefill_buffers = _alloc_prefill_buffers(max_tokens)
            self._prefill_buf_capacity = max_tokens
        return self._prefill_buffers

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        if self.backend == "bf16" and len(ids) > 1:
            # prefill the entire prompt; pred is the model's prediction at
            # position len(ids), i.e. the would-be first generated token.
            pred = self.prefill(ids)
        else:
            self.reset()
            for tid in ids[:-1]:
                self.step(tid)
            pred = self.step(ids[-1])
        out = []
        eos = self.tokenizer.eos_token_id
        for _ in range(max_tokens):
            if pred == eos:
                break
            out.append(pred)
            pred = self.step(pred)
        return self.tokenizer.decode(out, skip_special_tokens=True)
