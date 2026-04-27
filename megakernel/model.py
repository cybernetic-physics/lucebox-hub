"""Weight loading and decode API for Qwen3.5-0.8B bf16 megakernel."""

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

_decode = None


def _load_op():
    global _decode
    if _decode is None:
        import qwen35_megakernel_bf16_C
        _decode = torch.ops.qwen35_megakernel_bf16_C.decode


def load_weights(model_name="Qwen/Qwen3.5-0.8B", verbose=True):
    """Load Qwen3.5-0.8B weights as bf16 (no quantization)."""
    if not verbose:
        import os
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    from transformers import AutoModelForCausalLM, AutoTokenizer

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


class Decoder:
    """Stateful decoder for Qwen3.5-0.8B bf16 megakernel."""

    def __init__(self, weights=None, tokenizer=None,
                 model_name="Qwen/Qwen3.5-0.8B", verbose=True,
                 max_seq_len=MAX_SEQ_LEN):
        _load_op()

        if weights is None:
            weights, tokenizer = load_weights(model_name, verbose=verbose)
        self.tokenizer = tokenizer
        self._position = 0
        self._max_seq_len = max_seq_len
        self._weights = weights
        self._embed_weight = weights["embed_weight"]
        self._final_norm_weight = weights["final_norm_weight"]
        self._lm_head_weight = weights["lm_head_weight"]
        self._layer_weights_packed = _pack_layer_weights(weights["layer_data"])

        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        f32 = dict(dtype=torch.float32, device="cuda")
        i32 = dict(dtype=torch.int32, device="cuda")
        u32 = dict(dtype=torch.uint32, device="cuda")

        n_fa = sum(1 for t in LAYER_TYPE if t == 1)
        self._fa_k_cache = torch.zeros(n_fa, FA_NUM_KV_HEADS, max_seq_len, FA_HEAD_DIM, **bf16)
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
            self._position, self._max_seq_len,
        )
        self._position += 1
        return self._out_token.item()

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
