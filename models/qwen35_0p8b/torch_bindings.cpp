/**
 * PyTorch bindings for Qwen3.5-0.8B bf16 megakernel — decode.
 */

#include <Python.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/library.h>

#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)
#define _STRINGIFY(A) #A
#define STRINGIFY(A) _STRINGIFY(A)

#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)

#define REGISTER_EXTENSION(NAME)                                               \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                     \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT,                 \
                                        STRINGIFY(NAME), nullptr, 0, nullptr}; \
    return PyModule_Create(&module);                                           \
  }

struct LayerWeights {
    int layer_type;
    int _pad[3];
    void *ptrs[14];  // max(11 FA, 14 DN) pointers — all bf16, no scales
};

struct LayerWeightsNVFP4 {
    int layer_type;
    int group_size;
    int _pad[2];
    void *ptrs[24];  // hot decode weights become packed fp4 + per-group scales
};

extern "C" void launch_decode(
    int input_token_id, int *output_token_id,
    const void *embed_weight, const LayerWeights *layer_weights,
    const void *final_norm_weight, const void *lm_head_weight,
    void *fa_k_cache, void *fa_v_cache,
    void *dn_states, void *conv_bufs,
    void *hidden_buffer, void *g_activations, void *g_residual,
    void *g_qkv_scratch, void *g_kv_scratch, void *g_attn_out,
    void *g_mlp_inter, void *g_z_scratch, void *g_beta_scratch,
    void *g_alpha_scratch, void *g_normalized,
    unsigned int *barrier_counter, unsigned int *barrier_generation,
    float *block_max_vals, int *block_max_idxs,
    unsigned int *lm_sync_counter,
    int position, int max_seq_len, cudaStream_t stream);

extern "C" void launch_decode_nvfp4(
    int input_token_id, int *output_token_id,
    const void *embed_weight, const LayerWeightsNVFP4 *layer_weights,
    const void *final_norm_weight,
    const void *lm_head_weight_packed, const void *lm_head_scales,
    void *fa_k_cache, void *fa_v_cache,
    void *dn_states, void *conv_bufs,
    void *hidden_buffer, void *g_activations, void *g_residual,
    void *g_qkv_scratch, void *g_kv_scratch, void *g_attn_out,
    void *g_mlp_inter, void *g_z_scratch, void *g_beta_scratch,
    void *g_alpha_scratch, void *g_normalized,
    unsigned int *barrier_counter, unsigned int *barrier_generation,
    float *block_max_vals, int *block_max_idxs,
    unsigned int *lm_sync_counter,
    int position, int max_seq_len, int group_size, cudaStream_t stream);

extern "C" void launch_quantize_nvfp4_out(
    const void *weight, int rows, int cols, int group_size,
    void *packed_out, void *scales_out, cudaStream_t stream);

void decode(
    torch::Tensor output_token, int64_t input_token_id,
    torch::Tensor embed_weight, torch::Tensor layer_weights_packed,
    torch::Tensor final_norm_weight, torch::Tensor lm_head_weight,
    torch::Tensor fa_k_cache, torch::Tensor fa_v_cache,
    torch::Tensor dn_states, torch::Tensor conv_bufs,
    torch::Tensor hidden_buffer, torch::Tensor activations, torch::Tensor residual,
    torch::Tensor qkv_scratch, torch::Tensor kv_scratch, torch::Tensor attn_out,
    torch::Tensor mlp_inter, torch::Tensor z_scratch, torch::Tensor beta_scratch,
    torch::Tensor alpha_scratch, torch::Tensor normalized,
    torch::Tensor barrier_counter, torch::Tensor barrier_generation,
    torch::Tensor block_max_vals, torch::Tensor block_max_idxs,
    torch::Tensor lm_sync_counter, int64_t position, int64_t max_seq_len)
{
    launch_decode(
        (int)input_token_id, (int*)output_token.data_ptr(),
        embed_weight.data_ptr(),
        reinterpret_cast<const LayerWeights*>(layer_weights_packed.data_ptr()),
        final_norm_weight.data_ptr(), lm_head_weight.data_ptr(),
        fa_k_cache.data_ptr(), fa_v_cache.data_ptr(),
        dn_states.data_ptr(), conv_bufs.data_ptr(),
        hidden_buffer.data_ptr(), activations.data_ptr(), residual.data_ptr(),
        qkv_scratch.data_ptr(), kv_scratch.data_ptr(), attn_out.data_ptr(),
        mlp_inter.data_ptr(), z_scratch.data_ptr(), beta_scratch.data_ptr(),
        alpha_scratch.data_ptr(), normalized.data_ptr(),
        (unsigned int*)barrier_counter.data_ptr(), (unsigned int*)barrier_generation.data_ptr(),
        (float*)block_max_vals.data_ptr(), (int*)block_max_idxs.data_ptr(),
        (unsigned int*)lm_sync_counter.data_ptr(),
        (int)position, (int)max_seq_len,
        c10::cuda::getCurrentCUDAStream().stream());
}

void decode_nvfp4(
    torch::Tensor output_token, int64_t input_token_id,
    torch::Tensor embed_weight, torch::Tensor layer_weights_packed,
    torch::Tensor final_norm_weight,
    torch::Tensor lm_head_weight_packed, torch::Tensor lm_head_scales,
    torch::Tensor fa_k_cache, torch::Tensor fa_v_cache,
    torch::Tensor dn_states, torch::Tensor conv_bufs,
    torch::Tensor hidden_buffer, torch::Tensor activations, torch::Tensor residual,
    torch::Tensor qkv_scratch, torch::Tensor kv_scratch, torch::Tensor attn_out,
    torch::Tensor mlp_inter, torch::Tensor z_scratch, torch::Tensor beta_scratch,
    torch::Tensor alpha_scratch, torch::Tensor normalized,
    torch::Tensor barrier_counter, torch::Tensor barrier_generation,
    torch::Tensor block_max_vals, torch::Tensor block_max_idxs,
    torch::Tensor lm_sync_counter, int64_t position, int64_t max_seq_len,
    int64_t group_size)
{
    launch_decode_nvfp4(
        (int)input_token_id, (int*)output_token.data_ptr(),
        embed_weight.data_ptr(),
        reinterpret_cast<const LayerWeightsNVFP4*>(layer_weights_packed.data_ptr()),
        final_norm_weight.data_ptr(),
        lm_head_weight_packed.data_ptr(), lm_head_scales.data_ptr(),
        fa_k_cache.data_ptr(), fa_v_cache.data_ptr(),
        dn_states.data_ptr(), conv_bufs.data_ptr(),
        hidden_buffer.data_ptr(), activations.data_ptr(), residual.data_ptr(),
        qkv_scratch.data_ptr(), kv_scratch.data_ptr(), attn_out.data_ptr(),
        mlp_inter.data_ptr(), z_scratch.data_ptr(), beta_scratch.data_ptr(),
        alpha_scratch.data_ptr(), normalized.data_ptr(),
        (unsigned int*)barrier_counter.data_ptr(), (unsigned int*)barrier_generation.data_ptr(),
        (float*)block_max_vals.data_ptr(), (int*)block_max_idxs.data_ptr(),
        (unsigned int*)lm_sync_counter.data_ptr(),
        (int)position, (int)max_seq_len, (int)group_size,
        c10::cuda::getCurrentCUDAStream().stream());
}

void quantize_nvfp4_out(
    torch::Tensor packed_out,
    torch::Tensor scales_out,
    torch::Tensor weight,
    int64_t group_size)
{
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(weight.dim() == 2, "weight must be a 2D [out_dim, in_dim] tensor");
    TORCH_CHECK(weight.scalar_type() == torch::kBFloat16, "weight must be bfloat16");
    TORCH_CHECK(group_size > 0 && (group_size % 2) == 0, "group_size must be a positive even integer");

    auto rows = static_cast<int>(weight.size(0));
    auto cols = static_cast<int>(weight.size(1));
    TORCH_CHECK((cols % 2) == 0, "in_dim must be divisible by 2 for packed fp4 output");
    TORCH_CHECK((cols % group_size) == 0, "in_dim must be divisible by group_size");
    TORCH_CHECK(packed_out.is_cuda() && packed_out.is_contiguous(), "packed_out must be contiguous CUDA");
    TORCH_CHECK(scales_out.is_cuda() && scales_out.is_contiguous(), "scales_out must be contiguous CUDA");
    TORCH_CHECK(packed_out.scalar_type() == torch::kUInt8, "packed_out must be uint8");
    TORCH_CHECK(scales_out.scalar_type() == torch::kFloat16, "scales_out must be float16");
    TORCH_CHECK(
        packed_out.numel() == (int64_t)rows * (cols / 2),
        "packed_out has the wrong size");
    TORCH_CHECK(
        scales_out.numel() == (int64_t)rows * (cols / group_size),
        "scales_out has the wrong size");

    launch_quantize_nvfp4_out(
        weight.data_ptr(), rows, cols, (int)group_size,
        packed_out.data_ptr(), scales_out.data_ptr(),
        c10::cuda::getCurrentCUDAStream().stream());
}

// ===== Prefill BF16 =====

// Must mirror LoraPFSet in prefill.cu exactly.
struct LoraPFSet {
    const void *fa_q_A,    *fa_q_B;
    const void *fa_k_A,    *fa_k_B;
    const void *fa_v_A,    *fa_v_B;
    const void *fa_o_A,    *fa_o_B;
    const void *fa_gate_A, *fa_gate_B;
    const void *fa_up_A,   *fa_up_B;
    const void *fa_down_A, *fa_down_B;
    const void *dn_qkv_A,  *dn_qkv_B;
    const void *dn_z_A,    *dn_z_B;
    const void *dn_out_A,  *dn_out_B;
    const void *dn_gate_A, *dn_gate_B;
    const void *dn_up_A,   *dn_up_B;
    const void *dn_down_A, *dn_down_B;
};

// Must mirror SavedActivationsPF in prefill.cu. Each pointer, when
// non-null, is a flat [NUM_LAYERS, S, DIM] bf16 buffer the forward
// writes into at the respective per-layer checkpoint. All-null ⇒
// inference-only, zero overhead.
struct SavedActivationsPF {
    void *hidden_in;
    void *normalized_in;
    void *normalized_post_attn;
    void *mlp_inter;
    // Slice B.2 saves — see prefill.cu for the layout details.
    void *attn_out_pre_o;
    void *h_post_attn;
};

extern "C" void launch_prefill_bf16(
    const int *token_ids, int seq_len, int *output_token,
    const void *embed_weight, const LayerWeights *layers,
    const void *final_norm_w, const void *lm_head_w,
    void *fa_k_cache, void *fa_v_cache, void *dn_states, void *conv_bufs,
    void *hidden, void *residual, void *normalized,
    void *proj_buf, void *proj_buf2, void *attn_buf, void *mlp_buf,
    void *dn_out_buf, void *beta_buf, void *alpha_buf,
    void *final_normed, void *hidden_bf16_out,
    void *lm_bmv, void *lm_bmi,
    LoraPFSet lora, int lora_rank, float lora_scaling, void *lora_h_ws,
    SavedActivationsPF saved,
    int max_seq,
    cudaStream_t stream);

// Helper to derive the FA KV cache row count from its tensor shape.
// fa_k_cache shape is [n_fa, FA_NUM_KV_HEADS, MAX_SEQ_LEN, FA_HEAD_DIM].
static inline int fa_max_seq_from_cache(const torch::Tensor &fa_k_cache) {
    TORCH_CHECK(fa_k_cache.dim() == 4,
                "fa_k_cache must be 4-D [n_fa, kv_heads, max_seq, head_dim]");
    return (int)fa_k_cache.size(2);
}

extern "C" void launch_prefill_bf16_mega(
    const int *token_ids, int seq_len, int *output_token,
    const void *embed_weight, const LayerWeights *layers,
    const void *final_norm_w, const void *lm_head_w,
    void *fa_k_cache, void *fa_v_cache, void *dn_states, void *conv_bufs,
    void *hidden, void *residual, void *normalized,
    void *proj_buf, void *proj_buf2, void *attn_buf, void *mlp_buf,
    void *dn_out_buf, void *beta_buf, void *alpha_buf,
    void *final_normed, void *hidden_bf16_out,
    void *lm_bmv, void *lm_bmi,
    cudaStream_t stream);

void prefill_bf16(
    torch::Tensor output_token, torch::Tensor token_ids,
    torch::Tensor embed_weight, torch::Tensor layer_weights_packed,
    torch::Tensor final_norm_weight, torch::Tensor lm_head_weight,
    torch::Tensor fa_k_cache, torch::Tensor fa_v_cache,
    torch::Tensor dn_states, torch::Tensor conv_bufs,
    torch::Tensor hidden, torch::Tensor residual, torch::Tensor normalized,
    torch::Tensor proj_buf, torch::Tensor proj_buf2,
    torch::Tensor attn_buf, torch::Tensor mlp_buf,
    torch::Tensor dn_out_buf, torch::Tensor beta_buf, torch::Tensor alpha_buf,
    torch::Tensor final_normed, torch::Tensor hidden_bf16_out,
    torch::Tensor lm_bmv, torch::Tensor lm_bmi)
{
    LoraPFSet lora{};  // all-null: inference path, no extra work
    launch_prefill_bf16(
        (const int*)token_ids.data_ptr(), token_ids.size(0),
        (int*)output_token.data_ptr(),
        embed_weight.data_ptr(),
        reinterpret_cast<const LayerWeights*>(layer_weights_packed.data_ptr()),
        final_norm_weight.data_ptr(), lm_head_weight.data_ptr(),
        fa_k_cache.data_ptr(), fa_v_cache.data_ptr(),
        dn_states.data_ptr(), conv_bufs.data_ptr(),
        hidden.data_ptr(), residual.data_ptr(), normalized.data_ptr(),
        proj_buf.data_ptr(), proj_buf2.data_ptr(),
        attn_buf.data_ptr(), mlp_buf.data_ptr(),
        dn_out_buf.data_ptr(), beta_buf.data_ptr(), alpha_buf.data_ptr(),
        final_normed.data_ptr(), hidden_bf16_out.data_ptr(),
        lm_bmv.data_ptr(), lm_bmi.data_ptr(),
        lora, 0, 0.0f, nullptr,
        SavedActivationsPF{},
        fa_max_seq_from_cache(fa_k_cache),
        c10::cuda::getCurrentCUDAStream().stream());
}

// Helper: pull raw data_ptr from an optional Tensor (returns nullptr if
// the tensor is undefined or zero-sized so the kernel can null-check the
// corresponding LoRA slot to disable that specific projection).
static inline const void *opt_ptr(const torch::Tensor &t) {
    if (!t.defined() || t.numel() == 0) return nullptr;
    return t.data_ptr();
}

void prefill_bf16_with_lora(
    torch::Tensor output_token, torch::Tensor token_ids,
    torch::Tensor embed_weight, torch::Tensor layer_weights_packed,
    torch::Tensor final_norm_weight, torch::Tensor lm_head_weight,
    torch::Tensor fa_k_cache, torch::Tensor fa_v_cache,
    torch::Tensor dn_states, torch::Tensor conv_bufs,
    torch::Tensor hidden, torch::Tensor residual, torch::Tensor normalized,
    torch::Tensor proj_buf, torch::Tensor proj_buf2,
    torch::Tensor attn_buf, torch::Tensor mlp_buf,
    torch::Tensor dn_out_buf, torch::Tensor beta_buf, torch::Tensor alpha_buf,
    torch::Tensor final_normed, torch::Tensor hidden_bf16_out,
    torch::Tensor lm_bmv, torch::Tensor lm_bmi,
    // LoRA A/B tensors, one pair per trainable projection. Pass an empty
    // tensor to disable LoRA on that specific projection.
    torch::Tensor fa_q_A,    torch::Tensor fa_q_B,
    torch::Tensor fa_k_A,    torch::Tensor fa_k_B,
    torch::Tensor fa_v_A,    torch::Tensor fa_v_B,
    torch::Tensor fa_o_A,    torch::Tensor fa_o_B,
    torch::Tensor fa_gate_A, torch::Tensor fa_gate_B,
    torch::Tensor fa_up_A,   torch::Tensor fa_up_B,
    torch::Tensor fa_down_A, torch::Tensor fa_down_B,
    torch::Tensor dn_qkv_A,  torch::Tensor dn_qkv_B,
    torch::Tensor dn_z_A,    torch::Tensor dn_z_B,
    torch::Tensor dn_out_A,  torch::Tensor dn_out_B,
    torch::Tensor dn_gate_A, torch::Tensor dn_gate_B,
    torch::Tensor dn_up_A,   torch::Tensor dn_up_B,
    torch::Tensor dn_down_A, torch::Tensor dn_down_B,
    int64_t lora_rank, double lora_scaling, torch::Tensor lora_h_ws)
{
    LoraPFSet lora{
        opt_ptr(fa_q_A),    opt_ptr(fa_q_B),
        opt_ptr(fa_k_A),    opt_ptr(fa_k_B),
        opt_ptr(fa_v_A),    opt_ptr(fa_v_B),
        opt_ptr(fa_o_A),    opt_ptr(fa_o_B),
        opt_ptr(fa_gate_A), opt_ptr(fa_gate_B),
        opt_ptr(fa_up_A),   opt_ptr(fa_up_B),
        opt_ptr(fa_down_A), opt_ptr(fa_down_B),
        opt_ptr(dn_qkv_A),  opt_ptr(dn_qkv_B),
        opt_ptr(dn_z_A),    opt_ptr(dn_z_B),
        opt_ptr(dn_out_A),  opt_ptr(dn_out_B),
        opt_ptr(dn_gate_A), opt_ptr(dn_gate_B),
        opt_ptr(dn_up_A),   opt_ptr(dn_up_B),
        opt_ptr(dn_down_A), opt_ptr(dn_down_B),
    };
    launch_prefill_bf16(
        (const int*)token_ids.data_ptr(), token_ids.size(0),
        (int*)output_token.data_ptr(),
        embed_weight.data_ptr(),
        reinterpret_cast<const LayerWeights*>(layer_weights_packed.data_ptr()),
        final_norm_weight.data_ptr(), lm_head_weight.data_ptr(),
        fa_k_cache.data_ptr(), fa_v_cache.data_ptr(),
        dn_states.data_ptr(), conv_bufs.data_ptr(),
        hidden.data_ptr(), residual.data_ptr(), normalized.data_ptr(),
        proj_buf.data_ptr(), proj_buf2.data_ptr(),
        attn_buf.data_ptr(), mlp_buf.data_ptr(),
        dn_out_buf.data_ptr(), beta_buf.data_ptr(), alpha_buf.data_ptr(),
        final_normed.data_ptr(), hidden_bf16_out.data_ptr(),
        lm_bmv.data_ptr(), lm_bmi.data_ptr(),
        lora, (int)lora_rank, (float)lora_scaling, lora_h_ws.data_ptr(),
        SavedActivationsPF{},
        fa_max_seq_from_cache(fa_k_cache),
        c10::cuda::getCurrentCUDAStream().stream());
}

// ===== prefill_bf16_train_step =====
// Same as prefill_bf16_with_lora but also writes per-layer activation
// slabs that the backward kernels consume. Pass empty tensors for any
// save-slot you don't need — empty → nullptr → that save is skipped.
//
// Expected shapes (bf16, cuda):
//   hidden_in_save          : [NUM_LAYERS, S, HIDDEN]
//   normalized_in_save      : [NUM_LAYERS, S, HIDDEN]
//   normalized_post_attn_sv : [NUM_LAYERS, S, HIDDEN]
//   mlp_inter_save          : [NUM_LAYERS, S, INTER]
void prefill_bf16_train_step(
    torch::Tensor output_token, torch::Tensor token_ids,
    torch::Tensor embed_weight, torch::Tensor layer_weights_packed,
    torch::Tensor final_norm_weight, torch::Tensor lm_head_weight,
    torch::Tensor fa_k_cache, torch::Tensor fa_v_cache,
    torch::Tensor dn_states, torch::Tensor conv_bufs,
    torch::Tensor hidden, torch::Tensor residual, torch::Tensor normalized,
    torch::Tensor proj_buf, torch::Tensor proj_buf2,
    torch::Tensor attn_buf, torch::Tensor mlp_buf,
    torch::Tensor dn_out_buf, torch::Tensor beta_buf, torch::Tensor alpha_buf,
    torch::Tensor final_normed, torch::Tensor hidden_bf16_out,
    torch::Tensor lm_bmv, torch::Tensor lm_bmi,
    // LoRA A/B tensors (26 total)
    torch::Tensor fa_q_A,    torch::Tensor fa_q_B,
    torch::Tensor fa_k_A,    torch::Tensor fa_k_B,
    torch::Tensor fa_v_A,    torch::Tensor fa_v_B,
    torch::Tensor fa_o_A,    torch::Tensor fa_o_B,
    torch::Tensor fa_gate_A, torch::Tensor fa_gate_B,
    torch::Tensor fa_up_A,   torch::Tensor fa_up_B,
    torch::Tensor fa_down_A, torch::Tensor fa_down_B,
    torch::Tensor dn_qkv_A,  torch::Tensor dn_qkv_B,
    torch::Tensor dn_z_A,    torch::Tensor dn_z_B,
    torch::Tensor dn_out_A,  torch::Tensor dn_out_B,
    torch::Tensor dn_gate_A, torch::Tensor dn_gate_B,
    torch::Tensor dn_up_A,   torch::Tensor dn_up_B,
    torch::Tensor dn_down_A, torch::Tensor dn_down_B,
    int64_t lora_rank, double lora_scaling, torch::Tensor lora_h_ws,
    // Activation-save slabs (empty tensor ⇒ disable that save)
    torch::Tensor hidden_in_save,
    torch::Tensor normalized_in_save,
    torch::Tensor normalized_post_attn_save,
    torch::Tensor mlp_inter_save,
    // Slice B.2 saves — pass empty tensors to keep this op backward-compat.
    torch::Tensor attn_out_pre_o_save,
    torch::Tensor h_post_attn_save)
{
    LoraPFSet lora{
        opt_ptr(fa_q_A),    opt_ptr(fa_q_B),
        opt_ptr(fa_k_A),    opt_ptr(fa_k_B),
        opt_ptr(fa_v_A),    opt_ptr(fa_v_B),
        opt_ptr(fa_o_A),    opt_ptr(fa_o_B),
        opt_ptr(fa_gate_A), opt_ptr(fa_gate_B),
        opt_ptr(fa_up_A),   opt_ptr(fa_up_B),
        opt_ptr(fa_down_A), opt_ptr(fa_down_B),
        opt_ptr(dn_qkv_A),  opt_ptr(dn_qkv_B),
        opt_ptr(dn_z_A),    opt_ptr(dn_z_B),
        opt_ptr(dn_out_A),  opt_ptr(dn_out_B),
        opt_ptr(dn_gate_A), opt_ptr(dn_gate_B),
        opt_ptr(dn_up_A),   opt_ptr(dn_up_B),
        opt_ptr(dn_down_A), opt_ptr(dn_down_B),
    };
    SavedActivationsPF saved{
        const_cast<void*>(opt_ptr(hidden_in_save)),
        const_cast<void*>(opt_ptr(normalized_in_save)),
        const_cast<void*>(opt_ptr(normalized_post_attn_save)),
        const_cast<void*>(opt_ptr(mlp_inter_save)),
        const_cast<void*>(opt_ptr(attn_out_pre_o_save)),
        const_cast<void*>(opt_ptr(h_post_attn_save)),
    };
    launch_prefill_bf16(
        (const int*)token_ids.data_ptr(), token_ids.size(0),
        (int*)output_token.data_ptr(),
        embed_weight.data_ptr(),
        reinterpret_cast<const LayerWeights*>(layer_weights_packed.data_ptr()),
        final_norm_weight.data_ptr(), lm_head_weight.data_ptr(),
        fa_k_cache.data_ptr(), fa_v_cache.data_ptr(),
        dn_states.data_ptr(), conv_bufs.data_ptr(),
        hidden.data_ptr(), residual.data_ptr(), normalized.data_ptr(),
        proj_buf.data_ptr(), proj_buf2.data_ptr(),
        attn_buf.data_ptr(), mlp_buf.data_ptr(),
        dn_out_buf.data_ptr(), beta_buf.data_ptr(), alpha_buf.data_ptr(),
        final_normed.data_ptr(), hidden_bf16_out.data_ptr(),
        lm_bmv.data_ptr(), lm_bmi.data_ptr(),
        lora, (int)lora_rank, (float)lora_scaling, lora_h_ws.data_ptr(),
        saved,
        fa_max_seq_from_cache(fa_k_cache),
        c10::cuda::getCurrentCUDAStream().stream());
}

void prefill_bf16_mega(
    torch::Tensor output_token, torch::Tensor token_ids,
    torch::Tensor embed_weight, torch::Tensor layer_weights_packed,
    torch::Tensor final_norm_weight, torch::Tensor lm_head_weight,
    torch::Tensor fa_k_cache, torch::Tensor fa_v_cache,
    torch::Tensor dn_states, torch::Tensor conv_bufs,
    torch::Tensor hidden, torch::Tensor residual, torch::Tensor normalized,
    torch::Tensor proj_buf, torch::Tensor proj_buf2,
    torch::Tensor attn_buf, torch::Tensor mlp_buf,
    torch::Tensor dn_out_buf, torch::Tensor beta_buf, torch::Tensor alpha_buf,
    torch::Tensor final_normed, torch::Tensor hidden_bf16_out,
    torch::Tensor lm_bmv, torch::Tensor lm_bmi)
{
    launch_prefill_bf16_mega(
        (const int*)token_ids.data_ptr(), token_ids.size(0),
        (int*)output_token.data_ptr(),
        embed_weight.data_ptr(),
        reinterpret_cast<const LayerWeights*>(layer_weights_packed.data_ptr()),
        final_norm_weight.data_ptr(), lm_head_weight.data_ptr(),
        fa_k_cache.data_ptr(), fa_v_cache.data_ptr(),
        dn_states.data_ptr(), conv_bufs.data_ptr(),
        hidden.data_ptr(), residual.data_ptr(), normalized.data_ptr(),
        proj_buf.data_ptr(), proj_buf2.data_ptr(),
        attn_buf.data_ptr(), mlp_buf.data_ptr(),
        dn_out_buf.data_ptr(), beta_buf.data_ptr(), alpha_buf.data_ptr(),
        final_normed.data_ptr(), hidden_bf16_out.data_ptr(),
        lm_bmv.data_ptr(), lm_bmi.data_ptr(),
        c10::cuda::getCurrentCUDAStream().stream());
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("decode(Tensor output_token, int input_token_id, "
            "Tensor embed_weight, Tensor layer_weights_packed, "
            "Tensor final_norm_weight, Tensor lm_head_weight, "
            "Tensor fa_k_cache, Tensor fa_v_cache, Tensor dn_states, Tensor conv_bufs, "
            "Tensor hidden_buffer, Tensor activations, Tensor residual, "
            "Tensor qkv_scratch, Tensor kv_scratch, Tensor attn_out, "
            "Tensor mlp_inter, Tensor z_scratch, Tensor beta_scratch, "
            "Tensor alpha_scratch, Tensor normalized, "
            "Tensor barrier_counter, Tensor barrier_generation, "
            "Tensor block_max_vals, Tensor block_max_idxs, Tensor lm_sync_counter, "
            "int position, int max_seq_len) -> ()");
    ops.impl("decode", torch::kCUDA, &decode);

    ops.def("decode_nvfp4(Tensor output_token, int input_token_id, "
            "Tensor embed_weight, Tensor layer_weights_packed, "
            "Tensor final_norm_weight, Tensor lm_head_weight_packed, Tensor lm_head_scales, "
            "Tensor fa_k_cache, Tensor fa_v_cache, Tensor dn_states, Tensor conv_bufs, "
            "Tensor hidden_buffer, Tensor activations, Tensor residual, "
            "Tensor qkv_scratch, Tensor kv_scratch, Tensor attn_out, "
            "Tensor mlp_inter, Tensor z_scratch, Tensor beta_scratch, "
            "Tensor alpha_scratch, Tensor normalized, "
            "Tensor barrier_counter, Tensor barrier_generation, "
            "Tensor block_max_vals, Tensor block_max_idxs, Tensor lm_sync_counter, "
            "int position, int max_seq_len, int group_size) -> ()");
    ops.impl("decode_nvfp4", torch::kCUDA, &decode_nvfp4);

    ops.def("prefill_bf16(Tensor output_token, Tensor token_ids, "
            "Tensor embed_weight, Tensor layer_weights_packed, "
            "Tensor final_norm_weight, Tensor lm_head_weight, "
            "Tensor fa_k_cache, Tensor fa_v_cache, Tensor dn_states, Tensor conv_bufs, "
            "Tensor hidden, Tensor residual, Tensor normalized, "
            "Tensor proj_buf, Tensor proj_buf2, Tensor attn_buf, Tensor mlp_buf, "
            "Tensor dn_out_buf, Tensor beta_buf, Tensor alpha_buf, "
            "Tensor final_normed, Tensor hidden_bf16_out, "
            "Tensor lm_bmv, Tensor lm_bmi) -> ()");
    ops.impl("prefill_bf16", torch::kCUDA, &prefill_bf16);

    ops.def("prefill_bf16_with_lora(Tensor output_token, Tensor token_ids, "
            "Tensor embed_weight, Tensor layer_weights_packed, "
            "Tensor final_norm_weight, Tensor lm_head_weight, "
            "Tensor fa_k_cache, Tensor fa_v_cache, Tensor dn_states, Tensor conv_bufs, "
            "Tensor hidden, Tensor residual, Tensor normalized, "
            "Tensor proj_buf, Tensor proj_buf2, Tensor attn_buf, Tensor mlp_buf, "
            "Tensor dn_out_buf, Tensor beta_buf, Tensor alpha_buf, "
            "Tensor final_normed, Tensor hidden_bf16_out, "
            "Tensor lm_bmv, Tensor lm_bmi, "
            "Tensor fa_q_A, Tensor fa_q_B, "
            "Tensor fa_k_A, Tensor fa_k_B, "
            "Tensor fa_v_A, Tensor fa_v_B, "
            "Tensor fa_o_A, Tensor fa_o_B, "
            "Tensor fa_gate_A, Tensor fa_gate_B, "
            "Tensor fa_up_A, Tensor fa_up_B, "
            "Tensor fa_down_A, Tensor fa_down_B, "
            "Tensor dn_qkv_A, Tensor dn_qkv_B, "
            "Tensor dn_z_A, Tensor dn_z_B, "
            "Tensor dn_out_A, Tensor dn_out_B, "
            "Tensor dn_gate_A, Tensor dn_gate_B, "
            "Tensor dn_up_A, Tensor dn_up_B, "
            "Tensor dn_down_A, Tensor dn_down_B, "
            "int lora_rank, float lora_scaling, Tensor lora_h_ws) -> ()");
    ops.impl("prefill_bf16_with_lora", torch::kCUDA, &prefill_bf16_with_lora);

    ops.def("prefill_bf16_train_step(Tensor output_token, Tensor token_ids, "
            "Tensor embed_weight, Tensor layer_weights_packed, "
            "Tensor final_norm_weight, Tensor lm_head_weight, "
            "Tensor fa_k_cache, Tensor fa_v_cache, Tensor dn_states, Tensor conv_bufs, "
            "Tensor hidden, Tensor residual, Tensor normalized, "
            "Tensor proj_buf, Tensor proj_buf2, Tensor attn_buf, Tensor mlp_buf, "
            "Tensor dn_out_buf, Tensor beta_buf, Tensor alpha_buf, "
            "Tensor final_normed, Tensor hidden_bf16_out, "
            "Tensor lm_bmv, Tensor lm_bmi, "
            "Tensor fa_q_A, Tensor fa_q_B, "
            "Tensor fa_k_A, Tensor fa_k_B, "
            "Tensor fa_v_A, Tensor fa_v_B, "
            "Tensor fa_o_A, Tensor fa_o_B, "
            "Tensor fa_gate_A, Tensor fa_gate_B, "
            "Tensor fa_up_A, Tensor fa_up_B, "
            "Tensor fa_down_A, Tensor fa_down_B, "
            "Tensor dn_qkv_A, Tensor dn_qkv_B, "
            "Tensor dn_z_A, Tensor dn_z_B, "
            "Tensor dn_out_A, Tensor dn_out_B, "
            "Tensor dn_gate_A, Tensor dn_gate_B, "
            "Tensor dn_up_A, Tensor dn_up_B, "
            "Tensor dn_down_A, Tensor dn_down_B, "
            "int lora_rank, float lora_scaling, Tensor lora_h_ws, "
            "Tensor hidden_in_save, Tensor normalized_in_save, "
            "Tensor normalized_post_attn_save, Tensor mlp_inter_save, "
            "Tensor attn_out_pre_o_save, Tensor h_post_attn_save) -> ()");
    ops.impl("prefill_bf16_train_step", torch::kCUDA, &prefill_bf16_train_step);

    ops.def("prefill_bf16_mega(Tensor output_token, Tensor token_ids, "
            "Tensor embed_weight, Tensor layer_weights_packed, "
            "Tensor final_norm_weight, Tensor lm_head_weight, "
            "Tensor fa_k_cache, Tensor fa_v_cache, Tensor dn_states, Tensor conv_bufs, "
            "Tensor hidden, Tensor residual, Tensor normalized, "
            "Tensor proj_buf, Tensor proj_buf2, Tensor attn_buf, Tensor mlp_buf, "
            "Tensor dn_out_buf, Tensor beta_buf, Tensor alpha_buf, "
            "Tensor final_normed, Tensor hidden_bf16_out, "
            "Tensor lm_bmv, Tensor lm_bmi) -> ()");
    ops.impl("prefill_bf16_mega", torch::kCUDA, &prefill_bf16_mega);

    ops.def("quantize_nvfp4_out(Tensor packed_out, Tensor scales_out, Tensor weight, int group_size) -> ()");
    ops.impl("quantize_nvfp4_out", torch::kCUDA, &quantize_nvfp4_out);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
