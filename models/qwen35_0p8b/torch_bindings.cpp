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

#ifdef MEGAKERNEL_HAS_NVFP4
struct LayerWeightsNVFP4 {
    int layer_type;
    int group_size;
    int _pad[2];
    void *ptrs[24];  // hot decode weights become packed fp4 + per-group scales
};
#endif

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

#ifdef MEGAKERNEL_HAS_NVFP4
// New signature: lm_hidden_* + lm_logits_f16 buffers added between
// lm_head_scales and fa_k_cache. input_token is read from a device int
// (`output_token_id` is reused as input on the device side).
extern "C" void launch_decode_nvfp4(
    const int *input_token_ptr, int *output_token_id,
    const void *embed_weight, const LayerWeightsNVFP4 *layer_weights,
    const void *final_norm_weight,
    const void *lm_head_weight_packed, const void *lm_head_scales,
    void *lm_hidden_bf16, void *lm_hidden_packed, void *lm_hidden_scales, void *lm_logits_f16,
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

extern "C" void launch_decode_many_nvfp4(
    int *token_buffer, int *output_tokens, int steps,
    const void *embed_weight, const LayerWeightsNVFP4 *layer_weights,
    const void *final_norm_weight,
    const void *lm_head_weight_packed, const void *lm_head_scales,
    void *lm_hidden_bf16, void *lm_hidden_packed, void *lm_hidden_scales, void *lm_logits_f16,
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

extern "C" void launch_quantize_nvfp4_lm_out(
    const void *weight, int rows, int cols,
    void *packed_out, void *scales_out, cudaStream_t stream);

extern "C" void launch_lm_head_nvfp4_from_f32(
    const float *normalized,
    int *output_token_id,
    const void *lm_head_weight_packed, const void *lm_head_scales,
    void *lm_hidden_bf16, void *lm_hidden_packed, void *lm_hidden_scales, void *lm_logits_f16,
    float *block_max_vals, int *block_max_idxs,
    int group_size,
    cudaStream_t stream);

extern "C" void launch_prefill_megakernel_nvfp4(
    const int *token_ids, int seq_len, int *output_token_id,
    const void *embed_weight, const LayerWeightsNVFP4 *layer_weights,
    const void *final_norm_weight,
    const void *lm_head_weight_packed, const void *lm_head_scales,
    void *lm_hidden_bf16, void *lm_hidden_packed, void *lm_hidden_scales, void *lm_logits_f16,
    void *fa_k_cache, void *fa_v_cache,
    void *dn_states, void *conv_bufs,
    void *hidden_buffer, void *g_activations, void *g_residual,
    void *g_qkv_scratch, void *g_kv_scratch, void *g_attn_out,
    void *g_mlp_inter, void *g_z_scratch, void *g_beta_scratch,
    void *g_alpha_scratch, void *g_normalized,
    unsigned int *barrier_counter, unsigned int *barrier_generation,
    float *block_max_vals, int *block_max_idxs,
    unsigned int *lm_sync_counter,
    int max_seq_len, int group_size, cudaStream_t stream);

static void seed_token_buffer(torch::Tensor token_buffer, int token_id) {
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaError_t err = cudaMemcpyAsync(
        token_buffer.data_ptr(),
        &token_id,
        sizeof(token_id),
        cudaMemcpyHostToDevice,
        stream);
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync(token_buffer) failed: ",
                cudaGetErrorString(err));
}

// ===== NVFP4 KV cache (sm_121a). See nvfp4_kv.cuh for the format spec. =====
extern "C" void launch_kv_quant(
    const void *src_bf16, void *data, void *scales,
    int T, int H, cudaStream_t stream);
extern "C" void launch_kv_dequant(
    const void *data, const void *scales, void *dst_bf16,
    int T, int H, cudaStream_t stream);
extern "C" void launch_kv_qk_dot(
    const void *q_bf16, const void *k_data, const void *k_scales, void *scores,
    int T, int H, int Q_H, cudaStream_t stream);
extern "C" void launch_kv_attention(
    const void *q_bf16,
    const void *k_data, const void *k_scales,
    const void *v_data, const void *v_scales,
    void *out_bf16, void *lse_out,
    int T, int H, int Q_H, float attn_scale, cudaStream_t stream);
extern "C" void launch_kv_attention_split(
    const void *q_bf16,
    const void *k_data, const void *k_scales,
    const void *v_data, const void *v_scales,
    void *out_bf16, void *lse_out, void *partials,
    int T, int H, int Q_H, float attn_scale, int num_splits,
    cudaStream_t stream);

static constexpr int KV_HEAD_DIM    = 256;
static constexpr int KV_GROUP_SIZE  = 16;
static constexpr int KV_DATA_BYTES  = KV_HEAD_DIM / 2;     // 128
static constexpr int KV_SCALE_BYTES = KV_HEAD_DIM / KV_GROUP_SIZE;  // 16

static inline void check_kv_shapes(
    const torch::Tensor &bf16, const torch::Tensor &data, const torch::Tensor &scales,
    const char *bf16_name)
{
    TORCH_CHECK(bf16.is_cuda() && bf16.is_contiguous(), bf16_name, " must be contiguous CUDA");
    TORCH_CHECK(data.is_cuda() && data.is_contiguous(), "data must be contiguous CUDA");
    TORCH_CHECK(scales.is_cuda() && scales.is_contiguous(), "scales must be contiguous CUDA");
    TORCH_CHECK(bf16.scalar_type() == torch::kBFloat16, bf16_name, " must be bfloat16");
    TORCH_CHECK(data.scalar_type() == torch::kUInt8,   "data must be uint8");
    TORCH_CHECK(scales.scalar_type() == torch::kUInt8, "scales must be uint8 (E4M3 raw bytes)");
    TORCH_CHECK(bf16.dim() == 3,   bf16_name, " shape must be [T, H, ", KV_HEAD_DIM, "]");
    TORCH_CHECK(data.dim() == 3,   "data shape must be [T, H, ", KV_DATA_BYTES, "]");
    TORCH_CHECK(scales.dim() == 3, "scales shape must be [T, H, ", KV_SCALE_BYTES, "]");
    TORCH_CHECK(bf16.size(2) == KV_HEAD_DIM,     bf16_name, " head_dim must be ", KV_HEAD_DIM);
    TORCH_CHECK(data.size(2) == KV_DATA_BYTES,   "data last-dim must be ", KV_DATA_BYTES);
    TORCH_CHECK(scales.size(2) == KV_SCALE_BYTES, "scales last-dim must be ", KV_SCALE_BYTES);
    TORCH_CHECK(bf16.size(0) == data.size(0) && bf16.size(0) == scales.size(0),
                "T mismatch across bf16/data/scales");
    TORCH_CHECK(bf16.size(1) == data.size(1) && bf16.size(1) == scales.size(1),
                "H mismatch across bf16/data/scales");
}

void quantize_bf16_to_nvfp4_kv(torch::Tensor src, torch::Tensor data, torch::Tensor scales) {
    check_kv_shapes(src, data, scales, "src");
    int T = (int)src.size(0);
    int H = (int)src.size(1);
    launch_kv_quant(src.data_ptr(), data.data_ptr(), scales.data_ptr(),
                    T, H, c10::cuda::getCurrentCUDAStream().stream());
}

void dequantize_nvfp4_kv_to_bf16(torch::Tensor data, torch::Tensor scales, torch::Tensor dst) {
    check_kv_shapes(dst, data, scales, "dst");
    int T = (int)dst.size(0);
    int H = (int)dst.size(1);
    launch_kv_dequant(data.data_ptr(), scales.data_ptr(), dst.data_ptr(),
                      T, H, c10::cuda::getCurrentCUDAStream().stream());
}

void qk_dot_nvfp4(torch::Tensor q, torch::Tensor k_data, torch::Tensor k_scales,
                  torch::Tensor scores) {
    TORCH_CHECK(q.is_cuda() && q.is_contiguous() && q.scalar_type() == torch::kBFloat16,
                "q must be contiguous CUDA bf16");
    TORCH_CHECK(q.dim() == 2 && q.size(1) == KV_HEAD_DIM,
                "q shape must be [Q_H, ", KV_HEAD_DIM, "]");
    TORCH_CHECK(k_data.is_cuda() && k_data.is_contiguous() && k_data.scalar_type() == torch::kUInt8,
                "k_data must be contiguous CUDA uint8");
    TORCH_CHECK(k_scales.is_cuda() && k_scales.is_contiguous() && k_scales.scalar_type() == torch::kUInt8,
                "k_scales must be contiguous CUDA uint8");
    TORCH_CHECK(k_data.dim() == 3 && k_data.size(2) == KV_DATA_BYTES,
                "k_data shape must be [T, H, ", KV_DATA_BYTES, "]");
    TORCH_CHECK(k_scales.dim() == 3 && k_scales.size(2) == KV_SCALE_BYTES,
                "k_scales shape must be [T, H, ", KV_SCALE_BYTES, "]");
    TORCH_CHECK(scores.is_cuda() && scores.is_contiguous() && scores.scalar_type() == torch::kFloat,
                "scores must be contiguous CUDA float32");
    int T   = (int)k_data.size(0);
    int H   = (int)k_data.size(1);
    int Q_H = (int)q.size(0);
    TORCH_CHECK((Q_H % H) == 0, "Q_H (", Q_H, ") must be a multiple of H (", H, ") for GQA");
    TORCH_CHECK(scores.dim() == 2 && scores.size(0) == Q_H && scores.size(1) == T,
                "scores shape must be [Q_H=", Q_H, ", T=", T, "]");
    launch_kv_qk_dot(q.data_ptr(), k_data.data_ptr(), k_scales.data_ptr(), scores.data_ptr(),
                     T, H, Q_H, c10::cuda::getCurrentCUDAStream().stream());
}

void kv_attention_nvfp4(
    torch::Tensor q,
    torch::Tensor k_data, torch::Tensor k_scales,
    torch::Tensor v_data, torch::Tensor v_scales,
    torch::Tensor out,
    c10::optional<torch::Tensor> lse_out,
    double attn_scale)
{
    TORCH_CHECK(q.is_cuda() && q.is_contiguous() && q.scalar_type() == torch::kBFloat16,
                "q must be contiguous CUDA bf16");
    TORCH_CHECK(q.dim() == 2 && q.size(1) == KV_HEAD_DIM,
                "q shape must be [Q_H, ", KV_HEAD_DIM, "]");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.scalar_type() == torch::kBFloat16,
                "out must be contiguous CUDA bf16");
    TORCH_CHECK(out.sizes() == q.sizes(), "out shape must match q");
    TORCH_CHECK(k_data.is_cuda() && k_data.is_contiguous() && k_data.scalar_type() == torch::kUInt8,
                "k_data must be contiguous CUDA uint8");
    TORCH_CHECK(k_scales.is_cuda() && k_scales.is_contiguous() && k_scales.scalar_type() == torch::kUInt8,
                "k_scales must be contiguous CUDA uint8");
    TORCH_CHECK(v_data.is_cuda() && v_data.is_contiguous() && v_data.scalar_type() == torch::kUInt8,
                "v_data must be contiguous CUDA uint8");
    TORCH_CHECK(v_scales.is_cuda() && v_scales.is_contiguous() && v_scales.scalar_type() == torch::kUInt8,
                "v_scales must be contiguous CUDA uint8");
    TORCH_CHECK(k_data.dim() == 3 && k_data.size(2) == KV_DATA_BYTES,
                "k_data last-dim must be ", KV_DATA_BYTES);
    TORCH_CHECK(k_scales.dim() == 3 && k_scales.size(2) == KV_SCALE_BYTES,
                "k_scales last-dim must be ", KV_SCALE_BYTES);
    TORCH_CHECK(k_data.sizes() == v_data.sizes(), "K/V data shapes mismatch");
    TORCH_CHECK(k_scales.sizes() == v_scales.sizes(), "K/V scale shapes mismatch");
    int T   = (int)k_data.size(0);
    int H   = (int)k_data.size(1);
    int Q_H = (int)q.size(0);
    TORCH_CHECK((Q_H % H) == 0, "Q_H (", Q_H, ") must be a multiple of H (", H, ") for GQA");
    void *lse_ptr = nullptr;
    if (lse_out.has_value()) {
        const auto &t = *lse_out;
        TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == torch::kFloat
                    && t.dim() == 1 && t.size(0) == Q_H,
                    "lse_out must be a contiguous CUDA float32 [Q_H] tensor");
        lse_ptr = t.data_ptr();
    }
    launch_kv_attention(
        q.data_ptr(), k_data.data_ptr(), k_scales.data_ptr(),
        v_data.data_ptr(), v_scales.data_ptr(),
        out.data_ptr(), lse_ptr,
        T, H, Q_H, (float)attn_scale,
        c10::cuda::getCurrentCUDAStream().stream());
}

void kv_attention_split_nvfp4(
    torch::Tensor q,
    torch::Tensor k_data, torch::Tensor k_scales,
    torch::Tensor v_data, torch::Tensor v_scales,
    torch::Tensor out,
    torch::Tensor partials,
    c10::optional<torch::Tensor> lse_out,
    double attn_scale,
    int64_t num_splits)
{
    TORCH_CHECK(q.is_cuda() && q.is_contiguous() && q.scalar_type() == torch::kBFloat16,
                "q must be contiguous CUDA bf16");
    TORCH_CHECK(q.dim() == 2 && q.size(1) == KV_HEAD_DIM,
                "q shape must be [Q_H, ", KV_HEAD_DIM, "]");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.scalar_type() == torch::kBFloat16,
                "out must be contiguous CUDA bf16");
    TORCH_CHECK(out.sizes() == q.sizes(), "out shape must match q");
    TORCH_CHECK(k_data.dim() == 3 && k_data.size(2) == KV_DATA_BYTES,
                "k_data last-dim must be ", KV_DATA_BYTES);
    TORCH_CHECK(k_data.sizes() == v_data.sizes(), "K/V data shapes mismatch");
    TORCH_CHECK(k_scales.sizes() == v_scales.sizes(), "K/V scale shapes mismatch");
    TORCH_CHECK(num_splits > 0, "num_splits must be > 0");
    int T   = (int)k_data.size(0);
    int H   = (int)k_data.size(1);
    int Q_H = (int)q.size(0);
    TORCH_CHECK((Q_H % H) == 0, "Q_H must be a multiple of H");
    TORCH_CHECK(partials.is_cuda() && partials.is_contiguous()
                && partials.scalar_type() == torch::kFloat
                && partials.numel() >= (int64_t)Q_H * num_splits * (KV_HEAD_DIM + 2),
                "partials must be a contiguous CUDA float tensor of size >= Q_H*num_splits*(head+2)");
    void *lse_ptr = nullptr;
    if (lse_out.has_value()) {
        const auto &t = *lse_out;
        TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == torch::kFloat
                    && t.dim() == 1 && t.size(0) == Q_H,
                    "lse_out must be contiguous CUDA float32 [Q_H]");
        lse_ptr = t.data_ptr();
    }
    launch_kv_attention_split(
        q.data_ptr(), k_data.data_ptr(), k_scales.data_ptr(),
        v_data.data_ptr(), v_scales.data_ptr(),
        out.data_ptr(), lse_ptr, partials.data_ptr(),
        T, H, Q_H, (float)attn_scale, (int)num_splits,
        c10::cuda::getCurrentCUDAStream().stream());
}
#endif  // MEGAKERNEL_HAS_NVFP4

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

#ifdef MEGAKERNEL_HAS_NVFP4
void decode_nvfp4(
    torch::Tensor output_token, int64_t input_token_id,
    torch::Tensor embed_weight, torch::Tensor layer_weights_packed,
    torch::Tensor final_norm_weight,
    torch::Tensor lm_head_weight_packed, torch::Tensor lm_head_scales,
    torch::Tensor lm_hidden_bf16, torch::Tensor lm_hidden_packed,
    torch::Tensor lm_hidden_scales, torch::Tensor lm_logits_f16,
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
    seed_token_buffer(output_token, (int)input_token_id);
    launch_decode_nvfp4(
        (const int*)output_token.data_ptr(),
        (int*)output_token.data_ptr(),
        embed_weight.data_ptr(),
        reinterpret_cast<const LayerWeightsNVFP4*>(layer_weights_packed.data_ptr()),
        final_norm_weight.data_ptr(),
        lm_head_weight_packed.data_ptr(), lm_head_scales.data_ptr(),
        lm_hidden_bf16.data_ptr(), lm_hidden_packed.data_ptr(),
        lm_hidden_scales.data_ptr(), lm_logits_f16.data_ptr(),
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

void decode_many_nvfp4(
    torch::Tensor output_tokens,
    torch::Tensor token_buffer,
    int64_t input_token_id,
    torch::Tensor embed_weight, torch::Tensor layer_weights_packed,
    torch::Tensor final_norm_weight,
    torch::Tensor lm_head_weight_packed, torch::Tensor lm_head_scales,
    torch::Tensor lm_hidden_bf16, torch::Tensor lm_hidden_packed,
    torch::Tensor lm_hidden_scales, torch::Tensor lm_logits_f16,
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
    TORCH_CHECK(output_tokens.is_cuda(), "output_tokens must be CUDA");
    TORCH_CHECK(output_tokens.is_contiguous(), "output_tokens must be contiguous");
    TORCH_CHECK(output_tokens.scalar_type() == torch::kInt32, "output_tokens must be int32");
    TORCH_CHECK(output_tokens.dim() == 1, "output_tokens must be 1D");
    TORCH_CHECK(token_buffer.is_cuda(), "token_buffer must be CUDA");
    TORCH_CHECK(token_buffer.is_contiguous(), "token_buffer must be contiguous");
    TORCH_CHECK(token_buffer.scalar_type() == torch::kInt32, "token_buffer must be int32");
    TORCH_CHECK(token_buffer.numel() == 1, "token_buffer must contain exactly one int32 token");

    seed_token_buffer(token_buffer, (int)input_token_id);
    launch_decode_many_nvfp4(
        (int*)token_buffer.data_ptr(),
        (int*)output_tokens.data_ptr(),
        (int)output_tokens.numel(),
        embed_weight.data_ptr(),
        reinterpret_cast<const LayerWeightsNVFP4*>(layer_weights_packed.data_ptr()),
        final_norm_weight.data_ptr(),
        lm_head_weight_packed.data_ptr(), lm_head_scales.data_ptr(),
        lm_hidden_bf16.data_ptr(), lm_hidden_packed.data_ptr(),
        lm_hidden_scales.data_ptr(), lm_logits_f16.data_ptr(),
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

void prefill_megakernel_nvfp4(
    torch::Tensor output_token,
    torch::Tensor token_ids,
    torch::Tensor embed_weight, torch::Tensor layer_weights_packed,
    torch::Tensor final_norm_weight,
    torch::Tensor lm_head_weight_packed, torch::Tensor lm_head_scales,
    torch::Tensor lm_hidden_bf16, torch::Tensor lm_hidden_packed,
    torch::Tensor lm_hidden_scales, torch::Tensor lm_logits_f16,
    torch::Tensor fa_k_cache, torch::Tensor fa_v_cache,
    torch::Tensor dn_states, torch::Tensor conv_bufs,
    torch::Tensor hidden_buffer, torch::Tensor activations, torch::Tensor residual,
    torch::Tensor qkv_scratch, torch::Tensor kv_scratch, torch::Tensor attn_out,
    torch::Tensor mlp_inter, torch::Tensor z_scratch, torch::Tensor beta_scratch,
    torch::Tensor alpha_scratch, torch::Tensor normalized,
    torch::Tensor barrier_counter, torch::Tensor barrier_generation,
    torch::Tensor block_max_vals, torch::Tensor block_max_idxs,
    torch::Tensor lm_sync_counter, int64_t max_seq_len, int64_t group_size)
{
    TORCH_CHECK(token_ids.is_cuda() && token_ids.is_contiguous(),
                "token_ids must be contiguous CUDA");
    TORCH_CHECK(token_ids.scalar_type() == torch::kInt32, "token_ids must be int32");
    int seq_len = (int)token_ids.numel();
    launch_prefill_megakernel_nvfp4(
        (const int*)token_ids.data_ptr(), seq_len,
        (int*)output_token.data_ptr(),
        embed_weight.data_ptr(),
        reinterpret_cast<const LayerWeightsNVFP4*>(layer_weights_packed.data_ptr()),
        final_norm_weight.data_ptr(),
        lm_head_weight_packed.data_ptr(), lm_head_scales.data_ptr(),
        lm_hidden_bf16.data_ptr(), lm_hidden_packed.data_ptr(),
        lm_hidden_scales.data_ptr(), lm_logits_f16.data_ptr(),
        fa_k_cache.data_ptr(), fa_v_cache.data_ptr(),
        dn_states.data_ptr(), conv_bufs.data_ptr(),
        hidden_buffer.data_ptr(), activations.data_ptr(), residual.data_ptr(),
        qkv_scratch.data_ptr(), kv_scratch.data_ptr(), attn_out.data_ptr(),
        mlp_inter.data_ptr(), z_scratch.data_ptr(), beta_scratch.data_ptr(),
        alpha_scratch.data_ptr(), normalized.data_ptr(),
        (unsigned int*)barrier_counter.data_ptr(), (unsigned int*)barrier_generation.data_ptr(),
        (float*)block_max_vals.data_ptr(), (int*)block_max_idxs.data_ptr(),
        (unsigned int*)lm_sync_counter.data_ptr(),
        (int)max_seq_len, (int)group_size,
        c10::cuda::getCurrentCUDAStream().stream());
}

void lm_head_nvfp4_from_f32(
    torch::Tensor output_token,
    torch::Tensor normalized,
    torch::Tensor lm_head_weight_packed,
    torch::Tensor lm_head_scales,
    torch::Tensor lm_hidden_bf16,
    torch::Tensor lm_hidden_packed,
    torch::Tensor lm_hidden_scales,
    torch::Tensor lm_logits_f16,
    torch::Tensor block_max_vals,
    torch::Tensor block_max_idxs,
    int64_t group_size)
{
    TORCH_CHECK(normalized.is_cuda() && normalized.is_contiguous(),
                "normalized must be contiguous CUDA");
    TORCH_CHECK(normalized.scalar_type() == torch::kFloat32,
                "normalized must be f32");
    launch_lm_head_nvfp4_from_f32(
        (const float*)normalized.data_ptr(),
        (int*)output_token.data_ptr(),
        lm_head_weight_packed.data_ptr(), lm_head_scales.data_ptr(),
        lm_hidden_bf16.data_ptr(), lm_hidden_packed.data_ptr(),
        lm_hidden_scales.data_ptr(), lm_logits_f16.data_ptr(),
        (float*)block_max_vals.data_ptr(), (int*)block_max_idxs.data_ptr(),
        (int)group_size,
        c10::cuda::getCurrentCUDAStream().stream());
}

void quantize_nvfp4_lm_out(
    torch::Tensor packed_out,
    torch::Tensor scales_out,
    torch::Tensor weight)
{
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous(), "weight must be contiguous CUDA");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
    TORCH_CHECK(weight.scalar_type() == torch::kBFloat16, "weight must be bfloat16");
    TORCH_CHECK(packed_out.is_cuda() && packed_out.is_contiguous(), "packed_out must be contiguous CUDA");
    TORCH_CHECK(scales_out.is_cuda() && scales_out.is_contiguous(), "scales_out must be contiguous CUDA");
    TORCH_CHECK(packed_out.scalar_type() == torch::kUInt8, "packed_out must be uint8");
    TORCH_CHECK(scales_out.scalar_type() == torch::kUInt8, "scales_out must be uint8 (UE4M3)");
    auto rows = static_cast<int>(weight.size(0));
    auto cols = static_cast<int>(weight.size(1));
    launch_quantize_nvfp4_lm_out(
        weight.data_ptr(), rows, cols,
        packed_out.data_ptr(), scales_out.data_ptr(),
        c10::cuda::getCurrentCUDAStream().stream());
}
#endif  // MEGAKERNEL_HAS_NVFP4

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
    // Slice B.3b kernel-bwd saves (FA-only): Q post-RoPE/QKnorm, O
    // post-FA pre-gate, LSE log-sum-exp from cuDNN. Shapes:
    //   fa_q_save   [N_FA, S, FA_Q_HEADS, FA_HEAD_DIM] bf16
    //   fa_o_save   same
    //   fa_lse_save [N_FA, FA_Q_HEADS, S]              fp32
    void *fa_q_save;
    void *fa_o_save;
    void *fa_lse_save;
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

// 3090-tuned chunked DN forward (V_SPLITS=4, C=32). Used for inference S>=
// some threshold; matches HF torch_chunk_gated_delta_rule output within
// bf16 noise. See dn_chunked_3090.cu for the kernel itself.
extern "C" void launch_dn_chunked_3090(
    const void *q_base, const void *k_base, const void *v_base,
    const float *beta_base, const float *g_base,
    const float *state_in_base,
    void *y_base, float *state_out_base,
    int S, int H,
    int qkd_pos_stride, int v_pos_stride, int bd_pos_stride, int y_pos_stride,
    cudaStream_t stream);

extern "C" void launch_pf_decay_to_g_inplace(float *buf, int N, cudaStream_t stream);

void dn_chunked_3090(
    torch::Tensor q,           // [S, H, Dk] or strided [S, *] bf16
    torch::Tensor k,           // [S, H, Dk] or strided [S, *] bf16
    torch::Tensor v,           // [S, H, Dv] or strided [S, *] bf16
    torch::Tensor beta,        // [S, H] fp32 (sigmoid already applied)
    torch::Tensor g,           // [S, H] fp32 = log(decay)
    torch::Tensor state_in,    // [H, Dk, Dv] fp32
    torch::Tensor y,           // [S, H, Dv] bf16 (output)
    torch::Tensor state_out)   // [H, Dk, Dv] fp32 (output, may be empty)
{
    int64_t S = q.size(0);
    int64_t H = (q.dim() == 3) ? q.size(1) : (q.size(1) / 128);
    int64_t Dk = 128;
    int64_t Dv = 128;
    int qkd_stride = (int)(q.stride(0));      // tokens-to-tokens stride (in elements)
    int v_stride   = (int)(v.stride(0));
    int bd_stride  = (int)(beta.stride(0));
    int y_stride   = (int)(y.stride(0));
    float *state_out_ptr = state_out.numel() > 0 ? state_out.data_ptr<float>() : nullptr;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.get_device());
    launch_dn_chunked_3090(
        q.data_ptr(), k.data_ptr(), v.data_ptr(),
        beta.data_ptr<float>(), g.data_ptr<float>(),
        state_in.data_ptr<float>(),
        y.data_ptr(), state_out_ptr,
        (int)S, (int)H,
        qkd_stride, v_stride, bd_stride, y_stride,
        stream);
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
    torch::Tensor h_post_attn_save,
    // Slice B.3b kernel-bwd saves (FA only) — pass empty tensors to skip.
    torch::Tensor fa_q_save,
    torch::Tensor fa_o_save,
    torch::Tensor fa_lse_save)
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
        const_cast<void*>(opt_ptr(fa_q_save)),
        const_cast<void*>(opt_ptr(fa_o_save)),
        const_cast<void*>(opt_ptr(fa_lse_save)),
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

#ifdef MEGAKERNEL_HAS_NVFP4
    ops.def("decode_nvfp4(Tensor output_token, int input_token_id, "
            "Tensor embed_weight, Tensor layer_weights_packed, "
            "Tensor final_norm_weight, Tensor lm_head_weight_packed, Tensor lm_head_scales, "
            "Tensor lm_hidden_bf16, Tensor lm_hidden_packed, Tensor lm_hidden_scales, Tensor lm_logits_f16, "
            "Tensor fa_k_cache, Tensor fa_v_cache, Tensor dn_states, Tensor conv_bufs, "
            "Tensor hidden_buffer, Tensor activations, Tensor residual, "
            "Tensor qkv_scratch, Tensor kv_scratch, Tensor attn_out, "
            "Tensor mlp_inter, Tensor z_scratch, Tensor beta_scratch, "
            "Tensor alpha_scratch, Tensor normalized, "
            "Tensor barrier_counter, Tensor barrier_generation, "
            "Tensor block_max_vals, Tensor block_max_idxs, Tensor lm_sync_counter, "
            "int position, int max_seq_len, int group_size) -> ()");
    ops.impl("decode_nvfp4", torch::kCUDA, &decode_nvfp4);

    ops.def("decode_many_nvfp4(Tensor output_tokens, Tensor token_buffer, int input_token_id, "
            "Tensor embed_weight, Tensor layer_weights_packed, "
            "Tensor final_norm_weight, Tensor lm_head_weight_packed, Tensor lm_head_scales, "
            "Tensor lm_hidden_bf16, Tensor lm_hidden_packed, Tensor lm_hidden_scales, Tensor lm_logits_f16, "
            "Tensor fa_k_cache, Tensor fa_v_cache, Tensor dn_states, Tensor conv_bufs, "
            "Tensor hidden_buffer, Tensor activations, Tensor residual, "
            "Tensor qkv_scratch, Tensor kv_scratch, Tensor attn_out, "
            "Tensor mlp_inter, Tensor z_scratch, Tensor beta_scratch, "
            "Tensor alpha_scratch, Tensor normalized, "
            "Tensor barrier_counter, Tensor barrier_generation, "
            "Tensor block_max_vals, Tensor block_max_idxs, Tensor lm_sync_counter, "
            "int position, int max_seq_len, int group_size) -> ()");
    ops.impl("decode_many_nvfp4", torch::kCUDA, &decode_many_nvfp4);

    ops.def("quantize_nvfp4_lm_out(Tensor packed_out, Tensor scales_out, Tensor weight) -> ()");
    ops.impl("quantize_nvfp4_lm_out", torch::kCUDA, &quantize_nvfp4_lm_out);

    ops.def("lm_head_nvfp4_from_f32(Tensor output_token, Tensor normalized, "
            "Tensor lm_head_weight_packed, Tensor lm_head_scales, "
            "Tensor lm_hidden_bf16, Tensor lm_hidden_packed, "
            "Tensor lm_hidden_scales, Tensor lm_logits_f16, "
            "Tensor block_max_vals, Tensor block_max_idxs, "
            "int group_size) -> ()");
    ops.impl("lm_head_nvfp4_from_f32", torch::kCUDA, &lm_head_nvfp4_from_f32);

    ops.def("quantize_bf16_to_nvfp4_kv(Tensor src, Tensor(a!) data, Tensor(b!) scales) -> ()");
    ops.impl("quantize_bf16_to_nvfp4_kv", torch::kCUDA, &quantize_bf16_to_nvfp4_kv);

    ops.def("dequantize_nvfp4_kv_to_bf16(Tensor data, Tensor scales, Tensor(a!) dst) -> ()");
    ops.impl("dequantize_nvfp4_kv_to_bf16", torch::kCUDA, &dequantize_nvfp4_kv_to_bf16);

    ops.def("qk_dot_nvfp4(Tensor q, Tensor k_data, Tensor k_scales, Tensor(a!) scores) -> ()");
    ops.impl("qk_dot_nvfp4", torch::kCUDA, &qk_dot_nvfp4);

    ops.def("kv_attention_nvfp4(Tensor q, Tensor k_data, Tensor k_scales, "
            "Tensor v_data, Tensor v_scales, Tensor(a!) out, Tensor(b!)? lse_out, "
            "float attn_scale) -> ()");
    ops.impl("kv_attention_nvfp4", torch::kCUDA, &kv_attention_nvfp4);

    ops.def("kv_attention_split_nvfp4(Tensor q, Tensor k_data, Tensor k_scales, "
            "Tensor v_data, Tensor v_scales, Tensor(a!) out, Tensor(b!) partials, "
            "Tensor(c!)? lse_out, float attn_scale, int num_splits) -> ()");
    ops.impl("kv_attention_split_nvfp4", torch::kCUDA, &kv_attention_split_nvfp4);

    ops.def("prefill_megakernel_nvfp4(Tensor output_token, Tensor token_ids, "
            "Tensor embed_weight, Tensor layer_weights_packed, "
            "Tensor final_norm_weight, Tensor lm_head_weight_packed, Tensor lm_head_scales, "
            "Tensor lm_hidden_bf16, Tensor lm_hidden_packed, Tensor lm_hidden_scales, Tensor lm_logits_f16, "
            "Tensor fa_k_cache, Tensor fa_v_cache, Tensor dn_states, Tensor conv_bufs, "
            "Tensor hidden_buffer, Tensor activations, Tensor residual, "
            "Tensor qkv_scratch, Tensor kv_scratch, Tensor attn_out, "
            "Tensor mlp_inter, Tensor z_scratch, Tensor beta_scratch, "
            "Tensor alpha_scratch, Tensor normalized, "
            "Tensor barrier_counter, Tensor barrier_generation, "
            "Tensor block_max_vals, Tensor block_max_idxs, Tensor lm_sync_counter, "
            "int max_seq_len, int group_size) -> ()");
    ops.impl("prefill_megakernel_nvfp4", torch::kCUDA, &prefill_megakernel_nvfp4);
#endif  // MEGAKERNEL_HAS_NVFP4

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
            "Tensor attn_out_pre_o_save, Tensor h_post_attn_save, "
            "Tensor fa_q_save, Tensor fa_o_save, Tensor fa_lse_save) -> ()");
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

    ops.def("dn_chunked_3090(Tensor q, Tensor k, Tensor v, "
            "Tensor beta, Tensor g, Tensor state_in, "
            "Tensor(a!) y, Tensor(b!) state_out) -> ()");
    ops.impl("dn_chunked_3090", torch::kCUDA, &dn_chunked_3090);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
