/**
 * Python bindings for the Qwen3.x templated decode kernel scaffold.
 *
 * Exposes two smoke-test ops, one per Cfg specialization, so the build
 * pipeline can validate that both compile + link + execute on GB10:
 *
 *   qwen3x_C.mlp_smoke_0p8b(input, gain, w_gate, w_up, w_down,
 *                            sh_norm, g_gate, g_up, sh_inter, out)
 *   qwen3x_C.mlp_smoke_27b (same signature, larger shapes)
 *
 * The full decode op `qwen3x_C.decode(cfg_id, ...)` will land once the
 * DN V/QK split and FA layer are implemented (see kernel_decode.cu
 * TODOs).
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

#define REGISTER_EXTENSION(NAME)                                               \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                     \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT,                 \
                                        STRINGIFY(NAME), nullptr, 0, nullptr}; \
    return PyModule_Create(&module);                                           \
  }

// Mirror of YarnParams from rope.cuh — host side.
struct YarnParamsHost {
    float scale_factor;
    float beta_fast;
    float beta_slow;
    int   original_ctx_len;
    bool  enabled;
    int   _pad[3];
};
static_assert(sizeof(YarnParamsHost) >= 16, "YarnParams ABI");

extern "C" cudaError_t launch_decode_0p8b(
    void*, void*, void*, void*, void*, void*, void*,
    void*, void*, void*, void*, void*, void*,
    void*, void*, void*, void*, void*, void*,
    YarnParamsHost,
    int, int, int, int, int, int, void*, cudaStream_t);
extern "C" cudaError_t launch_decode_27b(
    void*, void*, void*, void*, void*, void*, void*,
    void*, void*, void*, void*, void*, void*,
    void*, void*, void*, void*, void*, void*,
    YarnParamsHost,
    int, int, int, int, int, int, void*, cudaStream_t);

extern "C" cudaError_t launch_prefill_naive_0p8b(
    const int32_t*, int,
    void*, void*, void*, void*, void*, void*, void*,
    void*, void*, void*, void*, void*, void*,
    void*, void*, void*, void*, void*, void*,
    YarnParamsHost,
    int, int, cudaStream_t);
// NOTE: prefill launcher signature unchanged (no debug capture in prefill yet).
extern "C" cudaError_t launch_prefill_naive_27b(
    const int32_t*, int,
    void*, void*, void*, void*, void*, void*, void*,
    void*, void*, void*, void*, void*, void*,
    void*, void*, void*, void*, void*, void*,
    YarnParamsHost,
    int, int, cudaStream_t);

extern "C" void launch_mlp_smoke_0p8b(
    const void *input, const void *gain,
    const void *w_gate, const void *w_up, const void *w_down,
    void *sh_norm, void *g_gate, void *g_up, void *sh_inter, void *out,
    cudaStream_t stream);

extern "C" void launch_mlp_smoke_27b(
    const void *input, const void *gain,
    const void *w_gate, const void *w_up, const void *w_down,
    void *sh_norm, void *g_gate, void *g_up, void *sh_inter, void *out,
    cudaStream_t stream);

static void check_mlp_inputs(const torch::Tensor &input, const torch::Tensor &gain,
                              const torch::Tensor &w_gate, const torch::Tensor &w_up,
                              const torch::Tensor &w_down,
                              const torch::Tensor &sh_norm,
                              const torch::Tensor &g_gate, const torch::Tensor &g_up,
                              const torch::Tensor &sh_inter,
                              const torch::Tensor &out,
                              int hidden, int intermediate)
{
    TORCH_CHECK(input.is_cuda() && input.is_contiguous()
                && input.scalar_type() == torch::kBFloat16
                && input.numel() == hidden,
                "input must be contig CUDA bf16 [", hidden, "]");
    TORCH_CHECK(gain.numel() == hidden && gain.scalar_type() == torch::kBFloat16,
                "gain must be bf16 [", hidden, "]");
    TORCH_CHECK(w_gate.numel() == (int64_t)intermediate * hidden, "w_gate shape");
    TORCH_CHECK(w_up.numel()   == (int64_t)intermediate * hidden, "w_up shape");
    TORCH_CHECK(w_down.numel() == (int64_t)hidden * intermediate, "w_down shape");
    TORCH_CHECK(sh_norm.numel() == hidden && sh_norm.scalar_type() == torch::kBFloat16,
                "sh_norm must be bf16 [", hidden, "]");
    TORCH_CHECK(g_gate.numel() == intermediate && g_gate.scalar_type() == torch::kFloat,
                "g_gate must be fp32 [", intermediate, "]");
    TORCH_CHECK(g_up.numel()   == intermediate && g_up.scalar_type() == torch::kFloat,
                "g_up must be fp32 [", intermediate, "]");
    TORCH_CHECK(sh_inter.numel() == intermediate && sh_inter.scalar_type() == torch::kBFloat16,
                "sh_inter must be bf16 [", intermediate, "]");
    TORCH_CHECK(out.numel() == hidden && out.scalar_type() == torch::kBFloat16,
                "out must be bf16 [", hidden, "]");
}

void mlp_smoke_0p8b(
    torch::Tensor input, torch::Tensor gain,
    torch::Tensor w_gate, torch::Tensor w_up, torch::Tensor w_down,
    torch::Tensor sh_norm, torch::Tensor g_gate, torch::Tensor g_up,
    torch::Tensor sh_inter, torch::Tensor out)
{
    check_mlp_inputs(input, gain, w_gate, w_up, w_down, sh_norm, g_gate, g_up,
                     sh_inter, out, /*hidden=*/1024, /*intermediate=*/3584);
    launch_mlp_smoke_0p8b(
        input.data_ptr(), gain.data_ptr(),
        w_gate.data_ptr(), w_up.data_ptr(), w_down.data_ptr(),
        sh_norm.data_ptr(), g_gate.data_ptr(), g_up.data_ptr(),
        sh_inter.data_ptr(), out.data_ptr(),
        c10::cuda::getCurrentCUDAStream().stream());
}

void mlp_smoke_27b(
    torch::Tensor input, torch::Tensor gain,
    torch::Tensor w_gate, torch::Tensor w_up, torch::Tensor w_down,
    torch::Tensor sh_norm, torch::Tensor g_gate, torch::Tensor g_up,
    torch::Tensor sh_inter, torch::Tensor out)
{
    check_mlp_inputs(input, gain, w_gate, w_up, w_down, sh_norm, g_gate, g_up,
                     sh_inter, out, /*hidden=*/5120, /*intermediate=*/17408);
    launch_mlp_smoke_27b(
        input.data_ptr(), gain.data_ptr(),
        w_gate.data_ptr(), w_up.data_ptr(), w_down.data_ptr(),
        sh_norm.data_ptr(), g_gate.data_ptr(), g_up.data_ptr(),
        sh_inter.data_ptr(), out.data_ptr(),
        c10::cuda::getCurrentCUDAStream().stream());
}

// ---------------------------------------------------------------------------
// Full decode op: dispatches to Cfg_0p8B or Cfg_27B specialization based on
// `model_id`. Layer-weight blob is a uint8 device tensor that mirrors the
// LayerWeights<Cfg> array layout (see weight_packer.py:pack_layer_weights).
// ---------------------------------------------------------------------------

static int default_num_blocks_for(int cfg_id) {
    // Heuristic: enough blocks to run Q_HEADS * 8 splits on the FA layer.
    // 0.8B: 8 * 8 = 64; 27B: 24 * 8 = 192. GB10 has ~64 SMs so we can
    // afford 64 blocks of cooperative grid (cuLaunchCooperativeKernel
    // requires the entire grid to fit on the device at once).
    int dev = 0; cudaGetDevice(&dev);
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    int target = (cfg_id == 0) ? 64 : 96;   // tuned modestly for 0.8B / 27B
    return std::min(target, std::max(sm_count, 1));
}

void decode_qwen3x(
    int64_t model_id,
    torch::Tensor embed_weight,
    torch::Tensor final_norm_weight,
    torch::Tensor layer_weights,
    torch::Tensor fa_k_cache, torch::Tensor fa_v_cache,
    torch::Tensor dn_states, torch::Tensor conv_bufs,
    torch::Tensor hidden_buffer, torch::Tensor g_residual,
    torch::Tensor g_qkv_scratch, torch::Tensor g_kv_scratch,
    torch::Tensor g_attn_out, torch::Tensor g_mlp_inter,
    torch::Tensor g_z_scratch, torch::Tensor g_beta_scratch, torch::Tensor g_alpha_scratch,
    torch::Tensor g_normalized, torch::Tensor g_fa_partials, torch::Tensor g_rope_inv_freq,
    int64_t input_token_id, int64_t position,
    int64_t pos_h, int64_t pos_w, int64_t max_seq_len,
    double yarn_scale, double yarn_beta_fast, double yarn_beta_slow,
    int64_t yarn_orig_ctx, bool yarn_enabled,
    int64_t num_blocks,
    c10::optional<torch::Tensor> g_layer_outputs)
{
    TORCH_CHECK(model_id == 0 || model_id == 1,
                "model_id must be 0 (Cfg_0p8B) or 1 (Cfg_27B)");
    TORCH_CHECK(embed_weight.is_cuda() && layer_weights.is_cuda(),
                "weights must be CUDA");

    YarnParamsHost yp;
    yp.scale_factor     = (float)yarn_scale;
    yp.beta_fast        = (float)yarn_beta_fast;
    yp.beta_slow        = (float)yarn_beta_slow;
    yp.original_ctx_len = (int)yarn_orig_ctx;
    yp.enabled          = yarn_enabled;
    for (int i = 0; i < 3; ++i) yp._pad[i] = 0;

    int nb = (num_blocks > 0) ? (int)num_blocks : default_num_blocks_for((int)model_id);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    auto launcher = (model_id == 0) ? &launch_decode_0p8b : &launch_decode_27b;
    void *layer_outs_ptr = nullptr;
    if (g_layer_outputs.has_value()) {
        const auto &t = *g_layer_outputs;
        TORCH_CHECK(t.is_cuda() && t.is_contiguous()
                    && t.scalar_type() == torch::kBFloat16,
                    "g_layer_outputs must be contiguous CUDA bf16");
        layer_outs_ptr = t.data_ptr();
    }
    cudaError_t err = launcher(
        embed_weight.data_ptr(), final_norm_weight.data_ptr(),
        layer_weights.data_ptr(),
        fa_k_cache.data_ptr(), fa_v_cache.data_ptr(),
        dn_states.data_ptr(), conv_bufs.data_ptr(),
        hidden_buffer.data_ptr(), g_residual.data_ptr(),
        g_qkv_scratch.data_ptr(), g_kv_scratch.data_ptr(),
        g_attn_out.data_ptr(), g_mlp_inter.data_ptr(),
        g_z_scratch.data_ptr(), g_beta_scratch.data_ptr(), g_alpha_scratch.data_ptr(),
        g_normalized.data_ptr(), g_fa_partials.data_ptr(), g_rope_inv_freq.data_ptr(),
        yp,
        (int)input_token_id, (int)position, (int)pos_h, (int)pos_w, (int)max_seq_len,
        nb, layer_outs_ptr, stream);
    TORCH_CHECK(err == cudaSuccess,
                "decode_qwen3x launch failed: ", cudaGetErrorString(err));
}

void prefill_qwen3x_naive(
    int64_t model_id,
    torch::Tensor tokens,                                  // [S] int32
    torch::Tensor embed_weight,
    torch::Tensor final_norm_weight,
    torch::Tensor layer_weights,
    torch::Tensor fa_k_cache, torch::Tensor fa_v_cache,
    torch::Tensor dn_states, torch::Tensor conv_bufs,
    torch::Tensor hidden_buffer, torch::Tensor g_residual,
    torch::Tensor g_qkv_scratch, torch::Tensor g_kv_scratch,
    torch::Tensor g_attn_out, torch::Tensor g_mlp_inter,
    torch::Tensor g_z_scratch, torch::Tensor g_beta_scratch, torch::Tensor g_alpha_scratch,
    torch::Tensor g_normalized, torch::Tensor g_fa_partials, torch::Tensor g_rope_inv_freq,
    int64_t max_seq_len,
    double yarn_scale, double yarn_beta_fast, double yarn_beta_slow,
    int64_t yarn_orig_ctx, bool yarn_enabled,
    int64_t num_blocks)
{
    TORCH_CHECK(model_id == 0 || model_id == 1, "model_id must be 0 or 1");
    TORCH_CHECK(tokens.is_cuda() && tokens.is_contiguous()
                && tokens.scalar_type() == torch::kInt32,
                "tokens must be contiguous CUDA int32 [S]");
    TORCH_CHECK(tokens.dim() == 1, "tokens must be 1-D");
    int S = (int)tokens.size(0);
    YarnParamsHost yp;
    yp.scale_factor     = (float)yarn_scale;
    yp.beta_fast        = (float)yarn_beta_fast;
    yp.beta_slow        = (float)yarn_beta_slow;
    yp.original_ctx_len = (int)yarn_orig_ctx;
    yp.enabled          = yarn_enabled;
    for (int i = 0; i < 3; ++i) yp._pad[i] = 0;
    int nb = (num_blocks > 0) ? (int)num_blocks : default_num_blocks_for((int)model_id);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    auto launcher = (model_id == 0) ? &launch_prefill_naive_0p8b
                                    : &launch_prefill_naive_27b;
    cudaError_t err = launcher(
        (const int32_t*)tokens.data_ptr(), S,
        embed_weight.data_ptr(), final_norm_weight.data_ptr(),
        layer_weights.data_ptr(),
        fa_k_cache.data_ptr(), fa_v_cache.data_ptr(),
        dn_states.data_ptr(), conv_bufs.data_ptr(),
        hidden_buffer.data_ptr(), g_residual.data_ptr(),
        g_qkv_scratch.data_ptr(), g_kv_scratch.data_ptr(),
        g_attn_out.data_ptr(), g_mlp_inter.data_ptr(),
        g_z_scratch.data_ptr(), g_beta_scratch.data_ptr(), g_alpha_scratch.data_ptr(),
        g_normalized.data_ptr(), g_fa_partials.data_ptr(), g_rope_inv_freq.data_ptr(),
        yp, (int)max_seq_len, nb, stream);
    TORCH_CHECK(err == cudaSuccess,
                "prefill_qwen3x_naive launch failed: ", cudaGetErrorString(err));
}

TORCH_LIBRARY(qwen3x_C, ops) {
    ops.def("mlp_smoke_0p8b(Tensor input, Tensor gain, "
            "Tensor w_gate, Tensor w_up, Tensor w_down, "
            "Tensor(a!) sh_norm, Tensor(b!) g_gate, Tensor(c!) g_up, "
            "Tensor(d!) sh_inter, Tensor(e!) out) -> ()");
    ops.impl("mlp_smoke_0p8b", torch::kCUDA, &mlp_smoke_0p8b);

    ops.def("mlp_smoke_27b(Tensor input, Tensor gain, "
            "Tensor w_gate, Tensor w_up, Tensor w_down, "
            "Tensor(a!) sh_norm, Tensor(b!) g_gate, Tensor(c!) g_up, "
            "Tensor(d!) sh_inter, Tensor(e!) out) -> ()");
    ops.impl("mlp_smoke_27b", torch::kCUDA, &mlp_smoke_27b);

    ops.def("prefill_qwen3x_naive(int model_id, Tensor tokens, "
            "Tensor embed_weight, Tensor final_norm_weight, Tensor layer_weights, "
            "Tensor(a!) fa_k_cache, Tensor(b!) fa_v_cache, "
            "Tensor(c!) dn_states, Tensor(d!) conv_bufs, "
            "Tensor(e!) hidden_buffer, Tensor(f!) g_residual, "
            "Tensor(g!) g_qkv_scratch, Tensor(h!) g_kv_scratch, "
            "Tensor(i!) g_attn_out, Tensor(j!) g_mlp_inter, "
            "Tensor(k!) g_z_scratch, Tensor(l!) g_beta_scratch, Tensor(m!) g_alpha_scratch, "
            "Tensor(n!) g_normalized, Tensor(o!) g_fa_partials, Tensor(p!) g_rope_inv_freq, "
            "int max_seq_len, float yarn_scale, float yarn_beta_fast, "
            "float yarn_beta_slow, int yarn_orig_ctx, bool yarn_enabled, int num_blocks) -> ()");
    ops.impl("prefill_qwen3x_naive", torch::kCUDA, &prefill_qwen3x_naive);

    ops.def("decode_qwen3x(int model_id, "
            "Tensor embed_weight, Tensor final_norm_weight, Tensor layer_weights, "
            "Tensor(a!) fa_k_cache, Tensor(b!) fa_v_cache, "
            "Tensor(c!) dn_states, Tensor(d!) conv_bufs, "
            "Tensor(e!) hidden_buffer, Tensor(f!) g_residual, "
            "Tensor(g!) g_qkv_scratch, Tensor(h!) g_kv_scratch, "
            "Tensor(i!) g_attn_out, Tensor(j!) g_mlp_inter, "
            "Tensor(k!) g_z_scratch, Tensor(l!) g_beta_scratch, Tensor(m!) g_alpha_scratch, "
            "Tensor(n!) g_normalized, Tensor(o!) g_fa_partials, Tensor(p!) g_rope_inv_freq, "
            "int input_token_id, int position, int pos_h, int pos_w, int max_seq_len, "
            "float yarn_scale, float yarn_beta_fast, float yarn_beta_slow, "
            "int yarn_orig_ctx, bool yarn_enabled, int num_blocks, "
            "Tensor(q!)? g_layer_outputs) -> ()");
    ops.impl("decode_qwen3x", torch::kCUDA, &decode_qwen3x);
}

REGISTER_EXTENSION(qwen3x_C)
