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

struct LayerWeights {
    int layer_type;
    int _pad[3];
    void *ptrs[14];
};

extern "C" void launch_prefill_megakernel(
    const int *token_ids, int *out_token,
    const void *embed, const LayerWeights *layers,
    const void *final_norm_w, const void *lm_head_w,
    const void *lora_a_q_all, const void *lora_b_q_all,
    int lora_rank, float lora_scaling, float *lora_h_ws,
    void *fa_k_cache, void *fa_v_cache,
    float *dn_states, float *conv_bufs,
    void *hidden, void *residual, void *normalized,
    void *proj_buf, void *proj_buf2,
    void *attn_buf, void *mlp_buf, void *dn_out_buf,
    float *beta_buf, float *alpha_buf,
    void *final_normed,
    float *lm_bmv, int *lm_bmi,
    int S, int max_seq,
    cudaStream_t stream);

void train_mega_forward(
    torch::Tensor out_token, torch::Tensor token_ids,
    torch::Tensor embed, torch::Tensor layers_packed,
    torch::Tensor final_norm_w, torch::Tensor lm_head_w,
    torch::Tensor lora_a_q, torch::Tensor lora_b_q,
    int64_t lora_rank, double lora_scaling, torch::Tensor lora_h_ws,
    torch::Tensor fa_k_cache, torch::Tensor fa_v_cache,
    torch::Tensor dn_states, torch::Tensor conv_bufs,
    torch::Tensor hidden, torch::Tensor residual, torch::Tensor normalized,
    torch::Tensor proj_buf, torch::Tensor proj_buf2,
    torch::Tensor attn_buf, torch::Tensor mlp_buf, torch::Tensor dn_out_buf,
    torch::Tensor beta_buf, torch::Tensor alpha_buf,
    torch::Tensor final_normed,
    torch::Tensor lm_bmv, torch::Tensor lm_bmi,
    int64_t max_seq, bool use_lora)
{
    int S = (int)token_ids.size(0);
    const void *a_ptr = use_lora ? lora_a_q.data_ptr() : nullptr;
    const void *b_ptr = use_lora ? lora_b_q.data_ptr() : nullptr;

    launch_prefill_megakernel(
        (const int *)token_ids.data_ptr(), (int *)out_token.data_ptr(),
        embed.data_ptr(),
        reinterpret_cast<const LayerWeights *>(layers_packed.data_ptr()),
        final_norm_w.data_ptr(), lm_head_w.data_ptr(),
        a_ptr, b_ptr,
        (int)lora_rank, (float)lora_scaling, (float *)lora_h_ws.data_ptr(),
        fa_k_cache.data_ptr(), fa_v_cache.data_ptr(),
        (float *)dn_states.data_ptr(), (float *)conv_bufs.data_ptr(),
        hidden.data_ptr(), residual.data_ptr(), normalized.data_ptr(),
        proj_buf.data_ptr(), proj_buf2.data_ptr(),
        attn_buf.data_ptr(), mlp_buf.data_ptr(), dn_out_buf.data_ptr(),
        (float *)beta_buf.data_ptr(), (float *)alpha_buf.data_ptr(),
        final_normed.data_ptr(),
        (float *)lm_bmv.data_ptr(), (int *)lm_bmi.data_ptr(),
        S, (int)max_seq,
        c10::cuda::getCurrentCUDAStream().stream());
}

TORCH_LIBRARY(TORCH_EXTENSION_NAME, ops) {
    ops.def("train_mega_forward(Tensor out_token, Tensor token_ids, "
            "Tensor embed, Tensor layers_packed, "
            "Tensor final_norm_w, Tensor lm_head_w, "
            "Tensor lora_a_q, Tensor lora_b_q, "
            "int lora_rank, float lora_scaling, Tensor lora_h_ws, "
            "Tensor fa_k_cache, Tensor fa_v_cache, "
            "Tensor dn_states, Tensor conv_bufs, "
            "Tensor hidden, Tensor residual, Tensor normalized, "
            "Tensor proj_buf, Tensor proj_buf2, "
            "Tensor attn_buf, Tensor mlp_buf, Tensor dn_out_buf, "
            "Tensor beta_buf, Tensor alpha_buf, "
            "Tensor final_normed, "
            "Tensor lm_bmv, Tensor lm_bmi, "
            "int max_seq, bool use_lora) -> ()");
    ops.impl("train_mega_forward", torch::kCUDA, &train_mega_forward);
}
REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
