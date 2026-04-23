/**
 * PyTorch bindings for the fused LoRA training megakernel.
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

extern "C" void launch_lora_train(
    const int *context_tokens, const int *target_tokens, const float *token_weights,
    const void *embedding, const void *output_weight,
    void *lora_a, void *lora_b,
    float *m_a, float *v_a, float *m_b, float *v_b,
    void *hidden, float *lora_h, float *logits_buf, float *grad_logits,
    float *grad_lora_a, float *grad_lora_b, float *grad_lora_h,
    float *selected_out, float *loss_out,
    int T, int H, int V, int R,
    float lr, float beta1, float beta2, float eps, float wd,
    float bias_correction1, float bias_correction2,
    int do_update,
    cudaStream_t stream);

void lora_train_step(
    torch::Tensor context_tokens, torch::Tensor target_tokens, torch::Tensor token_weights,
    torch::Tensor embedding, torch::Tensor output_weight,
    torch::Tensor lora_a, torch::Tensor lora_b,
    torch::Tensor m_a, torch::Tensor v_a, torch::Tensor m_b, torch::Tensor v_b,
    torch::Tensor hidden_ws, torch::Tensor lora_h_ws,
    torch::Tensor logits_ws, torch::Tensor grad_logits_ws,
    torch::Tensor grad_lora_a_ws, torch::Tensor grad_lora_b_ws, torch::Tensor grad_lora_h_ws,
    torch::Tensor selected_out, torch::Tensor loss_out,
    int64_t V, int64_t R,
    double lr, double beta1, double beta2, double eps, double wd,
    double bias_correction1, double bias_correction2,
    int64_t do_update)
{
    TORCH_CHECK(context_tokens.is_cuda() && context_tokens.scalar_type() == torch::kInt32,
                "context_tokens must be int32 CUDA");
    TORCH_CHECK(target_tokens.is_cuda() && target_tokens.scalar_type() == torch::kInt32,
                "target_tokens must be int32 CUDA");
    TORCH_CHECK(token_weights.is_cuda() && token_weights.scalar_type() == torch::kFloat32,
                "token_weights must be float32 CUDA");
    TORCH_CHECK(embedding.is_cuda() && embedding.scalar_type() == torch::kBFloat16,
                "embedding must be bfloat16 CUDA");
    TORCH_CHECK(output_weight.is_cuda() && output_weight.scalar_type() == torch::kBFloat16,
                "output_weight must be bfloat16 CUDA");
    TORCH_CHECK(lora_a.is_cuda() && lora_a.scalar_type() == torch::kBFloat16,
                "lora_a must be bfloat16 CUDA");
    TORCH_CHECK(lora_b.is_cuda() && lora_b.scalar_type() == torch::kBFloat16,
                "lora_b must be bfloat16 CUDA");

    int T = (int)context_tokens.size(0);
    int H = (int)embedding.size(1);

    launch_lora_train(
        (const int *)context_tokens.data_ptr(),
        (const int *)target_tokens.data_ptr(),
        (const float *)token_weights.data_ptr(),
        embedding.data_ptr(),
        output_weight.data_ptr(),
        lora_a.data_ptr(),
        lora_b.data_ptr(),
        (float *)m_a.data_ptr(),
        (float *)v_a.data_ptr(),
        (float *)m_b.data_ptr(),
        (float *)v_b.data_ptr(),
        hidden_ws.data_ptr(),
        (float *)lora_h_ws.data_ptr(),
        (float *)logits_ws.data_ptr(),
        (float *)grad_logits_ws.data_ptr(),
        (float *)grad_lora_a_ws.data_ptr(),
        (float *)grad_lora_b_ws.data_ptr(),
        (float *)grad_lora_h_ws.data_ptr(),
        (float *)selected_out.data_ptr(),
        (float *)loss_out.data_ptr(),
        T, H, (int)V, (int)R,
        (float)lr, (float)beta1, (float)beta2, (float)eps, (float)wd,
        (float)bias_correction1, (float)bias_correction2,
        (int)do_update,
        c10::cuda::getCurrentCUDAStream().stream());
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("lora_train_step("
            "Tensor context_tokens, Tensor target_tokens, Tensor token_weights, "
            "Tensor embedding, Tensor output_weight, "
            "Tensor lora_a, Tensor lora_b, "
            "Tensor m_a, Tensor v_a, Tensor m_b, Tensor v_b, "
            "Tensor hidden_ws, Tensor lora_h_ws, "
            "Tensor logits_ws, Tensor grad_logits_ws, "
            "Tensor grad_lora_a_ws, Tensor grad_lora_b_ws, Tensor grad_lora_h_ws, "
            "Tensor selected_out, Tensor loss_out, "
            "int V, int R, "
            "float lr, float beta1, float beta2, float eps, float wd, "
            "float bias_correction1, float bias_correction2, "
            "int do_update) -> ()");
    ops.impl("lora_train_step", torch::kCUDA, &lora_train_step);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
