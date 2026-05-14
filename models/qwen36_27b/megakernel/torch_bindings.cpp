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
}

REGISTER_EXTENSION(qwen3x_C)
