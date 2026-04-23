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

extern "C" void launch_tcgen05_gemm_one_tile(
    const void *A, const void *B, void *C,
    int M_TOTAL, int N_TOTAL, int K, cudaStream_t stream);

void tcgen05_gemm_one_tile(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    TORCH_CHECK(A.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(B.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(C.scalar_type() == torch::kFloat32);
    int M_TOTAL = (int)A.size(0);
    int N_TOTAL = (int)B.size(0);
    int K = (int)A.size(1);
    launch_tcgen05_gemm_one_tile(A.data_ptr(), B.data_ptr(), C.data_ptr(),
                                  M_TOTAL, N_TOTAL, K,
                                  c10::cuda::getCurrentCUDAStream().stream());
}

TORCH_LIBRARY(TORCH_EXTENSION_NAME, ops) {
    ops.def("tcgen05_gemm_one_tile(Tensor A, Tensor B, Tensor C) -> ()");
    ops.impl("tcgen05_gemm_one_tile", torch::kCUDA, &tcgen05_gemm_one_tile);
}
REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
