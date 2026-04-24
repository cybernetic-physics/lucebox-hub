/**
 * torch bindings for cutlass_train_C.
 *
 * Exposes the three CUTLASS-backed kernels as `torch.ops.cutlass_train_C.*`
 * ops. They can be called from Python once the kernel implementations
 * land; until then, calls return NotImplementedError-equivalent errors
 * so higher-level code can dispatch around unsupported ops while the
 * Phase-2 stack is being wired in.
 */

#include <Python.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
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


extern "C" cudaError_t cutlass_gemm_bf16_sm100_rowmajor(
    const __nv_bfloat16 *A, int lda,
    const __nv_bfloat16 *B, int ldb,
    __nv_bfloat16 *C, int ldc,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream);

extern "C" cudaError_t cutlass_fmha_bwd_sm100(
    const __nv_bfloat16 *Q, const __nv_bfloat16 *K, const __nv_bfloat16 *V,
    const __nv_bfloat16 *dO,
    const float *logsumexp,
    __nv_bfloat16 *dQ, __nv_bfloat16 *dK, __nv_bfloat16 *dV,
    int batch, int S, int Hq, int Hk, int D,
    float scale, bool causal,
    cudaStream_t stream);

extern "C" cudaError_t cutlass_deltanet_bwd_sm100(
    const __nv_bfloat16 *q, const __nv_bfloat16 *k, const __nv_bfloat16 *v,
    const float *beta, const float *decay,
    const float *state_saves,
    const __nv_bfloat16 *dy,
    __nv_bfloat16 *dq, __nv_bfloat16 *dk, __nv_bfloat16 *dv,
    float *dbeta, float *ddecay, float *dstate_init,
    int S, int H, int Dk, int Dv,
    int chunk_size,
    cudaStream_t stream);


TORCH_LIBRARY(cutlass_train_C, ops) {
    ops.def("cutlass_gemm_bf16(Tensor A, Tensor B, Tensor(a!) C, "
            "float alpha, float beta) -> ()");
    ops.impl("cutlass_gemm_bf16", torch::kCUDA, +[](
        torch::Tensor A, torch::Tensor B, torch::Tensor C,
        double alpha, double beta) {
            TORCH_CHECK(A.scalar_type() == torch::kBFloat16);
            TORCH_CHECK(B.scalar_type() == torch::kBFloat16);
            TORCH_CHECK(C.scalar_type() == torch::kBFloat16);
            int M = (int)A.size(0);
            int K = (int)A.size(1);
            int N = (int)B.size(1);
            TORCH_CHECK(B.size(0) == K);
            TORCH_CHECK(C.size(0) == M && C.size(1) == N);
            cudaError_t err = cutlass_gemm_bf16_sm100_rowmajor(
                (const __nv_bfloat16*)A.data_ptr(), (int)A.stride(0),
                (const __nv_bfloat16*)B.data_ptr(), (int)B.stride(0),
                (__nv_bfloat16*)C.data_ptr(), (int)C.stride(0),
                M, N, K, (float)alpha, (float)beta,
                c10::cuda::getCurrentCUDAStream().stream());
            TORCH_CHECK(err == cudaSuccess, "cutlass_gemm_bf16 failed: ",
                        cudaGetErrorString(err));
        });

    ops.def("fa_bwd(Tensor Q, Tensor K, Tensor V, Tensor dO, "
            "Tensor? logsumexp, "
            "Tensor(a!) dQ, Tensor(b!) dK, Tensor(c!) dV, "
            "float scale, bool causal) -> ()");
    ops.impl("fa_bwd", torch::kCUDA, +[](
        torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor dO,
        c10::optional<torch::Tensor> logsumexp,
        torch::Tensor dQ, torch::Tensor dK, torch::Tensor dV,
        double scale, bool causal) {
            TORCH_CHECK(Q.scalar_type() == torch::kBFloat16);
            int B_ = (int)Q.size(0);
            int S = (int)Q.size(1);
            int Hq = (int)Q.size(2);
            int D = (int)Q.size(3);
            int Hk = (int)K.size(2);
            const float *lse = logsumexp.has_value()
                ? (const float*)logsumexp->data_ptr()
                : nullptr;
            cudaError_t err = cutlass_fmha_bwd_sm100(
                (const __nv_bfloat16*)Q.data_ptr(),
                (const __nv_bfloat16*)K.data_ptr(),
                (const __nv_bfloat16*)V.data_ptr(),
                (const __nv_bfloat16*)dO.data_ptr(),
                lse,
                (__nv_bfloat16*)dQ.data_ptr(),
                (__nv_bfloat16*)dK.data_ptr(),
                (__nv_bfloat16*)dV.data_ptr(),
                B_, S, Hq, Hk, D, (float)scale, causal,
                c10::cuda::getCurrentCUDAStream().stream());
            TORCH_CHECK(err == cudaSuccess, "cutlass fa_bwd: ",
                        cudaGetErrorString(err));
        });

    ops.def("deltanet_bwd(Tensor q, Tensor k, Tensor v, "
            "Tensor beta, Tensor decay, Tensor state_saves, Tensor dy, "
            "Tensor(a!) dq, Tensor(b!) dk, Tensor(c!) dv, "
            "Tensor(d!) dbeta, Tensor(e!) ddecay, Tensor(f!) dstate_init, "
            "int chunk_size) -> ()");
    ops.impl("deltanet_bwd", torch::kCUDA, +[](
        torch::Tensor q, torch::Tensor k, torch::Tensor v,
        torch::Tensor beta, torch::Tensor decay, torch::Tensor state_saves,
        torch::Tensor dy,
        torch::Tensor dq, torch::Tensor dk, torch::Tensor dv,
        torch::Tensor dbeta, torch::Tensor ddecay, torch::Tensor dstate_init,
        int64_t chunk_size) {
            int S  = (int)q.size(0);
            int H  = (int)q.size(1);
            int Dk = (int)q.size(2);
            int Dv = (int)v.size(2);
            cudaError_t err = cutlass_deltanet_bwd_sm100(
                (const __nv_bfloat16*)q.data_ptr(),
                (const __nv_bfloat16*)k.data_ptr(),
                (const __nv_bfloat16*)v.data_ptr(),
                (const float*)beta.data_ptr(),
                (const float*)decay.data_ptr(),
                (const float*)state_saves.data_ptr(),
                (const __nv_bfloat16*)dy.data_ptr(),
                (__nv_bfloat16*)dq.data_ptr(),
                (__nv_bfloat16*)dk.data_ptr(),
                (__nv_bfloat16*)dv.data_ptr(),
                (float*)dbeta.data_ptr(), (float*)ddecay.data_ptr(),
                (float*)dstate_init.data_ptr(),
                S, H, Dk, Dv, (int)chunk_size,
                c10::cuda::getCurrentCUDAStream().stream());
            TORCH_CHECK(err == cudaSuccess, "cutlass deltanet_bwd: ",
                        cudaGetErrorString(err));
        });
}

REGISTER_EXTENSION(cutlass_train_C)
