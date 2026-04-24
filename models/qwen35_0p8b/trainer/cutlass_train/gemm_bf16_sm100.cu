/**
 * CUTLASS 3.x bf16 GEMM on sm_100 (Blackwell, B200).
 *
 * Drop-in replacement for our cuBLAS bf16 matmul, used by the per-layer
 * backward for linear projections (dX = dY @ W.T, dW accumulate, etc.).
 * Binds to a torch op so Python can call it directly from the trainer's
 * backward chain.
 *
 * The point isn't to beat cuBLAS (cuBLAS is already near-SOL for large
 * GEMMs on Blackwell). The point is that once FMHA bwd is a CUTLASS
 * kernel in the same .so, we can fuse or co-schedule matmuls +
 * attention in the same CUTLASS template machinery — fewer kernel
 * launches, fewer stream dependencies.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cute/tensor.hpp>


using namespace cute;


// Launcher: C = A @ B for row-major bf16 operands on sm_100.
// A: [M, K], B: [K, N], C: [M, N] bf16.
extern "C" cudaError_t cutlass_gemm_bf16_sm100_rowmajor(
    const __nv_bfloat16 *A, int lda,
    const __nv_bfloat16 *B, int ldb,
    __nv_bfloat16 *C, int ldc,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream)
{
    // cutlass::bfloat16_t shares ABI with __nv_bfloat16.
    using DTypeA = cutlass::bfloat16_t;
    using DTypeB = cutlass::bfloat16_t;
    using DTypeC = cutlass::bfloat16_t;
    using DTypeAccum = float;

    // Tile sizes: start modest, will be retuned when this is wired into
    // backward LoRA projections (small M, large K/N for LoRA A; large
    // M, small N/K for LoRA B).
    using TileShape        = Shape<_128, _128, _64>;
    using ClusterShape     = Shape<_1, _1, _1>;
    using KernelSchedule   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;

    using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
        DTypeA, cutlass::layout::RowMajor, 8,
        DTypeB, cutlass::layout::RowMajor, 8,
        DTypeAccum,
        TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename cutlass::epilogue::collective::DefaultEpilogue<
            cutlass::gemm::TagToStrideC_t<cutlass::layout::RowMajor>,
            cutlass::gemm::TagToStrideC_t<cutlass::layout::RowMajor>,
            cutlass::epilogue::thread::LinearCombination<DTypeC, 1, DTypeAccum, DTypeAccum>>::SharedStorage))>,
        KernelSchedule
      >::CollectiveOp;

    using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
        TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        DTypeAccum, DTypeAccum,
        DTypeC, cutlass::layout::RowMajor, 8,
        DTypeC, cutlass::layout::RowMajor, 8,
        EpilogueSchedule
      >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int>,
        CollectiveMainloop, CollectiveEpilogue>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {
          reinterpret_cast<const DTypeA*>(A),
          {lda, cute::_1{}, cute::_0{}},
          reinterpret_cast<const DTypeB*>(B),
          {ldb, cute::_1{}, cute::_0{}},
        },
        {
          {alpha, beta},
          reinterpret_cast<const DTypeC*>(C),
          {ldc, cute::_1{}, cute::_0{}},
          reinterpret_cast<DTypeC*>(C),
          {ldc, cute::_1{}, cute::_0{}},
        }
    };

    Gemm gemm;
    size_t ws = gemm.get_workspace_size(args);
    static void *ws_ptr = nullptr;
    static size_t ws_cap = 0;
    if (ws > ws_cap) {
        if (ws_ptr) cudaFree(ws_ptr);
        cudaMalloc(&ws_ptr, ws);
        ws_cap = ws;
    }
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
        return cudaErrorInvalidValue;
    }
    if (gemm.initialize(args, ws_ptr, stream) != cutlass::Status::kSuccess) {
        return cudaErrorInvalidValue;
    }
    if (gemm.run(stream) != cutlass::Status::kSuccess) {
        return cudaErrorInvalidDeviceFunction;
    }
    return cudaSuccess;
}
