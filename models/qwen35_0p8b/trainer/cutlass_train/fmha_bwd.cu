/**
 * CUTLASS sm_100 FMHA backward for Qwen3.5-0.8B FA layers.
 *
 * Qwen3.5-0.8B FA shapes per layer:
 *   Hq = 8, Hk = 2 (GQA 4:1), D = 256, D_VO = 256, bf16, causal.
 *
 * Wraps cutlass::fmha::device::Sm100FmhaBwd from examples/77_blackwell_fmha/
 * with bf16 Element. Produces dQ, dK, dV for causal GQA attention.
 *
 * Required forward artifacts (caller must save during fwd):
 *   O    : [B, S, Hq, D]   bf16    — attention output
 *   LSE  : [B, Hq, S]      fp32    — per-row log-sum-exp of scores
 *                                     after scaling but before softmax
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/kernel_hardware_info.h>
#include <cute/tensor.hpp>
#include <cute/layout.hpp>

// Order matters: fmha_fusion defines CausalMask / VariableLength that
// the kernel header uses internally.
#include "collective/fmha_fusion.hpp"
#include "device/fmha_device_bwd.hpp"

using namespace cute;

using Element       = cutlass::bfloat16_t;
using ElementAccum  = float;

// ProblemShape = (Q, K, D, D_VO, ((H_R, H_K), B))
using ProblemShape = cute::tuple<
    int, int, int, int,
    cute::tuple<cute::tuple<int, int>, int>>;

using TileShape  = Shape<_128, _128, _128>;
using ActiveMask = cutlass::fmha::collective::CausalMask<true>;

using BwdOp = cutlass::fmha::device::Sm100FmhaBwd<
    ProblemShape, Element, ElementAccum, TileShape,
    /*IsMla=*/ false, ActiveMask>;


// --- Stride builders (match kernel Argument types; see fmha_device_bwd.hpp
//     lines 102-121). Q/dQ/O/dO and K/V/dK/dV have different shapes in
//     the GQA dim ("H_R" stride = _0 for K/V since the same K/V head is
//     shared across H_R query heads).

static auto make_stride_qod(int S, int Hq, int Hk, int D, int B) {
    int H_R = Hq / Hk;
    int pos_stride = Hq * D;
    int hr_stride  = D;
    int hk_stride  = H_R * D;
    int b_stride   = (B == 1) ? 0 : Hq * D * S;
    return make_stride(pos_stride, _1{},
                       make_stride(make_stride(hr_stride, hk_stride), b_stride));
}

static auto make_stride_kv(int S, int Hk, int D, int B) {
    int pos_stride = Hk * D;
    int hk_stride  = D;
    int b_stride   = (B == 1) ? 0 : Hk * D * S;
    return make_stride(pos_stride, _1{},
                       make_stride(make_stride(_0{}, hk_stride), b_stride));
}

static auto make_stride_lse(int S, int Hq, int Hk, int B) {
    int H_R = Hq / Hk;
    int hr_stride = S;
    int hk_stride = H_R * S;
    int b_stride  = (B == 1) ? 0 : Hq * S;
    return make_stride(_1{},
                       make_stride(make_stride(hr_stride, hk_stride), b_stride));
}


extern "C" cudaError_t cutlass_fmha_bwd_sm100(
    const __nv_bfloat16 *Q,
    const __nv_bfloat16 *K,
    const __nv_bfloat16 *V,
    const __nv_bfloat16 *O,            // forward output
    const __nv_bfloat16 *dO,
    const float *logsumexp,            // [B, Hq, S] from forward softmax
    __nv_bfloat16 *dQ,
    __nv_bfloat16 *dK,
    __nv_bfloat16 *dV,
    int batch, int S, int Hq, int Hk, int D,
    float scale,
    bool causal,
    cudaStream_t stream)
{
    if (!causal)           return cudaErrorNotSupported;
    if (Hq % Hk != 0)      return cudaErrorInvalidValue;
    if (D != 128 && D != 256) return cudaErrorInvalidValue;

    int H_R = Hq / Hk;

    ProblemShape ps = make_tuple(
        S,                   // Q
        S,                   // K
        D,                   // D
        D,                   // D_VO (== D for us)
        make_tuple(make_tuple(H_R, Hk), batch));

    typename BwdOp::Arguments args{
        ps,
        reinterpret_cast<const Element*>(Q),  make_stride_qod(S, Hq, Hk, D, batch),
        reinterpret_cast<const Element*>(K),  make_stride_kv (S, Hk,    D, batch),
        reinterpret_cast<const Element*>(V),  make_stride_kv (S, Hk,    D, batch),
        reinterpret_cast<const Element*>(O),  make_stride_qod(S, Hq, Hk, D, batch),
        logsumexp,                            make_stride_lse(S, Hq, Hk,  batch),
        reinterpret_cast<const Element*>(dO), make_stride_qod(S, Hq, Hk, D, batch),
        reinterpret_cast<Element*>(dQ),       make_stride_qod(S, Hq, Hk, D, batch),
        reinterpret_cast<Element*>(dK),       make_stride_kv (S, Hk,    D, batch),
        reinterpret_cast<Element*>(dV),       make_stride_kv (S, Hk,    D, batch),
        scale,
        cutlass::KernelHardwareInfo{}
    };

    BwdOp op;
    if (op.can_implement(args) != cutlass::Status::kSuccess) {
        return cudaErrorInvalidConfiguration;
    }

    size_t ws_bytes = BwdOp::get_workspace_size(args);
    static void *ws_ptr = nullptr;
    static size_t ws_cap = 0;
    if (ws_bytes > ws_cap) {
        if (ws_ptr) cudaFree(ws_ptr);
        if (cudaMalloc(&ws_ptr, ws_bytes) != cudaSuccess) {
            return cudaErrorMemoryAllocation;
        }
        ws_cap = ws_bytes;
    }

    if (op.initialize(args, ws_ptr, stream) != cutlass::Status::kSuccess) {
        return cudaErrorInvalidValue;
    }
    if (op.run(stream) != cutlass::Status::kSuccess) {
        return cudaErrorLaunchFailure;
    }
    return cudaSuccess;
}
