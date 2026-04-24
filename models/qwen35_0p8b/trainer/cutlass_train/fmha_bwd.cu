/**
 * CUTLASS sm_100 FMHA backward for Qwen3.5-0.8B full-attention layers.
 *
 * Target kernel:
 *   cutlass::fmha::device::Sm100FmhaBwd<
 *       ProblemShape, Element, ElementAccumulator,
 *       TileShape, kIsMla, ActiveMask>
 *   from /root/cutlass/examples/77_blackwell_fmha/device/fmha_device_bwd.hpp
 *
 * Needed instantiation (our Qwen3.5-0.8B FA layer):
 *   Element            = cutlass::bfloat16_t   (NOT fp16 — example uses half_t)
 *   ElementAccumulator = float
 *   ProblemShape       = cute::tuple<int, int, int, int,
 *                            cute::tuple<cute::tuple<int,int>, int>>
 *                        = (Q, K, D, D_VO, ((H_R, H_K), B))
 *                        = (S, S, 256, 256, ((4, 2), 1))
 *     where H_R = Hq/Hk = 4 (GQA ratio), H_K = Hk = 2, B = 1.
 *   TileShape          = Shape<_128, _128, _128>  (Blackwell recipe)
 *   kIsMla             = false
 *   ActiveMask         = cutlass::fmha::collective::CausalMask
 *
 * Strides (from example lines 419-423):
 *   StrideQ   = Stride<int, _1, Stride<Stride<int, int>, int>>  // Q D ((H_R,H_K),B)
 *   StrideK   = Stride<int, _1, Stride<Stride<_0, int>, int>>   // GQA: K head is shared
 *                                                                  across H_R Q heads, stride 0
 *   StrideV   = StrideK
 *   StrideO   = StrideQ
 *   StrideLSE = Stride<_1, Stride<Stride<int, int>, int>>       // Q ((H_R,H_K),B)
 *
 * Required inputs (arguments struct, from example lines 688-703):
 *   Q, K, V          — from our forward (saved / live in fa_k/v_cache)
 *   O                — forward output (our `dn_out_buf` after gate)
 *   LSE              — log-sum-exp from forward softmax, PER QUERY ROW.
 *                      Our current forward DOES NOT save LSE; we'd need
 *                      to either:
 *                        (a) save LSE as a new [1, Hq, S] fp32 slab
 *                            during prefill_bf16_with_lora, OR
 *                        (b) recompute LSE via a separate pass (costs
 *                            one softmax.fwd per FA layer, ~1ms at 32k).
 *   dO               — grad from upstream (o_proj bwd)
 *   dQ, dK, dV       — outputs
 *   softmax_scale    = 1 / sqrt(D)
 *   hw_info          — SM count / runtime hints
 *
 * Porting the example's half_t path to bf16:
 *   The Sm100FmhaBwd kernel should support bf16 via the standard CUTLASS
 *   Element<->tensor-core dispatch, but the example only instantiates
 *   fp16/fp8. Substitute Element = cutlass::bfloat16_t and the tcgen05
 *   MMA dispatch should pick up bf16 automatically. If the compile fails
 *   (template deduction), we have two fallbacks:
 *     1. Do attention at fp16: cast Q/K/V/dO from bf16 → fp16 at launch,
 *        run the stock fp16 kernel, cast dQ/dK/dV back. Small accuracy
 *        hit vs bf16 in practice.
 *     2. Instantiate a fresh specialization of
 *        collective/sm100_fmha_bwd_kernel_tma_warpspecialized.hpp
 *        templated on bf16. Comments in that file note bf16 should just
 *        work via the existing MMA atom table.
 *
 * Time estimate: 1–2 days of focused CUTLASS 3.x work.
 *
 * Current state: TODO, returns cudaErrorNotSupported. Download-side of
 * the API (torch binding) is wired in torch_bindings.cpp so upstream
 * code can dispatch around it while this lands.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// CUTLASS FMHA backward headers (from examples/77_blackwell_fmha).
// NOTE: these paths are resolved via CMake include dirs.
// #include <device/fmha_device_bwd.hpp>
// #include <kernel/sm100_fmha_bwd_kernel_tma_warpspecialized.hpp>

extern "C" cudaError_t cutlass_fmha_bwd_sm100(
    const __nv_bfloat16 *Q,        // [B, S, Hq, D]
    const __nv_bfloat16 *K,        // [B, S, Hk, D]
    const __nv_bfloat16 *V,        // [B, S, Hk, D]
    const __nv_bfloat16 *dO,       // [B, S, Hq, D]
    const float *logsumexp,        // [B, Hq, S] from forward (if available)
    __nv_bfloat16 *dQ,             // [B, S, Hq, D]
    __nv_bfloat16 *dK,             // [B, S, Hk, D]
    __nv_bfloat16 *dV,             // [B, S, Hk, D]
    int batch, int S, int Hq, int Hk, int D,
    float scale,
    bool causal,
    cudaStream_t stream)
{
    // TODO: instantiate the sm100 FMHA bwd kernel with our shapes.
    // See examples/77_blackwell_fmha/77_blackwell_fmha_bwd.cu for the
    // template specialization pattern.
    (void)Q; (void)K; (void)V; (void)dO; (void)logsumexp;
    (void)dQ; (void)dK; (void)dV;
    (void)batch; (void)S; (void)Hq; (void)Hk; (void)D;
    (void)scale; (void)causal; (void)stream;
    return cudaErrorNotSupported;
}
