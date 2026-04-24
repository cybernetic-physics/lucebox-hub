/**
 * CUTLASS sm_100 FMHA backward for Qwen3.5-0.8B full-attention layers.
 *
 * Wraps the kernel from
 *   /root/cutlass/examples/77_blackwell_fmha/kernel/
 *     sm100_fmha_bwd_kernel_tma_warpspecialized.hpp
 *
 * Our FA layers have:
 *   Hq = 8 (query heads), Hk = 2 (kv heads, GQA), D = 256, bf16, causal.
 *
 * Input/output layouts:
 *   Q, K, V    : [B, S, H, D] bf16
 *   dO         : [B, S, H, D] bf16
 *   dQ, dK, dV : [B, S, H, D] bf16
 *
 * TODO (this file is a SCAFFOLD):
 *   - Pull in the template instantiation from the example's main .cu
 *     and specialize to (bf16, D=256, causal, GQA 4:1).
 *   - Expose as extern "C" for torch op binding.
 *   - Wire torch binding in CMakeLists.txt + a torch.library.
 *
 * The current file compiles against CUTLASS and exposes a launcher
 * signature, but the kernel dispatch is TODO — returns cudaErrorNotYetImplemented.
 * This lets downstream Python wiring proceed against a stable ABI
 * while the kernel specialization lands.
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
