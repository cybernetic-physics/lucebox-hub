/**
 * Templated decode kernel skeleton for Qwen3.x.
 *
 * This file is the Phase-1 on-ramp: it instantiates the Cfg-templated
 * primitives that the future full decode megakernel will be built out
 * of, and proves the build pipeline works for both Cfg_0p8B and
 * Cfg_27B specializations.
 *
 * What's implemented in this commit:
 *   - rmsnorm<Cfg>            : fp32-accumulated RMSNorm with gain
 *   - fused_silu_mul<Cfg>     : SwiGLU activation
 *   - matvec_bf16<Cfg>        : warp-strided bf16 GEMV (for proj layers)
 *   - mlp_forward<Cfg>        : MLP path (RMSNorm -> gate/up SwiGLU -> down)
 *
 * What's marked TODO (handed off to subsequent commits):
 *   - delta_net_layer<Cfg>    : gated DeltaNet recurrence. NEEDS WORK for
 *                               Cfg_27B because V heads (48) != QK heads
 *                               (16). Existing 0.8B code in
 *                               models/qwen35_0p8b/kernel.cu assumes
 *                               unified V/QK; the 27B path needs the
 *                               DN_V_PER_QK GQA-style replication.
 *   - full_attention_layer<Cfg>: GQA full attention. Mostly parameterizable
 *                               from the 0.8B implementation; FA_GQA_RATIO
 *                               is just larger (24/4=6 vs 8/2=4).
 *   - decode_kernel<Cfg>      : the persistent megakernel that walks all
 *                               NUM_LAYERS layers per token.
 *
 * The torch op `decode_qwen3x<Cfg>` is wired in torch_bindings.cpp and
 * dispatches to the right specialization at runtime via a model-id arg.
 *
 * Once the TODOs are filled in, this file becomes the canonical 27B
 * decode path; the 0.8B specialization stays a drop-in replacement for
 * the existing `models/qwen35_0p8b/kernel.cu` decode (regression-tested
 * against the bf16-mega correctness reference).
 */
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdint.h>

#include "Cfg.cuh"

namespace cg = cooperative_groups;
using namespace lucebox::qwen3x;

// ---------------------------------------------------------------------------
// Build knobs (mirrors models/qwen35_0p8b's tuning).
// ---------------------------------------------------------------------------
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
constexpr float RMS_EPS = 1e-6f;

// ---------------------------------------------------------------------------
// Primitives — fully templated on Cfg.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

template<typename Cfg>
__device__ void rmsnorm(
    const __nv_bfloat16 *__restrict__ in,
    const __nv_bfloat16 *__restrict__ gain,
    __nv_bfloat16 *__restrict__ out)
{
    constexpr int D = Cfg::HIDDEN;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    __shared__ float s[NUM_WARPS];

    float ssq = 0.0f;
    for (int i = tid; i < D; i += BLOCK_SIZE) {
        float v = __bfloat162float(in[i]);
        ssq += v * v;
    }
    ssq = warp_reduce_sum(ssq);
    if (lane == 0) s[warp_id] = ssq;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < NUM_WARPS) ? s[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) s[0] = rsqrtf(v / float(D) + RMS_EPS);
    }
    __syncthreads();
    float rstd = s[0];
    for (int i = tid; i < D; i += BLOCK_SIZE) {
        float v = __bfloat162float(in[i]);
        float g = __bfloat162float(__ldg(gain + i));
        out[i] = __float2bfloat16(v * rstd * (1.0f + g));
    }
}

template<typename Cfg>
__device__ void fused_silu_mul(
    const __nv_bfloat16 *__restrict__ gate,
    const __nv_bfloat16 *__restrict__ up,
    __nv_bfloat16 *__restrict__ out)
{
    constexpr int D = Cfg::INTERMEDIATE;
    int tid = threadIdx.x;
    for (int i = tid; i < D; i += BLOCK_SIZE) {
        float g = __bfloat162float(gate[i]);
        float u = __bfloat162float(up[i]);
        float silu;
        // silu(g) = g * sigmoid(g) = g / (1 + exp(-g))
        float ex; asm volatile("ex2.approx.ftz.f32 %0, %1;"
                                : "=f"(ex) : "f"(-g * 1.44269504088896340736f));
        silu = g / (1.0f + ex);
        out[i] = __float2bfloat16(silu * u);
    }
}

// Warp-strided bf16 GEMV: out[m] = sum_k weight[m,k] * input[k].
//   weight is row-major [out_dim, in_dim] bf16.
template<typename Cfg>
__device__ void matvec_bf16_row(
    const __nv_bfloat16 *__restrict__ input,
    const __nv_bfloat16 *__restrict__ weight,
    float *__restrict__ output,
    int out_dim, int in_dim)
{
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int rs = block_id * rows_per_block;
    int re = (rs + rows_per_block < out_dim) ? rs + rows_per_block : out_dim;

    for (int m_base = rs; m_base < re; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < re) {
            const __nv_bfloat16 *w_row = weight + (size_t)m * in_dim;
            float sum = 0.0f;
            for (int k = lane * 4; k < in_dim; k += WARP_SIZE * 4) {
                // Unrolled 4 bf16 per lane per iter.
                __nv_bfloat16 w0 = __ldg(w_row + k + 0);
                __nv_bfloat16 w1 = __ldg(w_row + k + 1);
                __nv_bfloat16 w2 = __ldg(w_row + k + 2);
                __nv_bfloat16 w3 = __ldg(w_row + k + 3);
                __nv_bfloat16 x0 = input[k + 0];
                __nv_bfloat16 x1 = input[k + 1];
                __nv_bfloat16 x2 = input[k + 2];
                __nv_bfloat16 x3 = input[k + 3];
                sum += __bfloat162float(w0) * __bfloat162float(x0);
                sum += __bfloat162float(w1) * __bfloat162float(x1);
                sum += __bfloat162float(w2) * __bfloat162float(x2);
                sum += __bfloat162float(w3) * __bfloat162float(x3);
            }
            sum = warp_reduce_sum(sum);
            if (lane == 0) output[m] = sum;
        }
    }
}

// ---------------------------------------------------------------------------
// MLP path — implemented (no DN V/QK split here, just plain GEMM/SwiGLU).
// ---------------------------------------------------------------------------
template<typename Cfg>
__device__ void mlp_forward(
    const __nv_bfloat16 *__restrict__ input,         // [HIDDEN]
    const __nv_bfloat16 *__restrict__ pre_mlp_gain,  // [HIDDEN]
    const __nv_bfloat16 *__restrict__ w_gate,        // [INTER, HIDDEN]
    const __nv_bfloat16 *__restrict__ w_up,          // [INTER, HIDDEN]
    const __nv_bfloat16 *__restrict__ w_down,        // [HIDDEN, INTER]
    __nv_bfloat16 *__restrict__ shmem_normalized,    // [HIDDEN] bf16
    float *__restrict__ g_gate,                       // [INTER] fp32 scratch
    float *__restrict__ g_up,                         // [INTER] fp32 scratch
    __nv_bfloat16 *__restrict__ shmem_inter,         // [INTER] bf16
    __nv_bfloat16 *__restrict__ out)                  // [HIDDEN] bf16
{
    rmsnorm<Cfg>(input, pre_mlp_gain, shmem_normalized);
    __syncthreads();
    matvec_bf16_row<Cfg>(shmem_normalized, w_gate, g_gate, Cfg::INTERMEDIATE, Cfg::HIDDEN);
    matvec_bf16_row<Cfg>(shmem_normalized, w_up,   g_up,   Cfg::INTERMEDIATE, Cfg::HIDDEN);
    __syncthreads();
    // SwiGLU into shmem_inter, then down-proj into out.
    // We do this in two passes; pack fp32 -> bf16 first.
    int tid = threadIdx.x;
    for (int i = tid; i < Cfg::INTERMEDIATE; i += BLOCK_SIZE) {
        shmem_inter[i] = __float2bfloat16(g_gate[i]);
    }
    __nv_bfloat16 *gate_bf = shmem_inter;
    // up_bf will reuse shmem_normalized scratch since it's HIDDEN-sized,
    // but INTERMEDIATE > HIDDEN for both configs — so we use g_up's
    // implicit bf16 reinterpret as a workaround; for the real impl this
    // becomes a fused single-pass kernel. For now: pack gate into
    // shmem_inter, then re-read g_up as fp32 and fuse silu*up into a
    // temporary buffer. The downprojection then reads the temporary.
    __syncthreads();
    // (Fused silu*up below — but we need a place to put the result.
    //  The cleanest path is: write fp32 silu*up into g_gate, then pack to bf16,
    //  then down-proj. Reuses g_gate as scratch.)
    for (int i = tid; i < Cfg::INTERMEDIATE; i += BLOCK_SIZE) {
        float g = g_gate[i];
        float u = g_up[i];
        float ex; asm volatile("ex2.approx.ftz.f32 %0, %1;"
                                : "=f"(ex) : "f"(-g * 1.44269504088896340736f));
        float silu = g / (1.0f + ex);
        shmem_inter[i] = __float2bfloat16(silu * u);
    }
    __syncthreads();
    // down-proj [HIDDEN, INTERMEDIATE] @ shmem_inter -> g_gate (fp32 acc),
    // then add into residual (output) and pack to bf16.
    matvec_bf16_row<Cfg>(shmem_inter, w_down, g_gate, Cfg::HIDDEN, Cfg::INTERMEDIATE);
    __syncthreads();
    for (int i = tid; i < Cfg::HIDDEN; i += BLOCK_SIZE) {
        float r = __bfloat162float(input[i]);   // residual
        out[i] = __float2bfloat16(r + g_gate[i]);
    }
}

// ---------------------------------------------------------------------------
// Explicit specializations so each Cfg gets a cubin entry. Without these
// the compiler may elide unused templates.
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(BLOCK_SIZE)
mlp_smoke_0p8b(
    const __nv_bfloat16 *input, const __nv_bfloat16 *gain,
    const __nv_bfloat16 *w_gate, const __nv_bfloat16 *w_up,
    const __nv_bfloat16 *w_down,
    __nv_bfloat16 *sh_norm, float *g_gate, float *g_up,
    __nv_bfloat16 *sh_inter, __nv_bfloat16 *out)
{
    mlp_forward<Cfg_0p8B>(input, gain, w_gate, w_up, w_down,
                           sh_norm, g_gate, g_up, sh_inter, out);
}

extern "C" __global__ void __launch_bounds__(BLOCK_SIZE)
mlp_smoke_27b(
    const __nv_bfloat16 *input, const __nv_bfloat16 *gain,
    const __nv_bfloat16 *w_gate, const __nv_bfloat16 *w_up,
    const __nv_bfloat16 *w_down,
    __nv_bfloat16 *sh_norm, float *g_gate, float *g_up,
    __nv_bfloat16 *sh_inter, __nv_bfloat16 *out)
{
    mlp_forward<Cfg_27B>(input, gain, w_gate, w_up, w_down,
                          sh_norm, g_gate, g_up, sh_inter, out);
}

// ---------------------------------------------------------------------------
// Host-side launchers, exposed via torch_bindings.cpp.
// Each one is a smoke-test entry point that runs a single MLP layer for
// the given Cfg. Used by the Phase-1 test to prove both specializations
// compile + link + execute.
// ---------------------------------------------------------------------------
extern "C" void launch_mlp_smoke_0p8b(
    const void *input, const void *gain,
    const void *w_gate, const void *w_up, const void *w_down,
    void *sh_norm, void *g_gate, void *g_up, void *sh_inter, void *out,
    cudaStream_t stream)
{
    mlp_smoke_0p8b<<<dim3(1), dim3(BLOCK_SIZE), 0, stream>>>(
        (const __nv_bfloat16 *)input, (const __nv_bfloat16 *)gain,
        (const __nv_bfloat16 *)w_gate, (const __nv_bfloat16 *)w_up,
        (const __nv_bfloat16 *)w_down,
        (__nv_bfloat16 *)sh_norm, (float *)g_gate, (float *)g_up,
        (__nv_bfloat16 *)sh_inter, (__nv_bfloat16 *)out);
}

extern "C" void launch_mlp_smoke_27b(
    const void *input, const void *gain,
    const void *w_gate, const void *w_up, const void *w_down,
    void *sh_norm, void *g_gate, void *g_up, void *sh_inter, void *out,
    cudaStream_t stream)
{
    mlp_smoke_27b<<<dim3(1), dim3(BLOCK_SIZE), 0, stream>>>(
        (const __nv_bfloat16 *)input, (const __nv_bfloat16 *)gain,
        (const __nv_bfloat16 *)w_gate, (const __nv_bfloat16 *)w_up,
        (const __nv_bfloat16 *)w_down,
        (__nv_bfloat16 *)sh_norm, (float *)g_gate, (float *)g_up,
        (__nv_bfloat16 *)sh_inter, (__nv_bfloat16 *)out);
}
