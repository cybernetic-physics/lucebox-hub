/**
 * RMSNorm primitives templated on Cfg. Two variants:
 *
 *   rmsnorm<Cfg>           -- in-place into a shared/global bf16 buffer
 *   rmsnorm_capture<Cfg>   -- block 0 also writes the unnormalized input
 *                             to a global residual buffer for later
 *                             addition (used between transformer layers)
 *
 * Qwen uses RMSNorm with **(1 + gain)** scaling — gain is stored as the
 * delta from 1.0, hence the `1.0f + g` term.
 */
#pragma once

#include "Cfg.cuh"
#include "helpers.cuh"

namespace lucebox::qwen3x {

#ifndef BLOCK_SIZE_RMS
#define BLOCK_SIZE_RMS BLOCK_SIZE
#endif

template<typename Cfg>
__device__ void rmsnorm(
    const __nv_bfloat16 *__restrict__ in,
    const __nv_bfloat16 *__restrict__ gain,
    __nv_bfloat16 *__restrict__ out)
{
    constexpr int D = Cfg::HIDDEN;
    constexpr float EPS = 1e-6f;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    __shared__ float s_reduce[NUM_WARPS];

    float ssq = 0.0f;
    for (int i = tid; i < D; i += BLOCK_SIZE) {
        float v = __bfloat162float(in[i]);
        ssq += v * v;
    }
    ssq = warp_reduce_sum_x(ssq);
    if (lane == 0) s_reduce[warp_id] = ssq;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < NUM_WARPS) ? s_reduce[lane] : 0.0f;
        v = warp_reduce_sum_x(v);
        if (lane == 0) s_reduce[0] = rsqrtf(v / float(D) + EPS);
    }
    __syncthreads();
    float rstd = s_reduce[0];
    for (int i = tid; i < D; i += BLOCK_SIZE) {
        float v = __bfloat162float(in[i]);
        float g = __bfloat162float(__ldg(gain + i));
        out[i] = __float2bfloat16(v * rstd * (1.0f + g));
    }
    __syncthreads();
}

// RMSNorm + captures the *unnormalized* input into a global residual
// buffer (block 0 only -- the residual is replicated to every block via
// __shared__ in the megakernel orchestrator).
template<typename Cfg>
__device__ void rmsnorm_capture(
    const __nv_bfloat16 *__restrict__ in,
    const __nv_bfloat16 *__restrict__ gain,
    __nv_bfloat16 *__restrict__ out,
    __nv_bfloat16 *__restrict__ g_residual)
{
    constexpr int D = Cfg::HIDDEN;
    constexpr float EPS = 1e-6f;
    int tid = threadIdx.x;
    int block_id = blockIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    __shared__ float s_reduce[NUM_WARPS];

    float ssq = 0.0f;
    for (int i = tid; i < D; i += BLOCK_SIZE) {
        float v = __bfloat162float(in[i]);
        out[i] = __float2bfloat16(v);     // staged copy
        ssq += v * v;
    }
    if (block_id == 0) {
        for (int i = tid; i < D; i += BLOCK_SIZE) g_residual[i] = out[i];
    }
    ssq = warp_reduce_sum_x(ssq);
    if (lane == 0) s_reduce[warp_id] = ssq;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < NUM_WARPS) ? s_reduce[lane] : 0.0f;
        v = warp_reduce_sum_x(v);
        if (lane == 0) s_reduce[0] = rsqrtf(v / float(D) + EPS);
    }
    __syncthreads();
    float rstd = s_reduce[0];
    for (int i = tid; i < D; i += BLOCK_SIZE) {
        float v = __bfloat162float(out[i]);
        float g = __bfloat162float(__ldg(gain + i));
        out[i] = __float2bfloat16(v * rstd * (1.0f + g));
    }
    __syncthreads();
}

}  // namespace lucebox::qwen3x
