/**
 * BF16 matvec primitives templated on Cfg.
 *
 *   matvec_bf16<Cfg>          -- generic warp-per-row GEMV
 *   matvec_gate_up_silu<Cfg>  -- fused gate+up+SiLU for MLP
 *   matvec_down_residual<Cfg> -- down-proj + residual add
 *   matvec_o_residual<Cfg>    -- O-proj + residual add
 *
 * All assume:
 *   - weight is row-major [out_dim, in_dim] bf16, in_dim divisible by 8
 *   - input is contiguous bf16 (or fp32 for the residual variants)
 *   - output is fp32 (or bf16 hidden for the residual variants)
 *   - block-level split: out_dim is sharded across blocks, rows across warps
 */
#pragma once

#include "Cfg.cuh"
#include "helpers.cuh"

namespace lucebox::qwen3x {

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

// Generic bf16 matvec.
template<typename Cfg>
__device__ void matvec_bf16(
    const __nv_bfloat16 *__restrict__ s_input,
    const __nv_bfloat16 *__restrict__ weight,
    float *__restrict__ output,
    int in_dim, int out_dim, int num_blocks)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int rs = block_id * rows_per_block;
    int re = (rs + rows_per_block < out_dim) ? rs + rows_per_block : out_dim;

    for (int m_base = rs; m_base < re; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < re) {
            const __nv_bfloat16 *w_row = weight + (size_t)m * in_dim;
            float sum = 0.0f;
            #pragma unroll 4
            for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint4 w_u4 = load_128bit(reinterpret_cast<const uint4 *>(w_row + k));
                sum += dot8_bf16(w_u4, s_input + k);
            }
            sum = warp_reduce_sum_x(sum);
            if (lane_id == 0) output[m] = sum;
        }
    }
}

// Fused gate+up+SiLU.
template<typename Cfg>
__device__ void matvec_gate_up_silu(
    const __nv_bfloat16 *__restrict__ s_input,
    const __nv_bfloat16 *__restrict__ gate_w,
    const __nv_bfloat16 *__restrict__ up_w,
    float *__restrict__ output,
    int in_dim, int out_dim, int num_blocks)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int rs = block_id * rows_per_block;
    int re = (rs + rows_per_block < out_dim) ? rs + rows_per_block : out_dim;

    for (int m_base = rs; m_base < re; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < re) {
            const __nv_bfloat16 *g_row = gate_w + (size_t)m * in_dim;
            const __nv_bfloat16 *u_row = up_w   + (size_t)m * in_dim;
            float gs = 0.0f, us = 0.0f;
            #pragma unroll 4
            for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint4 g_u4 = load_128bit(reinterpret_cast<const uint4 *>(g_row + k));
                uint4 u_u4 = load_128bit(reinterpret_cast<const uint4 *>(u_row + k));
                gs += dot8_bf16(g_u4, s_input + k);
                us += dot8_bf16(u_u4, s_input + k);
            }
            gs = warp_reduce_sum_x(gs);
            us = warp_reduce_sum_x(us);
            if (lane_id == 0) output[m] = fast_silu(gs) * us;
        }
    }
}

// Down-proj with residual add and bf16 output.
template<typename Cfg>
__device__ void matvec_down_residual(
    const float *__restrict__ s_input,
    const __nv_bfloat16 *__restrict__ weight,
    const __nv_bfloat16 *__restrict__ residual,
    __nv_bfloat16 *__restrict__ hidden_out,
    int in_dim, int out_dim, int num_blocks)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int rs = block_id * rows_per_block;
    int re = (rs + rows_per_block < out_dim) ? rs + rows_per_block : out_dim;

    for (int m_base = rs; m_base < re; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < re) {
            const __nv_bfloat16 *w_row = weight + (size_t)m * in_dim;
            float sum = 0.0f;
            for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint4 w_u4 = load_128bit(reinterpret_cast<const uint4 *>(w_row + k));
                const __nv_bfloat16 *w = reinterpret_cast<const __nv_bfloat16 *>(&w_u4);
                #pragma unroll
                for (int i = 0; i < 8; ++i) sum += __bfloat162float(w[i]) * s_input[k + i];
            }
            sum = warp_reduce_sum_x(sum);
            if (lane_id == 0)
                hidden_out[m] = __float2bfloat16(sum + __bfloat162float(residual[m]));
        }
    }
}

// O-proj with residual add (same structure, separate fn for clarity).
template<typename Cfg>
__device__ __forceinline__ void matvec_o_residual(
    const float *__restrict__ s_input,
    const __nv_bfloat16 *__restrict__ weight,
    const __nv_bfloat16 *__restrict__ residual,
    __nv_bfloat16 *__restrict__ hidden_out,
    int in_dim, int out_dim, int num_blocks)
{
    matvec_down_residual<Cfg>(s_input, weight, residual, hidden_out,
                               in_dim, out_dim, num_blocks);
}

}  // namespace lucebox::qwen3x
