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

// ---------------------------------------------------------------------------
// NVFP4 matvec primitives. Re-uses the same group-scaled FP4 format that
// the 0.8B path uses (FP4 E2M1 data + FP8 E4M3 per-group scales). Weight
// is stored as PackedMatrixNVFP4 = (data: uint8 [out_dim, in_dim/2],
// scales: __half [out_dim, in_dim/group_size]).
//
// S1 (per TODO.md): plug these into FA + DN + MLP layers to drop the 27B
// weight footprint from 50 GB BF16 to ~14 GB NVFP4. Pattern matches
// models/qwen35_0p8b/kernel_gb10_nvfp4.cu:matvec_nvfp4 but templated.
// ---------------------------------------------------------------------------

// FP4 E2M1 LUT: 8 positive magnitudes + 8 negatives.
//
// Constant-memory broadcast serializes when warp lanes access different
// indices — and in matvec each lane reads a different FP4 nibble, so we
// were paying 32 serial cycles per lookup. The fix is to load this
// 16-entry table into shared memory once per kernel; shared memory
// resolves divergent-index reads with only bank-conflict cost (16 banks
// for 16 elements → at worst 2-way conflict).
__device__ __constant__ float QWEN3X_FP4_E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
   -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
};

// Initialize a 16-float shared-memory LUT mirroring the constant LUT.
// Call once at the start of any block that will do NVFP4 matvecs.
__device__ __forceinline__ void load_fp4_lut_to_shmem(float *s_lut)
{
    if (threadIdx.x < 16) s_lut[threadIdx.x] = QWEN3X_FP4_E2M1_LUT[threadIdx.x];
    __syncthreads();
}

// Decode 4 FP4x2 bytes (32 bits = 8 FP4 values) into 8 dequantized floats,
// multiplied by `scale`, dotted against 8 bf16 activations. `lut` is a
// 16-entry shared-memory float table.
__device__ __forceinline__ float dot8_nvfp4_bf16(
    uint32_t packed, float scale, const __nv_bfloat16 *act,
    const float *__restrict__ lut)
{
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        unsigned int byte = (packed >> (i * 8)) & 0xff;
        sum += lut[byte & 0xf] * __bfloat162float(act[i * 2 + 0]);
        sum += lut[byte >> 4]  * __bfloat162float(act[i * 2 + 1]);
    }
    return sum * scale;
}

__device__ __forceinline__ float dot8_nvfp4_f32(
    uint32_t packed, float scale, const float *act,
    const float *__restrict__ lut)
{
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        unsigned int byte = (packed >> (i * 8)) & 0xff;
        sum += lut[byte & 0xf] * act[i * 2 + 0];
        sum += lut[byte >> 4]  * act[i * 2 + 1];
    }
    return sum * scale;
}

// Wide-load variants: take a uint4 (16 bytes = 32 FP4 values = one
// group at GROUP_SIZE=32) and one scale. ~4x fewer HBM transactions
// per warp than the uint32-load path.
__device__ __forceinline__ float dot32_nvfp4_bf16(
    uint4 packed4, float scale, const __nv_bfloat16 *act,
    const float *__restrict__ lut)
{
    float sum = 0.0f;
    uint32_t parts[4] = { packed4.x, packed4.y, packed4.z, packed4.w };
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        uint32_t packed = parts[g];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            unsigned int byte = (packed >> (i * 8)) & 0xff;
            sum += lut[byte & 0xf] * __bfloat162float(act[g * 8 + i * 2 + 0]);
            sum += lut[byte >> 4]  * __bfloat162float(act[g * 8 + i * 2 + 1]);
        }
    }
    return sum * scale;
}

__device__ __forceinline__ float dot32_nvfp4_f32(
    uint4 packed4, float scale, const float *act,
    const float *__restrict__ lut)
{
    float sum = 0.0f;
    uint32_t parts[4] = { packed4.x, packed4.y, packed4.z, packed4.w };
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        uint32_t packed = parts[g];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            unsigned int byte = (packed >> (i * 8)) & 0xff;
            sum += lut[byte & 0xf] * act[g * 8 + i * 2 + 0];
            sum += lut[byte >> 4]  * act[g * 8 + i * 2 + 1];
        }
    }
    return sum * scale;
}

// Packed NVFP4 weight matrix descriptor.
struct PackedMatrixNVFP4 {
    const uint8_t *data;          // [out_dim, in_dim / 2]   FP4x2 bytes
    const __half  *scales;        // [out_dim, in_dim / group_size]  FP16 scales
};

// NVFP4 matvec: out[m] = sum_k W[m, k] * input[k], W stored in NVFP4.
// Main loop uses uint4 (16-byte) loads at GROUP_SIZE=32 stride per
// lane (each lane handles 1 full group per iter). Tail loop handles
// any leftover < WARP_SIZE*32 elements via uint32 loads (e.g. 0.8B
// INTER=3584 has a 512-element tail past the 3*1024 main loop).
template<typename Cfg, int GROUP_SIZE = 32>
__device__ void matvec_nvfp4(
    const __nv_bfloat16 *__restrict__ s_input,
    PackedMatrixNVFP4 weight,
    float *__restrict__ output,
    int in_dim, int out_dim, int num_blocks)
{
    static_assert(GROUP_SIZE == 32, "matvec_nvfp4 requires GROUP_SIZE=32");
    __shared__ float s_fp4_lut[16];
    load_fp4_lut_to_shmem(s_fp4_lut);

    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int rs = block_id * rows_per_block;
    int re = (rs + rows_per_block < out_dim) ? rs + rows_per_block : out_dim;
    int row_bytes  = in_dim / 2;
    int row_scales = in_dim / GROUP_SIZE;
    int main_end = (in_dim / (WARP_SIZE * 32)) * (WARP_SIZE * 32);

    for (int m_base = rs; m_base < re; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < re) {
            const uint8_t *w_row = weight.data + (size_t)m * row_bytes;
            const __half *s_row  = weight.scales + (size_t)m * row_scales;
            float sum = 0.0f;
            #pragma unroll 2
            for (int k = lane_id * 32; k < main_end; k += WARP_SIZE * 32) {
                uint4 packed4 = load_128bit(
                    reinterpret_cast<const uint4 *>(w_row + (k / 2)));
                float scale = __half2float(__ldg(s_row + (k / GROUP_SIZE)));
                sum += dot32_nvfp4_bf16(packed4, scale, s_input + k, s_fp4_lut);
            }
            for (int k = main_end + lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint32_t packed = load_32bit(
                    reinterpret_cast<const uint32_t *>(w_row + (k / 2)));
                float scale = __half2float(__ldg(s_row + (k / GROUP_SIZE)));
                sum += dot8_nvfp4_bf16(packed, scale, s_input + k, s_fp4_lut);
            }
            sum = warp_reduce_sum_x(sum);
            if (lane_id == 0) output[m] = sum;
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

// ---------------------------------------------------------------------------
// NVFP4 fused / residual matvec variants. Mirror the BF16 trio so each layer
// only needs to swap the weight argument + matvec call. Group size pinned at
// 32 to match the packer in nvfp4_pack.py.
// ---------------------------------------------------------------------------

template<typename Cfg, int GROUP_SIZE = 32>
__device__ void matvec_gate_up_silu_nvfp4(
    const __nv_bfloat16 *__restrict__ s_input,
    PackedMatrixNVFP4 gate_w,
    PackedMatrixNVFP4 up_w,
    float *__restrict__ output,
    int in_dim, int out_dim, int num_blocks)
{
    __shared__ float s_fp4_lut[16];
    load_fp4_lut_to_shmem(s_fp4_lut);

    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int rs = block_id * rows_per_block;
    int re = (rs + rows_per_block < out_dim) ? rs + rows_per_block : out_dim;
    int row_bytes  = in_dim / 2;
    int row_scales = in_dim / GROUP_SIZE;
    int main_end = (in_dim / (WARP_SIZE * 32)) * (WARP_SIZE * 32);

    for (int m_base = rs; m_base < re; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < re) {
            const uint8_t *g_row = gate_w.data + (size_t)m * row_bytes;
            const __half  *gs_row = gate_w.scales + (size_t)m * row_scales;
            const uint8_t *u_row = up_w.data + (size_t)m * row_bytes;
            const __half  *us_row = up_w.scales + (size_t)m * row_scales;
            float gs = 0.0f, us = 0.0f;
            #pragma unroll 2
            for (int k = lane_id * 32; k < main_end; k += WARP_SIZE * 32) {
                uint4 gp = load_128bit(reinterpret_cast<const uint4 *>(g_row + (k / 2)));
                uint4 up = load_128bit(reinterpret_cast<const uint4 *>(u_row + (k / 2)));
                int si = k / GROUP_SIZE;
                float g_scale = __half2float(__ldg(gs_row + si));
                float u_scale = __half2float(__ldg(us_row + si));
                gs += dot32_nvfp4_bf16(gp, g_scale, s_input + k, s_fp4_lut);
                us += dot32_nvfp4_bf16(up, u_scale, s_input + k, s_fp4_lut);
            }
            for (int k = main_end + lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint32_t gp = load_32bit(reinterpret_cast<const uint32_t *>(g_row + (k / 2)));
                uint32_t up = load_32bit(reinterpret_cast<const uint32_t *>(u_row + (k / 2)));
                int si = k / GROUP_SIZE;
                float g_scale = __half2float(__ldg(gs_row + si));
                float u_scale = __half2float(__ldg(us_row + si));
                gs += dot8_nvfp4_bf16(gp, g_scale, s_input + k, s_fp4_lut);
                us += dot8_nvfp4_bf16(up, u_scale, s_input + k, s_fp4_lut);
            }
            gs = warp_reduce_sum_x(gs);
            us = warp_reduce_sum_x(us);
            if (lane_id == 0) output[m] = fast_silu(gs) * us;
        }
    }
}

template<typename Cfg, int GROUP_SIZE = 32>
__device__ void matvec_down_residual_nvfp4(
    const float *__restrict__ s_input,
    PackedMatrixNVFP4 weight,
    const __nv_bfloat16 *__restrict__ residual,
    __nv_bfloat16 *__restrict__ hidden_out,
    int in_dim, int out_dim, int num_blocks)
{
    __shared__ float s_fp4_lut[16];
    load_fp4_lut_to_shmem(s_fp4_lut);

    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int rs = block_id * rows_per_block;
    int re = (rs + rows_per_block < out_dim) ? rs + rows_per_block : out_dim;
    int row_bytes  = in_dim / 2;
    int row_scales = in_dim / GROUP_SIZE;
    int main_end = (in_dim / (WARP_SIZE * 32)) * (WARP_SIZE * 32);

    for (int m_base = rs; m_base < re; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < re) {
            const uint8_t *w_row = weight.data + (size_t)m * row_bytes;
            const __half  *s_row = weight.scales + (size_t)m * row_scales;
            float sum = 0.0f;
            #pragma unroll 2
            for (int k = lane_id * 32; k < main_end; k += WARP_SIZE * 32) {
                uint4 packed4 = load_128bit(
                    reinterpret_cast<const uint4 *>(w_row + (k / 2)));
                float scale = __half2float(__ldg(s_row + (k / GROUP_SIZE)));
                sum += dot32_nvfp4_f32(packed4, scale, s_input + k, s_fp4_lut);
            }
            for (int k = main_end + lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint32_t packed = load_32bit(
                    reinterpret_cast<const uint32_t *>(w_row + (k / 2)));
                float scale = __half2float(__ldg(s_row + (k / GROUP_SIZE)));
                sum += dot8_nvfp4_f32(packed, scale, s_input + k, s_fp4_lut);
            }
            sum = warp_reduce_sum_x(sum);
            if (lane_id == 0)
                hidden_out[m] = __float2bfloat16(sum + __bfloat162float(residual[m]));
        }
    }
}

template<typename Cfg, int GROUP_SIZE = 32>
__device__ __forceinline__ void matvec_o_residual_nvfp4(
    const float *__restrict__ s_input,
    PackedMatrixNVFP4 weight,
    const __nv_bfloat16 *__restrict__ residual,
    __nv_bfloat16 *__restrict__ hidden_out,
    int in_dim, int out_dim, int num_blocks)
{
    matvec_down_residual_nvfp4<Cfg, GROUP_SIZE>(s_input, weight, residual,
                                                  hidden_out, in_dim, out_dim, num_blocks);
}

}  // namespace lucebox::qwen3x
