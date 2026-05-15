/**
 * S7 - LM head argmax kernel templated on Cfg. Avoids the BF16->FP32
 * cast + torch matmul that the Python `_argmax_from_normalized` does
 * per decode step.
 *
 * Two-stage:
 *   1. lm_head_kernel<Cfg>: each block scans a row stripe of the
 *      [VOCAB, HIDDEN] BF16 weight, writes per-block (max_val, max_idx).
 *   2. lm_head_reduce_kernel: single block reduces over all blocks to
 *      pick the global max idx.
 *
 * Input:  g_normalized [Cfg::HIDDEN] fp32
 * Weight: lm_head_weight [Cfg::VOCAB_SIZE, Cfg::HIDDEN] bf16
 * Output: out_token_id  [1] int32
 *
 * No FP4 path yet; that's deferred to a later cuBLASLt-based variant.
 */
#include <cuda_runtime.h>
#include <stdint.h>

#include "Cfg.cuh"
#include "helpers.cuh"

namespace lucebox::qwen3x {

#ifndef LM_BLOCK_SIZE
#define LM_BLOCK_SIZE 256
#endif
constexpr int LM_WARP_SIZE = 32;
constexpr int LM_NUM_WARPS = LM_BLOCK_SIZE / LM_WARP_SIZE;

template<typename Cfg>
__global__ void lm_head_argmax_kernel(
    const float       *__restrict__ hidden,        // [HIDDEN] fp32
    const __nv_bfloat16 *__restrict__ weight,      // [VOCAB, HIDDEN] bf16
    float *__restrict__ block_max_vals,
    int   *__restrict__ block_max_idxs)
{
    constexpr int H = Cfg::HIDDEN;
    constexpr int V = Cfg::VOCAB_SIZE;
    __shared__ float s_hidden[Cfg::HIDDEN];
    for (int i = threadIdx.x; i < H; i += LM_BLOCK_SIZE) s_hidden[i] = hidden[i];
    __syncthreads();

    int warp_id = threadIdx.x / LM_WARP_SIZE;
    int lane_id = threadIdx.x % LM_WARP_SIZE;
    int rpb = (V + gridDim.x - 1) / gridDim.x;
    int rs = blockIdx.x * rpb;
    int re = min(rs + rpb, V);

    float local_max = -INFINITY;
    int   local_max_idx = -1;
    for (int m = rs + warp_id; m < re; m += LM_NUM_WARPS) {
        const __nv_bfloat16 *w_row = weight + (size_t)m * H;
        float sum = 0.0f;
        #pragma unroll 4
        for (int k = lane_id * 8; k < H; k += LM_WARP_SIZE * 8) {
            uint4 w_u4 = load_128bit(reinterpret_cast<const uint4 *>(w_row + k));
            const __nv_bfloat16 *wp = reinterpret_cast<const __nv_bfloat16 *>(&w_u4);
            #pragma unroll
            for (int i = 0; i < 8; ++i) sum += __bfloat162float(wp[i]) * s_hidden[k + i];
        }
        sum = warp_reduce_sum_x(sum);
        if (lane_id == 0 && sum > local_max) {
            local_max = sum; local_max_idx = m;
        }
    }
    local_max     = __shfl_sync(0xffffffff, local_max,     0);
    local_max_idx = __shfl_sync(0xffffffff, local_max_idx, 0);

    __shared__ float wm[LM_NUM_WARPS];
    __shared__ int   wi[LM_NUM_WARPS];
    if (lane_id == 0) { wm[warp_id] = local_max; wi[warp_id] = local_max_idx; }
    __syncthreads();
    if (warp_id == 0) {
        float mv = (lane_id < LM_NUM_WARPS) ? wm[lane_id] : -INFINITY;
        int   mi = (lane_id < LM_NUM_WARPS) ? wi[lane_id] : -1;
        #pragma unroll
        for (int o = LM_WARP_SIZE / 2; o > 0; o /= 2) {
            float ov = __shfl_down_sync(0xffffffff, mv, o);
            int   oi = __shfl_down_sync(0xffffffff, mi, o);
            if (ov > mv) { mv = ov; mi = oi; }
        }
        if (lane_id == 0) {
            block_max_vals[blockIdx.x] = mv;
            block_max_idxs[blockIdx.x] = mi;
        }
    }
}

__global__ void lm_head_argmax_reduce_kernel(
    const float *__restrict__ block_max_vals,
    const int   *__restrict__ block_max_idxs,
    int *__restrict__ output_token,
    int num_blocks)
{
    int tid = threadIdx.x;
    float bv = -INFINITY; int bi = -1;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        float v = block_max_vals[i];
        if (v > bv) { bv = v; bi = block_max_idxs[i]; }
    }
    __shared__ float sv[LM_BLOCK_SIZE];
    __shared__ int   si[LM_BLOCK_SIZE];
    sv[tid] = bv; si[tid] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sv[tid + s] > sv[tid]) {
            sv[tid] = sv[tid + s]; si[tid] = si[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) *output_token = si[0];
}

// Host launcher.
template<typename Cfg>
static cudaError_t launch_lm_head_argmax_impl(
    void *hidden, void *lm_head_weight,
    void *out_token_id,
    void *block_max_vals, void *block_max_idxs,
    int num_blocks,
    cudaStream_t stream)
{
    lm_head_argmax_kernel<Cfg><<<num_blocks, LM_BLOCK_SIZE, 0, stream>>>(
        (const float *)hidden,
        (const __nv_bfloat16 *)lm_head_weight,
        (float *)block_max_vals,
        (int *)block_max_idxs);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    lm_head_argmax_reduce_kernel<<<1, LM_BLOCK_SIZE, 0, stream>>>(
        (const float *)block_max_vals,
        (const int *)block_max_idxs,
        (int *)out_token_id,
        num_blocks);
    return cudaGetLastError();
}

extern "C" cudaError_t launch_lm_head_argmax_0p8b(
    void *hidden, void *lm_head_weight, void *out_token_id,
    void *block_max_vals, void *block_max_idxs,
    int num_blocks, cudaStream_t stream)
{
    return launch_lm_head_argmax_impl<Cfg_0p8B>(
        hidden, lm_head_weight, out_token_id,
        block_max_vals, block_max_idxs, num_blocks, stream);
}

extern "C" cudaError_t launch_lm_head_argmax_27b(
    void *hidden, void *lm_head_weight, void *out_token_id,
    void *block_max_vals, void *block_max_idxs,
    int num_blocks, cudaStream_t stream)
{
    return launch_lm_head_argmax_impl<Cfg_27B>(
        hidden, lm_head_weight, out_token_id,
        block_max_vals, block_max_idxs, num_blocks, stream);
}

}  // namespace lucebox::qwen3x
