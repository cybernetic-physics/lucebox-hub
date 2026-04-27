/**
 * Fused single-kernel decode for Qwen3.5-0.8B (hybrid DeltaNet + Full Attention).
 * ALL BF16: weights bf16, activations bf16, accumulation f32.
 * DeltaNet state: f32 (recurrence needs precision).
 *
 * Optimized for: NVIDIA RTX 3090 (sm_86, 82 SMs)
 * Model:         Qwen/Qwen3.5-0.8B (bf16 weights)
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <algorithm>
#include <cstdlib>

namespace cg = cooperative_groups;

// =============================================================================
// Model constants
// =============================================================================

constexpr int WARP_SIZE = 32;
constexpr int HIDDEN_SIZE = 1024;
constexpr int INTERMEDIATE_SIZE = 3584;
constexpr int NUM_LAYERS = 24;
constexpr float RMS_EPS = 1e-6f;
constexpr int VOCAB_SIZE = 248320;

// Full Attention
constexpr int FA_NUM_Q_HEADS = 8;
constexpr int FA_NUM_KV_HEADS = 2;
constexpr int FA_HEAD_DIM = 256;
constexpr int FA_GQA_RATIO = FA_NUM_Q_HEADS / FA_NUM_KV_HEADS;
constexpr int FA_Q_SIZE = FA_NUM_Q_HEADS * FA_HEAD_DIM;
constexpr int FA_GATE_SIZE = FA_Q_SIZE;
constexpr int FA_QPROJ_SIZE = FA_Q_SIZE + FA_GATE_SIZE;
constexpr int FA_KV_SIZE = FA_NUM_KV_HEADS * FA_HEAD_DIM;
constexpr int FA_ROTARY_DIM = 64;
constexpr float FA_ROPE_THETA = 10000000.0f;

// DeltaNet
constexpr int DN_NUM_HEADS = 16;
constexpr int DN_KEY_DIM = 128;
constexpr int DN_VALUE_DIM = 128;
constexpr int DN_CONV_KERNEL = 4;
constexpr int DN_QK_SIZE = DN_NUM_HEADS * DN_KEY_DIM;
constexpr int DN_V_SIZE = DN_NUM_HEADS * DN_VALUE_DIM;
constexpr int DN_CONV_CHANNELS = DN_QK_SIZE + DN_QK_SIZE + DN_V_SIZE;

constexpr int MAX_ACT_DIM = (HIDDEN_SIZE > INTERMEDIATE_SIZE) ? HIDDEN_SIZE : INTERMEDIATE_SIZE;

#ifndef NUM_BLOCKS
#define NUM_BLOCKS 82
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif

constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

#ifndef LM_NUM_BLOCKS
#define LM_NUM_BLOCKS 512
#endif
#ifndef LM_BLOCK_SIZE
#define LM_BLOCK_SIZE 256
#endif

__device__ __constant__ int LAYER_TYPE[NUM_LAYERS] = {
    0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1
};

// =============================================================================
// Weight structs — ALL BF16
// =============================================================================

struct FullAttnWeights {
    const __nv_bfloat16 *input_layernorm_weight;   // [1024]
    const __nv_bfloat16 *q_proj_weight;             // [4096, 1024]
    const __nv_bfloat16 *k_proj_weight;             // [512, 1024]
    const __nv_bfloat16 *v_proj_weight;             // [512, 1024]
    const __nv_bfloat16 *q_norm_weight;              // [256]
    const __nv_bfloat16 *k_norm_weight;              // [256]
    const __nv_bfloat16 *o_proj_weight;             // [1024, 2048]
    const __nv_bfloat16 *post_attn_layernorm_weight;
    const __nv_bfloat16 *gate_proj_weight;          // [3584, 1024]
    const __nv_bfloat16 *up_proj_weight;            // [3584, 1024]
    const __nv_bfloat16 *down_proj_weight;          // [1024, 3584]
};

struct DeltaNetWeights {
    const __nv_bfloat16 *input_layernorm_weight;
    const __nv_bfloat16 *qkv_proj_weight;           // [6144, 1024]
    const __nv_bfloat16 *z_proj_weight;             // [2048, 1024]
    const __nv_bfloat16 *beta_proj_weight;          // [16, 1024]
    const __nv_bfloat16 *alpha_proj_weight;         // [16, 1024]
    const __nv_bfloat16 *conv1d_weight;             // [6144, 1, 4]
    const __nv_bfloat16 *a_log;                     // [16]
    const __nv_bfloat16 *dt_bias;                   // [16]
    const __nv_bfloat16 *norm_weight;               // [128]
    const __nv_bfloat16 *out_proj_weight;           // [1024, 2048]
    const __nv_bfloat16 *post_attn_layernorm_weight;
    const __nv_bfloat16 *gate_proj_weight;
    const __nv_bfloat16 *up_proj_weight;
    const __nv_bfloat16 *down_proj_weight;
};

struct LayerWeights {
    int layer_type;
    int _pad[3];
    union {
        DeltaNetWeights dn;
        FullAttnWeights fa;
    };
};

// =============================================================================
// Grid-wide barrier (cooperative launch). The previous hand-rolled
// fence.acq_rel.gpu + atomicAdd barrier made no forward progress on sm_100
// (B200) when launched with a block count >= its SM count minus a few,
// presumably because L2 ordering differs on Blackwell datacenter. The NVFP4
// path (kernel_gb10_nvfp4.cu) already uses cg::this_grid().sync() and is
// happy on both GB10 and B200, so we use the same primitive here.
// =============================================================================

struct AtomicGridSync {
    __device__ __forceinline__ void sync() {
        cg::this_grid().sync();
    }
};

// =============================================================================
// Helpers
// =============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float fast_exp(float x) {
    float y; asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x * 1.44269504088896340736f)); return y;
}

__device__ __forceinline__ float fast_sigmoid(float x) {
    float y; asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(1.0f + fast_exp(-x))); return y;
}

__device__ __forceinline__ float fast_silu(float x) { return x * fast_sigmoid(x); }

__device__ __forceinline__ uint4 load_128bit(const uint4 *ptr) {
    uint4 out;
    asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w) : "l"(ptr));
    return out;
}

// BF16 dot product: 8 bf16 weights × 8 bf16 activations → f32
__device__ __forceinline__ float dot8_bf16(const uint4 &w_u4, const __nv_bfloat16 *act) {
    const __nv_bfloat16 *w = reinterpret_cast<const __nv_bfloat16 *>(&w_u4);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; i++)
        sum += __bfloat162float(w[i]) * __bfloat162float(act[i]);
    return sum;
}

// =============================================================================
// RMSNorm — reads bf16 input, writes bf16 output
// =============================================================================

__device__ void rmsnorm_redundant(
    const __nv_bfloat16 *__restrict__ input,
    const __nv_bfloat16 *__restrict__ weight,
    __nv_bfloat16 *__restrict__ s_out,        // shared memory bf16
    __nv_bfloat16 *__restrict__ g_residual)   // global bf16
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    __shared__ float smem_reduce[NUM_WARPS];

    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        float v = __bfloat162float(__ldg(input + i));
        s_out[i] = __float2bfloat16(v);
        local_sum_sq += v * v;
    }

    if (block_id == 0) {
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE)
            g_residual[i] = s_out[i];
    }

    local_sum_sq = warp_reduce_sum(local_sum_sq);
    if (lane_id == 0) smem_reduce[warp_id] = local_sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        float sum = (lane_id < NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane_id == 0)
            smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + RMS_EPS);
    }
    __syncthreads();

    float rstd = smem_reduce[0];
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        float w = __bfloat162float(__ldg(weight + i));
        float v = __bfloat162float(s_out[i]);
        s_out[i] = __float2bfloat16(v * rstd * (1.0f + w));
    }
    __syncthreads();
}

// RMSNorm from bf16 buffer (for post-attn norm)
__device__ void rmsnorm_from_bf16(
    const __nv_bfloat16 *__restrict__ input,
    const __nv_bfloat16 *__restrict__ weight,
    __nv_bfloat16 *__restrict__ s_out,
    __nv_bfloat16 *__restrict__ g_residual)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    __shared__ float smem_reduce[NUM_WARPS];

    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        float v = __bfloat162float(input[i]);
        s_out[i] = __float2bfloat16(v);
        local_sum_sq += v * v;
    }

    if (block_id == 0) {
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE)
            g_residual[i] = s_out[i];
    }

    local_sum_sq = warp_reduce_sum(local_sum_sq);
    if (lane_id == 0) smem_reduce[warp_id] = local_sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        float sum = (lane_id < NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane_id == 0)
            smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + RMS_EPS);
    }
    __syncthreads();

    float rstd = smem_reduce[0];
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        float w = __bfloat162float(__ldg(weight + i));
        float v = __bfloat162float(s_out[i]);
        s_out[i] = __float2bfloat16(v * rstd * (1.0f + w));
    }
    __syncthreads();
}

// =============================================================================
// BF16 Matvec: warp-per-row, activations in shared memory (bf16)
// =============================================================================

__device__ void matvec_bf16(
    const __nv_bfloat16 *__restrict__ s_input,  // shared memory bf16 [in_dim]
    const __nv_bfloat16 *__restrict__ weight,   // [out_dim, in_dim] bf16
    float *__restrict__ output,                  // [out_dim] f32 (accumulate in f32)
    int in_dim, int out_dim, int num_blocks)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, out_dim);

    for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < row_end) {
            const __nv_bfloat16 *w_row = weight + m * in_dim;
            float sum = 0.0f;
#pragma unroll 4
            for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint4 w_u4 = load_128bit(reinterpret_cast<const uint4 *>(w_row + k));
                sum += dot8_bf16(w_u4, s_input + k);
            }
            sum = warp_reduce_sum(sum);
            if (lane_id == 0) output[m] = sum;
        }
    }
}

// Fused gate+up+SiLU matvec (bf16 weights, bf16 activations)
__device__ void matvec_gate_up_silu_bf16(
    const __nv_bfloat16 *__restrict__ s_input,
    const __nv_bfloat16 *__restrict__ gate_weight,
    const __nv_bfloat16 *__restrict__ up_weight,
    float *__restrict__ output,
    int in_dim, int out_dim, int num_blocks)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, out_dim);

    for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < row_end) {
            const __nv_bfloat16 *g_row = gate_weight + m * in_dim;
            const __nv_bfloat16 *u_row = up_weight + m * in_dim;
            float gate_sum = 0.0f, up_sum = 0.0f;
#pragma unroll 4
            for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint4 g_u4 = load_128bit(reinterpret_cast<const uint4 *>(g_row + k));
                uint4 u_u4 = load_128bit(reinterpret_cast<const uint4 *>(u_row + k));
                gate_sum += dot8_bf16(g_u4, s_input + k);
                up_sum += dot8_bf16(u_u4, s_input + k);
            }
            gate_sum = warp_reduce_sum(gate_sum);
            up_sum = warp_reduce_sum(up_sum);
            if (lane_id == 0)
                output[m] = fast_silu(gate_sum) * up_sum;
        }
    }
}

// Down projection + residual → bf16 hidden
__device__ void matvec_down_residual_bf16(
    const float *__restrict__ s_input,           // shared [INTER] f32
    const __nv_bfloat16 *__restrict__ weight,    // [HIDDEN, INTER] bf16
    const __nv_bfloat16 *__restrict__ residual,  // [HIDDEN] bf16
    __nv_bfloat16 *__restrict__ hidden_out,      // [HIDDEN] bf16
    int in_dim, int out_dim, int num_blocks)
{
    // This needs f32 input (MLP intermediate is f32). Convert on the fly.
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, out_dim);

    for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < row_end) {
            const __nv_bfloat16 *w_row = weight + m * in_dim;
            float sum = 0.0f;
            // Weight is bf16, input is f32 — convert input to bf16 on the fly
            for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint4 w_u4 = load_128bit(reinterpret_cast<const uint4 *>(w_row + k));
                const __nv_bfloat16 *w = reinterpret_cast<const __nv_bfloat16 *>(&w_u4);
#pragma unroll
                for (int i = 0; i < 8; i++)
                    sum += __bfloat162float(w[i]) * s_input[k + i];
            }
            sum = warp_reduce_sum(sum);
            if (lane_id == 0)
                hidden_out[m] = __float2bfloat16(sum + __bfloat162float(residual[m]));
        }
    }
}

// O projection + residual → bf16
__device__ void matvec_o_residual_bf16(
    const float *__restrict__ s_input,           // shared [Q_SIZE] f32
    const __nv_bfloat16 *__restrict__ weight,
    const __nv_bfloat16 *__restrict__ residual,
    __nv_bfloat16 *__restrict__ hidden_out,
    int in_dim, int out_dim, int num_blocks)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, out_dim);

    for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < row_end) {
            const __nv_bfloat16 *w_row = weight + m * in_dim;
            float sum = 0.0f;
            for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint4 w_u4 = load_128bit(reinterpret_cast<const uint4 *>(w_row + k));
                const __nv_bfloat16 *w = reinterpret_cast<const __nv_bfloat16 *>(&w_u4);
#pragma unroll
                for (int i = 0; i < 8; i++)
                    sum += __bfloat162float(w[i]) * s_input[k + i];
            }
            sum = warp_reduce_sum(sum);
            if (lane_id == 0)
                hidden_out[m] = __float2bfloat16(sum + __bfloat162float(residual[m]));
        }
    }
}

// =============================================================================
// Full Attention layer (bf16)
// =============================================================================

__device__ void full_attention_layer(
    AtomicGridSync &grid,
    const FullAttnWeights &w,
    const __nv_bfloat16 *__restrict__ input,
    __nv_bfloat16 *__restrict__ k_cache,
    __nv_bfloat16 *__restrict__ v_cache,
    __nv_bfloat16 *__restrict__ g_residual,  // [HIDDEN] bf16
    float *__restrict__ g_activations,        // scratch f32
    float *__restrict__ g_q,                  // [FA_QPROJ_SIZE] f32
    float *__restrict__ g_kv,                 // [FA_KV_SIZE*2] f32
    float *__restrict__ g_attn_out,           // [FA_Q_SIZE] f32
    float *__restrict__ g_mlp_inter,          // [INTER] f32
    float *__restrict__ g_fa_partials,        // [num_blocks * FA_NUM_Q_HEADS * (FA_HEAD_DIM+2)]
    __nv_bfloat16 *__restrict__ hidden_out,   // [HIDDEN] bf16
    int position, int max_seq_len,
    __nv_bfloat16 *__restrict__ shmem)
{
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Phase 1: RMSNorm + QKV projection
    __nv_bfloat16 *s_norm = shmem;
    rmsnorm_redundant(input, w.input_layernorm_weight, s_norm, g_residual);

    matvec_bf16(s_norm, w.q_proj_weight, g_q, HIDDEN_SIZE, FA_QPROJ_SIZE, num_blocks);
    matvec_bf16(s_norm, w.k_proj_weight, g_kv, HIDDEN_SIZE, FA_KV_SIZE, num_blocks);
    matvec_bf16(s_norm, w.v_proj_weight, g_kv + FA_KV_SIZE, HIDDEN_SIZE, FA_KV_SIZE, num_blocks);
    grid.sync();

    // Phase 2: QK norm + partial RoPE + KV cache write
    if (block_id == 0) {
        float *k_buf = g_kv, *v_buf = g_kv + FA_KV_SIZE;
        for (int h = warp_id; h < FA_NUM_KV_HEADS; h += NUM_WARPS) {
            float *kh = k_buf + h * FA_HEAD_DIM, *vh = v_buf + h * FA_HEAD_DIM;
            __nv_bfloat16 *kc = k_cache + h * max_seq_len * FA_HEAD_DIM + position * FA_HEAD_DIM;
            __nv_bfloat16 *vc = v_cache + h * max_seq_len * FA_HEAD_DIM + position * FA_HEAD_DIM;
            float ss = 0; for (int i = lane_id; i < FA_HEAD_DIM; i += WARP_SIZE) ss += kh[i]*kh[i];
            ss = warp_reduce_sum(ss); float sc = rsqrtf(ss / float(FA_HEAD_DIM) + RMS_EPS);
            sc = __shfl_sync(0xffffffff, sc, 0);
            for (int i = lane_id; i < FA_HEAD_DIM; i += WARP_SIZE) {
                float normed = kh[i] * sc * (1.0f + __bfloat162float(__ldg(w.k_norm_weight + i)));
                if (i < FA_ROTARY_DIM) {
                    float fe = float(2*(i%(FA_ROTARY_DIM/2))) / float(FA_ROTARY_DIM);
                    float freq = float(position) / powf(FA_ROPE_THETA, fe);
                    float cv = cosf(freq), sv = sinf(freq);
                    int p = (i < FA_ROTARY_DIM/2) ? i+FA_ROTARY_DIM/2 : i-FA_ROTARY_DIM/2;
                    float pv = kh[p]*sc*(1.0f+__bfloat162float(__ldg(w.k_norm_weight+p)));
                    float rotated = (i < FA_ROTARY_DIM/2) ? (normed*cv - pv*sv) : (pv*sv + normed*cv);
                    kc[i] = __float2bfloat16(rotated);
                } else { kc[i] = __float2bfloat16(normed); }
                vc[i] = __float2bfloat16(vh[i]);
            }
        }
    }
    // Q norm + RoPE (all blocks)
    {
        int hpb = (FA_NUM_Q_HEADS + num_blocks - 1) / num_blocks;
        int hs = block_id * hpb, he = min(hs + hpb, FA_NUM_Q_HEADS);
        for (int qh = hs; qh < he; qh++) {
            float *qh_ptr = g_q + qh * FA_HEAD_DIM * 2;
            if (warp_id == 0) {
                float ss = 0; for (int i = lane_id; i < FA_HEAD_DIM; i += WARP_SIZE) ss += qh_ptr[i]*qh_ptr[i];
                ss = warp_reduce_sum(ss); float sc = rsqrtf(ss / float(FA_HEAD_DIM) + RMS_EPS);
                sc = __shfl_sync(0xffffffff, sc, 0);
                for (int i = lane_id; i < FA_HEAD_DIM; i += WARP_SIZE) {
                    float normed = qh_ptr[i]*sc*(1.0f+__bfloat162float(__ldg(w.q_norm_weight+i)));
                    if (i < FA_ROTARY_DIM) {
                        float fe = float(2*(i%(FA_ROTARY_DIM/2))) / float(FA_ROTARY_DIM);
                        float freq = float(position) / powf(FA_ROPE_THETA, fe);
                        float cv = cosf(freq), sv = sinf(freq);
                        int p = (i < FA_ROTARY_DIM/2) ? i+FA_ROTARY_DIM/2 : i-FA_ROTARY_DIM/2;
                        float pv = qh_ptr[p]*sc*(1.0f+__bfloat162float(__ldg(w.q_norm_weight+p)));
                        qh_ptr[i] = (i < FA_ROTARY_DIM/2) ? (normed*cv-pv*sv) : (pv*sv+normed*cv);
                    } else { qh_ptr[i] = normed; }
                }
            }
        }
    }
    grid.sync();

    // Phase 3: Split-K attention decode (online softmax + sigmoid gate).
    //
    // Old design: one block per query head; the rest of the SMs sat idle
    // during the FA scan. At long S that capped decode hard — only 8 of
    // 148 SMs busy.
    //
    // New design: each block handles one (query head, K-split) pair.
    //   block_id mod FA_NUM_Q_HEADS  -> my_qh
    //   block_id div FA_NUM_Q_HEADS  -> my_split
    // Active blocks fill the grid up to FA_NUM_Q_HEADS * num_splits; the
    // tail of blocks (if any) idle this phase but are needed for other
    // phases. Each active block computes a partial (max, sum_exp,
    // out_acc) over its slice of K positions, writes to a global
    // partials buffer, and grid-syncs. After the sync, blocks 0..7 each
    // reduce all splits for their query head into the final output.
    {
        int cache_len = position + 1;
        float attn_scale = 1.0f / sqrtf(float(FA_HEAD_DIM));
        constexpr int EPL = FA_HEAD_DIM / WARP_SIZE;
        constexpr int PARTIAL_STRIDE = FA_HEAD_DIM + 2;     // out + max + sum

        int num_splits = num_blocks / FA_NUM_Q_HEADS;
        if (num_splits < 1) num_splits = 1;
        int my_qh    = block_id % FA_NUM_Q_HEADS;
        int my_split = block_id / FA_NUM_Q_HEADS;
        bool active = (block_id < FA_NUM_Q_HEADS * num_splits);

        // Per-block warp-combine scratch lives in shared memory so it's
        // private to the block (no cross-block race).
        __shared__ float s_max[NUM_WARPS];
        __shared__ float s_sum[NUM_WARPS];
        __shared__ float s_out[NUM_WARPS * FA_HEAD_DIM];

        if (active) {
            int positions_per_split = (cache_len + num_splits - 1) / num_splits;
            int pos_start = my_split * positions_per_split;
            int pos_end   = min(pos_start + positions_per_split, cache_len);
            int kvh = my_qh / FA_GQA_RATIO;
            const float *q_head = g_q + my_qh * FA_HEAD_DIM * 2;

            float q_local[EPL];
            for (int e = 0; e < EPL; e++) q_local[e] = q_head[lane_id*EPL+e];

            float partial_max = -INFINITY, partial_sum = 0;
            float partial_acc[EPL];
            for (int e = 0; e < EPL; e++) partial_acc[e] = 0;

            for (int pos = pos_start + warp_id; pos < pos_end; pos += NUM_WARPS) {
                const __nv_bfloat16 *k_pos = k_cache + kvh*max_seq_len*FA_HEAD_DIM + pos*FA_HEAD_DIM;
                const __nv_bfloat16 *v_pos = v_cache + kvh*max_seq_len*FA_HEAD_DIM + pos*FA_HEAD_DIM;
                float score = 0;
                for (int e = 0; e < EPL; e++) score += q_local[e] * __bfloat162float(__ldg(k_pos + lane_id*EPL+e));
                score = warp_reduce_sum(score) * attn_scale;
                score = __shfl_sync(0xffffffff, score, 0);
                float old_max = partial_max; partial_max = fmaxf(partial_max, score);
                float exp_diff = fast_exp(old_max - partial_max);
                partial_sum = partial_sum * exp_diff + fast_exp(score - partial_max);
                float wt = fast_exp(score - partial_max);
                for (int e = 0; e < EPL; e++)
                    partial_acc[e] = partial_acc[e]*exp_diff + wt*__bfloat162float(__ldg(v_pos + lane_id*EPL+e));
            }
            if (lane_id == 0) { s_max[warp_id] = partial_max; s_sum[warp_id] = partial_sum; }
            for (int e = 0; e < EPL; e++) s_out[warp_id*FA_HEAD_DIM + lane_id*EPL+e] = partial_acc[e];
            __syncthreads();

            // Warp 0 combines warp-partials inside the block, then writes
            // to the global per-(split, qh) partials slot.
            if (warp_id == 0) {
                float bm = -INFINITY;
                for (int ww = 0; ww < NUM_WARPS; ww++) if (s_max[ww] > bm) bm = s_max[ww];
                float bs = 0;
                float bo[EPL]; for (int e = 0; e < EPL; e++) bo[e] = 0;
                for (int ww = 0; ww < NUM_WARPS; ww++) {
                    if (s_max[ww] > -INFINITY) {
                        float scale = fast_exp(s_max[ww] - bm);
                        bs += s_sum[ww] * scale;
                        for (int e = 0; e < EPL; e++) bo[e] += s_out[ww*FA_HEAD_DIM + lane_id*EPL+e] * scale;
                    }
                }
                float *p = g_fa_partials + (my_split * FA_NUM_Q_HEADS + my_qh) * PARTIAL_STRIDE;
                for (int e = 0; e < EPL; e++) p[lane_id*EPL+e] = bo[e];
                if (lane_id == 0) { p[FA_HEAD_DIM] = bm; p[FA_HEAD_DIM+1] = bs; }
            }
        }
    }
    grid.sync();

    // Reduce phase: blocks 0..FA_NUM_Q_HEADS-1 each combine all splits
    // for their query head into the final attention output.
    {
        constexpr int EPL = FA_HEAD_DIM / WARP_SIZE;
        constexpr int PARTIAL_STRIDE = FA_HEAD_DIM + 2;
        int num_splits = num_blocks / FA_NUM_Q_HEADS;
        if (num_splits < 1) num_splits = 1;

        if (block_id < FA_NUM_Q_HEADS && warp_id == 0) {
            int qh = block_id;
            float *q_head = g_q + qh * FA_HEAD_DIM * 2;
            float *gate_ptr = q_head + FA_HEAD_DIM;
            float *out_head = g_attn_out + qh * FA_HEAD_DIM;

            float global_max = -INFINITY;
            for (int s = 0; s < num_splits; s++) {
                float bm = g_fa_partials[(s * FA_NUM_Q_HEADS + qh) * PARTIAL_STRIDE + FA_HEAD_DIM];
                if (bm > global_max) global_max = bm;
            }
            float total_sum = 0;
            float total_out[EPL];
            for (int e = 0; e < EPL; e++) total_out[e] = 0;
            for (int s = 0; s < num_splits; s++) {
                float *p = g_fa_partials + (s * FA_NUM_Q_HEADS + qh) * PARTIAL_STRIDE;
                float bm = p[FA_HEAD_DIM];
                if (bm > -INFINITY) {
                    float scale = fast_exp(bm - global_max);
                    total_sum += p[FA_HEAD_DIM+1] * scale;
                    for (int e = 0; e < EPL; e++) total_out[e] += p[lane_id*EPL+e] * scale;
                }
            }
            float rcp = 1.0f / total_sum;
            for (int e = 0; e < EPL; e++) {
                int idx = lane_id*EPL+e;
                out_head[idx] = total_out[e] * rcp * fast_sigmoid(gate_ptr[idx]);
            }
        }
    }
    grid.sync();

    // Phase 4: O projection + residual → bf16
    {
        float *s_attn = reinterpret_cast<float *>(shmem);
        for (int i = threadIdx.x; i < FA_Q_SIZE; i += BLOCK_SIZE) s_attn[i] = g_attn_out[i];
        __syncthreads();
        matvec_o_residual_bf16(s_attn, w.o_proj_weight, g_residual, hidden_out, FA_Q_SIZE, HIDDEN_SIZE, num_blocks);
    }
    grid.sync();

    // Phase 5: Post-attn norm + MLP
    __nv_bfloat16 *s_act = shmem;
    rmsnorm_from_bf16(hidden_out, w.post_attn_layernorm_weight, s_act, g_residual);

    matvec_gate_up_silu_bf16(s_act, w.gate_proj_weight, w.up_proj_weight,
                              g_mlp_inter, HIDDEN_SIZE, INTERMEDIATE_SIZE, num_blocks);
    grid.sync();

    // Load MLP intermediate to shared (f32)
    float *s_mlp = reinterpret_cast<float *>(shmem);
    for (int i = threadIdx.x; i < INTERMEDIATE_SIZE; i += BLOCK_SIZE) s_mlp[i] = g_mlp_inter[i];
    __syncthreads();

    matvec_down_residual_bf16(s_mlp, w.down_proj_weight, g_residual, hidden_out,
                               INTERMEDIATE_SIZE, HIDDEN_SIZE, num_blocks);
    grid.sync();
}

// =============================================================================
// DeltaNet layer (bf16) — warp-cooperative state-in-registers recurrence
// =============================================================================

__device__ void deltanet_layer(
    AtomicGridSync &grid,
    const DeltaNetWeights &w,
    const __nv_bfloat16 *__restrict__ input,
    __nv_bfloat16 *__restrict__ g_residual,
    float *__restrict__ g_activations,
    float *__restrict__ g_qkv,
    float *__restrict__ g_z,
    float *__restrict__ g_beta,
    float *__restrict__ g_alpha,
    float *__restrict__ g_dn_out,
    float *__restrict__ g_mlp_inter,
    float *__restrict__ dn_state,     // [DN_NUM_HEADS, DN_KEY, DN_VAL] f32
    float *__restrict__ conv_buf,     // [DN_CONV_CH, DN_CONV_K] f32
    __nv_bfloat16 *__restrict__ hidden_out,
    int dn_layer_idx,
    __nv_bfloat16 *__restrict__ shmem)
{
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Phase 1: RMSNorm + projections
    __nv_bfloat16 *s_norm = shmem;
    rmsnorm_redundant(input, w.input_layernorm_weight, s_norm, g_residual);

    matvec_bf16(s_norm, w.qkv_proj_weight, g_qkv, HIDDEN_SIZE, DN_CONV_CHANNELS, num_blocks);
    matvec_bf16(s_norm, w.z_proj_weight, g_z, HIDDEN_SIZE, DN_V_SIZE, num_blocks);
    matvec_bf16(s_norm, w.beta_proj_weight, g_beta, HIDDEN_SIZE, DN_NUM_HEADS, num_blocks);
    matvec_bf16(s_norm, w.alpha_proj_weight, g_alpha, HIDDEN_SIZE, DN_NUM_HEADS, num_blocks);
    grid.sync();

    // Phase 2+3: Conv1d + recurrence (blocks 0-15 only)
    if (block_id < DN_NUM_HEADS) {
        int h = block_id;
        float *layer_conv = conv_buf + dn_layer_idx * DN_CONV_CHANNELS * DN_CONV_KERNEL;

        // Conv1d + SiLU
        __shared__ float s_q[DN_KEY_DIM], s_k[DN_KEY_DIM], s_v[DN_VALUE_DIM];
        int head_ch[3] = {h*DN_KEY_DIM, DN_QK_SIZE+h*DN_KEY_DIM, 2*DN_QK_SIZE+h*DN_VALUE_DIM};
        for (int region = 0; region < 3; region++) {
            int ch_base = head_ch[region], ch_count = (region < 2) ? DN_KEY_DIM : DN_VALUE_DIM;
            float *dst = (region == 0) ? s_q : (region == 1) ? s_k : s_v;
            for (int c = threadIdx.x; c < ch_count; c += BLOCK_SIZE) {
                int ch = ch_base + c;
                float h0=layer_conv[ch*DN_CONV_KERNEL+1], h1=layer_conv[ch*DN_CONV_KERNEL+2], h2=layer_conv[ch*DN_CONV_KERNEL+3];
                layer_conv[ch*DN_CONV_KERNEL]=h0; layer_conv[ch*DN_CONV_KERNEL+1]=h1;
                layer_conv[ch*DN_CONV_KERNEL+2]=h2; layer_conv[ch*DN_CONV_KERNEL+3]=g_qkv[ch];
                float co = 0;
                for (int t = 0; t < DN_CONV_KERNEL; t++)
                    co += layer_conv[ch*DN_CONV_KERNEL+t] * __bfloat162float(__ldg(w.conv1d_weight + ch*DN_CONV_KERNEL+t));
                dst[c] = fast_silu(co);
            }
        }

        // Beta/alpha activations
        if (threadIdx.x == 0) {
            g_beta[h] = fast_sigmoid(g_beta[h]);
            float a_log_val = __bfloat162float(__ldg(w.a_log + h));
            float dt_b = __bfloat162float(__ldg(w.dt_bias + h));
            float x = g_alpha[h] + dt_b;
            float sp = (x > 20.0f) ? x : logf(1.0f + fast_exp(x));
            g_alpha[h] = fast_exp(-fast_exp(a_log_val) * sp);
        }
        __syncthreads();

        // L2 normalize Q, K
        constexpr float Q_SCALE = 1.0f / 11.313708498984761f;
        if (warp_id == 0) {
            float sq = 0; for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) sq += s_q[i]*s_q[i];
            sq = warp_reduce_sum(sq); float n = rsqrtf(sq+1e-6f)*Q_SCALE;
            n = __shfl_sync(0xffffffff,n,0); for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) s_q[i] *= n;
        }
        if (warp_id == 1) {
            float sq = 0; for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) sq += s_k[i]*s_k[i];
            sq = warp_reduce_sum(sq); float n = rsqrtf(sq+1e-6f);
            n = __shfl_sync(0xffffffff,n,0); for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) s_k[i] *= n;
        }
        __syncthreads();

        float decay = g_alpha[h], beta = g_beta[h];

        // k·q dot
        __shared__ float s_kq;
        if (warp_id == 0) {
            float kq = 0; for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) kq += s_k[i]*s_q[i];
            kq = warp_reduce_sum(kq); if (lane_id == 0) s_kq = kq;
        }
        __syncthreads();
        float kq = s_kq;

        // Warp-cooperative recurrence (state in global memory — decode is 1 token, fine)
        float *state = dn_state + h * DN_KEY_DIM * DN_VALUE_DIM;
        float *out_head = g_dn_out + h * DN_VALUE_DIM;

        constexpr int J_PER_WARP = DN_VALUE_DIM / NUM_WARPS;
        constexpr int I_PER_LANE = DN_KEY_DIM / WARP_SIZE;

#pragma unroll
        for (int jj = 0; jj < J_PER_WARP; jj++) {
            int j = warp_id * J_PER_WARP + jj;
            float s_regs[I_PER_LANE], stk = 0, sqv = 0;
#pragma unroll
            for (int ii = 0; ii < I_PER_LANE; ii++) {
                int i = lane_id + ii * WARP_SIZE;
                float sv = state[j*DN_KEY_DIM+i]; s_regs[ii] = sv;
                stk += sv * s_k[i]; sqv += sv * s_q[i];
            }
            stk = warp_reduce_sum(stk); sqv = warp_reduce_sum(sqv);
            stk = __shfl_sync(0xffffffff,stk,0); sqv = __shfl_sync(0xffffffff,sqv,0);
            float error_j = (s_v[j] - stk) * beta;
            float o_j = decay * sqv + error_j * kq;
            if (lane_id == 0) out_head[j] = o_j;
#pragma unroll
            for (int ii = 0; ii < I_PER_LANE; ii++) {
                int i = lane_id + ii * WARP_SIZE;
                state[j*DN_KEY_DIM+i] = s_regs[ii] * decay + s_k[i] * error_j;
            }
        }

        // Gated RMSNorm
        __syncthreads();
        {
            __shared__ float smem_gnorm[NUM_WARPS];
            float sq = 0; for (int i = threadIdx.x; i < DN_VALUE_DIM; i += BLOCK_SIZE) sq += out_head[i]*out_head[i];
            sq = warp_reduce_sum(sq); if (lane_id == 0) smem_gnorm[warp_id] = sq; __syncthreads();
            if (warp_id == 0) { float v = (lane_id < NUM_WARPS) ? smem_gnorm[lane_id] : 0; v = warp_reduce_sum(v); if (lane_id == 0) smem_gnorm[0] = rsqrtf(v/DN_VALUE_DIM + RMS_EPS); }
            __syncthreads(); float rstd = smem_gnorm[0];
            for (int i = threadIdx.x; i < DN_VALUE_DIM; i += BLOCK_SIZE) {
                float normed = out_head[i] * rstd * __bfloat162float(__ldg(w.norm_weight + i));
                float gate = fast_silu(g_z[h*DN_VALUE_DIM+i]);
                out_head[i] = normed * gate;
            }
        }
    } else {
        // Idle blocks: could prefetch weights
    }
    grid.sync();

    // Phase 4: Out projection + residual → bf16
    {
        float *s_dn = reinterpret_cast<float *>(shmem);
        for (int i = threadIdx.x; i < DN_V_SIZE; i += BLOCK_SIZE) s_dn[i] = g_dn_out[i];
        __syncthreads();
        matvec_o_residual_bf16(s_dn, w.out_proj_weight, g_residual, hidden_out, DN_V_SIZE, HIDDEN_SIZE, num_blocks);
    }
    grid.sync();

    // Phase 5: Post-attn norm + MLP
    __nv_bfloat16 *s_act = shmem;
    rmsnorm_from_bf16(hidden_out, w.post_attn_layernorm_weight, s_act, g_residual);

    matvec_gate_up_silu_bf16(s_act, w.gate_proj_weight, w.up_proj_weight,
                              g_mlp_inter, HIDDEN_SIZE, INTERMEDIATE_SIZE, num_blocks);
    grid.sync();

    float *s_mlp = reinterpret_cast<float *>(shmem);
    for (int i = threadIdx.x; i < INTERMEDIATE_SIZE; i += BLOCK_SIZE) s_mlp[i] = g_mlp_inter[i];
    __syncthreads();
    matvec_down_residual_bf16(s_mlp, w.down_proj_weight, g_residual, hidden_out,
                               INTERMEDIATE_SIZE, HIDDEN_SIZE, num_blocks);
    grid.sync();
}

// =============================================================================
// LM Head: vocab projection + argmax
// =============================================================================

__global__ void lm_head_kernel(
    const float *__restrict__ hidden,
    const __nv_bfloat16 *__restrict__ weight,   // [VOCAB, HIDDEN] bf16
    float *__restrict__ block_max_vals,
    int *__restrict__ block_max_idxs)
{
    __shared__ float s_hidden[HIDDEN_SIZE];
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LM_BLOCK_SIZE) s_hidden[i] = hidden[i];
    __syncthreads();

    int warp_id = threadIdx.x / WARP_SIZE, lane_id = threadIdx.x % WARP_SIZE;
    int num_warps = LM_BLOCK_SIZE / WARP_SIZE;
    int rpb = (VOCAB_SIZE + gridDim.x - 1) / gridDim.x;
    int rs = blockIdx.x * rpb, re = min(rs + rpb, VOCAB_SIZE);

    float local_max = -INFINITY; int local_max_idx = -1;
    for (int m = rs + warp_id; m < re; m += num_warps) {
        const __nv_bfloat16 *w_row = weight + m * HIDDEN_SIZE;
        float sum = 0;
#pragma unroll 4
        for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
            uint4 w_u4 = load_128bit(reinterpret_cast<const uint4 *>(w_row + k));
            const __nv_bfloat16 *wp = reinterpret_cast<const __nv_bfloat16 *>(&w_u4);
            for (int i = 0; i < 8; i++) sum += __bfloat162float(wp[i]) * s_hidden[k+i];
        }
        sum = warp_reduce_sum(sum);
        if (lane_id == 0 && sum > local_max) { local_max = sum; local_max_idx = m; }
    }
    local_max = __shfl_sync(0xffffffff, local_max, 0);
    local_max_idx = __shfl_sync(0xffffffff, local_max_idx, 0);

    __shared__ float wm[32]; __shared__ int wi[32];
    if (lane_id == 0) { wm[warp_id] = local_max; wi[warp_id] = local_max_idx; }
    __syncthreads();
    if (warp_id == 0) {
        float mv = (lane_id < num_warps) ? wm[lane_id] : -INFINITY;
        int mi = (lane_id < num_warps) ? wi[lane_id] : -1;
        for (int o = WARP_SIZE/2; o > 0; o /= 2) {
            float ov = __shfl_down_sync(0xffffffff, mv, o);
            int oi = __shfl_down_sync(0xffffffff, mi, o);
            if (ov > mv) { mv = ov; mi = oi; }
        }
        if (lane_id == 0) { block_max_vals[blockIdx.x] = mv; block_max_idxs[blockIdx.x] = mi; }
    }
}

__global__ void lm_head_reduce_kernel_bf16(
    const float *__restrict__ block_max_vals,
    const int *__restrict__ block_max_idxs,
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
    __shared__ int si[LM_BLOCK_SIZE];
    sv[tid] = bv; si[tid] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sv[tid + s] > sv[tid]) { sv[tid] = sv[tid + s]; si[tid] = si[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) *output_token = si[0];
}

// =============================================================================
// Main decode kernel
// =============================================================================

__global__ void __launch_bounds__(BLOCK_SIZE, 1)
decode_kernel(
    const __nv_bfloat16 *__restrict__ embed_weight,
    const __nv_bfloat16 *__restrict__ final_norm_weight,
    const __nv_bfloat16 *__restrict__ lm_head_weight,
    const LayerWeights *__restrict__ layer_weights,
    __nv_bfloat16 *__restrict__ fa_k_cache,
    __nv_bfloat16 *__restrict__ fa_v_cache,
    float *__restrict__ dn_states,
    float *__restrict__ conv_bufs,
    __nv_bfloat16 *__restrict__ hidden_buffer,
    float *__restrict__ g_activations,
    __nv_bfloat16 *__restrict__ g_residual,
    float *__restrict__ g_qkv_scratch,
    float *__restrict__ g_kv_scratch,
    float *__restrict__ g_attn_out,
    float *__restrict__ g_mlp_inter,
    float *__restrict__ g_z_scratch,
    float *__restrict__ g_beta_scratch,
    float *__restrict__ g_alpha_scratch,
    float *__restrict__ g_normalized,
    float *__restrict__ g_fa_partials,
    unsigned int *__restrict__ barrier_counter,
    unsigned int *__restrict__ barrier_generation,
    int input_token_id, int position, int max_seq_len)
{
    int block_id = blockIdx.x;
    // barrier_counter / barrier_generation are kept on the arg list for ABI
    // compatibility but unused now that we sync via cg::this_grid().
    (void)barrier_counter;
    (void)barrier_generation;

    AtomicGridSync grid{};

    // Shared memory: large enough for max(HIDDEN_SIZE bf16, INTERMEDIATE_SIZE f32)
    __shared__ __align__(16) char shmem_raw[MAX_ACT_DIM * sizeof(float)];
    __nv_bfloat16 *shmem_bf16 = reinterpret_cast<__nv_bfloat16 *>(shmem_raw);

    const __nv_bfloat16 *embed_row = embed_weight + input_token_id * HIDDEN_SIZE;

    int fa_kv_stride = FA_NUM_KV_HEADS * max_seq_len * FA_HEAD_DIM;
    int dn_state_stride = DN_NUM_HEADS * DN_KEY_DIM * DN_VALUE_DIM;

    int dn_layer_idx = 0, fa_layer_idx = 0;

    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        const __nv_bfloat16 *layer_input = (layer == 0) ? embed_row : hidden_buffer;

        if (LAYER_TYPE[layer] == 0) {
            deltanet_layer(
                grid, layer_weights[layer].dn, layer_input,
                g_residual, g_activations, g_qkv_scratch, g_z_scratch,
                g_beta_scratch, g_alpha_scratch, g_attn_out, g_mlp_inter,
                dn_states + dn_layer_idx * dn_state_stride,
                conv_bufs, hidden_buffer, dn_layer_idx, shmem_bf16);
            dn_layer_idx++;
        } else {
            full_attention_layer(
                grid, layer_weights[layer].fa, layer_input,
                fa_k_cache + fa_layer_idx * fa_kv_stride,
                fa_v_cache + fa_layer_idx * fa_kv_stride,
                g_residual, g_activations, g_qkv_scratch, g_kv_scratch,
                g_attn_out, g_mlp_inter, g_fa_partials, hidden_buffer,
                position, max_seq_len, shmem_bf16);
            fa_layer_idx++;
        }
    }

    // Final RMSNorm (block 0 only)
    if (block_id == 0) {
        __shared__ float smem_reduce[NUM_WARPS];
        int warp_id = threadIdx.x / WARP_SIZE, lane_id = threadIdx.x % WARP_SIZE;
        float local_sum_sq = 0;
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
            float v = __bfloat162float(hidden_buffer[i]); g_activations[i] = v; local_sum_sq += v*v;
        }
        local_sum_sq = warp_reduce_sum(local_sum_sq);
        if (lane_id == 0) smem_reduce[warp_id] = local_sum_sq; __syncthreads();
        if (warp_id == 0) { float sum = (lane_id < NUM_WARPS) ? smem_reduce[lane_id] : 0; sum = warp_reduce_sum(sum); if (lane_id == 0) smem_reduce[0] = rsqrtf(sum/HIDDEN_SIZE + RMS_EPS); }
        __syncthreads(); float rstd = smem_reduce[0];
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
            float wt = __bfloat162float(__ldg(final_norm_weight + i));
            g_normalized[i] = g_activations[i] * rstd * (1.0f + wt);
        }
    }
}

// =============================================================================
// C entry point
// =============================================================================

extern "C" void launch_decode(
    int input_token_id, int *output_token_id,
    const void *embed_weight, const LayerWeights *layer_weights,
    const void *final_norm_weight,
    const void *lm_head_weight,
    void *fa_k_cache, void *fa_v_cache,
    void *dn_states, void *conv_bufs,
    void *hidden_buffer, void *g_activations, void *g_residual,
    void *g_qkv_scratch, void *g_kv_scratch, void *g_attn_out,
    void *g_mlp_inter, void *g_z_scratch, void *g_beta_scratch,
    void *g_alpha_scratch, void *g_normalized,
    unsigned int *barrier_counter, unsigned int *barrier_generation,
    float *block_max_vals, int *block_max_idxs,
    unsigned int *lm_sync_counter,
    int position, int max_seq_len, cudaStream_t stream)
{
    static int cached_decode_blocks = 0;
    static int cached_lm_blocks = 0;
    if (cached_decode_blocks == 0) {
        if (const char *override_blocks = std::getenv("MEGAKERNEL_DECODE_BLOCKS")) {
            int v = std::atoi(override_blocks);
            if (v > 0) cached_decode_blocks = v;
        }
        if (cached_decode_blocks == 0) {
            int device = 0;
            cudaGetDevice(&device);
            cudaDeviceProp prop{};
            cudaGetDeviceProperties(&prop, device);
            int active = 0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active, decode_kernel, BLOCK_SIZE, 0);
            cached_decode_blocks = std::max(1, active * prop.multiProcessorCount);
        }
    }
    if (cached_lm_blocks == 0) {
        if (const char *override_blocks = std::getenv("MEGAKERNEL_LM_BLOCKS")) {
            int v = std::atoi(override_blocks);
            if (v > 0) cached_lm_blocks = v;
        }
        if (cached_lm_blocks == 0) {
            int device = 0;
            cudaGetDevice(&device);
            cudaDeviceProp prop{};
            cudaGetDeviceProperties(&prop, device);
            int active = 0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active, lm_head_kernel, LM_BLOCK_SIZE, 0);
            int resident = std::max(1, active * prop.multiProcessorCount);
            int target = std::max(resident, prop.multiProcessorCount * 8);
            cached_lm_blocks = std::min(1024, target);
        }
    }

    // Lazy-allocate the FA-decode split-K partials buffer. Sized to
    // accommodate up to cached_decode_blocks splits, each holding
    // FA_NUM_Q_HEADS * (FA_HEAD_DIM + 2) floats. ~1.2 MB on B200.
    static float *g_fa_partials = nullptr;
    static int g_fa_partials_blocks = 0;
    if (g_fa_partials_blocks < cached_decode_blocks) {
        if (g_fa_partials) cudaFree(g_fa_partials);
        size_t bytes = (size_t)cached_decode_blocks * FA_NUM_Q_HEADS *
                       (FA_HEAD_DIM + 2) * sizeof(float);
        cudaMalloc(&g_fa_partials, bytes);
        g_fa_partials_blocks = cached_decode_blocks;
    }

    const void *embed_w_arg = embed_weight;
    const void *final_norm_w_arg = final_norm_weight;
    const void *lm_head_w_arg = lm_head_weight;
    const LayerWeights *layer_weights_arg = layer_weights;
    void *fa_k_cache_arg = fa_k_cache;
    void *fa_v_cache_arg = fa_v_cache;
    void *dn_states_arg = dn_states;
    void *conv_bufs_arg = conv_bufs;
    void *hidden_buffer_arg = hidden_buffer;
    void *g_activations_arg = g_activations;
    void *g_residual_arg = g_residual;
    void *g_qkv_scratch_arg = g_qkv_scratch;
    void *g_kv_scratch_arg = g_kv_scratch;
    void *g_attn_out_arg = g_attn_out;
    void *g_mlp_inter_arg = g_mlp_inter;
    void *g_z_scratch_arg = g_z_scratch;
    void *g_beta_scratch_arg = g_beta_scratch;
    void *g_alpha_scratch_arg = g_alpha_scratch;
    void *g_normalized_arg = g_normalized;
    float *g_fa_partials_arg = g_fa_partials;
    unsigned int *barrier_counter_arg = barrier_counter;
    unsigned int *barrier_generation_arg = barrier_generation;
    int input_token_id_arg = input_token_id;
    int position_arg = position;
    int max_seq_len_arg = max_seq_len;

    void *decode_args[] = {
        (void *)&embed_w_arg,
        (void *)&final_norm_w_arg,
        (void *)&lm_head_w_arg,
        (void *)&layer_weights_arg,
        (void *)&fa_k_cache_arg, (void *)&fa_v_cache_arg,
        (void *)&dn_states_arg, (void *)&conv_bufs_arg,
        (void *)&hidden_buffer_arg,
        (void *)&g_activations_arg, (void *)&g_residual_arg,
        (void *)&g_qkv_scratch_arg, (void *)&g_kv_scratch_arg,
        (void *)&g_attn_out_arg, (void *)&g_mlp_inter_arg,
        (void *)&g_z_scratch_arg, (void *)&g_beta_scratch_arg,
        (void *)&g_alpha_scratch_arg, (void *)&g_normalized_arg,
        (void *)&g_fa_partials_arg,
        (void *)&barrier_counter_arg, (void *)&barrier_generation_arg,
        (void *)&input_token_id_arg, (void *)&position_arg, (void *)&max_seq_len_arg,
    };
    cudaLaunchCooperativeKernel(
        (void *)decode_kernel,
        dim3(cached_decode_blocks),
        dim3(BLOCK_SIZE),
        decode_args,
        0,
        stream);

    (void)lm_sync_counter;

    lm_head_kernel<<<cached_lm_blocks, LM_BLOCK_SIZE, 0, stream>>>(
        (const float *)g_normalized,
        (const __nv_bfloat16 *)lm_head_weight,
        block_max_vals, block_max_idxs);
    lm_head_reduce_kernel_bf16<<<1, LM_BLOCK_SIZE, 0, stream>>>(
        block_max_vals, block_max_idxs, output_token_id, cached_lm_blocks);
}

