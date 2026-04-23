/**
 * True prefill megakernel for Qwen 3.5-0.8B.
 *
 * One cooperative CUDA dispatch — all 24 layers (hybrid DeltaNet +
 * Full Attention) processed inside a single persistent kernel with
 * grid.sync() between phases. No cuBLAS, no inter-kernel launches,
 * no CPU round-trips.
 *
 * Matches the math of ../megakernel/prefill.cu exactly so outputs can
 * be diffed element-for-element — the difference is that prefill.cu
 * issues ~480 launches per forward and this kernel issues ONE.
 *
 * bf16 weights, bf16 activations, f32 accumulation.
 * DeltaNet state kept in f32 (recurrence needs precision).
 *
 * Target: NVIDIA B200 (sm_100, 148 SMs). Compiles on sm_80+.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <mma.h>
#include <cmath>

namespace cg = cooperative_groups;
namespace wmma = nvcuda::wmma;

// ============================================================================
// Model constants — Qwen 3.5-0.8B
// ============================================================================

constexpr int WARP_SIZE = 32;
constexpr int HIDDEN = 1024;
constexpr int INTER = 3584;
constexpr int VOCAB = 248320;
constexpr int NUM_LAYERS = 24;
constexpr float RMS_EPS = 1e-6f;

// Full Attention
constexpr int FA_Q_HEADS = 8;
constexpr int FA_KV_HEADS = 2;
constexpr int FA_HEAD_DIM = 256;
constexpr int FA_GQA = FA_Q_HEADS / FA_KV_HEADS;
constexpr int FA_Q_SIZE = FA_Q_HEADS * FA_HEAD_DIM;       // 2048
constexpr int FA_QPROJ_SIZE = FA_Q_SIZE * 2;               // 4096 (q+gate)
constexpr int FA_KV_SIZE = FA_KV_HEADS * FA_HEAD_DIM;      // 512
constexpr int FA_ROT_DIM = 64;
constexpr float FA_ROPE_THETA = 10000000.0f;

// DeltaNet
constexpr int DN_HEADS = 16;
constexpr int DN_KEY = 128;
constexpr int DN_VAL = 128;
constexpr int DN_CONV_K = 4;
constexpr int DN_QK_SIZE = DN_HEADS * DN_KEY;              // 2048
constexpr int DN_V_SIZE = DN_HEADS * DN_VAL;               // 2048
constexpr int DN_CONV_CH = DN_QK_SIZE * 2 + DN_V_SIZE;     // 6144

constexpr int N_FA = 6;
constexpr int N_DN = 18;

__device__ __constant__ int LAYER_TYPE[NUM_LAYERS] = {
    0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1
};

// ============================================================================
// Weight layout — packed per layer; shared across both layer types.
// For FA  (layer_type=1): 11 bf16 tensor ptrs (input_norm, q, k, v, q_norm,
//                          k_norm, o, post_norm, gate, up, down)
// For DN  (layer_type=0): 14 bf16 tensor ptrs (input_norm, qkv, z, beta, alpha,
//                          conv, A_log, dt_bias, dn_norm, out, post_norm,
//                          gate, up, down)
// ============================================================================

struct LayerWeights {
    int layer_type;
    int _pad[3];
    void *ptrs[14];
};

#ifndef PM_BLOCK_SIZE
#define PM_BLOCK_SIZE 512
#endif

#ifndef PM_NUM_BLOCKS
#define PM_NUM_BLOCKS 148
#endif

constexpr int NW = PM_BLOCK_SIZE / WARP_SIZE;  // 16 warps/block

// ============================================================================
// Reduction primitives
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v, float *smem) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float x = (lane < NW) ? smem[lane] : 0.0f;
        x = warp_reduce_sum(x);
        if (lane == 0) smem[0] = x;
    }
    __syncthreads();
    return smem[0];
}

__device__ __forceinline__ uint4 load_128(const void *p) {
    uint4 r;
    asm volatile("ld.global.L1::no_allocate.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}

__device__ __forceinline__ float dot8_bf16(const uint4 &a, const __nv_bfloat16 *b) {
    const __nv_bfloat16 *av = reinterpret_cast<const __nv_bfloat16 *>(&a);
    float s = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) s += __bfloat162float(av[i]) * __bfloat162float(b[i]);
    return s;
}

__device__ __forceinline__ float fast_silu(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float fast_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// ============================================================================
// Phase: embedding lookup
// Blocks cover the full [S, HIDDEN] = S*1024 elements. Threads stride.
// ============================================================================

__device__ void phase_embed(
    const int *__restrict__ tokens,
    const __nv_bfloat16 *__restrict__ embed,
    __nv_bfloat16 *__restrict__ hidden,
    int S)
{
    int total = S * HIDDEN;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < total; i += stride) {
        int s = i / HIDDEN;
        int h = i % HIDDEN;
        hidden[i] = embed[tokens[s] * HIDDEN + h];
    }
}

// ============================================================================
// Phase: per-seq RMSNorm.
//   out[s, :] = in[s, :] * rsqrt(mean(in[s, :]^2) + eps) * (1 + weight[:])
//   res[s, :] = in[s, :]   (saved for post-attn residual add)
// One block per sequence position. Threads cover HIDDEN within a row.
// ============================================================================

__device__ void phase_rmsnorm(
    const __nv_bfloat16 *__restrict__ in,
    const __nv_bfloat16 *__restrict__ weight,
    __nv_bfloat16 *__restrict__ out,
    __nv_bfloat16 *__restrict__ res,
    int S, int D)
{
    __shared__ float smem[NW];
    for (int s = blockIdx.x; s < S; s += gridDim.x) {
        const __nv_bfloat16 *r_in = in + s * D;
        __nv_bfloat16 *r_out = out + s * D;
        __nv_bfloat16 *r_res = res + s * D;

        float sq = 0.0f;
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            float v = __bfloat162float(r_in[i]);
            r_res[i] = r_in[i];
            sq += v * v;
        }
        float mean_sq = block_reduce_sum(sq, smem);
        float rstd = rsqrtf(mean_sq / float(D) + RMS_EPS);

        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            float v = __bfloat162float(r_in[i]);
            float w = __bfloat162float(weight[i]);
            r_out[i] = __float2bfloat16(v * rstd * (1.0f + w));
        }
    }
}

// ============================================================================
// GEMM primitives: WMMA 16×16×16 bf16 tiles, f32 accumulation,
// cp.async double-buffered K-pipeline.
//
// Layout per block:
//   Each block owns a [M_BLK=16, N_BLK=NW*16=256] output macro tile.
//   All NW warps share the same 16-row A-slab; each warp's 16-col B-slab
//   is contiguous within a single shared-memory B buffer.
//
// Pipeline:
//   Stage N's cp.async is issued while stage N-1's MMA runs. cp.async.wait_group
//   blocks only on commits older than the one in flight, letting the tensor
//   cores overlap with loads.
//
// Memory:
//   s_A[2][16][K_TILE=32]                = 2 KB
//   s_B[2][N_BLK=256][K_TILE=32]         = 32 KB
//   s_store[NW=16][16*16 f32]            = 16 KB
//   total                                = 50 KB → needs opt-in via
//                                          cudaFuncAttributeMaxDynamicSharedMemorySize
//
// Preconditions:
//   - S % 16 == 0, K % 32 == 0, N % 256 == 0  (all satisfied by Qwen 3.5-0.8B)
// ============================================================================

constexpr int GEMM_M_BLK = 16;
constexpr int GEMM_N_BLK = NW * 16;   // 256
constexpr int GEMM_K_TILE = 32;       // 2 MMAs per K-chunk

__device__ __forceinline__ void cp_async_16B(void *smem_dst, const void *global_src) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                 :: "r"(smem_addr), "l"(global_src));
}

template <typename T>
__device__ void gemm_wmma_core(
    const __nv_bfloat16 *__restrict__ X, int K,
    const __nv_bfloat16 *__restrict__ W,
    T *__restrict__ Y,
    const __nv_bfloat16 *__restrict__ R,
    int S, int N)
{
    const int tid = threadIdx.x;
    const int warp_block = tid >> 5;
    const int lane = tid & 31;

    const int tiles_M = S >> 4;
    const int tiles_N = N / GEMM_N_BLK;
    const int total_macro = tiles_M * tiles_N;

    // Shared memory layout — offsets in bytes from smem_raw base.
    //   s_A: 2 * 16 * 32 * 2 = 2048  B
    //   s_B: 2 * 256 * 32 * 2 = 32768 B
    //   s_store: NW * 256 * 4 = 16384 B
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16 *s_A = reinterpret_cast<__nv_bfloat16 *>(smem_raw);
    __nv_bfloat16 *s_B = s_A + 2 * GEMM_M_BLK * GEMM_K_TILE;          // after A
    float *s_store = reinterpret_cast<float *>(
        s_B + 2 * GEMM_N_BLK * GEMM_K_TILE);                           // after B
    float *s_tile_warp = s_store + warp_block * 256;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a0, a1;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b0, b1;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // A-load helper: warp 0 loads 16 rows × 32 cols = 512 bf16 = 1024 bytes
    // via 64 × 16-byte cp.asyncs (each covers 8 bf16 = 16 bytes).
    // Warp 0 (32 threads) → 2 loads per thread, 4 col-segments per row.
    auto load_A = [&](int buf, int s_start, int k_off) {
        __nv_bfloat16 *A_buf = s_A + buf * GEMM_M_BLK * GEMM_K_TILE;
        if (warp_block == 0) {
            #pragma unroll
            for (int it = 0; it < 2; it++) {
                int i = lane + it * 32;         // 0..64
                int r = i >> 2;                  // 0..16
                int c = (i & 3) << 3;            // 0, 8, 16, 24
                __nv_bfloat16 *dst = A_buf + r * GEMM_K_TILE + c;
                const __nv_bfloat16 *src = X + (s_start + r) * K + k_off + c;
                cp_async_16B(dst, src);
            }
        }
    };

    // B-load helper: 512 threads load 256 rows × 32 cols = 8192 bf16 = 16384 bytes
    // via 1024 × 16-byte cp.asyncs. 512 threads → 2 loads per thread, each row has
    // 4 col-segments (0, 8, 16, 24 in bf16 offsets).
    auto load_B = [&](int buf, int n_block_start, int k_off) {
        __nv_bfloat16 *B_buf = s_B + buf * GEMM_N_BLK * GEMM_K_TILE;
        #pragma unroll
        for (int it = 0; it < 2; it++) {
            int i = tid + it * 512;       // 0..1024
            int r = i >> 2;                // 0..256
            int c = (i & 3) << 3;          // 0, 8, 16, 24
            __nv_bfloat16 *dst = B_buf + r * GEMM_K_TILE + c;
            const __nv_bfloat16 *src = W + (n_block_start + r) * K + k_off + c;
            cp_async_16B(dst, src);
        }
    };

    for (int macro = blockIdx.x; macro < total_macro; macro += gridDim.x) {
        int tm = macro / tiles_N;
        int tnb = macro % tiles_N;
        int s_start = tm << 4;
        int n_block_start = tnb * GEMM_N_BLK;
        int n_start = n_block_start + (warp_block << 4);

        wmma::fill_fragment(c_frag, 0.0f);

        // Prologue: issue stage 0 load.
        load_A(0, s_start, 0);
        load_B(0, n_block_start, 0);
        asm volatile("cp.async.commit_group;");

        int buf = 0;
        #pragma unroll 1
        for (int k = 0; k < K; k += GEMM_K_TILE) {
            int next_k = k + GEMM_K_TILE;
            int next_buf = buf ^ 1;

            // Issue next stage (if any).
            if (next_k < K) {
                load_A(next_buf, s_start, next_k);
                load_B(next_buf, n_block_start, next_k);
                asm volatile("cp.async.commit_group;");
                asm volatile("cp.async.wait_group 1;");
            } else {
                asm volatile("cp.async.wait_group 0;");
            }
            __syncthreads();

            // MMA on current buf.
            __nv_bfloat16 *A_buf = s_A + buf * GEMM_M_BLK * GEMM_K_TILE;
            __nv_bfloat16 *B_buf = s_B + buf * GEMM_N_BLK * GEMM_K_TILE
                                         + (warp_block * 16) * GEMM_K_TILE;
            wmma::load_matrix_sync(a0, A_buf + 0,  GEMM_K_TILE);
            wmma::load_matrix_sync(a1, A_buf + 16, GEMM_K_TILE);
            wmma::load_matrix_sync(b0, B_buf + 0,  GEMM_K_TILE);
            wmma::load_matrix_sync(b1, B_buf + 16, GEMM_K_TILE);
            wmma::mma_sync(c_frag, a0, b0, c_frag);
            wmma::mma_sync(c_frag, a1, b1, c_frag);

            __syncthreads();
            buf = next_buf;
        }

        // Store accumulator.
        wmma::store_matrix_sync(s_tile_warp, c_frag, 16, wmma::mem_row_major);
        #pragma unroll
        for (int i = lane; i < 256; i += 32) {
            int r = i >> 4;
            int c = i & 15;
            int idx = (s_start + r) * N + (n_start + c);
            float v = s_tile_warp[i];
            if (R != nullptr) v += __bfloat162float(R[idx]);
            if constexpr (std::is_same<T, __nv_bfloat16>::value) {
                Y[idx] = __float2bfloat16(v);
            } else {
                Y[idx] = v;
            }
        }
    }
}

// ============================================================================
// LoRA forward phases. For a LoRA-adapted linear y = x @ Wᵀ + s·(x @ A) @ B:
//   phase_lora_h computes the rank-R hidden:   lora_h[s, r] = Σ_k x[s, k] * A[k, r]
//   phase_lora_b_add adds to the base proj:    proj[s, n] += s·Σ_r lora_h[s, r] * B[r, n]
//
// A layout: [K_in, LORA_R] row-major.
// B layout: [LORA_R, K_out] row-major.
//
// Generic over K_in (input dim) — caller passes it so one kernel covers
// q/k/v (K_in=HIDDEN), o (K_in=FA_Q_SIZE), down (K_in=INTER), DN out
// (K_in=DN_V_SIZE) etc.
// ============================================================================

__device__ void phase_lora_h(
    const __nv_bfloat16 *__restrict__ X,
    const __nv_bfloat16 *__restrict__ A,
    float *__restrict__ lora_h,
    int S, int K_in, int LORA_R)
{
    int total = S * LORA_R;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int idx = tid; idx < total; idx += stride) {
        int s = idx / LORA_R;
        int r = idx % LORA_R;
        const __nv_bfloat16 *x_row = X + (size_t)s * K_in;
        float acc = 0.0f;
        #pragma unroll 4
        for (int h = 0; h < K_in; h++) {
            acc += __bfloat162float(x_row[h]) * __bfloat162float(A[h * LORA_R + r]);
        }
        lora_h[idx] = acc;
    }
}

// Pointer bundle for every trainable LoRA A/B pair. Each element points at
// the start of a [N_layers_of_this_type, K_in, R] or [N_layers_of_this_type, R, K_out]
// contiguous buffer. Nullable (disables LoRA on that specific linear).
struct LoraSet {
    const __nv_bfloat16 *fa_q_A,    *fa_q_B;
    const __nv_bfloat16 *fa_k_A,    *fa_k_B;
    const __nv_bfloat16 *fa_v_A,    *fa_v_B;
    const __nv_bfloat16 *fa_o_A,    *fa_o_B;
    const __nv_bfloat16 *fa_gate_A, *fa_gate_B;
    const __nv_bfloat16 *fa_up_A,   *fa_up_B;
    const __nv_bfloat16 *fa_down_A, *fa_down_B;
    const __nv_bfloat16 *dn_qkv_A,  *dn_qkv_B;
    const __nv_bfloat16 *dn_z_A,    *dn_z_B;
    const __nv_bfloat16 *dn_out_A,  *dn_out_B;
    const __nv_bfloat16 *dn_gate_A, *dn_gate_B;
    const __nv_bfloat16 *dn_up_A,   *dn_up_B;
    const __nv_bfloat16 *dn_down_A, *dn_down_B;
};

__device__ void phase_lora_b_add(
    const float *__restrict__ lora_h,            // [S, LORA_R]
    const __nv_bfloat16 *__restrict__ B,         // [LORA_R, N]
    __nv_bfloat16 *__restrict__ proj,            // [S, N]  accumulated in place
    int S, int N, int LORA_R, float scaling)
{
    int total = S * N;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int idx = tid; idx < total; idx += stride) {
        int s = idx / N;
        int n = idx % N;
        const float *h_row = lora_h + (size_t)s * LORA_R;
        float acc = __bfloat162float(proj[idx]);
        #pragma unroll
        for (int r = 0; r < 32; r++) {
            if (r >= LORA_R) break;
            acc += h_row[r] * __bfloat162float(B[r * N + n]) * scaling;
        }
        proj[idx] = __float2bfloat16(acc);
    }
}

__device__ __forceinline__ void gemm_bf16_bf16(
    const __nv_bfloat16 *X, int K, const __nv_bfloat16 *W,
    __nv_bfloat16 *Y, int S, int N)
{
    gemm_wmma_core<__nv_bfloat16>(X, K, W, Y, nullptr, S, N);
}

__device__ __forceinline__ void gemm_bf16_f32(
    const __nv_bfloat16 *X, int K, const __nv_bfloat16 *W,
    float *Y, int S, int N)
{
    gemm_wmma_core<float>(X, K, W, Y, nullptr, S, N);
}

__device__ __forceinline__ void gemm_bf16_add_residual(
    const __nv_bfloat16 *X, int K, const __nv_bfloat16 *W,
    const __nv_bfloat16 *R, __nv_bfloat16 *Y, int S, int N)
{
    gemm_wmma_core<__nv_bfloat16>(X, K, W, Y, R, S, N);
}

// ============================================================================
// SwiGLU (SiLU(gate) * up)  element-wise, in place into `out`
// ============================================================================

__device__ void phase_silu_mul(
    const __nv_bfloat16 *__restrict__ gate,
    const __nv_bfloat16 *__restrict__ up,
    __nv_bfloat16 *__restrict__ out,
    int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < N; i += stride) {
        float g = __bfloat162float(gate[i]);
        float u = __bfloat162float(up[i]);
        out[i] = __float2bfloat16(fast_silu(g) * u);
    }
}

// ============================================================================
// FA: per-head Q/K RMSNorm + partial RoPE + KV cache write.
// For each (s, head), RMSNorm across HEAD_DIM, apply (1+w), rotate first
// FA_ROT_DIM channels by RoPE at position `s`. Write K/V cache.
//
// Layout of q[s, :] = [q_head_0_data(HD) | gate_0(HD) | q_head_1_data | gate_1 | ...]
// (per prefill.cu's FA_QPROJ = 4096 = 8 heads * 2 * 256).
// ============================================================================

__device__ void phase_fa_qk_norm_rope(
    __nv_bfloat16 *__restrict__ q,        // [S, FA_QPROJ] in-place
    __nv_bfloat16 *__restrict__ k,        // [S, FA_KV_SIZE]
    const __nv_bfloat16 *__restrict__ v,  // [S, FA_KV_SIZE]
    const __nv_bfloat16 *__restrict__ qnw,
    const __nv_bfloat16 *__restrict__ knw,
    __nv_bfloat16 *__restrict__ k_cache,  // [KV_HEADS, max_seq, HD]
    __nv_bfloat16 *__restrict__ v_cache,  // [KV_HEADS, max_seq, HD]
    int S, int max_seq)
{
    // Distribute (s, head) work across grid. 1 warp per (s, head).
    int total_q = S * FA_Q_HEADS;
    int total_k = S * FA_KV_HEADS;
    int total = total_q + total_k;

    int warps_per_grid = gridDim.x * NW;
    int warp_grid = blockIdx.x * NW + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;

    for (int idx = warp_grid; idx < total; idx += warps_per_grid) {
        if (idx < total_q) {
            int pos = idx / FA_Q_HEADS;
            int head = idx % FA_Q_HEADS;
            __nv_bfloat16 *qh = q + pos * FA_QPROJ_SIZE + head * FA_HEAD_DIM * 2;

            float ss = 0;
            for (int i = lane; i < FA_HEAD_DIM; i += 32) {
                float v = __bfloat162float(qh[i]);
                ss += v * v;
            }
            ss = warp_reduce_sum(ss);
            float sc = rsqrtf(ss / float(FA_HEAD_DIM) + RMS_EPS);
            for (int i = lane; i < FA_HEAD_DIM; i += 32) {
                float normed = __bfloat162float(qh[i]) * sc *
                               (1.0f + __bfloat162float(qnw[i]));
                if (i < FA_ROT_DIM) {
                    float fe = float(2 * (i % (FA_ROT_DIM / 2))) / float(FA_ROT_DIM);
                    float freq = float(pos) / powf(FA_ROPE_THETA, fe);
                    float cv = cosf(freq), sv = sinf(freq);
                    int p = (i < FA_ROT_DIM / 2) ? i + FA_ROT_DIM / 2 : i - FA_ROT_DIM / 2;
                    float pv = __bfloat162float(qh[p]) * sc *
                               (1.0f + __bfloat162float(qnw[p]));
                    float rotated = (i < FA_ROT_DIM / 2) ? (normed * cv - pv * sv)
                                                          : (pv * sv + normed * cv);
                    qh[i] = __float2bfloat16(rotated);
                } else {
                    qh[i] = __float2bfloat16(normed);
                }
            }
        } else {
            int kidx = idx - total_q;
            int pos = kidx / FA_KV_HEADS;
            int head = kidx % FA_KV_HEADS;
            __nv_bfloat16 *kh = k + pos * FA_KV_SIZE + head * FA_HEAD_DIM;
            const __nv_bfloat16 *vh = v + pos * FA_KV_SIZE + head * FA_HEAD_DIM;
            __nv_bfloat16 *kc = k_cache + head * max_seq * FA_HEAD_DIM + pos * FA_HEAD_DIM;
            __nv_bfloat16 *vc = v_cache + head * max_seq * FA_HEAD_DIM + pos * FA_HEAD_DIM;

            float ss = 0;
            for (int i = lane; i < FA_HEAD_DIM; i += 32) {
                float v = __bfloat162float(kh[i]);
                ss += v * v;
            }
            ss = warp_reduce_sum(ss);
            float sc = rsqrtf(ss / float(FA_HEAD_DIM) + RMS_EPS);
            for (int i = lane; i < FA_HEAD_DIM; i += 32) {
                float normed = __bfloat162float(kh[i]) * sc *
                               (1.0f + __bfloat162float(knw[i]));
                float fk;
                if (i < FA_ROT_DIM) {
                    float fe = float(2 * (i % (FA_ROT_DIM / 2))) / float(FA_ROT_DIM);
                    float freq = float(pos) / powf(FA_ROPE_THETA, fe);
                    float cv = cosf(freq), sv = sinf(freq);
                    int p = (i < FA_ROT_DIM / 2) ? i + FA_ROT_DIM / 2 : i - FA_ROT_DIM / 2;
                    float pv = __bfloat162float(kh[p]) * sc *
                               (1.0f + __bfloat162float(knw[p]));
                    fk = (i < FA_ROT_DIM / 2) ? (normed * cv - pv * sv)
                                               : (pv * sv + normed * cv);
                } else {
                    fk = normed;
                }
                kh[i] = __float2bfloat16(fk);
                kc[i] = __float2bfloat16(fk);
                vc[i] = vh[i];
            }
        }
    }
}

// ============================================================================
// FA: causal attention with online softmax + sigmoid output gate.
// Layout: q[s, :] = [head0_q | head0_gate | head1_q | head1_gate | ...]
// Output: out[s, FA_Q_SIZE] — no gate concatenation, just gated values.
// One warp per (s, q_head). Streams K/V across past positions.
// ============================================================================

__device__ void phase_fa_causal_attn(
    const __nv_bfloat16 *__restrict__ q,   // [S, FA_QPROJ]
    const __nv_bfloat16 *__restrict__ k,   // [S, FA_KV_SIZE]
    const __nv_bfloat16 *__restrict__ v,   // [S, FA_KV_SIZE]
    __nv_bfloat16 *__restrict__ out,       // [S, FA_Q_SIZE]
    int S)
{
    const int total = S * FA_Q_HEADS;
    constexpr int EPL = FA_HEAD_DIM / 32;
    const float scale = 1.0f / sqrtf(float(FA_HEAD_DIM));
    int warps_per_grid = gridDim.x * NW;
    int warp_grid = blockIdx.x * NW + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;

    for (int idx = warp_grid; idx < total; idx += warps_per_grid) {
        int pos = idx / FA_Q_HEADS;
        int qh = idx % FA_Q_HEADS;
        int kvh = qh / FA_GQA;

        const __nv_bfloat16 *qv = q + pos * FA_QPROJ_SIZE + qh * FA_HEAD_DIM * 2;
        const __nv_bfloat16 *gv = qv + FA_HEAD_DIM;
        __nv_bfloat16 *ov = out + pos * FA_Q_SIZE + qh * FA_HEAD_DIM;

        float ql[EPL];
        #pragma unroll
        for (int e = 0; e < EPL; e++) ql[e] = __bfloat162float(qv[lane * EPL + e]);

        float oa[EPL] = {};
        float mx = -1e30f, se = 0;

        for (int kp = 0; kp <= pos; kp++) {
            const __nv_bfloat16 *kv_ = k + kp * FA_KV_SIZE + kvh * FA_HEAD_DIM;
            const __nv_bfloat16 *vv = v + kp * FA_KV_SIZE + kvh * FA_HEAD_DIM;
            float sc = 0;
            #pragma unroll
            for (int e = 0; e < EPL; e++) sc += ql[e] * __bfloat162float(kv_[lane * EPL + e]);
            sc = warp_reduce_sum(sc) * scale;
            float om = mx;
            mx = fmaxf(mx, sc);
            float ed = expf(om - mx);
            se = se * ed + expf(sc - mx);
            float wt = expf(sc - mx);
            #pragma unroll
            for (int e = 0; e < EPL; e++)
                oa[e] = oa[e] * ed + wt * __bfloat162float(vv[lane * EPL + e]);
        }
        float rs = 1.0f / se;
        #pragma unroll
        for (int e = 0; e < EPL; e++) {
            int i = lane * EPL + e;
            float g = 1.0f / (1.0f + expf(-__bfloat162float(gv[i])));
            ov[i] = __float2bfloat16(oa[e] * rs * g);
        }
    }
}

// ============================================================================
// DN: "tiny" matvec for beta/alpha projections — [S, DN_HEADS] output from
// [S, HIDDEN] input. One warp per (s, head).
// ============================================================================

__device__ void phase_dn_beta_alpha_matvec(
    const __nv_bfloat16 *__restrict__ X,   // [S, HIDDEN]
    const __nv_bfloat16 *__restrict__ W,   // [DN_HEADS, HIDDEN]
    float *__restrict__ Y,                  // [S, DN_HEADS]
    int S)
{
    int total = S * DN_HEADS;
    int warps_per_grid = gridDim.x * NW;
    int warp_grid = blockIdx.x * NW + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    for (int idx = warp_grid; idx < total; idx += warps_per_grid) {
        int s = idx / DN_HEADS;
        int n = idx % DN_HEADS;
        const __nv_bfloat16 *x = X + s * HIDDEN;
        const __nv_bfloat16 *w = W + n * HIDDEN;
        float acc = 0.0f;
        for (int k = lane * 8; k < HIDDEN; k += 32 * 8) {
            uint4 wu = load_128(reinterpret_cast<const uint4 *>(w + k));
            acc += dot8_bf16(wu, x + k);
        }
        acc = warp_reduce_sum(acc);
        if (lane == 0) Y[idx] = acc;
    }
}

// ============================================================================
// DN: state-in-registers recurrence for prefill.
// One block per head (16 blocks participate; others idle for this phase).
// Sequential over S, state held in 32 registers per warp.
//
// Exactly mirrors prefill.cu's pf_deltanet_recurrence. Output is bf16.
// ============================================================================

__device__ void phase_dn_recurrence(
    const __nv_bfloat16 *__restrict__ qkv_proj,  // [S, DN_CONV_CH]
    const __nv_bfloat16 *__restrict__ z_proj,    // [S, DN_V_SIZE]
    const float       *__restrict__ beta_proj,    // [S, DN_HEADS]
    const float       *__restrict__ alpha_proj,   // [S, DN_HEADS]
    const __nv_bfloat16 *__restrict__ conv_w,    // [DN_CONV_CH, DN_CONV_K]
    const __nv_bfloat16 *__restrict__ a_log,     // [DN_HEADS]
    const __nv_bfloat16 *__restrict__ dt_bias,   // [DN_HEADS]
    const __nv_bfloat16 *__restrict__ norm_w,    // [DN_VAL]
    float             *__restrict__ state,        // [DN_HEADS, DN_KEY, DN_VAL]
    float             *__restrict__ conv_buf,     // [DN_CONV_CH, DN_CONV_K]
    __nv_bfloat16     *__restrict__ output,       // [S, DN_V_SIZE]
    int S)
{
    int h = blockIdx.x;
    if (h >= DN_HEADS) return;
    int tid = threadIdx.x, wid = tid / 32, lid = tid % 32;
    constexpr int NWARPS = PM_BLOCK_SIZE / 32;   // 16
    constexpr float Q_SCALE = 1.0f / 11.313708498984761f;
    constexpr int CPW = DN_VAL / NWARPS;   // 8
    constexpr int RPL = DN_KEY / 32;        // 4

    float a_log_val = __bfloat162float(a_log[h]);
    float dt_b = __bfloat162float(dt_bias[h]);

    __shared__ float s_q[DN_KEY], s_k[DN_KEY], s_v[DN_VAL];
    __shared__ float s_beta, s_decay;
    __shared__ float s_gnorm[NWARPS];

    float *my_state = state + h * DN_KEY * DN_VAL;

    float sreg[CPW * RPL];
    for (int jj = 0; jj < CPW; jj++) {
        int j = wid * CPW + jj;
        for (int ii = 0; ii < RPL; ii++)
            sreg[jj * RPL + ii] = my_state[j * DN_KEY + lid + ii * 32];
    }

    for (int t = 0; t < S; t++) {
        // Conv1d+SiLU for Q,K,V channels (within this head's slice).
        for (int c = tid; c < DN_KEY; c += PM_BLOCK_SIZE) {
            int ch = h * DN_KEY + c;
            float h0 = conv_buf[ch * DN_CONV_K + 1];
            float h1 = conv_buf[ch * DN_CONV_K + 2];
            float h2 = conv_buf[ch * DN_CONV_K + 3];
            conv_buf[ch * DN_CONV_K + 0] = h0;
            conv_buf[ch * DN_CONV_K + 1] = h1;
            conv_buf[ch * DN_CONV_K + 2] = h2;
            conv_buf[ch * DN_CONV_K + 3] = __bfloat162float(qkv_proj[t * DN_CONV_CH + ch]);
            float co = 0;
            for (int k = 0; k < DN_CONV_K; k++)
                co += conv_buf[ch * DN_CONV_K + k] * __bfloat162float(conv_w[ch * DN_CONV_K + k]);
            s_q[c] = fast_silu(co);
        }
        for (int c = tid; c < DN_KEY; c += PM_BLOCK_SIZE) {
            int ch = DN_QK_SIZE + h * DN_KEY + c;
            float h0 = conv_buf[ch * DN_CONV_K + 1];
            float h1 = conv_buf[ch * DN_CONV_K + 2];
            float h2 = conv_buf[ch * DN_CONV_K + 3];
            conv_buf[ch * DN_CONV_K + 0] = h0;
            conv_buf[ch * DN_CONV_K + 1] = h1;
            conv_buf[ch * DN_CONV_K + 2] = h2;
            conv_buf[ch * DN_CONV_K + 3] = __bfloat162float(qkv_proj[t * DN_CONV_CH + ch]);
            float co = 0;
            for (int k = 0; k < DN_CONV_K; k++)
                co += conv_buf[ch * DN_CONV_K + k] * __bfloat162float(conv_w[ch * DN_CONV_K + k]);
            s_k[c] = fast_silu(co);
        }
        for (int c = tid; c < DN_VAL; c += PM_BLOCK_SIZE) {
            int ch = 2 * DN_QK_SIZE + h * DN_VAL + c;
            float h0 = conv_buf[ch * DN_CONV_K + 1];
            float h1 = conv_buf[ch * DN_CONV_K + 2];
            float h2 = conv_buf[ch * DN_CONV_K + 3];
            conv_buf[ch * DN_CONV_K + 0] = h0;
            conv_buf[ch * DN_CONV_K + 1] = h1;
            conv_buf[ch * DN_CONV_K + 2] = h2;
            conv_buf[ch * DN_CONV_K + 3] = __bfloat162float(qkv_proj[t * DN_CONV_CH + ch]);
            float co = 0;
            for (int k = 0; k < DN_CONV_K; k++)
                co += conv_buf[ch * DN_CONV_K + k] * __bfloat162float(conv_w[ch * DN_CONV_K + k]);
            s_v[c] = fast_silu(co);
        }
        __syncthreads();

        // L2 normalize Q (with scale) and K.
        if (wid == 0) {
            float sq = 0;
            for (int i = lid; i < DN_KEY; i += 32) sq += s_q[i] * s_q[i];
            sq = warp_reduce_sum(sq);
            float n = rsqrtf(sq + 1e-6f) * Q_SCALE;
            for (int i = lid; i < DN_KEY; i += 32) s_q[i] *= n;
        }
        if (wid == 1) {
            float sq = 0;
            for (int i = lid; i < DN_KEY; i += 32) sq += s_k[i] * s_k[i];
            sq = warp_reduce_sum(sq);
            float n = rsqrtf(sq + 1e-6f);
            for (int i = lid; i < DN_KEY; i += 32) s_k[i] *= n;
        }
        __syncthreads();

        if (tid == 0) {
            s_beta = 1.0f / (1.0f + expf(-beta_proj[t * DN_HEADS + h]));
            float x = alpha_proj[t * DN_HEADS + h] + dt_b;
            float sp = (x > 20.0f) ? x : logf(1.0f + expf(x));
            s_decay = expf(-expf(a_log_val) * sp);
        }
        __syncthreads();
        float beta = s_beta, decay = s_decay;
        __nv_bfloat16 *out_h = output + t * DN_V_SIZE + h * DN_VAL;

        // State-in-registers recurrence.
        // sreg layout: for warp wid, columns j = wid*CPW .. wid*CPW+CPW-1.
        // Within each column j, 32 lanes cover DN_KEY/32=4 row groups.
        // s_k and s_q are cached in shared memory.
        for (int jj = 0; jj < CPW; jj++) {
            int j = wid * CPW + jj;
            float kv = 0;
            for (int ii = 0; ii < RPL; ii++) kv += sreg[jj * RPL + ii] * s_k[lid + ii * 32];
            kv = warp_reduce_sum(kv);
            float delta = (s_v[j] - decay * kv) * beta;
            float attn = 0;
            for (int ii = 0; ii < RPL; ii++) {
                sreg[jj * RPL + ii] = decay * sreg[jj * RPL + ii] + s_k[lid + ii * 32] * delta;
                attn += sreg[jj * RPL + ii] * s_q[lid + ii * 32];
            }
            attn = warp_reduce_sum(attn);
            if (lid == 0) out_h[j] = __float2bfloat16(attn);
        }
        __syncthreads();

        // Gated RMSNorm: RMSNorm over DN_VAL, then multiply by silu(z).
        const __nv_bfloat16 *z_h = z_proj + t * DN_V_SIZE + h * DN_VAL;
        float sq2 = 0;
        for (int i = tid; i < DN_VAL; i += PM_BLOCK_SIZE) {
            float v = __bfloat162float(out_h[i]);
            sq2 += v * v;
        }
        sq2 = warp_reduce_sum(sq2);
        if (lid == 0) s_gnorm[wid] = sq2;
        __syncthreads();
        if (wid == 0) {
            float v = (lid < NWARPS) ? s_gnorm[lid] : 0;
            v = warp_reduce_sum(v);
            if (lid == 0) s_gnorm[0] = rsqrtf(v / float(DN_VAL) + RMS_EPS);
        }
        __syncthreads();
        float rstd = s_gnorm[0];
        for (int i = tid; i < DN_VAL; i += PM_BLOCK_SIZE) {
            float n = __bfloat162float(out_h[i]) * rstd * __bfloat162float(norm_w[i]);
            out_h[i] = __float2bfloat16(n * fast_silu(__bfloat162float(z_h[i])));
        }
        __syncthreads();
    }

    // Write state back.
    for (int jj = 0; jj < CPW; jj++) {
        int j = wid * CPW + jj;
        for (int ii = 0; ii < RPL; ii++)
            my_state[j * DN_KEY + lid + ii * 32] = sreg[jj * RPL + ii];
    }
}

// ============================================================================
// Final RMSNorm + LM head (argmax).
// Operates only on the last sequence position (as in prefill.cu).
// ============================================================================

__device__ void phase_final_norm_last(
    const __nv_bfloat16 *__restrict__ hidden,   // [S, HIDDEN]
    const __nv_bfloat16 *__restrict__ w,
    __nv_bfloat16 *__restrict__ normed,         // [HIDDEN]
    int S)
{
    if (blockIdx.x != 0) return;
    __shared__ float smem[NW];
    const __nv_bfloat16 *row = hidden + (S - 1) * HIDDEN;
    float sq = 0;
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) {
        float v = __bfloat162float(row[i]);
        sq += v * v;
    }
    float mean_sq = block_reduce_sum(sq, smem);
    float rstd = rsqrtf(mean_sq / float(HIDDEN) + RMS_EPS);
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) {
        float v = __bfloat162float(row[i]);
        normed[i] = __float2bfloat16(v * rstd * (1.0f + __bfloat162float(w[i])));
    }
}

// LM head argmax across VOCAB. Block-strided reduction into (block_max_val,
// block_max_idx), followed by a final-reduce inside block 0.
__device__ void phase_lm_head_argmax(
    const __nv_bfloat16 *__restrict__ normed,   // [HIDDEN]
    const __nv_bfloat16 *__restrict__ w,        // [VOCAB, HIDDEN]
    float *__restrict__ block_max_vals,
    int   *__restrict__ block_max_idxs,
    int   *__restrict__ out_token)
{
    __shared__ __nv_bfloat16 s_h[HIDDEN];
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) s_h[i] = normed[i];
    __syncthreads();

    int wid = threadIdx.x / 32, lid = threadIdx.x % 32;
    int rpb = (VOCAB + gridDim.x - 1) / gridDim.x;
    int rs = blockIdx.x * rpb, re = min(rs + rpb, VOCAB);
    float best_v = -1e30f;
    int best_i = -1;
    for (int m = rs + wid; m < re; m += NW) {
        const __nv_bfloat16 *wr = w + m * HIDDEN;
        float s = 0;
        #pragma unroll 4
        for (int k = lid * 8; k < HIDDEN; k += 32 * 8) {
            uint4 wu = load_128(reinterpret_cast<const uint4 *>(wr + k));
            s += dot8_bf16(wu, s_h + k);
        }
        s = warp_reduce_sum(s);
        if (lid == 0 && s > best_v) { best_v = s; best_i = m; }
    }
    best_v = __shfl_sync(0xffffffff, best_v, 0);
    best_i = __shfl_sync(0xffffffff, best_i, 0);

    __shared__ float w_v[NW];
    __shared__ int   w_i[NW];
    if (lid == 0) { w_v[wid] = best_v; w_i[wid] = best_i; }
    __syncthreads();

    if (wid == 0) {
        float mv = (lid < NW) ? w_v[lid] : -1e30f;
        int   mi = (lid < NW) ? w_i[lid] : -1;
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_down_sync(0xffffffff, mv, o);
            int   oi = __shfl_down_sync(0xffffffff, mi, o);
            if (ov > mv) { mv = ov; mi = oi; }
        }
        if (lid == 0) {
            block_max_vals[blockIdx.x] = mv;
            block_max_idxs[blockIdx.x] = mi;
        }
    }
}

// Final reduce over per-block (max, idx) pairs. Run by block 0 only.
__device__ void phase_lm_final_reduce(
    const float *__restrict__ vals,
    const int   *__restrict__ idxs,
    int   *__restrict__ out_token,
    int nb)
{
    if (blockIdx.x != 0) return;
    int tid = threadIdx.x;
    float best = -1e30f;
    int bi = -1;
    for (int i = tid; i < nb; i += blockDim.x) {
        float v = vals[i];
        if (v > best) { best = v; bi = idxs[i]; }
    }
    __shared__ float sv[PM_BLOCK_SIZE];
    __shared__ int   si[PM_BLOCK_SIZE];
    sv[tid] = best; si[tid] = bi;
    __syncthreads();
    for (int s = PM_BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s && sv[tid + s] > sv[tid]) {
            sv[tid] = sv[tid + s];
            si[tid] = si[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) *out_token = si[0];
}

// ============================================================================
// Per-layer FA / DN. Each is a sequence of GEMMs / per-phase work separated
// by grid.sync(). They're inlined inside the megakernel loop below.
// ============================================================================

// ============================================================================
// Megakernel entry point.
// ============================================================================

extern "C" __global__ void __launch_bounds__(PM_BLOCK_SIZE, 1) prefill_megakernel(
    const int *__restrict__ token_ids,      // [S]
    int       *__restrict__ out_token,       // [1]
    // Weights
    const __nv_bfloat16 *__restrict__ embed,
    const LayerWeights  *__restrict__ layers,
    const __nv_bfloat16 *__restrict__ final_norm_w,
    const __nv_bfloat16 *__restrict__ lm_head_w,
    // LoRA configuration for all trainable linears (pass NULL pointers to disable
    // per linear). Pointer layouts encoded per LoraSet member comment.
    LoraSet lora,
    int lora_rank,
    float lora_scaling,
    float *__restrict__ lora_h_ws,            // [S, LORA_R] scratch
    // Persistent caches (zeroed by host before first call).
    __nv_bfloat16 *__restrict__ fa_k_cache,  // [N_FA, KV_HEADS, max_seq, HD]
    __nv_bfloat16 *__restrict__ fa_v_cache,
    float         *__restrict__ dn_states,   // [N_DN, DN_HEADS*DN_KEY*DN_VAL]
    float         *__restrict__ conv_bufs,   // [N_DN, DN_CONV_CH*DN_CONV_K]
    // Per-forward workspace buffers.
    __nv_bfloat16 *__restrict__ hidden,      // [S, HIDDEN]
    __nv_bfloat16 *__restrict__ residual,    // [S, HIDDEN]
    __nv_bfloat16 *__restrict__ normalized,  // [S, HIDDEN]
    __nv_bfloat16 *__restrict__ proj_buf,    // [S, max_proj_dim]
    __nv_bfloat16 *__restrict__ proj_buf2,
    __nv_bfloat16 *__restrict__ attn_buf,
    __nv_bfloat16 *__restrict__ mlp_buf,
    __nv_bfloat16 *__restrict__ dn_out_buf,  // [S, DN_V_SIZE] / [S, FA_Q_SIZE]
    float         *__restrict__ beta_buf,    // [S, DN_HEADS]
    float         *__restrict__ alpha_buf,
    __nv_bfloat16 *__restrict__ final_normed, // [HIDDEN]
    float         *__restrict__ lm_bmv,       // [gridDim.x]
    int           *__restrict__ lm_bmi,
    int S, int max_seq)
{
    cg::grid_group grid = cg::this_grid();

    const int fa_stride = FA_KV_HEADS * max_seq * FA_HEAD_DIM;
    const int dn_state_stride = DN_HEADS * DN_KEY * DN_VAL;
    const int dn_conv_stride = DN_CONV_CH * DN_CONV_K;

    // ------ embed -----------------------------------------------------------
    phase_embed(token_ids, embed, hidden, S);
    grid.sync();

    int fa_idx = 0, dn_idx = 0;
    for (int li = 0; li < NUM_LAYERS; li++) {
        const LayerWeights &lw = layers[li];
        int lt = LAYER_TYPE[li];

        const __nv_bfloat16 *norm_w = (const __nv_bfloat16 *)lw.ptrs[0];

        // input RMSNorm
        phase_rmsnorm(hidden, norm_w, normalized, residual, S, HIDDEN);
        grid.sync();

        if (lt == 0) {
            // ---------- DeltaNet layer ---------------------------------------
            const __nv_bfloat16 *qkv_w = (const __nv_bfloat16 *)lw.ptrs[1];
            const __nv_bfloat16 *z_w   = (const __nv_bfloat16 *)lw.ptrs[2];
            const __nv_bfloat16 *beta_w  = (const __nv_bfloat16 *)lw.ptrs[3];
            const __nv_bfloat16 *alpha_w = (const __nv_bfloat16 *)lw.ptrs[4];
            const __nv_bfloat16 *conv_w  = (const __nv_bfloat16 *)lw.ptrs[5];
            const __nv_bfloat16 *a_log   = (const __nv_bfloat16 *)lw.ptrs[6];
            const __nv_bfloat16 *dt_bias = (const __nv_bfloat16 *)lw.ptrs[7];
            const __nv_bfloat16 *dn_norm = (const __nv_bfloat16 *)lw.ptrs[8];
            const __nv_bfloat16 *out_w   = (const __nv_bfloat16 *)lw.ptrs[9];
            const __nv_bfloat16 *post_norm = (const __nv_bfloat16 *)lw.ptrs[10];
            const __nv_bfloat16 *gate_w  = (const __nv_bfloat16 *)lw.ptrs[11];
            const __nv_bfloat16 *up_w    = (const __nv_bfloat16 *)lw.ptrs[12];
            const __nv_bfloat16 *down_w  = (const __nv_bfloat16 *)lw.ptrs[13];

            // Macro: apply LoRA for one linear. X is the input; Y_BUF is the
            // output buffer to accumulate into; A_ALL / B_ALL are the packed
            // per-layer weight arrays (nullable). K_IN is the input feature
            // dimension, K_OUT the output feature dimension.
            #define APPLY_LORA(X, Y_BUF, A_ALL, B_ALL, LAYER_IDX, K_IN, K_OUT) \
                if ((A_ALL) != nullptr && (B_ALL) != nullptr) {                \
                    const __nv_bfloat16 *__A = (A_ALL)                         \
                        + (size_t)(LAYER_IDX) * (K_IN) * lora_rank;            \
                    const __nv_bfloat16 *__B = (B_ALL)                         \
                        + (size_t)(LAYER_IDX) * lora_rank * (K_OUT);           \
                    phase_lora_h((X), __A, lora_h_ws, S, (K_IN), lora_rank);   \
                    grid.sync();                                               \
                    phase_lora_b_add(lora_h_ws, __B, (Y_BUF), S,               \
                                     (K_OUT), lora_rank, lora_scaling);        \
                    grid.sync();                                               \
                }

            // QKV / Z / beta / alpha projections
            gemm_bf16_bf16(normalized, HIDDEN, qkv_w, proj_buf, S, DN_CONV_CH);
            gemm_bf16_bf16(normalized, HIDDEN, z_w, proj_buf2, S, DN_V_SIZE);
            phase_dn_beta_alpha_matvec(normalized, beta_w,  beta_buf,  S);
            phase_dn_beta_alpha_matvec(normalized, alpha_w, alpha_buf, S);
            grid.sync();

            // LoRA: qkv, z
            APPLY_LORA(normalized, proj_buf,  lora.dn_qkv_A, lora.dn_qkv_B, dn_idx, HIDDEN, DN_CONV_CH);
            APPLY_LORA(normalized, proj_buf2, lora.dn_z_A,   lora.dn_z_B,   dn_idx, HIDDEN, DN_V_SIZE);

            // Recurrence (operates on proj_buf / proj_buf2 which now include LoRA)
            phase_dn_recurrence(
                proj_buf, proj_buf2, beta_buf, alpha_buf,
                conv_w, a_log, dt_bias, dn_norm,
                dn_states + dn_idx * dn_state_stride,
                conv_bufs + dn_idx * dn_conv_stride,
                dn_out_buf, S);
            grid.sync();

            // Out projection + residual add
            gemm_bf16_add_residual(dn_out_buf, DN_V_SIZE, out_w, residual, hidden, S, HIDDEN);
            grid.sync();
            // LoRA on out_proj: add scaling*(dn_out @ A) @ B to hidden directly
            // (math-equivalent to LoRA-before-residual because residual is linear).
            APPLY_LORA(dn_out_buf, hidden, lora.dn_out_A, lora.dn_out_B, dn_idx, DN_V_SIZE, HIDDEN);

            // Post-attn RMSNorm
            phase_rmsnorm(hidden, post_norm, normalized, residual, S, HIDDEN);
            grid.sync();

            // gate/up
            gemm_bf16_bf16(normalized, HIDDEN, gate_w, proj_buf,  S, INTER);
            gemm_bf16_bf16(normalized, HIDDEN, up_w,   proj_buf2, S, INTER);
            grid.sync();
            APPLY_LORA(normalized, proj_buf,  lora.dn_gate_A, lora.dn_gate_B, dn_idx, HIDDEN, INTER);
            APPLY_LORA(normalized, proj_buf2, lora.dn_up_A,   lora.dn_up_B,   dn_idx, HIDDEN, INTER);

            // silu(gate) * up
            phase_silu_mul(proj_buf, proj_buf2, mlp_buf, S * INTER);
            grid.sync();

            // down + residual
            gemm_bf16_add_residual(mlp_buf, INTER, down_w, residual, hidden, S, HIDDEN);
            grid.sync();
            APPLY_LORA(mlp_buf, hidden, lora.dn_down_A, lora.dn_down_B, dn_idx, INTER, HIDDEN);

            dn_idx++;
        } else {
            // ---------- Full Attention layer ---------------------------------
            const __nv_bfloat16 *q_w  = (const __nv_bfloat16 *)lw.ptrs[1];
            const __nv_bfloat16 *k_w  = (const __nv_bfloat16 *)lw.ptrs[2];
            const __nv_bfloat16 *v_w  = (const __nv_bfloat16 *)lw.ptrs[3];
            const __nv_bfloat16 *q_nw = (const __nv_bfloat16 *)lw.ptrs[4];
            const __nv_bfloat16 *k_nw = (const __nv_bfloat16 *)lw.ptrs[5];
            const __nv_bfloat16 *o_w  = (const __nv_bfloat16 *)lw.ptrs[6];
            const __nv_bfloat16 *post_norm = (const __nv_bfloat16 *)lw.ptrs[7];
            const __nv_bfloat16 *gate_w = (const __nv_bfloat16 *)lw.ptrs[8];
            const __nv_bfloat16 *up_w   = (const __nv_bfloat16 *)lw.ptrs[9];
            const __nv_bfloat16 *down_w = (const __nv_bfloat16 *)lw.ptrs[10];

            // Q / K / V projections
            gemm_bf16_bf16(normalized, HIDDEN, q_w, proj_buf,  S, FA_QPROJ_SIZE);
            gemm_bf16_bf16(normalized, HIDDEN, k_w, proj_buf2, S, FA_KV_SIZE);
            gemm_bf16_bf16(normalized, HIDDEN, v_w, attn_buf,  S, FA_KV_SIZE);
            grid.sync();

            // LoRA: q, k, v
            APPLY_LORA(normalized, proj_buf,  lora.fa_q_A, lora.fa_q_B, fa_idx, HIDDEN, FA_QPROJ_SIZE);
            APPLY_LORA(normalized, proj_buf2, lora.fa_k_A, lora.fa_k_B, fa_idx, HIDDEN, FA_KV_SIZE);
            APPLY_LORA(normalized, attn_buf,  lora.fa_v_A, lora.fa_v_B, fa_idx, HIDDEN, FA_KV_SIZE);

            // QK norm + RoPE + KV cache write
            phase_fa_qk_norm_rope(
                proj_buf, proj_buf2, attn_buf, q_nw, k_nw,
                fa_k_cache + fa_idx * fa_stride,
                fa_v_cache + fa_idx * fa_stride,
                S, max_seq);
            grid.sync();

            // Causal attention
            phase_fa_causal_attn(proj_buf, proj_buf2, attn_buf, dn_out_buf, S);
            grid.sync();

            // O projection + residual add
            gemm_bf16_add_residual(dn_out_buf, FA_Q_SIZE, o_w, residual, hidden, S, HIDDEN);
            grid.sync();
            APPLY_LORA(dn_out_buf, hidden, lora.fa_o_A, lora.fa_o_B, fa_idx, FA_Q_SIZE, HIDDEN);

            // Post-attn norm
            phase_rmsnorm(hidden, post_norm, normalized, residual, S, HIDDEN);
            grid.sync();

            // gate/up
            gemm_bf16_bf16(normalized, HIDDEN, gate_w, proj_buf,  S, INTER);
            gemm_bf16_bf16(normalized, HIDDEN, up_w,   proj_buf2, S, INTER);
            grid.sync();
            APPLY_LORA(normalized, proj_buf,  lora.fa_gate_A, lora.fa_gate_B, fa_idx, HIDDEN, INTER);
            APPLY_LORA(normalized, proj_buf2, lora.fa_up_A,   lora.fa_up_B,   fa_idx, HIDDEN, INTER);

            phase_silu_mul(proj_buf, proj_buf2, mlp_buf, S * INTER);
            grid.sync();

            gemm_bf16_add_residual(mlp_buf, INTER, down_w, residual, hidden, S, HIDDEN);
            grid.sync();
            APPLY_LORA(mlp_buf, hidden, lora.fa_down_A, lora.fa_down_B, fa_idx, INTER, HIDDEN);

            fa_idx++;
        }
    }
    #undef APPLY_LORA

    // ------ final norm (last token only) -----------------------------------
    phase_final_norm_last(hidden, final_norm_w, final_normed, S);
    grid.sync();

    // ------ LM head + argmax -----------------------------------------------
    phase_lm_head_argmax(final_normed, lm_head_w, lm_bmv, lm_bmi, out_token);
    grid.sync();

    phase_lm_final_reduce(lm_bmv, lm_bmi, out_token, gridDim.x);
}

// ============================================================================
// Host launch.
// ============================================================================

extern "C" void launch_prefill_megakernel(
    const int *token_ids, int *out_token,
    const void *embed,
    const LayerWeights *layers,
    const void *final_norm_w, const void *lm_head_w,
    LoraSet lora,
    int lora_rank, float lora_scaling, float *lora_h_ws,
    void *fa_k_cache, void *fa_v_cache,
    float *dn_states, float *conv_bufs,
    void *hidden, void *residual, void *normalized,
    void *proj_buf, void *proj_buf2,
    void *attn_buf, void *mlp_buf, void *dn_out_buf,
    float *beta_buf, float *alpha_buf,
    void *final_normed,
    float *lm_bmv, int *lm_bmi,
    int S, int max_seq,
    cudaStream_t stream)
{
    // Dynamic shared memory:
    //   s_A:     2 × 16 × 32 × 2 = 2048 bytes
    //   s_B:     2 × 256 × 32 × 2 = 32768 bytes
    //   s_store: NW × 256 × 4 = 16384 bytes
    //   total:   51200 bytes (50 KB). >48 KB default needs opt-in.
    constexpr size_t smem_bytes =
        2 * 16 * 32 * sizeof(__nv_bfloat16) +
        2 * 256 * 32 * sizeof(__nv_bfloat16) +
        (size_t)NW * 256 * sizeof(float);

    static bool smem_configured = false;
    if (!smem_configured) {
        cudaFuncSetAttribute(
            (void *)prefill_megakernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem_bytes);
        smem_configured = true;
    }

    // Cooperative launch requires ALL blocks concurrently resident.
    // Query occupancy (WMMA register pressure can drop it below 1/SM) and
    // clamp. Printed on first call for visibility.
    static int safe_num_blocks = 0;
    if (safe_num_blocks == 0) {
        int dev;
        cudaGetDevice(&dev);
        int sm_count;
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
        int blocks_per_sm = 0;
        cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, (void *)prefill_megakernel,
            PM_BLOCK_SIZE, smem_bytes);
        safe_num_blocks = sm_count * blocks_per_sm;
        (void)err;
        if (safe_num_blocks < 1) safe_num_blocks = 1;
        if (safe_num_blocks > PM_NUM_BLOCKS) safe_num_blocks = PM_NUM_BLOCKS;
    }
    dim3 grid(safe_num_blocks);
    dim3 block(PM_BLOCK_SIZE);

    void *args[] = {
        (void *)&token_ids, (void *)&out_token,
        (void *)&embed, (void *)&layers,
        (void *)&final_norm_w, (void *)&lm_head_w,
        (void *)&lora,
        (void *)&lora_rank, (void *)&lora_scaling, (void *)&lora_h_ws,
        (void *)&fa_k_cache, (void *)&fa_v_cache,
        (void *)&dn_states, (void *)&conv_bufs,
        (void *)&hidden, (void *)&residual, (void *)&normalized,
        (void *)&proj_buf, (void *)&proj_buf2,
        (void *)&attn_buf, (void *)&mlp_buf, (void *)&dn_out_buf,
        (void *)&beta_buf, (void *)&alpha_buf,
        (void *)&final_normed,
        (void *)&lm_bmv, (void *)&lm_bmi,
        (void *)&S, (void *)&max_seq,
    };

    cudaLaunchCooperativeKernel(
        (void *)prefill_megakernel, grid, block, args, smem_bytes, stream);
}

// ============================================================================
// Fused AdamW — single kernel updates an arbitrary-size flat parameter
// buffer. The training wrapper can call it once on a concatenated view of
// every LoRA A/B pair to get a single-dispatch optimizer step.
// ============================================================================

extern "C" __global__ void fused_adamw_kernel(
    __nv_bfloat16 *__restrict__ params,    // bf16 parameters, updated in place
    float *__restrict__ m,                  // fp32 first-moment state
    float *__restrict__ v,                  // fp32 second-moment state
    const float *__restrict__ grad,         // fp32 gradients
    long long numel, int step,
    float lr, float beta1, float beta2, float eps, float wd,
    float bias_correction1, float bias_correction2)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = tid; i < numel; i += stride) {
        float g = grad[i];
        float mi = beta1 * m[i] + (1.0f - beta1) * g;
        float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
        m[i] = mi;
        v[i] = vi;
        float m_hat = mi / bias_correction1;
        float v_hat = vi / bias_correction2;
        float p = __bfloat162float(params[i]);
        p -= lr * (m_hat / (sqrtf(v_hat) + eps) + wd * p);
        params[i] = __float2bfloat16(p);
    }
}

extern "C" void launch_fused_adamw(
    void *params_bf16, float *m, float *v, const float *grad,
    long long numel, int step,
    float lr, float beta1, float beta2, float eps, float wd,
    cudaStream_t stream)
{
    const float bc1 = 1.0f - powf(beta1, (float)step);
    const float bc2 = 1.0f - powf(beta2, (float)step);

    int block_size = 256;
    int grid_size = (int)((numel + block_size - 1) / block_size);
    if (grid_size > 65535 * 2) grid_size = 65535 * 2;   // cap for large models
    if (grid_size < 1) grid_size = 1;

    fused_adamw_kernel<<<grid_size, block_size, 0, stream>>>(
        (__nv_bfloat16 *)params_bf16, m, v, grad,
        numel, step, lr, beta1, beta2, eps, wd, bc1, bc2);
}
