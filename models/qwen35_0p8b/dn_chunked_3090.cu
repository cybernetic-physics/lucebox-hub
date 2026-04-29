/**
 * DeltaNet chunked forward — bf16 tensor cores via nvcuda::wmma.
 *
 * SM86 (RTX 3090) variant. Differs from trainer/dn_chunked.cu by
 * splitting the Dv state across V_SPLITS=4 blocks per head so the
 * shared memory fits in 99 KB (the 3090 opt-in cap; B200 has 228 KB).
 *
 * Tile sizes:
 *   V_SPLITS = 4   -> Dv_block = Dv / V_SPLITS = 32
 *   C        = 32  -> half the chunk size of the B200 kernel
 *   Grid     = (H * V_SPLITS,)  (= 64 blocks for Qwen3.5-0.8B's 16 DN heads)
 *   Block    = 256 threads = 8 warps
 *
 * Shared memory (~ 85 KB, fits the 99 KB SM86 opt-in limit):
 *   state[Dk * Dv_block]      fp32   16 KB    persistent state slice
 *   buf_attn[C*C]             fp32    4 KB    attn0 -> T -> attn_in
 *   buf_decay[C*C]            fp32    4 KB    decay_mask
 *   warp_scratch[8*256]       fp32    8 KB    wmma_gemm_bf16 staging
 *   s_g_cs / s_exp_cs / s_beta fp32  3*128 B  per-chunk scalars
 *   state_bf[Dk * Dv_block]   bf16    8 KB    state cast for tensor cores
 *   buf_q[C * Dk]             bf16    8 KB
 *   buf_k[C * Dk]             bf16    8 KB
 *   buf_kbeta[C * Dk]         bf16    8 KB
 *   buf_kcd[C * Dk]           bf16    8 KB    (also used as [C, Dv_block])
 *   buf_vbeta[C * Dv_block]   bf16    2 KB
 *   buf_attn_bf[C*C]          bf16    2 KB
 *
 * Ports the algorithm from trainer/dn_chunked.cu line-for-line; the only
 * structural change is V-splitting the state and v columns across blocks
 * (V_SPLITS dimension introduced into blockIdx.x and into the per-block
 * Dv-axis bounds).
 *
 * Inputs:
 *   q, k, v   : packed bf16, layout [S, DN_CONV_CH] for prefill (we use
 *               qkd_pos_stride / v_pos_stride to address per-head q/k/v
 *               within the packed row).
 *   beta_h    : fp32 sigmoid(beta), shape [S, H] strided by bd_pos_stride.
 *   g_h       : fp32 log(decay), shape [S, H] strided by bd_pos_stride.
 *   state_in  : fp32 [H, Dk, Dv]  — full state coming in.
 *   y_out     : bf16 [S, H, Dv]   — full output written back per v-split.
 *
 * Caller is responsible for converting alpha_buf (= decay = exp(g)) into
 * g via pf_decay_to_g_inplace before launching this kernel.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>

using namespace nvcuda::wmma;


#ifndef DN3090_DK
#define DN3090_DK 128
#endif
#ifndef DN3090_DV
#define DN3090_DV 128
#endif
#ifndef DN3090_C
#define DN3090_C 32
#endif
#ifndef DN3090_VSPLITS
#define DN3090_VSPLITS 4
#endif

constexpr int DN3090_DV_BLOCK = DN3090_DV / DN3090_VSPLITS;
static_assert(DN3090_DV_BLOCK >= 16, "Dv_block must be at least 16 for wmma");
static_assert(DN3090_DV_BLOCK % 16 == 0, "Dv_block must be a multiple of 16");
static_assert(DN3090_C % 16 == 0, "C must be a multiple of 16");
static_assert(DN3090_DK % 16 == 0, "Dk must be a multiple of 16");


// ===================== WMMA gemm primitives =====================
// Matches trainer/dn_chunked.cu's wmma_gemm helpers but kept local so the
// inference module can be built independently.

template <int M, int N, int K, bool A_COL = false, bool B_COL = false>
__device__ inline void wmma_gemm_fp32(
    const __nv_bfloat16 *A, int ld_a,
    const __nv_bfloat16 *B, int ld_b,
    float *C, int ld_c,
    int warp_id, int n_warps,
    bool accumulate)
{
    constexpr int TM = 16, TN = 16, TK = 16;
    constexpr int M_TILES = M / TM;
    constexpr int N_TILES = N / TN;
    constexpr int K_TILES = K / TK;

    int n_out_tiles = M_TILES * N_TILES;
    for (int t = warp_id; t < n_out_tiles; t += n_warps) {
        int mt = t / N_TILES;
        int nt = t - mt * N_TILES;

        fragment<accumulator, TM, TN, TK, float> c_frag;
        if (accumulate) {
            load_matrix_sync(c_frag, C + mt * TM * ld_c + nt * TN, ld_c, mem_row_major);
        } else {
            fill_fragment(c_frag, 0.0f);
        }

        for (int kt = 0; kt < K_TILES; kt++) {
            using ALayout = std::conditional_t<A_COL, col_major, row_major>;
            using BLayout = std::conditional_t<B_COL, col_major, row_major>;
            fragment<matrix_a, TM, TN, TK, __nv_bfloat16, ALayout> a_frag;
            fragment<matrix_b, TM, TN, TK, __nv_bfloat16, BLayout> b_frag;
            // For row_major matrix_a [M, K] with leading dim ld_a (= K stride):
            //   tile (mt, kt) top-left at offset mt*TM*ld_a + kt*TK.
            // For col_major matrix_a [M, K] with leading dim ld_a (= M stride):
            //   A[m, k] = ptr[m + k*ld_a], tile top-left at mt*TM + kt*TK*ld_a.
            const __nv_bfloat16 *a_ptr = A_COL
                ? A + mt * TM + kt * TK * ld_a
                : A + mt * TM * ld_a + kt * TK;
            // For row_major matrix_b [K, N] with leading dim ld_b (= N stride):
            //   tile (kt, nt) top-left at kt*TK*ld_b + nt*TN.
            // For col_major matrix_b [K, N] with leading dim ld_b (= K stride):
            //   B[k, n] = ptr[k + n*ld_b], tile top-left at kt*TK + nt*TN*ld_b.
            const __nv_bfloat16 *b_ptr = B_COL
                ? B + kt * TK + nt * TN * ld_b
                : B + kt * TK * ld_b + nt * TN;
            load_matrix_sync(a_frag, a_ptr, ld_a);
            load_matrix_sync(b_frag, b_ptr, ld_b);
            mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        store_matrix_sync(C + mt * TM * ld_c + nt * TN, c_frag, ld_c, mem_row_major);
    }
}

// Same but C is bf16 — cast through fp32 fragment via warp_scratch.
template <int M, int N, int K>
__device__ inline void wmma_gemm_bf16(
    const __nv_bfloat16 *A, int ld_a,
    const __nv_bfloat16 *B, int ld_b,
    __nv_bfloat16 *C, int ld_c,
    int warp_id, int n_warps,
    float *warp_scratch /* [n_warps * 16 * 16] */)
{
    constexpr int TM = 16, TN = 16, TK = 16;
    constexpr int M_TILES = M / TM;
    constexpr int N_TILES = N / TN;
    constexpr int K_TILES = K / TK;

    int n_out_tiles = M_TILES * N_TILES;
    for (int t = warp_id; t < n_out_tiles; t += n_warps) {
        int mt = t / N_TILES;
        int nt = t - mt * N_TILES;

        fragment<accumulator, TM, TN, TK, float> c_frag;
        fill_fragment(c_frag, 0.0f);
        for (int kt = 0; kt < K_TILES; kt++) {
            fragment<matrix_a, TM, TN, TK, __nv_bfloat16, row_major> a_frag;
            fragment<matrix_b, TM, TN, TK, __nv_bfloat16, row_major> b_frag;
            load_matrix_sync(a_frag, A + mt * TM * ld_a + kt * TK, ld_a);
            load_matrix_sync(b_frag, B + kt * TK * ld_b + nt * TN, ld_b);
            mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        // Cast fp32 fragment -> bf16 via shared-mem scratch.
        float *scratch = warp_scratch + warp_id * 256;
        store_matrix_sync(scratch, c_frag, 16, mem_row_major);
        __syncwarp();
        int lane = threadIdx.x & 31;
        for (int idx = lane; idx < 256; idx += 32) {
            int i = idx >> 4;
            int j = idx & 15;
            C[(mt * TM + i) * ld_c + (nt * TN + j)] = __float2bfloat16(scratch[idx]);
        }
        __syncwarp();
    }
}


// ===================== chunked fwd kernel (V-split) =====================
__global__ void __launch_bounds__(256, 1)
dn_chunked_3090_fwd_kernel(
    const __nv_bfloat16 *__restrict__ q_base,
    const __nv_bfloat16 *__restrict__ k_base,
    const __nv_bfloat16 *__restrict__ v_base,
    const float *__restrict__ beta_base,
    const float *__restrict__ g_base,
    const float *__restrict__ state_in_base,
    __nv_bfloat16 *__restrict__ y_base,
    float *__restrict__ state_out_base,
    int S,
    int qkd_pos_stride,        // stride from token t to t+1 in q/k (= DN_CONV_CH for prefill packed buf)
    int v_pos_stride,          // stride for v
    int bd_pos_stride,         // stride for beta/g (= DN_HEADS)
    int y_pos_stride)          // stride for y output
{
    constexpr int Dk = DN3090_DK;
    constexpr int Dv = DN3090_DV;
    constexpr int Dvb = DN3090_DV_BLOCK;
    constexpr int C  = DN3090_C;

    int tid = threadIdx.x;
    int nt  = blockDim.x;
    int warp = tid >> 5;
    int n_warps = nt >> 5;

    int block_id = blockIdx.x;
    int h  = block_id / DN3090_VSPLITS;
    int vs = block_id - h * DN3090_VSPLITS;
    int v_off = vs * Dvb;

    // Per-head views into [S, ...]-strided tensors. q/k are [S, H*Dk]
    // packed (or wider in prefill); v is [S, H*Dv]. Caller picks the
    // base pointer for head h.
    const __nv_bfloat16 *q  = q_base  + h * Dk;
    const __nv_bfloat16 *k  = k_base  + h * Dk;
    const __nv_bfloat16 *v  = v_base  + h * Dv;
    const float *beta_h     = beta_base + h;
    const float *g_h        = g_base    + h;
    const float *state_in_h = state_in_base + h * Dk * Dv;
    __nv_bfloat16 *y_h      = y_base + h * Dv;
    float *state_out_h      = state_out_base ? state_out_base + h * Dk * Dv : nullptr;

    int n_chunks = (S + C - 1) / C;

    // ----- shared memory layout -----
    extern __shared__ unsigned char smem_raw[];
    float *state         = (float*)smem_raw;                            // [Dk, Dvb]      fp32
    float *buf_attn      = state    + Dk * Dvb;                         // [C, C]         fp32
    float *buf_decay     = buf_attn + C * C;                            // [C, C]         fp32
    float *warp_scratch  = buf_decay + C * C;                           // [n_warps * 256] fp32
    float *s_g_cs        = warp_scratch + 8 * 256;                      // [C]
    float *s_exp_cs      = s_g_cs   + C;                                // [C]
    float *s_beta        = s_exp_cs + C;                                // [C]
    __nv_bfloat16 *bf16_base = (__nv_bfloat16*)(s_beta + C);
    __nv_bfloat16 *state_bf  = bf16_base;                               // [Dk, Dvb]   bf16
    __nv_bfloat16 *buf_q     = state_bf + Dk * Dvb;                     // [C, Dk]
    __nv_bfloat16 *buf_k     = buf_q    + C * Dk;                       // [C, Dk]
    __nv_bfloat16 *buf_kbeta = buf_k    + C * Dk;                       // [C, Dk]
    __nv_bfloat16 *buf_kcd   = buf_kbeta + C * Dk;                      // [C, Dk] (also [C, Dvb])
    __nv_bfloat16 *buf_vbeta = buf_kcd  + C * Dk;                       // [C, Dvb]
    __nv_bfloat16 *buf_attn_bf = buf_vbeta + C * Dvb;                   // [C, C]

    // ----- Load initial state slice -----
    // state_in_h is [Dk, Dv]; we own columns [v_off, v_off+Dvb).
    for (int i = tid; i < Dk * Dvb; i += nt) {
        int row = i / Dvb;
        int col = i - row * Dvb;
        state[row * Dvb + col] = state_in_h[row * Dv + v_off + col];
    }
    __syncthreads();


    for (int c = 0; c < n_chunks; c++) {
        int t0 = c * C;
        int chunk_len = (S - t0 < C) ? (S - t0) : C;

        // ----- Load q, k for chunk; pad with zeros if chunk_len < C -----
        for (int idx = tid; idx < C * Dk; idx += nt) {
            int j = idx / Dk;
            int d = idx - j * Dk;
            __nv_bfloat16 vq = (j < chunk_len) ? q[(t0 + j) * qkd_pos_stride + d]
                                               : __float2bfloat16(0.0f);
            __nv_bfloat16 vk = (j < chunk_len) ? k[(t0 + j) * qkd_pos_stride + d]
                                               : __float2bfloat16(0.0f);
            buf_q[idx] = vq;
            buf_k[idx] = vk;
        }
        // Load v slice (only Dvb columns).
        for (int idx = tid; idx < C * Dvb; idx += nt) {
            int j = idx / Dvb;
            int d = idx - j * Dvb;
            buf_vbeta[idx] = (j < chunk_len) ? v[(t0 + j) * v_pos_stride + v_off + d]
                                             : __float2bfloat16(0.0f);
        }
        // Load beta and g into scalar buffers (one slot per chunk-step).
        if (tid < C) {
            float bj = (tid < chunk_len) ? beta_h[(t0 + tid) * bd_pos_stride] : 0.0f;
            float gj = (tid < chunk_len) ? g_h   [(t0 + tid) * bd_pos_stride] : 0.0f;
            s_beta[tid] = bj;
            s_g_cs[tid] = gj;
        }
        __syncthreads();

        // Cumsum g over the chunk; thread 0 does it sequentially (C=32).
        if (tid == 0) {
            float acc = 0.0f;
            for (int i = 0; i < C; i++) {
                acc += s_g_cs[i];
                s_g_cs[i] = acc;
            }
        }
        __syncthreads();
        if (tid < C) s_exp_cs[tid] = expf(s_g_cs[tid]);
        __syncthreads();

        // ----- decay_mask[i, j] = exp(g_cs[i] - g_cs[j]) for i >= j else 0 -----
        for (int idx = tid; idx < C * C; idx += nt) {
            int i = idx / C;
            int j = idx - i * C;
            if (i >= j && i < chunk_len && j < chunk_len) {
                buf_decay[idx] = expf(s_g_cs[i] - s_g_cs[j]);
            } else {
                buf_decay[idx] = 0.0f;
            }
        }
        __syncthreads();

        // ----- k_beta[j, d] = k[j, d] * beta[j];  v_beta likewise -----
        for (int idx = tid; idx < C * Dk; idx += nt) {
            int j = idx / Dk;
            float bj = (j < chunk_len) ? s_beta[j] : 0.0f;
            float kk = __bfloat162float(buf_k[idx]);
            buf_kbeta[idx] = __float2bfloat16(bj * kk);
        }
        for (int idx = tid; idx < C * Dvb; idx += nt) {
            int j = idx / Dvb;
            float bj = (j < chunk_len) ? s_beta[j] : 0.0f;
            float vv = __bfloat162float(buf_vbeta[idx]);
            buf_vbeta[idx] = __float2bfloat16(bj * vv);
        }
        __syncthreads();

        // ----- attn0 = -(k_beta @ k.T) -----
        wmma_gemm_fp32<C, C, Dk, false, true>(
            buf_kbeta, Dk, buf_k, Dk, buf_attn, C, warp, n_warps, false);
        __syncthreads();

        for (int idx = tid; idx < C * C; idx += nt) {
            int i = idx / C, j = idx - i * C;
            if (i > j) buf_attn[idx] = -buf_attn[idx] * buf_decay[idx];
            else       buf_attn[idx] = 0.0f;
        }
        __syncthreads();

        // ----- T = (I - tril(attn0))^{-1} via in-place sequential update on buf_attn -----
        // Same algorithm as the B200 kernel; uses s_exp_cs as scratch (C=32 floats).
        for (int i = 1; i < C; i++) {
            for (int j = tid; j < i; j += nt) {
                float orig = buf_attn[i * C + j];
                float acc = 0.0f;
                for (int l = j + 1; l < i; l++) {
                    acc += buf_attn[i * C + l] * buf_attn[l * C + j];
                }
                s_exp_cs[j] = orig + acc;
            }
            __syncthreads();
            for (int j = tid; j < i; j += nt) {
                buf_attn[i * C + j] = s_exp_cs[j];
            }
            __syncthreads();
        }
        // Add identity: T = attn + I
        for (int idx = tid; idx < C; idx += nt) {
            buf_attn[idx * C + idx] += 1.0f;
        }
        // Restore exp_cs (we trampled it).
        if (tid < C) s_exp_cs[tid] = expf(s_g_cs[tid]);
        __syncthreads();
        // Cast T (fp32) -> bf16 in buf_attn_bf.
        for (int idx = tid; idx < C * C; idx += nt) {
            buf_attn_bf[idx] = __float2bfloat16(buf_attn[idx]);
        }
        __syncthreads();

        // ----- v_new = T @ v_beta -----  (writes to buf_kcd[:, :Dvb] then copy back)
        wmma_gemm_bf16<C, Dvb, C>(
            buf_attn_bf, C, buf_vbeta, Dvb, buf_kcd, Dvb, warp, n_warps, warp_scratch);
        __syncthreads();

        // Copy buf_kcd -> buf_vbeta as the new v_new.
        for (int idx = tid; idx < C * Dvb; idx += nt) buf_vbeta[idx] = buf_kcd[idx];
        __syncthreads();

        // ----- k_cd = T @ (k_beta * exp_g_cs) -----
        for (int idx = tid; idx < C * Dk; idx += nt) {
            int j = idx / Dk;
            float ec = (j < chunk_len) ? s_exp_cs[j] : 0.0f;
            float kb = __bfloat162float(buf_kbeta[idx]);
            buf_kbeta[idx] = __float2bfloat16(kb * ec);
        }
        __syncthreads();
        wmma_gemm_bf16<C, Dk, C>(
            buf_attn_bf, C, buf_kbeta, Dk, buf_kcd, Dk, warp, n_warps, warp_scratch);
        __syncthreads();

        // ----- v_prime = k_cd @ state[:, v_off..v_off+Dvb]; v_new -= v_prime  -----
        // State slice is [Dk, Dvb] (8 KB bf16). Whole slice fits — no two-halves.
        for (int idx = tid; idx < Dk * Dvb; idx += nt) {
            state_bf[idx] = __float2bfloat16(state[idx]);
        }
        __syncthreads();
        // [C, Dvb] = k_cd [C, Dk] @ state_bf [Dk, Dvb]
        wmma_gemm_fp32<C, Dvb, Dk>(
            buf_kcd, Dk, state_bf, Dvb, buf_attn, Dvb,
            warp, n_warps, false);
        __syncthreads();
        // v_new -= buf_attn (fp32)
        for (int idx = tid; idx < C * Dvb; idx += nt) {
            float v_old = __bfloat162float(buf_vbeta[idx]);
            buf_vbeta[idx] = __float2bfloat16(v_old - buf_attn[idx]);
        }
        __syncthreads();

        // ----- attn_in = q @ k.T * decay_mask  -----
        wmma_gemm_fp32<C, C, Dk, false, true>(
            buf_q, Dk, buf_k, Dk, buf_attn, C, warp, n_warps, false);
        __syncthreads();
        for (int idx = tid; idx < C * C; idx += nt) {
            int i = idx / C, j = idx - i * C;
            if (i >= j && i < chunk_len && j < chunk_len) {
                buf_attn[idx] *= buf_decay[idx];
            } else {
                buf_attn[idx] = 0.0f;
            }
        }
        __syncthreads();
        for (int idx = tid; idx < C * C; idx += nt) {
            buf_attn_bf[idx] = __float2bfloat16(buf_attn[idx]);
        }
        __syncthreads();

        // ----- attn_int = (q * exp_g_cs) @ state[:, v_off..]  -----
        for (int idx = tid; idx < C * Dk; idx += nt) {
            int j = idx / Dk;
            float ec = (j < chunk_len) ? s_exp_cs[j] : 0.0f;
            float qq = __bfloat162float(buf_q[idx]);
            buf_q[idx] = __float2bfloat16(qq * ec);
        }
        __syncthreads();
        // attn_int [C, Dvb] = buf_q [C, Dk] @ state_bf [Dk, Dvb]
        // (state_bf is still the cast slice from above — we didn't overwrite it.)
        // Output to buf_kcd reused (was k_cd; fits [C, Dvb] = 2 KB).
        wmma_gemm_fp32<C, Dvb, Dk>(
            buf_q, Dk, state_bf, Dvb, buf_attn, Dvb,
            warp, n_warps, false);
        __syncthreads();
        for (int idx = tid; idx < C * Dvb; idx += nt) {
            buf_kcd[idx] = __float2bfloat16(buf_attn[idx]);  // attn_int_bf
        }
        __syncthreads();

        // ----- y_chunk = attn_int + attn_in_bf @ v_new -----
        wmma_gemm_fp32<C, Dvb, C>(
            buf_attn_bf, C, buf_vbeta, Dvb, buf_attn, Dvb,
            warp, n_warps, false);
        __syncthreads();
        for (int idx = tid; idx < C * Dvb; idx += nt) {
            int j = idx / Dvb;
            int d = idx - j * Dvb;
            if (j >= chunk_len) continue;
            float ai = __bfloat162float(buf_kcd[idx]);
            float ax = buf_attn[idx];
            y_h[(t0 + j) * y_pos_stride + v_off + d] = __float2bfloat16(ai + ax);
        }
        __syncthreads();

        // ----- state_slice = state_slice * exp(g_total) + (k * exp(g_total - g_cs))^T @ v_new -----
        float g_total = s_g_cs[chunk_len - 1];
        float exp_total = expf(g_total);
        for (int idx = tid; idx < Dk * Dvb; idx += nt) state[idx] *= exp_total;
        __syncthreads();
        // k_decay[j, d] = k[j, d] * exp(g_total - g_cs[j])  (reuse buf_kbeta)
        for (int idx = tid; idx < C * Dk; idx += nt) {
            int j = idx / Dk;
            float scale = (j < chunk_len) ? expf(g_total - s_g_cs[j]) : 0.0f;
            float kk = __bfloat162float(buf_k[idx]);
            buf_kbeta[idx] = __float2bfloat16(kk * scale);
        }
        __syncthreads();
        // state_slice += k_decay.T @ v_new
        wmma_gemm_fp32<Dk, Dvb, C, true, false>(
            buf_kbeta, Dk, buf_vbeta, Dvb, state, Dvb, warp, n_warps, true);
        __syncthreads();
    }

    // Persist final state slice into state_out (caller owns layout).
    if (state_out_h) {
        for (int i = tid; i < Dk * Dvb; i += nt) {
            int row = i / Dvb;
            int col = i - row * Dvb;
            state_out_h[row * Dv + v_off + col] = state[i];
        }
    }
}


// =================== launcher (host-callable) ====================
// Caller arranges:
//   - g_base contains log-decay (run pf_decay_to_g_inplace first if you have decay).
//   - state_in_base / state_out_base: [H, Dk, Dv] fp32 row-major.
//   - q_base / k_base point at the per-head q/k row (i.e. base + offset_for_head_0
//     of the packed qkv buffer).
extern "C" void launch_dn_chunked_3090(
    const void *q_base_v,
    const void *k_base_v,
    const void *v_base_v,
    const float *beta_base,
    const float *g_base,
    const float *state_in_base,
    void *y_base_v,
    float *state_out_base,
    int S, int H,
    int qkd_pos_stride, int v_pos_stride, int bd_pos_stride, int y_pos_stride,
    cudaStream_t stream)
{
    const __nv_bfloat16 *q_base = (const __nv_bfloat16 *)q_base_v;
    const __nv_bfloat16 *k_base = (const __nv_bfloat16 *)k_base_v;
    const __nv_bfloat16 *v_base = (const __nv_bfloat16 *)v_base_v;
    __nv_bfloat16 *y_base       = (__nv_bfloat16 *)y_base_v;
    constexpr int Dk  = DN3090_DK;
    constexpr int Dv  = DN3090_DV;
    constexpr int Dvb = DN3090_DV_BLOCK;
    constexpr int C   = DN3090_C;
    constexpr int n_warps = 8;

    size_t smem_fp32 = ((size_t)Dk * Dvb + 2 * (size_t)C * C + n_warps * 256 + 3 * C) * sizeof(float);
    size_t smem_bf16 = ((size_t)Dk * Dvb + 4 * (size_t)C * Dk + (size_t)C * Dvb + (size_t)C * C) * sizeof(__nv_bfloat16);
    size_t smem = smem_fp32 + smem_bf16;

    cudaError_t set_err = cudaFuncSetAttribute(
        dn_chunked_3090_fwd_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem);
    if (set_err != cudaSuccess) {
        printf("[dn_chunked_3090] cudaFuncSetAttribute failed: %s (smem=%zu)\n",
               cudaGetErrorString(set_err), smem);
    }

    int threads = n_warps * 32;
    dim3 grid(H * DN3090_VSPLITS);
    dn_chunked_3090_fwd_kernel<<<grid, threads, smem, stream>>>(
        q_base, k_base, v_base, beta_base, g_base,
        state_in_base, y_base, state_out_base,
        S, qkd_pos_stride, v_pos_stride, bd_pos_stride, y_pos_stride);
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        printf("[dn_chunked_3090] launch failed: %s (grid=%d, threads=%d, smem=%zu)\n",
               cudaGetErrorString(launch_err), H * DN3090_VSPLITS, threads, smem);
    }
}


// In-place log: alpha[i] = logf(alpha[i]).  Used to convert the prefill
// alpha_buf (= decay = exp(g)) into g for the chunked kernel.
extern "C" __global__ void pf_decay_to_g_inplace_kernel(float *buf, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) buf[i] = logf(buf[i]);
}

extern "C" void launch_pf_decay_to_g_inplace(float *buf, int N, cudaStream_t stream) {
    int blk = 256;
    int grid = (N + blk - 1) / blk;
    pf_decay_to_g_inplace_kernel<<<grid, blk, 0, stream>>>(buf, N);
}
