/**
 * Parallel-scan DeltaNet forward — three-phase pipeline.
 *
 * Phase A: extract per-chunk linear operators (M_chunk, B_chunk) such that
 *
 *     state_after_chunk_c = M_chunk_c @ state_before_chunk_c + B_chunk_c
 *
 *   where (derivation in dn_parallel_scan_proto.py):
 *
 *     M_chunk = exp_g_total * I - k_decay.T @ T @ k_cd        [Dk x Dk]
 *     B_chunk = k_decay.T @ T @ v_beta                         [Dk x Dv]
 *
 *   Grid: (H, n_chunks). Per block: process one chunk's (k, v, beta, g)
 *   inputs, compute the operator, write to global scratch.
 *
 * Phase D: Hillis-Steele parallel prefix scan over (M, B). Each stage `d`
 *   composes (A2, B2) ∘ (A1, B1) = (A2 @ A1, A2 @ B1 + B2) at stride d
 *   across chunks. log2(n_chunks) stages with grid sync between.
 *
 * Phase C: per-chunk output. Each block re-derives intra-chunk T/k_cd/...,
 *   reads its prefix state (= M_prefix_{c-1} @ state_0 + B_prefix_{c-1};
 *   for c=0 it's state_0), computes y_chunk = attn_int + attn_in @ v_new,
 *   writes to dn_out_buf. The last chunk also writes the final state back
 *   to state_out for the next call (decode or next prefill).
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <mma.h>

namespace cg = cooperative_groups;
using namespace nvcuda::wmma;

#ifndef DN_KEY_DEFAULT
#define DN_KEY_DEFAULT 128
#endif
#ifndef DN_VAL_DEFAULT
#define DN_VAL_DEFAULT 128
#endif
#ifndef DN_CHUNK
#define DN_CHUNK 64
#endif

namespace ps {

constexpr int Dk = DN_KEY_DEFAULT;
constexpr int Dv = DN_VAL_DEFAULT;
constexpr int C  = DN_CHUNK;

// =====================================================================
// WMMA gemm helpers (m16n16k16, bf16 inputs, fp32 accumulator).
// Distributed by warp_id across [M_TILES, N_TILES] output tiles.
// =====================================================================
template <int M, int N, int K, bool A_COL = false, bool B_COL = false>
__device__ inline void wmma_gemm_fp32(
    const __nv_bfloat16 *A, int ld_a,
    const __nv_bfloat16 *B, int ld_b,
    float *C, int ld_c,
    int warp_id, int n_warps,
    bool accumulate)
{
    constexpr int TM = 16, TN = 16, TK = 16;
    constexpr int M_TILES = (M + TM - 1) / TM;
    constexpr int N_TILES = (N + TN - 1) / TN;
    constexpr int K_TILES = (K + TK - 1) / TK;
    int n_out_tiles = M_TILES * N_TILES;
    for (int t = warp_id; t < n_out_tiles; t += n_warps) {
        int mt = t / N_TILES;
        int nt = t % N_TILES;
        fragment<accumulator, TM, TN, TK, float> c_frag;
        if (accumulate) {
            load_matrix_sync(c_frag, C + (mt * TM) * ld_c + (nt * TN), ld_c, mem_row_major);
        } else {
            fill_fragment(c_frag, 0.0f);
        }
        #pragma unroll
        for (int kt = 0; kt < K_TILES; kt++) {
            using A_layout = typename std::conditional<A_COL, col_major, row_major>::type;
            using B_layout = typename std::conditional<B_COL, col_major, row_major>::type;
            fragment<matrix_a, TM, TN, TK, __nv_bfloat16, A_layout> a_frag;
            fragment<matrix_b, TM, TN, TK, __nv_bfloat16, B_layout> b_frag;
            const __nv_bfloat16 *a_ptr = A_COL
                ? (A + (kt * TK) * ld_a + (mt * TM))
                : (A + (mt * TM) * ld_a + (kt * TK));
            const __nv_bfloat16 *b_ptr = B_COL
                ? (B + (nt * TN) * ld_b + (kt * TK))
                : (B + (kt * TK) * ld_b + (nt * TN));
            load_matrix_sync(a_frag, a_ptr, ld_a);
            load_matrix_sync(b_frag, b_ptr, ld_b);
            mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        store_matrix_sync(C + (mt * TM) * ld_c + (nt * TN), c_frag, ld_c, mem_row_major);
    }
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

// =====================================================================
// Phase A: per-chunk operator extraction.
//
// Input  (per (h, chunk_c) block):
//   q/k/v/beta/g — read straight out of qkv_proj/beta_buf/g_buf (no
//                   pre-transpose). q is unused in Phase A.
//
// Output (to global scratch):
//   M[h, c, :, :]  — [Dk, Dk] fp32
//   B[h, c, :, :]  — [Dk, Dv] fp32
//
// Algorithm (chunk-internal — same as dn_chunked.cu's chunk processing):
//   g_cs[i]    = sum_{j<=i} g[c0+j]
//   exp_g_cs[i]= exp(g_cs[i])
//   decay[i,j] = exp(g_cs[i] - g_cs[j])           [C, C]
//   k_beta[i]  = k[c0+i] * beta[c0+i]
//   v_beta[i]  = v[c0+i] * beta[c0+i]
//   attn0[i,j] = -(k_beta[i] @ k[c0+j].T)*decay[i,j], zero diag-and-above
//   T          = (I - tril(attn0))^-1, then T += I  (sequential row solve)
//   k_cd[i]    = k[c0+i] * exp_g_cs[i]
//   k_decay[i] = k[c0+i] * exp(g_total - g_cs[i])
// Then:
//   M = exp(g_total) * I - k_decay.T @ T @ k_cd     [Dk, Dk]
//   B = k_decay.T @ T @ v_beta                       [Dk, Dv]
// =====================================================================

__global__ void __launch_bounds__(512, 1)
phase_a_extract_kernel(
    const __nv_bfloat16 *__restrict__ q_base,   // unused, kept for arg symmetry
    const __nv_bfloat16 *__restrict__ k_base,
    const __nv_bfloat16 *__restrict__ v_base,
    const float *__restrict__ beta_base,
    const float *__restrict__ g_base,
    float *__restrict__ M_out,                  // [H, n_chunks, Dk, Dk]
    float *__restrict__ B_out,                  // [H, n_chunks, Dk, Dv]
    int S, int H, int n_chunks,
    int qkd_stride, int v_stride, int bd_stride)
{
    int H_idx = blockIdx.y;
    int c_idx = blockIdx.x;
    if (H_idx >= H || c_idx >= n_chunks) return;
    int t0 = c_idx * C;
    int chunk_len = (S - t0 < C) ? (S - t0) : C;

    int tid = threadIdx.x;
    int nt  = blockDim.x;
    int warp = tid >> 5;
    int n_warps = nt >> 5;
    int lane = tid & 31;

    // Per-(h) input slices.
    const __nv_bfloat16 *k = k_base + H_idx * Dk;
    const __nv_bfloat16 *v = v_base + H_idx * Dv;
    const float *beta_h    = beta_base + H_idx;
    const float *g_h       = g_base    + H_idx;

    // Output destinations for this (h, c).
    float *M_dst = M_out + ((size_t)H_idx * n_chunks + c_idx) * Dk * Dk;
    float *B_dst = B_out + ((size_t)H_idx * n_chunks + c_idx) * Dk * Dv;

    // ----- Shared memory -----
    extern __shared__ unsigned char smem_raw[];
    float *buf_attn      = (float*)smem_raw;                  // [C, C]   fp32 attn0 / T / scratch
    float *buf_decay     = buf_attn + C * C;                  // [C, C]   fp32 decay_mask
    float *warp_scratch  = buf_decay + C * C;                 // 16*256 fp32 (always 16 warps slot)
    float *s_g_cs        = warp_scratch + 16 * 256;
    float *s_exp_cs      = s_g_cs   + C;
    float *s_g_remain    = s_exp_cs + C;                      // exp(g_total - g_cs[i])
    float *s_beta_v      = s_g_remain + C;                    // beta values
    __nv_bfloat16 *bf16_base = (__nv_bfloat16*)(s_beta_v + C);
    __nv_bfloat16 *buf_k     = bf16_base;                     // [C, Dk]
    __nv_bfloat16 *buf_kbeta = buf_k    + C * Dk;             // [C, Dk]
    __nv_bfloat16 *buf_kcd   = buf_kbeta + C * Dk;            // [C, Dk]
    __nv_bfloat16 *buf_kdecay = buf_kcd  + C * Dk;            // [C, Dk] -- transposed for matmul
    __nv_bfloat16 *buf_vbeta = buf_kdecay + C * Dk;           // [C, Dv]
    __nv_bfloat16 *buf_T     = buf_vbeta  + C * Dv;           // [C, C] bf16
    __nv_bfloat16 *buf_tk_cd = buf_T      + C * C;            // [C, Dk] = T @ k_cd
    __nv_bfloat16 *buf_tv    = buf_tk_cd + C * Dk;            // [C, Dv] = T @ v_beta

    // ----- Load k, v, beta, g -----
    // k: [C, Dk] from k_base[(t0+i) * qkd_stride + c]
    for (int idx = tid; idx < C * Dk; idx += nt) {
        int i = idx / Dk, c = idx - i * Dk;
        if (i < chunk_len) {
            buf_k[i * Dk + c] = k[(t0 + i) * qkd_stride + c];
        } else {
            buf_k[i * Dk + c] = __float2bfloat16(0.0f);
        }
    }
    for (int idx = tid; idx < C * Dv; idx += nt) {
        int i = idx / Dv, d = idx - i * Dv;
        if (i < chunk_len) {
            float vv = __bfloat162float(v[(t0 + i) * v_stride + d]);
            float bb = beta_h[(t0 + i) * bd_stride];
            buf_vbeta[i * Dv + d] = __float2bfloat16(vv * bb);
        } else {
            buf_vbeta[i * Dv + d] = __float2bfloat16(0.0f);
        }
    }
    for (int i = tid; i < C; i += nt) {
        s_beta_v[i] = (i < chunk_len) ? beta_h[(t0 + i) * bd_stride] : 0.0f;
    }
    // g cumulative sum (one warp, two passes — C=64 = 2 * warp_size).
    if (warp == 0) {
        // First half [0, 32).
        float g_lo = (lane < chunk_len) ? g_h[(t0 + lane) * bd_stride] : 0.0f;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            float n = __shfl_up_sync(0xffffffff, g_lo, o);
            if (lane >= o) g_lo += n;
        }
        s_g_cs[lane] = g_lo;
        float lo_total = __shfl_sync(0xffffffff, g_lo, 31);
        // Second half [32, 64).
        int j = lane + 32;
        float g_hi = (j < chunk_len) ? g_h[(t0 + j) * bd_stride] : 0.0f;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            float n = __shfl_up_sync(0xffffffff, g_hi, o);
            if (lane >= o) g_hi += n;
        }
        s_g_cs[j] = lo_total + g_hi;
    }
    __syncthreads();
    // exp_g_cs and exp(g_total - g_cs)
    float g_total = s_g_cs[chunk_len > 0 ? chunk_len - 1 : 0];
    for (int i = tid; i < C; i += nt) {
        float gi = (i < chunk_len) ? s_g_cs[i] : 0.0f;
        s_exp_cs[i]    = expf(gi);
        s_g_remain[i]  = expf(g_total - gi);
    }
    __syncthreads();

    // ----- Build k_beta, k_cd, k_decay -----
    for (int idx = tid; idx < C * Dk; idx += nt) {
        int i = idx / Dk, c = idx - i * Dk;
        float kk = __bfloat162float(buf_k[idx]);
        float bb = (i < chunk_len) ? s_beta_v[i] : 0.0f;
        float ec = s_exp_cs[i];
        float er = s_g_remain[i];
        buf_kbeta [idx] = __float2bfloat16(kk * bb);
        buf_kcd   [idx] = __float2bfloat16(kk * ec);
        buf_kdecay[idx] = __float2bfloat16(kk * er);
    }
    // ----- decay_mask[i, j] = exp(g_cs[i] - g_cs[j]) -----
    for (int idx = tid; idx < C * C; idx += nt) {
        int i = idx / C, j = idx - i * C;
        if (i < chunk_len && j < chunk_len && i >= j) {
            buf_decay[idx] = expf(s_g_cs[i] - s_g_cs[j]);
        } else {
            buf_decay[idx] = 0.0f;
        }
    }
    __syncthreads();

    // ----- attn0 = -(k_beta @ k.T) * decay_mask, diag-and-above zeroed -----
    // attn0[i, j] = sum_d k_beta[i, d] * k[j, d], scale by -decay_mask[i, j].
    // Use buf_attn as fp32 accumulator.
    wmma_gemm_fp32<C, C, Dk, false, true>(
        buf_kbeta, Dk, buf_k, Dk, buf_attn, C, warp, n_warps, false);
    __syncthreads();
    // Apply -decay_mask, zero diag-and-above.
    for (int idx = tid; idx < C * C; idx += nt) {
        int i = idx / C, j = idx - i * C;
        if (i > j && i < chunk_len && j < chunk_len) {
            buf_attn[idx] = -buf_attn[idx] * buf_decay[idx];
        } else {
            buf_attn[idx] = 0.0f;
        }
    }
    __syncthreads();

    // ----- T = (I - tril(attn0))^-1, then T += I -----
    // Sequential row solve: for i in [0, C), for j in [0, i):
    //   row_i[j] -= sum_{k=j+1..i-1} row_i[k] * row_k[j]    -- actually it's
    //   T[i, j] = attn0[i, j] + sum_{k=j+1..i-1} T[i, k] * attn0[k, j]
    // (matches dn_chunked.cu's solve.) Let one warp handle the sequential
    // row updates.
    // T = (I - tril(attn0))^-1 via sequential row update with a scratch
    // buffer (no in-row read/write race). Use first C floats of warp_scratch.
    for (int i = 1; i < C; i++) {
        for (int j = tid; j < i; j += nt) {
            float orig = buf_attn[i * C + j];
            float acc = 0.0f;
            for (int l = j + 1; l < i; l++) {
                acc += buf_attn[i * C + l] * buf_attn[l * C + j];
            }
            warp_scratch[j] = orig + acc;
        }
        __syncthreads();
        for (int j = tid; j < i; j += nt) {
            buf_attn[i * C + j] = warp_scratch[j];
        }
        __syncthreads();
    }
    // Now buf_attn holds T (lower-strict-triangular). Set diagonal to 1
    // and upper-triangular to 0 explicitly. Cast to bf16 in buf_T.
    for (int idx = tid; idx < C * C; idx += nt) {
        int i = idx / C, j = idx - i * C;
        float t = (i == j) ? 1.0f : (i > j ? buf_attn[idx] : 0.0f);
        buf_T[idx] = __float2bfloat16(t);
    }
    __syncthreads();

    // ----- buf_tk_cd = T @ k_cd  (bf16 [C, Dk]) -----
    // Use fp32 accumulator in buf_attn (overwrite).
    wmma_gemm_fp32<C, Dk, C, false, false>(
        buf_T, C, buf_kcd, Dk, buf_attn, Dk, warp, n_warps, false);
    __syncthreads();
    for (int idx = tid; idx < C * Dk; idx += nt) {
        buf_tk_cd[idx] = __float2bfloat16(buf_attn[idx]);
    }
    __syncthreads();

    // ----- buf_tv = T @ v_beta (bf16 [C, Dv]) -----
    wmma_gemm_fp32<C, Dv, C, false, false>(
        buf_T, C, buf_vbeta, Dv, buf_attn, Dv, warp, n_warps, false);
    __syncthreads();
    for (int idx = tid; idx < C * Dv; idx += nt) {
        buf_tv[idx] = __float2bfloat16(buf_attn[idx]);
    }
    __syncthreads();

    // ----- M_chunk = exp(g_total) * I - k_decay.T @ buf_tk_cd -----
    // k_decay is buf_kdecay [C, Dk]; transpose -> [Dk, C].
    // Output: M_dst [Dk, Dk].
    wmma_gemm_fp32<Dk, Dk, C, true, false>(
        buf_kdecay, Dk, buf_tk_cd, Dk, M_dst, Dk, warp, n_warps, false);
    __syncthreads();
    // Negate, then add exp(g_total) on diagonal.
    float exp_total = expf(g_total);
    for (int idx = tid; idx < Dk * Dk; idx += nt) {
        int i = idx / Dk, j = idx - i * Dk;
        float v = -M_dst[idx];
        if (i == j) v += exp_total;
        M_dst[idx] = v;
    }

    // ----- B_chunk = k_decay.T @ buf_tv -----
    wmma_gemm_fp32<Dk, Dv, C, true, false>(
        buf_kdecay, Dk, buf_tv, Dv, B_dst, Dv, warp, n_warps, false);
}


// =====================================================================
// Phase D: Hillis-Steele parallel scan stage.
//
// At stage `d` (1, 2, 4, ..., < n_chunks):
//   For each (h, c) with c >= d:
//     M_new[h, c] = M_old[h, c] @ M_old[h, c-d]
//     B_new[h, c] = M_old[h, c] @ B_old[h, c-d] + B_old[h, c]
//
// Implemented as two kernel launches per stage (or use double buffering).
// We use double buffering: ping-pong between (M, B) and (M_alt, B_alt).
// =====================================================================

__global__ void __launch_bounds__(512, 1)
phase_d_compose_stage_kernel(
    const float *__restrict__ M_in,
    const float *__restrict__ B_in,
    float *__restrict__ M_out,
    float *__restrict__ B_out,
    int H, int n_chunks, int d)
{
    int H_idx = blockIdx.y;
    int c_idx = blockIdx.x;
    if (H_idx >= H || c_idx >= n_chunks) return;

    int tid = threadIdx.x;
    int nt  = blockDim.x;
    int warp = tid >> 5;
    int n_warps = nt >> 5;

    const float *M_self = M_in + ((size_t)H_idx * n_chunks + c_idx) * Dk * Dk;
    const float *B_self = B_in + ((size_t)H_idx * n_chunks + c_idx) * Dk * Dv;
    float *M_dst = M_out + ((size_t)H_idx * n_chunks + c_idx) * Dk * Dk;
    float *B_dst = B_out + ((size_t)H_idx * n_chunks + c_idx) * Dk * Dv;

    if (c_idx < d) {
        // Pass-through: copy.
        for (int idx = tid; idx < Dk * Dk; idx += nt) M_dst[idx] = M_self[idx];
        for (int idx = tid; idx < Dk * Dv; idx += nt) B_dst[idx] = B_self[idx];
        return;
    }

    const float *M_pred = M_in + ((size_t)H_idx * n_chunks + (c_idx - d)) * Dk * Dk;
    const float *B_pred = B_in + ((size_t)H_idx * n_chunks + (c_idx - d)) * Dk * Dv;

    // M_dst = M_self @ M_pred.
    // B_dst = M_self @ B_pred + B_self.
    // Stage M_self / M_pred / B_pred to bf16 staging in shared (saves bandwidth
    // and lets us use WMMA bf16 inputs with fp32 accumulator).
    extern __shared__ unsigned char smem_raw[];
    __nv_bfloat16 *Ms_bf = (__nv_bfloat16*)smem_raw;             // [Dk, Dk]
    __nv_bfloat16 *Mp_bf = Ms_bf + Dk * Dk;                       // [Dk, Dk]
    __nv_bfloat16 *Bp_bf = Mp_bf + Dk * Dk;                       // [Dk, Dv]
    for (int idx = tid; idx < Dk * Dk; idx += nt) Ms_bf[idx] = __float2bfloat16(M_self[idx]);
    for (int idx = tid; idx < Dk * Dk; idx += nt) Mp_bf[idx] = __float2bfloat16(M_pred[idx]);
    for (int idx = tid; idx < Dk * Dv; idx += nt) Bp_bf[idx] = __float2bfloat16(B_pred[idx]);
    __syncthreads();

    // M_dst = Ms_bf @ Mp_bf  (Dk x Dk x Dk)
    wmma_gemm_fp32<Dk, Dk, Dk, false, false>(
        Ms_bf, Dk, Mp_bf, Dk, M_dst, Dk, warp, n_warps, false);
    // B_dst = Ms_bf @ Bp_bf + B_self
    // Initialize B_dst with B_self, then accumulate.
    for (int idx = tid; idx < Dk * Dv; idx += nt) B_dst[idx] = B_self[idx];
    __syncthreads();
    wmma_gemm_fp32<Dk, Dv, Dk, false, false>(
        Ms_bf, Dk, Bp_bf, Dv, B_dst, Dv, warp, n_warps, true);
}


// =====================================================================
// Phase C: per-chunk output.
//
// Re-runs the chunk-internal computation (T, k_cd, decay_mask, etc.) and
// produces y_chunk = q @ k.T * decay_mask @ v_new + (q * exp_g_cs) @ s
// where s = state_at_chunk_start (= M_prefix_{c-1} @ state_0 + B_prefix_{c-1};
// for c=0 it's state_0).
//
// Last chunk also writes the final state to state_out (from the inclusive
// prefix at the last chunk).
// =====================================================================

__global__ void __launch_bounds__(512, 1)
phase_c_output_kernel(
    const __nv_bfloat16 *__restrict__ q_base,
    const __nv_bfloat16 *__restrict__ k_base,
    const __nv_bfloat16 *__restrict__ v_base,
    const float *__restrict__ beta_base,
    const float *__restrict__ g_base,
    const float *__restrict__ M_prefix,    // inclusive prefix [H, n_chunks, Dk, Dk]
    const float *__restrict__ B_prefix,    // inclusive prefix [H, n_chunks, Dk, Dv]
    const float *__restrict__ state_in,    // [H, Dk, Dv]
    __nv_bfloat16 *__restrict__ y,          // [S, H*Dv]
    float *__restrict__ state_out,          // [H, Dk, Dv] — only last chunk writes
    int S, int H, int n_chunks,
    int qkd_stride, int v_stride, int bd_stride, int y_stride)
{
    int H_idx = blockIdx.y;
    int c_idx = blockIdx.x;
    if (H_idx >= H || c_idx >= n_chunks) return;
    int t0 = c_idx * C;
    int chunk_len = (S - t0 < C) ? (S - t0) : C;

    int tid = threadIdx.x;
    int nt  = blockDim.x;
    int warp = tid >> 5;
    int n_warps = nt >> 5;
    int lane = tid & 31;

    const __nv_bfloat16 *q = q_base + H_idx * Dk;
    const __nv_bfloat16 *k = k_base + H_idx * Dk;
    const __nv_bfloat16 *v = v_base + H_idx * Dv;
    const float *beta_h    = beta_base + H_idx;
    const float *g_h       = g_base    + H_idx;

    // Compute s = state_at_chunk_start for this chunk.
    // s = (c==0) ? state_in[h] : M_prefix[h, c-1] @ state_in[h] + B_prefix[h, c-1]
    // We'll materialize s in shared as [Dk, Dv] fp32.
    extern __shared__ unsigned char smem_raw[];
    float *s_state       = (float*)smem_raw;                 // [Dk, Dv] = 64 KB
    float *buf_attn      = s_state + Dk * Dv;                // [C, C] = 16 KB
    float *buf_decay     = buf_attn + C * C;                 // [C, C] = 16 KB
    float *warp_scratch  = buf_decay + C * C;                // 16 KB
    float *s_g_cs        = warp_scratch + 16 * 256;
    float *s_exp_cs      = s_g_cs   + C;
    float *s_beta_v      = s_exp_cs + C;
    __nv_bfloat16 *bf16_base = (__nv_bfloat16*)(s_beta_v + C);
    __nv_bfloat16 *state_bf  = bf16_base;                    // [Dk, Dv] = 32 KB
    __nv_bfloat16 *buf_q     = state_bf + Dk * Dv;           // [C, Dk] = 16 KB
    __nv_bfloat16 *buf_k     = buf_q   + C * Dk;             // [C, Dk]
    __nv_bfloat16 *buf_kcd   = buf_k   + C * Dk;             // [C, Dk]
    __nv_bfloat16 *buf_kbeta = buf_kcd + C * Dk;             // [C, Dk]
    __nv_bfloat16 *buf_vbeta = buf_kbeta + C * Dk;           // [C, Dv]
    __nv_bfloat16 *buf_T     = buf_vbeta + C * Dv;           // [C, C]
    __nv_bfloat16 *buf_attn_bf = buf_T + C * C;              // [C, C]

    // Compute s.
    const float *state_in_h = state_in + H_idx * Dk * Dv;
    if (c_idx == 0) {
        for (int idx = tid; idx < Dk * Dv; idx += nt) s_state[idx] = state_in_h[idx];
    } else {
        const float *M_p = M_prefix + ((size_t)H_idx * n_chunks + (c_idx - 1)) * Dk * Dk;
        const float *B_p = B_prefix + ((size_t)H_idx * n_chunks + (c_idx - 1)) * Dk * Dv;
        // Stage M_p, state_in_h to bf16
        for (int idx = tid; idx < Dk * Dk; idx += nt) state_bf[idx] = __float2bfloat16(M_p[idx]);
        for (int idx = tid; idx < Dk * Dv; idx += nt) s_state[idx] = B_p[idx];
        __syncthreads();
        // Need B_pred + M_pred @ state_in. state_in is already fp32 [Dk, Dv].
        // Stage state_in to bf16 in buf_q (reused as scratch) ... actually let's
        // reuse buf_kbeta as a [Dk, Dv] bf16 scratch (it's [C, Dk] = 16 KB but
        // here we need [Dk, Dv] = 32 KB).
        // Use buf_T + buf_attn_bf + buf_kbeta + buf_kcd contiguous = 8+8+16+16 = 48 KB
        // Or simpler: cast in place using state_bf temporarily
        // Hmm we already used state_bf for M_p. Let's stage state_in into a fresh region.
        // Just reuse buf_q (16 KB) — but state_in is [Dk, Dv] = 32 KB. Need a 32 KB region.
        // Use buf_kbeta + buf_kcd contiguous (16+16=32 KB).
        __nv_bfloat16 *state_in_bf = buf_kbeta;
        for (int idx = tid; idx < Dk * Dv; idx += nt) state_in_bf[idx] = __float2bfloat16(state_in_h[idx]);
        __syncthreads();
        // s_state = B_p (already in s_state) + M_p @ state_in_bf
        wmma_gemm_fp32<Dk, Dv, Dk, false, false>(
            state_bf, Dk, state_in_bf, Dv, s_state, Dv, warp, n_warps, true);
        __syncthreads();
    }
    __syncthreads();

    // Re-compute chunk intermediates.
    for (int idx = tid; idx < C * Dk; idx += nt) {
        int i = idx / Dk, c = idx - i * Dk;
        if (i < chunk_len) {
            buf_q[idx] = q[(t0 + i) * qkd_stride + c];
            buf_k[idx] = k[(t0 + i) * qkd_stride + c];
        } else {
            buf_q[idx] = __float2bfloat16(0);
            buf_k[idx] = __float2bfloat16(0);
        }
    }
    for (int idx = tid; idx < C * Dv; idx += nt) {
        int i = idx / Dv, d = idx - i * Dv;
        if (i < chunk_len) {
            float vv = __bfloat162float(v[(t0 + i) * v_stride + d]);
            float bb = beta_h[(t0 + i) * bd_stride];
            buf_vbeta[idx] = __float2bfloat16(vv * bb);
        } else {
            buf_vbeta[idx] = __float2bfloat16(0);
        }
    }
    for (int i = tid; i < C; i += nt) {
        s_beta_v[i] = (i < chunk_len) ? beta_h[(t0 + i) * bd_stride] : 0.0f;
    }
    // g_cs cumulative sum (one warp, two passes).
    if (warp == 0) {
        float g_lo = (lane < chunk_len) ? g_h[(t0 + lane) * bd_stride] : 0.0f;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            float n = __shfl_up_sync(0xffffffff, g_lo, o);
            if (lane >= o) g_lo += n;
        }
        s_g_cs[lane] = g_lo;
        float lo_total = __shfl_sync(0xffffffff, g_lo, 31);
        int j = lane + 32;
        float g_hi = (j < chunk_len) ? g_h[(t0 + j) * bd_stride] : 0.0f;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            float n = __shfl_up_sync(0xffffffff, g_hi, o);
            if (lane >= o) g_hi += n;
        }
        s_g_cs[j] = lo_total + g_hi;
    }
    __syncthreads();
    for (int i = tid; i < C; i += nt) {
        float gi = (i < chunk_len) ? s_g_cs[i] : 0.0f;
        s_exp_cs[i] = expf(gi);
    }
    __syncthreads();

    // k_cd = k * exp_g_cs; k_beta = k * beta. (we don't need k_decay in Phase C.)
    for (int idx = tid; idx < C * Dk; idx += nt) {
        int i = idx / Dk;
        float kk = __bfloat162float(buf_k[idx]);
        float ec = s_exp_cs[i];
        float bb = (i < chunk_len) ? s_beta_v[i] : 0.0f;
        buf_kcd  [idx] = __float2bfloat16(kk * ec);
        buf_kbeta[idx] = __float2bfloat16(kk * bb);
    }
    // decay_mask
    for (int idx = tid; idx < C * C; idx += nt) {
        int i = idx / C, j = idx - i * C;
        if (i < chunk_len && j < chunk_len && i >= j) {
            buf_decay[idx] = expf(s_g_cs[i] - s_g_cs[j]);
        } else {
            buf_decay[idx] = 0.0f;
        }
    }
    __syncthreads();

    // Recompute T (same as Phase A).
    wmma_gemm_fp32<C, C, Dk, false, true>(
        buf_kbeta, Dk, buf_k, Dk, buf_attn, C, warp, n_warps, false);
    __syncthreads();
    for (int idx = tid; idx < C * C; idx += nt) {
        int i = idx / C, j = idx - i * C;
        if (i > j && i < chunk_len && j < chunk_len) {
            buf_attn[idx] = -buf_attn[idx] * buf_decay[idx];
        } else {
            buf_attn[idx] = 0.0f;
        }
    }
    __syncthreads();
    // T = (I - tril(attn0))^-1 via sequential row update with a scratch
    // buffer (no in-row read/write race). Use first C floats of warp_scratch.
    for (int i = 1; i < C; i++) {
        for (int j = tid; j < i; j += nt) {
            float orig = buf_attn[i * C + j];
            float acc = 0.0f;
            for (int l = j + 1; l < i; l++) {
                acc += buf_attn[i * C + l] * buf_attn[l * C + j];
            }
            warp_scratch[j] = orig + acc;
        }
        __syncthreads();
        for (int j = tid; j < i; j += nt) {
            buf_attn[i * C + j] = warp_scratch[j];
        }
        __syncthreads();
    }
    for (int idx = tid; idx < C * C; idx += nt) {
        int i = idx / C, j = idx - i * C;
        float t = (i == j) ? 1.0f : (i > j ? buf_attn[idx] : 0.0f);
        buf_T[idx] = __float2bfloat16(t);
    }
    __syncthreads();

    // ----- Compute attn_in = q @ k.T * decay_mask FIRST (while decay_mask
    // is still valid in buf_decay), cast to bf16 in buf_attn_bf. -----
    wmma_gemm_fp32<C, C, Dk, false, true>(
        buf_q, Dk, buf_k, Dk, buf_attn, C, warp, n_warps, false);
    __syncthreads();
    for (int idx = tid; idx < C * C; idx += nt) {
        int i = idx / C, j = idx - i * C;
        if (i < chunk_len && j < chunk_len && i >= j) {
            buf_attn_bf[idx] = __float2bfloat16(buf_attn[idx] * buf_decay[idx]);
        } else {
            buf_attn_bf[idx] = __float2bfloat16(0.0f);
        }
    }
    __syncthreads();

    // ----- v_new = T @ v_beta - T @ k_cd @ s -----
    // Compute T@v_beta into buf_attn (fp32 [C, Dv]).
    wmma_gemm_fp32<C, Dv, C, false, false>(
        buf_T, C, buf_vbeta, Dv, buf_attn, Dv, warp, n_warps, false);
    __syncthreads();
    // Stage s_state to bf16.
    for (int idx = tid; idx < Dk * Dv; idx += nt) state_bf[idx] = __float2bfloat16(s_state[idx]);
    __syncthreads();
    // T @ k_cd -> bf16 buf_tkcd. We use buf_decay (now free; we stashed
    // attn_in into buf_attn_bf above) as fp32 scratch for the matmul.
    __nv_bfloat16 *buf_tkcd = buf_kbeta;   // [C, Dk] = 16 KB
    wmma_gemm_fp32<C, Dk, C, false, false>(
        buf_T, C, buf_kcd, Dk, buf_decay, Dk, warp, n_warps, false);
    __syncthreads();
    for (int idx = tid; idx < C * Dk; idx += nt) buf_tkcd[idx] = __float2bfloat16(buf_decay[idx]);
    __syncthreads();
    // (T @ k_cd) @ s_state into buf_decay (fp32 [C, Dv]).
    wmma_gemm_fp32<C, Dv, Dk, false, false>(
        buf_tkcd, Dk, state_bf, Dv, buf_decay, Dv, warp, n_warps, false);
    __syncthreads();
    // v_new = buf_attn - buf_decay; cast to bf16 in buf_vbeta.
    for (int idx = tid; idx < C * Dv; idx += nt) {
        buf_vbeta[idx] = __float2bfloat16(buf_attn[idx] - buf_decay[idx]);
    }
    __syncthreads();

    // attn_int = (q * exp_g_cs) @ s.   Scale q in place.
    for (int idx = tid; idx < C * Dk; idx += nt) {
        int i = idx / Dk;
        float ec = (i < chunk_len) ? s_exp_cs[i] : 0.0f;
        float qq = __bfloat162float(buf_q[idx]);
        buf_q[idx] = __float2bfloat16(qq * ec);
    }
    __syncthreads();
    // y_chunk = attn_in @ v_new + (q * exp_cs) @ state.
    // First: buf_attn = (q * exp_cs) @ state_bf (still in bf16).
    wmma_gemm_fp32<C, Dv, Dk, false, false>(
        buf_q, Dk, state_bf, Dv, buf_attn, Dv, warp, n_warps, false);
    __syncthreads();
    // Add attn_in @ v_new (accumulate into buf_attn).
    wmma_gemm_fp32<C, Dv, C, false, false>(
        buf_attn_bf, C, buf_vbeta, Dv, buf_attn, Dv, warp, n_warps, true);
    __syncthreads();

    // Write y_chunk to output.
    __nv_bfloat16 *y_h = y + H_idx * Dv;
    for (int idx = tid; idx < C * Dv; idx += nt) {
        int j = idx / Dv, d = idx - j * Dv;
        if (j < chunk_len) {
            y_h[(t0 + j) * y_stride + d] = __float2bfloat16(buf_attn[idx]);
        }
    }

    // Last chunk: write final state to state_out.
    if (c_idx == n_chunks - 1 && state_out) {
        // state_after = M_prefix[h, last] @ state_in[h] + B_prefix[h, last]
        const float *M_last = M_prefix + ((size_t)H_idx * n_chunks + c_idx) * Dk * Dk;
        const float *B_last = B_prefix + ((size_t)H_idx * n_chunks + c_idx) * Dk * Dv;
        float *state_out_h = state_out + H_idx * Dk * Dv;
        // Stage state_in to bf16 (reuse state_bf already populated... actually
        // state_bf currently has the chunk-start state, not state_in. Re-stage.)
        for (int idx = tid; idx < Dk * Dv; idx += nt) state_bf[idx] = __float2bfloat16(state_in_h[idx]);
        // M_last to bf16 staging — reuse buf_kbeta region (32 KB).
        __nv_bfloat16 *M_bf = (__nv_bfloat16*)buf_q;       // [Dk, Dk] = 32 KB
        // buf_q is [C*Dk] = 16 KB. Need 32 KB. Use buf_q + buf_k contiguous (32 KB).
        // The two buffers ARE contiguous in our layout (buf_k = buf_q + C*Dk).
        for (int idx = tid; idx < Dk * Dk; idx += nt) M_bf[idx] = __float2bfloat16(M_last[idx]);
        // Initialize state_out with B_last.
        for (int idx = tid; idx < Dk * Dv; idx += nt) state_out_h[idx] = B_last[idx];
        __syncthreads();
        // Accumulate M_last @ state_in onto state_out.
        wmma_gemm_fp32<Dk, Dv, Dk, false, false>(
            M_bf, Dk, state_bf, Dv, state_out_h, Dv, warp, n_warps, true);
    }
}


// =====================================================================
// Launcher.
// =====================================================================

static cudaError_t launch_inner(
    const __nv_bfloat16 *q, const __nv_bfloat16 *k, const __nv_bfloat16 *v,
    const float *beta, const float *g, const float *state_in,
    __nv_bfloat16 *y, float *state_out,
    int S, int H,
    int qkd_stride, int v_stride, int bd_stride, int y_stride,
    cudaStream_t stream)
{
    int n_chunks = (S + C - 1) / C;

    // --- Lazy alloc operator scratch (M, B) in two ping-pong copies. ---
    static float *g_M_a = nullptr;
    static float *g_B_a = nullptr;
    static float *g_M_b = nullptr;
    static float *g_B_b = nullptr;
    static int g_alloc_chunks = 0;
    static int g_alloc_H = 0;
    if (n_chunks > g_alloc_chunks || H > g_alloc_H) {
        if (g_M_a) cudaFree(g_M_a);
        if (g_B_a) cudaFree(g_B_a);
        if (g_M_b) cudaFree(g_M_b);
        if (g_B_b) cudaFree(g_B_b);
        size_t M_bytes = (size_t)H * n_chunks * Dk * Dk * sizeof(float);
        size_t B_bytes = (size_t)H * n_chunks * Dk * Dv * sizeof(float);
        cudaMalloc(&g_M_a, M_bytes);
        cudaMalloc(&g_B_a, B_bytes);
        cudaMalloc(&g_M_b, M_bytes);
        cudaMalloc(&g_B_b, B_bytes);
        g_alloc_chunks = n_chunks;
        g_alloc_H = H;
    }

    // --- Phase A: extract per-chunk operators. ---
    {
        int threads = 512;
        // Shared-mem layout (must match phase_a_extract_kernel).
        size_t smem_fp32 = ((size_t)2 * C * C + 16 * 256 + 4 * C) * sizeof(float);
        size_t smem_bf16 = ((size_t)5 * C * Dk + C * Dv + C * C + C * Dv) * sizeof(__nv_bfloat16);
        size_t smem = smem_fp32 + smem_bf16;
        cudaFuncSetAttribute(phase_a_extract_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        dim3 grid((unsigned)n_chunks, (unsigned)H);
        phase_a_extract_kernel<<<grid, threads, smem, stream>>>(
            q, k, v, beta, g, g_M_a, g_B_a, S, H, n_chunks,
            qkd_stride, v_stride, bd_stride);
    }

    // --- Phase D: log2(n_chunks) Hillis-Steele stages with ping-pong. ---
    {
        int threads = 512;
        size_t smem = (size_t)(2 * Dk * Dk + Dk * Dv) * sizeof(__nv_bfloat16);
        cudaFuncSetAttribute(phase_d_compose_stage_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        dim3 grid((unsigned)n_chunks, (unsigned)H);
        float *M_in = g_M_a, *B_in = g_B_a;
        float *M_out = g_M_b, *B_out = g_B_b;
        for (int d = 1; d < n_chunks; d <<= 1) {
            phase_d_compose_stage_kernel<<<grid, threads, smem, stream>>>(
                M_in, B_in, M_out, B_out, H, n_chunks, d);
            // ping-pong
            float *tmp;
            tmp = M_in; M_in = M_out; M_out = tmp;
            tmp = B_in; B_in = B_out; B_out = tmp;
        }
        // After the loop, M_in / B_in hold the final inclusive prefix.
        if (M_in != g_M_a) {
            // Move data so g_M_a / g_B_a hold the result (Phase C reads from g_M_a/g_B_a).
            cudaMemcpyAsync(g_M_a, M_in, (size_t)H * n_chunks * Dk * Dk * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(g_B_a, B_in, (size_t)H * n_chunks * Dk * Dv * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
        }
    }

    // --- Phase C: per-chunk output. ---
    {
        int threads = 512;
        size_t smem_fp32 = ((size_t)Dk * Dv + 2 * C * C + 16 * 256 + 3 * C) * sizeof(float);
        size_t smem_bf16 = ((size_t)Dk * Dv + 5 * C * Dk + C * Dv + 2 * C * C) * sizeof(__nv_bfloat16);
        size_t smem = smem_fp32 + smem_bf16;
        cudaFuncSetAttribute(phase_c_output_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        dim3 grid((unsigned)n_chunks, (unsigned)H);
        phase_c_output_kernel<<<grid, threads, smem, stream>>>(
            q, k, v, beta, g, g_M_a, g_B_a, state_in, y, state_out,
            S, H, n_chunks, qkd_stride, v_stride, bd_stride, y_stride);
    }

    return cudaGetLastError();
}

}  // namespace ps

extern "C" cudaError_t launch_dn_parallel_scan_fwd(
    const __nv_bfloat16 *q, const __nv_bfloat16 *k, const __nv_bfloat16 *v,
    const float *beta, const float *g, const float *state_in,
    __nv_bfloat16 *y, float *state_out,
    int S, int H,
    int qkd_stride, int v_stride, int bd_stride, int y_stride,
    cudaStream_t stream)
{
    return ps::launch_inner(
        q, k, v, beta, g, state_in, y, state_out,
        S, H, qkd_stride, v_stride, bd_stride, y_stride, stream);
}
