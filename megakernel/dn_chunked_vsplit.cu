/**
 * DeltaNet chunked forward — bf16 tensor cores via nvcuda::wmma.
 *
 * Algorithm: matches `dn_chunked_proto.chunked_fwd` (which matches HF's
 * torch_chunk_gated_delta_rule). Per chunk c of size C=64:
 *
 *   k_beta   = k * beta                                     (elementwise)
 *   v_beta   = v * beta                                     (elementwise)
 *   attn0    = -(k_beta @ k.T) * decay_mask, diag-and-above zeroed
 *   T        = (I - tril(attn0))^(-1) via the standard sequential row update
 *              (followed by T += I)
 *   v_new    = T @ v_beta
 *   k_cd     = T @ (k_beta * exp_g_cs)
 *   v_prime  = k_cd @ state
 *   v_new   -= v_prime
 *   attn_in  = q @ k.T * decay_mask  (lower-and-diag kept)
 *   attn_int = (q * exp_g_cs) @ state
 *   y_chunk  = attn_int + attn_in @ v_new
 *   state    = state * exp_g_chunk_total
 *            + (k * exp(g_chunk_total - g_cs))^T @ v_new
 *
 * Block: 256 threads = 8 warps per head. Grid: (H,).
 *
 * Shared memory layout (217 KB, fits B200's 228 KB per-block limit):
 *   state[Dk*Dv]      fp32                                    64 KB  persistent
 *   state_bf[Dk*Dv/2] bf16  (one half at a time)              16 KB  per-chunk
 *   buf_q[C*Dk]       bf16  q -> q_scaled                     16 KB
 *   buf_k[C*Dk]       bf16  k (used as k.T via col_major)     16 KB
 *   buf_kbeta[C*Dk]   bf16  k_beta -> v_prime -> k_decay      16 KB
 *   buf_vbeta[C*Dv]   bf16  v_beta -> v_new                   16 KB
 *   buf_kcd[C*Dk]     bf16  k_cd -> attn_int                  16 KB
 *   buf_attn[C*C]     fp32  attn0 -> T -> attn_in             16 KB
 *   buf_attn_bf[C*C]  bf16  T_bf -> attn_in_bf                 8 KB
 *   buf_decay[C*C]    fp32  decay_mask                        16 KB
 *   scalars (g_cs, exp_cs, beta_buf)                          ~1 KB
 *
 * The clever bit: instead of explicitly building k.T as a separate buffer,
 * we load k row-major as a matrix_b with col_major layout in WMMA. Same
 * memory, different fragment interpretation. Likewise k_decay.T as
 * matrix_a col_major.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

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


// ===================== WMMA gemm primitives =====================
// All ops use 16x16x16 tiles, bf16 inputs, fp32 accumulator.

// C = A @ B  (C fp32 row-major in shared, accumulate=overwrite or add)
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
        int nt = t % N_TILES;
        fragment<accumulator, TM, TN, TK, float> c_frag;
        if (accumulate) {
            load_matrix_sync(c_frag,
                             C + (mt * TM) * ld_c + (nt * TN),
                             ld_c, mem_row_major);
        } else {
            fill_fragment(c_frag, 0.0f);
        }
        #pragma unroll
        for (int kt = 0; kt < K_TILES; kt++) {
            using A_layout = typename std::conditional<A_COL, col_major, row_major>::type;
            using B_layout = typename std::conditional<B_COL, col_major, row_major>::type;
            fragment<matrix_a, TM, TN, TK, __nv_bfloat16, A_layout> a_frag;
            fragment<matrix_b, TM, TN, TK, __nv_bfloat16, B_layout> b_frag;
            // For row_major A: tile at (mt, kt). offset (mt*TM)*ld_a + kt*TK.
            // For col_major A: A is conceptually [M, K]; in mem (col-major) the
            // tile (mt, kt) is at column kt*TK row mt*TM = (kt*TK)*ld_a + mt*TM.
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
        store_matrix_sync(C + (mt * TM) * ld_c + (nt * TN),
                          c_frag, ld_c, mem_row_major);
    }
}

// C = A @ B with C bf16 output (overwrites). Uses fp32 accumulator internally.
// `warp_scratch` is a SHARED-mem scratch buffer with at least 256 floats per
// active warp (each warp writes its tile to its own slot).
template <int M, int N, int K, bool A_COL = false, bool B_COL = false>
__device__ inline void wmma_gemm_bf16(
    const __nv_bfloat16 *A, int ld_a,
    const __nv_bfloat16 *B, int ld_b,
    __nv_bfloat16 *C, int ld_c,
    int warp_id, int n_warps,
    float *warp_scratch /* [n_warps * 256] */)
{
    constexpr int TM = 16, TN = 16, TK = 16;
    constexpr int M_TILES = M / TM;
    constexpr int N_TILES = N / TN;
    constexpr int K_TILES = K / TK;

    int n_out_tiles = M_TILES * N_TILES;
    float *my_scratch = warp_scratch + warp_id * (TM * TN);
    int lane = threadIdx.x & 31;
    for (int t = warp_id; t < n_out_tiles; t += n_warps) {
        int mt = t / N_TILES;
        int nt = t % N_TILES;
        fragment<accumulator, TM, TN, TK, float> c_frag;
        fill_fragment(c_frag, 0.0f);
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
        // Cast fp32 fragment to bf16 via shared-mem scratch.
        store_matrix_sync(my_scratch, c_frag, TN, mem_row_major);
        for (int e = lane; e < TM * TN; e += 32) {
            int i = e / TN, j = e - i * TN;
            C[(mt * TM + i) * ld_c + (nt * TN + j)] = __float2bfloat16(my_scratch[e]);
        }
    }
}


// ===================== chunked fwd kernel =====================
// Optionally writes `state_chunks_base` [H, n_chunks+1, Dk, Dv] fp32 — entry
// 0 is state_init, entries 1..n_chunks are end-of-chunk states. Used by the
// chunked backward kernel to replay forward intermediates per chunk.
__global__ void dn_chunked_vsplit_kernel(
    const __nv_bfloat16 *__restrict__ q_base,
    const __nv_bfloat16 *__restrict__ k_base,
    const __nv_bfloat16 *__restrict__ v_base,
    const float *__restrict__ beta_base,
    const float *__restrict__ g_base,
    const float *__restrict__ state_in_base,
    __nv_bfloat16 *__restrict__ y_base,
    float *__restrict__ state_out_base,
    float *__restrict__ state_chunks_base,   // [H, n_chunks+1, Dk, Dv] or null
    int S,
    int qkd_pos_stride,
    int v_pos_stride,
    int bd_pos_stride,
    int y_pos_stride)
{
    constexpr int Dk = DN_KEY_DEFAULT;
    // V-split: Dv is the per-block V slice size. With V_SPLITS=8 each
    // block owns a [Dk, 16] state slice and the kernel processes state as
    // a single Dv-wide pass instead of two Dv/2 halves (since Dv/2=8 is
    // smaller than the WMMA tile width of 16). Grid: H * V_SPLITS = 128
    // blocks, which fills 148-SM B200 to ~86% occupancy.
    constexpr int Dv = 16;
    constexpr int V_SPLITS = 8;
    constexpr int Dv_full = Dv * V_SPLITS;   // 128
    constexpr int N_VHALVES = 1;             // Dv == one WMMA tile, no splitting
    constexpr int VHALF = Dv;                // == Dv when N_VHALVES == 1
    constexpr int C  = DN_CHUNK;

    int tid = threadIdx.x;
    int nt  = blockDim.x;
    int warp = tid >> 5;
    int n_warps = nt >> 5;
    int h  = blockIdx.x / V_SPLITS;
    int vs = blockIdx.x % V_SPLITS;
    int v_offset = vs * Dv;                  // offset into the head's full V dim

    // Per-(head, v_split) views. q/k are shared across v_splits of the same
    // head and hit the same qkv_proj rows; v / y / state are sliced.
    const __nv_bfloat16 *q  = q_base  + h * Dk;
    const __nv_bfloat16 *k  = k_base  + h * Dk;
    const __nv_bfloat16 *v  = v_base  + h * Dv_full + v_offset;
    const float *beta_h     = beta_base + h;
    const float *g_h        = g_base    + h;
    // state laid out [H, Dk, Dv_full] in global; per block we own a
    // [Dk, Dv] slice. The kernel reads/writes through state_in_h with
    // stride Dv_full between rows of i (handled in the load/store loops
    // below), but the slice itself is the contiguous [Dk*Dv] block in
    // shared memory.
    const float *state_in_h = state_in_base + h * Dk * Dv_full + v_offset;
    __nv_bfloat16 *y_h      = y_base + h * Dv_full + v_offset;
    float *state_out_h      = state_out_base ? state_out_base + h * Dk * Dv_full + v_offset : nullptr;

    int n_chunks = (S + C - 1) / C;
    float *state_chunks_h   = state_chunks_base
        ? state_chunks_base + h * (size_t)(n_chunks + 1) * Dk * Dv : nullptr;

    // ----- shared memory layout -----
    // Lay out fp32 buffers first (16-byte alignment guaranteed), then bf16.
    extern __shared__ unsigned char smem_raw[];
    float *state         = (float*)smem_raw;                    // [Dk, Dv]      fp32
    float *buf_attn      = state    + Dk * Dv;                  // [C, C]        fp32
    float *buf_decay     = buf_attn + C * C;                    // [C, C]        fp32
    float *warp_scratch  = buf_decay + C * C;                   // [n_warps * 256] fp32 (16 KB at 16 warps)
    float *s_g_cs        = warp_scratch + 16 * 256;             // [C]
    float *s_exp_cs      = s_g_cs   + C;                        // [C]
    float *s_beta        = s_exp_cs + C;                        // [C]
    __nv_bfloat16 *bf16_base = (__nv_bfloat16*)(s_beta + C);
    __nv_bfloat16 *state_bf  = bf16_base;                  // [Dk, VHALF]    bf16  (whole slice when V_SPLITS=8)
    __nv_bfloat16 *buf_q     = state_bf + Dk * VHALF;      // [C, Dk]
    __nv_bfloat16 *buf_k     = buf_q    + C * Dk;          // [C, Dk]
    __nv_bfloat16 *buf_kbeta = buf_k    + C * Dk;          // [C, Dk]
    __nv_bfloat16 *buf_vbeta = buf_kbeta + C * Dk;         // [C, Dv]
    __nv_bfloat16 *buf_kcd   = buf_vbeta + C * Dv;         // [C, Dk]  (also [C, Dv] for attn_int)
    __nv_bfloat16 *buf_attn_bf = buf_kcd + C * Dk;         // [C, C]   bf16

    // ----- Load initial state (strided: global has Dv_full stride per row) -----
    for (int i = tid; i < Dk * Dv; i += nt) {
        int row = i / Dv;
        int col = i - row * Dv;
        state[i] = state_in_h[row * Dv_full + col];
    }
    __syncthreads();

    // Save state_chunks[0] = state_init (entry 0 of the per-head slab).
    if (state_chunks_h) {
        for (int i = tid; i < Dk * Dv; i += nt) state_chunks_h[i] = state[i];
    }
    __syncthreads();

    for (int c = 0; c < n_chunks; c++) {
        int t0 = c * C;
        int chunk_len = (S - t0 < C) ? (S - t0) : C;

        // ----- Load q, k, v for chunk; pad with zeros if chunk_len < C -----
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
        for (int idx = tid; idx < C * Dv; idx += nt) {
            int j = idx / Dv;
            int d = idx - j * Dv;
            buf_vbeta[idx] = (j < chunk_len) ? v[(t0 + j) * v_pos_stride + d]
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

        // Cumsum g over the chunk; thread 0 does it sequentially (C=64).
        if (tid == 0) {
            float acc = 0.0f;
            for (int i = 0; i < C; i++) {
                acc += s_g_cs[i];
                s_g_cs[i] = acc;
            }
        }
        __syncthreads();
        // exp(g_cs)
        if (tid < C) s_exp_cs[tid] = expf(s_g_cs[tid]);

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
        for (int idx = tid; idx < C * Dv; idx += nt) {
            int j = idx / Dv;
            float bj = (j < chunk_len) ? s_beta[j] : 0.0f;
            float vv = __bfloat162float(buf_vbeta[idx]);
            buf_vbeta[idx] = __float2bfloat16(bj * vv);
        }
        __syncthreads();

        // ----- attn0 = -(k_beta @ k.T) -----
        // A = buf_kbeta [C, Dk] row_major. B = k.T = buf_k loaded col_major [Dk, C].
        wmma_gemm_fp32<C, C, Dk, false, true>(
            buf_kbeta, Dk, buf_k, Dk, buf_attn, C, warp, n_warps, false);
        __syncthreads();
        // attn0 *= -decay_mask, but zero diag-and-above.
        for (int idx = tid; idx < C * C; idx += nt) {
            int i = idx / C, j = idx - i * C;
            if (i > j) buf_attn[idx] = -buf_attn[idx] * buf_decay[idx];
            else       buf_attn[idx] = 0.0f;
        }
        __syncthreads();

        // ----- T construction: in-place sequential update on buf_attn -----
        // for i in 1..C-1:
        //   for j in 0..i-1:
        //     attn[i, j] = attn[i, j] + sum_{l=j+1..i-1} attn[i, l] * attn[l, j]
        // Each inner row update can race-condition with itself (read+write of
        // attn[i, l]) — solve by computing all updated values into a scratch
        // first, then copying back. We use s_g_cs as scratch (tiny).
        // (Actually s_g_cs only holds C floats; we need C scratch elements
        // for one row's i-1 values. Fits.)
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

        // ----- v_new = T @ v_beta  (in place: write to buf_vbeta) -----
        // The matmul is M=C, N=Dv, K=C. Reads from buf_vbeta into v_beta op.
        // Output overwrites buf_vbeta. Need a temp [C, Dv] bf16.
        // Use buf_kcd (which is [C, Dk]=16KB; we need [C, Dv]=16KB — same size).
        wmma_gemm_bf16<C, Dv, C>(
            buf_attn_bf, C, buf_vbeta, Dv, buf_kcd, Dv, warp, n_warps, warp_scratch);
        __syncthreads();
        // Copy buf_kcd -> buf_vbeta as the new v_new.
        for (int idx = tid; idx < C * Dv; idx += nt) buf_vbeta[idx] = buf_kcd[idx];
        __syncthreads();

        // ----- k_cd = T @ (k_beta * exp_g_cs) -----
        // Compute k_beta * exp_g_cs into buf_kbeta in place.
        for (int idx = tid; idx < C * Dk; idx += nt) {
            int j = idx / Dk;
            float ec = (j < chunk_len) ? s_exp_cs[j] : 0.0f;
            float kb = __bfloat162float(buf_kbeta[idx]);
            buf_kbeta[idx] = __float2bfloat16(kb * ec);
        }
        __syncthreads();
        // k_cd = T @ buf_kbeta  -> buf_kcd
        wmma_gemm_bf16<C, Dk, C>(
            buf_attn_bf, C, buf_kbeta, Dk, buf_kcd, Dk, warp, n_warps, warp_scratch);
        __syncthreads();

        // ----- v_prime = k_cd @ state; v_new -= v_prime  -----
        // Do in two halves of Dv: cast state[:, half] to bf16 in state_bf,
        // do matmul accumulating to a scratch [C, Dv/2] fp32, then subtract
        // from v_new and continue with the second half.
        // We use buf_attn (fp32 [C, C] = 16KB = 4096 floats) as scratch since
        // C*Dv/2 = 64*64 = 4096 fp32 — exactly fits.
        for (int half = 0; half < N_VHALVES; half++) {
            int dv_off = half * VHALF;
            // Cast state[:, dv_off..dv_off+VHALF-1] -> state_bf [Dk, VHALF].
            for (int idx = tid; idx < Dk * VHALF; idx += nt) {
                int i = idx / VHALF;
                int d = idx - i * VHALF;
                state_bf[i * VHALF + d] = __float2bfloat16(state[i * Dv + dv_off + d]);
            }
            __syncthreads();
            // [C, VHALF] = k_cd [C, Dk] @ state_bf [Dk, VHALF]
            wmma_gemm_fp32<C, VHALF, Dk>(
                buf_kcd, Dk, state_bf, VHALF, buf_attn, VHALF,
                warp, n_warps, false);
            __syncthreads();
            // v_new[:, dv_off..] -= buf_attn (fp32)
            for (int idx = tid; idx < C * VHALF; idx += nt) {
                int j = idx / VHALF;
                int d = idx - j * VHALF;
                float v_old = __bfloat162float(buf_vbeta[j * Dv + dv_off + d]);
                buf_vbeta[j * Dv + dv_off + d] = __float2bfloat16(v_old - buf_attn[idx]);
            }
            __syncthreads();
        }

        // ----- attn_in = q @ k.T * decay_mask  -----
        // A = buf_q [C, Dk]; B = k.T from buf_k col_major [Dk, C].
        wmma_gemm_fp32<C, C, Dk, false, true>(
            buf_q, Dk, buf_k, Dk, buf_attn, C, warp, n_warps, false);
        __syncthreads();
        // Apply decay_mask (lower-and-diag kept, above zeroed).
        for (int idx = tid; idx < C * C; idx += nt) {
            int i = idx / C, j = idx - i * C;
            if (i >= j && i < chunk_len && j < chunk_len) {
                buf_attn[idx] *= buf_decay[idx];
            } else {
                buf_attn[idx] = 0.0f;
            }
        }
        __syncthreads();
        // Cast to bf16 in buf_attn_bf.
        for (int idx = tid; idx < C * C; idx += nt) {
            buf_attn_bf[idx] = __float2bfloat16(buf_attn[idx]);
        }
        __syncthreads();

        // ----- attn_int = (q * exp_g_cs) @ state  -----
        // Scale q in place: buf_q[j, d] *= exp_g_cs[j].
        for (int idx = tid; idx < C * Dk; idx += nt) {
            int j = idx / Dk;
            float ec = (j < chunk_len) ? s_exp_cs[j] : 0.0f;
            float qq = __bfloat162float(buf_q[idx]);
            buf_q[idx] = __float2bfloat16(qq * ec);
        }
        __syncthreads();
        // attn_int [C, Dv] = buf_q [C, Dk] @ state [Dk, Dv]. N_VHALVES passes.
        // Output to buf_kcd (reused; was k_cd, free now after v_prime done).
        for (int half = 0; half < N_VHALVES; half++) {
            int dv_off = half * VHALF;
            for (int idx = tid; idx < Dk * VHALF; idx += nt) {
                int i = idx / VHALF;
                int d = idx - i * VHALF;
                state_bf[i * VHALF + d] = __float2bfloat16(state[i * Dv + dv_off + d]);
            }
            __syncthreads();
            wmma_gemm_fp32<C, VHALF, Dk>(
                buf_q, Dk, state_bf, VHALF, buf_attn, VHALF,
                warp, n_warps, false);
            __syncthreads();
            // Cast buf_attn (fp32 [C, VHALF]) to bf16 in buf_kcd[:, dv_off..]
            for (int idx = tid; idx < C * VHALF; idx += nt) {
                int j = idx / VHALF;
                int d = idx - j * VHALF;
                buf_kcd[j * Dv + dv_off + d] = __float2bfloat16(buf_attn[idx]);
            }
            __syncthreads();
        }

        // ----- y_chunk = attn_int + attn_in_bf @ v_new -----
        // attn_in_bf is in buf_attn_bf [C, C]. v_new is in buf_vbeta [C, Dv].
        // We compute attn_in @ v_new in two halves (Dv/2 each), accumulate
        // into buf_attn (fp32), then add attn_int (in buf_kcd) and write y.
        for (int half = 0; half < N_VHALVES; half++) {
            int dv_off = half * VHALF;
            wmma_gemm_fp32<C, VHALF, C>(
                buf_attn_bf, C, buf_vbeta + dv_off, Dv,
                buf_attn, VHALF, warp, n_warps, false);
            __syncthreads();
            for (int idx = tid; idx < C * VHALF; idx += nt) {
                int j = idx / VHALF;
                int d = idx - j * VHALF;
                if (j >= chunk_len) continue;
                float ai = __bfloat162float(buf_kcd[j * Dv + dv_off + d]);
                float ax = buf_attn[idx];
                y_h[(t0 + j) * y_pos_stride + dv_off + d] = __float2bfloat16(ai + ax);
            }
            __syncthreads();
        }

        // ----- state = state * exp(g_chunk_total) + (k * exp(g_total - g_cs))^T @ v_new -----
        float g_total = s_g_cs[chunk_len - 1];
        float exp_total = expf(g_total);
        for (int idx = tid; idx < Dk * Dv; idx += nt) state[idx] *= exp_total;
        __syncthreads();
        // k_decay[j, d] = k[j, d] * exp(g_total - g_cs[j]) -> reuse buf_kbeta
        for (int idx = tid; idx < C * Dk; idx += nt) {
            int j = idx / Dk;
            float scale = (j < chunk_len) ? expf(g_total - s_g_cs[j]) : 0.0f;
            float kk = __bfloat162float(buf_k[idx]);
            buf_kbeta[idx] = __float2bfloat16(kk * scale);
        }
        __syncthreads();
        // state += k_decay.T @ v_new
        // A = k_decay.T from buf_kbeta col_major [Dk, C]; B = buf_vbeta [C, Dv].
        // Output: state [Dk, Dv], accumulate=true.
        wmma_gemm_fp32<Dk, Dv, C, true, false>(
            buf_kbeta, Dk, buf_vbeta, Dv, state, Dv, warp, n_warps, true);
        __syncthreads();

        // Save state at end of chunk c into state_chunks[c+1].
        if (state_chunks_h) {
            float *dst = state_chunks_h + (size_t)(c + 1) * Dk * Dv;
            for (int i = tid; i < Dk * Dv; i += nt) dst[i] = state[i];
        }
        __syncthreads();
    }

    if (state_out_h) {
        for (int i = tid; i < Dk * Dv; i += nt) {
            int row = i / Dv;
            int col = i - row * Dv;
            state_out_h[row * Dv_full + col] = state[i];
        }
    }
}


// ===================== launcher =====================
//
// Wraps `dn_chunked_vsplit_kernel` with explicit byte-strides for q/k/v and
// beta/g. This is the only difference vs the b200-train trainer version:
// instead of assuming q/k/v are dense [S, H, D] buffers, we let the
// caller pass strides so the kernel can read directly out of prefill's
// interleaved qkv_proj scratch ([S, DN_CONV_CH] where each S row holds
// q | k | v back-to-back).
//
// Caller responsibilities:
//   - q/k/v point at the start of their channel slice within qkv_proj.
//   - beta and g are [S, H] flat (bd_stride = H).
//   - state_in/state_out are [H, Dk, Dv] fp32 (state_in may equal state_out).
//   - state_chunks may be nullptr — only used by the (training-side) bwd.
//
// Inference does NOT need state_chunks; pass nullptr.
extern "C" cudaError_t launch_dn_chunked_vsplit_fwd(
    const __nv_bfloat16 *q, const __nv_bfloat16 *k, const __nv_bfloat16 *v,
    const float *beta, const float *g, const float *state_in,
    __nv_bfloat16 *y, float *state_out,
    float *state_chunks,                  // nullptr for inference
    int S, int H,
    int qkd_stride, int v_stride, int bd_stride,
    int y_stride,
    cudaStream_t stream)
{
    constexpr int Dk = DN_KEY_DEFAULT;
    // Match kernel: V_SPLITS=8 so Dv per block is 16 and the kernel
    // processes state in a single Dv-wide pass (state_bf is [Dk, Dv]).
    constexpr int Dv = 16;
    constexpr int V_SPLITS = 8;
    constexpr int C  = DN_CHUNK;
    size_t smem_fp32 = ((size_t)Dk * Dv + 2 * (size_t)C * C + 16 * 256 + 3 * C) * sizeof(float);
    size_t smem_bf16 = ((size_t)Dk * Dv + 4 * (size_t)C * Dk + (size_t)C * Dv + (size_t)C * C) * sizeof(__nv_bfloat16);
    size_t smem = smem_fp32 + smem_bf16;
    int threads = 512;
    cudaFuncSetAttribute(dn_chunked_vsplit_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem);
    // The kernel already uses the strides we pass; nothing to do but
    // forward them. For prefill's qkv_proj the kernel expects qkd_stride
    // bytes-of-bf16 between successive S rows of q (and the same buffer
    // also holds k starting `DN_QK_SIZE` later — caller has already
    // shifted the k pointer).
    dn_chunked_vsplit_kernel<<<H * V_SPLITS, threads, smem, stream>>>(
        q, k, v, beta, g, state_in, y, state_out, state_chunks,
        S, qkd_stride, v_stride, bd_stride, y_stride);
    return cudaGetLastError();
}
