/**
 * BF16 Prefill: cuBLAS bf16 GEMM + standalone recurrence kernel.
 * Weights bf16, activations bf16, state f32. No quantization, no conversion.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// ATen / torch for sdpa (cuDNN FlashAttention at large S).
#include <torch/torch.h>
#include <ATen/ATen.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

constexpr int HIDDEN = 1024;
constexpr int INTER = 3584;
constexpr int VOCAB = 248320;
constexpr float RMS_EPS = 1e-6f;

constexpr int FA_Q_HEADS = 8;
constexpr int FA_KV_HEADS = 2;
constexpr int FA_HEAD_DIM = 256;
constexpr int FA_GQA = FA_Q_HEADS / FA_KV_HEADS;
constexpr int FA_Q_SIZE = FA_Q_HEADS * FA_HEAD_DIM;
constexpr int FA_QPROJ_SIZE = FA_Q_SIZE * 2;
constexpr int FA_KV_SIZE = FA_KV_HEADS * FA_HEAD_DIM;
constexpr int FA_ROT_DIM = 64;
constexpr float FA_ROPE_THETA = 10000000.0f;

constexpr int DN_HEADS = 16;
constexpr int DN_KEY = 128;
constexpr int DN_VAL = 128;
constexpr int DN_CONV_K = 4;
constexpr int DN_QK_SIZE = DN_HEADS * DN_KEY;
constexpr int DN_V_SIZE = DN_HEADS * DN_VAL;
constexpr int DN_CONV_CH = DN_QK_SIZE * 2 + DN_V_SIZE;

constexpr int NUM_LAYERS = 24;
constexpr int LAYER_TYPE[24] = {0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1};

struct PFLayerWeights { int layer_type; int _pad[3]; void *ptrs[14]; };

__device__ __forceinline__ float pf_warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o); return v;
}
__device__ __forceinline__ float pf_silu(float x) { return x / (1.0f + expf(-x)); }

// Embedding
__global__ void pf_embed(const int *ids, const __nv_bfloat16 *embed, __nv_bfloat16 *out, int S) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= S * HIDDEN) return;
    out[idx] = embed[ids[idx / HIDDEN] * HIDDEN + idx % HIDDEN];
}

// Batched RMSNorm: bf16 in → bf16 out, saves bf16 residual
__global__ void pf_rmsnorm(const __nv_bfloat16 *in, const __nv_bfloat16 *w,
    __nv_bfloat16 *out, __nv_bfloat16 *res, int S, int D) {
    int s = blockIdx.x; if (s >= S) return;
    int tid = threadIdx.x, wid = tid/32, lid = tid%32;
    __shared__ float smem[32];
    const __nv_bfloat16 *ri = in + s*D;
    __nv_bfloat16 *ro = out + s*D, *rr = res + s*D;
    float sq = 0;
    for (int i = tid; i < D; i += blockDim.x) { float v = __bfloat162float(ri[i]); rr[i] = ri[i]; sq += v*v; }
    sq = pf_warp_sum(sq); if(lid==0) smem[wid]=sq; __syncthreads();
    if(wid==0){float v=(lid<blockDim.x/32)?smem[lid]:0;v=pf_warp_sum(v);if(lid==0)smem[0]=rsqrtf(v/D+RMS_EPS);}
    __syncthreads(); float rstd = smem[0];
    for (int i = tid; i < D; i += blockDim.x) {
        float v = __bfloat162float(ri[i]) * rstd * (1.0f + __bfloat162float(w[i]));
        ro[i] = __float2bfloat16(v);
    }
}

// bf16 matvec for tiny projections (beta/alpha)
__global__ void pf_bf16_matvec(const __nv_bfloat16 *in, const __nv_bfloat16 *w, float *out, int S, int K, int N) {
    int idx = blockIdx.x; if (idx >= S * N) return;
    int s = idx / N, n = idx % N, lid = threadIdx.x;
    const __nv_bfloat16 *ir = in + s*K, *wr = w + n*K;
    float sum = 0;
    for (int k = lid; k < K; k += 32) sum += __bfloat162float(ir[k]) * __bfloat162float(wr[k]);
    sum = pf_warp_sum(sum);
    if (lid == 0) out[idx] = sum;
}

// bf16 result + bf16 residual → bf16 output
__global__ void pf_add_residual_bf16(const __nv_bfloat16 *a, const __nv_bfloat16 *b, __nv_bfloat16 *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
}

// SiLU(gate) * up — bf16 inputs → bf16 output
__global__ void pf_silu_mul_bf16(const __nv_bfloat16 *gate, const __nv_bfloat16 *up, __nv_bfloat16 *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) { float g = __bfloat162float(gate[i]); out[i] = __float2bfloat16(pf_silu(g) * __bfloat162float(up[i])); }
}

// Extract the Q-only half of Qwen3.5's gated [S, H, 2*D] FA projection
// output into a dense [S, H, D] bf16 buffer, so sdpa / cuDNN FA can run.
// The source was written by pf_qk_norm_rope into proj_buf (post-RoPE,
// post-QK-norm values at offset 0..D of each head's 2*D slot).
__global__ void pf_fa_extract_q_dense(
    const __nv_bfloat16 *q_interleaved,   // [S, FA_Q_HEADS, 2*FA_HEAD_DIM]
    __nv_bfloat16 *q_dense,                // [S, FA_Q_HEADS, FA_HEAD_DIM]
    int S)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = S * FA_Q_HEADS * FA_HEAD_DIM;
    if (i >= total) return;
    int d = i % FA_HEAD_DIM;
    int hd = i / FA_HEAD_DIM;
    int h = hd % FA_Q_HEADS;
    int s = hd / FA_Q_HEADS;
    q_dense[i] = q_interleaved[s * FA_Q_HEADS * 2 * FA_HEAD_DIM
                               + h * 2 * FA_HEAD_DIM + d];
}

// Apply Qwen3.5's gated-attention output gate and write to the final
// per-position dense output. Reads sdpa's native [1, H, S, D] layout
// directly — avoids a transpose+contiguous 128 MB copy at S=32k × 6
// FA layers.
//
// out[s, h, d] = attn_out_BHSD[h, s, d] * sigmoid(gate[s, h, d])
// gate lives in the second half of each head's slot in q_interleaved.
__global__ void pf_fa_apply_gate_bf16(
    const __nv_bfloat16 *attn_out_BHSD,   // [1, H, S, D] as left by sdpa
    const __nv_bfloat16 *q_interleaved,   // [S, H, 2*D]
    __nv_bfloat16 *out,                    // [S, H, D] final dense output
    int S)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = S * FA_Q_HEADS * FA_HEAD_DIM;
    if (i >= total) return;
    int d = i % FA_HEAD_DIM;
    int hd = i / FA_HEAD_DIM;
    int h = hd % FA_Q_HEADS;
    int s = hd / FA_Q_HEADS;
    // Read attn_out from [H, S, D] layout.
    const __nv_bfloat16 *a_src = attn_out_BHSD
        + h * S * FA_HEAD_DIM + s * FA_HEAD_DIM + d;
    // Read gate from interleaved [S, H, 2D] layout (second half).
    const __nv_bfloat16 *g_src = q_interleaved
        + s * FA_Q_HEADS * 2 * FA_HEAD_DIM
        + h * 2 * FA_HEAD_DIM + FA_HEAD_DIM + d;
    float g = __bfloat162float(*g_src);
    float gsig = 1.0f / (1.0f + expf(-g));
    float a = __bfloat162float(*a_src);
    out[i] = __float2bfloat16(a * gsig);
}

// ===== DeltaNet per-step prep kernel =====
//
// Before the V-split recurrence runs its per-step loop, precompute the
// post-silu post-L2normalize q/k, post-silu v, sigmoid(beta) and
// decay = exp(-exp(a_log)*softplus(alpha+dt_bias)) for all (t, head)
// pairs in parallel. All of these are O(1) per-step operations that
// don't depend on the recurrence state — pulling them out of the hot
// loop turns the recurrence into pure V-local arithmetic with no per-
// step __syncthreads (the qkv_proj / beta / decay slices each thread
// needs are independent across threads).
//
// We write post-silu/post-norm values in-place over qkv_proj, and
// sigmoid(beta) / decay in-place over beta_buf / alpha_buf.
// conv_buf's final-4-taps update is done by pf_dn_conv_buf_update
// afterwards to avoid in-kernel read/write races on conv_buf.
//
// Grid: (S, DN_HEADS). Block: 128 threads (4 warps). Each block handles
// one (t, h) — computes conv for the 384 per-head channels (3 per
// thread), normalizes q and k, and writes processed beta/decay.
// IMPORTANT: pf_dn_prep must NOT overwrite qkv_proj in place. Grid scheduling
// on CUDA is unordered across blocks within a single kernel launch, so a
// block at (t, h) may read qkv_proj[t-3 .. t-1] AFTER the block at (t-1, h)
// has already written post-silu/post-norm values there. That races in the
// conv1d inputs and corrupts downstream DeltaNet state.
//
// Fix: write to a SEPARATE `qkv_prepped` output buffer. The downstream
// recurrence kernel reads from this buffer, not qkv_proj.
__global__ void pf_dn_prep(
    const __nv_bfloat16 *qkv_proj,           // IN only (raw, stays intact)
    __nv_bfloat16 *qkv_prepped,              // OUT post-conv-silu-norm
    const __nv_bfloat16 *conv_w,
    const float *conv_buf,                   // IN only (history from prior prefill)
    const __nv_bfloat16 *a_log,
    const __nv_bfloat16 *dt_bias,
    float *beta_buf,                          // IN raw / OUT sigmoid(raw)
    float *alpha_buf,                         // IN raw / OUT decay
    int S)
{
    int t = blockIdx.x;
    int h = blockIdx.y;
    if (t >= S || h >= DN_HEADS) return;

    int tid = threadIdx.x, wid = tid/32, lid = tid%32;
    constexpr int QKV_CH = 2 * DN_KEY + DN_VAL;     // 384 per-head channels
    constexpr float Q_SCALE = 1.0f / 11.313708498984761f;

    __shared__ float s_q[DN_KEY], s_k[DN_KEY], s_v[DN_VAL];

    // Conv: each thread handles up to ceil(384/128) = 3 channels.
    for (int c = tid; c < QKV_CH; c += blockDim.x) {
        int gch;
        if (c < DN_KEY) gch = h * DN_KEY + c;
        else if (c < 2 * DN_KEY) gch = DN_QK_SIZE + h * DN_KEY + (c - DN_KEY);
        else gch = 2 * DN_QK_SIZE + h * DN_VAL + (c - 2 * DN_KEY);

        // Read 4 conv taps. For pos < 0 fall back to conv_buf (prior-prefill
        // history). conv_buf[gch*4 + (pos+4)] when pos in [-3, -1] maps to
        // conv_buf[1], [2], [3] which are x[-3], x[-2], x[-1] respectively.
        float x[4];
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            int pos = t - 3 + k;
            if (pos >= 0) {
                x[k] = __bfloat162float(qkv_proj[pos * DN_CONV_CH + gch]);
            } else {
                x[k] = conv_buf[gch * DN_CONV_K + (pos + 4)];
            }
        }
        float co = 0;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            co += x[k] * __bfloat162float(conv_w[gch * DN_CONV_K + k]);
        }
        float silu_out = pf_silu(co);

        if (c < DN_KEY)           s_q[c] = silu_out;
        else if (c < 2 * DN_KEY)  s_k[c - DN_KEY] = silu_out;
        else                       s_v[c - 2 * DN_KEY] = silu_out;
    }
    __syncthreads();

    // Normalize q (warp 0), normalize k (warp 1), beta/decay (warp 2 lane 0).
    // All disjoint shared-memory targets; one sync fences them all.
    if (wid == 0) {
        float sq = 0;
        for (int i = lid; i < DN_KEY; i += 32) sq += s_q[i] * s_q[i];
        sq = pf_warp_sum(sq);
        float n = rsqrtf(sq + 1e-6f) * Q_SCALE;
        n = __shfl_sync(0xffffffff, n, 0);
        for (int i = lid; i < DN_KEY; i += 32) s_q[i] *= n;
    } else if (wid == 1) {
        float sq = 0;
        for (int i = lid; i < DN_KEY; i += 32) sq += s_k[i] * s_k[i];
        sq = pf_warp_sum(sq);
        float n = rsqrtf(sq + 1e-6f);
        n = __shfl_sync(0xffffffff, n, 0);
        for (int i = lid; i < DN_KEY; i += 32) s_k[i] *= n;
    } else if (wid == 2 && lid == 0) {
        float raw_beta = beta_buf[t * DN_HEADS + h];
        float raw_alpha = alpha_buf[t * DN_HEADS + h];
        beta_buf[t * DN_HEADS + h] = 1.0f / (1.0f + expf(-raw_beta));
        float x_val = raw_alpha + __bfloat162float(dt_bias[h]);
        float sp = (x_val > 20.0f) ? x_val : logf(1.0f + expf(x_val));
        alpha_buf[t * DN_HEADS + h] = expf(-expf(__bfloat162float(a_log[h])) * sp);
    }
    __syncthreads();

    // Write post-silu/post-norm values to the SEPARATE output buffer.
    for (int c = tid; c < QKV_CH; c += blockDim.x) {
        int gch;
        float val;
        if (c < DN_KEY) {
            gch = h * DN_KEY + c;
            val = s_q[c];
        } else if (c < 2 * DN_KEY) {
            gch = DN_QK_SIZE + h * DN_KEY + (c - DN_KEY);
            val = s_k[c - DN_KEY];
        } else {
            gch = 2 * DN_QK_SIZE + h * DN_VAL + (c - 2 * DN_KEY);
            val = s_v[c - 2 * DN_KEY];
        }
        qkv_prepped[t * DN_CONV_CH + gch] = __float2bfloat16(val);
    }
}

// After pf_dn_prep: update conv_buf with the last 4 raw qkv_proj positions.
// This kernel reads from a separate buffer because pf_dn_prep overwrote
// qkv_proj. We pass the raw buffer here. Actually we use conv_buf_save
// allocated externally. Alternative: call this kernel BEFORE pf_dn_prep.
//
// We launch it BEFORE pf_dn_prep while qkv_proj still holds raw values.
__global__ void pf_dn_conv_buf_update(
    const __nv_bfloat16 *qkv_proj_raw,       // raw qkv_proj, pre-pre-pass
    float *conv_buf,
    int S)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= DN_CONV_CH) return;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int t_pos = S - 4 + i;
        float v;
        if (t_pos >= 0) {
            v = __bfloat162float(qkv_proj_raw[t_pos * DN_CONV_CH + tid]);
        } else {
            // S < 4 (unusual): inherit from prior history appropriately.
            v = conv_buf[tid * DN_CONV_K + (t_pos + 4)];
        }
        conv_buf[tid * DN_CONV_K + i] = v;
    }
}

// ===== V-split DeltaNet recurrence (post-prep version) =====
//
// Reads PRE-PROCESSED qkv_proj (post-silu, post-L2norm) and PRE-PROCESSED
// beta_buf (sigmoid'd) and alpha_buf (= decay). Does pure recurrence: no
// conv, no normalize, no beta/decay compute in the hot loop. Each thread
// reads its own RPL-sized q/k slice from global directly into registers —
// no shared-memory staging, no per-step __syncthreads.
//
// Expected reduction vs the in-kernel-prep vsplit: removes ~50 % of
// per-step compute AND both per-step __syncthreads.
__global__ void __launch_bounds__(512, 1)
pf_deltanet_recurrence_vsplit_prepped(
    const __nv_bfloat16 *qkv_proj_prepped,   // from pf_dn_prep: post-silu, post-norm
    const float *beta_buf_prepped,            // sigmoid(raw beta)
    const float *decay_buf_prepped,           // = exp(-exp(a_log) * softplus(...))
    float *state,
    __nv_bfloat16 *output,                    // [S, DN_V_SIZE] bf16
    int S, int num_v_splits)
{
    int block_id = blockIdx.x;
    int h = block_id / num_v_splits;
    int v_split = block_id % num_v_splits;
    if (h >= DN_HEADS) return;
    int tid = threadIdx.x, wid = tid/32, lid = tid%32;
    int NWARPS = blockDim.x / 32;

    int v_slice_size = DN_VAL / num_v_splits;
    int v_start = v_split * v_slice_size;
    int CPW_V   = v_slice_size / NWARPS;

    float *my_state = state + h * DN_KEY * DN_VAL;

    constexpr int RPL = DN_KEY / 32;
    float sreg[8 * RPL];
    #pragma unroll
    for (int i = 0; i < 8 * RPL; i++) sreg[i] = 0.0f;

    // Load state slice.
    for (int jj = 0; jj < CPW_V; jj++) {
        int j_global = v_start + wid * CPW_V + jj;
        #pragma unroll
        for (int ii = 0; ii < RPL; ii++) {
            sreg[jj*RPL+ii] = my_state[j_global * DN_KEY + lid + ii*32];
        }
    }

    // Global pointers for this head's q/k/v base offsets in qkv_proj.
    int q_base = h * DN_KEY;
    int k_base = DN_QK_SIZE + h * DN_KEY;
    int v_base_gch = 2 * DN_QK_SIZE + h * DN_VAL;

    for (int t = 0; t < S; t++) {
        const __nv_bfloat16 *q_row = qkv_proj_prepped + t * DN_CONV_CH + q_base;
        const __nv_bfloat16 *k_row = qkv_proj_prepped + t * DN_CONV_CH + k_base;
        const __nv_bfloat16 *v_row = qkv_proj_prepped + t * DN_CONV_CH + v_base_gch;

        // Each lane loads its 4-element slice of q and k from global.
        float sk_cache[RPL], sq_cache[RPL];
        #pragma unroll
        for (int ii = 0; ii < RPL; ii++) {
            sk_cache[ii] = __bfloat162float(k_row[lid + ii*32]);
            sq_cache[ii] = __bfloat162float(q_row[lid + ii*32]);
        }

        float beta  = beta_buf_prepped [t * DN_HEADS + h];
        float decay = decay_buf_prepped[t * DN_HEADS + h];

        __nv_bfloat16 *out_t_h = output + t * DN_V_SIZE + h * DN_VAL;

        #pragma unroll
        for (int jj = 0; jj < CPW_V; jj++) {
            int j_local = wid * CPW_V + jj;
            int j_global = v_start + j_local;
            float v_val = __bfloat162float(v_row[j_global]);

            float kv = 0;
            #pragma unroll
            for (int ii = 0; ii < RPL; ii++) kv += sreg[jj*RPL+ii] * sk_cache[ii];
            kv = pf_warp_sum(kv);
            kv = __shfl_sync(0xffffffff, kv, 0);
            float delta = (v_val - decay * kv) * beta;
            float attn = 0;
            #pragma unroll
            for (int ii = 0; ii < RPL; ii++) {
                sreg[jj*RPL+ii] = decay * sreg[jj*RPL+ii] + sk_cache[ii] * delta;
                attn += sreg[jj*RPL+ii] * sq_cache[ii];
            }
            attn = pf_warp_sum(attn);
            if (lid == 0) out_t_h[j_global] = __float2bfloat16(attn);
        }
        // No __syncthreads — every thread's reads are its own regs/global.
    }

    // State write-back.
    for (int jj = 0; jj < CPW_V; jj++) {
        int j_global = v_start + wid * CPW_V + jj;
        #pragma unroll
        for (int ii = 0; ii < RPL; ii++) {
            my_state[j_global * DN_KEY + lid + ii*32] = sreg[jj*RPL+ii];
        }
    }
}

// ===== V-split DeltaNet recurrence =====
//
// Key observation: the per-step work in the recurrence is almost entirely
// local per V-column of state. For each j in [0, DN_VAL):
//   kv[j]    = sum_i state[i, j] * k[i]             (local to j)
//   delta[j] = (v[j] - decay * kv[j]) * beta        (local to j)
//   state[i,j] = decay*state[i,j] + k[i]*delta[j]   (local to j)
//   attn[j]  = sum_i state[i, j] * q[i]             (local to j)
// Only the gated-RMSNorm at the end of each step reduces across V. We pull
// that out into a separate kernel (pf_deltanet_gnorm below) and split V
// across multiple blocks per head — each block owns a V slice and runs
// the full S sequential steps on its slice. The conv1d + q/k-L2-normalize
// + beta/decay prep stays per-block (redundant for q/k across V splits,
// which is OK: q/k conv is ~0.5% of per-step work).
//
// Before: 16 blocks (one per head) × 520 steps sequential.
// After:  (H * num_v_splits) blocks × 520 steps sequential — but per-step
//         wall shrinks because each block's recurrence inner loop covers
//         only V/num_v_splits j-values, and the gnorm cross-V reduction
//         is hoisted out. On B200 (148 SMs) we go from 16 SMs used to 64
//         with num_v_splits=4.
__global__ void __launch_bounds__(512, 1)
pf_deltanet_recurrence_vsplit(
    const __nv_bfloat16 *qkv_proj,
    const float *beta_proj, const float *alpha_proj,
    const __nv_bfloat16 *conv_w, const __nv_bfloat16 *a_log,
    const __nv_bfloat16 *dt_bias,
    float *state, float *conv_buf,
    __nv_bfloat16 *output,       // [S, DN_V_SIZE] bf16 — raw attn per step, gnorm applied later
    int S, int num_v_splits)
{
    int block_id = blockIdx.x;
    int h = block_id / num_v_splits;
    int v_split = block_id % num_v_splits;
    if (h >= DN_HEADS) return;
    int tid = threadIdx.x, wid = tid/32, lid = tid%32;
    int NWARPS = blockDim.x / 32;               // runtime-variable; supports 128/256/512 threads
    constexpr float Q_SCALE = 1.0f / 11.313708498984761f;
    constexpr int QKV_CH = 2 * DN_KEY + DN_VAL;

    int v_slice_size = DN_VAL / num_v_splits;   // 128, 64, 32, or 16
    int v_start = v_split * v_slice_size;
    int v_end   = v_start + v_slice_size;
    int CPW_V   = v_slice_size / NWARPS;        // j's per warp in this slice

    float a_log_val = __bfloat162float(a_log[h]);
    float dt_b = __bfloat162float(dt_bias[h]);

    // Shared layout — same as the full-V version, but s_v and s_out only
    // need to be V_SLICE long (the block's local V range). We still allocate
    // full DN_VAL to keep the indexing identical; waste is small (<2 KB).
    __shared__ float s_q[DN_KEY], s_k[DN_KEY], s_v[DN_VAL];
    __shared__ float s_beta, s_decay;
    __shared__ float s_conv[QKV_CH * DN_CONV_K];
    __shared__ float s_conv_w[QKV_CH * DN_CONV_K];

    float *my_state = state + h * DN_KEY * DN_VAL;

    constexpr int RPL = DN_KEY / 32;            // 4
    float sreg[8 * RPL];                        // max CPW_V=8 × RPL=4 when num_v_splits=1
    // zero the unused tail (when CPW_V < 8)
    #pragma unroll
    for (int i = 0; i < 8 * RPL; i++) sreg[i] = 0.0f;

    // Load my V-slice of state into registers.
    for (int jj = 0; jj < CPW_V; jj++) {
        int j_local = wid * CPW_V + jj;
        int j_global = v_start + j_local;
        #pragma unroll
        for (int ii = 0; ii < RPL; ii++) {
            sreg[jj*RPL+ii] = my_state[j_global * DN_KEY + lid + ii*32];
        }
    }

    auto ch_global = [&] (int c) -> int {
        if (c < DN_KEY) return h * DN_KEY + c;
        if (c < 2 * DN_KEY) return DN_QK_SIZE + h * DN_KEY + (c - DN_KEY);
        return 2 * DN_QK_SIZE + h * DN_VAL + (c - 2 * DN_KEY);
    };

    // One-time: load conv state + conv weights for all 384 per-head channels.
    // Every V-split block redundantly loads q/k channels; v channels overlap
    // for their slice. Fine — it's 6 KB each, one-shot.
    for (int c = tid; c < QKV_CH; c += blockDim.x) {
        int gch = ch_global(c);
        const float *src_state = conv_buf + gch * DN_CONV_K;
        const __nv_bfloat16 *src_w = conv_w + gch * DN_CONV_K;
        float *dst_state = s_conv + c * DN_CONV_K;
        float *dst_w = s_conv_w + c * DN_CONV_K;
        #pragma unroll
        for (int k = 0; k < DN_CONV_K; k++) {
            dst_state[k] = src_state[k];
            dst_w[k] = __bfloat162float(src_w[k]);
        }
    }
    __syncthreads();

    for (int t = 0; t < S; t++) {
        // Fused conv1d+SiLU pass for all per-head channels.
        for (int c = tid; c < QKV_CH; c += blockDim.x) {
            int gch = ch_global(c);
            float *cs = s_conv + c * DN_CONV_K;
            const float *cw = s_conv_w + c * DN_CONV_K;
            float new_x = __bfloat162float(qkv_proj[t * DN_CONV_CH + gch]);
            float h0 = cs[1], h1 = cs[2], h2 = cs[3];
            cs[0] = h0; cs[1] = h1; cs[2] = h2; cs[3] = new_x;
            float co = h0 * cw[0] + h1 * cw[1] + h2 * cw[2] + new_x * cw[3];
            float silu_out = pf_silu(co);
            if (c < DN_KEY)           s_q[c] = silu_out;
            else if (c < 2 * DN_KEY)  s_k[c - DN_KEY] = silu_out;
            else                       s_v[c - 2 * DN_KEY] = silu_out;
        }
        __syncthreads();

        // Normalize q/k (warps 0, 1), beta/decay (warp 2 lane 0) — parallel.
        if(wid==0){float sq=0;for(int i=lid;i<DN_KEY;i+=32)sq+=s_q[i]*s_q[i];sq=pf_warp_sum(sq);float n=rsqrtf(sq+1e-6f)*Q_SCALE;n=__shfl_sync(0xffffffff,n,0);for(int i=lid;i<DN_KEY;i+=32)s_q[i]*=n;}
        if(wid==1){float sq=0;for(int i=lid;i<DN_KEY;i+=32)sq+=s_k[i]*s_k[i];sq=pf_warp_sum(sq);float n=rsqrtf(sq+1e-6f);n=__shfl_sync(0xffffffff,n,0);for(int i=lid;i<DN_KEY;i+=32)s_k[i]*=n;}
        if(wid==2 && lid==0){s_beta=1.f/(1.f+expf(-beta_proj[t*DN_HEADS+h]));float x=alpha_proj[t*DN_HEADS+h]+dt_b;float sp=(x>20.f)?x:logf(1.f+expf(x));s_decay=expf(-expf(a_log_val)*sp);}
        __syncthreads();

        float beta = s_beta, decay = s_decay;

        // Cache the lane's slice of s_q/s_k in registers.
        float sk_cache[RPL];
        float sq_cache[RPL];
        #pragma unroll
        for (int ii = 0; ii < RPL; ii++) {
            sk_cache[ii] = s_k[lid + ii*32];
            sq_cache[ii] = s_q[lid + ii*32];
        }

        // Recurrence over this block's V slice. Writes raw attn (bf16)
        // directly to output[t, h, j_global] — gnorm + silu-mul applied in
        // the follow-up kernel.
        __nv_bfloat16 *out_t_h = output + t * DN_V_SIZE + h * DN_VAL;

        #pragma unroll
        for (int jj = 0; jj < CPW_V; jj++) {
            int j_local = wid * CPW_V + jj;
            int j_global = v_start + j_local;
            float kv = 0;
            #pragma unroll
            for (int ii = 0; ii < RPL; ii++) kv += sreg[jj*RPL+ii] * sk_cache[ii];
            kv = pf_warp_sum(kv);
            kv = __shfl_sync(0xffffffff, kv, 0);
            float delta = (s_v[j_global] - decay * kv) * beta;
            float attn = 0;
            #pragma unroll
            for (int ii = 0; ii < RPL; ii++) {
                sreg[jj*RPL+ii] = decay * sreg[jj*RPL+ii] + sk_cache[ii] * delta;
                attn += sreg[jj*RPL+ii] * sq_cache[ii];
            }
            attn = pf_warp_sum(attn);
            if (lid == 0) out_t_h[j_global] = __float2bfloat16(attn);
        }
        // No end-of-iter sync needed: next t's conv+z writes target per-thread
        // disjoint slots in s_conv / s_q / s_k / s_v; the sync at the top of
        // the next iteration's phase-A is sufficient.
    }

    // Write state for this V slice back to global.
    for (int jj = 0; jj < CPW_V; jj++) {
        int j_local = wid * CPW_V + jj;
        int j_global = v_start + j_local;
        #pragma unroll
        for (int ii = 0; ii < RPL; ii++) {
            my_state[j_global * DN_KEY + lid + ii*32] = sreg[jj*RPL+ii];
        }
    }

    // Only v_split=0 writes back conv_buf for this head (all splits compute
    // the same q/k conv state; v conv state is per-slice but we currently
    // write all channels' conv_buf from split 0 too, since every split ran
    // the full conv over all 384 channels identically).
    if (v_split == 0) {
        __syncthreads();
        for (int c = tid; c < QKV_CH; c += blockDim.x) {
            int gch = ch_global(c);
            float *dst_state = conv_buf + gch * DN_CONV_K;
            const float *src_state = s_conv + c * DN_CONV_K;
            #pragma unroll
            for (int k = 0; k < DN_CONV_K; k++) {
                dst_state[k] = src_state[k];
            }
        }
    }
}

// Gated RMSNorm over V for the DeltaNet output: one block per (t, head),
// reduces the raw attn across V, applies rstd * norm_w and silu-mul with z.
__global__ void pf_deltanet_gnorm(
    __nv_bfloat16 *output,          // [S, DN_V_SIZE] bf16, raw-in / gnormed-out
    const __nv_bfloat16 *z_proj,     // [S, DN_V_SIZE] bf16
    const __nv_bfloat16 *norm_w,     // [DN_VAL] bf16 (shared across heads)
    int S)
{
    int s = blockIdx.x / DN_HEADS;
    int h = blockIdx.x - s * DN_HEADS;
    if (s >= S) return;
    int tid = threadIdx.x;
    constexpr int V = DN_VAL;
    __shared__ float s_attn[V];
    __shared__ float s_sum_warp[4];  // 128 threads = 4 warps

    __nv_bfloat16 *out_row = output + s * DN_V_SIZE + h * V;
    const __nv_bfloat16 *z_row = z_proj + s * DN_V_SIZE + h * V;

    float a = __bfloat162float(out_row[tid]);
    s_attn[tid] = a;
    float sq = a * a;
    sq = pf_warp_sum(sq);
    int wid = tid >> 5;
    int lid = tid & 31;
    if (lid == 0) s_sum_warp[wid] = sq;
    __syncthreads();

    float total = s_sum_warp[0] + s_sum_warp[1] + s_sum_warp[2] + s_sum_warp[3];
    float rstd = rsqrtf(total / V + RMS_EPS);

    float a_f = s_attn[tid];
    float z_v = __bfloat162float(z_row[tid]);
    float nw = __bfloat162float(norm_w[tid]);
    out_row[tid] = __float2bfloat16(a_f * rstd * nw * pf_silu(z_v));
}

// ===== Standalone DeltaNet recurrence (kept for reference / num_v_splits=1) =====
__global__ void __launch_bounds__(512, 1)
pf_deltanet_recurrence(
    const __nv_bfloat16 *qkv_proj, const __nv_bfloat16 *z_proj,
    const float *beta_proj, const float *alpha_proj,
    const __nv_bfloat16 *conv_w, const __nv_bfloat16 *a_log,
    const __nv_bfloat16 *dt_bias, const __nv_bfloat16 *norm_w,
    float *state, float *conv_buf, __nv_bfloat16 *output, int S)
{
    int h = blockIdx.x; if (h >= DN_HEADS) return;
    int tid = threadIdx.x, wid = tid/32, lid = tid%32;
    constexpr int NWARPS = 16;
    constexpr float Q_SCALE = 1.0f / 11.313708498984761f;
    constexpr int QKV_CH = 2 * DN_KEY + DN_VAL;  // channels per head: 384

    float a_log_val = __bfloat162float(a_log[h]);
    float dt_b = __bfloat162float(dt_bias[h]);

    __shared__ float s_q[DN_KEY], s_k[DN_KEY], s_v[DN_VAL];
    __shared__ float s_beta, s_decay;
    __shared__ float s_gnorm[NWARPS];
    // Conv1d ring-buffer: [QKV_CH][DN_CONV_K] flattened. 6 KB per block.
    __shared__ float s_conv[QKV_CH * DN_CONV_K];
    // Per-head conv weights cached in shared memory (per-head slice of conv_w).
    // One-time load amortised across all S time steps.
    __shared__ float s_conv_w[QKV_CH * DN_CONV_K];
    // Cached per-head z projection row for the current t (reused in gated norm).
    __shared__ float s_z[DN_VAL];
    // Per-step recurrence output, reused by the tail gated-RMSNorm so we
    // avoid 3 global round-trips through out_h on every step.
    __shared__ float s_out[DN_VAL];
    // Per-head norm_w cached in shared — constant across all S steps.
    __shared__ float s_norm_w[DN_VAL];

    float *my_state = state + h * DN_KEY * DN_VAL;
    float *my_conv = conv_buf + /*head offset computed below per channel*/ 0;
    (void)my_conv;

    // Load state into registers
    constexpr int CPW = DN_VAL / NWARPS;  // 8
    constexpr int RPL = DN_KEY / 32;       // 4
    float sreg[CPW * RPL];  // 32 floats

    for (int jj = 0; jj < CPW; jj++) {
        int j = wid * CPW + jj;
        for (int ii = 0; ii < RPL; ii++)
            sreg[jj*RPL+ii] = my_state[j*DN_KEY + lid+ii*32];
    }

    // Map a per-head local channel c in [0, QKV_CH) to the global
    // conv_buf / conv_w / qkv_proj channel index.
    auto ch_global = [&] (int c) -> int {
        if (c < DN_KEY) return h * DN_KEY + c;
        if (c < 2 * DN_KEY) return DN_QK_SIZE + h * DN_KEY + (c - DN_KEY);
        return 2 * DN_QK_SIZE + h * DN_VAL + (c - 2 * DN_KEY);
    };

    // One-time: load conv state, conv weights, and norm_w for this head
    // into shared memory.
    for (int c = tid; c < QKV_CH; c += blockDim.x) {
        int gch = ch_global(c);
        const float *src_state = conv_buf + gch * DN_CONV_K;
        const __nv_bfloat16 *src_w = conv_w + gch * DN_CONV_K;
        float *dst_state = s_conv + c * DN_CONV_K;
        float *dst_w = s_conv_w + c * DN_CONV_K;
        #pragma unroll
        for (int k = 0; k < DN_CONV_K; k++) {
            dst_state[k] = src_state[k];
            dst_w[k] = __bfloat162float(src_w[k]);
        }
    }
    for (int i = tid; i < DN_VAL; i += blockDim.x) {
        s_norm_w[i] = __bfloat162float(norm_w[i]);
    }
    __syncthreads();

    for (int t = 0; t < S; t++) {
        // Single fused conv1d + SiLU pass over all 384 per-head channels.
        // Each thread handles at most ceil(384/512) = 1 channel.
        for (int c = tid; c < QKV_CH; c += blockDim.x) {
            int gch = ch_global(c);
            float *cs = s_conv + c * DN_CONV_K;
            const float *cw = s_conv_w + c * DN_CONV_K;
            float new_x = __bfloat162float(qkv_proj[t * DN_CONV_CH + gch]);
            // Shift: [x_{t-3}, x_{t-2}, x_{t-1}, x_t] in-place, then write new_x.
            float h0 = cs[1], h1 = cs[2], h2 = cs[3];
            cs[0] = h0; cs[1] = h1; cs[2] = h2; cs[3] = new_x;
            float co = h0 * cw[0] + h1 * cw[1] + h2 * cw[2] + new_x * cw[3];
            float silu_out = pf_silu(co);
            if (c < DN_KEY)           s_q[c] = silu_out;
            else if (c < 2 * DN_KEY)  s_k[c - DN_KEY] = silu_out;
            else                       s_v[c - 2 * DN_KEY] = silu_out;
        }
        // Prefetch z-row in parallel with conv — they go to disjoint arrays.
        {
            const __nv_bfloat16 *z_h_bf = z_proj + t * DN_V_SIZE + h * DN_VAL;
            for (int i = tid; i < DN_VAL; i += blockDim.x) {
                s_z[i] = __bfloat162float(z_h_bf[i]);
            }
        }
        __syncthreads();

        // L2 normalize q (warp 0), L2 normalize k (warp 1), and beta/decay
        // scalars (warp 2 lane 0). All three operate on disjoint shared
        // targets, so they can run in parallel behind a single sync.
        if(wid==0){float sq=0;for(int i=lid;i<DN_KEY;i+=32)sq+=s_q[i]*s_q[i];sq=pf_warp_sum(sq);float n=rsqrtf(sq+1e-6f)*Q_SCALE;n=__shfl_sync(0xffffffff,n,0);for(int i=lid;i<DN_KEY;i+=32)s_q[i]*=n;}
        if(wid==1){float sq=0;for(int i=lid;i<DN_KEY;i+=32)sq+=s_k[i]*s_k[i];sq=pf_warp_sum(sq);float n=rsqrtf(sq+1e-6f);n=__shfl_sync(0xffffffff,n,0);for(int i=lid;i<DN_KEY;i+=32)s_k[i]*=n;}
        if(wid==2 && lid==0){s_beta=1.f/(1.f+expf(-beta_proj[t*DN_HEADS+h]));float x=alpha_proj[t*DN_HEADS+h]+dt_b;float sp=(x>20.f)?x:logf(1.f+expf(x));s_decay=expf(-expf(a_log_val)*sp);}
        __syncthreads();
        float beta = s_beta, decay = s_decay;
        __nv_bfloat16 *out_h = output + t * DN_V_SIZE + h * DN_VAL;

        // State-in-registers recurrence. Cache each lane's RPL=4 slice of
        // s_q and s_k into registers once per step so the CPW=8 inner-loop
        // iterations read from regs instead of shared (was 8×4×2 = 64
        // shared reads per step per warp for s_k/s_q — now 4+4 once).
        float sk_cache[RPL];
        float sq_cache[RPL];
        #pragma unroll
        for (int ii = 0; ii < RPL; ii++) {
            sk_cache[ii] = s_k[lid + ii*32];
            sq_cache[ii] = s_q[lid + ii*32];
        }

        #pragma unroll
        for (int jj = 0; jj < CPW; jj++) {
            int j = wid * CPW + jj;
            float kv = 0;
            #pragma unroll
            for (int ii = 0; ii < RPL; ii++) kv += sreg[jj*RPL+ii] * sk_cache[ii];
            kv = pf_warp_sum(kv); kv = __shfl_sync(0xffffffff, kv, 0);
            float delta = (s_v[j] - decay * kv) * beta;
            float attn = 0;
            #pragma unroll
            for (int ii = 0; ii < RPL; ii++) {
                sreg[jj*RPL+ii] = decay * sreg[jj*RPL+ii] + sk_cache[ii] * delta;
                attn += sreg[jj*RPL+ii] * sq_cache[ii];
            }
            attn = pf_warp_sum(attn);
            if (lid == 0) s_out[j] = attn;
        }
        __syncthreads();

        // Gated RMSNorm → bf16 output. After the per-warp partial-sum
        // write+sync, every thread reads all 16 partials and computes rstd
        // locally, avoiding the second "warp 0 reduces then everyone syncs
        // to pick up rstd" shared-memory round trip. Saves one
        // __syncthreads per step (×520 steps ×18 layers per prefill).
        float sq2=0; for(int i=tid;i<DN_VAL;i+=512){ float v=s_out[i]; sq2+=v*v; }
        sq2=pf_warp_sum(sq2); if(lid==0) s_gnorm[wid]=sq2; __syncthreads();

        float total = 0;
        #pragma unroll
        for (int w = 0; w < NWARPS; w++) total += s_gnorm[w];
        float rstd = rsqrtf(total / DN_VAL + RMS_EPS);

        for(int i=tid;i<DN_VAL;i+=512){
            float n = s_out[i] * rstd * s_norm_w[i];
            out_h[i] = __float2bfloat16(n * pf_silu(s_z[i]));
        }
        // No __syncthreads() here: the next iteration's conv+z prefetch
        // writes to s_conv / s_q / s_k / s_v / s_z, and each thread only
        // touches its own slot of those arrays (stride = blockDim.x). The
        // first sync of the next iteration fences before any cross-thread
        // read (phase B reads s_q/s_k across warps). Saves one sync per
        // step (×520 steps ×18 layers = ~9k syncs removed per prefill).
    }

    // Write state back
    for (int jj = 0; jj < CPW; jj++) {
        int j = wid * CPW + jj;
        for (int ii = 0; ii < RPL; ii++)
            my_state[j*DN_KEY + lid+ii*32] = sreg[jj*RPL+ii];
    }

    // Write conv ring-buffer back to global so the next prefill call picks
    // up the trailing 4 taps. Only the shifted entries need to persist; we
    // push the whole thing.
    __syncthreads();
    for (int c = tid; c < QKV_CH; c += blockDim.x) {
        int gch = ch_global(c);
        float *dst_state = conv_buf + gch * DN_CONV_K;
        const float *src_state = s_conv + c * DN_CONV_K;
        #pragma unroll
        for (int k = 0; k < DN_CONV_K; k++) {
            dst_state[k] = src_state[k];
        }
    }
}

// ===== QK norm + RoPE + KV cache =====
__global__ void pf_qk_norm_rope(
    __nv_bfloat16 *q, __nv_bfloat16 *k, const __nv_bfloat16 *v,
    const __nv_bfloat16 *qnw, const __nv_bfloat16 *knw,
    __nv_bfloat16 *k_cache, __nv_bfloat16 *v_cache, int S, int max_seq)
{
    int idx = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lid = threadIdx.x % 32;
    int total_q = S * FA_Q_HEADS, total_k = S * FA_KV_HEADS;
    if (idx < total_q) {
        int pos = idx / FA_Q_HEADS, head = idx % FA_Q_HEADS;
        __nv_bfloat16 *qh = q + pos * FA_QPROJ_SIZE + head * FA_HEAD_DIM * 2;
        float ss = 0; for (int i = lid; i < FA_HEAD_DIM; i += 32) { float v = __bfloat162float(qh[i]); ss += v*v; }
        ss = pf_warp_sum(ss); float sc = rsqrtf(ss/FA_HEAD_DIM+RMS_EPS); sc = __shfl_sync(0xffffffff,sc,0);
        for (int i = lid; i < FA_HEAD_DIM; i += 32) {
            float normed = __bfloat162float(qh[i])*sc*(1.f+__bfloat162float(qnw[i]));
            if (i < FA_ROT_DIM) {
                float fe=float(2*(i%(FA_ROT_DIM/2)))/FA_ROT_DIM; float freq=float(pos)/powf(FA_ROPE_THETA,fe);
                float cv=cosf(freq),sv=sinf(freq); int p=(i<FA_ROT_DIM/2)?i+FA_ROT_DIM/2:i-FA_ROT_DIM/2;
                float pv=__bfloat162float(qh[p])*sc*(1.f+__bfloat162float(qnw[p]));
                qh[i]=__float2bfloat16((i<FA_ROT_DIM/2)?(normed*cv-pv*sv):(pv*sv+normed*cv));
            } else qh[i]=__float2bfloat16(normed);
        }
    }
    int kidx = idx - total_q;
    if (idx >= total_q && kidx < total_k) {
        int pos = kidx / FA_KV_HEADS, head = kidx % FA_KV_HEADS;
        __nv_bfloat16 *kh = k + pos*FA_KV_SIZE + head*FA_HEAD_DIM;
        const __nv_bfloat16 *vh = v + pos*FA_KV_SIZE + head*FA_HEAD_DIM;
        __nv_bfloat16 *kc = k_cache + head*max_seq*FA_HEAD_DIM + pos*FA_HEAD_DIM;
        __nv_bfloat16 *vc = v_cache + head*max_seq*FA_HEAD_DIM + pos*FA_HEAD_DIM;
        float ss = 0; for (int i = lid; i < FA_HEAD_DIM; i += 32) { float v = __bfloat162float(kh[i]); ss += v*v; }
        ss = pf_warp_sum(ss); float sc = rsqrtf(ss/FA_HEAD_DIM+RMS_EPS); sc = __shfl_sync(0xffffffff,sc,0);
        for (int i = lid; i < FA_HEAD_DIM; i += 32) {
            float normed = __bfloat162float(kh[i])*sc*(1.f+__bfloat162float(knw[i])); float fk;
            if (i < FA_ROT_DIM) {
                float fe=float(2*(i%(FA_ROT_DIM/2)))/FA_ROT_DIM; float freq=float(pos)/powf(FA_ROPE_THETA,fe);
                float cv=cosf(freq),sv=sinf(freq); int p=(i<FA_ROT_DIM/2)?i+FA_ROT_DIM/2:i-FA_ROT_DIM/2;
                float pv=__bfloat162float(kh[p])*sc*(1.f+__bfloat162float(knw[p]));
                fk=(i<FA_ROT_DIM/2)?(normed*cv-pv*sv):(pv*sv+normed*cv);
            } else fk=normed;
            kh[i]=__float2bfloat16(fk); kc[i]=__float2bfloat16(fk); vc[i]=vh[i];
        }
    }
}

// ===== Causal attention (bf16 Q/K/V, f32 accumulation, bf16 output) =====
__global__ void pf_causal_attn(const __nv_bfloat16 *q, const __nv_bfloat16 *k,
    const __nv_bfloat16 *v, __nv_bfloat16 *out, int S)
{
    int idx = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lid = threadIdx.x % 32;
    if (idx >= S * FA_Q_HEADS) return;
    int pos = idx / FA_Q_HEADS, qh = idx % FA_Q_HEADS, kvh = qh / FA_GQA;
    float scale = 1.0f / sqrtf(float(FA_HEAD_DIM));
    constexpr int EPL = FA_HEAD_DIM / 32;
    const __nv_bfloat16 *qv = q + pos*FA_QPROJ_SIZE + qh*FA_HEAD_DIM*2;
    const __nv_bfloat16 *gv = qv + FA_HEAD_DIM;
    __nv_bfloat16 *ov = out + pos*FA_Q_SIZE + qh*FA_HEAD_DIM;
    float ql[EPL]; for(int e=0;e<EPL;e++) ql[e]=__bfloat162float(qv[lid*EPL+e]);
    float oa[EPL]={}; float mx=-1e30f, se=0;
    for (int kp = 0; kp <= pos; kp++) {
        const __nv_bfloat16 *kv=k+kp*FA_KV_SIZE+kvh*FA_HEAD_DIM;
        const __nv_bfloat16 *vv=v+kp*FA_KV_SIZE+kvh*FA_HEAD_DIM;
        float sc=0; for(int e=0;e<EPL;e++) sc+=ql[e]*__bfloat162float(kv[lid*EPL+e]);
        sc=pf_warp_sum(sc)*scale; sc=__shfl_sync(0xffffffff,sc,0);
        float om=mx; mx=fmaxf(mx,sc); float ed=expf(om-mx); se=se*ed+expf(sc-mx);
        float wt=expf(sc-mx); for(int e=0;e<EPL;e++) oa[e]=oa[e]*ed+wt*__bfloat162float(vv[lid*EPL+e]);
    }
    float rs=1.f/se;
    for(int e=0;e<EPL;e++){int i=lid*EPL+e;float g=1.f/(1.f+expf(-__bfloat162float(gv[i])));ov[i]=__float2bfloat16(oa[e]*rs*g);}
}

// Final norm
__global__ void pf_final_norm(const __nv_bfloat16 *hidden, const __nv_bfloat16 *w,
    __nv_bfloat16 *normed, __nv_bfloat16 *hidden_out, int S) {
    int tid=threadIdx.x, wid=tid/32, lid=tid%32;
    __shared__ float smem[16];
    const __nv_bfloat16 *row = hidden + (S-1)*HIDDEN;
    float sq=0; for(int i=tid;i<HIDDEN;i+=blockDim.x){float v=__bfloat162float(row[i]);sq+=v*v;}
    sq=pf_warp_sum(sq);if(lid==0)smem[wid]=sq;__syncthreads();
    if(wid==0){float v=(lid<blockDim.x/32)?smem[lid]:0;v=pf_warp_sum(v);if(lid==0)smem[0]=rsqrtf(v/HIDDEN+RMS_EPS);}
    __syncthreads();float rstd=smem[0];
    for(int i=tid;i<HIDDEN;i+=blockDim.x){
        float v=__bfloat162float(row[i]);
        normed[i]=__float2bfloat16(v*rstd*(1.f+__bfloat162float(w[i])));
        hidden_out[i]=row[i];
    }
}

// LM head: bf16 weight × bf16 hidden
__global__ void pf_lm_head(const __nv_bfloat16 *hidden, const __nv_bfloat16 *w,
    float *bmv, int *bmi, int N) {
    __shared__ __nv_bfloat16 s_h[HIDDEN];
    for(int i=threadIdx.x;i<HIDDEN;i+=blockDim.x) s_h[i]=hidden[i];
    __syncthreads();
    int wid=threadIdx.x/32, lid=threadIdx.x%32, nw=blockDim.x/32;
    int rpb=(N+gridDim.x-1)/gridDim.x, rs=blockIdx.x*rpb, re=min(rs+rpb,N);
    float lm=-1e30f; int li=-1;
    for(int m=rs+wid;m<re;m+=nw){const __nv_bfloat16 *wr=w+m*HIDDEN;float s=0;
        for(int k=lid*8;k<HIDDEN;k+=32*8){for(int i=0;i<8;i++)s+=__bfloat162float(wr[k+i])*__bfloat162float(s_h[k+i]);}
        s=pf_warp_sum(s);if(lid==0&&s>lm){lm=s;li=m;}}
    lm=__shfl_sync(0xffffffff,lm,0);li=__shfl_sync(0xffffffff,li,0);
    __shared__ float wm[32]; __shared__ int wi[32];
    if(lid==0){wm[wid]=lm;wi[wid]=li;}__syncthreads();
    if(wid==0){float mv=(lid<nw)?wm[lid]:-1e30f;int mi=(lid<nw)?wi[lid]:-1;
        for(int o=16;o>0;o>>=1){float ov=__shfl_down_sync(0xffffffff,mv,o);int oi=__shfl_down_sync(0xffffffff,mi,o);if(ov>mv){mv=ov;mi=oi;}}
        if(lid==0){bmv[blockIdx.x]=mv;bmi[blockIdx.x]=mi;}}
}
__global__ void pf_lm_reduce(const float *bmv, const int *bmi, int *out, int nb) {
    int tid=threadIdx.x; float best=-1e30f; int bi=-1;
    for(int i=tid;i<nb;i+=blockDim.x){float v=bmv[i];if(v>best){best=v;bi=bmi[i];}}
    __shared__ float sv[256]; __shared__ int si[256];
    sv[tid]=best;si[tid]=bi;__syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){if(tid<s&&sv[tid+s]>sv[tid]){sv[tid]=sv[tid+s];si[tid]=si[tid+s];}__syncthreads();}
    if(tid==0)*out=si[0];
}

// ===== cuBLAS bf16 GEMM =====
static void cublas_bf16_gemm(cublasHandle_t h,
    const __nv_bfloat16 *A, const __nv_bfloat16 *B, __nv_bfloat16 *C,
    int S, int N, int K) {
    float alpha = 1.0f, beta_val = 0.0f;
    cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, S, K,
        &alpha, B, CUDA_R_16BF, K, A, CUDA_R_16BF, K,
        &beta_val, C, CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

// ===== cuBLAS bf16 GEMM, "NN" layout (row-major X @ W without transposing W) =====
// Used for LoRA matmuls where A is stored as [K_in, R] and B as [R, K_out].
// Supports alpha/beta so the B-matmul can accumulate into the base proj
// buffer via beta=1, saving a separate residual-add kernel.
static void cublas_bf16_gemm_nn(cublasHandle_t h,
    const __nv_bfloat16 *X, const __nv_bfloat16 *W, __nv_bfloat16 *Y,
    int S, int N, int K, float alpha, float beta) {
    cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, S, K,
        &alpha, W, CUDA_R_16BF, N, X, CUDA_R_16BF, K,
        &beta, Y, CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

// ===== LoRA pointer bundle (mirrors LoraSet in the trainer's kernel.cu) =====
// Each A is a contiguous buffer over (per-type-layers, K_in, R); each B is
// (per-type-layers, R, K_out). Nullable per entry. When all are null or
// lora_rank == 0, the forward path is a bit-exact copy of the
// inference-only prefill (no extra work).
struct LoraPFSet {
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

// Fire the LoRA A/B cuBLAS pair for one projection, accumulating into
// Y (which already holds the base projection output). Skips entirely if
// A or B is null or rank==0 — the whole branch collapses to a single null
// check in the captured graph.
static inline void apply_lora_linear(cublasHandle_t h,
    const __nv_bfloat16 *X,
    const __nv_bfloat16 *A_layer, const __nv_bfloat16 *B_layer,
    __nv_bfloat16 *Y_base, __nv_bfloat16 *lora_h_ws,
    int S, int N, int K_in, int lora_rank, float lora_scaling)
{
    if (A_layer == nullptr || B_layer == nullptr || lora_rank <= 0) return;
    // lora_h = X @ A_layer   (S, K_in) @ (K_in, R) → (S, R)
    cublas_bf16_gemm_nn(h, X, A_layer, lora_h_ws,
                        S, lora_rank, K_in, 1.0f, 0.0f);
    // Y_base += scaling * (lora_h @ B_layer)   (S, R) @ (R, N) → (S, N)
    cublas_bf16_gemm_nn(h, lora_h_ws, B_layer, Y_base,
                        S, N, lora_rank, lora_scaling, 1.0f);
}

// ===== Saved activations for training backward =====
//
// During training we need to recover per-layer intermediate activations
// when running backward. Each pointer here, when non-null, is a flat
// [NUM_LAYERS, S, DIM] bf16 buffer that the forward writes to at the
// corresponding checkpoint. All null ⇒ inference-only, zero overhead.
//
// hidden_in[L]           ← input to layer L (= output of layer L-1)
// normalized_in[L]       ← output of input RMSnorm of layer L
// normalized_post_attn[L]← output of post-attn RMSnorm of layer L
// mlp_inter[L]           ← output of silu(gate)*up of layer L
//
// DIM = HIDDEN for the first three, INTER for mlp_inter.
struct SavedActivationsPF {
    __nv_bfloat16 *hidden_in;
    __nv_bfloat16 *normalized_in;
    __nv_bfloat16 *normalized_post_attn;
    __nv_bfloat16 *mlp_inter;
};

static inline void save_bf16_slab(__nv_bfloat16 *dst, const __nv_bfloat16 *src,
                                  int layer_idx, int S, int dim, cudaStream_t stream) {
    if (dst == nullptr) return;
    size_t bytes = (size_t)S * dim * sizeof(__nv_bfloat16);
    cudaMemcpyAsync(dst + (size_t)layer_idx * S * dim, src, bytes,
                    cudaMemcpyDeviceToDevice, stream);
}

// ===== Prefill body (capturable). All device work lives here so we can
//       wrap it in cudaStreamBeginCapture and replay via cudaGraphLaunch. =====
// Helper: per-projection LoRA offset into a per-type-layer-major
// (layer_idx, K_IN, R) / (layer_idx, R, K_OUT) buffer. Returns nullptr
// when the base pointer is null (LoRA disabled for this slot).
static inline const __nv_bfloat16 *lora_layer_ptr_a(
    const __nv_bfloat16 *base, int layer_idx, int K_in, int rank) {
    if (base == nullptr) return nullptr;
    return base + (size_t)layer_idx * K_in * rank;
}
static inline const __nv_bfloat16 *lora_layer_ptr_b(
    const __nv_bfloat16 *base, int layer_idx, int K_out, int rank) {
    if (base == nullptr) return nullptr;
    return base + (size_t)layer_idx * rank * K_out;
}

static void prefill_bf16_body(
    cublasHandle_t cublas,
    const PFLayerWeights *hl,              // host-side mirror of layers[]
    const int *token_ids, int S, int *output_token,
    const __nv_bfloat16 *embed_weight,
    const __nv_bfloat16 *final_norm_w, const __nv_bfloat16 *lm_head_w,
    __nv_bfloat16 *fa_k_cache, __nv_bfloat16 *fa_v_cache,
    float *dn_states, float *conv_bufs,
    __nv_bfloat16 *hidden, __nv_bfloat16 *residual, __nv_bfloat16 *normalized,
    __nv_bfloat16 *proj_buf, __nv_bfloat16 *proj_buf2,
    __nv_bfloat16 *attn_buf, __nv_bfloat16 *mlp_buf,
    __nv_bfloat16 *dn_out_buf,
    float *beta_buf, float *alpha_buf,
    __nv_bfloat16 *final_normed, __nv_bfloat16 *hidden_bf16_out,
    float *lm_bmv, int *lm_bmi,
    // LoRA (optional — null pointers + rank==0 → inference-only path)
    LoraPFSet lora,
    int lora_rank, float lora_scaling, __nv_bfloat16 *lora_h_ws,
    // Saved activations for backward (optional — all-null → inference-only)
    SavedActivationsPF saved,
    // Output buffer for pf_dn_prep (sized [S, DN_CONV_CH] bf16). Must be
    // DISTINCT from proj_buf so blocks in pf_dn_prep don't race on reads
    // of the raw qkv values at (t-3..t-1) positions — see pf_dn_prep docs.
    __nv_bfloat16 *dn_qkv_prepped_scratch,
    // Scratch for the sdpa-based FA attention path. Sized [S, FA_Q_SIZE] bf16.
    // fa_q_dense holds the de-interleaved Q; fa_sdpa_out holds cuDNN FA's
    // output before we apply the Qwen3.5 gated-attention gate.
    __nv_bfloat16 *fa_q_dense_scratch,
    __nv_bfloat16 *fa_sdpa_out_scratch,
    cudaStream_t stream)
{
    int bk = (S*HIDDEN+255)/256;
    pf_embed<<<bk, 256, 0, stream>>>(token_ids, embed_weight, hidden, S);

    // Convenience wrapper — one line per projection, still nops out when
    // the per-type-layer pointer is null.
    #define APPLY_LORA(A_BASE, B_BASE, X, Y, K_IN, K_OUT, LAYER_IDX)               \
        apply_lora_linear(cublas, (X),                                             \
            lora_layer_ptr_a((A_BASE), (LAYER_IDX), (K_IN),  lora_rank),           \
            lora_layer_ptr_b((B_BASE), (LAYER_IDX), (K_OUT), lora_rank),           \
            (Y), lora_h_ws, S, (K_OUT), (K_IN), lora_rank, lora_scaling)

    // KV-cache stride MUST match the fa_k_cache / fa_v_cache tensor the
    // caller allocated. We use 32768 = 32 k context, which also matches
    // the pf_qk_norm_rope stride arg below.
    constexpr int PREFILL_KV_MAX = 32768;
    int fa_stride = FA_KV_HEADS * PREFILL_KV_MAX * FA_HEAD_DIM;
    int dn_stride = DN_HEADS * DN_KEY * DN_VAL;
    int fa_idx = 0, dn_idx = 0;

    for (int li = 0; li < NUM_LAYERS; li++) {
        const PFLayerWeights &lw = hl[li];
        int lt = LAYER_TYPE[li];

        // Save pre-norm hidden for input-RMSnorm bwd.
        save_bf16_slab(saved.hidden_in, hidden, li, S, HIDDEN, stream);

        const __nv_bfloat16 *norm_w = (const __nv_bfloat16 *)lw.ptrs[0];
        pf_rmsnorm<<<S, 512, 0, stream>>>(hidden, norm_w, normalized, residual, S, HIDDEN);

        // Save post-input-RMSnorm for QKV/DN-QKV bwd (LoRA A grad input).
        save_bf16_slab(saved.normalized_in, normalized, li, S, HIDDEN, stream);

        if (lt == 0) {
            // DeltaNet
            const __nv_bfloat16 *qkv_w=(const __nv_bfloat16*)lw.ptrs[1];
            const __nv_bfloat16 *z_w=(const __nv_bfloat16*)lw.ptrs[2];
            const __nv_bfloat16 *beta_w=(const __nv_bfloat16*)lw.ptrs[3];
            const __nv_bfloat16 *alpha_w=(const __nv_bfloat16*)lw.ptrs[4];
            const __nv_bfloat16 *conv_w=(const __nv_bfloat16*)lw.ptrs[5];
            const __nv_bfloat16 *a_log=(const __nv_bfloat16*)lw.ptrs[6];
            const __nv_bfloat16 *dt_bias=(const __nv_bfloat16*)lw.ptrs[7];
            const __nv_bfloat16 *dn_norm=(const __nv_bfloat16*)lw.ptrs[8];
            const __nv_bfloat16 *out_w=(const __nv_bfloat16*)lw.ptrs[9];
            const __nv_bfloat16 *post_norm=(const __nv_bfloat16*)lw.ptrs[10];
            const __nv_bfloat16 *gate_w=(const __nv_bfloat16*)lw.ptrs[11];
            const __nv_bfloat16 *up_w=(const __nv_bfloat16*)lw.ptrs[12];
            const __nv_bfloat16 *down_w=(const __nv_bfloat16*)lw.ptrs[13];

            // cuBLAS projections — direct bf16, no conversion.
            // LoRA applied in-place on each projection output when ptrs are
            // non-null; an identity no-op when LoRA is disabled.
            cublas_bf16_gemm(cublas, normalized, qkv_w, proj_buf, S, DN_CONV_CH, HIDDEN);
            APPLY_LORA(lora.dn_qkv_A, lora.dn_qkv_B, normalized, proj_buf, HIDDEN, DN_CONV_CH, dn_idx);
            cublas_bf16_gemm(cublas, normalized, z_w, proj_buf2, S, DN_V_SIZE, HIDDEN);
            APPLY_LORA(lora.dn_z_A, lora.dn_z_B, normalized, proj_buf2, HIDDEN, DN_V_SIZE, dn_idx);
            pf_bf16_matvec<<<S*DN_HEADS, 32, 0, stream>>>(normalized, beta_w, beta_buf, S, HIDDEN, DN_HEADS);
            pf_bf16_matvec<<<S*DN_HEADS, 32, 0, stream>>>(normalized, alpha_w, alpha_buf, S, HIDDEN, DN_HEADS);

            // Pre-pass: compute conv+silu+L2norm(q,k) for all (t, h) in
            // parallel, compute sigmoid(beta) and decay. Writes POST-processed
            // values to dn_qkv_prepped (NOT proj_buf — see pf_dn_prep docs
            // for the race-free buffer requirement). Grid (S, DN_HEADS).
            //
            // IMPORTANT: this must run BEFORE pf_dn_conv_buf_update so that
            // pf_dn_prep reads the PRIOR conv history from conv_bufs. If we
            // updated conv_bufs first, pf_dn_prep at t=0..2 would see this
            // prefill's own tail positions as "prior history" and silently
            // corrupt the recurrence. (Prior code had this reversed because
            // pf_dn_prep used to overwrite qkv_proj in-place; now that it
            // writes to a separate scratch we reorder.)
            {
                dim3 grid(S, DN_HEADS);
                pf_dn_prep<<<grid, 128, 0, stream>>>(
                    proj_buf, dn_qkv_prepped_scratch, conv_w,
                    conv_bufs + dn_idx*DN_CONV_CH*DN_CONV_K,
                    a_log, dt_bias, beta_buf, alpha_buf, S);
            }

            // Persist the last 4 raw qkv positions into conv_bufs for the
            // next prefill/decode call's conv history.
            {
                int conv_update_blocks = (DN_CONV_CH + 127) / 128;
                pf_dn_conv_buf_update<<<conv_update_blocks, 128, 0, stream>>>(
                    proj_buf, conv_bufs + dn_idx*DN_CONV_CH*DN_CONV_K, S);
            }

            // V-split recurrence: each block owns a V-slice of one head's
            // recurrence state and runs the full S sequential steps on it.
            // Reads pre-processed q/k/v/beta/decay from dn_qkv_prepped.
            int num_v_splits = 8;
            int dn_block_size = 512;
            if (const char *env = std::getenv("MEGAKERNEL_DN_V_SPLITS")) {
                int v = std::atoi(env);
                if (v == 1 || v == 2 || v == 4 || v == 8) num_v_splits = v;
            }
            if (const char *env = std::getenv("MEGAKERNEL_DN_BLOCK_SIZE")) {
                int b = std::atoi(env);
                if (b == 128 || b == 256 || b == 512) dn_block_size = b;
            }
            int v_slice = DN_V_SIZE / num_v_splits;
            int nwarps_want = dn_block_size / 32;
            while (v_slice < nwarps_want) { dn_block_size >>= 1; nwarps_want = dn_block_size / 32; }
            pf_deltanet_recurrence_vsplit_prepped<<<DN_HEADS * num_v_splits, dn_block_size, 0, stream>>>(
                dn_qkv_prepped_scratch, beta_buf, alpha_buf,
                dn_states + dn_idx*dn_stride,
                dn_out_buf, S, num_v_splits);
            // Gnorm + silu-mul with z, applied across full V per (t, head).
            pf_deltanet_gnorm<<<S * DN_HEADS, 128, 0, stream>>>(
                dn_out_buf, proj_buf2, dn_norm, S);

            // Out projection + residual
            cublas_bf16_gemm(cublas, dn_out_buf, out_w, proj_buf, S, HIDDEN, DN_V_SIZE);
            APPLY_LORA(lora.dn_out_A, lora.dn_out_B, dn_out_buf, proj_buf, DN_V_SIZE, HIDDEN, dn_idx);
            pf_add_residual_bf16<<<bk, 256, 0, stream>>>(proj_buf, residual, hidden, S*HIDDEN);

            // MLP
            pf_rmsnorm<<<S, 512, 0, stream>>>(hidden, post_norm, normalized, residual, S, HIDDEN);
            save_bf16_slab(saved.normalized_post_attn, normalized, li, S, HIDDEN, stream);
            cublas_bf16_gemm(cublas, normalized, gate_w, proj_buf, S, INTER, HIDDEN);
            APPLY_LORA(lora.dn_gate_A, lora.dn_gate_B, normalized, proj_buf, HIDDEN, INTER, dn_idx);
            cublas_bf16_gemm(cublas, normalized, up_w, proj_buf2, S, INTER, HIDDEN);
            APPLY_LORA(lora.dn_up_A, lora.dn_up_B, normalized, proj_buf2, HIDDEN, INTER, dn_idx);
            int mlp_bk = (S*INTER+255)/256;
            pf_silu_mul_bf16<<<mlp_bk, 256, 0, stream>>>(proj_buf, proj_buf2, mlp_buf, S*INTER);
            save_bf16_slab(saved.mlp_inter, mlp_buf, li, S, INTER, stream);
            cublas_bf16_gemm(cublas, mlp_buf, down_w, proj_buf, S, HIDDEN, INTER);
            APPLY_LORA(lora.dn_down_A, lora.dn_down_B, mlp_buf, proj_buf, INTER, HIDDEN, dn_idx);
            pf_add_residual_bf16<<<bk, 256, 0, stream>>>(proj_buf, residual, hidden, S*HIDDEN);

            dn_idx++;
        } else {
            // Full Attention
            const __nv_bfloat16 *q_w=(const __nv_bfloat16*)lw.ptrs[1];
            const __nv_bfloat16 *k_w=(const __nv_bfloat16*)lw.ptrs[2];
            const __nv_bfloat16 *v_w=(const __nv_bfloat16*)lw.ptrs[3];
            const __nv_bfloat16 *q_nw=(const __nv_bfloat16*)lw.ptrs[4];
            const __nv_bfloat16 *k_nw=(const __nv_bfloat16*)lw.ptrs[5];
            const __nv_bfloat16 *o_w=(const __nv_bfloat16*)lw.ptrs[6];
            const __nv_bfloat16 *post_norm=(const __nv_bfloat16*)lw.ptrs[7];
            const __nv_bfloat16 *gate_w=(const __nv_bfloat16*)lw.ptrs[8];
            const __nv_bfloat16 *up_w=(const __nv_bfloat16*)lw.ptrs[9];
            const __nv_bfloat16 *down_w=(const __nv_bfloat16*)lw.ptrs[10];

            cublas_bf16_gemm(cublas, normalized, q_w, proj_buf, S, FA_QPROJ_SIZE, HIDDEN);
            APPLY_LORA(lora.fa_q_A, lora.fa_q_B, normalized, proj_buf, HIDDEN, FA_QPROJ_SIZE, fa_idx);
            cublas_bf16_gemm(cublas, normalized, k_w, proj_buf2, S, FA_KV_SIZE, HIDDEN);
            APPLY_LORA(lora.fa_k_A, lora.fa_k_B, normalized, proj_buf2, HIDDEN, FA_KV_SIZE, fa_idx);
            cublas_bf16_gemm(cublas, normalized, v_w, attn_buf, S, FA_KV_SIZE, HIDDEN);
            APPLY_LORA(lora.fa_v_A, lora.fa_v_B, normalized, attn_buf, HIDDEN, FA_KV_SIZE, fa_idx);

            int total_heads = S*(FA_Q_HEADS+FA_KV_HEADS);
            pf_qk_norm_rope<<<(total_heads+15)/16, 512, 0, stream>>>(
                proj_buf, proj_buf2, attn_buf, q_nw, k_nw,
                fa_k_cache + fa_idx*fa_stride, fa_v_cache + fa_idx*fa_stride, S, PREFILL_KV_MAX);

            // Dense Q → [S, FA_Q_HEADS, FA_HEAD_DIM] bf16 (strip out the
            // gate half of the q_proj output so cuDNN FA sees a packed
            // tensor). Then call at::scaled_dot_product_attention, which
            // routes to cuDNN FlashAttention-2 on sm_100. Using sdpa
            // makes attention O(S * d) in memory (no materialized S×S
            // matrix) and uses bf16 tensor cores — scalar pf_causal_attn
            // saturates launch/bf16-scalar throughput beyond S≈8k.
            {
                int blocks_q = (S * FA_Q_SIZE + 255) / 256;
                pf_fa_extract_q_dense<<<blocks_q, 256, 0, stream>>>(
                    proj_buf, fa_q_dense_scratch, S);

                auto opts = torch::TensorOptions()
                    .dtype(torch::kBFloat16).device(torch::kCUDA);

                // Q: already [S, H, D] contiguous in q_dense.
                auto q_t = torch::from_blob(
                    (void *)fa_q_dense_scratch,
                    {1, S, (int64_t)FA_Q_HEADS, (int64_t)FA_HEAD_DIM},
                    opts).transpose(1, 2);   // [1, H, S, D]

                // K/V: strided view of the per-layer KV cache slice.
                // Layout: [FA_KV_HEADS, PREFILL_KV_MAX, FA_HEAD_DIM].
                // We view the first S rows as [1, FA_KV_HEADS, S, FA_HEAD_DIM]
                // with strides that skip the unused tail of MAX_SEQ rows.
                int64_t kv_head_stride = (int64_t)PREFILL_KV_MAX * FA_HEAD_DIM;
                auto k_t = torch::from_blob(
                    (void *)(fa_k_cache + fa_idx * fa_stride),
                    {1, (int64_t)FA_KV_HEADS, S, (int64_t)FA_HEAD_DIM},
                    {0, kv_head_stride, (int64_t)FA_HEAD_DIM, 1},
                    opts);
                auto v_t = torch::from_blob(
                    (void *)(fa_v_cache + fa_idx * fa_stride),
                    {1, (int64_t)FA_KV_HEADS, S, (int64_t)FA_HEAD_DIM},
                    {0, kv_head_stride, (int64_t)FA_HEAD_DIM, 1},
                    opts);

                // sdpa: causal, GQA (8 Q heads × 2 KV heads).
                auto o_t = at::scaled_dot_product_attention(
                    q_t, k_t, v_t,
                    /*attn_mask=*/c10::nullopt,
                    /*dropout_p=*/0.0,
                    /*is_causal=*/true,
                    /*scale=*/c10::nullopt,
                    /*enable_gqa=*/true);   // [1, H, S, D] contiguous
                auto o_cont = o_t.contiguous();  // cheap no-op if already

                // Apply Qwen3.5 attention-output gate — gate kernel reads
                // sdpa's native [H, S, D] layout directly (no transpose,
                // no intermediate copy) and writes dense [S, H, D] out.
                pf_fa_apply_gate_bf16<<<blocks_q, 256, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(o_cont.data_ptr()),
                    proj_buf, dn_out_buf, S);
            }

            cublas_bf16_gemm(cublas, dn_out_buf, o_w, proj_buf, S, HIDDEN, FA_Q_SIZE);
            APPLY_LORA(lora.fa_o_A, lora.fa_o_B, dn_out_buf, proj_buf, FA_Q_SIZE, HIDDEN, fa_idx);
            pf_add_residual_bf16<<<bk, 256, 0, stream>>>(proj_buf, residual, hidden, S*HIDDEN);

            // MLP
            pf_rmsnorm<<<S, 512, 0, stream>>>(hidden, post_norm, normalized, residual, S, HIDDEN);
            save_bf16_slab(saved.normalized_post_attn, normalized, li, S, HIDDEN, stream);
            cublas_bf16_gemm(cublas, normalized, gate_w, proj_buf, S, INTER, HIDDEN);
            APPLY_LORA(lora.fa_gate_A, lora.fa_gate_B, normalized, proj_buf, HIDDEN, INTER, fa_idx);
            cublas_bf16_gemm(cublas, normalized, up_w, proj_buf2, S, INTER, HIDDEN);
            APPLY_LORA(lora.fa_up_A, lora.fa_up_B, normalized, proj_buf2, HIDDEN, INTER, fa_idx);
            int mlp_bk = (S*INTER+255)/256;
            pf_silu_mul_bf16<<<mlp_bk, 256, 0, stream>>>(proj_buf, proj_buf2, mlp_buf, S*INTER);
            save_bf16_slab(saved.mlp_inter, mlp_buf, li, S, INTER, stream);
            cublas_bf16_gemm(cublas, mlp_buf, down_w, proj_buf, S, HIDDEN, INTER);
            APPLY_LORA(lora.fa_down_A, lora.fa_down_B, mlp_buf, proj_buf, INTER, HIDDEN, fa_idx);
            pf_add_residual_bf16<<<bk, 256, 0, stream>>>(proj_buf, residual, hidden, S*HIDDEN);

            fa_idx++;
        }
    }
    #undef APPLY_LORA

    pf_final_norm<<<1, 512, 0, stream>>>(hidden, final_norm_w, final_normed, hidden_bf16_out, S);

    int lm_blocks = 512;
    pf_lm_head<<<lm_blocks, 256, 0, stream>>>(final_normed, lm_head_w, lm_bmv, lm_bmi, VOCAB);
    pf_lm_reduce<<<1, 256, 0, stream>>>(lm_bmv, lm_bmi, output_token, lm_blocks);
}

// ===== Graph cache. Each entry is a compiled cudaGraphExec for one
//       specific (seq_len, all-pointer) tuple. The first call with a given
//       tuple captures + instantiates. Subsequent calls just cudaGraphLaunch.
//       final_bench.py reuses the same Decoder + buffers across all 8 prefill
//       runs, so after the first one we pay ~no per-call overhead. =====
struct PrefillGraphKey {
    int seq_len;
    const int *token_ids;
    int *output_token;
    const void *embed_weight;
    const PFLayerWeights *layers;
    const void *final_norm_w;
    const void *lm_head_w;
    const void *fa_k_cache;
    const void *fa_v_cache;
    const void *dn_states;
    const void *conv_bufs;
    const void *hidden;
    const void *residual;
    const void *normalized;
    const void *proj_buf;
    const void *proj_buf2;
    const void *attn_buf;
    const void *mlp_buf;
    const void *dn_out_buf;
    const void *beta_buf;
    const void *alpha_buf;
    const void *final_normed;
    const void *hidden_bf16_out;
    const void *lm_bmv;
    const void *lm_bmi;
    LoraPFSet lora;
    int lora_rank;
    float lora_scaling;
    const void *lora_h_ws;
    // Saved-activation slabs: distinct training vs inference vs grad-check
    // invocations get distinct cached graphs.
    SavedActivationsPF saved;
};

struct PrefillGraphEntry {
    PrefillGraphKey key;
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    int eager_runs;                 // eager runs before first capture attempt
    PFLayerWeights hl[NUM_LAYERS];  // host-side mirror captured with this graph
};

static constexpr int MAX_PREFILL_GRAPHS = 4;
static PrefillGraphEntry g_prefill_graph_cache[MAX_PREFILL_GRAPHS];
static int g_prefill_graph_count = 0;

static bool keys_equal(const PrefillGraphKey &a, const PrefillGraphKey &b) {
    return std::memcmp(&a, &b, sizeof(PrefillGraphKey)) == 0;
}

// ===== Main orchestrator =====
extern "C" void launch_prefill_bf16(
    const int *token_ids, int seq_len, int *output_token,
    const __nv_bfloat16 *embed_weight, const PFLayerWeights *layers,
    const __nv_bfloat16 *final_norm_w, const __nv_bfloat16 *lm_head_w,
    __nv_bfloat16 *fa_k_cache, __nv_bfloat16 *fa_v_cache,
    float *dn_states, float *conv_bufs,
    // Scratch (ALL bf16 except state/conv which are f32)
    __nv_bfloat16 *hidden, __nv_bfloat16 *residual, __nv_bfloat16 *normalized,
    __nv_bfloat16 *proj_buf, __nv_bfloat16 *proj_buf2,
    __nv_bfloat16 *attn_buf, __nv_bfloat16 *mlp_buf,
    __nv_bfloat16 *dn_out_buf,
    float *beta_buf, float *alpha_buf,
    __nv_bfloat16 *final_normed, __nv_bfloat16 *hidden_bf16_out,
    float *lm_bmv, int *lm_bmi,
    // LoRA (optional — null ptrs + rank==0 means "inference only", fast path)
    LoraPFSet lora, int lora_rank, float lora_scaling,
    __nv_bfloat16 *lora_h_ws,
    // Saved activations for training backward (optional — all-null = off)
    SavedActivationsPF saved,
    cudaStream_t stream)
{
    static cublasHandle_t cublas = nullptr;
    static void *cublas_workspace = nullptr;
    static cudaStream_t capture_stream = nullptr;
    static constexpr size_t CUBLAS_WORKSPACE_BYTES = 32ull * 1024 * 1024;  // 32 MB, fits sm_100 recommended
    if (!cublas) {
        cublasCreate(&cublas);
        cublasSetMathMode(cublas, CUBLAS_DEFAULT_MATH);
        // Pre-allocate a workspace so cuBLAS doesn't try to allocate on
        // stream during graph capture. Without this, the first GEMM under
        // capture triggers cudaErrorInvalidValue.
        cudaMalloc(&cublas_workspace, CUBLAS_WORKSPACE_BYTES);
        cublasSetWorkspace(cublas, cublas_workspace, CUBLAS_WORKSPACE_BYTES);
        // Private non-default stream for capture. PyTorch's
        // getCurrentCUDAStream() may hand us the legacy default stream, and
        // cudaStreamBeginCapture is not permitted on the default stream.
        cudaStreamCreateWithFlags(&capture_stream, cudaStreamNonBlocking);
    }
    cublasSetStream(cublas, stream);

    // Internal scratch for pf_dn_prep's race-free output. Sized to the
    // largest seq_len seen so far. Kept in a static allocator so we don't
    // need Python callers to plumb yet another buffer.
    static __nv_bfloat16 *dn_qkv_prepped = nullptr;
    static size_t dn_qkv_prepped_capacity = 0;
    size_t need_bytes = (size_t)seq_len * DN_CONV_CH * sizeof(__nv_bfloat16);
    if (need_bytes > dn_qkv_prepped_capacity) {
        if (dn_qkv_prepped) cudaFree(dn_qkv_prepped);
        cudaMalloc(&dn_qkv_prepped, need_bytes);
        dn_qkv_prepped_capacity = need_bytes;
    }

    // Internal scratch for the sdpa-based FA attention path:
    //   q_dense   [S, FA_Q_HEADS,     FA_HEAD_DIM] bf16
    //   sdpa_out  [S, FA_Q_HEADS,     FA_HEAD_DIM] bf16
    // At S=32k this is 2 * 32k * 8 * 256 * 2 = 128 MB total. Grow-as-needed.
    static __nv_bfloat16 *fa_q_dense = nullptr;
    static __nv_bfloat16 *fa_sdpa_out = nullptr;
    static size_t fa_scratch_capacity = 0;
    size_t fa_need = (size_t)seq_len * FA_Q_SIZE * sizeof(__nv_bfloat16);
    if (fa_need > fa_scratch_capacity) {
        if (fa_q_dense)  cudaFree(fa_q_dense);
        if (fa_sdpa_out) cudaFree(fa_sdpa_out);
        cudaMalloc(&fa_q_dense,  fa_need);
        cudaMalloc(&fa_sdpa_out, fa_need);
        fa_scratch_capacity = fa_need;
    }

    bool disable_graph = std::getenv("MEGAKERNEL_PREFILL_NOGRAPH") != nullptr;

    PrefillGraphKey key{};
    key.seq_len = seq_len;
    key.token_ids = token_ids;
    key.output_token = output_token;
    key.embed_weight = embed_weight;
    key.layers = layers;
    key.final_norm_w = final_norm_w;
    key.lm_head_w = lm_head_w;
    key.fa_k_cache = fa_k_cache;
    key.fa_v_cache = fa_v_cache;
    key.dn_states = dn_states;
    key.conv_bufs = conv_bufs;
    key.hidden = hidden;
    key.residual = residual;
    key.normalized = normalized;
    key.proj_buf = proj_buf;
    key.proj_buf2 = proj_buf2;
    key.attn_buf = attn_buf;
    key.mlp_buf = mlp_buf;
    key.dn_out_buf = dn_out_buf;
    key.beta_buf = beta_buf;
    key.alpha_buf = alpha_buf;
    key.final_normed = final_normed;
    key.hidden_bf16_out = hidden_bf16_out;
    key.lm_bmv = lm_bmv;
    key.lm_bmi = lm_bmi;
    key.lora = lora;
    key.lora_rank = lora_rank;
    key.lora_scaling = lora_scaling;
    key.lora_h_ws = lora_h_ws;
    key.saved = saved;

    PrefillGraphEntry *hit = nullptr;
    if (!disable_graph) {
        for (int i = 0; i < g_prefill_graph_count; i++) {
            if (keys_equal(g_prefill_graph_cache[i].key, key)) {
                hit = &g_prefill_graph_cache[i];
                break;
            }
        }
    }

    if (hit && hit->exec) {
        // Launch cached graph on the dedicated capture stream, with event
        // plumbing so the caller stream waits on the replay result.
        cudaEvent_t hit_evt;
        cudaEventCreateWithFlags(&hit_evt, cudaEventDisableTiming);
        cudaEventRecord(hit_evt, stream);
        cudaStreamWaitEvent(capture_stream, hit_evt, 0);
        cudaGraphLaunch(hit->exec, capture_stream);
        cudaEventRecord(hit_evt, capture_stream);
        cudaStreamWaitEvent(stream, hit_evt, 0);
        cudaEventDestroy(hit_evt);
        return;
    }

    // Host-side mirror of weight pointer table. Needed before capture so we
    // can refer to per-layer pointers from the host during kernel enqueues.
    PFLayerWeights hl_local[NUM_LAYERS];
    cudaMemcpy(hl_local, layers, NUM_LAYERS * sizeof(PFLayerWeights),
               cudaMemcpyDeviceToHost);

    // First eager warmup pass for a new key: lets cuBLAS do any workspace
    // allocation / heuristic selection outside of stream capture. We do it
    // once per key, then capture on the second call.
    int eager_warmups = 1;
    if (const char *env = std::getenv("MEGAKERNEL_PREFILL_WARMUPS")) {
        int v = std::atoi(env);
        if (v >= 0) eager_warmups = v;
    }

    if (disable_graph || (hit && hit->eager_runs < eager_warmups)) {
        prefill_bf16_body(
            cublas, hl_local,
            token_ids, seq_len, output_token,
            embed_weight, final_norm_w, lm_head_w,
            fa_k_cache, fa_v_cache, dn_states, conv_bufs,
            hidden, residual, normalized,
            proj_buf, proj_buf2, attn_buf, mlp_buf, dn_out_buf,
            beta_buf, alpha_buf,
            final_normed, hidden_bf16_out,
            lm_bmv, lm_bmi, lora, lora_rank, lora_scaling, lora_h_ws, saved, dn_qkv_prepped, fa_q_dense, fa_sdpa_out, stream);  // to body
        if (hit) hit->eager_runs++;
        return;
    }
    if (!hit) {
        // Record the key and run one eager pass before attempting capture.
        if (g_prefill_graph_count < MAX_PREFILL_GRAPHS) {
            PrefillGraphEntry &entry = g_prefill_graph_cache[g_prefill_graph_count++];
            entry.key = key;
            entry.graph = nullptr;
            entry.exec = nullptr;
            entry.eager_runs = 1;
            std::memcpy(entry.hl, hl_local, sizeof(hl_local));
        }
        prefill_bf16_body(
            cublas, hl_local,
            token_ids, seq_len, output_token,
            embed_weight, final_norm_w, lm_head_w,
            fa_k_cache, fa_v_cache, dn_states, conv_bufs,
            hidden, residual, normalized,
            proj_buf, proj_buf2, attn_buf, mlp_buf, dn_out_buf,
            beta_buf, alpha_buf,
            final_normed, hidden_bf16_out,
            lm_bmv, lm_bmi, lora, lora_rank, lora_scaling, lora_h_ws, saved, dn_qkv_prepped, fa_q_dense, fa_sdpa_out, stream);  // to body
        return;
    }

    // hit exists with eager_runs >= eager_warmups: capture into a graph.
    // Capture on our private stream (PyTorch's current stream may be the
    // legacy default stream, which cannot be captured). Sync the caller
    // stream into the capture stream before, and back to it after launch.
    bool verbose = std::getenv("MEGAKERNEL_PREFILL_GRAPH_DEBUG") != nullptr;
    cudaEvent_t join_evt;
    cudaEventCreateWithFlags(&join_evt, cudaEventDisableTiming);
    cudaEventRecord(join_evt, stream);
    cudaStreamWaitEvent(capture_stream, join_evt, 0);
    cublasSetStream(cublas, capture_stream);
    cudaError_t err;
    err = cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeRelaxed);
    if (err != cudaSuccess) {
        if (verbose) fprintf(stderr, "[prefill-graph] BeginCapture failed: %s\n", cudaGetErrorString(err));
        cudaGetLastError();  // clear
        goto fallback_eager;
    }
    prefill_bf16_body(
        cublas, hl_local,
        token_ids, seq_len, output_token,
        embed_weight, final_norm_w, lm_head_w,
        fa_k_cache, fa_v_cache, dn_states, conv_bufs,
        hidden, residual, normalized,
        proj_buf, proj_buf2, attn_buf, mlp_buf, dn_out_buf,
        beta_buf, alpha_buf,
        final_normed, hidden_bf16_out,
        lm_bmv, lm_bmi, lora, lora_rank, lora_scaling, lora_h_ws, saved, dn_qkv_prepped, fa_q_dense, fa_sdpa_out, capture_stream);
    {
        cudaGraph_t graph = nullptr;
        err = cudaStreamEndCapture(capture_stream, &graph);
        if (err != cudaSuccess) {
            if (verbose) fprintf(stderr, "[prefill-graph] EndCapture failed: %s\n", cudaGetErrorString(err));
            cudaGetLastError();
            if (graph) cudaGraphDestroy(graph);
            // Stream is now in an unknown state; rerun eagerly to recover.
            goto fallback_eager;
        }
        cudaGraphExec_t exec = nullptr;
        err = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
        if (err != cudaSuccess) {
            if (verbose) fprintf(stderr, "[prefill-graph] Instantiate failed: %s\n", cudaGetErrorString(err));
            cudaGetLastError();
            cudaGraphDestroy(graph);
            goto fallback_eager;
        }
        // hit is non-null here (we branched through the warmup path first).
        hit->graph = graph;
        hit->exec = exec;
        std::memcpy(hit->hl, hl_local, sizeof(hl_local));
        // Launch on the capture stream, then plumb the result back to the
        // caller's stream so subsequent PyTorch ops see the dependency.
        err = cudaGraphLaunch(exec, capture_stream);
        if (err != cudaSuccess && verbose) {
            fprintf(stderr, "[prefill-graph] GraphLaunch failed: %s\n", cudaGetErrorString(err));
        }
        cudaEventRecord(join_evt, capture_stream);
        cudaStreamWaitEvent(stream, join_evt, 0);
        cublasSetStream(cublas, stream);
        cudaEventDestroy(join_evt);
        return;
    }

fallback_eager:
    cublasSetStream(cublas, stream);
    cudaEventDestroy(join_evt);
    prefill_bf16_body(
        cublas, hl_local,
        token_ids, seq_len, output_token,
        embed_weight, final_norm_w, lm_head_w,
        fa_k_cache, fa_v_cache, dn_states, conv_bufs,
        hidden, residual, normalized,
        proj_buf, proj_buf2, attn_buf, mlp_buf, dn_out_buf,
        beta_buf, alpha_buf,
        final_normed, hidden_bf16_out,
        lm_bmv, lm_bmi, lora, lora_rank, lora_scaling, lora_h_ws, saved, dn_qkv_prepped, fa_q_dense, fa_sdpa_out, stream);
    if (hit) {
        hit->eager_runs++;  // stay in warmup until capture finally works
    }
}
