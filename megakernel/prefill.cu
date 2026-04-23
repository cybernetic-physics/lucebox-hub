/**
 * BF16 Prefill: cuBLAS bf16 GEMM + standalone recurrence kernel.
 * Weights bf16, activations bf16, state f32. No quantization, no conversion.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

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

// ===== Standalone DeltaNet recurrence (state-in-registers, bf16 I/O, f32 state) =====
//
// Prefill recurrence: 16 heads, one CUDA block per head, state held in
// per-thread registers (16 warps × 32 lanes × 32 floats = 16384 = 128×128).
// Before the B200 port this kernel burned the majority of prefill wall time
// because the per-head conv1d shift lived in *global memory* and did
// 3 global-read + 3 global-write + 4 global-read per channel per step.
//
// B200 version: keep the whole conv1d ring-buffer (2×DN_KEY + DN_VAL
// channels × 4 taps = 1536 f32 = 6 KB) and the per-step weight taps in
// shared memory, and fuse the 3 per-group conv loops into a single
// 384-thread pass that uses all 3 of q/k/v channels at once instead of
// serialising the block through three barely-populated waves.
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

        // State-in-registers recurrence. Write fp32 result to s_out so the
        // gated-RMSNorm tail can read it without a bf16 round-trip through
        // global memory.
        for (int jj = 0; jj < CPW; jj++) {
            int j = wid * CPW + jj;
            float kv = 0;
            for (int ii = 0; ii < RPL; ii++) kv += sreg[jj*RPL+ii] * s_k[lid+ii*32];
            kv = pf_warp_sum(kv); kv = __shfl_sync(0xffffffff, kv, 0);
            float delta = (s_v[j] - decay * kv) * beta;
            float attn = 0;
            for (int ii = 0; ii < RPL; ii++) {
                sreg[jj*RPL+ii] = decay * sreg[jj*RPL+ii] + s_k[lid+ii*32] * delta;
                attn += sreg[jj*RPL+ii] * s_q[lid+ii*32];
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

// ===== Prefill body (capturable). All device work lives here so we can
//       wrap it in cudaStreamBeginCapture and replay via cudaGraphLaunch. =====
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
    cudaStream_t stream)
{
    int bk = (S*HIDDEN+255)/256;
    pf_embed<<<bk, 256, 0, stream>>>(token_ids, embed_weight, hidden, S);

    int fa_stride = FA_KV_HEADS * 2048 * FA_HEAD_DIM;
    int dn_stride = DN_HEADS * DN_KEY * DN_VAL;
    int fa_idx = 0, dn_idx = 0;

    for (int li = 0; li < NUM_LAYERS; li++) {
        const PFLayerWeights &lw = hl[li];
        int lt = LAYER_TYPE[li];

        const __nv_bfloat16 *norm_w = (const __nv_bfloat16 *)lw.ptrs[0];
        pf_rmsnorm<<<S, 512, 0, stream>>>(hidden, norm_w, normalized, residual, S, HIDDEN);

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

            // cuBLAS projections — direct bf16, no conversion!
            cublas_bf16_gemm(cublas, normalized, qkv_w, proj_buf, S, DN_CONV_CH, HIDDEN);
            cublas_bf16_gemm(cublas, normalized, z_w, proj_buf2, S, DN_V_SIZE, HIDDEN);
            pf_bf16_matvec<<<S*DN_HEADS, 32, 0, stream>>>(normalized, beta_w, beta_buf, S, HIDDEN, DN_HEADS);
            pf_bf16_matvec<<<S*DN_HEADS, 32, 0, stream>>>(normalized, alpha_w, alpha_buf, S, HIDDEN, DN_HEADS);

            // Standalone recurrence
            pf_deltanet_recurrence<<<DN_HEADS, 512, 0, stream>>>(
                proj_buf, proj_buf2, beta_buf, alpha_buf,
                conv_w, a_log, dt_bias, dn_norm,
                dn_states + dn_idx*dn_stride,
                conv_bufs + dn_idx*DN_CONV_CH*DN_CONV_K,
                dn_out_buf, S);

            // Out projection + residual
            cublas_bf16_gemm(cublas, dn_out_buf, out_w, proj_buf, S, HIDDEN, DN_V_SIZE);
            pf_add_residual_bf16<<<bk, 256, 0, stream>>>(proj_buf, residual, hidden, S*HIDDEN);

            // MLP
            pf_rmsnorm<<<S, 512, 0, stream>>>(hidden, post_norm, normalized, residual, S, HIDDEN);
            cublas_bf16_gemm(cublas, normalized, gate_w, proj_buf, S, INTER, HIDDEN);
            cublas_bf16_gemm(cublas, normalized, up_w, proj_buf2, S, INTER, HIDDEN);
            int mlp_bk = (S*INTER+255)/256;
            pf_silu_mul_bf16<<<mlp_bk, 256, 0, stream>>>(proj_buf, proj_buf2, mlp_buf, S*INTER);
            cublas_bf16_gemm(cublas, mlp_buf, down_w, proj_buf, S, HIDDEN, INTER);
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
            cublas_bf16_gemm(cublas, normalized, k_w, proj_buf2, S, FA_KV_SIZE, HIDDEN);
            cublas_bf16_gemm(cublas, normalized, v_w, attn_buf, S, FA_KV_SIZE, HIDDEN);

            int total_heads = S*(FA_Q_HEADS+FA_KV_HEADS);
            pf_qk_norm_rope<<<(total_heads+15)/16, 512, 0, stream>>>(
                proj_buf, proj_buf2, attn_buf, q_nw, k_nw,
                fa_k_cache + fa_idx*fa_stride, fa_v_cache + fa_idx*fa_stride, S, 2048);

            pf_causal_attn<<<(S*FA_Q_HEADS+15)/16, 512, 0, stream>>>(
                proj_buf, proj_buf2, attn_buf, dn_out_buf, S);

            cublas_bf16_gemm(cublas, dn_out_buf, o_w, proj_buf, S, HIDDEN, FA_Q_SIZE);
            pf_add_residual_bf16<<<bk, 256, 0, stream>>>(proj_buf, residual, hidden, S*HIDDEN);

            // MLP
            pf_rmsnorm<<<S, 512, 0, stream>>>(hidden, post_norm, normalized, residual, S, HIDDEN);
            cublas_bf16_gemm(cublas, normalized, gate_w, proj_buf, S, INTER, HIDDEN);
            cublas_bf16_gemm(cublas, normalized, up_w, proj_buf2, S, INTER, HIDDEN);
            int mlp_bk = (S*INTER+255)/256;
            pf_silu_mul_bf16<<<mlp_bk, 256, 0, stream>>>(proj_buf, proj_buf2, mlp_buf, S*INTER);
            cublas_bf16_gemm(cublas, mlp_buf, down_w, proj_buf, S, HIDDEN, INTER);
            pf_add_residual_bf16<<<bk, 256, 0, stream>>>(proj_buf, residual, hidden, S*HIDDEN);

            fa_idx++;
        }
    }

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
            lm_bmv, lm_bmi, stream);
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
            lm_bmv, lm_bmi, stream);
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
        lm_bmv, lm_bmi, capture_stream);
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
        lm_bmv, lm_bmi, stream);
    if (hit) {
        hit->eager_runs++;  // stay in warmup until capture finally works
    }
}
