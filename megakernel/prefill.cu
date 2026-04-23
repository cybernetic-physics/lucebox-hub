/**
 * BF16 Prefill: cuBLAS bf16 GEMM + standalone recurrence kernel.
 * Weights bf16, activations bf16, state f32. No quantization, no conversion.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifndef PREFILL_DN_BLOCK_SIZE
#define PREFILL_DN_BLOCK_SIZE 256
#endif

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
constexpr int FA_ROT_PAIRS = FA_ROT_DIM / 2;

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

extern "C" bool launch_lm_head_cublaslt_bf16_top1(
    const void *hidden_bf16,
    const void *lm_head_weight_packed,
    const void *lm_head_scales,
    void *lm_hidden_bf16,
    void *lm_hidden_packed,
    void *lm_hidden_scales,
    void *lm_logits_f16,
    float *block_max_vals,
    int *block_max_idxs,
    int *output_token_id,
    cudaStream_t stream);

__device__ __constant__ float PF_ROPE_INV_FREQ[FA_ROT_PAIRS] = {
    1.000000000000000000e+00f, 6.042963902381328634e-01f,
    3.651741272548377215e-01f, 2.206734069084589911e-01f,
    1.333521432163324028e-01f, 8.058421877614818651e-02f,
    4.869675251658631132e-02f, 2.942727176209281731e-02f,
    1.778279410038922925e-02f, 1.074607828321317432e-02f,
    6.493816315762112983e-03f, 3.924189758484536265e-03f,
    2.371373705661655382e-03f, 1.433012570236962685e-03f,
    8.659643233600653866e-04f, 5.232991146814947340e-04f,
    3.162277660168379394e-04f, 1.910952974970440477e-04f,
    1.154781984689458215e-04f, 6.978305848598663529e-05f,
    4.216965034285822237e-05f, 2.548296747979346413e-05f,
    1.539926526059491854e-05f, 9.305720409296990429e-06f,
    5.623413251903491208e-06f, 3.398208328942559268e-06f,
    2.053525026457146066e-06f, 1.240937760751719527e-06f,
    7.498942093324558477e-07f, 4.531583637600817928e-07f,
    2.738419634264361394e-07f, 1.654817099943181354e-07f
};

__device__ __forceinline__ float pf_warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o); return v;
}
__device__ __forceinline__ float pf_sigmoid(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float pf_silu(float x) { return x * pf_sigmoid(x); }

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
template <int BLOCK_THREADS>
__global__ void __launch_bounds__(BLOCK_THREADS, 1)
pf_deltanet_recurrence(
    const __nv_bfloat16 *qkv_proj, const __nv_bfloat16 *z_proj,
    const float *beta_proj, const float *alpha_proj,
    const __nv_bfloat16 *conv_w, const __nv_bfloat16 *a_log,
    const __nv_bfloat16 *dt_bias, const __nv_bfloat16 *norm_w,
    float *state, float *conv_buf, __nv_bfloat16 *output, int S)
{
    int h = blockIdx.x; if (h >= DN_HEADS) return;
    int tid = threadIdx.x, wid = tid/32, lid = tid%32;
    constexpr int NWARPS = BLOCK_THREADS / 32;
    static_assert(BLOCK_THREADS % 32 == 0, "DeltaNet prefill block size must be warp-aligned");
    static_assert(DN_VAL % NWARPS == 0, "DeltaNet value width must divide the warp count");
    static_assert(DN_KEY == DN_VAL, "DeltaNet prefill kernel assumes equal key/value widths");
    constexpr float Q_SCALE = 1.0f / 11.313708498984761f;
    constexpr int HEAD_CHANNELS = DN_KEY + DN_KEY + DN_VAL;

    float a_log_val = __bfloat162float(a_log[h]);
    float a_scale = __expf(a_log_val);
    float dt_b = __bfloat162float(dt_bias[h]);

    __shared__ float s_q[DN_KEY], s_k[DN_KEY], s_v[DN_VAL], s_out[DN_VAL];
    __shared__ float s_beta, s_decay;
    __shared__ float s_qnorm[NWARPS], s_knorm[NWARPS], s_gnorm[NWARPS];
    __shared__ float s_conv[HEAD_CHANNELS * DN_CONV_K];
    __shared__ float s_conv_w[HEAD_CHANNELS * DN_CONV_K];

    float *my_state = state + h * DN_KEY * DN_VAL;
    const int q_base = h * DN_KEY;
    const int k_base = DN_QK_SIZE + h * DN_KEY;
    const int v_base = 2 * DN_QK_SIZE + h * DN_VAL;

    // Stage this head's conv ring buffer and weights in shared memory.
    for (int idx = tid; idx < HEAD_CHANNELS * DN_CONV_K; idx += BLOCK_THREADS) {
        int channel = idx / DN_CONV_K;
        int tap = idx % DN_CONV_K;
        int global_ch = (channel < DN_KEY)
            ? (q_base + channel)
            : ((channel < 2 * DN_KEY)
                ? (k_base + channel - DN_KEY)
                : (v_base + channel - 2 * DN_KEY));
        s_conv[idx] = conv_buf[global_ch * DN_CONV_K + tap];
        s_conv_w[idx] = __bfloat162float(conv_w[global_ch * DN_CONV_K + tap]);
    }
    __syncthreads();

    // Load state into registers
    constexpr int CPW = DN_VAL / NWARPS;  // 8
    constexpr int RPL = DN_KEY / 32;       // 4
    float sreg[CPW * RPL];  // 32 floats

    for (int jj = 0; jj < CPW; jj++) {
        int j = wid * CPW + jj;
        for (int ii = 0; ii < RPL; ii++)
            sreg[jj*RPL+ii] = my_state[j*DN_KEY + lid+ii*32];
    }

    for (int t = 0; t < S; t++) {
        const __nv_bfloat16 *qkv_t = qkv_proj + t * DN_CONV_CH;
        const __nv_bfloat16 *z_h = z_proj + t * DN_V_SIZE + h * DN_VAL;
        const float beta_proj_val = beta_proj[t * DN_HEADS + h];
        const float alpha_proj_val = alpha_proj[t * DN_HEADS + h];

        // Conv1d + SiLU for q, k, v with head-local ring buffers staged in shared memory.
        for (int c = tid; c < HEAD_CHANNELS; c += BLOCK_THREADS) {
            int base = c * DN_CONV_K;
            float h0 = s_conv[base + 1];
            float h1 = s_conv[base + 2];
            float h2 = s_conv[base + 3];
            s_conv[base + 0] = h0;
            s_conv[base + 1] = h1;
            s_conv[base + 2] = h2;

            float in_val;
            if (c < DN_KEY) {
                in_val = __bfloat162float(qkv_t[q_base + c]);
            } else if (c < 2 * DN_KEY) {
                in_val = __bfloat162float(qkv_t[k_base + c - DN_KEY]);
            } else {
                in_val = __bfloat162float(qkv_t[v_base + c - 2 * DN_KEY]);
            }
            s_conv[base + 3] = in_val;

            float co = 0.0f;
#pragma unroll
            for (int k = 0; k < DN_CONV_K; k++) {
                co += s_conv[base + k] * s_conv_w[base + k];
            }
            float act = pf_silu(co);
            if (c < DN_KEY) {
                s_q[c] = act;
            } else if (c < 2 * DN_KEY) {
                s_k[c - DN_KEY] = act;
            } else {
                s_v[c - 2 * DN_KEY] = act;
            }
        }
        __syncthreads();

        // L2 normalize
        float q_sq = 0.0f;
        float k_sq = 0.0f;
        for (int i = tid; i < DN_KEY; i += BLOCK_THREADS) {
            q_sq += s_q[i] * s_q[i];
            k_sq += s_k[i] * s_k[i];
        }
        q_sq = pf_warp_sum(q_sq);
        k_sq = pf_warp_sum(k_sq);
        if (lid == 0) {
            s_qnorm[wid] = q_sq;
            s_knorm[wid] = k_sq;
        }
        __syncthreads();
        if (wid == 0) {
            float q_total = (lid < NWARPS) ? s_qnorm[lid] : 0.0f;
            float k_total = (lid < NWARPS) ? s_knorm[lid] : 0.0f;
            q_total = pf_warp_sum(q_total);
            k_total = pf_warp_sum(k_total);
            if (lid == 0) {
                s_qnorm[0] = rsqrtf(q_total + 1e-6f) * Q_SCALE;
                s_knorm[0] = rsqrtf(k_total + 1e-6f);
            }
        }
        __syncthreads();
        float q_norm = s_qnorm[0];
        float k_norm = s_knorm[0];
        for (int i = tid; i < DN_KEY; i += BLOCK_THREADS) {
            s_q[i] *= q_norm;
            s_k[i] *= k_norm;
        }
        __syncthreads();

        if (tid == 0) {
            s_beta = pf_sigmoid(beta_proj_val);
            float x = alpha_proj_val + dt_b;
            float sp = (x > 20.0f) ? x : log1pf(__expf(x));
            s_decay = __expf(-a_scale * sp);
        }
        __syncthreads();
        float beta = s_beta, decay = s_decay;
        __nv_bfloat16 *out_h = output + t * DN_V_SIZE + h * DN_VAL;

        // State-in-registers recurrence
        for (int jj = 0; jj < CPW; jj++) {
            int j = wid * CPW + jj;
            float kv = 0.0f;
            for (int ii = 0; ii < RPL; ii++) {
                kv += sreg[jj * RPL + ii] * s_k[lid + ii * 32];
            }
            kv = pf_warp_sum(kv);
            kv = __shfl_sync(0xffffffff, kv, 0);
            float delta = (s_v[j] - decay * kv) * beta;
            float attn = 0.0f;
            for (int ii = 0; ii < RPL; ii++) {
                int key_idx = lid + ii * 32;
                float new_state = decay * sreg[jj * RPL + ii] + s_k[key_idx] * delta;
                sreg[jj * RPL + ii] = new_state;
                attn += new_state * s_q[key_idx];
            }
            attn = pf_warp_sum(attn);
            if (lid == 0) s_out[j] = attn;
        }
        __syncthreads();

        // Gated RMSNorm → bf16 output, keeping the pre-norm output in shared memory.
        float sq2 = 0.0f;
        for (int i = tid; i < DN_VAL; i += BLOCK_THREADS) {
            float v = s_out[i];
            sq2 += v * v;
        }
        sq2 = pf_warp_sum(sq2);
        if (lid == 0) s_gnorm[wid] = sq2;
        __syncthreads();
        if (wid == 0) {
            float v = (lid < NWARPS) ? s_gnorm[lid] : 0.0f;
            v = pf_warp_sum(v);
            if (lid == 0) s_gnorm[0] = rsqrtf(v / DN_VAL + RMS_EPS);
        }
        __syncthreads();
        float rstd = s_gnorm[0];
        for (int i = tid; i < DN_VAL; i += BLOCK_THREADS) {
            float n = s_out[i] * rstd * __bfloat162float(norm_w[i]);
            out_h[i] = __float2bfloat16(n * pf_silu(__bfloat162float(z_h[i])));
        }
    }

    // Spill the updated conv ring buffer back once per prompt.
    for (int idx = tid; idx < HEAD_CHANNELS * DN_CONV_K; idx += BLOCK_THREADS) {
        int channel = idx / DN_CONV_K;
        int tap = idx % DN_CONV_K;
        int global_ch = (channel < DN_KEY)
            ? (q_base + channel)
            : ((channel < 2 * DN_KEY)
                ? (k_base + channel - DN_KEY)
                : (v_base + channel - 2 * DN_KEY));
        conv_buf[global_ch * DN_CONV_K + tap] = s_conv[idx];
    }

    // Write state back
    for (int jj = 0; jj < CPW; jj++) {
        int j = wid * CPW + jj;
        for (int ii = 0; ii < RPL; ii++)
            my_state[j*DN_KEY + lid+ii*32] = sreg[jj*RPL+ii];
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
                float angle = float(pos) * PF_ROPE_INV_FREQ[i & (FA_ROT_PAIRS - 1)];
                float sv, cv;
                __sincosf(angle, &sv, &cv);
                int p=(i<FA_ROT_DIM/2)?i+FA_ROT_DIM/2:i-FA_ROT_DIM/2;
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
                float angle = float(pos) * PF_ROPE_INV_FREQ[i & (FA_ROT_PAIRS - 1)];
                float sv, cv;
                __sincosf(angle, &sv, &cv);
                int p=(i<FA_ROT_DIM/2)?i+FA_ROT_DIM/2:i-FA_ROT_DIM/2;
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
        float om=mx; mx=fmaxf(mx,sc); float ed=__expf(om-mx); float wexp=__expf(sc-mx); se=se*ed+wexp;
        float wt=wexp; for(int e=0;e<EPL;e++) oa[e]=oa[e]*ed+wt*__bfloat162float(vv[lid*EPL+e]);
    }
    float rs=1.f/se;
    for(int e=0;e<EPL;e++){int i=lid*EPL+e;float g=pf_sigmoid(__bfloat162float(gv[i]));ov[i]=__float2bfloat16(oa[e]*rs*g);}
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

// ===== Main orchestrator =====
static void launch_prefill_bf16_impl(
    const int *token_ids, int seq_len, int *output_token,
    const __nv_bfloat16 *embed_weight, const PFLayerWeights *layers,
    const __nv_bfloat16 *final_norm_w, const __nv_bfloat16 *lm_head_w,
    const void *lm_head_weight_packed, const void *lm_head_scales,
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
    __nv_bfloat16 *lm_hidden_bf16, uint8_t *lm_hidden_packed,
    uint8_t *lm_hidden_scales, __half *lm_logits_f16,
    cudaStream_t stream)
{
    static cublasHandle_t cublas = nullptr;
    if (!cublas) cublasCreate(&cublas);
    cublasSetStream(cublas, stream);

    static PFLayerWeights hl[NUM_LAYERS];
    static bool copied = false;
    if (!copied) { cudaMemcpy(hl, layers, NUM_LAYERS*sizeof(PFLayerWeights), cudaMemcpyDeviceToHost); copied = true; }

    int S = seq_len;
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

            cublas_bf16_gemm(cublas, normalized, qkv_w, proj_buf, S, DN_CONV_CH, HIDDEN);
            cublas_bf16_gemm(cublas, normalized, z_w, proj_buf2, S, DN_V_SIZE, HIDDEN);
            pf_bf16_matvec<<<S*DN_HEADS, 32, 0, stream>>>(normalized, beta_w, beta_buf, S, HIDDEN, DN_HEADS);
            pf_bf16_matvec<<<S*DN_HEADS, 32, 0, stream>>>(normalized, alpha_w, alpha_buf, S, HIDDEN, DN_HEADS);

            // Standalone recurrence
            pf_deltanet_recurrence<PREFILL_DN_BLOCK_SIZE><<<DN_HEADS, PREFILL_DN_BLOCK_SIZE, 0, stream>>>(
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

    bool used_nvfp4_lm = false;
    if (lm_head_weight_packed && lm_head_scales && lm_hidden_bf16 && lm_hidden_packed &&
        lm_hidden_scales && lm_logits_f16) {
        used_nvfp4_lm = launch_lm_head_cublaslt_bf16_top1(
            final_normed,
            lm_head_weight_packed,
            lm_head_scales,
            lm_hidden_bf16,
            lm_hidden_packed,
            lm_hidden_scales,
            lm_logits_f16,
            lm_bmv,
            lm_bmi,
            output_token,
            stream);
    }
    if (!used_nvfp4_lm) {
        int lm_blocks = 512;
        pf_lm_head<<<lm_blocks, 256, 0, stream>>>(final_normed, lm_head_w, lm_bmv, lm_bmi, VOCAB);
        pf_lm_reduce<<<1, 256, 0, stream>>>(lm_bmv, lm_bmi, output_token, lm_blocks);
    }
}

extern "C" void launch_prefill_bf16(
    const int *token_ids, int seq_len, int *output_token,
    const void *embed_weight, const PFLayerWeights *layers,
    const void *final_norm_w, const void *lm_head_w,
    void *fa_k_cache, void *fa_v_cache, void *dn_states, void *conv_bufs,
    void *hidden, void *residual, void *normalized,
    void *proj_buf, void *proj_buf2, void *attn_buf, void *mlp_buf,
    void *dn_out_buf, void *beta_buf, void *alpha_buf,
    void *final_normed, void *hidden_bf16_out,
    void *lm_bmv, void *lm_bmi,
    cudaStream_t stream)
{
    launch_prefill_bf16_impl(
        token_ids, seq_len, output_token,
        (const __nv_bfloat16 *)embed_weight, layers,
        (const __nv_bfloat16 *)final_norm_w, (const __nv_bfloat16 *)lm_head_w,
        nullptr, nullptr,
        (__nv_bfloat16 *)fa_k_cache, (__nv_bfloat16 *)fa_v_cache,
        (float *)dn_states, (float *)conv_bufs,
        (__nv_bfloat16 *)hidden, (__nv_bfloat16 *)residual, (__nv_bfloat16 *)normalized,
        (__nv_bfloat16 *)proj_buf, (__nv_bfloat16 *)proj_buf2,
        (__nv_bfloat16 *)attn_buf, (__nv_bfloat16 *)mlp_buf,
        (__nv_bfloat16 *)dn_out_buf,
        (float *)beta_buf, (float *)alpha_buf,
        (__nv_bfloat16 *)final_normed, (__nv_bfloat16 *)hidden_bf16_out,
        (float *)lm_bmv, (int *)lm_bmi,
        nullptr, nullptr, nullptr, nullptr,
        stream);
}

extern "C" void launch_prefill_bf16_nvfp4_lm(
    const int *token_ids, int seq_len, int *output_token,
    const void *embed_weight, const PFLayerWeights *layers,
    const void *final_norm_w, const void *lm_head_w,
    const void *lm_head_weight_packed, const void *lm_head_scales,
    void *fa_k_cache, void *fa_v_cache, void *dn_states, void *conv_bufs,
    void *hidden, void *residual, void *normalized,
    void *proj_buf, void *proj_buf2, void *attn_buf, void *mlp_buf,
    void *dn_out_buf, void *beta_buf, void *alpha_buf,
    void *final_normed, void *hidden_bf16_out,
    void *lm_bmv, void *lm_bmi,
    void *lm_hidden_bf16, void *lm_hidden_packed,
    void *lm_hidden_scales, void *lm_logits_f16,
    cudaStream_t stream)
{
    launch_prefill_bf16_impl(
        token_ids, seq_len, output_token,
        (const __nv_bfloat16 *)embed_weight, layers,
        (const __nv_bfloat16 *)final_norm_w, (const __nv_bfloat16 *)lm_head_w,
        lm_head_weight_packed, lm_head_scales,
        (__nv_bfloat16 *)fa_k_cache, (__nv_bfloat16 *)fa_v_cache,
        (float *)dn_states, (float *)conv_bufs,
        (__nv_bfloat16 *)hidden, (__nv_bfloat16 *)residual, (__nv_bfloat16 *)normalized,
        (__nv_bfloat16 *)proj_buf, (__nv_bfloat16 *)proj_buf2,
        (__nv_bfloat16 *)attn_buf, (__nv_bfloat16 *)mlp_buf,
        (__nv_bfloat16 *)dn_out_buf,
        (float *)beta_buf, (float *)alpha_buf,
        (__nv_bfloat16 *)final_normed, (__nv_bfloat16 *)hidden_bf16_out,
        (float *)lm_bmv, (int *)lm_bmi,
        (__nv_bfloat16 *)lm_hidden_bf16, (uint8_t *)lm_hidden_packed,
        (uint8_t *)lm_hidden_scales, (__half *)lm_logits_f16,
        stream);
}
