/**
 * Full-attention layer (Gated Attention with QK-norm + RoPE) templated on Cfg.
 *
 * Ported from `models/qwen35_0p8b/kernel.cu:full_attention_layer`,
 * generalized for:
 *   - Cfg-parameterized dims (Cfg_0p8B: 8/2/256, Cfg_27B: 24/4/256)
 *   - YaRN + MRoPE rotary (for 27B's 262k+ context)
 *
 * Required scratch buffers (caller-provided):
 *   g_residual[HIDDEN]      bf16  -- captured input for residual add
 *   g_q[FA_QPROJ_SIZE]      fp32  -- Q + gate post-proj
 *   g_kv[FA_KV_SIZE * 2]    fp32  -- K then V post-proj
 *   g_attn_out[FA_Q_SIZE]   fp32  -- attention output before O proj
 *   g_fa_partials           fp32  -- [num_splits * Q_HEADS * (HEAD_DIM+2)]
 *   shmem[HIDDEN]           bf16  -- per-block s_norm
 *
 * Caller also passes the FA layer weights (struct templated on Cfg) and
 * the persistent K/V cache (BF16 or NVFP4-packed, see kv_cache.cuh).
 */
#pragma once

#include "Cfg.cuh"
#include "helpers.cuh"
#include "matvec.cuh"
#include "rmsnorm.cuh"
#include "rope.cuh"

namespace lucebox::qwen3x {

template<typename Cfg>
struct FullAttnWeights {
    const __nv_bfloat16 *input_layernorm_weight;   // [HIDDEN]
    const __nv_bfloat16 *q_proj_weight;            // [FA_QPROJ_SIZE, HIDDEN]
    const __nv_bfloat16 *k_proj_weight;            // [FA_KV_SIZE, HIDDEN]
    const __nv_bfloat16 *v_proj_weight;            // [FA_KV_SIZE, HIDDEN]
    const __nv_bfloat16 *q_norm_weight;            // [FA_HEAD_DIM]
    const __nv_bfloat16 *k_norm_weight;            // [FA_HEAD_DIM]
    const __nv_bfloat16 *o_proj_weight;            // [HIDDEN, FA_Q_SIZE]
    const __nv_bfloat16 *post_attn_layernorm_weight; // [HIDDEN]
    const __nv_bfloat16 *gate_proj_weight;         // [INTER, HIDDEN]
    const __nv_bfloat16 *up_proj_weight;           // [INTER, HIDDEN]
    const __nv_bfloat16 *down_proj_weight;         // [HIDDEN, INTER]
};

// NVFP4 variant: norms stay BF16 (small, frequently accessed), projections
// switch to packed FP4 + FP16 scales. Memory drops ~3.5×.
template<typename Cfg>
struct FullAttnWeightsNVFP4 {
    const __nv_bfloat16 *input_layernorm_weight;
    PackedMatrixNVFP4    q_proj;
    PackedMatrixNVFP4    k_proj;
    PackedMatrixNVFP4    v_proj;
    const __nv_bfloat16 *q_norm_weight;
    const __nv_bfloat16 *k_norm_weight;
    PackedMatrixNVFP4    o_proj;
    const __nv_bfloat16 *post_attn_layernorm_weight;
    PackedMatrixNVFP4    gate_proj;
    PackedMatrixNVFP4    up_proj;
    PackedMatrixNVFP4    down_proj;
};

// One-head RMSNorm + RoPE. lane-cooperative within a warp.
template<typename Cfg>
__device__ void head_norm_rope(
    float *__restrict__ head,                 // [FA_HEAD_DIM] fp32 (per-head Q or K)
    const __nv_bfloat16 *__restrict__ norm_w, // [FA_HEAD_DIM]
    int pos_t, int pos_h, int pos_w,
    const float *__restrict__ inv_freq,
    const YarnParams &yp,
    const MRopeSections &sections,
    int lane_id)
{
    constexpr int D    = Cfg::FA_HEAD_DIM;
    constexpr int R    = Cfg::FA_ROTARY_DIM;
    constexpr float EPS = 1e-6f;

    // Per-head RMSNorm.
    float ss = 0.0f;
    for (int i = lane_id; i < D; i += WARP_SIZE) ss += head[i] * head[i];
    ss = warp_reduce_sum_x(ss);
    float sc = rsqrtf(ss / float(D) + EPS);
    sc = __shfl_sync(0xffffffff, sc, 0);
    for (int i = lane_id; i < D; i += WARP_SIZE) {
        float w = __bfloat162float(__ldg(norm_w + i));
        head[i] = head[i] * sc * (1.0f + w);
    }

    // RoPE on the first R dims; rest pass through.
    rope_apply<Cfg>(head, pos_t, pos_h, pos_w, inv_freq, yp, sections, lane_id);
}

// Full-attention layer forward (single token; KV cache is grown by one
// position). Caller is responsible for grid syncing between sections.
template<typename Cfg>
__device__ void full_attention_layer(
    AtomicGridSync &grid,
    const FullAttnWeights<Cfg> &w,
    const __nv_bfloat16 *__restrict__ input,   // [HIDDEN]
    __nv_bfloat16 *__restrict__ k_cache,        // [FA_KV_HEADS, MAX_SEQ, FA_HEAD_DIM]
    __nv_bfloat16 *__restrict__ v_cache,        // same shape
    __nv_bfloat16 *__restrict__ g_residual,     // [HIDDEN]
    float *__restrict__ g_q,                    // [FA_QPROJ_SIZE]
    float *__restrict__ g_kv,                   // [FA_KV_SIZE * 2]
    float *__restrict__ g_attn_out,             // [FA_Q_SIZE]
    float *__restrict__ g_fa_partials,          // [num_splits * Q_HEADS * (HEAD_DIM+2)]
    float *__restrict__ g_rope_inv_freq,        // [FA_ROTARY_DIM/2]
    const YarnParams &yp,
    const MRopeSections &sections,
    int position, int max_seq_len,
    int pos_h, int pos_w,                       // 0 for text-only
    __nv_bfloat16 *__restrict__ shmem,
    __nv_bfloat16 *__restrict__ hidden_out)
{
    constexpr int H      = Cfg::HIDDEN;
    constexpr int Q_H    = Cfg::FA_NUM_Q_HEADS;
    constexpr int KV_H   = Cfg::FA_NUM_KV_HEADS;
    constexpr int D      = Cfg::FA_HEAD_DIM;
    constexpr int GQA    = Cfg::FA_GQA_RATIO;
    constexpr int Q_SIZE = Cfg::FA_Q_SIZE;
    constexpr int QPROJ  = Cfg::FA_QPROJ_SIZE;     // Q + gate
    constexpr int KV_SIZE = Cfg::FA_KV_SIZE;

    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    __nv_bfloat16 *s_norm = shmem;

    // 1. Pre-attn RMSNorm + capture input as residual.
    rmsnorm_capture<Cfg>(input, w.input_layernorm_weight, s_norm, g_residual);

    // 2. Q + gate, K, V projections.
    matvec_bf16<Cfg>(s_norm, w.q_proj_weight, g_q, H, QPROJ,    num_blocks);
    matvec_bf16<Cfg>(s_norm, w.k_proj_weight, g_kv, H, KV_SIZE,  num_blocks);
    matvec_bf16<Cfg>(s_norm, w.v_proj_weight, g_kv + KV_SIZE,
                      H, KV_SIZE, num_blocks);
    grid.sync();

    // 3a. K-norm + RoPE + write to k_cache. Block 0 only; each warp handles
    //     one KV head.
    if (block_id == 0) {
        float *k_buf = g_kv;
        float *v_buf = g_kv + KV_SIZE;
        for (int h = warp_id; h < KV_H; h += NUM_WARPS) {
            float *kh = k_buf + h * D;
            head_norm_rope<Cfg>(kh, w.k_norm_weight, position, pos_h, pos_w,
                                g_rope_inv_freq, yp, sections, lane_id);
            __nv_bfloat16 *kc = k_cache + (size_t)h * max_seq_len * D
                                 + (size_t)position * D;
            __nv_bfloat16 *vc = v_cache + (size_t)h * max_seq_len * D
                                 + (size_t)position * D;
            for (int i = lane_id; i < D; i += WARP_SIZE) {
                kc[i] = __float2bfloat16(kh[i]);
                vc[i] = __float2bfloat16(v_buf[h * D + i]);
            }
        }
    }

    // 3b. Q-norm + RoPE (split across blocks; one query head per warp slot).
    // Q-proj output layout: per-head [Q_h, gate_h] interleaved with stride 2*D.
    // Split layout was tested and made things worse (cos=0.948 at layer 3 vs
    // 0.997 with interleaved), so interleaved is correct.
    int hpb = (Q_H + num_blocks - 1) / num_blocks;
    int hs = block_id * hpb;
    int he = min(hs + hpb, Q_H);
    for (int qh = hs; qh < he; ++qh) {
        if (warp_id == 0) {
            float *qhp = g_q + qh * D * 2;
            head_norm_rope<Cfg>(qhp, w.q_norm_weight, position, pos_h, pos_w,
                                g_rope_inv_freq, yp, sections, lane_id);
        }
    }
    grid.sync();

    // 4. Split-K online-softmax attention. Each (qh, split) -> one block.
    {
        int cache_len = position + 1;
        float attn_scale = 1.0f / sqrtf(float(D));
        constexpr int EPL = D / WARP_SIZE;
        constexpr int PARTIAL_STRIDE = D + 2;

        int num_splits = num_blocks / Q_H;
        if (num_splits < 1) num_splits = 1;
        int my_qh    = block_id % Q_H;
        int my_split = block_id / Q_H;
        bool active = (block_id < Q_H * num_splits);

        __shared__ float s_max[NUM_WARPS];
        __shared__ float s_sum[NUM_WARPS];
        __shared__ float s_out[NUM_WARPS * Cfg::FA_HEAD_DIM];

        if (active) {
            int per_split = (cache_len + num_splits - 1) / num_splits;
            int t_start = my_split * per_split;
            int t_end   = min(t_start + per_split, cache_len);
            int kvh = my_qh / GQA;
            const float *qh = g_q + my_qh * D * 2;     // interleaved: Q first half of pair

            float q_local[EPL];
            #pragma unroll
            for (int e = 0; e < EPL; ++e) q_local[e] = qh[lane_id * EPL + e];

            float partial_max = -INFINITY, partial_sum = 0.0f;
            float partial_acc[EPL];
            #pragma unroll
            for (int e = 0; e < EPL; ++e) partial_acc[e] = 0.0f;

            for (int t = t_start + warp_id; t < t_end; t += NUM_WARPS) {
                const __nv_bfloat16 *k_p = k_cache + (size_t)kvh * max_seq_len * D
                                            + (size_t)t * D;
                const __nv_bfloat16 *v_p = v_cache + (size_t)kvh * max_seq_len * D
                                            + (size_t)t * D;
                float score = 0.0f;
                #pragma unroll
                for (int e = 0; e < EPL; ++e)
                    score += q_local[e] * __bfloat162float(__ldg(k_p + lane_id * EPL + e));
                score = warp_reduce_sum_x(score) * attn_scale;
                score = __shfl_sync(0xffffffff, score, 0);

                float old_max = partial_max;
                partial_max = fmaxf(partial_max, score);
                float exp_diff = fast_exp(old_max - partial_max);
                partial_sum = partial_sum * exp_diff + fast_exp(score - partial_max);
                float wt = fast_exp(score - partial_max);
                #pragma unroll
                for (int e = 0; e < EPL; ++e)
                    partial_acc[e] = partial_acc[e] * exp_diff
                                     + wt * __bfloat162float(__ldg(v_p + lane_id * EPL + e));
            }
            if (lane_id == 0) { s_max[warp_id] = partial_max; s_sum[warp_id] = partial_sum; }
            #pragma unroll
            for (int e = 0; e < EPL; ++e)
                s_out[warp_id * D + lane_id * EPL + e] = partial_acc[e];
            __syncthreads();

            // Warp 0 reduces across warps inside this block.
            if (warp_id == 0) {
                float bm = -INFINITY;
                for (int w_ = 0; w_ < NUM_WARPS; ++w_) bm = fmaxf(bm, s_max[w_]);
                float bs = 0.0f;
                float bo[EPL];
                #pragma unroll
                for (int e = 0; e < EPL; ++e) bo[e] = 0.0f;
                for (int w_ = 0; w_ < NUM_WARPS; ++w_) {
                    if (s_max[w_] > -INFINITY) {
                        float sc = fast_exp(s_max[w_] - bm);
                        bs += s_sum[w_] * sc;
                        #pragma unroll
                        for (int e = 0; e < EPL; ++e)
                            bo[e] += s_out[w_ * D + lane_id * EPL + e] * sc;
                    }
                }
                int slot = (my_qh * num_splits + my_split) * PARTIAL_STRIDE;
                #pragma unroll
                for (int e = 0; e < EPL; ++e) g_fa_partials[slot + lane_id * EPL + e] = bo[e];
                if (lane_id == 0) {
                    g_fa_partials[slot + D]     = bm;
                    g_fa_partials[slot + D + 1] = bs;
                }
            }
        }
    }
    grid.sync();

    // 5. Reduce per-(qh) splits to final attn output, apply gate.
    {
        int num_splits = num_blocks / Q_H;
        if (num_splits < 1) num_splits = 1;
        constexpr int EPL = D / WARP_SIZE;
        constexpr int PARTIAL_STRIDE = D + 2;

        if (block_id < Q_H) {
            int qh = block_id;
            const float *qhp = g_q + qh * D * 2;
            const float *gate = qhp + D;
            if (warp_id == 0) {
                float gm = -INFINITY;
                for (int s = 0; s < num_splits; ++s)
                    gm = fmaxf(gm, g_fa_partials[(qh * num_splits + s) * PARTIAL_STRIDE + D]);
                float gs = 0.0f;
                float go[EPL];
                #pragma unroll
                for (int e = 0; e < EPL; ++e) go[e] = 0.0f;
                for (int s = 0; s < num_splits; ++s) {
                    float m = g_fa_partials[(qh * num_splits + s) * PARTIAL_STRIDE + D];
                    if (m == -INFINITY) continue;
                    float ss = g_fa_partials[(qh * num_splits + s) * PARTIAL_STRIDE + D + 1];
                    float ww = fast_exp(m - gm);
                    gs += ss * ww;
                    #pragma unroll
                    for (int e = 0; e < EPL; ++e)
                        go[e] += g_fa_partials[(qh * num_splits + s) * PARTIAL_STRIDE
                                                + lane_id * EPL + e] * ww;
                }
                float rcp = (gs > 0.0f) ? (1.0f / gs) : 0.0f;
                #pragma unroll
                for (int e = 0; e < EPL; ++e) {
                    int idx = lane_id * EPL + e;
                    // Output gate. C3 experiment showed sigmoid is closer to
                    // HF than silu (silu made cos=0.81 at layer 3 vs 0.997
                    // with sigmoid). Despite the config saying
                    // output_gate_type="swish", HF's runtime must be applying
                    // plain sigmoid here. TODO: confirm in HF source.
                    g_attn_out[qh * D + idx] = go[e] * rcp * fast_sigmoid(gate[idx]);
                }
            }
        }
    }
    grid.sync();

    // 6. O projection + residual.
    matvec_o_residual<Cfg>(g_attn_out, w.o_proj_weight, g_residual, hidden_out,
                            Q_SIZE, H, num_blocks);
    grid.sync();

    // 7. Post-attn RMSnorm + MLP + residual. Every layer (FA or DN) has the
    //    same post-attention path: norm -> gate/up SwiGLU -> down -> +residual.
    //    Missing this for FA was the root cause of the ~0.003 cos drift per
    //    FA pass we tracked down in C3.
    __nv_bfloat16 *s_act = shmem;
    rmsnorm_capture<Cfg>(hidden_out, w.post_attn_layernorm_weight, s_act, g_residual);

    constexpr int INTER = Cfg::INTERMEDIATE;
    // Reuse g_attn_out as the f32 [INTER] mlp_inter scratch — the gated
    // attention output is no longer needed after step 6.
    float *g_mlp_inter = g_attn_out;
    matvec_gate_up_silu<Cfg>(s_act, w.gate_proj_weight, w.up_proj_weight,
                              g_mlp_inter, H, INTER, num_blocks);
    grid.sync();

    {
        float *s_mlp = reinterpret_cast<float *>(shmem);
        for (int i = threadIdx.x; i < INTER; i += BLOCK_SIZE) s_mlp[i] = g_mlp_inter[i];
        __syncthreads();
        matvec_down_residual<Cfg>(s_mlp, w.down_proj_weight, g_residual, hidden_out,
                                    INTER, H, num_blocks);
    }
    grid.sync();
}

}  // namespace lucebox::qwen3x
