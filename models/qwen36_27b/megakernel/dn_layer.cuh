/**
 * Gated DeltaNet layer templated on Cfg — with V/QK split (new for 27B).
 *
 * 0.8B  : DN_NUM_QK_HEADS == DN_NUM_V_HEADS == 16 (unified, V_PER_QK = 1)
 * 27B   : DN_NUM_QK_HEADS = 16, DN_NUM_V_HEADS = 48 (V_PER_QK = 3)
 *
 * Each V head has its own recurrent state of shape [DN_VALUE_DIM, DN_KEY_DIM].
 * The Q, K projections are shared across V_PER_QK V heads that map to the
 * same QK head (GQA-style).
 *
 * Weight layouts (consumed by the layer):
 *   qkv_proj         [DN_CONV_CH, HIDDEN] bf16
 *                       DN_CONV_CH = 2 * DN_QK_SIZE + DN_V_SIZE
 *                       layout: [Q, K, V] concatenated along output dim
 *   z_proj           [DN_V_SIZE, HIDDEN] bf16              (output gate)
 *   beta_proj        [DN_NUM_V_HEADS, HIDDEN] bf16         (per-V scalar)
 *   alpha_proj       [DN_NUM_V_HEADS, HIDDEN] bf16         (per-V scalar)
 *   conv1d_weight    [DN_CONV_CH, DN_CONV_KERNEL] bf16
 *   a_log            [DN_NUM_V_HEADS]                       (recurrent decay)
 *   dt_bias          [DN_NUM_V_HEADS]
 *   norm_weight      [DN_VALUE_DIM]                         (final group-RMSNorm)
 *   out_proj         [HIDDEN, DN_V_SIZE]
 *   post_attn_layernorm_weight + MLP weights as usual
 *
 * Recurrence per (v_head, value_dim_j):
 *   state[v, j, :] holds previous K vector contribution for output dim j
 *   stk = state[v, j, :] dot k[qk_head]
 *   sqv = state[v, j, :] dot q[qk_head]
 *   error_j = (v[v_head, j] - stk) * beta[v_head]
 *   o[v_head, j] = decay[v_head] * sqv + error_j * (k . q)
 *   state[v, j, :] = state[v, j, :] * decay[v_head] + k[qk_head] * error_j
 *
 * Followed by group-RMSNorm and SiLU(z)-gate.
 */
#pragma once

#include "Cfg.cuh"
#include "helpers.cuh"
#include "matvec.cuh"
#include "rmsnorm.cuh"

namespace lucebox::qwen3x {

template<typename Cfg>
struct DeltaNetWeights {
    const __nv_bfloat16 *input_layernorm_weight;     // [HIDDEN]
    const __nv_bfloat16 *qkv_proj_weight;            // [DN_CONV_CH, HIDDEN]
    const __nv_bfloat16 *z_proj_weight;              // [DN_V_SIZE, HIDDEN]
    const __nv_bfloat16 *beta_proj_weight;           // [DN_NUM_V_HEADS, HIDDEN]
    const __nv_bfloat16 *alpha_proj_weight;          // [DN_NUM_V_HEADS, HIDDEN]
    const __nv_bfloat16 *conv1d_weight;              // [DN_CONV_CH, DN_CONV_KERNEL]
    const __nv_bfloat16 *a_log;                      // [DN_NUM_V_HEADS]
    const __nv_bfloat16 *dt_bias;                    // [DN_NUM_V_HEADS]
    const __nv_bfloat16 *norm_weight;                // [DN_VALUE_DIM]
    const __nv_bfloat16 *out_proj_weight;            // [HIDDEN, DN_V_SIZE]
    const __nv_bfloat16 *post_attn_layernorm_weight; // [HIDDEN]
    const __nv_bfloat16 *gate_proj_weight;           // [INTER, HIDDEN]
    const __nv_bfloat16 *up_proj_weight;             // [INTER, HIDDEN]
    const __nv_bfloat16 *down_proj_weight;           // [HIDDEN, INTER]
};

// NVFP4 variant: norms, conv1d, a_log, dt_bias stay BF16. Linear projections
// move to packed FP4 + FP16 scales.
template<typename Cfg>
struct DeltaNetWeightsNVFP4 {
    const __nv_bfloat16 *input_layernorm_weight;
    PackedMatrixNVFP4    qkv_proj;
    PackedMatrixNVFP4    z_proj;
    PackedMatrixNVFP4    beta_proj;
    PackedMatrixNVFP4    alpha_proj;
    const __nv_bfloat16 *conv1d_weight;      // small, stays bf16
    const __nv_bfloat16 *a_log;
    const __nv_bfloat16 *dt_bias;
    const __nv_bfloat16 *norm_weight;
    PackedMatrixNVFP4    out_proj;
    const __nv_bfloat16 *post_attn_layernorm_weight;
    PackedMatrixNVFP4    gate_proj;
    PackedMatrixNVFP4    up_proj;
    PackedMatrixNVFP4    down_proj;
};

template<typename Cfg>
__device__ void delta_net_layer(
    AtomicGridSync &grid,
    const DeltaNetWeights<Cfg> &w,
    const __nv_bfloat16 *__restrict__ input,
    __nv_bfloat16 *__restrict__ g_residual,
    float *__restrict__ g_qkv,           // [DN_CONV_CH]
    float *__restrict__ g_z,             // [DN_V_SIZE]
    float *__restrict__ g_beta,          // [DN_NUM_V_HEADS]
    float *__restrict__ g_alpha,         // [DN_NUM_V_HEADS]
    float *__restrict__ g_dn_out,        // [DN_V_SIZE]
    float *__restrict__ g_mlp_inter,     // [INTER]
    float *__restrict__ dn_state,        // [DN_NUM_V_HEADS, DN_VAL, DN_KEY] persistent
    float *__restrict__ conv_buf,        // [DN_CONV_CH, DN_CONV_KERNEL] persistent
    __nv_bfloat16 *__restrict__ hidden_out,
    __nv_bfloat16 *__restrict__ shmem)
{
    constexpr int H        = Cfg::HIDDEN;
    constexpr int INTER    = Cfg::INTERMEDIATE;
    constexpr int V_HEADS  = Cfg::DN_NUM_V_HEADS;
    constexpr int QK_HEADS = Cfg::DN_NUM_QK_HEADS;
    constexpr int V_PER_QK = Cfg::DN_V_PER_QK;
    constexpr int KEY      = Cfg::DN_KEY_DIM;
    constexpr int VAL      = Cfg::DN_VALUE_DIM;
    constexpr int QK_SIZE  = Cfg::DN_QK_SIZE;
    constexpr int V_SIZE   = Cfg::DN_V_SIZE;
    constexpr int CONV_CH  = Cfg::DN_CONV_CH;
    constexpr int CONV_K   = Cfg::DN_CONV_KERNEL;

    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    __nv_bfloat16 *s_norm = shmem;

    // 1. RMSNorm + capture residual.
    rmsnorm_capture<Cfg>(input, w.input_layernorm_weight, s_norm, g_residual);

    // 2. Projections.
    matvec_bf16<Cfg>(s_norm, w.qkv_proj_weight,    g_qkv,   H, CONV_CH, num_blocks);
    matvec_bf16<Cfg>(s_norm, w.z_proj_weight,      g_z,     H, V_SIZE,  num_blocks);
    matvec_bf16<Cfg>(s_norm, w.beta_proj_weight,   g_beta,  H, V_HEADS, num_blocks);
    matvec_bf16<Cfg>(s_norm, w.alpha_proj_weight,  g_alpha, H, V_HEADS, num_blocks);
    grid.sync();

    // 3. Conv1d + SiLU + recurrence. One block per V head.
    if (block_id < V_HEADS) {
        int v_head  = block_id;
        int qk_head = v_head / V_PER_QK;

        __shared__ float s_q[Cfg::DN_KEY_DIM];
        __shared__ float s_k[Cfg::DN_KEY_DIM];
        __shared__ float s_v[Cfg::DN_VALUE_DIM];

        // Three regions to conv1d: Q[qk_head], K[qk_head], V[v_head].
        struct Region { int ch_base; int count; float *dst; };
        Region regs[3] = {
            { qk_head * KEY,             KEY, s_q },
            { QK_SIZE + qk_head * KEY,   KEY, s_k },
            { 2*QK_SIZE + v_head  * VAL, VAL, s_v },
        };
        for (int r = 0; r < 3; ++r) {
            const Region &R = regs[r];
            for (int c = threadIdx.x; c < R.count; c += BLOCK_SIZE) {
                int ch = R.ch_base + c;
                // Shift the conv ring buffer by 1 (kernel size CONV_K = 4).
                float h0 = conv_buf[ch * CONV_K + 1];
                float h1 = conv_buf[ch * CONV_K + 2];
                float h2 = conv_buf[ch * CONV_K + 3];
                conv_buf[ch * CONV_K + 0] = h0;
                conv_buf[ch * CONV_K + 1] = h1;
                conv_buf[ch * CONV_K + 2] = h2;
                conv_buf[ch * CONV_K + 3] = g_qkv[ch];
                float co = 0.0f;
                #pragma unroll
                for (int t = 0; t < CONV_K; ++t)
                    co += conv_buf[ch * CONV_K + t]
                          * __bfloat162float(__ldg(w.conv1d_weight + ch * CONV_K + t));
                R.dst[c] = fast_silu(co);
            }
        }

        // Beta / alpha activations — per V head.
        if (threadIdx.x == 0) {
            g_beta[v_head] = fast_sigmoid(g_beta[v_head]);
            float a_log_val = __bfloat162float(__ldg(w.a_log + v_head));
            float dt_b      = __bfloat162float(__ldg(w.dt_bias + v_head));
            float x  = g_alpha[v_head] + dt_b;
            float sp = (x > 20.0f) ? x : logf(1.0f + fast_exp(x));
            g_alpha[v_head] = fast_exp(-fast_exp(a_log_val) * sp);
        }
        __syncthreads();

        // L2 normalize Q, K (per-head). Q gets the constant /sqrt(128)
        // factor baked in.
        constexpr float Q_SCALE = 1.0f / 11.313708498984761f;  // 1/sqrt(128)
        if (warp_id == 0) {
            float sq = 0.0f;
            for (int i = lane_id; i < KEY; i += WARP_SIZE) sq += s_q[i] * s_q[i];
            sq = warp_reduce_sum_x(sq);
            float n = rsqrtf(sq + 1e-6f) * Q_SCALE;
            n = __shfl_sync(0xffffffff, n, 0);
            for (int i = lane_id; i < KEY; i += WARP_SIZE) s_q[i] *= n;
        }
        if (warp_id == 1) {
            float sq = 0.0f;
            for (int i = lane_id; i < KEY; i += WARP_SIZE) sq += s_k[i] * s_k[i];
            sq = warp_reduce_sum_x(sq);
            float n = rsqrtf(sq + 1e-6f);
            n = __shfl_sync(0xffffffff, n, 0);
            for (int i = lane_id; i < KEY; i += WARP_SIZE) s_k[i] *= n;
        }
        __syncthreads();

        float decay = g_alpha[v_head];
        float beta  = g_beta[v_head];

        __shared__ float s_kq;
        if (warp_id == 0) {
            float kq = 0.0f;
            for (int i = lane_id; i < KEY; i += WARP_SIZE) kq += s_k[i] * s_q[i];
            kq = warp_reduce_sum_x(kq);
            if (lane_id == 0) s_kq = kq;
        }
        __syncthreads();
        float kq = s_kq;

        // Recurrence: state for this V head.
        float *state = dn_state + (size_t)v_head * KEY * VAL;
        float *out_head = g_dn_out + v_head * VAL;

        constexpr int J_PER_WARP = VAL / NUM_WARPS;
        constexpr int I_PER_LANE = KEY / WARP_SIZE;

        #pragma unroll
        for (int jj = 0; jj < J_PER_WARP; ++jj) {
            int j = warp_id * J_PER_WARP + jj;
            float s_regs[I_PER_LANE], stk = 0.0f, sqv = 0.0f;
            #pragma unroll
            for (int ii = 0; ii < I_PER_LANE; ++ii) {
                int i  = lane_id + ii * WARP_SIZE;
                float sv = state[j * KEY + i];
                s_regs[ii] = sv;
                stk += sv * s_k[i];
                sqv += sv * s_q[i];
            }
            stk = warp_reduce_sum_x(stk);
            sqv = warp_reduce_sum_x(sqv);
            stk = __shfl_sync(0xffffffff, stk, 0);
            sqv = __shfl_sync(0xffffffff, sqv, 0);
            float error_j = (s_v[j] - stk) * beta;
            float o_j = decay * sqv + error_j * kq;
            if (lane_id == 0) out_head[j] = o_j;
            #pragma unroll
            for (int ii = 0; ii < I_PER_LANE; ++ii) {
                int i = lane_id + ii * WARP_SIZE;
                state[j * KEY + i] = s_regs[ii] * decay + s_k[i] * error_j;
            }
        }

        // Per-head group RMSNorm + SiLU(z) gate.
        __syncthreads();
        {
            __shared__ float s_grms[NUM_WARPS];
            float ss = 0.0f;
            for (int i = threadIdx.x; i < VAL; i += BLOCK_SIZE)
                ss += out_head[i] * out_head[i];
            ss = warp_reduce_sum_x(ss);
            if (lane_id == 0) s_grms[warp_id] = ss;
            __syncthreads();
            if (warp_id == 0) {
                float v = (lane_id < NUM_WARPS) ? s_grms[lane_id] : 0.0f;
                v = warp_reduce_sum_x(v);
                if (lane_id == 0) s_grms[0] = rsqrtf(v / float(VAL) + 1e-6f);
            }
            __syncthreads();
            float rstd = s_grms[0];
            for (int i = threadIdx.x; i < VAL; i += BLOCK_SIZE) {
                float normed = out_head[i] * rstd
                               * __bfloat162float(__ldg(w.norm_weight + i));
                float gate   = fast_silu(g_z[v_head * VAL + i]);
                out_head[i]  = normed * gate;
            }
        }
    }
    grid.sync();

    // 4. Out projection + residual.
    {
        float *s_dn = reinterpret_cast<float *>(shmem);
        for (int i = threadIdx.x; i < V_SIZE; i += BLOCK_SIZE) s_dn[i] = g_dn_out[i];
        __syncthreads();
        matvec_o_residual<Cfg>(s_dn, w.out_proj_weight, g_residual, hidden_out,
                                V_SIZE, H, num_blocks);
    }
    grid.sync();

    // 5. Post-attn RMSNorm + MLP (uses same shmem region; residual is
    //    `hidden_out` here, captured into g_residual by rmsnorm_capture).
    __nv_bfloat16 *s_act = shmem;
    rmsnorm_capture<Cfg>(hidden_out, w.post_attn_layernorm_weight, s_act, g_residual);

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
