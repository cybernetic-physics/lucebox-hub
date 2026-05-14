/**
 * Full templated decode megakernel for Qwen3.x.
 *
 * Walks all NUM_LAYERS layers of the hybrid Gated DeltaNet + Gated
 * Attention network in a single persistent kernel dispatch. Same
 * pattern as `models/qwen35_0p8b/kernel.cu:decode_kernel`, but every
 * device function is templated on `Cfg` so the same source compiles
 * for both Cfg_0p8B and Cfg_27B.
 *
 * Build target: sm_120+ for Cfg_27B (FA HEAD_DIM=256, INTERMEDIATE=17408
 * push register pressure but stay within the SM's resource limits).
 * Cfg_0p8B builds at any arch.
 *
 * Correctness status:
 *   - Primitives (rmsnorm, matvec, mlp_forward) smoke-tested against
 *     PyTorch reference -- pass at both Cfg specializations.
 *   - FA and DN layers ported from 0.8B reference faithfully; the DN
 *     V/QK split (DN_V_PER_QK = 3 for 27B) is new code.
 *   - End-to-end HF correctness on Qwen3.6-27B requires the 54 GB BF16
 *     weight pull (see ../reference/capture_hf_reference.py). The
 *     regression harness in ../test/test_correctness_vs_hf.py runs
 *     against HF once weights are present.
 */
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "Cfg.cuh"
#include "helpers.cuh"
#include "matvec.cuh"
#include "rmsnorm.cuh"
#include "rope.cuh"
#include "fa_layer.cuh"
#include "dn_layer.cuh"

namespace lucebox::qwen3x {

// Variant per-layer weight pointer block: union-style with both FA and DN.
template<typename Cfg>
struct LayerWeights {
    int layer_type;       // 0 = DN, 1 = FA  (matches FamilyInvariants::is_fa_layer)
    int _pad;
    union {
        FullAttnWeights<Cfg> fa;
        DeltaNetWeights<Cfg> dn;
    };
};

template<typename Cfg>
__global__ void __launch_bounds__(BLOCK_SIZE, 1)
decode_kernel_impl(
    const __nv_bfloat16 *__restrict__ embed_weight,        // [VOCAB, HIDDEN]
    const __nv_bfloat16 *__restrict__ final_norm_weight,   // [HIDDEN]
    const LayerWeights<Cfg> *__restrict__ layer_weights,
    __nv_bfloat16 *__restrict__ fa_k_cache,                // [N_FA, KV_H, MAX_SEQ, HEAD]
    __nv_bfloat16 *__restrict__ fa_v_cache,
    float *__restrict__ dn_states,                          // [N_DN, V_H, VAL, KEY]
    float *__restrict__ conv_bufs,                          // [N_DN, CONV_CH, CONV_K]
    __nv_bfloat16 *__restrict__ hidden_buffer,              // [HIDDEN]
    __nv_bfloat16 *__restrict__ g_residual,                 // [HIDDEN]
    float *__restrict__ g_qkv_scratch,                      // [max(FA_QPROJ, DN_CONV_CH)]
    float *__restrict__ g_kv_scratch,                       // [FA_KV_SIZE * 2]
    float *__restrict__ g_attn_out,                         // [max(FA_Q_SIZE, DN_V_SIZE)]
    float *__restrict__ g_mlp_inter,                        // [INTER]
    float *__restrict__ g_z_scratch,                        // [DN_V_SIZE]
    float *__restrict__ g_beta_scratch,                     // [DN_NUM_V_HEADS]
    float *__restrict__ g_alpha_scratch,                    // [DN_NUM_V_HEADS]
    float *__restrict__ g_normalized,                       // [HIDDEN] fp32 final
    float *__restrict__ g_fa_partials,                      // for FA split-K
    float *__restrict__ g_rope_inv_freq,                    // [FA_ROTARY_DIM/2]
    YarnParams yp,
    int input_token_id,
    int position,
    int pos_h, int pos_w,  // 0 for text-only
    int max_seq_len)
{
    AtomicGridSync grid{};
    constexpr int H        = Cfg::HIDDEN;
    constexpr int KV_H     = Cfg::FA_NUM_KV_HEADS;
    constexpr int HEAD     = Cfg::FA_HEAD_DIM;
    constexpr int VAL      = Cfg::DN_VALUE_DIM;
    constexpr int KEY      = Cfg::DN_KEY_DIM;
    constexpr int V_HEADS  = Cfg::DN_NUM_V_HEADS;
    constexpr int CONV_CH  = Cfg::DN_CONV_CH;
    constexpr int CONV_K   = Cfg::DN_CONV_KERNEL;

    __shared__ __align__(16)
        char shmem_raw[(Cfg::INTERMEDIATE > Cfg::HIDDEN ? Cfg::INTERMEDIATE
                                                        : Cfg::HIDDEN) * sizeof(float)];
    __nv_bfloat16 *shmem_bf16 = reinterpret_cast<__nv_bfloat16 *>(shmem_raw);

    // 1. Embed lookup -> hidden_buffer (block 0 writes; everyone reads
    //    through hidden_buffer below).
    if (blockIdx.x == 0) {
        const __nv_bfloat16 *erow = embed_weight + (size_t)input_token_id * H;
        for (int i = threadIdx.x; i < H; i += BLOCK_SIZE) hidden_buffer[i] = erow[i];
    }
    // 2. Precompute RoPE inv-freq table (lane-bounded, runs once).
    if (blockIdx.x == 0 && threadIdx.x < Cfg::FA_ROTARY_DIM / 2) {
        compute_base_inv_freq<Cfg>(g_rope_inv_freq);
    }
    grid.sync();

    // 3. Layer loop.
    int fa_kv_stride = KV_H * max_seq_len * HEAD;
    size_t dn_state_stride = (size_t)V_HEADS * KEY * VAL;
    size_t conv_stride = (size_t)CONV_CH * CONV_K;
    int dn_layer_idx = 0, fa_layer_idx = 0;
    MRopeSections sections = MROPE_QWEN36;

    #pragma unroll 1
    for (int layer = 0; layer < Cfg::NUM_LAYERS; ++layer) {
        const __nv_bfloat16 *layer_input = hidden_buffer;
        if (layer_weights[layer].layer_type == 0) {
            delta_net_layer<Cfg>(
                grid, layer_weights[layer].dn, layer_input,
                g_residual,
                g_qkv_scratch, g_z_scratch,
                g_beta_scratch, g_alpha_scratch,
                g_attn_out, g_mlp_inter,
                dn_states + (size_t)dn_layer_idx * dn_state_stride,
                conv_bufs + (size_t)dn_layer_idx * conv_stride,
                hidden_buffer, shmem_bf16);
            ++dn_layer_idx;
        } else {
            full_attention_layer<Cfg>(
                grid, layer_weights[layer].fa, layer_input,
                fa_k_cache + (size_t)fa_layer_idx * fa_kv_stride,
                fa_v_cache + (size_t)fa_layer_idx * fa_kv_stride,
                g_residual, g_qkv_scratch, g_kv_scratch,
                g_attn_out, g_fa_partials,
                g_rope_inv_freq, yp, sections,
                position, max_seq_len, pos_h, pos_w,
                shmem_bf16, hidden_buffer);
            ++fa_layer_idx;
        }
    }

    // 4. Final RMSNorm (block 0 only; result lives in g_normalized as fp32
    //    for downstream LM head consumption).
    if (blockIdx.x == 0) {
        __shared__ float s_reduce[NUM_WARPS];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        float ssq = 0.0f;
        for (int i = threadIdx.x; i < H; i += BLOCK_SIZE) {
            float v = __bfloat162float(hidden_buffer[i]);
            g_normalized[i] = v;
            ssq += v * v;
        }
        ssq = warp_reduce_sum_x(ssq);
        if (lane_id == 0) s_reduce[warp_id] = ssq;
        __syncthreads();
        if (warp_id == 0) {
            float v = (lane_id < NUM_WARPS) ? s_reduce[lane_id] : 0.0f;
            v = warp_reduce_sum_x(v);
            if (lane_id == 0) s_reduce[0] = rsqrtf(v / float(H) + 1e-6f);
        }
        __syncthreads();
        float rstd = s_reduce[0];
        for (int i = threadIdx.x; i < H; i += BLOCK_SIZE) {
            float w = __bfloat162float(__ldg(final_norm_weight + i));
            g_normalized[i] = g_normalized[i] * rstd * (1.0f + w);
        }
    }
}

// Explicit instantiations.
template __global__ void decode_kernel_impl<Cfg_0p8B>(
    const __nv_bfloat16*, const __nv_bfloat16*,
    const LayerWeights<Cfg_0p8B>*,
    __nv_bfloat16*, __nv_bfloat16*, float*, float*,
    __nv_bfloat16*, __nv_bfloat16*,
    float*, float*, float*, float*, float*, float*, float*, float*,
    float*, float*, YarnParams,
    int, int, int, int, int);
template __global__ void decode_kernel_impl<Cfg_27B>(
    const __nv_bfloat16*, const __nv_bfloat16*,
    const LayerWeights<Cfg_27B>*,
    __nv_bfloat16*, __nv_bfloat16*, float*, float*,
    __nv_bfloat16*, __nv_bfloat16*,
    float*, float*, float*, float*, float*, float*, float*, float*,
    float*, float*, YarnParams,
    int, int, int, int, int);

}  // namespace lucebox::qwen3x
