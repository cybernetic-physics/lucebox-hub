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

// Variant per-layer weight pointer block. 192 bytes so the union can hold
// either a BF16 layer (DN_bf16 = 112B is biggest) or an NVFP4 layer
// (DN_nvfp4 = 168B is biggest). Python packer stride must match exactly
// — see PACK_STRUCT in weight_packer.py.
//
//   layer_type values:
//     0 = DN_bf16   1 = FA_bf16   2 = DN_nvfp4   3 = FA_nvfp4
template<typename Cfg>
struct LayerWeights {
    int layer_type;
    int _pad0;
    union {
        FullAttnWeights<Cfg>       fa;
        DeltaNetWeights<Cfg>       dn;
        FullAttnWeightsNVFP4<Cfg>  fa_nvfp4;
        DeltaNetWeightsNVFP4<Cfg>  dn_nvfp4;
        char _force_size[184];   // 184 + 8 header = 192 byte struct
    };
};
static_assert(sizeof(LayerWeights<Cfg_0p8B>) == 192,
              "LayerWeights<Cfg_0p8B> must be 192 bytes");
static_assert(sizeof(LayerWeights<Cfg_27B>)  == 192,
              "LayerWeights<Cfg_27B> must be 192 bytes");

template<typename Cfg, bool USE_NVFP4 = false>
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
    int max_seq_len,
    __nv_bfloat16 *__restrict__ g_layer_outputs,  // optional [NUM_LAYERS, HIDDEN]
    const int *__restrict__ input_token_id_dev)   // optional device ptr; overrides input_token_id when non-null
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

    // Dynamic shmem (extern __shared__): for Cfg_27B this region is
    // max(INTERMEDIATE, HIDDEN) * sizeof(float) = 17408 * 4 = 69632
    // bytes, which exceeds the 48 KB static-shmem ABI cap. The host
    // launcher (launch_decode_impl below) opts the kernel into the
    // 100 KB Blackwell shmem limit via cudaFuncSetAttribute(
    // cudaFuncAttributeMaxDynamicSharedMemorySize, ...) before the
    // cooperative-grid launch.
    extern __shared__ __align__(16) char shmem_raw[];
    __nv_bfloat16 *shmem_bf16 = reinterpret_cast<__nv_bfloat16 *>(shmem_raw);

    // 1. Embed lookup -> hidden_buffer (block 0 writes; everyone reads
    //    through hidden_buffer below). If `input_token_id_dev` is non-
    //    null, read the token id from device memory (lets the caller
    //    chain decode<-lm_head_argmax without host sync between
    //    iterations; required for CUDA Graph capture).
    int tok_id = input_token_id;
    if (input_token_id_dev != nullptr) tok_id = input_token_id_dev[0];
    if (blockIdx.x == 0) {
        const __nv_bfloat16 *erow = embed_weight + (size_t)tok_id * H;
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
        int lt = layer_weights[layer].layer_type;
        if constexpr (USE_NVFP4) {
            if (lt == 2) {
                delta_net_layer_nvfp4<Cfg>(
                    grid, layer_weights[layer].dn_nvfp4, layer_input,
                    g_residual,
                    g_qkv_scratch, g_z_scratch,
                    g_beta_scratch, g_alpha_scratch,
                    g_attn_out, g_mlp_inter,
                    dn_states + (size_t)dn_layer_idx * dn_state_stride,
                    conv_bufs + (size_t)dn_layer_idx * conv_stride,
                    position,
                    hidden_buffer, shmem_bf16);
                ++dn_layer_idx;
            } else {  // lt == 3 (FA_nvfp4)
                full_attention_layer_nvfp4<Cfg>(
                    grid, layer_weights[layer].fa_nvfp4, layer_input,
                    fa_k_cache + (size_t)fa_layer_idx * fa_kv_stride,
                    fa_v_cache + (size_t)fa_layer_idx * fa_kv_stride,
                    g_residual, g_qkv_scratch, g_kv_scratch,
                    g_attn_out, g_fa_partials,
                    g_rope_inv_freq, yp, sections,
                    position, max_seq_len, pos_h, pos_w,
                    shmem_bf16, hidden_buffer);
                ++fa_layer_idx;
            }
        } else {
            if (lt == 0) {
                delta_net_layer<Cfg>(
                    grid, layer_weights[layer].dn, layer_input,
                    g_residual,
                    g_qkv_scratch, g_z_scratch,
                    g_beta_scratch, g_alpha_scratch,
                    g_attn_out, g_mlp_inter,
                    dn_states + (size_t)dn_layer_idx * dn_state_stride,
                    conv_bufs + (size_t)dn_layer_idx * conv_stride,
                    position,
                    hidden_buffer, shmem_bf16);
                ++dn_layer_idx;
            } else {  // lt == 1
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
        // Debug capture: write hidden_buffer to layer-indexed slot if requested.
        if (g_layer_outputs != nullptr) {
            grid.sync();
            if (blockIdx.x == 0) {
                __nv_bfloat16 *dst = g_layer_outputs + (size_t)layer * H;
                for (int i = threadIdx.x; i < H; i += BLOCK_SIZE) {
                    dst[i] = hidden_buffer[i];
                }
            }
            grid.sync();
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

// ---------------------------------------------------------------------------
// Host-side launchers — extern C wrappers around the cooperative-grid
// dispatch. Each calls the corresponding Cfg specialization.
// ---------------------------------------------------------------------------
template<typename Cfg, bool USE_NVFP4>
static cudaError_t launch_decode_impl(
    void *embed_weight, void *final_norm_weight,
    void *layer_weights,
    void *fa_k_cache, void *fa_v_cache,
    void *dn_states, void *conv_bufs,
    void *hidden_buffer, void *g_residual,
    void *g_qkv_scratch, void *g_kv_scratch,
    void *g_attn_out, void *g_mlp_inter,
    void *g_z_scratch, void *g_beta_scratch, void *g_alpha_scratch,
    void *g_normalized, void *g_fa_partials, void *g_rope_inv_freq,
    YarnParams yp,
    int input_token_id, int position, int pos_h, int pos_w, int max_seq_len,
    int num_blocks,
    void *g_layer_outputs,
    const void *input_token_id_dev,
    cudaStream_t stream)
{
    void *args[] = {
        &embed_weight, &final_norm_weight, &layer_weights,
        &fa_k_cache, &fa_v_cache, &dn_states, &conv_bufs,
        &hidden_buffer, &g_residual,
        &g_qkv_scratch, &g_kv_scratch, &g_attn_out, &g_mlp_inter,
        &g_z_scratch, &g_beta_scratch, &g_alpha_scratch,
        &g_normalized, &g_fa_partials, &g_rope_inv_freq,
        &yp,
        &input_token_id, &position, &pos_h, &pos_w, &max_seq_len,
        &g_layer_outputs,
        &input_token_id_dev,
    };
    dim3 grid(num_blocks);
    dim3 block(BLOCK_SIZE);
    constexpr int DYN_SHMEM_BYTES =
        (Cfg::INTERMEDIATE > Cfg::HIDDEN ? Cfg::INTERMEDIATE
                                         : Cfg::HIDDEN) * (int)sizeof(float);
    // Opt the kernel into the Blackwell extended shmem limit
    // (sm_120/121 allows up to 100 KB dynamic shmem per block, vs
    // the 48 KB default). The attribute is per-kernel and persists
    // for the lifetime of the CUDA context; set it once with a
    // function-local static guard so we don't pay the syscall
    // every launch.
    static bool attr_set = false;
    if (!attr_set) {
        cudaError_t e = cudaFuncSetAttribute(
            (void *)decode_kernel_impl<Cfg, USE_NVFP4>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SHMEM_BYTES);
        if (e != cudaSuccess) return e;
        // Also pin the SM L1/shmem carveout to all-shmem so the driver
        // doesn't shrink the per-block shmem allocation between
        // launches. Without this, repeated cooperative launches with
        // > 48 KB dyn shmem on sm_121a sometimes return garbage from
        // the dynamic-shmem region on the second decoder.
        e = cudaFuncSetAttribute(
            (void *)decode_kernel_impl<Cfg, USE_NVFP4>,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared);
        if (e != cudaSuccess) return e;
        attr_set = true;
    }
    return cudaLaunchCooperativeKernel(
        (void *)decode_kernel_impl<Cfg, USE_NVFP4>, grid, block, args,
        DYN_SHMEM_BYTES, stream);
}

extern "C" cudaError_t launch_decode_0p8b(
    void *embed_weight, void *final_norm_weight, void *layer_weights,
    void *fa_k_cache, void *fa_v_cache, void *dn_states, void *conv_bufs,
    void *hidden_buffer, void *g_residual,
    void *g_qkv_scratch, void *g_kv_scratch, void *g_attn_out, void *g_mlp_inter,
    void *g_z_scratch, void *g_beta_scratch, void *g_alpha_scratch,
    void *g_normalized, void *g_fa_partials, void *g_rope_inv_freq,
    YarnParams yp,
    int input_token_id, int position, int pos_h, int pos_w, int max_seq_len,
    int num_blocks, void *g_layer_outputs,
    const void *input_token_id_dev, cudaStream_t stream)
{
    return launch_decode_impl<Cfg_0p8B, false>(
        embed_weight, final_norm_weight, layer_weights,
        fa_k_cache, fa_v_cache, dn_states, conv_bufs,
        hidden_buffer, g_residual,
        g_qkv_scratch, g_kv_scratch, g_attn_out, g_mlp_inter,
        g_z_scratch, g_beta_scratch, g_alpha_scratch,
        g_normalized, g_fa_partials, g_rope_inv_freq, yp,
        input_token_id, position, pos_h, pos_w, max_seq_len,
        num_blocks, g_layer_outputs, input_token_id_dev, stream);
}

extern "C" cudaError_t launch_decode_27b(
    void *embed_weight, void *final_norm_weight, void *layer_weights,
    void *fa_k_cache, void *fa_v_cache, void *dn_states, void *conv_bufs,
    void *hidden_buffer, void *g_residual,
    void *g_qkv_scratch, void *g_kv_scratch, void *g_attn_out, void *g_mlp_inter,
    void *g_z_scratch, void *g_beta_scratch, void *g_alpha_scratch,
    void *g_normalized, void *g_fa_partials, void *g_rope_inv_freq,
    YarnParams yp,
    int input_token_id, int position, int pos_h, int pos_w, int max_seq_len,
    int num_blocks, void *g_layer_outputs,
    const void *input_token_id_dev, cudaStream_t stream)
{
    return launch_decode_impl<Cfg_27B, false>(
        embed_weight, final_norm_weight, layer_weights,
        fa_k_cache, fa_v_cache, dn_states, conv_bufs,
        hidden_buffer, g_residual,
        g_qkv_scratch, g_kv_scratch, g_attn_out, g_mlp_inter,
        g_z_scratch, g_beta_scratch, g_alpha_scratch,
        g_normalized, g_fa_partials, g_rope_inv_freq, yp,
        input_token_id, position, pos_h, pos_w, max_seq_len,
        num_blocks, g_layer_outputs, input_token_id_dev, stream);
}

extern "C" cudaError_t launch_decode_0p8b_nvfp4(
    void *embed_weight, void *final_norm_weight, void *layer_weights,
    void *fa_k_cache, void *fa_v_cache, void *dn_states, void *conv_bufs,
    void *hidden_buffer, void *g_residual,
    void *g_qkv_scratch, void *g_kv_scratch, void *g_attn_out, void *g_mlp_inter,
    void *g_z_scratch, void *g_beta_scratch, void *g_alpha_scratch,
    void *g_normalized, void *g_fa_partials, void *g_rope_inv_freq,
    YarnParams yp,
    int input_token_id, int position, int pos_h, int pos_w, int max_seq_len,
    int num_blocks, void *g_layer_outputs,
    const void *input_token_id_dev, cudaStream_t stream)
{
    return launch_decode_impl<Cfg_0p8B, true>(
        embed_weight, final_norm_weight, layer_weights,
        fa_k_cache, fa_v_cache, dn_states, conv_bufs,
        hidden_buffer, g_residual,
        g_qkv_scratch, g_kv_scratch, g_attn_out, g_mlp_inter,
        g_z_scratch, g_beta_scratch, g_alpha_scratch,
        g_normalized, g_fa_partials, g_rope_inv_freq, yp,
        input_token_id, position, pos_h, pos_w, max_seq_len,
        num_blocks, g_layer_outputs, input_token_id_dev, stream);
}

extern "C" cudaError_t launch_decode_27b_nvfp4(
    void *embed_weight, void *final_norm_weight, void *layer_weights,
    void *fa_k_cache, void *fa_v_cache, void *dn_states, void *conv_bufs,
    void *hidden_buffer, void *g_residual,
    void *g_qkv_scratch, void *g_kv_scratch, void *g_attn_out, void *g_mlp_inter,
    void *g_z_scratch, void *g_beta_scratch, void *g_alpha_scratch,
    void *g_normalized, void *g_fa_partials, void *g_rope_inv_freq,
    YarnParams yp,
    int input_token_id, int position, int pos_h, int pos_w, int max_seq_len,
    int num_blocks, void *g_layer_outputs,
    const void *input_token_id_dev, cudaStream_t stream)
{
    return launch_decode_impl<Cfg_27B, true>(
        embed_weight, final_norm_weight, layer_weights,
        fa_k_cache, fa_v_cache, dn_states, conv_bufs,
        hidden_buffer, g_residual,
        g_qkv_scratch, g_kv_scratch, g_attn_out, g_mlp_inter,
        g_z_scratch, g_beta_scratch, g_alpha_scratch,
        g_normalized, g_fa_partials, g_rope_inv_freq, yp,
        input_token_id, position, pos_h, pos_w, max_seq_len,
        num_blocks, g_layer_outputs, input_token_id_dev, stream);
}

// Explicit instantiations.
template __global__ void decode_kernel_impl<Cfg_0p8B>(
    const __nv_bfloat16*, const __nv_bfloat16*,
    const LayerWeights<Cfg_0p8B>*,
    __nv_bfloat16*, __nv_bfloat16*, float*, float*,
    __nv_bfloat16*, __nv_bfloat16*,
    float*, float*, float*, float*, float*, float*, float*, float*,
    float*, float*, YarnParams,
    int, int, int, int, int,
    __nv_bfloat16*, const int*);
template __global__ void decode_kernel_impl<Cfg_27B>(
    const __nv_bfloat16*, const __nv_bfloat16*,
    const LayerWeights<Cfg_27B>*,
    __nv_bfloat16*, __nv_bfloat16*, float*, float*,
    __nv_bfloat16*, __nv_bfloat16*,
    float*, float*, float*, float*, float*, float*, float*, float*,
    float*, float*, YarnParams,
    int, int, int, int, int,
    __nv_bfloat16*, const int*);

}  // namespace lucebox::qwen3x
