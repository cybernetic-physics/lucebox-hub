/**
 * Multi-token prefill megakernel for Qwen3.x — templated on Cfg.
 *
 * Strategy in this commit (working, correctness-first):
 *
 *   prefill_loop_impl<Cfg>(... S, ...)  -- runs `decode_kernel_impl<Cfg>`
 *   in a host-side loop, one token at a time, positions 0..S-1. Each
 *   step writes to the KV cache and advances the DN recurrent state.
 *   Same correctness as the per-token decode path; ~S× slower than an
 *   optimized parallel-S prefill.
 *
 *   This is exposed as `qwen3x_C.prefill_qwen3x_naive` so the
 *   correctness harness can validate against HF without waiting for
 *   the optimized prefill to land.
 *
 * TODO (next session, ~3 days): write a proper parallel-S prefill kernel
 * that tiles the S-dimension across blocks. The structure is:
 *
 *   for each layer:
 *     matvec [S, H] -> [S, *]  via cuBLAS or hand-rolled tile
 *     per-position head-norm + RoPE
 *     batched FA over the S × S triangle (causal mask)
 *     DN sequential recurrence (cannot parallelize over S)
 *     out-proj + residual
 *     post-attn norm + MLP
 *
 * The DN recurrence is inherently sequential (state at t depends on
 * state at t-1). It can be chunked (Mamba-style) but that's the bulk
 * of the optimization effort. For first-shot correctness validation,
 * the per-token loop is fine.
 */
#include <cuda_runtime.h>
#include <stdint.h>

#include "Cfg.cuh"
#include "rope.cuh"

namespace lucebox::qwen3x {

// Forward declarations of the per-token launchers (defined in
// kernel_decode_full.cu).
extern "C" cudaError_t launch_decode_0p8b(
    void *embed_weight, void *final_norm_weight, void *layer_weights,
    void *fa_k_cache, void *fa_v_cache, void *dn_states, void *conv_bufs,
    void *hidden_buffer, void *g_residual,
    void *g_qkv_scratch, void *g_kv_scratch, void *g_attn_out, void *g_mlp_inter,
    void *g_z_scratch, void *g_beta_scratch, void *g_alpha_scratch,
    void *g_normalized, void *g_fa_partials, void *g_rope_inv_freq,
    YarnParams yp,
    int input_token_id, int position, int pos_h, int pos_w, int max_seq_len,
    int num_blocks, void *g_layer_outputs, cudaStream_t stream);

extern "C" cudaError_t launch_decode_27b(
    void *embed_weight, void *final_norm_weight, void *layer_weights,
    void *fa_k_cache, void *fa_v_cache, void *dn_states, void *conv_bufs,
    void *hidden_buffer, void *g_residual,
    void *g_qkv_scratch, void *g_kv_scratch, void *g_attn_out, void *g_mlp_inter,
    void *g_z_scratch, void *g_beta_scratch, void *g_alpha_scratch,
    void *g_normalized, void *g_fa_partials, void *g_rope_inv_freq,
    YarnParams yp,
    int input_token_id, int position, int pos_h, int pos_w, int max_seq_len,
    int num_blocks, void *g_layer_outputs, cudaStream_t stream);

extern "C" cudaError_t launch_decode_0p8b_nvfp4(
    void *embed_weight, void *final_norm_weight, void *layer_weights,
    void *fa_k_cache, void *fa_v_cache, void *dn_states, void *conv_bufs,
    void *hidden_buffer, void *g_residual,
    void *g_qkv_scratch, void *g_kv_scratch, void *g_attn_out, void *g_mlp_inter,
    void *g_z_scratch, void *g_beta_scratch, void *g_alpha_scratch,
    void *g_normalized, void *g_fa_partials, void *g_rope_inv_freq,
    YarnParams yp,
    int input_token_id, int position, int pos_h, int pos_w, int max_seq_len,
    int num_blocks, void *g_layer_outputs, cudaStream_t stream);

extern "C" cudaError_t launch_decode_27b_nvfp4(
    void *embed_weight, void *final_norm_weight, void *layer_weights,
    void *fa_k_cache, void *fa_v_cache, void *dn_states, void *conv_bufs,
    void *hidden_buffer, void *g_residual,
    void *g_qkv_scratch, void *g_kv_scratch, void *g_attn_out, void *g_mlp_inter,
    void *g_z_scratch, void *g_beta_scratch, void *g_alpha_scratch,
    void *g_normalized, void *g_fa_partials, void *g_rope_inv_freq,
    YarnParams yp,
    int input_token_id, int position, int pos_h, int pos_w, int max_seq_len,
    int num_blocks, void *g_layer_outputs, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Naive prefill: S sequential decode calls. Token i is read from
// host_token_ids[i] (a device-side int32 array). Last position's logits
// are computable from g_normalized after the loop completes.
// ---------------------------------------------------------------------------
template<int CFG_ID>
static cudaError_t prefill_naive_impl(
    const int32_t *device_token_ids,    // [S] int32
    int S,
    void *embed_weight, void *final_norm_weight, void *layer_weights,
    void *fa_k_cache, void *fa_v_cache, void *dn_states, void *conv_bufs,
    void *hidden_buffer, void *g_residual,
    void *g_qkv_scratch, void *g_kv_scratch, void *g_attn_out, void *g_mlp_inter,
    void *g_z_scratch, void *g_beta_scratch, void *g_alpha_scratch,
    void *g_normalized, void *g_fa_partials, void *g_rope_inv_freq,
    YarnParams yp, int max_seq_len, int num_blocks,
    void *g_layer_outputs,    // optional [NUM_LAYERS, HIDDEN] — captures
                              // ONLY the last decode step's layer outputs
    int start_position,        // offset into KV cache; 0 for fresh prefill
    cudaStream_t stream)
{
    // Copy each token id to host (could be batched, but S is at most 32k
    // and the copy is small — bound by the kernel launch latency, not the
    // memcpy.)
    auto launcher = (CFG_ID == 0) ? &launch_decode_0p8b
                  : (CFG_ID == 1) ? &launch_decode_27b
                  : (CFG_ID == 2) ? &launch_decode_0p8b_nvfp4
                                  : &launch_decode_27b_nvfp4;
    for (int pos_rel = 0; pos_rel < S; ++pos_rel) {
        int pos = pos_rel + start_position;
        int32_t tok = 0;
        cudaError_t err = cudaMemcpyAsync(&tok, device_token_ids + pos_rel,
                                           sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) return err;
        cudaStreamSynchronize(stream);

        // Capture per-layer outputs only on the LAST step so the buffer
        // ends up holding the final-position hidden states (matching HF's
        // hidden_states[i][:, -1, :] semantic).
        void *capture_this_step = (pos_rel == S - 1) ? g_layer_outputs : nullptr;

        err = launcher(
            embed_weight, final_norm_weight, layer_weights,
            fa_k_cache, fa_v_cache, dn_states, conv_bufs,
            hidden_buffer, g_residual,
            g_qkv_scratch, g_kv_scratch, g_attn_out, g_mlp_inter,
            g_z_scratch, g_beta_scratch, g_alpha_scratch,
            g_normalized, g_fa_partials, g_rope_inv_freq, yp,
            // text-only: pass pos for all 3 MRoPE axes so all rotary
            // pairs rotate (matches HF's text-only position_ids).
            (int)tok, pos, pos, pos, max_seq_len, num_blocks,
            capture_this_step, stream);
        if (err != cudaSuccess) return err;
    }
    return cudaSuccess;
}

extern "C" cudaError_t launch_prefill_naive_0p8b(
    const int32_t *device_token_ids, int S,
    void *embed_weight, void *final_norm_weight, void *layer_weights,
    void *fa_k_cache, void *fa_v_cache, void *dn_states, void *conv_bufs,
    void *hidden_buffer, void *g_residual,
    void *g_qkv_scratch, void *g_kv_scratch, void *g_attn_out, void *g_mlp_inter,
    void *g_z_scratch, void *g_beta_scratch, void *g_alpha_scratch,
    void *g_normalized, void *g_fa_partials, void *g_rope_inv_freq,
    YarnParams yp, int max_seq_len, int num_blocks,
    void *g_layer_outputs, int start_position, cudaStream_t stream)
{
    return prefill_naive_impl<0>(
        device_token_ids, S,
        embed_weight, final_norm_weight, layer_weights,
        fa_k_cache, fa_v_cache, dn_states, conv_bufs,
        hidden_buffer, g_residual,
        g_qkv_scratch, g_kv_scratch, g_attn_out, g_mlp_inter,
        g_z_scratch, g_beta_scratch, g_alpha_scratch,
        g_normalized, g_fa_partials, g_rope_inv_freq,
        yp, max_seq_len, num_blocks, g_layer_outputs, start_position, stream);
}

extern "C" cudaError_t launch_prefill_naive_27b(
    const int32_t *device_token_ids, int S,
    void *embed_weight, void *final_norm_weight, void *layer_weights,
    void *fa_k_cache, void *fa_v_cache, void *dn_states, void *conv_bufs,
    void *hidden_buffer, void *g_residual,
    void *g_qkv_scratch, void *g_kv_scratch, void *g_attn_out, void *g_mlp_inter,
    void *g_z_scratch, void *g_beta_scratch, void *g_alpha_scratch,
    void *g_normalized, void *g_fa_partials, void *g_rope_inv_freq,
    YarnParams yp, int max_seq_len, int num_blocks,
    void *g_layer_outputs, int start_position, cudaStream_t stream)
{
    return prefill_naive_impl<1>(
        device_token_ids, S,
        embed_weight, final_norm_weight, layer_weights,
        fa_k_cache, fa_v_cache, dn_states, conv_bufs,
        hidden_buffer, g_residual,
        g_qkv_scratch, g_kv_scratch, g_attn_out, g_mlp_inter,
        g_z_scratch, g_beta_scratch, g_alpha_scratch,
        g_normalized, g_fa_partials, g_rope_inv_freq,
        yp, max_seq_len, num_blocks, g_layer_outputs, start_position, stream);
}

extern "C" cudaError_t launch_prefill_naive_0p8b_nvfp4(
    const int32_t *device_token_ids, int S,
    void *embed_weight, void *final_norm_weight, void *layer_weights,
    void *fa_k_cache, void *fa_v_cache, void *dn_states, void *conv_bufs,
    void *hidden_buffer, void *g_residual,
    void *g_qkv_scratch, void *g_kv_scratch, void *g_attn_out, void *g_mlp_inter,
    void *g_z_scratch, void *g_beta_scratch, void *g_alpha_scratch,
    void *g_normalized, void *g_fa_partials, void *g_rope_inv_freq,
    YarnParams yp, int max_seq_len, int num_blocks,
    void *g_layer_outputs, int start_position, cudaStream_t stream)
{
    return prefill_naive_impl<2>(
        device_token_ids, S,
        embed_weight, final_norm_weight, layer_weights,
        fa_k_cache, fa_v_cache, dn_states, conv_bufs,
        hidden_buffer, g_residual,
        g_qkv_scratch, g_kv_scratch, g_attn_out, g_mlp_inter,
        g_z_scratch, g_beta_scratch, g_alpha_scratch,
        g_normalized, g_fa_partials, g_rope_inv_freq,
        yp, max_seq_len, num_blocks, g_layer_outputs, start_position, stream);
}

extern "C" cudaError_t launch_prefill_naive_27b_nvfp4(
    const int32_t *device_token_ids, int S,
    void *embed_weight, void *final_norm_weight, void *layer_weights,
    void *fa_k_cache, void *fa_v_cache, void *dn_states, void *conv_bufs,
    void *hidden_buffer, void *g_residual,
    void *g_qkv_scratch, void *g_kv_scratch, void *g_attn_out, void *g_mlp_inter,
    void *g_z_scratch, void *g_beta_scratch, void *g_alpha_scratch,
    void *g_normalized, void *g_fa_partials, void *g_rope_inv_freq,
    YarnParams yp, int max_seq_len, int num_blocks,
    void *g_layer_outputs, int start_position, cudaStream_t stream)
{
    return prefill_naive_impl<3>(
        device_token_ids, S,
        embed_weight, final_norm_weight, layer_weights,
        fa_k_cache, fa_v_cache, dn_states, conv_bufs,
        hidden_buffer, g_residual,
        g_qkv_scratch, g_kv_scratch, g_attn_out, g_mlp_inter,
        g_z_scratch, g_beta_scratch, g_alpha_scratch,
        g_normalized, g_fa_partials, g_rope_inv_freq,
        yp, max_seq_len, num_blocks, g_layer_outputs, start_position, stream);
}

}  // namespace lucebox::qwen3x
