// cuDNN FlashAttention-2 forward, called from launch_prefill_bf16's FA
// layer body. Replaces the old pf_causal_attn (O(S²) walk-keys-per-query
// CUDA kernel) with a thin C++ wrapper around
// at::_scaled_dot_product_flash_attention — the same cuDNN FA-2 kernel
// fa_bwd_flash.py already uses for the backward.
//
// Why this isn't in prefill.cu: aten/torch headers don't compile under
// nvcc cleanly. Keeping this in a plain .cpp file decouples the
// concerns — prefill.cu does CUDA orchestration, this file does aten
// dispatch. They communicate through a flat extern "C" interface.
//
// Scope: handles GQA (FA_Q_HEADS=8 over FA_KV_HEADS=2), causal mask, and
// the Qwen3-Next "gated query" layout where each query head has both a
// query and a sigmoid-gated multiplier interleaved as
// `[Q[D], Gate[D]]` per (position, head).

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Mirrors models/qwen35_0p8b/model.py constants. Kept local because
// neither prefill.cu nor torch_bindings.cpp exposes a shared header.
namespace fa_consts {
constexpr int FA_Q_HEADS  = 8;
constexpr int FA_KV_HEADS = 2;
constexpr int FA_HEAD_DIM = 256;
constexpr int FA_GQA      = FA_Q_HEADS / FA_KV_HEADS;     // 4
constexpr int FA_Q_SIZE       = FA_Q_HEADS  * FA_HEAD_DIM;     // 2048
constexpr int FA_QPROJ_SIZE   = FA_Q_SIZE   * 2;               // 4096 (Q + Gate)
constexpr int FA_KV_SIZE      = FA_KV_HEADS * FA_HEAD_DIM;     // 512
}  // namespace fa_consts

// ---------------------------------------------------------------------------
// launch_fa_attn_aten
//
// Inputs (all device pointers, bf16 unless noted):
//   q_proj          : [S, FA_Q_HEADS, 2, FA_HEAD_DIM]
//                     = [S, FA_QPROJ_SIZE]
//                     pos×head innermost stride: [Q[D], Gate[D]]
//                     (matches what pf_qk_norm_rope writes for the
//                      query side)
//   k_cache         : [FA_KV_HEADS, max_seq, FA_HEAD_DIM]
//   v_cache         : same shape as k_cache
//                     (we only read positions 0..S-1; the cache may be
//                      longer for future decode-step appends)
//   out             : [S, FA_Q_SIZE] — receives sigmoid(Gate) * O
//   S               : current sequence length
//   max_seq         : KV cache row count (stride into k_cache/v_cache)
//   stream          : CUDA stream the call runs on
//
// Notes:
//   - cuDNN FA-2 supports GQA natively when Hq == Hk × group_size; we
//     repeat-interleave K/V to FA_Q_HEADS heads so this works on torch
//     versions whose at:: API hasn't yet exposed the GQA flag. The
//     interleave is a kernel-driven copy on the same stream, ~50 µs at
//     S=8K, dwarfed by the compute savings.
//   - The sigmoid-gate multiplication is done in a small fused kernel
//     defined in prefill.cu (`pf_apply_gate`). This file just calls it
//     after the FA forward via a function pointer.
extern "C" void pf_apply_gate_launch(
    const __nv_bfloat16 *attn_out,
    const __nv_bfloat16 *q_proj,
    __nv_bfloat16 *out,
    int S,
    cudaStream_t stream);

extern "C" void launch_fa_attn_aten(
    const __nv_bfloat16 *q_proj,
    const __nv_bfloat16 *k_cache,
    const __nv_bfloat16 *v_cache,
    __nv_bfloat16 *out,
    int S,
    int max_seq,
    cudaStream_t stream)
{
    using namespace fa_consts;
    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);

    // Build a Q view that strips the gate. q_proj is laid out as
    // [S, FA_Q_HEADS, 2, FA_HEAD_DIM] (each head has Q + Gate). Slicing
    // dim 2 to 0 gives [S, FA_Q_HEADS, FA_HEAD_DIM]. Permute to
    // [1, FA_Q_HEADS, S, FA_HEAD_DIM] and contiguous() so cuDNN sees
    // a clean tensor (the strided view from select+permute won't match
    // cuDNN's contiguous-row expectation).
    auto q_packed = torch::from_blob(
        const_cast<__nv_bfloat16 *>(q_proj),
        {1, S, FA_Q_HEADS, 2, FA_HEAD_DIM}, opts_bf16);
    auto q = q_packed.select(3, 0)                      // [1, S, FA_Q_HEADS, FA_HEAD_DIM]
                     .permute({0, 2, 1, 3})             // [1, FA_Q_HEADS, S, FA_HEAD_DIM]
                     .contiguous();

    // K/V cache views, sliced to current S and reshaped to [B=1, H, S, D].
    auto k_full = torch::from_blob(
        const_cast<__nv_bfloat16 *>(k_cache),
        {1, FA_KV_HEADS, max_seq, FA_HEAD_DIM}, opts_bf16);
    auto v_full = torch::from_blob(
        const_cast<__nv_bfloat16 *>(v_cache),
        {1, FA_KV_HEADS, max_seq, FA_HEAD_DIM}, opts_bf16);
    auto k = k_full.narrow(2, 0, S).contiguous();
    auto v = v_full.narrow(2, 0, S).contiguous();

    // GQA expansion — repeat each KV head FA_GQA times to match Hq.
    // (The native enable_gqa flag isn't always exposed via the
    // _scaled_dot_product_flash_attention C++ entry on this torch.)
    auto k_e = k.repeat_interleave(FA_GQA, /*dim=*/1);  // [1, Hq, S, D]
    auto v_e = v.repeat_interleave(FA_GQA, /*dim=*/1);

    // cuDNN FA-2 forward via the same op fa_bwd_flash uses on the bwd side.
    double scale = 1.0 / std::sqrt((double)FA_HEAD_DIM);
    auto r = at::_scaled_dot_product_flash_attention(
        q, k_e, v_e,
        /*dropout_p=*/0.0,
        /*is_causal=*/true,
        /*return_debug_mask=*/false,
        /*scale=*/c10::optional<double>(scale));
    auto O_bhsd = std::get<0>(r);  // [1, FA_Q_HEADS, S, FA_HEAD_DIM]

    // Reshape O to [S, FA_Q_HEADS, FA_HEAD_DIM] = [S, FA_Q_SIZE] flat.
    auto O_shd = O_bhsd.permute({0, 2, 1, 3}).contiguous();
    auto O_flat = O_shd.view({S, FA_Q_SIZE});

    // Apply sigmoid-gate from the second half of q_proj and write to
    // `out`. The kernel reads gate at the per-(pos, head) slot
    // q_proj[pos][head][1][:].
    pf_apply_gate_launch(
        (const __nv_bfloat16 *)O_flat.data_ptr(),
        q_proj,
        out,
        S,
        stream);
}
