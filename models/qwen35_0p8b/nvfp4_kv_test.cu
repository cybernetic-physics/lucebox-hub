/**
 * Standalone test kernels for the NVFP4 KV cache format.
 *
 * Three kernels:
 *   1. kv_quant_roundtrip_kernel       -- bf16 -> packed fp4 + E4M3 -> bf16
 *      (and exposes the packed buffers so PyTorch can also inspect them)
 *   2. kv_quant_kernel                 -- bf16 -> packed fp4 + E4M3 only
 *   3. kv_qk_dot_kernel                -- Q (bf16) . K (NVFP4) -> float score
 *
 * Layouts (T = number of tokens, H = number of kv heads, D = head_dim = 256):
 *
 *   bf16:        [T, H, D]              bf16
 *   data:        [T, H, D/2]            uint8
 *   scales:      [T, H, D/16]           uint8
 *   q (for dot): [H, D]                 bf16   (single-token Q)
 *   scores:      [T, H]                 float
 *
 * Launch: grid (H, T), 32-thread (1 warp) block. Each warp handles one
 * (token, head). Coalesced 128-bit loads (8 bf16 / 32 bytes) per lane.
 */

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 1200

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "nvfp4_kv.cuh"

namespace {

constexpr int HEAD_DIM   = 256;
constexpr int DATA_BYTES = HEAD_DIM / 2;
constexpr int SCALE_BYTES = HEAD_DIM / nvfp4_kv::KV_GROUP_SIZE;

}  // namespace

// ---------------------------------------------------------------------------
// 1. Quantize only (used by the python op to produce the packed buffers).
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(32)
kv_quant_kernel(
    const __nv_bfloat16 *__restrict__ src,   // [T, H, D]
    uint8_t *__restrict__ data,              // [T, H, D/2]
    uint8_t *__restrict__ scales,            // [T, H, D/16]
    int T, int H)
{
    int t = blockIdx.x;
    int h = blockIdx.y;
    if (h >= H || t >= T) return;
    int lane = threadIdx.x;

    size_t bf16_off  = (size_t(t) * H + h) * HEAD_DIM;
    size_t data_off  = (size_t(t) * H + h) * DATA_BYTES;
    size_t scale_off = (size_t(t) * H + h) * SCALE_BYTES;

    nvfp4_kv::encode_row<HEAD_DIM>(
        src + bf16_off, data + data_off, scales + scale_off, lane);
}

// ---------------------------------------------------------------------------
// 2. Dequantize only.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(32)
kv_dequant_kernel(
    const uint8_t *__restrict__ data,        // [T, H, D/2]
    const uint8_t *__restrict__ scales,      // [T, H, D/16]
    __nv_bfloat16 *__restrict__ dst,         // [T, H, D]
    int T, int H)
{
    int t = blockIdx.x;
    int h = blockIdx.y;
    if (h >= H || t >= T) return;
    int lane = threadIdx.x;

    size_t bf16_off  = (size_t(t) * H + h) * HEAD_DIM;
    size_t data_off  = (size_t(t) * H + h) * DATA_BYTES;
    size_t scale_off = (size_t(t) * H + h) * SCALE_BYTES;

    nvfp4_kv::decode_row_bf16<HEAD_DIM>(
        data + data_off, scales + scale_off, dst + bf16_off, lane);
}

// ---------------------------------------------------------------------------
// 3. Q . K dot product, K loaded from NVFP4 packed.
//   q:      [Q_H, D] bf16   (single-token Q, all query heads of one layer)
//   k_data: [T, H, D/2]
//   scales: [T, H, D/16]
//   scores: [Q_H, T] float
//   gqa:    Q_H / H  (query heads per kv head)
//
// Each warp computes one (token, query_head) score. We re-broadcast the
// kv-head index from q_head / gqa.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(32)
kv_qk_dot_kernel(
    const __nv_bfloat16 *__restrict__ q,     // [Q_H, D]
    const uint8_t *__restrict__ k_data,      // [T, H, D/2]
    const uint8_t *__restrict__ k_scales,    // [T, H, D/16]
    float *__restrict__ scores,              // [Q_H, T]
    int T, int H, int Q_H)
{
    int t  = blockIdx.x;
    int qh = blockIdx.y;
    if (qh >= Q_H || t >= T) return;
    int lane = threadIdx.x;

    int gqa = Q_H / H;
    int h   = qh / gqa;

    constexpr int EPL = HEAD_DIM / 32;  // 8
    float q_local[EPL];
    const __nv_bfloat16 *q_row = q + qh * HEAD_DIM;
    #pragma unroll
    for (int e = 0; e < EPL; ++e) {
        q_local[e] = __bfloat162float(q_row[lane * EPL + e]);
    }

    size_t data_off  = (size_t(t) * H + h) * DATA_BYTES;
    size_t scale_off = (size_t(t) * H + h) * SCALE_BYTES;

    float dot = nvfp4_kv::dot_q_k_nvfp4<HEAD_DIM>(
        q_local, k_data + data_off, k_scales + scale_off, lane);

    if (lane == 0) {
        scores[size_t(qh) * T + t] = dot;
    }
}

// ---------------------------------------------------------------------------
// 4. Fused softmax attention over NVFP4 K and V.
//
// Mirrors the FA inner loop in kernel.cu (decode path, single Q token):
//   for each pos in [0, T):
//       score = (Q . K[pos]) * attn_scale
//       online softmax
//       out_acc[:] = exp_diff * out_acc[:] + softmax_wt * V[pos]
//   out[:] = out_acc / sum_exp
//
//   q:        [Q_H, D] bf16
//   k_data:   [T, H, D/2]   uint8
//   k_scales: [T, H, D/16]  uint8
//   v_data:   [T, H, D/2]   uint8
//   v_scales: [T, H, D/16]  uint8
//   out:      [Q_H, D]      bf16
//   lse_out:  [Q_H]         float  (optional, set NULL to skip)
//
// One warp per query head. Used by the python test suite to validate
// end-to-end attention agreement with a bf16 reference.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(32)
kv_attention_kernel(
    const __nv_bfloat16 *__restrict__ q,
    const uint8_t *__restrict__ k_data,
    const uint8_t *__restrict__ k_scales,
    const uint8_t *__restrict__ v_data,
    const uint8_t *__restrict__ v_scales,
    __nv_bfloat16 *__restrict__ out,
    float *__restrict__ lse_out,
    int T, int H, int Q_H, float attn_scale)
{
    int qh = blockIdx.x;
    if (qh >= Q_H) return;
    int lane = threadIdx.x;

    int gqa = Q_H / H;
    int h   = qh / gqa;
    constexpr int EPL = HEAD_DIM / 32;

    float q_local[EPL];
    const __nv_bfloat16 *q_row = q + qh * HEAD_DIM;
    #pragma unroll
    for (int e = 0; e < EPL; ++e) {
        q_local[e] = __bfloat162float(q_row[lane * EPL + e]);
    }

    float max_score = -INFINITY;
    float sum_exp   = 0.0f;
    float out_acc[EPL];
    #pragma unroll
    for (int e = 0; e < EPL; ++e) out_acc[e] = 0.0f;

    for (int t = 0; t < T; ++t) {
        size_t off_d = (size_t(t) * H + h) * DATA_BYTES;
        size_t off_s = (size_t(t) * H + h) * SCALE_BYTES;

        float score = nvfp4_kv::dot_q_k_nvfp4<HEAD_DIM>(
            q_local, k_data + off_d, k_scales + off_s, lane) * attn_scale;

        float old_max = max_score;
        max_score = fmaxf(max_score, score);
        float exp_diff = __expf(old_max - max_score);   // == 1 on the first iter
        sum_exp = sum_exp * exp_diff + __expf(score - max_score);
        float wt = __expf(score - max_score);

        #pragma unroll
        for (int e = 0; e < EPL; ++e) out_acc[e] *= exp_diff;

        nvfp4_kv::accum_v_nvfp4<HEAD_DIM>(
            out_acc, wt, v_data + off_d, v_scales + off_s, lane);
    }

    float inv = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    __nv_bfloat16 *out_row = out + qh * HEAD_DIM;
    #pragma unroll
    for (int e = 0; e < EPL; ++e) {
        out_row[lane * EPL + e] = __float2bfloat16(out_acc[e] * inv);
    }
    if (lane == 0 && lse_out != nullptr) {
        lse_out[qh] = logf(fmaxf(sum_exp, 1e-30f)) + max_score;
    }
}

// ---------------------------------------------------------------------------
// 5. Split-K fused attention. Each block handles one (query_head, K-split)
//    pair and writes a partial (max, sum_exp, out_acc) to a workspace. A
//    second kernel reduces partials -> final softmax output and LSE.
//
//    Grid: (num_splits, Q_H).
//    partials layout: [Q_H, num_splits, HEAD_DIM + 2]   float
//                                              ^------ index [HEAD_DIM]   = max
//                                              ^------ index [HEAD_DIM+1] = sum_exp
//
//    Replaces the serial-T attention with SM-parallel work, hitting HBM
//    bandwidth for the K/V scan rather than warp single-thread latency.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(32)
kv_attention_split_kernel(
    const __nv_bfloat16 *__restrict__ q,
    const uint8_t *__restrict__ k_data,
    const uint8_t *__restrict__ k_scales,
    const uint8_t *__restrict__ v_data,
    const uint8_t *__restrict__ v_scales,
    float *__restrict__ partials,
    int T, int H, int Q_H, float attn_scale, int num_splits)
{
    int my_split = blockIdx.x;
    int qh       = blockIdx.y;
    if (qh >= Q_H || my_split >= num_splits) return;
    int lane = threadIdx.x;

    int gqa = Q_H / H;
    int h   = qh / gqa;
    constexpr int EPL = HEAD_DIM / 32;
    constexpr int PARTIAL_STRIDE = HEAD_DIM + 2;

    int per_split = (T + num_splits - 1) / num_splits;
    int t_start = my_split * per_split;
    int t_end   = min(t_start + per_split, T);

    float q_local[EPL];
    const __nv_bfloat16 *q_row = q + qh * HEAD_DIM;
    #pragma unroll
    for (int e = 0; e < EPL; ++e) {
        q_local[e] = __bfloat162float(q_row[lane * EPL + e]);
    }

    float max_score = -INFINITY;
    float sum_exp   = 0.0f;
    float out_acc[EPL];
    #pragma unroll
    for (int e = 0; e < EPL; ++e) out_acc[e] = 0.0f;

    for (int t = t_start; t < t_end; ++t) {
        size_t off_d = (size_t(t) * H + h) * DATA_BYTES;
        size_t off_s = (size_t(t) * H + h) * SCALE_BYTES;
        float score = nvfp4_kv::dot_q_k_nvfp4<HEAD_DIM>(
            q_local, k_data + off_d, k_scales + off_s, lane) * attn_scale;
        float old_max = max_score;
        max_score = fmaxf(max_score, score);
        float exp_diff = __expf(old_max - max_score);
        sum_exp = sum_exp * exp_diff + __expf(score - max_score);
        float wt = __expf(score - max_score);
        #pragma unroll
        for (int e = 0; e < EPL; ++e) out_acc[e] *= exp_diff;
        nvfp4_kv::accum_v_nvfp4<HEAD_DIM>(
            out_acc, wt, v_data + off_d, v_scales + off_s, lane);
    }

    // If this split saw no positions (e.g. T < num_splits), mark inactive.
    if (t_start >= t_end) max_score = -INFINITY;

    float *po = partials + (size_t(qh) * num_splits + my_split) * PARTIAL_STRIDE;
    #pragma unroll
    for (int e = 0; e < EPL; ++e) {
        po[lane * EPL + e] = out_acc[e];
    }
    if (lane == 0) {
        po[HEAD_DIM]     = max_score;
        po[HEAD_DIM + 1] = sum_exp;
    }
}

__global__ void __launch_bounds__(32)
kv_attention_reduce_kernel(
    const float *__restrict__ partials,
    __nv_bfloat16 *__restrict__ out,
    float *__restrict__ lse_out,
    int Q_H, int num_splits)
{
    int qh = blockIdx.x;
    if (qh >= Q_H) return;
    int lane = threadIdx.x;
    constexpr int EPL = HEAD_DIM / 32;
    constexpr int PARTIAL_STRIDE = HEAD_DIM + 2;

    float global_max = -INFINITY;
    for (int s = 0; s < num_splits; ++s) {
        float m = partials[(size_t(qh) * num_splits + s) * PARTIAL_STRIDE + HEAD_DIM];
        if (m > global_max) global_max = m;
    }
    float global_sum = 0.0f;
    float global_out[EPL];
    #pragma unroll
    for (int e = 0; e < EPL; ++e) global_out[e] = 0.0f;

    for (int s = 0; s < num_splits; ++s) {
        const float *po = partials + (size_t(qh) * num_splits + s) * PARTIAL_STRIDE;
        float m  = po[HEAD_DIM];
        if (m == -INFINITY) continue;   // empty split
        float ss = po[HEAD_DIM + 1];
        float w  = __expf(m - global_max);
        global_sum += ss * w;
        #pragma unroll
        for (int e = 0; e < EPL; ++e) global_out[e] += po[lane * EPL + e] * w;
    }
    float inv = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
    __nv_bfloat16 *out_row = out + qh * HEAD_DIM;
    #pragma unroll
    for (int e = 0; e < EPL; ++e) {
        out_row[lane * EPL + e] = __float2bfloat16(global_out[e] * inv);
    }
    if (lane == 0 && lse_out != nullptr) {
        lse_out[qh] = logf(fmaxf(global_sum, 1e-30f)) + global_max;
    }
}

// ---------------------------------------------------------------------------
// Host-side launchers (called from the torch binding TU).
// ---------------------------------------------------------------------------
extern "C" {

void launch_kv_quant(
    const void *src_bf16, void *data, void *scales,
    int T, int H, cudaStream_t stream)
{
    if (T == 0 || H == 0) return;
    dim3 grid(T, H);
    kv_quant_kernel<<<grid, 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(src_bf16),
        reinterpret_cast<uint8_t *>(data),
        reinterpret_cast<uint8_t *>(scales),
        T, H);
}

void launch_kv_dequant(
    const void *data, const void *scales, void *dst_bf16,
    int T, int H, cudaStream_t stream)
{
    if (T == 0 || H == 0) return;
    dim3 grid(T, H);
    kv_dequant_kernel<<<grid, 32, 0, stream>>>(
        reinterpret_cast<const uint8_t *>(data),
        reinterpret_cast<const uint8_t *>(scales),
        reinterpret_cast<__nv_bfloat16 *>(dst_bf16),
        T, H);
}

void launch_kv_qk_dot(
    const void *q_bf16, const void *k_data, const void *k_scales, void *scores,
    int T, int H, int Q_H, cudaStream_t stream)
{
    if (T == 0 || H == 0 || Q_H == 0) return;
    dim3 grid(T, Q_H);
    kv_qk_dot_kernel<<<grid, 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(q_bf16),
        reinterpret_cast<const uint8_t *>(k_data),
        reinterpret_cast<const uint8_t *>(k_scales),
        reinterpret_cast<float *>(scores),
        T, H, Q_H);
}

void launch_kv_attention(
    const void *q_bf16,
    const void *k_data, const void *k_scales,
    const void *v_data, const void *v_scales,
    void *out_bf16, void *lse_out,
    int T, int H, int Q_H, float attn_scale, cudaStream_t stream)
{
    if (Q_H == 0 || H == 0) return;
    dim3 grid(Q_H);
    kv_attention_kernel<<<grid, 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(q_bf16),
        reinterpret_cast<const uint8_t *>(k_data),
        reinterpret_cast<const uint8_t *>(k_scales),
        reinterpret_cast<const uint8_t *>(v_data),
        reinterpret_cast<const uint8_t *>(v_scales),
        reinterpret_cast<__nv_bfloat16 *>(out_bf16),
        reinterpret_cast<float *>(lse_out),
        T, H, Q_H, attn_scale);
}

void launch_kv_attention_split(
    const void *q_bf16,
    const void *k_data, const void *k_scales,
    const void *v_data, const void *v_scales,
    void *out_bf16, void *lse_out, void *partials,
    int T, int H, int Q_H, float attn_scale, int num_splits,
    cudaStream_t stream)
{
    if (Q_H == 0 || H == 0 || num_splits <= 0) return;
    dim3 grid_split(num_splits, Q_H);
    kv_attention_split_kernel<<<grid_split, 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(q_bf16),
        reinterpret_cast<const uint8_t *>(k_data),
        reinterpret_cast<const uint8_t *>(k_scales),
        reinterpret_cast<const uint8_t *>(v_data),
        reinterpret_cast<const uint8_t *>(v_scales),
        reinterpret_cast<float *>(partials),
        T, H, Q_H, attn_scale, num_splits);
    dim3 grid_reduce(Q_H);
    kv_attention_reduce_kernel<<<grid_reduce, 32, 0, stream>>>(
        reinterpret_cast<const float *>(partials),
        reinterpret_cast<__nv_bfloat16 *>(out_bf16),
        reinterpret_cast<float *>(lse_out),
        Q_H, num_splits);
}

}  // extern "C"

#else   // sm < 120 -- emit stubs so the host-side aten binding still links.

#include <cuda_runtime.h>
extern "C" {
void launch_kv_quant(const void *, void *, void *, int, int, cudaStream_t) {}
void launch_kv_dequant(const void *, const void *, void *, int, int, cudaStream_t) {}
void launch_kv_qk_dot(const void *, const void *, const void *, void *,
                      int, int, int, cudaStream_t) {}
void launch_kv_attention(const void *, const void *, const void *,
                         const void *, const void *, void *, void *,
                         int, int, int, float, cudaStream_t) {}
void launch_kv_attention_split(const void *, const void *, const void *,
                               const void *, const void *, void *, void *, void *,
                               int, int, int, float, int, cudaStream_t) {}
}

#endif  // __CUDA_ARCH__ >= 1200
