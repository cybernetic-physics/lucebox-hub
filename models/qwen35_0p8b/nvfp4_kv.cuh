/**
 * NVFP4 KV cache format for the Qwen3.5-0.8B megakernel on GB10 (sm_121a).
 *
 * Storage layout (per K or V cache):
 *   data:   [n_layers, n_kv_heads, max_seq, FA_HEAD_DIM / 2]   uint8 (packed FP4x2)
 *   scales: [n_layers, n_kv_heads, max_seq, FA_HEAD_DIM / 16]  uint8 (E4M3 FP8)
 *
 * Block size: 16 elements along head_dim. Matches NVIDIA NVFP4 spec (4-bit E2M1
 * mantissa, FP8 E4M3 per-block scale). FA_HEAD_DIM=256 -> 16 scale bytes per
 * (head, position), 128 data bytes per (head, position).
 *
 * Memory footprint vs bf16 baseline at the default config
 * (n_layers=6, n_kv_heads=2, max_seq=65536, head_dim=256):
 *   bf16:  6 * 2 * 65536 * 256 * 2 = 384 MB per cache    -> K+V = 768 MB
 *   nvfp4: 6 * 2 * 65536 * (128 + 16) = 108 MB per cache -> K+V = 216 MB  (3.55x)
 *
 * Encode path:
 *   absmax over 16 bf16 elements -> scale_f = absmax / 6.0 (max NVFP4 magnitude)
 *   -> E4M3 round (this is the stored scale)
 *   -> dequant E4M3 -> half -> float (use the decoded scale's inverse for quant
 *      so encode+decode is bit-exact when the absmax is representable)
 *   -> for each elem: code = float2_to_fp4x2((x / decoded_scale, y / decoded_scale))
 *
 * Decode path:
 *   load packed FP4x2 byte -> two E2M1 codes -> FP4_E2M1_LUT lookup * scale
 *
 * Arch gate: the FP4/FP8 conversion intrinsics require sm_120+ (`__CUDA_ARCH__
 * >= 1200`). Including this header from a sm_86 device pass produces an empty
 * cubin — same pattern as kernel_gb10_nvfp4.cu.
 */

#pragma once

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 1200

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

namespace nvfp4_kv {

constexpr int KV_GROUP_SIZE     = 16;   // elements per FP8 scale
constexpr float NVFP4_ABSMAX    = 6.0f; // max representable |E2M1| value
constexpr float MIN_SCALE       = 1.0f / float(1 << 28);

// FP4_E2M1 LUT: {±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
__device__ __constant__ float FP4_E2M1_LUT_KV[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
   -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
};

// ---------------------------------------------------------------------------
// Scale helpers
// ---------------------------------------------------------------------------

// Convert a positive float to E4M3 FP8 storage byte (saturating).
__device__ __forceinline__ uint8_t f32_to_e4m3(float v) {
    __nv_fp8_storage_t s = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    return static_cast<uint8_t>(s);
}

// Decode an E4M3 byte back to float (the exact value the encoder must use as
// the divisor for round-trip cleanliness).
__device__ __forceinline__ float e4m3_to_f32(uint8_t code) {
    __half_raw h = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(code), __NV_E4M3);
    return __half2float(*reinterpret_cast<__half *>(&h));
}

// Convert two floats (already scaled to the [-6, 6] FP4 range) to a packed
// FP4x2 byte: low nibble = lo, high nibble = hi.
__device__ __forceinline__ uint8_t f2_to_fp4x2(float lo, float hi) {
    float2 pair = make_float2(lo, hi);
    __nv_fp4x2_storage_t p = __nv_cvt_float2_to_fp4x2(pair, __NV_E2M1, cudaRoundNearest);
    return static_cast<uint8_t>(p);
}

// ---------------------------------------------------------------------------
// Warp-cooperative encode of one (head_dim=256) row.
//   src_bf16:   [FA_HEAD_DIM] bf16
//   out_data:   [FA_HEAD_DIM/2] uint8       (packed FP4x2)
//   out_scales: [FA_HEAD_DIM/16] uint8      (E4M3 FP8)
//
// Threading: one warp does the whole row. Each lane handles 8 elems
// (head_dim/32 = 8). The 16-elem block boundaries land on lane pairs (2 lanes
// per scale group). Lane n stores element 8n .. 8n+7, which spans:
//   - n=0..1   -> scale block 0      (elems 0-15)
//   - n=2..3   -> scale block 1      (elems 16-31)
//   - ...
//   - n=30..31 -> scale block 15     (elems 240-255)
// So lane n is in scale block (n >> 1), and is "low half" if n is even.
// ---------------------------------------------------------------------------
template<int HEAD_DIM = 256>
__device__ __forceinline__ void encode_row(
    const __nv_bfloat16 *__restrict__ src_bf16,
    uint8_t *__restrict__ out_data,
    uint8_t *__restrict__ out_scales,
    int lane)
{
    static_assert(HEAD_DIM == 256, "encode_row currently assumes head_dim=256");
    constexpr int EPL = HEAD_DIM / 32;            // = 8
    constexpr int N_SCALES = HEAD_DIM / KV_GROUP_SIZE;  // = 16

    // Load 8 bf16 elements as 4 bf162.
    __nv_bfloat162 vals[EPL / 2];
    const __nv_bfloat162 *src2 = reinterpret_cast<const __nv_bfloat162 *>(src_bf16 + lane * EPL);
    #pragma unroll
    for (int i = 0; i < EPL / 2; ++i) vals[i] = src2[i];

    // Per-lane absmax over the 8 elements.
    float my_absmax = 0.0f;
    #pragma unroll
    for (int i = 0; i < EPL / 2; ++i) {
        float2 f = __bfloat1622float2(vals[i]);
        my_absmax = fmaxf(my_absmax, fmaxf(fabsf(f.x), fabsf(f.y)));
    }

    // Each scale group covers 2 lanes -> reduce within lane pair via xor 1.
    float partner = __shfl_xor_sync(0xffffffff, my_absmax, 1);
    float group_absmax = fmaxf(my_absmax, partner);
    float scale_f = fmaxf(group_absmax * (1.0f / NVFP4_ABSMAX), MIN_SCALE);

    // E4M3 round of the scale. Both lanes in the pair end up with the same
    // E4M3 byte because they reduced from the same absmax. The even lane in
    // each pair is responsible for writing.
    uint8_t scale_e4m3 = f32_to_e4m3(scale_f);
    float   decoded    = e4m3_to_f32(scale_e4m3);
    float   inv_scale  = (decoded > 0.0f) ? (1.0f / decoded) : 0.0f;

    int scale_idx = lane >> 1;
    if ((lane & 1) == 0 && scale_idx < N_SCALES) {
        out_scales[scale_idx] = scale_e4m3;
    }

    // Quantize: pack 8 elems into 4 FP4x2 bytes (uint32_t).
    uint32_t packed = 0;
    #pragma unroll
    for (int i = 0; i < EPL / 2; ++i) {
        float2 f = __bfloat1622float2(vals[i]);
        f.x *= inv_scale;
        f.y *= inv_scale;
        uint8_t byte = f2_to_fp4x2(f.x, f.y);
        packed |= static_cast<uint32_t>(byte) << (i * 8);
    }
    // Write 4 bytes per lane = 32 bytes per scale group; lanes 0..1 of a
    // scale group write contiguously.
    uint32_t *out_u32 = reinterpret_cast<uint32_t *>(out_data + lane * (EPL / 2));
    *out_u32 = packed;
}

// ---------------------------------------------------------------------------
// Warp-cooperative dequant to bf16 (round-trip helper / fallback dequant).
//   src_data:   [FA_HEAD_DIM/2] uint8
//   src_scales: [FA_HEAD_DIM/16] uint8
//   out_bf16:   [FA_HEAD_DIM] bf16
// ---------------------------------------------------------------------------
template<int HEAD_DIM = 256>
__device__ __forceinline__ void decode_row_bf16(
    const uint8_t *__restrict__ src_data,
    const uint8_t *__restrict__ src_scales,
    __nv_bfloat16 *__restrict__ out_bf16,
    int lane)
{
    static_assert(HEAD_DIM == 256, "decode_row_bf16 currently assumes head_dim=256");
    constexpr int EPL = HEAD_DIM / 32;  // 8 elems per lane

    int scale_idx = lane >> 1;
    float scale = e4m3_to_f32(src_scales[scale_idx]);

    uint32_t packed = *reinterpret_cast<const uint32_t *>(src_data + lane * (EPL / 2));
    __nv_bfloat16 *dst = out_bf16 + lane * EPL;

    #pragma unroll
    for (int i = 0; i < EPL / 2; ++i) {
        uint8_t byte = (packed >> (i * 8)) & 0xff;
        float lo = FP4_E2M1_LUT_KV[byte & 0xf] * scale;
        float hi = FP4_E2M1_LUT_KV[byte >> 4]  * scale;
        dst[i * 2 + 0] = __float2bfloat16(lo);
        dst[i * 2 + 1] = __float2bfloat16(hi);
    }
}

// ---------------------------------------------------------------------------
// Warp-cooperative scalar dot product: q (bf16, 256) . k (NVFP4, 256) -> float
// Each lane holds 8 elements of q in registers (q_local[EPL]).
//
// Returns the lane-0 broadcast scalar dot product across the warp.
// ---------------------------------------------------------------------------
template<int HEAD_DIM = 256>
__device__ __forceinline__ float dot_q_k_nvfp4(
    const float *__restrict__ q_local,           // [EPL] f32 per lane
    const uint8_t *__restrict__ k_data,          // [HEAD_DIM/2] packed
    const uint8_t *__restrict__ k_scales,        // [HEAD_DIM/16] E4M3
    int lane)
{
    constexpr int EPL = HEAD_DIM / 32;  // 8

    int scale_idx = lane >> 1;
    float scale = e4m3_to_f32(k_scales[scale_idx]);

    uint32_t packed = *reinterpret_cast<const uint32_t *>(k_data + lane * (EPL / 2));

    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < EPL / 2; ++i) {
        uint8_t byte = (packed >> (i * 8)) & 0xff;
        sum += FP4_E2M1_LUT_KV[byte & 0xf] * q_local[i * 2 + 0];
        sum += FP4_E2M1_LUT_KV[byte >> 4]  * q_local[i * 2 + 1];
    }
    sum *= scale;

    // Warp reduce.
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    return sum;
}

// ---------------------------------------------------------------------------
// Warp-cooperative weighted accumulate: out[e] += wt * v[e]
// Each lane accumulates EPL contiguous V elements into out_local[EPL].
// ---------------------------------------------------------------------------
template<int HEAD_DIM = 256>
__device__ __forceinline__ void accum_v_nvfp4(
    float *out_local,                            // [EPL] f32 per lane (in/out)
    float wt,
    const uint8_t *__restrict__ v_data,
    const uint8_t *__restrict__ v_scales,
    int lane)
{
    constexpr int EPL = HEAD_DIM / 32;  // 8

    int scale_idx = lane >> 1;
    float scale = e4m3_to_f32(v_scales[scale_idx]);

    uint32_t packed = *reinterpret_cast<const uint32_t *>(v_data + lane * (EPL / 2));

    #pragma unroll
    for (int i = 0; i < EPL / 2; ++i) {
        uint8_t byte = (packed >> (i * 8)) & 0xff;
        out_local[i * 2 + 0] += wt * FP4_E2M1_LUT_KV[byte & 0xf] * scale;
        out_local[i * 2 + 1] += wt * FP4_E2M1_LUT_KV[byte >> 4]  * scale;
    }
}

}  // namespace nvfp4_kv

#endif  // __CUDA_ARCH__ >= 1200
