/**
 * Shared device-side helpers for the Qwen3.x templated kernels.
 * Math intrinsics, load helpers, warp reductions. Header-only.
 */
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdint.h>

namespace lucebox::qwen3x {

namespace cg = cooperative_groups;

// ---- math intrinsics ----
__device__ __forceinline__ float fast_exp(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;"
                 : "=f"(y) : "f"(x * 1.44269504088896340736f));
    return y;
}
__device__ __forceinline__ float fast_sigmoid(float x) {
    float y;
    asm volatile("rcp.approx.ftz.f32 %0, %1;"
                 : "=f"(y) : "f"(1.0f + fast_exp(-x)));
    return y;
}
__device__ __forceinline__ float fast_silu(float x) { return x * fast_sigmoid(x); }

// ---- 128-bit coalesced load via L1-no-allocate ----
__device__ __forceinline__ uint4 load_128bit(const uint4 *ptr) {
    uint4 out;
    asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w) : "l"(ptr));
    return out;
}

__device__ __forceinline__ uint32_t load_32bit(const uint32_t *ptr) {
    uint32_t out;
    asm volatile("ld.global.L1::no_allocate.b32 %0, [%1];" : "=r"(out) : "l"(ptr));
    return out;
}

// ---- 8x bf16 . 8x bf16 dot in fp32 acc ----
__device__ __forceinline__ float dot8_bf16(const uint4 &w_u4, const __nv_bfloat16 *act) {
    const __nv_bfloat16 *w = reinterpret_cast<const __nv_bfloat16 *>(&w_u4);
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) sum += __bfloat162float(w[i]) * __bfloat162float(act[i]);
    return sum;
}

// ---- warp reductions ----
__device__ __forceinline__ float warp_reduce_sum_x(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, v, o);
        v = fmaxf(v, other);
    }
    return v;
}

// ---- grid-cooperative sync (decode megakernel needs this) ----
struct AtomicGridSync {
    __device__ __forceinline__ void sync() { cg::this_grid().sync(); }
};

}  // namespace lucebox::qwen3x
