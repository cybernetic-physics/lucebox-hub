/**
 * Standalone tcgen05.mma bf16 → fp32 GEMM for NVIDIA Blackwell (sm_100).
 *
 * Shape: M=128, N=256, K=64 bf16. One MMA shape, multiple K-steps.
 * Using 128B swizzle mode because K=64 bf16 = 128 bytes per row, which
 * matches 128B swizzle granularity natively — row-major loads work.
 *
 * Reference descriptor pattern from https://gau-nernst.github.io/tcgen05/ :
 *   SBO = 8 * 128 (stride between 8×128B tiles)
 *   bit 46  = 1  (leading-dim)
 *   bits 61-63 = 2  (128B swizzle mode)
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

constexpr int M = 128;
constexpr int N = 256;
constexpr int K = 64;           // bf16 elements; row width = 128 bytes (one 128B super-block)
constexpr int K_BYTES = K * 2;  // 128 bytes
constexpr int MMA_K = 16;       // bf16 K per tcgen05.mma call
constexpr int K_STEPS = K / MMA_K;   // 4

// desc_encode matches gau-nernst: 14-bit field, value >> 4.
__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4;
}

// make_desc: unswizzled base address + SBO stride + leading-dim flag + swizzle mode.
// For 128B swizzle, SBO = 8 * 128 = 1024 bytes.
__device__ __forceinline__ uint64_t make_desc(uint32_t smem_addr) {
    constexpr int SBO = 8 * 128;
    return desc_encode((uint64_t)smem_addr)
         | (desc_encode((uint64_t)SBO) << 32)
         | (1ULL << 46)        // leading-dim
         | (2ULL << 61);       // 128B swizzle
}

extern "C" __global__ void tcgen05_gemm_one_tile(
    const __nv_bfloat16 *__restrict__ A,   // [M=128, K=64]
    const __nv_bfloat16 *__restrict__ B,   // [N=256, K=64]
    float               *__restrict__ C)   // [M=128, N=256] fp32
{
    // 128B-aligned shared buffers: A 16 KB + B 32 KB = 48 KB, in dynamic shared
    // to stay under the PTXAS static-shared 48 KB compile-time limit.
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    __nv_bfloat16 *s_A = reinterpret_cast<__nv_bfloat16 *>(smem_raw);
    __nv_bfloat16 *s_B = reinterpret_cast<__nv_bfloat16 *>(smem_raw + M * K * sizeof(__nv_bfloat16));
    __shared__ uint32_t tmem_slot;
    __shared__ uint64_t mma_mbar;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    // --------- 1. Allocate TMEM ---------
    if (warp == 0) {
        uint32_t slot_smem = static_cast<uint32_t>(__cvta_generic_to_shared(&tmem_slot));
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
            :: "r"(slot_smem), "n"(N));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
    }

    // --------- 2. Init mbarrier ---------
    if (tid == 0) {
        uint32_t bar_smem = static_cast<uint32_t>(__cvta_generic_to_shared(&mma_mbar));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(bar_smem));
        asm volatile("fence.mbarrier_init.release.cluster;\n");
    }
    __syncthreads();
    const uint32_t taddr = tmem_slot;

    // --------- 3. Load A, B to shared via cp.async WITH 128B swizzle applied.
    // Raw row-major must be rearranged: for row r and 16-byte chunk index
    // chunk_idx, the swizzled physical chunk position is chunk_idx ^ (r & 7).
    // Each row has 128 bytes = 8 chunks of 16 bytes (= 8 bf16 elements).
    // So row r, bf16 offset c (where c is a multiple of 8):
    //   chunk = c >> 3
    //   swizzled_chunk = chunk ^ (r & 7)
    //   write to s_*[r * K + (swizzled_chunk << 3)]
    // 128B swizzle applies within each 128-byte super-block independently.
    // In bf16 units: 128 bytes = 64 bf16, 8 chunks of 8 bf16 each.
    auto swz_offset = [](int r, int c_bf16) -> int {
        int super_idx = c_bf16 >> 6;                 // which 128B super-block
        int local_c   = c_bf16 & 63;                 // bf16 offset within super
        int local_chunk = local_c >> 3;              // 8-bf16-wide chunk (0..7)
        int sw_chunk   = local_chunk ^ (r & 7);      // XOR swizzle
        int sw_local_c = (sw_chunk << 3) | (local_c & 7);
        return (super_idx << 6) | sw_local_c;
    };

    // A: 128 rows × 8 chunks = 1024 loads.
    {
        for (int it = 0; it < 2; it++) {
            int i = tid + it * 512;
            int r = i / 8;
            int c = (i % 8) * 8;                      // bf16 offset: 0, 8, 16, ..., 56
            if (r < M) {
                int sw_c = swz_offset(r, c);
                __nv_bfloat16 *dst = s_A + r * K + sw_c;
                const __nv_bfloat16 *src = A + r * K + c;
                uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                             :: "r"(smem_addr), "l"(src));
            }
        }
    }
    // B: 256 rows × 8 chunks = 2048 loads.
    {
        for (int it = 0; it < 4; it++) {
            int i = tid + it * 512;
            int r = i / 8;
            int c = (i % 8) * 8;
            if (r < N) {
                int sw_c = swz_offset(r, c);
                __nv_bfloat16 *dst = s_B + r * K + sw_c;
                const __nv_bfloat16 *src = B + r * K + c;
                uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                             :: "r"(smem_addr), "l"(src));
            }
        }
    }
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    // --------- 4. Issue MMA chain by one thread ---------
    //   i_desc encoding for bf16 → fp32:
    //     [4]    dtype = 1 (FP32)
    //     [7]    atype = 1 (BF16)
    //     [10]   btype = 1 (BF16)
    //     [17:23] MMA_N >> 3   (N in units of 8)
    //     [24:28] MMA_M >> 4   (M in units of 16)
    constexpr uint32_t i_desc =
        (1U << 4)
      | (1U << 7)
      | (1U << 10)
      | ((uint32_t)(N >> 3) << 17)
      | ((uint32_t)(M >> 4) << 24);

    if (tid == 0) {
        uint32_t a_smem_base = static_cast<uint32_t>(__cvta_generic_to_shared(s_A));
        uint32_t b_smem_base = static_cast<uint32_t>(__cvta_generic_to_shared(s_B));
        uint32_t bar_smem = static_cast<uint32_t>(__cvta_generic_to_shared(&mma_mbar));

        // First MMA: enable_input_d=0  (reset accumulator).
        {
            uint64_t a_desc = make_desc(a_smem_base);
            uint64_t b_desc = make_desc(b_smem_base);
            asm volatile(
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 0;\n"
                :: "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(i_desc));
        }
        // Remaining K-steps accumulate into same TMEM accumulator.
        for (int ks = 1; ks < K_STEPS; ks++) {
            uint32_t a_smem = a_smem_base + (uint32_t)(ks * MMA_K * 2);
            uint32_t b_smem = b_smem_base + (uint32_t)(ks * MMA_K * 2);
            uint64_t a_desc = make_desc(a_smem);
            uint64_t b_desc = make_desc(b_smem);
            asm volatile(
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 1;\n"
                :: "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(i_desc));
        }
        asm volatile(
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
            :: "r"(bar_smem));
    }

    // --------- 5. Wait for MMA completion ---------
    if (tid == 0) {
        uint32_t bar_smem = static_cast<uint32_t>(__cvta_generic_to_shared(&mma_mbar));
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "waitLoop:\n"
            "  mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n"
            "  @p bra done;\n"
            "  bra.uni waitLoop;\n"
            "done:\n"
            "}\n"
            :: "r"(bar_smem));
    }
    __syncthreads();

    // --------- 6. Read back TMEM accumulator ---------
    // .32x32b requires row_offset to be 32-aligned. Each warp reads 32 rows
    // × some cols per .x8 call. From empirical testing, the (lane, reg)
    // layout is still under investigation — my best guess has ~25% of
    // positions correct. The first 10 matches cluster at rows 0, 1 of each
    // 32-row warp slice, suggesting reg 0 and reg 1 land at offsets 0 and 1
    // within the warp slice but regs 2..7 go somewhere else in TMEM.
    //
    // TODO: determine exact fragment layout. Candidates to try:
    //   - Non-contiguous row_map {0, 1, 8, 9, 16, 17, 24, 25} (tested, same match count)
    //   - Per-quad layout (tested, alignment error — .32x32b hard-requires 32-row stride)
    //   - Maybe layout depends on `.x8` vector width: .x16 might straightforwardly
    //     cover 32 rows × 8 cols in a simple [t][i] pattern.
    //
    // Leaving the original lane=row-stride layout so AT LEAST (0,0) and rows 0,1
    // match, until the layout is pinned down in a followup session.
    // Best known-decent: layout A (lane=row, reg=col consecutive). Hits ~13%
    // of positions (reg 0 correct at each col iter; regs 1..7 land in the
    // wrong TMEM col mapping). This is the state to continue debugging from
    // next session with the PTX ISA spec / CUTLASS TMEM layout source open.
    for (int n_off = 0; n_off < N; n_off += 8) {
        if (warp < 4) {
            int row_offset = warp * 32;
            uint32_t addr = taddr + (uint32_t)(row_offset << 16) + (uint32_t)n_off;
            float t0, t1, t2, t3, t4, t5, t6, t7;
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
                : "=f"(t0), "=f"(t1), "=f"(t2), "=f"(t3),
                  "=f"(t4), "=f"(t5), "=f"(t6), "=f"(t7)
                : "r"(addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;\n");
            int out_row = row_offset + lane;
            if (out_row < M) {
                float *dst = C + out_row * N + n_off;
                dst[0] = t0; dst[1] = t1; dst[2] = t2; dst[3] = t3;
                dst[4] = t4; dst[5] = t5; dst[6] = t6; dst[7] = t7;
            }
        }
    }
    __syncthreads();

    // --------- 7. Free TMEM ---------
    if (warp == 0) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
            :: "r"(taddr), "n"(N));
    }
}

extern "C" void launch_tcgen05_gemm_one_tile(
    const void *A, const void *B, void *C, cudaStream_t stream)
{
    // Shared usage: s_A 16 KB + s_B 32 KB + a few bytes for slots = ~48 KB.
    // Opt into the higher per-block shared memory bucket.
    static bool cfg = false;
    if (!cfg) {
        cudaFuncSetAttribute((void *)tcgen05_gemm_one_tile,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
        cfg = true;
    }
    dim3 grid(1);
    dim3 block(512);   // 16 warps — still 4 warps handle the readback
    constexpr size_t smem_bytes = (M + N) * K * sizeof(__nv_bfloat16);
    tcgen05_gemm_one_tile<<<grid, block, smem_bytes, stream>>>(
        (const __nv_bfloat16 *)A, (const __nv_bfloat16 *)B, (float *)C);
}
