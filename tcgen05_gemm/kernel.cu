/**
 * Standalone tcgen05.mma bf16 → fp32 GEMM for NVIDIA Blackwell (sm_100).
 *
 * Scope of this first cut: ONE MMA tile of shape [M=128, N=256, K=16], no
 * K-loop, no TMA — cp.async into a shared memory buffer, then tcgen05.mma
 * into a TMEM accumulator, then tcgen05.ld back to registers and write
 * out to global fp32. The goal is to get the instruction sequence right
 * before scaling up to a tiled GEMM.
 *
 * Reference: https://gau-nernst.github.io/tcgen05/
 *
 * WARNING: raw tcgen05 PTX is only valid on sm_100+ (Blackwell datacenter).
 * Do not compile for sm_90 or earlier.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

constexpr int M = 128;
constexpr int N = 256;
constexpr int K = 16;   // one MMA K-step; bf16 → K_bytes = 32

// --------- Shared-memory descriptor encoder ---------
// tcgen05.mma reads A and B via 64-bit shared-memory descriptors. Layout
// per PTX spec: { base_addr, SBO, LBO, flags, swizzle }. We use the minimal
// form: unswizzled (mode 0), contiguous layout, leading dim = 1.
//
// Bits 0–13  : (base_addr >> 4)       - 16-byte-aligned start in shared memory
// Bits 16–29 : (LBO >> 4)              - stride between 8×... "matrix boxes"
// Bits 32–45 : (SBO >> 4)              - stride along K between boxes
// Bit 46     : leading-dim-major       - 1 for matrix-a row-major / matrix-b col-major
// Bits 61–63 : swizzle mode            - 0 = none (we use this)
__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4;
}

// Attempt 2: per CUTLASS sm_100 source (include/cute/arch/mma_sm100_desc.hpp),
// the 64-bit descriptor layout for tcgen05.mma SMEM operands is:
//   [ 0:13] start address (>> 4)
//   [14:15] reserved
//   [16:29] LBO (>> 4)
//   [30:31] reserved
//   [32:45] SBO (>> 4)
//   [46]    leading-dim byte swap (LBO-major flag)
//   [47:48] reserved
//   [49:51] matrix base offset
//   [52:60] reserved
//   [61:63] swizzle mode  (0 = none, 1 = 32B, 2 = 64B, 3 = 128B)
__device__ __forceinline__ uint64_t make_desc_a(uint32_t smem_addr, int LBO, int SBO) {
    return desc_encode((uint64_t)smem_addr)
         | (desc_encode((uint64_t)LBO) << 16)
         | (desc_encode((uint64_t)SBO) << 32)
         | (1ULL << 46);  // leading-dim major
}

__device__ __forceinline__ uint64_t make_desc_b(uint32_t smem_addr, int LBO, int SBO) {
    return desc_encode((uint64_t)smem_addr)
         | (desc_encode((uint64_t)LBO) << 16)
         | (desc_encode((uint64_t)SBO) << 32)
         | (1ULL << 46);
}

// --------- Kernel ---------
extern "C" __global__ void tcgen05_gemm_one_tile(
    const __nv_bfloat16 *__restrict__ A,   // [M=128, K=16] row-major
    const __nv_bfloat16 *__restrict__ B,   // [N=256, K=16] row-major  (B^T view: K × N)
    float               *__restrict__ C)   // [M=128, N=256] row-major fp32
{
    __shared__ __align__(16) __nv_bfloat16 s_A[M * K];  // 128 × 16 bf16 = 4096 B
    __shared__ __align__(16) __nv_bfloat16 s_B[N * K];  // 256 × 16 bf16 = 8192 B
    __shared__ uint32_t tmem_slot;                        // where tcgen05.alloc writes TMEM base
    __shared__ uint64_t mma_mbar;                         // mbarrier for MMA completion

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    // --------- 1. Allocate TMEM (one warp, synchronized across block) ---------
    // Allocate 256 "columns" (N=256), each 32-bit wide. This covers a [M×N] fp32
    // accumulator laid out across the TMEM sub-partitions.
    if (warp == 0) {
        uint32_t slot_smem = static_cast<uint32_t>(__cvta_generic_to_shared(&tmem_slot));
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
            :: "r"(slot_smem), "n"(N));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
    }
    // --------- 2. Init mbarrier for MMA completion ---------
    if (tid == 0) {
        uint32_t bar_smem = static_cast<uint32_t>(__cvta_generic_to_shared(&mma_mbar));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(bar_smem));
        asm volatile("fence.mbarrier_init.release.cluster;\n");
    }
    __syncthreads();
    const uint32_t taddr = tmem_slot;

    // --------- 3. Copy A, B from global to shared ---------
    // A: 128 × 16 bf16 = 4096 bytes = 256 x 16B. 256 threads do 1 load each.
    if (tid < 256) {
        int r = tid >> 1;       // 0..128
        int c = (tid & 1) << 3; // 0 or 8
        s_A[r * K + c + 0] = A[r * K + c + 0];
        s_A[r * K + c + 1] = A[r * K + c + 1];
        s_A[r * K + c + 2] = A[r * K + c + 2];
        s_A[r * K + c + 3] = A[r * K + c + 3];
        s_A[r * K + c + 4] = A[r * K + c + 4];
        s_A[r * K + c + 5] = A[r * K + c + 5];
        s_A[r * K + c + 6] = A[r * K + c + 6];
        s_A[r * K + c + 7] = A[r * K + c + 7];
    }
    // B: 256 × 16 bf16 = 8192 bytes = 512 × 16B. All 512 threads do 1 load each.
    {
        int r = tid >> 1;        // 0..256
        int c = (tid & 1) << 3;
        if (r < N) {
            s_B[r * K + c + 0] = B[r * K + c + 0];
            s_B[r * K + c + 1] = B[r * K + c + 1];
            s_B[r * K + c + 2] = B[r * K + c + 2];
            s_B[r * K + c + 3] = B[r * K + c + 3];
            s_B[r * K + c + 4] = B[r * K + c + 4];
            s_B[r * K + c + 5] = B[r * K + c + 5];
            s_B[r * K + c + 6] = B[r * K + c + 6];
            s_B[r * K + c + 7] = B[r * K + c + 7];
        }
    }
    __syncthreads();

    // --------- 4. Issue MMA via one thread ---------
    // i_desc (32-bit) layout per tcgen05 spec:
    //   [4]    dtype = 1 (FP32 accumulator)
    //   [7]    atype = 1 (BF16)
    //   [10]   btype = 1 (BF16)
    //   [17:23] MMA_N >> 3  (N in units of 8)
    //   [24:28] MMA_M >> 4  (M in units of 16)
    constexpr uint32_t i_desc =
        (1U << 4)
      | (1U << 7)
      | (1U << 10)
      | ((uint32_t)(N >> 3) << 17)
      | ((uint32_t)(M >> 4) << 24);

    if (tid == 0) {
        uint32_t a_smem = static_cast<uint32_t>(__cvta_generic_to_shared(s_A));
        uint32_t b_smem = static_cast<uint32_t>(__cvta_generic_to_shared(s_B));
        // For unswizzled bf16 kind::f16 MMA with M=128, K=16:
        //   8×16B box is 8 rows × 8 bf16 elements = 128 bytes
        //   K-boxes: 2 (covers K=16), stride = 128 bytes (SBO)
        //   M-boxes: 16 (covers M=128), stride between them = K-boxes*128 = 256 B (LBO)
        // Row-major storage gives different offsets though — assuming box-major shared.
        uint64_t a_desc = make_desc_a(a_smem, /*LBO=*/ 256, /*SBO=*/ 128);
        uint64_t b_desc = make_desc_b(b_smem, /*LBO=*/ 256, /*SBO=*/ 128);

        uint32_t bar_smem = static_cast<uint32_t>(__cvta_generic_to_shared(&mma_mbar));

        // enable_input_d=0 => reset accumulator (first mma of the K-loop).
        asm volatile(
            "{\n"
            "  tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 0;\n"
            "  tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%4];\n"
            "}\n"
            :: "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(bar_smem));
    }

    // --------- 5. Wait for MMA completion via mbarrier ---------
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

    // --------- 6. Read back TMEM into registers ---------
    // tcgen05.ld.sync.aligned.32x32b.x8: each warp reads 32 rows × 8 cols of fp32.
    // With M=128 and 32 rows per warp, we need 4 warps to cover the M dim.
    // For N=256 and 8 cols per load, we need 32 iterations to cover N.
    // total warps of work = 4 (M) × 32 (N) = 128 "ld" calls.
    // We have (blockDim.x / 32) warps. With blockDim.x=128, we have 4 warps, each
    // does 32 iterations covering all 256 cols for one M-slice.
    for (int n_off = 0; n_off < N; n_off += 8) {
        if (warp < 4) {   // 4 warps cover M=128
            int row_offset = warp * 32;                 // 0, 32, 64, 96
            // addr = taddr | (row_offset << 16) | n_off
            uint32_t addr = taddr + (row_offset << 16) + n_off;
            float t0, t1, t2, t3, t4, t5, t6, t7;
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
                : "=f"(t0), "=f"(t1), "=f"(t2), "=f"(t3),
                  "=f"(t4), "=f"(t5), "=f"(t6), "=f"(t7)
                : "r"(addr));
            // tcgen05.wait::ld ensures loaded values are visible.
            asm volatile("tcgen05.wait::ld.sync.aligned;\n");

            // For this layout, lane `lane` owns element row=(row_offset + lane),
            // cols=(n_off..n_off+8). Write directly to C.
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

// Host launcher.
extern "C" void launch_tcgen05_gemm_one_tile(
    const void *A, const void *B, void *C, cudaStream_t stream)
{
    dim3 grid(1);
    dim3 block(128);   // 4 warps — enough to cover M=128 in the readback step
    tcgen05_gemm_one_tile<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16 *)A, (const __nv_bfloat16 *)B, (float *)C);
}
