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
constexpr int K = 256;          // bf16 elements; multi K-tile layout
constexpr int K_TILE = 64;      // bf16 per K-tile (= 128 bytes, one 128B super-block)
constexpr int K_TILES = K / K_TILE;
constexpr int MMA_K = 16;       // bf16 K per tcgen05.mma call
constexpr int K_STEPS = K / MMA_K;
constexpr int MMAS_PER_TILE = K_TILE / MMA_K;    // 4

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

// Per-block tile dimensions (fixed). Grid dim controls output N tiling.
//   blockIdx.x: which N-tile this block owns.
//   Output C[M, N_TOTAL] — M=128 fixed, N_TOTAL = gridDim.x * N.
extern "C" __global__ void tcgen05_gemm_one_tile(
    const __nv_bfloat16 *__restrict__ A,      // [M=128, K]
    const __nv_bfloat16 *__restrict__ B_all,  // [N_TOTAL, K]
    float               *__restrict__ C_all,  // [M, N_TOTAL] fp32
    int N_TOTAL)
{
    const int block_n_idx = blockIdx.x;
    const int n_offset = block_n_idx * N;
    const __nv_bfloat16 *B = B_all + (size_t)n_offset * K;
    float               *C = C_all + (size_t)n_offset;  // will write stride N_TOTAL
    // K-tile-major layout: K_TILES copies of [M × K_TILE] and [N × K_TILE],
    // each swizzled per-tile. For K=128 (2 K-tiles): A = 32 KB, B = 64 KB.
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    __nv_bfloat16 *s_A = reinterpret_cast<__nv_bfloat16 *>(smem_raw);
    __nv_bfloat16 *s_B = reinterpret_cast<__nv_bfloat16 *>(
        smem_raw + K_TILES * M * K_TILE * sizeof(__nv_bfloat16));
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

    // --------- 3. Load A, B to shared in K-TILE-MAJOR layout with 128B swizzle.
    // For a row-major source A[M, K] bf16, we rearrange into K_TILES copies of
    // [M × K_TILE bf16], each copy swizzled by (row & 7). Global bf16 offset
    // c ∈ [0, K) split as (k_tile = c / K_TILE, local_c = c % K_TILE), then
    // the swizzled in-tile position is (row * K_TILE + (chunk^(r&7)) << 3).
    //
    // Standard 128B swizzle for 8×16B chunks within a 128B row.
    auto swz_offset = [](int r, int local_c) -> int {
        int chunk    = local_c >> 3;            // 0..7 chunks of 8 bf16 each
        int sw_chunk = chunk ^ (r & 7);
        return (sw_chunk << 3) | (local_c & 7);
    };

    // A: M * K bf16 total = (M * K / 8) loads of 8 bf16 each.
    {
        constexpr int A_TOTAL = M * K / 8;   // 8 bf16 per cp.async
        constexpr int A_ITERS = (A_TOTAL + 511) / 512;
        for (int it = 0; it < A_ITERS; it++) {
            int i = tid + it * 512;
            if (i >= A_TOTAL) break;
            int r = i / (K / 8);               // row in [0, M)
            int c_chunk = i % (K / 8);         // chunk in [0, K/8)
            int c = c_chunk * 8;               // bf16 offset in [0, K)
            int k_tile = c / K_TILE;
            int local_c = c % K_TILE;
            int sw_local_c = swz_offset(r, local_c);
            __nv_bfloat16 *dst = s_A + k_tile * M * K_TILE + r * K_TILE + sw_local_c;
            const __nv_bfloat16 *src = A + r * K + c;
            uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                         :: "r"(smem_addr), "l"(src));
        }
    }
    // B: N * K bf16 total = (N * K / 8) loads.
    {
        constexpr int B_TOTAL = N * K / 8;
        constexpr int B_ITERS = (B_TOTAL + 511) / 512;
        for (int it = 0; it < B_ITERS; it++) {
            int i = tid + it * 512;
            if (i >= B_TOTAL) break;
            int r = i / (K / 8);
            int c_chunk = i % (K / 8);
            int c = c_chunk * 8;
            int k_tile = c / K_TILE;
            int local_c = c % K_TILE;
            int sw_local_c = swz_offset(r, local_c);
            __nv_bfloat16 *dst = s_B + k_tile * N * K_TILE + r * K_TILE + sw_local_c;
            const __nv_bfloat16 *src = B + r * K + c;
            uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                         :: "r"(smem_addr), "l"(src));
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
        // Remaining K-steps. K-tile-major: each K_TILE (64 bf16 = 128 bytes)
        // is one contiguous 128B-swizzled block of M*128 or N*128 bytes.
        // Within one tile, MMA_K=16 advances are 32 bytes. Between tiles,
        // jump by M*K_TILE*2 bytes for A and N*K_TILE*2 for B.
        for (int ks = 1; ks < K_STEPS; ks++) {
            int k_tile_idx = ks / MMAS_PER_TILE;
            int k_in_tile  = ks % MMAS_PER_TILE;
            uint32_t a_smem = a_smem_base
                + (uint32_t)(k_tile_idx * M * K_TILE * 2)   // bytes to next K-tile of A
                + (uint32_t)(k_in_tile * MMA_K * 2);         // within tile
            uint32_t b_smem = b_smem_base
                + (uint32_t)(k_tile_idx * N * K_TILE * 2)
                + (uint32_t)(k_in_tile * MMA_K * 2);
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
                float *dst = C + out_row * N_TOTAL + n_off;
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
    const void *A, const void *B, void *C, int N_TOTAL, cudaStream_t stream)
{
    static bool cfg = false;
    if (!cfg) {
        cudaFuncSetAttribute((void *)tcgen05_gemm_one_tile,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);
        cfg = true;
    }
    dim3 grid(N_TOTAL / N);
    dim3 block(512);
    constexpr size_t smem_bytes = (M + N) * K * sizeof(__nv_bfloat16);
    tcgen05_gemm_one_tile<<<grid, block, smem_bytes, stream>>>(
        (const __nv_bfloat16 *)A, (const __nv_bfloat16 *)B, (float *)C, N_TOTAL);
}
