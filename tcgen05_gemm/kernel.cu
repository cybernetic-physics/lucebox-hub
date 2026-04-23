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
constexpr int K = 1024;         // bf16 elements; multi K-tile layout with streaming
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

// Per-block tile: [M_BLK=128, N_BLK=256]. Grid dim controls M and N tiling.
//   blockIdx.x: which N-tile this block owns.
//   blockIdx.y: which M-tile this block owns.
//   A[M_TOTAL, K], B[N_TOTAL, K], C[M_TOTAL, N_TOTAL].
extern "C" __global__ void tcgen05_gemm_one_tile(
    const __nv_bfloat16 *__restrict__ A_all,
    const __nv_bfloat16 *__restrict__ B_all,
    float               *__restrict__ C_all,
    int M_TOTAL, int N_TOTAL)
{
    const int block_n_idx = blockIdx.x;
    const int block_m_idx = blockIdx.y;
    const int n_offset = block_n_idx * N;
    const int m_offset = block_m_idx * M;
    const __nv_bfloat16 *A = A_all + (size_t)m_offset * K;
    const __nv_bfloat16 *B = B_all + (size_t)n_offset * K;
    float               *C = C_all + (size_t)m_offset * N_TOTAL + n_offset;
    // TRIPLE-buffered K-streaming so cp.async for tile kt+2 can overlap with
    // MMA on tile kt. With 2 buffers the MMA's async shared-mem reads race
    // against cp.async overwrites; with 3 buffers there's always a buffer
    // gap between "being MMA'd" and "being written".
    //   s_A: 3 × M × K_TILE bf16  = 3 × 128 × 128 = 48 KB
    //   s_B: 3 × N × K_TILE bf16  = 3 × 256 × 128 = 96 KB
    //   total dynamic: 144 KB (opt-in via cudaFuncSetAttribute).
    constexpr int S_A_TILE_BYTES = M * K_TILE * 2;
    constexpr int S_B_TILE_BYTES = N * K_TILE * 2;
    constexpr int NBUF = 4;
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    __nv_bfloat16 *s_A = reinterpret_cast<__nv_bfloat16 *>(smem_raw);
    __nv_bfloat16 *s_B = reinterpret_cast<__nv_bfloat16 *>(
        smem_raw + NBUF * S_A_TILE_BYTES);
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

    // --------- 3. K-streaming load helpers ---------
    auto swz_offset = [](int r, int local_c) -> int {
        int chunk    = local_c >> 3;
        int sw_chunk = chunk ^ (r & 7);
        return (sw_chunk << 3) | (local_c & 7);
    };

    // Load ONE K-tile (K_TILE bf16 per row) of A into s_A[buf].
    // Uses cp.async, does NOT commit_group — caller groups multiple loads.
    auto load_a_tile = [&](int kt, int buf) {
        __nv_bfloat16 *A_buf = s_A + buf * (M * K_TILE);
        constexpr int A_TILE_TOTAL = M * K_TILE / 8;     // # of 16-byte cp.asyncs
        constexpr int A_TILE_ITERS = (A_TILE_TOTAL + 511) / 512;
        #pragma unroll
        for (int it = 0; it < A_TILE_ITERS; it++) {
            int i = tid + it * 512;
            if (i >= A_TILE_TOTAL) break;
            int r = i / (K_TILE / 8);
            int c_chunk = i % (K_TILE / 8);
            int c = c_chunk * 8;
            int sw_c = swz_offset(r, c);
            __nv_bfloat16 *dst = A_buf + r * K_TILE + sw_c;
            const __nv_bfloat16 *src = A + r * K + (kt * K_TILE) + c;
            uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                         :: "r"(smem_addr), "l"(src));
        }
    };
    auto load_b_tile = [&](int kt, int buf) {
        __nv_bfloat16 *B_buf = s_B + buf * (N * K_TILE);
        constexpr int B_TILE_TOTAL = N * K_TILE / 8;
        constexpr int B_TILE_ITERS = (B_TILE_TOTAL + 511) / 512;
        #pragma unroll
        for (int it = 0; it < B_TILE_ITERS; it++) {
            int i = tid + it * 512;
            if (i >= B_TILE_TOTAL) break;
            int r = i / (K_TILE / 8);
            int c_chunk = i % (K_TILE / 8);
            int c = c_chunk * 8;
            int sw_c = swz_offset(r, c);
            __nv_bfloat16 *dst = B_buf + r * K_TILE + sw_c;
            const __nv_bfloat16 *src = B + r * K + (kt * K_TILE) + c;
            uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                         :: "r"(smem_addr), "l"(src));
        }
    };

    // Prologue: issue stage-0 AND stage-1 loads (up to 2 ahead).
    load_a_tile(0, 0);
    load_b_tile(0, 0);
    asm volatile("cp.async.commit_group;\n");
    if (K_TILES > 1) {
        load_a_tile(1, 1);
        load_b_tile(1, 1);
        asm volatile("cp.async.commit_group;\n");
    }

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

    uint32_t a_smem_base_0 = static_cast<uint32_t>(__cvta_generic_to_shared(s_A));
    uint32_t b_smem_base_0 = static_cast<uint32_t>(__cvta_generic_to_shared(s_B));
    uint32_t bar_smem = static_cast<uint32_t>(__cvta_generic_to_shared(&mma_mbar));

    // Main loop: triple-buffered. buf = kt % 3. On each iter:
    //   1. Wait for tile kt to finish loading (via cp.async.wait_group).
    //   2. Issue prefetch for tile kt+2 (if exists) into (kt+2) % 3.
    //   3. MMA thread 0 issues 4 MMAs on buf = kt%3. This is async; MMAs
    //      continue during next iter's cp.async.
    //   4. __syncthreads once per iter so cp.async threads in iter kt+1 see
    //      consistent shared-mem state with MMA thread's descriptor reads.
    for (int kt = 0; kt < K_TILES; kt++) {
        int buf = kt % NBUF;
        int prefetch_kt = kt + 2;
        int prefetch_buf = prefetch_kt % NBUF;

        // Wait for tile kt to be ready. We have up to 2 loads in flight
        // (tile kt+1 already queued, maybe tile kt+2 about to queue).
        if (prefetch_kt < K_TILES) {
            load_a_tile(prefetch_kt, prefetch_buf);
            load_b_tile(prefetch_kt, prefetch_buf);
            asm volatile("cp.async.commit_group;\n");
            asm volatile("cp.async.wait_group 2;\n");
        } else if (kt + 1 < K_TILES) {
            asm volatile("cp.async.wait_group 1;\n");
        } else {
            asm volatile("cp.async.wait_group 0;\n");
        }
        __syncthreads();

        if (tid == 0) {
            uint32_t a_base = a_smem_base_0 + buf * S_A_TILE_BYTES;
            uint32_t b_base = b_smem_base_0 + buf * S_B_TILE_BYTES;
            #pragma unroll
            for (int k_in_tile = 0; k_in_tile < MMAS_PER_TILE; k_in_tile++) {
                uint32_t a_smem = a_base + (uint32_t)(k_in_tile * MMA_K * 2);
                uint32_t b_smem = b_base + (uint32_t)(k_in_tile * MMA_K * 2);
                uint64_t a_desc = make_desc(a_smem);
                uint64_t b_desc = make_desc(b_smem);
                int enable_d = (kt == 0 && k_in_tile == 0) ? 0 : 1;
                if (enable_d == 0) {
                    asm volatile(
                        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 0;\n"
                        :: "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(i_desc));
                } else {
                    asm volatile(
                        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 1;\n"
                        :: "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(i_desc));
                }
            }
        }
        // No post-MMA __syncthreads: MMAs are async and will read buf
        // BEFORE next-next iter's cp.async overwrites it (3-buffer gap).
    }

    if (tid == 0) {
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
    const void *A, const void *B, void *C,
    int M_TOTAL, int N_TOTAL, cudaStream_t stream)
{
    static bool cfg = false;
    if (!cfg) {
        cudaFuncSetAttribute((void *)tcgen05_gemm_one_tile,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);
        cfg = true;
    }
    dim3 grid(N_TOTAL / N, M_TOTAL / M);
    dim3 block(512);
    constexpr size_t smem_bytes = 4 * (M + N) * K_TILE * sizeof(__nv_bfloat16);
    tcgen05_gemm_one_tile<<<grid, block, smem_bytes, stream>>>(
        (const __nv_bfloat16 *)A, (const __nv_bfloat16 *)B, (float *)C,
        M_TOTAL, N_TOTAL);
}
