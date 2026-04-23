/**
 * tcgen05 bf16→fp32 GEMM v5: 2-SM MMA (cta_group::2).
 *
 * Structure from gau-nernst/learn-cuda/02e_matmul_sm100/matmul_v5.cu
 * (their 1169 TFLOPs reference at M=N=K=4096).
 *
 * Key v4 → v5 changes:
 *   - __cluster_dims__(2, 1, 1): CTAs launch in pairs on adjacent SMs.
 *   - MMA_M = 256 (BLOCK_M * CTA_GROUP). tcgen05.mma.cta_group::2 issues
 *     one MMA that covers 128 rows on SM0 and 128 rows on SM1.
 *   - Each CTA loads only its half of B (BLOCK_N / 2 = 128 cols).
 *   - MMA issued only by CTA rank 0; result mcast signals both CTAs.
 *   - TMA mbar init with count=2 (both CTAs arrive); MMA mbar multicast.
 */

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// ----------------------- PTX helpers -----------------------

__device__ __forceinline__ uint32_t elect_sync_once() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t.reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t}"
        : "+r"(pred) : "r"(0xFFFFFFFFU));
    return pred;
}

__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_addr), "r"(count));
}

__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t.reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t}"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks));
}

template <int CTA_GROUP>
__device__ __forceinline__ void tma_3d_g2s(int dst, const void *tmap, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::%6 "
        "[%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(mbar_addr), "n"(CTA_GROUP) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, int size) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(size) : "memory");
}

template <int CTA_GROUP>
__device__ __forceinline__ void tcgen05_mma_f16(int taddr, uint64_t a_desc, uint64_t b_desc,
                                                 uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::%5.kind::f16 [%0], %1, %2, %3, p;\n\t}"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(enable_input_d), "n"(CTA_GROUP));
}

template <int CTA_GROUP>
__device__ __forceinline__ void tcgen05_commit_mcast(int mbar_addr, int16_t cta_mask) {
    asm volatile(
        "tcgen05.commit.cta_group::%2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 "
        "[%0], %1;"
        :: "r"(mbar_addr), "h"(cta_mask), "n"(CTA_GROUP) : "memory");
}

template <int CTA_GROUP>
__device__ __forceinline__ void tcgen05_alloc(int smem_addr, int ncols) {
    asm volatile("tcgen05.alloc.cta_group::%2.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(smem_addr), "r"(ncols), "n"(CTA_GROUP));
}

template <int CTA_GROUP>
__device__ __forceinline__ void tcgen05_dealloc(int taddr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::%2.sync.aligned.b32 %0, %1;"
                 :: "r"(taddr), "r"(ncols), "n"(CTA_GROUP));
}

__device__ __forceinline__ constexpr uint64_t desc_encode(uint64_t x) {
    return (x & 0x3'FFFFULL) >> 4ULL;
}

// ----------------------- Kernel -----------------------

constexpr int NUM_WARPS = 4;
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 256;
constexpr int BLOCK_K = 64;
constexpr int CTA_GROUP = 2;
constexpr int NUM_STAGES = 7;      // per-CTA shared (128+128) × 64 × 7 × 2 = 224 KB, fits B200
constexpr int MMA_K = 16;
constexpr int MMA_M = BLOCK_M * CTA_GROUP;   // 256

extern "C" __global__ __cluster_dims__(CTA_GROUP, 1, 1)
__launch_bounds__(TB_SIZE) void tcgen05_gemm_v5_kernel(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B_tmap,
    float *C_ptr,
    int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int warp_id = tid / WARP_SIZE;

    const int grid_n = N / BLOCK_N;
    // bid must run along M-mode first so cta_group::2 pairs consecutive
    // clusters on the same N-column across SM0/SM1.
    constexpr int GROUP_M = CTA_GROUP;
    const int bid_m = bid / (grid_n * GROUP_M) * GROUP_M + (bid % GROUP_M);
    const int bid_n = (bid / GROUP_M) % grid_n;

    const int off_m = bid_m * BLOCK_M;
    const int off_n = bid_n * BLOCK_N;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    extern __shared__ __align__(1024) char smem_ptr[];
    const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
    constexpr int A_size = BLOCK_M * BLOCK_K * 2;
    constexpr int B_size = (BLOCK_N / CTA_GROUP) * BLOCK_K * 2;

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ uint64_t mbars[NUM_STAGES * 2 + 1];
    __shared__ int tmem_addr[1];
    const int tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbars));
    const int mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;
    const int mainloop_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;

    if (warp_id == 0 && elect_sync_once()) {
        for (int i = 0; i < NUM_STAGES; i++) {
            mbarrier_init(tma_mbar_addr + i * 8, CTA_GROUP);
            mbarrier_init(mma_mbar_addr + i * 8, 1);
        }
        mbarrier_init(mainloop_mbar_addr, 1);
        asm volatile("fence.mbarrier_init.release.cluster;");
    } else if (warp_id == 1) {
        const int addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr));
        tcgen05_alloc<CTA_GROUP>(addr, BLOCK_N);
    }

    // Cluster-wide barrier (both CTAs synchronize)
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    const int taddr = tmem_addr[0];
    int phase = 0;

    constexpr uint32_t i_desc = (1U << 4U)
                              | (1U << 7U)
                              | (1U << 10U)
                              | ((uint32_t)BLOCK_N >> 3U << 17U)
                              | ((uint32_t)MMA_M    >> 4U << 24U);

    auto load = [&](int iter_k) {
        const int stage_id = iter_k % NUM_STAGES;
        if (iter_k >= NUM_STAGES)
            mbarrier_wait(mma_mbar_addr + stage_id * 8, phase ^ 1);
        if (stage_id == NUM_STAGES - 1)
            phase ^= 1;

        // Target the TMA mbar on CTA0 (clear bit 24 in the cluster-shared address).
        const int mbar_addr = (tma_mbar_addr + stage_id * 8) & 0xFEFFFFFF;
        const int A_smem = smem + stage_id * (A_size + B_size);
        const int B_smem = A_smem + A_size;

        const int off_k = iter_k * BLOCK_K;
        tma_3d_g2s<CTA_GROUP>(A_smem, &A_tmap, 0, off_m, off_k / 64, mbar_addr);
        tma_3d_g2s<CTA_GROUP>(
            B_smem, &B_tmap, 0,
            off_n + cta_rank * (BLOCK_N / CTA_GROUP), off_k / 64, mbar_addr);
        mbarrier_arrive_expect_tx(mbar_addr, A_size + B_size);
    };

    auto compute = [&](int iter_k) {
        const int stage_id = iter_k % NUM_STAGES;
        mbarrier_wait(tma_mbar_addr + stage_id * 8, phase);
        asm volatile("tcgen05.fence::after_thread_sync;");
        if (stage_id == NUM_STAGES - 1)
            phase ^= 1;

        const int A_smem = smem + stage_id * (A_size + B_size);
        const int B_smem = A_smem + A_size;

        auto make_desc = [](int addr) -> uint64_t {
            const int SBO = 8 * 128;
            return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
        };

        {
            int enable_d = (iter_k == 0) ? 0 : 1;
            tcgen05_mma_f16<CTA_GROUP>(taddr, make_desc(A_smem), make_desc(B_smem), i_desc, enable_d);
            for (int k2 = 1; k2 < 64 / MMA_K; k2++) {
                uint64_t a_desc = make_desc(A_smem + k2 * 32);
                uint64_t b_desc = make_desc(B_smem + k2 * 32);
                tcgen05_mma_f16<CTA_GROUP>(taddr, a_desc, b_desc, i_desc, 1);
            }
        }
        for (int k1 = 1; k1 < BLOCK_K / 64; k1++)
            for (int k2 = 0; k2 < 64 / MMA_K; k2++) {
                uint64_t a_desc = make_desc(A_smem + k1 * BLOCK_M * 128 + k2 * 32);
                uint64_t b_desc = make_desc(B_smem + k1 * (BLOCK_N / CTA_GROUP) * 128 + k2 * 32);
                tcgen05_mma_f16<CTA_GROUP>(taddr, a_desc, b_desc, i_desc, 1);
            }
        constexpr int16_t cta_mask = (1 << CTA_GROUP) - 1;
        tcgen05_commit_mcast<CTA_GROUP>(mma_mbar_addr + stage_id * 8, cta_mask);
    };

    const int num_iters = K / BLOCK_K;
    if (warp_id == 0 && elect_sync_once()) {
        for (int iter_k = 0; iter_k < num_iters; iter_k++) load(iter_k);
    } else if (cta_rank == 0 && warp_id == 1 && elect_sync_once()) {
        for (int iter_k = 0; iter_k < num_iters; iter_k++) compute(iter_k);
        constexpr int16_t cta_mask = (1 << CTA_GROUP) - 1;
        tcgen05_commit_mcast<CTA_GROUP>(mainloop_mbar_addr, cta_mask);
    }
    __syncthreads();
    mbarrier_wait(mainloop_mbar_addr, 0);
    asm volatile("tcgen05.fence::after_thread_sync;");

    // Epilogue: TMEM address uses cta_rank*128 (each SM holds half of MMA_M=256).
    // Output row uses off_m + tid (each CTA has its own off_m already set up
    // by the cluster-paired bid_m).
    const int lane_id = tid & 31;
    for (int n_off = 0; n_off < BLOCK_N; n_off += 8) {
        float tmp[8];
        const int addr = taddr + (((cta_rank * 128) + (warp_id * 32)) << 16) + n_off;
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
            "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]),
              "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7])
            : "r"(addr));
        asm volatile("tcgen05.wait::ld.sync.aligned;");

        // tid = warp_id * 32 + lane_id ∈ [0, 128). Output row = off_m + tid.
        float *dst = C_ptr + (off_m + tid) * N + (off_n + n_off);
        dst[0] = tmp[0]; dst[1] = tmp[1]; dst[2] = tmp[2]; dst[3] = tmp[3];
        dst[4] = tmp[4]; dst[5] = tmp[5]; dst[6] = tmp[6]; dst[7] = tmp[7];
    }
    __syncthreads();
    if (warp_id == 0)
        tcgen05_dealloc<CTA_GROUP>(taddr, BLOCK_N);
}

static void init_tmap_3d_128B(
    CUtensorMap *tmap, const __nv_bfloat16 *ptr,
    uint64_t global_height, uint64_t global_width,
    uint32_t shared_height, uint32_t shared_width)
{
    constexpr uint32_t rank = 3;
    uint64_t globalDim[rank]         = {64, global_height, global_width / 64};
    uint64_t globalStrides[rank - 1] = {global_width * sizeof(__nv_bfloat16), 128};
    uint32_t boxDim[rank]            = {64, shared_height, shared_width / 64};
    uint32_t elementStrides[rank]    = {1, 1, 1};

    cuTensorMapEncodeTiled(
        tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, (void *)ptr,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

extern "C" void launch_tcgen05_gemm_one_tile(
    const void *A, const void *B, void *C,
    int M_TOTAL, int N_TOTAL, int K, cudaStream_t stream)
{
    // MA tmap covers the FULL B since both CTAs index into it by cta_rank.
    CUtensorMap A_tmap, B_tmap;
    init_tmap_3d_128B(&A_tmap, (const __nv_bfloat16 *)A, M_TOTAL, K, BLOCK_M, BLOCK_K);
    init_tmap_3d_128B(&B_tmap, (const __nv_bfloat16 *)B, N_TOTAL, K, BLOCK_N / CTA_GROUP, BLOCK_K);

    const int grid = (M_TOTAL / BLOCK_M) * (N_TOTAL / BLOCK_N);
    const int size_AB = (BLOCK_M + BLOCK_N / CTA_GROUP) * BLOCK_K * NUM_STAGES;
    const int smem_bytes = size_AB * sizeof(__nv_bfloat16);

    static bool cfg = false;
    if (!cfg) {
        cudaFuncSetAttribute((void *)tcgen05_gemm_v5_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        cfg = true;
    }
    tcgen05_gemm_v5_kernel<<<grid, TB_SIZE, smem_bytes, stream>>>(
        A_tmap, B_tmap, (float *)C, M_TOTAL, N_TOTAL, K);
}
