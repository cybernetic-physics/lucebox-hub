/**
 * Blackwell sm_100 tcgen05.mma bf16→fp32 GEMM with warp specialization + TMA.
 *
 * Structure (from gau-nernst/learn-cuda/02e_matmul_sm100/matmul_v4.cu —
 *   a proven 1053 TFLOPs reference, ~78% of cuBLAS at M=N=K=4096):
 *
 *   Warp 0 (TMA producer): loops over K-iters, fires cp.async.bulk.tensor.3d
 *     loads for each new stage, arrives-on-tx on per-stage mbars.
 *   Warp 1 (MMA consumer): loops over K-iters, waits on per-stage tma mbars,
 *     issues tcgen05.mma chain, commits on per-stage mma mbars.
 *   Warps 2-3: only participate in the epilogue (reading tmem → writing C).
 *
 * Pipelining: NUM_STAGES overlapped TMA loads and MMAs. TMA warp waits on
 * mma mbar of stage s-NUM_STAGES before re-filling it; MMA warp waits on
 * tma mbar of stage s before issuing MMAs.
 *
 * Preconditions: M % 128 == 0, N % BLOCK_N == 0, K % BLOCK_K == 0.
 */

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// ----------------------- helpers (mirror common.h) -------------------------

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

__device__ __forceinline__ void tma_3d_g2s(int dst, const void *tmap, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1 "
        "[%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(mbar_addr) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, int size) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(size) : "memory");
}

__device__ __forceinline__ void tcgen05_mma_f16(int taddr, uint64_t a_desc, uint64_t b_desc,
                                                 uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(enable_input_d));
}

__device__ __forceinline__ constexpr uint64_t desc_encode(uint64_t x) {
    return (x & 0x3'FFFFULL) >> 4ULL;
}

// ---------------------------- Kernel ---------------------------------------

constexpr int NUM_WARPS = 4;
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 64;
constexpr int NUM_STAGES = 7;
constexpr int MMA_K = 16;

extern "C" __global__ __launch_bounds__(TB_SIZE) void tcgen05_gemm_v4_kernel(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B_tmap,
    float *C_ptr,
    int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int warp_id = tid / WARP_SIZE;

    const int grid_n = N / BLOCK_N;
    const int bid_m = bid / grid_n;
    const int bid_n = bid % grid_n;
    const int off_m = bid_m * BLOCK_M;
    const int off_n = bid_n * BLOCK_N;

    extern __shared__ __align__(1024) char smem_ptr[];
    const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
    constexpr int A_size = BLOCK_M * BLOCK_K * 2;     // bf16 bytes
    constexpr int B_size = BLOCK_N * BLOCK_K * 2;

    // Layout of mbar array: NUM_STAGES tma + NUM_STAGES mma + 1 epilogue.
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ uint64_t mbars[NUM_STAGES * 2 + 1];
    __shared__ int tmem_addr[1];
    const int tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbars));
    const int mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;
    const int mainloop_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;

    if (warp_id == 0 && elect_sync_once()) {
        for (int i = 0; i < NUM_STAGES * 2 + 1; i++)
            mbarrier_init(tma_mbar_addr + i * 8, 1);
        asm volatile("fence.mbarrier_init.release.cluster;");
    } else if (warp_id == 1) {
        const int addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr));
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     :: "r"(addr), "r"(BLOCK_N));
    }
    __syncthreads();
    const int taddr = tmem_addr[0];

    int phase = 0;

    constexpr uint32_t i_desc = (1U << 4U)
                              | (1U << 7U)
                              | (1U << 10U)
                              | ((uint32_t)BLOCK_N >> 3U << 17U)
                              | ((uint32_t)BLOCK_M >> 4U << 24U);

    auto load = [&](int iter_k) {
        const int stage_id = iter_k % NUM_STAGES;
        // Wait for MMA warp to have consumed the previous round's use of this stage.
        // Initial mma mbar phase is 0 (unfired), so we wait for 1 to know mma
        // already completed — but mbar is pre-set as "ready" by mbarrier_init + phase 0.
        // Actually: we want to wait for the mma-side signal AFTER first full cycle.
        if (iter_k >= NUM_STAGES)
            mbarrier_wait(mma_mbar_addr + stage_id * 8, phase ^ 1);
        if (stage_id == NUM_STAGES - 1)
            phase ^= 1;

        const int mbar_addr = tma_mbar_addr + stage_id * 8;
        const int A_smem = smem + stage_id * (A_size + B_size);
        const int B_smem = A_smem + A_size;

        const int off_k = iter_k * BLOCK_K;
        tma_3d_g2s(A_smem, &A_tmap, 0, off_m, off_k / 64, mbar_addr);
        tma_3d_g2s(B_smem, &B_tmap, 0, off_n, off_k / 64, mbar_addr);
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

        // First MMA clears accumulator on first iter_k, else accumulates.
        // Structure: K dim of block has BLOCK_K/64 groups of 64 bf16, each with
        // 64/MMA_K=4 MMAs. For BLOCK_K=64 that's 1 group × 4 MMAs per compute call.
        {
            int enable_d = (iter_k == 0) ? 0 : 1;
            tcgen05_mma_f16(taddr, make_desc(A_smem), make_desc(B_smem), i_desc, enable_d);
            for (int k2 = 1; k2 < 64 / MMA_K; k2++) {
                uint64_t a_desc = make_desc(A_smem + k2 * 32);
                uint64_t b_desc = make_desc(B_smem + k2 * 32);
                tcgen05_mma_f16(taddr, a_desc, b_desc, i_desc, 1);
            }
        }
        for (int k1 = 1; k1 < BLOCK_K / 64; k1++)
            for (int k2 = 0; k2 < 64 / MMA_K; k2++) {
                uint64_t a_desc = make_desc(A_smem + k1 * BLOCK_M * 128 + k2 * 32);
                uint64_t b_desc = make_desc(B_smem + k1 * BLOCK_N * 128 + k2 * 32);
                tcgen05_mma_f16(taddr, a_desc, b_desc, i_desc, 1);
            }
        asm volatile(
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
            :: "r"(mma_mbar_addr + stage_id * 8) : "memory");
    };

    const int num_iters = K / BLOCK_K;
    if (warp_id == 0 && elect_sync_once()) {
        for (int iter_k = 0; iter_k < num_iters; iter_k++) load(iter_k);
    } else if (warp_id == 1 && elect_sync_once()) {
        for (int iter_k = 0; iter_k < num_iters; iter_k++) compute(iter_k);
        asm volatile(
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
            :: "r"(mainloop_mbar_addr) : "memory");
    }
    __syncthreads();
    mbarrier_wait(mainloop_mbar_addr, 0);
    asm volatile("tcgen05.fence::after_thread_sync;");

    // Epilogue: all 4 warps read tmem, each warp handles 32 rows of M=128.
    // For output col stride of 8 per .x8: loop n_off 0..BLOCK_N step 8.
    const int lane_id = tid & 31;
    for (int n_off = 0; n_off < BLOCK_N; n_off += 8) {
        float tmp[8];
        const int addr = taddr + ((warp_id * 32) << 16) + n_off;
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
            "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]),
              "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7])
            : "r"(addr));
        asm volatile("tcgen05.wait::ld.sync.aligned;");

        const int out_row = warp_id * 32 + lane_id;
        if (out_row < BLOCK_M) {
            float *dst = C_ptr + (off_m + out_row) * N + (off_n + n_off);
            dst[0] = tmp[0]; dst[1] = tmp[1]; dst[2] = tmp[2]; dst[3] = tmp[3];
            dst[4] = tmp[4]; dst[5] = tmp[5]; dst[6] = tmp[6]; dst[7] = tmp[7];
        }
    }
    __syncthreads();
    if (warp_id == 0)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     :: "r"(taddr), "r"(BLOCK_N));
}

// Host-side helper to build a 3D tensor map matching the TMA load pattern.
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
    CUtensorMap A_tmap, B_tmap;
    init_tmap_3d_128B(&A_tmap, (const __nv_bfloat16 *)A, M_TOTAL, K, BLOCK_M, BLOCK_K);
    init_tmap_3d_128B(&B_tmap, (const __nv_bfloat16 *)B, N_TOTAL, K, BLOCK_N, BLOCK_K);

    const int grid = (M_TOTAL / BLOCK_M) * (N_TOTAL / BLOCK_N);
    const int size_AB = (BLOCK_M + BLOCK_N) * BLOCK_K * NUM_STAGES;
    const int smem_bytes = size_AB * sizeof(__nv_bfloat16);

    static bool cfg = false;
    if (!cfg) {
        cudaFuncSetAttribute((void *)tcgen05_gemm_v4_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        cfg = true;
    }
    tcgen05_gemm_v4_kernel<<<grid, TB_SIZE, smem_bytes, stream>>>(
        A_tmap, B_tmap, (float *)C, M_TOTAL, N_TOTAL, K);
}
