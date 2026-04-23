# tcgen05 bf16 → fp32 GEMM — CORRECT on Blackwell (sm_100)

Standalone reference for the Blackwell-native `tcgen05.mma` instruction
with bf16 inputs and fp32 accumulation.

## Status: ✅ 100% numerically correct

`M=128, N=256, K=64` bf16 → fp32. 32768/32768 output positions match
torch reference to bf16 MMA precision.

### Instruction sequence (all verified)

1. `tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32` — allocate TMEM for accumulator
2. `tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned`
3. `mbarrier.init.shared::cta.b64` — completion barrier
4. `cp.async.ca.shared.global` with **128B swizzle applied to destination address**
5. `cp.async.commit_group` + `cp.async.wait_group 0`
6. `tcgen05.mma.cta_group::1.kind::f16 [tmem], a_desc, b_desc, i_desc, enable_input_d`
   run in a K-loop (`enable_input_d=0` on first call to reset accumulator, `1` after)
7. `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64`
8. `mbarrier.try_wait.parity.shared::cta.b64`
9. `tcgen05.ld.sync.aligned.32x32b.x8.b32` — read fp32 accumulator into registers
10. `tcgen05.wait::ld.sync.aligned`
11. `tcgen05.dealloc.cta_group::1.sync.aligned.b32`

### Key unlock: 128B swizzle on cp.async writes

The bug that took days to pin down: the MMA's shared-memory descriptor
with `swizzle mode = 2` (`2ULL << 61`) tells it to read data in
128B-swizzled form, but raw row-major `cp.async` writes leave data
un-swizzled. Physically, each row's 8 16-byte chunks must be XORed with
the row's lower 3 bits:

```
for row r, logical bf16 offset c in 0..63 (covers one 128B super-block):
    chunk      = c >> 3         (0..7, the 16-byte chunk index)
    sw_chunk   = chunk ^ (r & 7)
    physical_c = sw_chunk << 3
    dst = &shared[r * K + physical_c]
```

The diagnostic that found it: setting `A[:, 16]=1, B[j, 16]=j/1024` and
looking for outputs showed K-step 1 contributed only at output cols
{0, 8, 16, 24}, the stride-8 signature of 128B swizzle reading
non-swizzled data. Fix: apply that swizzle during the cp.async.

### 128B swizzle table (for posterity)

For a row r (0..7 relevant) and chunk c (0..7):

```
physical_chunk[r][c]:
 r\c   0 1 2 3 4 5 6 7
  0    0 1 2 3 4 5 6 7  ← identity (r & 7 = 0)
  1    1 0 3 2 5 4 7 6  ← flip pairs
  2    2 3 0 1 6 7 4 5
  3    3 2 1 0 7 6 5 4
  4    4 5 6 7 0 1 2 3  ← 128B swap
  5    5 4 7 6 1 0 3 2
  6    6 7 4 5 2 3 0 1
  7    7 6 5 4 3 2 1 0
```

## Benchmark at M=128 N=256 K=64 (4 MFLOPs)

```
tcgen05 (bf16 → fp32)      :  22.5 μs
torch.matmul bf16 (cuBLAS) :  14.5 μs   (cuBLAS 1.56× faster at this size)
torch.matmul fp32 (cuBLAS) :  39.6 μs
```

At this tiny shape both kernels are launch-overhead-limited (4 MFLOPs in
22 μs = 180 GFLOPs, way below B200 peak of ~2 PFLOPs). The comparison is
unfair — real throughput needs M/N/K in the thousands.

**Scaling up requires two more pieces:**

1. **K-tile-major layout for K > 64** — currently K=64 fits one 128B
   super-block per row. For K=128+ rows span multiple super-blocks, and
   the descriptor advance by 32 bytes per K-step crosses block
   boundaries, breaking layout. Fix: store A/B as K-tile-major (each
   K-tile = BLOCK_M rows × 128 bytes = 16 KB, concatenated in K),
   advance descriptor by `k_tile_idx * BLOCK_M * 128` per tile per
   gau-nernst's pattern.

2. **M/N tiling across blocks** — one cooperative grid launch with each
   block handling a different `(m_tile, n_tile)` output tile. TMEM alloc
   amortizes across all MMA calls within a block.

Both are mechanical extensions of the working single-tile kernel.

## Next session: integrate into prefill_megakernel

With tcgen05 correctness established:

1. Add K-tile-major load helpers (applicable for K=1024 = 16 K-tiles).
2. Tile across N in a grid loop.
3. Package as `gemm_tcgen05_core` matching `gemm_wmma_core`'s signature.
4. Swap it into `../prefill_megakernel/kernel.cu`.
5. Bench full Qwen 3.5-0.8B prefill forward vs existing cuBLAS path.

Expected: 1.2–1.8× cuBLAS for the full megakernel at S=256 (per
published Blackwell tcgen05 kernel numbers). Path to 2×+ requires TMA
for loads (eliminates cp.async overhead) and producer/consumer warp
specialization per the ThunderKittens Blackwell pattern.

## Files

- `kernel.cu` — working tcgen05 GEMM (one tile)
- `torch_bindings.cpp` — `torch.ops.tcgen05_gemm_C.tcgen05_gemm_one_tile`
- `setup.py` — CUDAExtension targeting sm_100a
- `test.py` — torch-reference correctness (passes)
- `probe.py` — TMEM position → logical col decoder
- `probe_kstep.py` — isolates K-step contributions (found the bug)
- `bench.py` — walltime vs torch/cuBLAS
