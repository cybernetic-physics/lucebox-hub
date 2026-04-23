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

## Benchmark at realistic scale

### M=1024, N=4096, K=1024 (8.6 GFLOPs)

```
tcgen05 (bf16 → fp32)      :   32.8 μs    ~262 TFLOPs   (13% of B200 bf16 peak)
torch.matmul bf16 (cuBLAS) :   11.1 μs    ~738 TFLOPs   (37% of peak)
torch.matmul fp32 (cuBLAS) :  182.5 μs     (fp32 path, not tensor cores)
```

tcgen05 is **2.95× slower than cuBLAS at realistic scale.** This is
the honest state at current session end.

### Extensions that landed

1. **K-tile-major shared memory layout** — K_TILES copies of
   [M × K_TILE bf16] / [N × K_TILE bf16], each with 128B swizzle
   applied to its rows. Descriptor advance between K-tiles is
   `M * K_TILE * 2` bytes. Tested correct at K=64, K=128, K=256, K=1024.
2. **cp.async K-streaming** — 2-buffer ping-pong loads one K-tile
   ahead of MMA, so arbitrary K is supported with fixed 96 KB of
   shared memory.
3. **2D grid tiling** — `(blockIdx.x, blockIdx.y)` = `(n_tile, m_tile)`.
   128 blocks for M=1024 N=4096 — uses most of B200's 148 SMs.

### What closes the remaining 2.95× gap

1. **Producer/consumer warp specialization** (biggest single win).
   Current kernel has all 512 threads cp.async'ing together, then 511
   wait while thread 0 issues 4 MMAs. Split into 2 producer warps
   doing TMA + 14 consumer warps doing MMA — mbarriers handle
   handoff. This is the ThunderKittens Blackwell pattern and gets
   "only one bubble in the whole tensor pipeline".
2. **TMA instead of cp.async** — `cp.async.bulk.tensor.2d` with a
   host-constructed tensor map. Hardware does the swizzle for us.
   Roughly halves load latency vs cp.async.
3. **`cta_group::2`** — two CTAs sharing TMEM support MMA_M=256,
   i.e. twice the per-call FLOPs.
4. **Register-level MMA issue pipelining** — hoist the descriptor
   recomputation so thread 0 issues with near-zero idle cycles.

Each is 1 focused session of careful PTX work. None of them needs
tcgen05 infrastructure redesign — the alloc / mbarrier / commit /
wait / ld pattern is all correct, it's about filling the pipeline.

## Next session: integrate into prefill_megakernel

With tcgen05 correctness established and K-streaming working:

1. Package as `gemm_tcgen05_core` matching `gemm_wmma_core`'s
   signature in the prefill megakernel.
2. Swap it into `../prefill_megakernel/kernel.cu`.
3. Bench full Qwen 3.5-0.8B prefill forward vs existing cuBLAS path.

Expected even at current 2.95× per-GEMM deficit: roughly similar to
current WMMA shipping state (0.74× of cuBLAS in full prefill) —
because the GEMM time is a fraction of total time that's dominated
by DN recurrence + RMSNorms + attention. Closing the GEMM gap to 1×
cuBLAS (via producer/consumer above) and then beyond unlocks
1.5–2× cuBLAS for the full prefill.

## Files

- `kernel.cu` — working tcgen05 GEMM (one tile)
- `torch_bindings.cpp` — `torch.ops.tcgen05_gemm_C.tcgen05_gemm_one_tile`
- `setup.py` — CUDAExtension targeting sm_100a
- `test.py` — torch-reference correctness (passes)
- `probe.py` — TMEM position → logical col decoder
- `probe_kstep.py` — isolates K-step contributions (found the bug)
- `bench.py` — walltime vs torch/cuBLAS
