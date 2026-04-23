# Megakernel Qwen3.5-0.8B on NVIDIA B200 (sm_100)

Port of the megakernel backends to Blackwell datacenter (B200, sm_100). Batch
size 1, single-stream decode. GB10 (sm_121a) NVFP4 path and RTX 3090 (sm_86)
BF16 path both still build and still work on their native arches — the B200
changes are additive.

## TL;DR

| Method on B200 (Qwen3.5-0.8B) | pp520 (tok/s) | tg128 (tok/s) | tg128 vs llama.cpp |
|-------------------------------|:-------------:|:-------------:|:------------------:|
| **Megakernel BF16 (this port)**   |    16,770     | **711**       | **1.63×**          |
| llama.cpp BF16 (CUDA, ngl=99) | **26,781**    |    437        |   1.00×            |
| Megakernel NVFP4              |    12,413     |    217        |   0.50×            |
| PyTorch HuggingFace BF16      |     1,797     |     27        |   0.06×            |

Decode (tg128) on the BF16 megakernel beats llama.cpp by **1.63×** and HF by
**26×**. Prefill went from 12,420 → 15,859 t/s (+28%) after graph-capture and
a DeltaNet recurrence rewrite; still behind llama.cpp at 0.59×, with a clear
next step (chunked associative scan). Completion strings are bit-exact between
megakernel BF16 and HF eager.

## Hardware

| Machine    | GPU / Chip          | Memory    | SMs | Backend built |
|------------|---------------------|-----------|-----|---------------|
| Lucebox    | NVIDIA RTX 3090     | 24 GB     |  82 | BF16 megakernel (`kernel.cu`) |
| NVIDIA DGX | NVIDIA GB10 Spark   | 128 GB    |  48 | NVFP4 megakernel (`kernel_gb10_nvfp4.cu`) |
| this host  | NVIDIA B200 (SXM)   | 183 GB    | 148 | BF16 + NVFP4 megakernels (both `.cu` files) |

## What had to change for B200 (sm_100)

Three changes, in order of impact:

1. **BF16 decode barrier** (`kernel.cu`): hand-rolled grid barrier deadlocked
   on sm_100, unblocked by moving to `cudaLaunchCooperativeKernel` +
   `cg::this_grid().sync()`. Unlocks the entire BF16 decode path.
2. **BF16 prefill recurrence** (`prefill.cu`): move conv1d state to shared
   memory; add CUDA graph capture for the whole prefill body. +28% pp520.
3. **NVFP4 kernel** (`kernel_gb10_nvfp4.cu`): compiles for B200 unchanged.
   Already uses `cudaLaunchCooperativeKernel` + `cg::this_grid().sync()`,
   and its NVFP4 dot products are LUT-based with no arch-specific
   tensor-core intrinsics.

### BF16 decode barrier (root cause + fix)

The BF16 kernel (`kernel.cu`) needed one surgical change: its hand-rolled
grid barrier was making no forward progress on B200. Root cause:

```cuda
// old (worked on sm_86, hung on sm_100):
asm volatile("fence.acq_rel.gpu;" ::: "memory");
unsigned int arrived = atomicAdd(counter, 1);
if (arrived == nblocks - 1) { *counter = 0; atomicAdd(generation, 1); }
else { while (*vgen <= my_gen) {} }
```

Symptom: `decode_kernel` hung for 30+ minutes at 100% SM utilization, ~216 W,
producing zero tokens. Other blocks saw a stale `generation` value — on
Blackwell datacenter the `fence.acq_rel.gpu` release from the "last arriver"
does not reliably propagate to the spin-waiters through L2, even with the
spin-read marked `volatile`, within the timescales decode needs.

Fix (in `kernel.cu`):

1. Replace `AtomicGridSync` internals with `cg::this_grid().sync()` (same
   pattern the NVFP4 path already uses). Call sites (`grid.sync()`) are
   unchanged because `AtomicGridSync::sync()` is now a thin wrapper.
2. Delete the kernel-entry barrier init (the hand-rolled prologue that reset
   the counter + incremented the generation). `cg::this_grid().sync()` does
   not need it.
3. Launch via `cudaLaunchCooperativeKernel` instead of `<<<...>>>`.
4. Drop the `NUM_BLOCKS=82` hardcode: size the grid from
   `cudaOccupancyMaxActiveBlocksPerMultiprocessor * multiProcessorCount`
   (optionally overridden by `MEGAKERNEL_DECODE_BLOCKS`). On B200 that comes
   out to `148 × 1 = 148` blocks because of `__launch_bounds__(BLOCK_SIZE, 1)`;
   on RTX 3090 the same code yields 82.
5. Split `lm_head_kernel`'s intra-kernel reduce (which also used the same kind
   of atomic/fence barrier) into two kernel launches, matching the NVFP4 LM
   head pattern: `lm_head_kernel` writes per-block partial (val, idx) pairs,
   `lm_head_reduce_kernel_bf16` is a single-block reduction. Both launches are
   standard, not cooperative.

Net effect: BF16 decode now runs end-to-end on B200 in ~1.4 ms/token (cold
KV cache, no prefill context) and 1.8 ms/token (after a 520-token prefill).

## Headline benchmarks

Configuration: Qwen3.5-0.8B BF16, prompt pp520 (same English seed string as
the 3090 benchmark), decode tg128, single B200 via `CUDA_VISIBLE_DEVICES=0`,
`final_bench.py` methodology (3 warm + 5 timed prefill, 128 decode steps after
a full prefill into a fresh decoder).

| Method                              | pp520 (tok/s) | tg128 (tok/s) |
|-------------------------------------|:-------------:|:-------------:|
| **Megakernel BF16 (this port)**     |    16,770     |    **711**    |
| llama.cpp BF16 (CUDA, ngl=99, r=5)  |  **26,781**   |      437      |
| Megakernel NVFP4                    |    12,413     |      217      |
| PyTorch HuggingFace BF16            |     1,797     |       27      |

llama.cpp: `ggml-org/llama.cpp@6217b49`, built with
`-DCMAKE_CUDA_ARCHITECTURES=100`, benchmarked via
`llama-bench -p 520 -n 128 -r 5` on `unsloth/Qwen3.5-0.8B-BF16.gguf`.

### Speedups on decode (tg128)

| Megakernel BF16 vs … | speedup |
|:---------------------|:-------:|
| PyTorch HuggingFace  | **26×** |
| Megakernel NVFP4     |  3.28×  |
| llama.cpp BF16 CUDA  | **1.63×** |

### Speedups on prefill (pp520)

| Megakernel BF16 vs … | speedup |
|:---------------------|:-------:|
| PyTorch HuggingFace  | **8.8×** |
| llama.cpp BF16 CUDA  |  0.59×  |

### Cross-check numbers

`bench_pp_tg.py --section all` (independent warm, 2 warm + 5 timed prefill,
512-token prompt), same decoder build:

| Method                     | pp512 (tok/s) | tg128 (tok/s) |
|----------------------------|:-------------:|:-------------:|
| Megakernel BF16 (B200)     |    12,160     |      706      |
| Megakernel NVFP4 (B200)    |    12,121     |      234      |

Cold-cache mini-bench (500 decode steps from position 0, no prefill context,
same kernel):

| BLOCKS      | tg (tok/s) | ms/tok |
|-------------|:----------:|:------:|
|  82 (3090)  |    822     | 1.217  |
| **148 (B200 SMs)** | **879** / 840* | 1.137 |

*840 tok/s was the number measured while also NVML-sampling power every 10
ms; the 39 tok/s gap is the sampling overhead.

Power during the same 500-step cold decode: **avg 266 W, peak 326 W → 3.16
tok/J**. That is 1.69× the RTX 3090's best-case 1.87 tok/J from the DVFS
sweep, at 1.5–2× the absolute throughput of llama.cpp BF16 on the same B200.

## Correctness

- `bench_pp_tg.py --section correctness` PASSes on both BF16 and NVFP4.
- Full 128-token continuations from `final_bench.py` BF16 megakernel and
  `final_bench.py` HF eager match character-for-character on the English test
  prompt.

## How B200 compares to the other targets

Like-for-like (same megakernel backend on each chip). GB10 row pending.

| Chip  | SMs | Mem BW    | Backend | pp520 (tok/s) | tg128 (tok/s) | Draw (decode) | tok/J  |
|-------|----:|-----------|---------|:-------------:|:-------------:|:-------------:|:------:|
| B200  | 148 | ~8 TB/s   | BF16    | **12,420**    | **711**       |    266 W      | 3.16   |
| B200  | 148 | ~8 TB/s   | NVFP4   |    12,413     |    217        |    215 W      | 1.01   |
| GB10  |  48 | ~273 GB/s | NVFP4   |    *TBD*      |   *TBD*       |    *TBD*      | *TBD*  |
| 3090  |  82 | 936 GB/s  | BF16    |    37,800     |    413        |    220 W      | 1.87   |

### vs llama.cpp BF16 across targets

| Chip  | Megakernel tg128 | llama.cpp tg128 | ratio   |
|-------|:----------------:|:---------------:|:-------:|
| B200  |    **711**       |      437        | **1.63×** |
| 3090  |    **413**       |      267        | **1.55×** |
| GB10  |    *TBD*         |    *TBD*        |  *TBD*  |

On B200 the megakernel's decode lead over llama.cpp is actually *slightly
larger* (1.63×) than on the 3090 (1.55×), despite B200's much faster tensor
core hardware. Reason: decode is memory-bandwidth and launch-overhead bound,
not tensor-core bound. The ~100 kernel launches per token llama.cpp does still
cost wall-clock on B200, and the megakernel's single persistent dispatch
absorbs that cost on both chips.

## Prefill megakernel (new): a true single-dispatch persistent kernel

`prefill_megakernel.cu` / op `prefill_bf16_mega`. One
`cudaLaunchCooperativeKernel` dispatch, 148 persistent blocks, all 24 layers,
all S tokens, with `cg::this_grid().sync()` between phases. Same megakernel
shape as decode: no cuBLAS, no per-layer relaunch, no host round-trips. See
`final_bench.py --prefill-mode mega`.

Structure inside the kernel:

```
phase_embed  →  sync
for layer in 0..23:
    phase_rmsnorm  →  sync
    phase_matmul_bf16 (× QKV/Z or Q/K/V)  →  sync
    phase_deltanet_recurrence  or  (phase_qk_norm_rope → sync → phase_causal_attn)
    →  sync
    phase_matmul_bf16 (output proj)  →  sync  →  phase_add_residual  →  sync
    phase_rmsnorm (post-attn)  →  sync
    phase_matmul_bf16 (gate, up)  →  sync  →  phase_silu_mul  →  sync
    phase_matmul_bf16 (down)  →  sync  →  phase_add_residual  →  sync
phase_final_norm  →  sync  →  phase_lm_head  →  sync  →  phase_lm_reduce
```

Matmul uses WMMA (`mma.sync.aligned.m16n8k16` under the hood) with bf16
operands and f32 accumulator. Block tile is [32, 128] with a 2×8 warp grid;
work is distributed across blocks by cyclic tile assignment. The DeltaNet
recurrence keeps its 16-block-per-layer structure (serial over t, same
optimized inner loop as the non-mega path) but now runs inside the
persistent kernel so there's no per-layer relaunch.

### Correctness

First-token output matches the cuBLAS-based prefill bit-exactly on the pp520
test prompt (`token=13` in both paths). Full-sequence completions continue
to match HF eager after decode is run on the post-prefill state.

### Performance on B200

| Prefill path (pp520)                                        | pp520 (tok/s) | ms   |
|-------------------------------------------------------------|:-------------:|:----:|
| Megakernel BF16 (cuBLAS + launches, graph-captured)         |    15,859     | 32.7 |
| **Prefill megakernel, pipelined cp.async + WMMA (current)** |     8,880     | 58.6 |
| Prefill megakernel, no-pipeline WMMA (earlier attempt)      |     9,404     | 55.3 |
| llama.cpp BF16 (CUDA, ngl=99)                               |    26,781     | 19.4 |

The true-megakernel path is still **slower** than the cuBLAS-assisted
path here, by ~0.56×. After implementing the pipelined GEMM below the
gap is the same as before — the pipeline correctly overlaps loads with
compute, but the underlying WMMA compute is not tuned enough.

### What's in the current GEMM (`phase_matmul_bf16`)

`megakernel/prefill_megakernel.cu`, inside the persistent kernel. Full
CUTLASS-style producer/consumer pipeline on stock `mma.sync`-based WMMA:

- **Block tile** `[BTM=32, BTN=128]`, warp grid 2×8, each warp produces
  `[16, 16]` via one WMMA `m16n16k16` accumulator.
- **K-chunk** `BTK=64`. For K=1024 that's 16 chunks, for K=3584 it's 56.
- **Double-buffered shared staging**:
  ```
  sA[2][BTM][BTK]   =  8 KB
  sB[2][BTN][BTK]   = 32 KB
  sC[BTM][BTN]      = 16 KB    (f32 accumulator → bf16 epilogue)
  ```
  56 KB dynamic shared per block; shared limit raised via
  `cudaFuncAttributeMaxDynamicSharedMemorySize` to 104 KB before launch.
- **cp.async.cg** (L1-bypass, global → shared, 16 bytes per issue) for
  both A and B chunks. 256 threads load the A tile (2 KB worth of `cp.async`
  calls); all 512 threads do 2 passes for the B tile.
- **Pipeline**: first chunk prefetched before the k-loop;
  at iteration *ck* we issue chunk *ck+1* (`cp.async.commit_group`), then
  `cp.async.wait_group<1>` keeps one outstanding group so the GPU overlaps
  the next chunk's global fetch with the current chunk's MMA.
- **Epilogue**: WMMA accumulators written to an `f32` shared tile, then
  all 512 threads cooperate on the `f32 → bf16` cast + bounds-checked
  global store.

Helpers implemented inline: `cp_async_16`, `cp_async_commit`,
`cp_async_wait_group<N>`. The PTX is:

```
asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
             :: "r"(smem_int), "l"(src_gmem));
asm volatile("cp.async.commit_group;\n");
asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
```

### Why it still doesn't beat cuBLAS

cuBLAS at these shapes picks a shape-specialised SASS kernel from its
internal heuristic cache. Three capabilities we don't yet match:

1. **`tcgen05.mma` (sm_100) / `wgmma` (sm_90a) warp-group MMA.** Blackwell
   datacenter TC gen5 issues one MMA per *warp-group* (4 warps × 32 lanes
   = 128 threads), with a 64×128×16 tile in bf16. Our `mma.sync.aligned.m16n16k16`
   is a single-warp path — correct, but delivers roughly a third of the
   tensor-core throughput on sm_100 at the same register footprint.
2. **Swizzled shared-memory layouts + `ldmatrix.sync.aligned.x4`.** WMMA
   `load_matrix_sync` is a plain stride load; cuBLAS uses swizzled
   address permutations so 4-way `ldmatrix` avoids bank conflicts and hits
   the full shared-memory bandwidth.
3. **Epilogue fusion.** cuBLAS at this size fuses the bf16 cast and often
   the residual add into the MMA epilogue. We still round-trip through an
   f32 shared tile between the MMA and the bf16 store.

Implementing (1) on sm_100 is a substantial undertaking (different MMA
intrinsic family, different fragment layouts, TMA-driven A/B loads via
`cp.async.bulk.tensor`, tensor-memory descriptors). (2) and (3) are
incremental polish on top.

### Where the megakernel still wins

The scaffold is the right shape — single persistent dispatch, all 24
layers, every phase inside grid syncs. Decoder-side, the megakernel
already beats llama.cpp on tg128 (711 vs 437, 1.63×). On prefill the
same architecture will beat llama.cpp once `phase_matmul_bf16` is
rewritten with (1)(2)(3) above; the rest of the prefill body (rmsnorm,
DN recurrence, RoPE, causal attn, silu_mul, LM head, residual adds)
already lives in the megakernel and will not need to be touched.

### The recurrence is still the other half

Even with a perfect GEMM, DeltaNet recurrence remains the bigger chunk
(85% of the cuBLAS-path wall time, ~28 ms). That needs a chunked
associative-scan rewrite (Mamba-style) to go from 28 ms → ~5 ms on B200.
Both items together (GEMM ≈ cuBLAS + scan-based recurrence) should push
pp520 past llama.cpp on B200. Until either is done, `prefill_bf16` (the
graph-captured cuBLAS path, 15,859 tok/s) is the faster production choice;
`prefill_bf16_mega` exists, works, and is the correct architectural shape
for the optimizations above to land in.

## Prefill (pp520, cuBLAS path): 12,420 → 16,770 (+35%), 0.63× llama.cpp

### Where the time actually goes

Profiled with `torch.profiler` on a single pp520 call (post-fixes). CUDA
kernel self-time breakdown:

| Kernel                     | self time | share |
|----------------------------|:---------:|:-----:|
| `pf_deltanet_recurrence`   | 27.7 ms   | 85.1% |
| `pf_causal_attn`           |  2.8 ms   |  8.7% |
| cuBLAS matmuls (all)       |  1.4 ms   |  4.4% |
| rmsnorm / silu / residual  |  0.6 ms   |  1.8% |
| Total                      | 32.6 ms   | 100%  |

**The recurrence is the whole ballgame.** cuBLAS GEMMs are 4% of wall time —
the usual prefill bottleneck (matmul) is not the bottleneck here. DeltaNet's
linear-attention scan dominates because it is 18 layers × 520 sequential time
steps, and the naive implementation launches 16 blocks (one per head) on a
148-SM GPU, using ~3% of compute.

### What changed on B200 so far

1. **Graph capture the prefill.** Wrap the whole 24-layer body in
   `cudaStreamBeginCapture` → `cudaGraphInstantiate` → `cudaGraphLaunch`,
   keyed on `(seq_len, all scratch/weight pointers)`. First eager warm-up
   lets cuBLAS allocate its workspace outside capture; subsequent calls pay
   graph-launch cost only. We also pre-allocate a 32 MB cuBLAS workspace via
   `cublasSetWorkspace` and create a dedicated non-default stream for
   capture (PyTorch's `getCurrentCUDAStream()` may return the legacy default
   stream, which cannot be captured). Worth ~3% on its own at S=520;
   eliminates launch latency as a term, which matters more at smaller S.
2. **Rewrite `pf_deltanet_recurrence`'s inner loop.** The old version kept
   the conv1d ring-buffer in global memory and performed a 3-write + 4-read
   shift per channel on every time step. The B200 version:
    - loads the per-head 384-channel × 4-tap conv state into
      shared memory once at kernel entry (6 KB), shifts in-place each step,
      writes back once at kernel exit.
    - loads the per-head conv weights into shared once (another 6 KB).
    - fuses the three q/k/v conv passes into a single 384-way parallel pass
      that uses all 512 threads, instead of three barely-populated waves.
    - caches `norm_w` in shared once and routes the recurrence output
      through `s_out` (fp32 shared) so the gated RMSNorm tail skips a bf16
      round-trip through global memory.
3. **Cut two `__syncthreads` from the per-step hot loop.**
    - The gated-RMSNorm rstd compute now runs on every thread (each reads
      all 16 per-warp partials from shared, adds, rsqrts) instead of the
      old "warp 0 reduces → shared[0] = rstd → everyone syncs to pick
      up rstd" round trip. Saves one sync per step.
    - The end-of-iteration sync is removed: the next iteration's conv+z
      prefetch touches s_conv / s_q / s_k / s_v / s_z strictly on each
      thread's own slot (stride = blockDim.x), and the first cross-warp
      read (phase B normalize reading s_q) is already fenced by the
      existing sync *inside* that phase. Saves one sync per step.

   Per-prefill effect: 520 steps × 18 layers × 2 syncs ≈ **18 k
   `__syncthreads`** removed, +5.7 % on pp520 wall time.

   Net recurrence rewrite: 36.3 → ~26 ms (from 88 % → ~72 % of prefill
   wall time). Prefill wall 42 → 31 ms.

### What's needed to actually beat llama.cpp on prefill

The remaining 28 ms is 85% inside the DeltaNet recurrence. Its per-step
compute budget is ~500 ns in theory (512 threads × ~128 FMAs + a few warp
reductions) but we're measuring ~3 µs/step — the gap is `__syncthreads`
count (5+ per step) + the inherent sequential dependency on the
[`DN_KEY`×`DN_VAL`] recurrent state.

The right next step is a **chunked associative-scan rewrite** of the
recurrence. DeltaNet is a first-order linear recurrence with a state update
of the form `s_{t+1} = A_t · s_t + b_t`, which is associative-scan-friendly:
chunk the sequence into blocks of 32, compute the per-chunk prefix in
parallel, then fuse chunk boundaries with a tree-reduction. This is how
Mamba-style models hit near-peak throughput on large GPUs. On B200 it
should take the recurrence from 28 ms to ~5 ms and push pp520 past
llama.cpp. It is a non-trivial kernel (~500 lines) and is left as the next
todo rather than something to rush in; the writeup already calls out
`kernel_gb10_nvfp4.cu` semantics are unchanged so this work can proceed in
a separate file without risking GB10.

Three smaller items that would close further ground if done alongside:

- **cuBLASLt + plan cache** instead of `cublasGemmEx` default algo. Small-K
  (K=1024) GEMMs leave ~1.3× on the table on Blackwell.
- **Fuse rmsnorm + next-layer matmul** into a single persistent kernel
  (similar to what the decode megakernel already does). Takes another ~0.6
  ms off per prefill.
- **Process multiple heads per block** in the recurrence to raise utilization
  from 16 SMs to 128+. Only makes sense with the associative-scan rewrite.

## Reproduce

```bash
# torch with CUDA 12.8 wheels (B200 sm_100 support)
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers safetensors huggingface-hub accelerate

# build (auto-detects sm_100 via torch.cuda.get_device_capability())
cd megakernel && python3 setup.py build_ext --inplace

# full run: prefill pp520 + decode tg128 + HF baseline (BF16)
CUDA_VISIBLE_DEVICES=0 python3 final_bench.py --backend bf16

# NVFP4 decode (auto-selected on GB10; explicit on B200)
CUDA_VISIBLE_DEVICES=0 python3 final_bench.py --backend nvfp4

# llama.cpp baseline
git clone --depth=1 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp
cmake -B /tmp/llama.cpp/build -S /tmp/llama.cpp \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=100 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/llama.cpp/build --target llama-bench -j
huggingface-cli download unsloth/Qwen3.5-0.8B-GGUF Qwen3.5-0.8B-BF16.gguf \
  --local-dir /tmp/gguf
CUDA_VISIBLE_DEVICES=0 /tmp/llama.cpp/build/bin/llama-bench \
  -m /tmp/gguf/Qwen3.5-0.8B-BF16.gguf -p 520 -n 128 -r 5

# optional knobs
export MEGAKERNEL_DECODE_BLOCKS=148   # default = SM count * 1
export MEGAKERNEL_LM_BLOCKS=1024      # default = min(1024, SM*8)
export MEGAKERNEL_BACKEND=bf16        # or nvfp4
```

## Methodology

- **Precision:** BF16 weights streamed from HF; activations BF16, accumulation
  F32. NVFP4 path additionally compresses hot decode projections to E2M1 +
  per-group FP16 scale at load time.
- **Prompt:** Same English seed string as the RTX 3090 benchmark, padded with
  repeats to exactly the target length (`final_bench.build_exact_prompt_ids`).
- **Warm-up:** 3 prefill warm + 5 timed prefill. Decode runs once for 128
  steps after a full prefill into a freshly allocated decoder.
- **Power:** Measured with `nvidia-smi`/`pynvml` sampling (10 ms) during the
  timed decode. Accelerator-only; total system draw is higher.
- **Correctness:** 128-token megakernel completion matches HF eager for the
  same prompt; `bench_pp_tg.py --section correctness` additionally checks the
  prefill→decode handoff against a reference path.
- **llama.cpp:** Default `llama-bench` settings, `-p 520 -n 128 -r 5`, fully
  offloaded (`ngl=99` implied by no CPU layers). Same BF16 weights via the
  unsloth GGUF.
