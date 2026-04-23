# Megakernel Qwen3.5-0.8B on NVIDIA B200 (sm_100)

Port of the megakernel backends to Blackwell datacenter (B200, sm_100). Batch
size 1, single-stream decode. GB10 (sm_121a) NVFP4 path and RTX 3090 (sm_86)
BF16 path both still build and still work on their native arches — the B200
changes are additive.

## TL;DR

| Method on B200 (Qwen3.5-0.8B) | pp520 (tok/s) | tg128 (tok/s) | tg128 vs llama.cpp |
|-------------------------------|:-------------:|:-------------:|:------------------:|
| **Megakernel BF16 (this port)**   | 12,420        | **711**       | **1.63×**          |
| llama.cpp BF16 (CUDA, ngl=99) | **26,781**    |    437        |   1.00×            |
| Megakernel NVFP4              | 12,413        |    217        |   0.50×            |
| PyTorch HuggingFace BF16      |  2,100        |     30        |   0.07×            |

Decode (tg128) on the BF16 megakernel beats llama.cpp by **1.63×** and HF by
**23.5×**. Prefill is still behind llama.cpp (0.46×) — see the prefill section.
Completion strings are bit-exact between megakernel BF16 and HF eager.

## Hardware

| Machine    | GPU / Chip          | Memory    | SMs | Backend built |
|------------|---------------------|-----------|-----|---------------|
| Lucebox    | NVIDIA RTX 3090     | 24 GB     |  82 | BF16 megakernel (`kernel.cu`) |
| NVIDIA DGX | NVIDIA GB10 Spark   | 128 GB    |  48 | NVFP4 megakernel (`kernel_gb10_nvfp4.cu`) |
| this host  | NVIDIA B200 (SXM)   | 183 GB    | 148 | BF16 + NVFP4 megakernels (both `.cu` files) |

## What had to change for B200 (sm_100)

The NVFP4 kernel (`kernel_gb10_nvfp4.cu`) compiles for B200 unchanged — it
already uses `cudaLaunchCooperativeKernel` + `cg::this_grid().sync()`, and its
NVFP4 dot products are LUT-based with no arch-specific tensor-core intrinsics.

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
| **Megakernel BF16 (this port)**     |    12,420     |    **711**    |
| llama.cpp BF16 (CUDA, ngl=99, r=5)  |  **26,781**   |      437      |
| Megakernel NVFP4                    |    12,413     |      217      |
| PyTorch HuggingFace BF16            |     2,100     |       30      |

llama.cpp: `ggml-org/llama.cpp@6217b49`, built with
`-DCMAKE_CUDA_ARCHITECTURES=100`, benchmarked via
`llama-bench -p 520 -n 128 -r 5` on `unsloth/Qwen3.5-0.8B-BF16.gguf`.

### Speedups on decode (tg128)

| Megakernel BF16 vs … | speedup |
|:---------------------|:-------:|
| PyTorch HuggingFace  | **23.5×** |
| Megakernel NVFP4     |  3.28×  |
| llama.cpp BF16 CUDA  | **1.63×** |

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

## Prefill (pp520): still behind llama.cpp, 0.46×

The B200 BF16 prefill path (`prefill.cu`) is cuBLAS `cublasGemmEx` +
`COMPUTE_32F` per-layer plus ~10 small custom kernels per layer (rmsnorm,
silu_mul, residual, rope, causal attention, recurrence). 24 layers × ~15
kernel launches ≈ 360 launches for a 520-token prefill, each cuBLAS call
picking a default algo. On B200 the small-K (K=1024, N=1024 or 3584) GEMMs
don't saturate tensor cores, and the launch burst dominates over the compute.
llama.cpp gets ~2× the prefill throughput because it pipes everything through
ggml + cudaGraphs with FlashAttention-style fused kernels.

Paths to close the gap (none done yet):

1. **cuBLASLt + heuristic cache.** Replace `cublasGemmEx` with `cublasLtMatmul`,
   pick a good algo per shape once and cache the plan. On small-K, this
   commonly buys 1.3–1.6× on Blackwell.
2. **Graph capture the prefill.** Wrap the 24-layer prefill in a
   `cudaStreamBeginCapture` → `cudaGraphInstantiate` → `cudaGraphLaunch` once
   per `max_seq_len` bucket. Eliminates launch latency, which is the dominant
   cost at 520 tokens.
3. **Fuse rmsnorm + matmul + bias/residual into a single custom kernel.**
   Same trick the decode path uses, applied to prefill. Most of the payoff
   comes from items 1 + 2, though.

For the target "faster than llama.cpp at decode," those are future work.

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
