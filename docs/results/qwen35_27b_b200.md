# DFlash27B on NVIDIA B200 (sm_100)

Port of the DFlash + DDTree Qwen3.5-27B inference path (which the original
repo ships for RTX 3090 / sm_86) to Blackwell datacenter (B200, sm_100).

## Build changes

Single line in `CMakeLists.txt`:

```cmake
# Was:
set_target_properties(dflash27b PROPERTIES CUDA_ARCHITECTURES "86")
# Now (parameterised, default sm_100 for B200):
if(NOT DEFINED DFLASH_CUDA_ARCH)
    set(DFLASH_CUDA_ARCH "100")
endif()
set_target_properties(dflash27b PROPERTIES CUDA_ARCHITECTURES "${DFLASH_CUDA_ARCH}")
```

Build:

```bash
cd dflash
git submodule update --init --recursive
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=100 -DCMAKE_BUILD_TYPE=Release
cmake --build build --target test_generate test_dflash -j
```

## Numbers on B200 (Qwen3.5-27B Q4_K_M, HumanEval 5-prompt mean, n_gen=256)

| Mode    | 3090 (blog) | B200 (this port) | Ratio |
|---------|:-----------:|:----------------:|:-----:|
| AR (`test_generate`) |  37.78 tok/s |  **36.78 tok/s** | 0.97× |
| DFlash+DDTree b=22   | 129.52 tok/s |  **80.57 tok/s** | 0.62× |
| Acceptance length    |   8.31       |    8.25          | 0.99× |

AR matches the 3090 within noise. DFlash+DDTree is 38 % slower than the
3090 despite identical draft acceptance (AL 8.25 vs 8.31). All the gap
sits in the per-step target-model forward, **not** in the draft or the
verify/accept logic.

## Why B200 is slower than the 3090 on this workload

Profiled with nsys (`test_generate`, n_gen=64). Top GPU kernels:

| Kernel                                | Share | Notes |
|---------------------------------------|:-----:|:------|
| `mul_mat_vec_q<type=Q4_K, bs=1>`      | 34.7% | Q4_K matvec, `__dp4a` integer dot |
| `mul_mat_vec_q<type=Q6_K, bs=1>`      | 13.8% | Q6_K matvec (subset of K_M) |
| `mul_mat_vec_q<type=Q5_K, bs=1>`      | 10.4% | Q5_K matvec |
| `mul_mat_vec_q<type=Q4_K, bs=1> (v2)` |  7.5% | Same, different flags |
| `quantize_q8_1`                       |  6.3% | Activation quantization for Q_K |
| `rms_norm_f32`                        |  5.0% | |
| `flash_attn_ext_f16`                  |  2.5% | FlashAttn decode |
| other                                 | 19.8% | |

Total `mul_mat_vec_q` + its `quantize_q8_1` feeder = **72.7 %** of GPU
time. That is the scalar-`__dp4a` matvec path ggml uses for
batch_size=1 decode. It was tuned for Turing/Ampere and is
compute-bound (not bandwidth-bound) on B200 — the INT8 dot-product
throughput on B200 is nowhere near its BF16 tensor-core throughput, so
the chip's 8 TB/s HBM and 2.25 PFLOPS BF16 TC are both idle while
`__dp4a` lanes saturate.

ggml *does* have Blackwell-aware MMA paths (`blackwell_mma_available`
in `common.cuh`), but:

1. It only flips on for `GGML_CUDA_CC_BLACKWELL = 1200` (sm_120 —
   consumer Blackwell like RTX 5090). B200 is sm_100; ggml's comment
   explicitly calls this out: *"BW spans CC 1000, 1100 & 1200, we are
   integrating Tensor Core instructions available to 1200 family"*.
2. Even when it flips on, it's currently only used for `GGML_TYPE_MXFP4`,
   not Q4_K / Q5_K / Q6_K.
3. The `mmq` (batched tensor-core) path gates on a batch-size
   threshold. Spec-decode verify batches (budget=22 ⇒ up to 22 tokens
   per target forward) may or may not hit that threshold depending on
   shape — and even when hit, the Blackwell-specific tile sizes still
   need sm_120.

So on B200, with a Q4_K_M-quantized model, ggml falls through to the
old `mul_mat_vec_q` scalar path for every layer of every forward pass.
That path runs at roughly the same speed on a 3090 and a B200 — which
is why our AR number equals the 3090's despite an order-of-magnitude
more available compute.

## Proper path to speed-up

1. **Extend ggml's Blackwell MMA check to sm_100.** The instructions
   it uses (`cutlass` sm_120 GEMMs) may or may not be
   binary-compatible with sm_100a. If they are, this is a one-line
   fix; if not, need a separate sm_100 tile set.
2. **Write a sm_100-specific Q4_K / Q5_K / Q6_K mul_mat_vec kernel**
   that uses `mma.sync` (f16/bf16 accumulate) or, ideally,
   `tcgen05.mma` to do the quantized-weight dequant+matmul via tensor
   cores. This is the real win — the Q4_K weight-read bandwidth is
   ~4 GB/layer, a fraction of B200's 8 TB/s budget; making the
   matmul actually tensor-core-bound would put decode latency close
   to the 2 ms per-token theoretical minimum instead of today's
   ~27 ms.
3. **Force the `mmq` batched path for small batches on sm_100.** Even
   without a full Blackwell-TC rewrite, the Ampere/Ada MMQ
   implementation (`turing_mma_available`) runs on sm_100 and should
   beat `mul_mat_vec_q` by 2-3× at spec-decode batch sizes.

Items 1 & 3 are one-afternoon changes in `ggml-cuda`. Item 2 is the
real multi-day fix. None of the dflash27b C++ code itself (the
~2000 lines in this repo) is on the critical path — the bottleneck
sits one layer below, in ggml's matmul kernel selection.

## Reproduce

```bash
# Prep
git submodule update --init --recursive
cd dflash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=100 -DCMAKE_BUILD_TYPE=Release
cmake --build build --target test_generate test_dflash -j

# Weights (~20 GB total)
huggingface-cli download unsloth/Qwen3.5-27B-GGUF Qwen3.5-27B-Q4_K_M.gguf \
  --local-dir models/
huggingface-cli download z-lab/Qwen3.5-27B-DFlash model.safetensors \
  --local-dir models/draft/

# Smoke run
CUDA_VISIBLE_DEVICES=0 python3 scripts/run.py \
  --prompt "def fibonacci(n):" \
  --draft models/draft/model.safetensors \
  --n-gen 256

# 5-prompt HumanEval bench (uses test_generate + test_dflash directly)
CUDA_VISIBLE_DEVICES=0 python3 scripts/bench_llm.py
```
