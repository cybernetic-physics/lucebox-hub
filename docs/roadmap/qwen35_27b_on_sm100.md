# Qwen3.5-27B + DFlash + DDTree on B200 (sm_100)

What it'll take to push the 27B DFlash path past the RTX 3090 number
(129.5 tok/s HumanEval mean) on a B200. Ordered by blast radius.

## Where we are today

5-prompt HumanEval mean, Qwen3.5-27B Q4_K_M, n_gen=256, ddtree budget=22:

|                        | 3090 (blog) | B200 (current) | ratio |
|------------------------|:-----------:|:--------------:|:-----:|
| AR (`test_generate`)   | 37.78       | 36.78          | 0.97× |
| DFlash+DDTree          | 129.52      | 80.57          | 0.62× |
| Acceptance length      | 8.31        | 8.25           | 0.99× |

The `dflash27b` glue is correct — AR and AL match the 3090. All the
missing 38 % sits in ggml's CUDA matmul kernels.

## Why it's slow: ggml's `GGML_CUDA_CC_BLACKWELL` excludes sm_100

From `deps/llama.cpp/ggml/src/ggml-cuda/common.cuh`:

```cpp
// BW spans CC 1000, 1100 & 1200, we are integrating Tensor Core
// instructions available to 1200 family
#define GGML_CUDA_CC_BLACKWELL 1200
#define GGML_CUDA_CC_DGX_SPARK 1210
#define GGML_CUDA_CC_RUBIN     1300
```

B200 is `cc == 1000`. ggml's Blackwell MMA specialization only fires
for cc ≥ 1200 (RTX 5090 / sm_120). Everything else on B200 falls
through to the Turing/Ampere path — scalar `__dp4a` for matvecs,
Ampere tile sizes for `mmq`.

nsys profile (AR, n_gen=64, top GPU time):

| Kernel                                | Share |
|---------------------------------------|:-----:|
| `mul_mat_vec_q<Q4_K>`                 | 34.7 % |
| `mul_mat_vec_q<Q5_K>`                 | 10.4 % |
| `mul_mat_vec_q<Q6_K>` (+ Q6_K v2)     | 13.8 % |
| `mul_mat_vec_q<Q4_K>` (flag variant)  |  7.5 % |
| `quantize_q8_1` (activation prep)     |  6.3 % |
| `rms_norm_f32`                        |  5.0 % |
| `flash_attn_ext_f16`                  |  2.5 % |
| other                                 | 19.8 % |

DFlash run profile (verify batches, budget=22):

| Kernel                                | Share |
|---------------------------------------|:-----:|
| `mul_mat_q<Q4_K, tile=24>`            | 19.4 % |
| `mul_mat_q<Q4_K, tile=16>`            | 11.2 % |
| `mul_mat_q_stream_k_fixup`            |  7.9 % + 4.9 % |
| `mul_mat_q<Q6_K, tile=24>`            |  5.6 % |
| `mul_mat_q<Q5_K, tile=24>`            |  5.3 % |
| `gated_delta_net_cuda<fp16>`          |  3.0 % |

So the DFlash path *does* reach the `mmq` batched kernels (good — it's
already using tensor cores via Turing MMA tiles), but the tile sizes
are Ampere-era and the Blackwell tcgen05 path is gated off.

## The plan, priority-ordered

### 1. Extend `blackwell_mma_available()` to include sm_100 — 1 afternoon

Single-line change in `common.cuh`, but requires auditing every
`__CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL` block in the mma / mmq
headers to verify the instructions used are actually available on
sm_100 and not sm_120-only.

- `GGML_CUDA_CC_BLACKWELL` = 1200 today. Lower to 1000, then selectively
  `#ifdef __CUDA_ARCH__ >= 1200` inside any code block that uses
  sm_120-specific intrinsics (e.g. MXFP4 native block format).
- The `cutlass` sm_120 MMA shapes used by ggml's Blackwell path are
  *probably* ABI-compatible with sm_100a; verify by compiling with
  `-arch=sm_100a` and inspecting SASS.
- Expected win: small for our Q4_K workload (MXFP4 is the main
  beneficiary today). Useful as a platform foundation for items 2–3.

### 2. Force `mmq` path for small batches on sm_100 — 1 afternoon

`ggml_cuda_should_use_mmq()` gates on batch size. On sm_100 the
`mul_mat_vec_q` path is weak (pure `__dp4a`) and `mul_mat_q` via
Turing MMA tiles is much better even at batch=1. Lower the threshold
for our cc.

- Edit `MMQ_DP4A_MAX_BATCH_SIZE` path: if `cc == 1000 && turing_mma`,
  prefer mmq for batches ≥ 1 on Q4_K/Q5_K/Q6_K types.
- Expected win: **2-3× on AR decode** (37 → 75-100 tok/s) because AR
  decode is the only place where batch=1 matvec fires. Proportionally
  smaller on DFlash since verify is already batched.

### 3. Write a sm_100-specific Q4_K / Q5_K / Q6_K `mul_mat_vec` kernel — 2-3 days

The real win. The current `__dp4a` kernel ignores B200's tensor cores
entirely. A sm_100 kernel that does quant dequant + matmul via
`mma.sync.aligned.m16n8k16` (bf16/fp16 accumulate, tile sizes tuned
for B200's L2 and SM count) should get close to bandwidth-bound
performance.

- New file: `ggml-cuda/mmv-sm100.cu` with `mul_mat_vec_sm100<Q4_K>`,
  `<Q5_K>`, `<Q6_K>`.
- Dispatcher in `ggml-cuda.cu`: if `cc == 1000 && type in {Q4_K, Q5_K,
  Q6_K}`, route to `mmv-sm100`.
- Structure: each warp unpacks a Q_K block (super-block of 256 weights
  with K-byte scales), converts to bf16 in registers, does `mma.sync`
  against an activation tile.
- Expected win: **decode bandwidth-bound instead of compute-bound**.
  At 8 TB/s, 16 GB Q4_K_M streams in ~2 ms per token. Today we're at
  ~27 ms. Margin: up to 10× on AR, proportionally 2-3× on the whole
  DFlash+DDTree loop (which still has non-matmul overhead).

### 4. BF16 path for 27B on B200 — 1 day code, 1 hour weights

B200 has 180 GB HBM. Qwen3.5-27B BF16 is ~54 GB. It fits, with
headroom for a long KV cache. The main hurdles are tooling, not perf:

- `gguf_target_loader.cpp` uses `gguf_init_from_file` (single file).
  Qwen3.5-27B BF16 is distributed as a split GGUF (2 files). Add the
  standard ggml split merge (or use `llama-gguf-split --merge`
  offline).
- Verify the qwen35 graph code handles BF16 weights (it should —
  `target_feat` is already BF16; only the gemm call types change).
- Expected: **native BF16 matmul via cuBLAS or ggml's BF16 mmq.**
  No dequant overhead. Should beat the Q4_K_M path even with items
  1–3 done.
- Caveat: draft cost becomes a larger fraction, since the target side
  accelerates faster than the draft side (BF16 3B draft already on
  tensor cores). DDTree budget may want re-tuning.

### 5. DeltaNet recurrence chunked scan — 3-5 days

The `gated_delta_net_cuda` kernel at 3 % of wall time today isn't
critical on the decode side (single-token recurrence step), but
prefill for long contexts is sequential-over-t and becomes the
bottleneck at ≥16K context.

- The scan rewrite we outlined for the 0.8B prefill (chunked
  associative scan, Mamba-style) applies directly — DeltaNet's
  recurrence is Kronecker-separable on K, so per-chunk `(A_K, b)`
  composes across chunks as `[K,K] × [K,V]` matmuls.
- Code lives in `deps/llama.cpp/src/models/delta-net-base.cpp` and its
  CUDA side under `ggml-cuda/`.
- Expected win: **prefill at 32 K+ context drops from minutes to
  seconds**. Matches z-lab's B200 numbers (the reference target).

### 6. Tuning pass on `mmq` tile sizes — 1 week

Even without a new kernel, the Ampere-era tile sizes are suboptimal
for B200's SM count and L2 size. A sweep over `MMQ_X`, `MMQ_Y`, and
`MMQ_NWARPS` per-type would pick better defaults for cc=1000.

- Auto-tune script: bench all (type, tile) combinations on a
  representative prefill+decode workload, write best per type to a
  config header.
- Expected win: 10-30 % on DFlash verify batches. Useful polish
  after items 1–3 land.

## Critical-path estimate

Do items 1 + 2 + 3 in order. Each validates independently against the
AR baseline (same output, faster). Expected end state:

- **AR**: 37 → 200-300 tok/s (bandwidth-bound on Q4_K_M; far above
  3090).
- **DFlash+DDTree**: 80 → 200+ tok/s (3090: 129.5). Pulls ahead of
  3090 once the target forward stops being the bottleneck.

Item 4 (BF16) is a parallel track — it's a larger model but easier
kernel story. Good fallback if items 1–3 run into CUDA-compat issues
on sm_100.

Items 5 and 6 are follow-ups once the AR ceiling is lifted.

## Deliverables, acceptance criteria

Per item above: (1) builds on B200 with `-DCMAKE_CUDA_ARCHITECTURES=100`,
(2) bit-exact output tokens vs reference on HumanEval 5-prompt,
(3) `scripts/bench_llm.py` HumanEval mean strictly faster than prior
commit, (4) one commit per item with the before/after numbers in the
message.

## Where the changes land

- Items 1, 2, 3, 6 are patches on the pinned `luce-dflash` branch of
  the llama.cpp fork (`deps/llama.cpp/ggml/src/ggml-cuda/`). Up-stream
  later if the maintainers want them.
- Item 4 is split across `dflash27b/src/gguf_target_loader.cpp` and
  the llama.cpp fork's BF16 kernels.
- Item 5 lives primarily in the llama.cpp fork's `ggml-cuda/` and
  `src/models/delta-net-base.cpp`.

Nothing in the dflash27b glue code itself is on the critical path.
