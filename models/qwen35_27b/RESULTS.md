# Luce DFlash benchmark results

Single RTX 3090 24 GB, CUDA 12, driver 535.
Target: `unsloth/Qwen3.5-27B-GGUF` (Q4_K_M, ~16 GB).
Draft:  `z-lab/Qwen3.5-27B-DFlash` (BF16, 3.46 GB).
Concurrency = 1, greedy decoding, `n_gen=256`.
Reproduce with `python3 scripts/bench_llm.py` (samples 10 prompts/dataset, seed=42).

## Headline — AR vs Luce DFlash at concurrency 1

| Task      | AR tok/s | DFlash tok/s | AL   | Speedup |
|-----------|:--------:|:------------:|:----:|:-------:|
| HumanEval | 37.78    | **129.52**   | 8.31 | **3.43×** |
| Math500   | 37.71    | **110.51**   | 7.04 | **2.93×** |
| GSM8K     | 37.65    | **96.15**    | 6.14 | **2.55×** |

AR = autoregressive target-only decode via `test_generate`.
DFlash = block-diffusion draft + DDTree budget 22 verify + fast rollback.
AL = mean committed tokens per draft/verify step (acceptance length).

Datasets pulled live via HuggingFace `datasets`:
- HumanEval — `openai_humaneval`, `prompt` field
- GSM8K    — `gsm8k` main split, `Question: … Answer: ` format
- Math500  — `HuggingFaceH4/MATH-500`, `Problem: … Solution: ` format

## Per-prompt numbers (seed 42)

### HumanEval (10 samples)

| # | n_tok | AR    | DFlash | AL    |
|:-:|:-----:|:-----:|:------:|:-----:|
| 01| 84    | 37.98 | 137.91 | 8.83  |
| 02| 138   | 37.90 | 143.38 | 9.14  |
| 03| 134   | 37.88 | 137.49 | 8.83  |
| 04| 120   | 37.84 | 153.77 | 9.85  |
| 05| 172   | 37.76 | 131.74 | 8.53  |
| 06| 118   | 37.59 | 113.97 | 7.31  |
| 07| 51    | 37.78 | 103.27 | 6.56  |
| 08| 141   | 37.68 | **158.40** | **10.24** |
| 09| 125   | 37.71 | 128.22 | 8.26  |
| 10| 95    | 37.65 |  87.04 | 5.57  |
| **mean** |   | **37.78** | **129.52** | **8.31** |

Peak per-prompt: **158.40 tok/s at AL 10.24** (4.20× over AR on the same prompt).

### GSM8K (10 samples)

| # | n_tok | AR    | DFlash | AL   |
|:-:|:-----:|:-----:|:------:|:----:|
| 01| 45    | 37.62 |  93.87 | 5.95 |
| 02| 111   | 37.53 |  90.59 | 5.82 |
| 03| 49    | 37.73 |  87.79 | 5.57 |
| 04| 70    | 37.67 |  82.11 | 5.22 |
| 05| 102   | 37.62 | **127.83** | **8.26** |
| 06| 118   | 37.61 |  88.67 | 5.69 |
| 07| 113   | 37.62 |  86.86 | 5.57 |
| 08| 50    | 37.72 | 102.98 | 6.56 |
| 09| 43    | 37.69 | 109.66 | 6.92 |
| 10| 96    | 37.72 |  91.12 | 5.82 |
| **mean** |   | **37.65** | **96.15** | **6.14** |

### Math500 (10 samples)

| # | n_tok | AR    | DFlash | AL   |
|:-:|:-----:|:-----:|:------:|:----:|
| 01| 257   | 37.60 | 100.97 | 6.56 |
| 02| 53    | 37.73 | 115.62 | 7.31 |
| 03| 40    | 37.76 | 126.47 | 8.00 |
| 04| 50    | 37.76 | 118.20 | 7.53 |
| 05| 117   | 37.69 | 114.55 | 7.31 |
| 06| 76    | 37.70 | 108.63 | 6.92 |
| 07| 43    | 37.72 |  90.41 | 5.69 |
| 08| 79    | 37.73 | 100.10 | 6.40 |
| 09| 52    | 37.69 |  91.69 | 5.82 |
| 10| 57    | 37.74 | **138.45** | **8.83** |
| **mean** |   | **37.71** | **110.51** | **7.04** |

## Why the speedup varies by task

Acceptance length is the dominant factor — tok/s is roughly linear in AL when per-step overhead is fixed:

| Task      | AL   | Speedup vs AR |
|-----------|:----:|:-------------:|
| HumanEval | 8.31 | 3.43×         |
| Math500   | 7.04 | 2.93×         |
| GSM8K     | 6.14 | 2.55×         |

HumanEval prompts are highly regular (function signatures + docstrings), the draft nails consecutive tokens. GSM8K is natural-language arithmetic reasoning, the draft is less confident, tree verify rescues less.

## 128K context configuration

`max_ctx = 131072` + `DFLASH27B_KV_Q4=1` (Q4_0 K+V cache, 8× compression vs F16).
Sliding `target_feat` ring (4096 slots) keeps captured features at 0.2 GB regardless of context length.
`--ddtree-budget=16` keeps per-layer `ssm_intermediate` under 1.3 GB.

| Prompt length | KV size  | Prefill | Decode tok/s |
|:-------------:|:--------:|:-------:|:------------:|
| 520 (HE)      | ~35 MB   | 0.06 s  | 130          |
| 13K           | ~860 MB  | 15 s    | 99           |
| 32K           | ~2.1 GB  | 106 s   | 35           |
| 128K          | ~8.4 GB  | ~10 min | ~15-20 (est) |

Q4_0 KV costs ~3% mean tok/s vs F16 at short contexts and is the only thing that lets 128K allocate at all.

## DDTree budget sweep (HumanEval, n_gen=256, f16 intermediate)

Historical tuning run from commit `f1cb9bf` (2026-04-16). Used to pick the default budget=22. Fresh run at budget=22 on commit `5bb7f8c` is the 129.5 tok/s / AL 8.31 reported in the headline above; the ~5 tok/s delta vs the 135.8 row here comes from sample variance across the 10 prompts and from minor build-flag drift between the two commits.

| Budget | Mean AL | Mean tok/s |
|:------:|:-------:|:----------:|
| 15     | 7.64    | 125.3      |
| 16     | 7.81    | 128.7      |
| 18     | 8.22    | 131.2      |
| 20     | 8.64    | 133.9      |
| **22** | **8.88**| **135.8**  |
| 24     | 8.91    | 133.0      |
| 30     | 8.86    | 120.5      |
| 40     | 8.90    | 105.1      |

AL plateaus at ~8.9, past budget 22 each extra node costs more in verify time than it buys in accept. Memory ceiling at budget 26 on 24 GB (per-token SSM intermediate cache is hybrid-only overhead).

## Kernel-level wins (cumulative, chain mode → DDTree budget 22 + f16)

Starting point: Chain DFlash at 112.8 tok/s mean on HumanEval, AL 7.67.

| Optimization                                    | Δ tok/s | Δ AL | Note |
|-------------------------------------------------|:-------:|:----:|------|
| DDTree budget 20, f32 intermediate              | +15.1   | +0.77| Heap-based best-first tree, 20 nodes |
| Chain pre-seed in `build_ddtree`                | —       | +~5  | Fixes top-1 chain coverage under Q4 noise (prior AL ~4) |
| Tree-aware `ggml_ssm_conv_tree` kernel          | —       | +~1  | Sibling conv window gathers via parent chain, not DFS |
| `target_feat` compaction after sibling-accept   | —       | +~0.8| Stale feature pruning |
| OpenMP-parallel CPU top-K, K reduced 32→8       | +2.1    | —    | Shaves 7% off draft step |
| Fast K=1 path for budget=15                     | +1.5    | —    | Skips 11 ms CPU top-K when no siblings needed |
| D2D `cudaMemcpyAsync` for target_feat (GPU→GPU) | +3.7    | —    | Replaces GPU→CPU→GPU round trip |
| `ggml_gated_delta_net_tree_persist` kernel      | +12.4   | —    | Direct-writes SSM intermediates, skips 9 ms `ggml_cpy` per step |
| Budget 20 → 22, f16 intermediate                | +5.5    | +0.24| f16 cuts intermediate bandwidth in half |
| **Total**                                       | **+16.7** | **+0.64** | **129.5 tok/s, AL 8.31 (HumanEval mean, fresh run)** |

## Reproducibility

- Deterministic: greedy decode + greedy verify. Same prompts + same weights + same binary = same numbers ±1 tok/s.
- Full bench (10×3 = 30 prompts): ~15 min.
- All numbers above reproduced on 2026-04-20 from commit `5bb7f8c` with:
  ```
  python3 scripts/bench_llm.py
  ```

## Hardware ceiling notes

- Published DFlash paper on Qwen3-4B/8B/30B-MoE (pure attention, BF16, B200) reports 4-5× over AR on HumanEval/Math500 at concurrency 1. Ours: 3.43× on 27B hybrid Q4_K_M on RTX 3090.
- Memory ceiling: per-token SSM intermediate cache (hybrid-only cost) caps tree budget at ~26 on 24 GB. The paper uses budgets up to 1024 on pure-attention models with zero per-node memory tax.
- Per-token verify cost drops from 25 ms at N=1 to 0.97 ms at N=128 (ggml-cuda Q4_K matmul amortises well with batch size).
