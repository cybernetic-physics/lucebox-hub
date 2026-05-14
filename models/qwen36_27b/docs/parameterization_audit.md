# Parameterization audit — `models/qwen35_0p8b/*.cu` for 27B support

Every `constexpr` in the 0.8B kernel files that encodes a model dimension
and needs to become parameterized to support 27B. Source of truth on the
right is the Qwen3.6-27B config (HF model card).

## Per-file inventory

| file | line | constant | 0.8B value | 27B value | parameter source |
|---|---:|---|---:|---:|---|
| `kernel.cu`              |  24 | `HIDDEN_SIZE`        |  1024 |  5120 | `ModelConfig::hidden`            |
| `kernel.cu`              |  25 | `INTERMEDIATE_SIZE`  |  3584 | 17408 | `ModelConfig::intermediate`      |
| `kernel.cu`              |  26 | `NUM_LAYERS`         |    24 |    64 | `ModelConfig::num_layers`        |
| `kernel.cu`              |  28 | `VOCAB_SIZE`         | 248320| 248320 | **same** — keep constexpr        |
| `kernel.cu`              |  31 | `FA_NUM_Q_HEADS`     |     8 |    24 | `ModelConfig::fa_q_heads`        |
| `kernel.cu`              |  32 | `FA_NUM_KV_HEADS`    |     2 |     4 | `ModelConfig::fa_kv_heads`       |
| `kernel.cu`              |  33 | `FA_HEAD_DIM`        |   256 |   256 | **same** — keep constexpr        |
| `kernel.cu`              |  39 | `FA_ROTARY_DIM`      |    64 |    64 | **same** — keep constexpr        |
| `kernel.cu`              |  40 | `FA_ROPE_THETA`      |   1e7 |   1e7 | **same** — keep constexpr        |
| `kernel.cu`              |  43 | `DN_NUM_HEADS` (V/QK)|    16 |  48/16 | **split**: `ModelConfig::dn_v_heads`, `ModelConfig::dn_qk_heads` |
| `kernel.cu`              |  44 | `DN_KEY_DIM`         |   128 |   128 | **same** — keep constexpr        |
| `kernel.cu`              |  45 | `DN_VALUE_DIM`       |   128 |   128 | **same** — keep constexpr        |
| `kernel.cu`              |  46 | `DN_CONV_KERNEL`     |     4 |     4 | **same** — keep constexpr        |
| `kernel.cu`              |  ?? | `LAYER_TYPE[24]`     | repeating 0,0,0,1 | repeating 0,0,0,1, **length 64** | length comes from `num_layers`, pattern is invariant |
| `prefill_megakernel.cu`  |  32-55 | (same family, prefixed `FA_`/`DN_` w/o `_DIM`) | same | same | mirror `kernel.cu` |
| `prefill_megakernel.cu`  |  71-84 | `WM=WN=WK=16`, `BTM=32`, `BTN=128`, `BTK=64`, `WARPS_M=2`, `WARPS_N=8` | tuning | tuning | **keep constexpr** — these are GEMM tile params, retuning happens in Phase 2 against 27B's 5120/17408 shapes |
| `prefill_megakernel.cu`  |  37 | `MAX_SEQ` (prefill batch)|  2048 | runtime | **runtime** — context can vary |
| `prefill_bw.cu`          |  60-97 | (same family as kernel.cu) | same | same | mirror `kernel.cu` |
| `prefill_bw.cu`          |  31 | `PREFILL_DN_BLOCKS_PER_HEAD` |  8 |  8 (re-tune) | **keep #define**, retune later |
| `prefill_bw.cu`          |  90-93 | `NVFP4_TC_*` block scaling   | same | same | **same** — NVFP4 layout is model-independent |
| `kernel_gb10_nvfp4.cu`   |  36-69 | (same family) | same | same | mirror `kernel.cu` |
| `kernel_gb10_nvfp4.cu`   |  80 | `LM_BLOCK_SIZE`      |   256 |   256 | tuning |
| `kernel_gb10_nvfp4.cu`   |  92-125 | `ROPE_INV_FREQ[FA_ROTARY_DIM/2]` (precomputed cosines) | 32 | 32 | **same** — head_dim and rotary_dim unchanged |

## What's actually parameter-equivalent across 0.8B and 27B

A small subset of constants are **identical** between the two models and
should stay as `constexpr` — they're effectively part of the Qwen3.x
family invariant:

```
FA_HEAD_DIM      = 256
FA_ROTARY_DIM    = 64
FA_ROPE_THETA    = 1e7
DN_KEY_DIM       = 128
DN_VALUE_DIM     = 128
DN_CONV_KERNEL   = 4
DN_QK_HEADS      = 16     # (0.8B: 16, 27B: 16 — note 27B splits into 48 V / 16 QK)
VOCAB_SIZE       = 248320
LAYER_PATTERN    = (0, 0, 0, 1)  # DN DN DN FA — repeats every 4
```

The actually-varying parameters between 0.8B and 27B reduce to:

```
ModelConfig {
    int num_layers;       // 24 or 64
    int hidden;           // 1024 or 5120
    int intermediate;     // 3584 or 17408
    int fa_q_heads;       // 8 or 24
    int fa_kv_heads;      // 2 or 4
    int dn_v_heads;       // 16 or 48      (only the V branch changes)
    int rope_scaling;     // 1 or YaRN (27B-only)
};
```

7 ints. The rest is identical.

## Recommended refactor approach

**Option A — Template specialization** (preferred for hot path):
Wrap each kernel in a template `<typename Cfg>` and provide
`Cfg_0p8B`, `Cfg_27B` as compile-time tag types. nvcc emits one cubin
per spec; both run at full inlined-constexpr speed. Cost: every kernel
becomes a template; cubin size roughly 2× (acceptable).

**Option B — `__constant__` memory config** (simpler):
A single `__constant__ ModelConfig cfg;` populated by host at load
time. Kernels read `cfg.hidden` etc. Cost: a few extra `LDC.32` ops per
loop iteration but `__constant__` is broadcast-fetch + L1-hit cheap
(~tens of cycles per load, amortized).

**Recommendation: Option A** for the inner FA/DN math, **Option B** for
the outer prefill-megakernel loop variables (which are accessed once per
layer transition, not per element).

For now, keep both models as separate TUs and template the device
functions in `kernel.cu` etc. on a `Cfg` tag struct. Existing 0.8B path
gets `Cfg_0p8B`, the 27B path will get `Cfg_27B`. The `<Cfg>` template
parameter threads through the device-side call graph but the launcher
APIs (host-side) stay model-agnostic.

## Files NOT requiring touch

- `nvfp4_kv.cuh` — head_dim=256 hardcoded but same on both models. KV
  grouping is independent of model size. **No changes needed.**
- `nvfp4_kv_test.cu` — model-agnostic. **No changes needed.**
- `dn_chunked_3090.cu` — sm_86 fallback. Update later in a separate
  pass if needed for 27B-on-3090 inference (will need NVFP4 weights to
  fit).

## Estimated effort

Phase 1 refactor: **1–2 days** for an experienced CUDA dev. The bulk is
mechanical: open each `constexpr int FOO = N` site and replace with
`Cfg::FOO`. Touch points are ~50 lines across 5 files. Verification:
run all existing 0.8B tests (`bench_lora_e2e.py`, perplexity, etc) and
confirm bit-equality with the current main-branch outputs.
