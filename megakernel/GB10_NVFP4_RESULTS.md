# GB10 NVFP4 Results

This note records the current GB10-focused NVFP4 results for the Luce
megakernel path on `Qwen/Qwen3.5-0.8B`, including the refreshed side-by-side
comparison against `llama.cpp` on the same machine.

Date of latest update: `2026-04-22`

## Environment

- GPU: `NVIDIA GB10`
- Compute capability: `12.1`
- Practical target: `sm_121a`
- CUDA toolkit: `13.2.78`
- Torch env: `/home/sparkz/dreamzero/.venv-cu132-src`
- Luce repo: `/home/sparkz/lucebox-hub`
- llama.cpp repo: `/home/sparkz/llama.cpp`

## What changed in this pass

The main performance change in this pass was moving the LM head onto a real
Blackwell tensor-core path using raw `cuBLASLt` block-scaled FP4.

The working shape is:

- A: LM-head weights in `CUDA_R_4F_E2M1`
- B: runtime hidden tile quantized to `CUDA_R_4F_E2M1`
- C/D: `CUDA_R_16F`
- compute: `CUBLAS_COMPUTE_32F`
- scales:
  - `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3` for A
  - `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3` for B

The bug that originally blocked this path was not the matmul descriptor. It
was the LM hidden quantizer launch:

- `LM_HEAD_TC_N = 16`
- the quantizer launch used `rows / 128`
- for `rows = 16`, that produced a zero-dimension launch

That left CUDA in an error state and made the LM-head code fall back to the
scalar NVFP4 path.

The critical fix was changing the launch to ceil-div and then validating that
the live path reports:

```text
lm_head_cublaslt: used=1
```

The prompt-side change after that was in the DeltaNet prefill path. The old
prefill recurrence only launched `16` blocks total, one per DeltaNet head,
which left most of the GB10 idle during prompt processing. The current prompt
path now:

- targets `sm_121a` explicitly in the extension build
- tiles the DeltaNet recurrence across value channels
- launches `4` blocks per head in the hot recurrence
- separates the post-recurrence RMSNorm+gate step into its own CUDA kernel so
  the tiled recurrence stays mathematically exact

The best validated setting on this GB10 was:

- `PREFILL_DN_BLOCKS_PER_HEAD = 4`
- `PREFILL_DN_BLOCK_SIZE = 256`

That moved the focused prompt benchmark from the earlier `pp520 ≈ 11.3k tok/s`
range into the `16k tok/s` range on clean runs.

On top of that, the current hybrid prefill path now uses CUDA graph replay for
repeated prompt lengths. The captured sequence keeps the existing CUDA/cuBLAS
prompt schedule, but replays it as a persistent graph instead of rebuilding the
same layer-by-layer launch train every time. This is enabled by default for the
NVFP4 hybrid prompt path and can be disabled with:

```bash
MEGAKERNEL_PREFILL_GRAPH=0
```

The latest prompt-side cut after graph replay was reducing the number of BF16
projection GEMMs that the hybrid prefill path launches per layer. Instead of
running separate `q+k+v`, `qkv+z`, and `gate+up` matmuls, the current code now:

- pre-builds prompt-only fused BF16 weights for:
  - attention `qkv`
  - DeltaNet `qkvz`
  - MLP `gate+up`
- uses fewer, larger GEMMs during prefill
- teaches the consumer CUDA kernels to read those fused row-strided outputs
  directly, instead of paying split-copy kernels

That change moved the current prompt benchmark from the high-`15k` range to the
low-`16k` range on the same machine and workload, without regressing decode.

The next prompt-side win after that came from the CUDA kernels themselves, not
from more launch plumbing:

- the DeltaNet recurrence now writes its per-token tile outputs directly from
  the producing warp instead of staging them through shared memory and paying an
  extra full-block barrier every token
- the causal-attention inner loop now processes BF16 values in pairs, reducing
  the scalar conversion/load overhead in the `Q·K` and `P·V` loops

That pushed the current prompt path from the low-`16k` range into the
`17.3k tok/s` range on clean runs while keeping decode flat.

The latest prompt-side pass tightened the hottest custom CUDA work again:

- `pf_deltanet_recurrence_tiled<128, 8>` now updates its 4-tap conv history
  with aligned `float4` loads/stores, keeps normalized `q` / `k` values in
  registers inside the inner recurrence loop, and uses `fmaf()` in the
  state-update math
- the default `PREFILL_FA_BLOCK_SIZE=256` attention path now maps one CTA to
  one prompt position and stages K/V tiles in shared memory, so the grouped
  query heads sharing a KV head stop rereading the same K/V rows from global
  memory during prefill

The best validated setting for that new attention path on this GB10 was a
shared K/V tile of `8` prompt positions. A larger `16`-position tile regressed
throughput and was backed out.

The next pass after that removed another full-block phase from the DeltaNet
recurrence and tightened the prompt epilogue kernels:

- the value-side conv path inside `pf_deltanet_recurrence_tiled<128, 8>` is now
  warp-local, so each warp updates and consumes its own `v` tile directly in
  registers instead of paying a CTA-wide barrier for work that never leaves the
  warp
- `pf_deltanet_finalize`, `pf_add_residual_bf16`, `pf_rmsnorm`, and
  `pf_silu_mul_fused_bf16` now operate on BF16 pairs where possible, which
  materially cut the prompt-side epilogue cost on this GB10

The next larger prompt-side cut was splitting DeltaNet q/k preparation out of
the tiled recurrence:

- `pf_deltanet_prepare_qk` now runs once per head for the whole prompt, updates
  the q/k conv state once, normalizes q/k once, and writes the normalized q/k
  vectors into scratch
- `pf_deltanet_recurrence_tiled<128, 8>` now consumes those precomputed q/k
  vectors instead of redoing the q/k conv + norm work in every value tile
- on this GB10, the best launch for that new q/k kernel was `256` threads per
  head; `128` was slower, and the old recurrence-only structure was slower than
  either split version
- the winning q/k version goes one step further and gives each thread ownership
  of one q/k channel for the whole prompt, so the 4-tap conv history and conv
  weights stay in registers and shared memory is only used for the per-head q
  and k norm reductions
- I also tried the same register-owned rewrite for the value-side conv path
  inside `pf_deltanet_recurrence_tiled<128, 8>`, but that one was not a stable
  win in the repo benchmark and was backed out

## Current Luce results

### Repo benchmark

Command:

```bash
env HF_HOME=/home/sparkz/hf_cache \
    MEGAKERNEL_BACKEND=nvfp4 \
    /home/sparkz/dreamzero/.venv-cu132-src/bin/python \
    /home/sparkz/lucebox-hub/megakernel/final_bench.py \
    --skip-hf
```

Validated result:

- `pp520 = 19479 tok/s`
- `tg128 = 183 tok/s`
- prompt time: `26.7 ms`
- decode time: `699.5 ms`

Approximate combined `520 + 128` latency:

- `726.2 ms`

### Focused pp / tg harness

Command:

```bash
env HF_HOME=/home/sparkz/hf_cache \
    MEGAKERNEL_BACKEND=nvfp4 \
    /home/sparkz/dreamzero/.venv-cu132-src/bin/python \
    /home/sparkz/lucebox-hub/megakernel/bench_pp_tg.py \
    --section pp --prompt-tokens 520
```

Validated result:

- `pp520 = 20567.1 tok/s`
- `25.3 ms`

## Prompt tensor-core projection pass

The next stable prompt-side gain came from moving the prompt projection GEMMs
onto raw `cuBLASLt` block-scaled NVFP4 tensor cores while keeping the
numerically sensitive DeltaNet `beta/alpha` control projections in BF16.

Two details matter for the working path:

- the prompt-side `cuBLASLt` FP4 helper now writes BF16 output directly instead
  of staging FP16 and converting afterward
- `MEGAKERNEL_PREFILL_TC=1` now enables only the validated prompt projection
  tensor-core path by default; `MEGAKERNEL_PREFILL_TC_GATE_UP=1` remains
  experimental and is still not used for the stable headline

### Prompt-only benchmark

Command:

```bash
env HF_HOME=/home/sparkz/hf_cache \
    MEGAKERNEL_BACKEND=nvfp4 \
    MEGAKERNEL_PREFILL_TC=1 \
    /home/sparkz/dreamzero/.venv-cu132-src/bin/python \
    /home/sparkz/lucebox-hub/megakernel/bench_pp_tg.py \
    --section pp --prompt-tokens 520 --measure-runs 3
```

Validated result:

- `pp520 = 22411.4 tok/s`
- `23.2 ms`

Relative to the previous stable non-TC prompt path (`20567.1 tok/s`), that is
about `+8.9%` prompt throughput.

### Repo headline benchmark

Command:

```bash
env HF_HOME=/home/sparkz/hf_cache \
    MEGAKERNEL_BACKEND=nvfp4 \
    MEGAKERNEL_PREFILL_TC=1 \
    /home/sparkz/dreamzero/.venv-cu132-src/bin/python \
    /home/sparkz/lucebox-hub/megakernel/final_bench.py --skip-hf
```

Validated result:

- `pp520 = 22696 tok/s`
- `tg128 = 183 tok/s`
- combined `520 + 128` latency: about `723.9 ms`

### Decode-only benchmark

Command:

```bash
env HF_HOME=/home/sparkz/hf_cache \
    MEGAKERNEL_BACKEND=nvfp4 \
    /home/sparkz/dreamzero/.venv-cu132-src/bin/python \
    /home/sparkz/lucebox-hub/megakernel/bench_pp_tg.py \
    --section tg --json-result
```

Validated result:

- `tg128 = 197.3 tok/s`
- `648.9 ms` total in that harness

This harness is still useful for relative decode progression, but the current
headline number for the repo is the `final_bench.py` result above.

### Prompt-side profiler delta

On the q/k-split build before the register-owner rewrite, `torch.profiler`
reported:

- `pf_deltanet_recurrence_tiled<128, 8>`: `8.331 ms`
- `pf_deltanet_prepare_qk<256>`: `5.771 ms`
- total self CUDA in the profiled prefill: `27.687 ms`

On the current winning q/k-precompute build, the same profiler flow reported:

- `pf_deltanet_recurrence_tiled<128, 8>`: `8.249 ms`
- `pf_deltanet_prepare_qk<256>`: `3.545 ms`
- total self CUDA in the profiled prefill: `25.416 ms`

So the main gain in this pass was not a dramatic recurrence change; it was
cutting `pf_deltanet_prepare_qk<256>` down by about `38.6%` and lowering total
prompt-side CUDA time by about `8.2%` in that profiler sample.

## Fresh llama.cpp re-measure

Official checkout used:

- repo: `https://github.com/ggml-org/llama.cpp`
- local path: `/home/sparkz/llama.cpp`
- commit: `0d0764dfd257c0ae862525c05778207f87b99b1c`

Model used:

- GGUF: `/home/sparkz/models/gguf/Qwen3.5-0.8B-BF16.gguf`

Important caveat:

- this is `llama.cpp` BF16 GGUF, not NVFP4
- official `llama.cpp` did not have an NVFP4 GGUF export path for this run

### Prompt processing

Command:

```bash
/home/sparkz/llama.cpp/build/bin/llama-bench \
  -m /home/sparkz/models/gguf/Qwen3.5-0.8B-BF16.gguf \
  -o jsonl -r 5 -ngl 99 -fa 1 -p 520 -n 0 -b 2048 -ub 520
```

Result:

- `pp520 = 14150.99 tok/s`
- `36.89 ms`

### Token generation

Command:

```bash
/home/sparkz/llama.cpp/build/bin/llama-bench \
  -m /home/sparkz/models/gguf/Qwen3.5-0.8B-BF16.gguf \
  -o jsonl -r 5 -ngl 99 -fa 0 -p 0 -n 128
```

Result:

- `tg128 = 135.10 tok/s`
- `947.42 ms`

### Combined prompt + generation

Command:

```bash
/home/sparkz/llama.cpp/build/bin/llama-bench \
  -m /home/sparkz/models/gguf/Qwen3.5-0.8B-BF16.gguf \
  -o jsonl -r 5 -ngl 99 -fa 1 -b 2048 -ub 520 -pg 520,128
```

Result:

- `pg520,128 = 639.74 tok/s`
- `1012.91 ms`

## Current comparison

| workload | Luce NVFP4 | llama.cpp BF16 | winner |
| --- | ---: | ---: | --- |
| `pp520` | `22696 tok/s` | `14150.99 tok/s` | `Luce NVFP4` |
| `tg128` | `183 tok/s` | `135.10 tok/s` | `Luce NVFP4` |
| combined `520 + 128` latency | `723.9 ms` | `1012.9 ms` | `Luce NVFP4` |

Relative to the clean refreshed `llama.cpp` run:

- Luce prompt ingest is about `60.4%` faster
- Luce decode is about `35.5%` faster
- Luce total `520 + 128` latency is about `28.5%` lower

## Current conclusion

On the current code:

- Luce NVFP4 is now ahead of the refreshed BF16 `llama.cpp` run on prompt
  ingest
- Luce NVFP4 remains clearly ahead on decode
- Luce NVFP4 is also clearly ahead on end-to-end latency for the `520 + 128`
  workload that matters here

The next practical optimization target is no longer the LM head. The current
prompt-side top hotspot is still the tiled DeltaNet recurrence, followed by the
remaining prompt-side attention and the still-BF16-only `gate_up` path. The
next useful cuts are either:

- another rewrite of the DeltaNet prefill recurrence around shared head-level
  state reuse, or
- fixing and re-enabling the prompt-side `gate_up` NVFP4 tensor-core path, or
- another shared-memory / tiling rewrite of the prefill attention kernel.
