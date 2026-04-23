# GB10 NVFP4 Results

This note records the current GB10-focused NVFP4 results for the Luce
megakernel path on `Qwen/Qwen3.5-0.8B`, including the refreshed side-by-side
comparison against `llama.cpp` on the same machine.

Date of latest update: `2026-04-22` to `2026-04-23`

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

- `pp520 = 10786 tok/s`
- `tg128 = 180 tok/s`
- prompt time: `48.2 ms`
- decode time: `712.9 ms`

Approximate combined `520 + 128` latency:

- `761.1 ms`

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

- `tg128 = 88.0 tok/s`
- `1454.2 ms` total in that harness

This harness is still useful for relative decode progression, but the current
headline number for the repo is the `final_bench.py` result above.

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
| `pp520` | `10786 tok/s` | `14150.99 tok/s` | `llama.cpp` |
| `tg128` | `180 tok/s` | `135.10 tok/s` | `Luce NVFP4` |
| combined `520 + 128` latency | `761.1 ms` | `1012.9 ms` | `Luce NVFP4` |

Relative to the clean refreshed `llama.cpp` run:

- Luce prompt ingest is about `23.8%` slower
- Luce decode is about `33.2%` faster
- Luce total `520 + 128` latency is about `24.9%` lower

## Current conclusion

On the current code:

- `llama.cpp` is still better at prompt ingest on this model
- Luce NVFP4 is now clearly better on decode
- Luce NVFP4 is also clearly better on end-to-end latency for the `520 + 128`
  workload that matters here

The next practical optimization target is no longer the LM head. The next
target is moving more of the projection work off scalar FP4 unpack loops and
onto SM12x tensor-core grouped GEMMs.
