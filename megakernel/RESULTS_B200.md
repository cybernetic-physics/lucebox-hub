# NVIDIA B200 (sm_100) port — Qwen3.5-0.8B BF16

## Headline

```
                            pp520 (tok/s)   tg128 (tok/s)
  Megakernel BF16             40,635            719          (this branch, B200 GPU 0)
  Megakernel NVFP4            29,994            327          (gb10 path on B200)
  PyTorch HuggingFace         12,919             36
  llama.cpp BF16             ~26,800           ~440
```

`final_bench.py --backend bf16` runs the BF16 megakernel pipeline
(`prefill_bf16` + the BF16 decode megakernel). `final_bench.py
--backend nvfp4` runs the gb10 NVFP4 path (`prefill_bf16_nvfp4_lm`
+ LUT-based NVFP4 decode).

## Full shape sweep — fork-parent-b200 vs upstream/main vs SGLang

Methodology: `bench_shapes.py --runs 5 --warmup 3 --decode-tokens 64
--decode-warmup 8`. SGLang via `sglang.bench_one_batch --batch-size 1
--input S --output 64`, taking the steady-state benchmark line (not
the warmup). Both backends are batch-1 single-stream. SGLang reports
`Prefill throughput = S / prefill_latency` and `Decode median
throughput = 1 / median_step_latency`; we read the same way.

### B200 (sm_100, this host, GPU 0)

Prefill (tok/s):

| S | upstream/main | this branch | sglang | best |
|---:|---:|---:|---:|---|
| 32 | 6,701 | **12,360** | 811 | this branch |
| 64 | 9,115 | **20,072** | 1,938 | this branch |
| 128 | 10,865 | **29,687** | 4,124 | this branch |
| 256 | 11,810 | **37,128** | 8,842 | this branch |
| 512 | 12,122 | **37,614** | 14,671 | this branch |
| 1,024 | 12,285 | **37,884** | 30,753 | this branch |
| 2,048 | 11,619 | 32,292 | **61,766** | sglang |
| 4,096 | — | 23,070 | **127,422** | sglang |
| 8,192 | — | 13,411 | **217,503** | sglang |
| 16,384 | — | 8,182 | **201,577** | sglang |
| 32,768 | — | 4,605 | **158,908** | sglang |
| 65,536 | — | **2,449** | n/a | this branch |

Decode (tok/s, after S-token prefill):

| S | upstream/main | this branch | sglang | best |
|---:|---:|---:|---:|---|
| 32 | hang | **919** | 484 | this branch |
| 64 | hang | **903** | 487 | this branch |
| 128 | hang | **872** | 552 | this branch |
| 256 | hang | **819** | 520 | this branch |
| 512 | hang | **730** | 488 | this branch |
| 1,024 | hang | **598** | 441 | this branch |
| 2,048 | hang | 439 | **450** | sglang |
| 4,096 | hang | 230 | **423** | sglang |
| 8,192 | hang | 113 | **367** | sglang |
| 16,384 | hang | 58 | **282** | sglang |
| 32,768 | hang | 30 | **193** | sglang |
| 65,536 | hang | **11** | n/a | this branch |

upstream/main BF16 decode hangs on B200 — its hand-rolled grid
barrier doesn't progress on 148 SMs. The cooperative-grid-sync fix
in this branch is what makes BF16 decode functional on sm_100.

### RTX 3090 (sm_86, vast.ai PCIe 4.0×8, 350 W stock)

Prefill (tok/s):

| S | upstream/main | this branch | sglang | best |
|---:|---:|---:|---:|---|
| 32 | 4,355 | **5,347** | 1,351 | this branch |
| 64 | 6,571 | **9,560** | 2,747 | this branch |
| 128 | 8,453 | **14,795** | 5,285 | this branch |
| 256 | 8,901 | **17,058** | 10,859 | this branch |
| 512 | 8,635 | 15,316 | **20,942** | sglang |
| 1,024 | 8,519 | 14,767 | **35,368** | sglang |
| 2,048 | 7,987 | 13,340 | **41,658** | sglang |
| 4,096 | — | 10,252 | **43,203** | sglang |
| 8,192 | — | 6,805 | **41,434** | sglang |
| 16,384 | — | 4,036 | **37,221** | sglang |
| 32,768 | — | **2,197** | OOM | this branch |
| 65,536 | — | **1,151** | n/a | this branch |

Decode (tok/s, after S-token prefill):

| S | upstream/main | this branch | sglang | best |
|---:|---:|---:|---:|---|
| 32 | 430 | **434** | 335 | this branch |
| 64 | 443 | **444** | 336 | this branch |
| 128 | 439 | **440** | 334 | this branch |
| 256 | 431 | **432** | 334 | this branch |
| 512 | 419 | **419** | 324 | this branch |
| 1,024 | 394 | **395** | 336 | this branch |
| 2,048 | n/a | **439** | 333 | this branch |
| 4,096 | — | 290 | **328** | sglang |
| 8,192 | — | 215 | **320** | sglang |
| 16,384 | — | 141 | **307** | sglang |
| 32,768 | — | **84** | OOM | this branch |
| 65,536 | — | **46** | n/a | this branch |

`upstream/main` columns are capped at S = 2,048 because the upstream
binding hardcodes `FA_KV_HEADS * 2048 * FA_HEAD_DIM` as the KV
stride; S > 2,048 silently corrupts the cache. This branch lifts
that cap (the launcher reads `max_seq` from `fa_k_cache.size(2)`).

## Headlines

- **Short-context prefill is ours.** S ≤ 256 on the 3090 and S ≤
  1,024 on B200 — fork-parent-b200 beats both upstream/main and
  SGLang by 1.6–4×. The persistent-decode + low-launch-overhead
  thesis holds at small S where SGLang's CUDA-graph capture and
  flashinfer setup cost dominates real work.
- **Long-context prefill is SGLang's.** On B200 at S = 8 k SGLang
  does 217 k tok/s vs our 13 k — a 16× gap. cuBLAS sm_100 calls and
  flashinfer's tensor-core attention dwarf our handwritten WMMA in
  this regime. Same shape on the 3090 SGLang wins by 6×.
- **Decode at short context is ours.** ~30 % faster than SGLang on
  the 3090 across S = 32..1,024; ~30–60 % faster on B200 in the same
  range. Persistent decode megakernel is exactly the right tool.
- **Decode at long context is SGLang's.** Past S = 2 k our
  per-step FA scan walks the whole KV cache linearly; SGLang's
  flashinfer decode kernel is much better at hiding that latency.

## Crossover map

```
                          this branch beats sglang
                         ┌──────────────────────────────┐
                         │           prefill            │  decode
  3090 (sm_86)           │  S ≤ 256                     │  S ≤ 2,048
  B200 (sm_100)          │  S ≤ 1,024                   │  S ≤ 1,024
                         └──────────────────────────────┘
```

## What did NOT change

- **sm_86 (RTX 3090)**: All `is_blackwell` gates in `setup.py`
  evaluate false on sm_86 — same sources, same defines, same arch
  flag as upstream. Verified on a vast.ai 3090 (see table above):
  identical decode within 1 % noise; prefill +23 % to +92 % across
  shapes from arch-agnostic recurrence/sync improvements that
  happen to also help sm_86.
- **sm_120 / sm_121a (RTX 50 / DGX Spark / GB10)**: same sources,
  same defines as before; the gb10 NVFP4 path is untouched.
- **NVFP4 path on B200**: the LUT-based NVFP4 decode runs (29,994
  pp520 / 327 tg128) but is slower than the new BF16 path because
  it intentionally avoids tensor cores. Recommended use of B200 is
  `--backend bf16`. NVFP4 decode tuning for B200 is deferred.

## Methodology

- `bench_shapes.py` is in this repo and runs in a single Python
  process: weights load once, KV cache and scratch sized for the
  largest S in the sweep, per-S buffers re-allocated each step.
- SGLang sweep uses `sglang.bench_one_batch` per shape (model
  reloaded per call) — slower wall-clock but the steady-state
  numbers are equivalent.
- All decode timings: 8 warmup steps then 64 timed steps; report
  median per-step throughput.
- All numbers are batch 1, single stream, no other workload pinned
  to the same GPU.
- Torch versions used: 2.11+cu128 (B200) and 2.5.1+cu124 (3090) for
  the megakernel runs; 2.9.1 (both boxes) after the SGLang install
  upgraded torch.

## Where the long-S prefill gap actually lives

Profiling the prefill at S = 8 k on B200, our path spends ≈ 60 %
inside `pf_deltanet_recurrence_vsplit_prepped` (still serial-over-t,
and we now have one block-pair per (head, v_split) but 24 layers ×
~16 blocks = 384 blocks per layer-step, not enough at that S to
saturate 148 SMs each microsecond), ≈ 30 % inside the cuBLAS GEMM
calls (working as advertised), and the rest in DN-prep and
`pf_qk_norm_rope`. Two structural levers stand out for closing the
gap to SGLang:

1. **Chunked associative-scan DeltaNet recurrence.** The current
   recurrence is exactly serial: 1 step depends on step t-1. The
   delta-net update `S_t = decay * S_{t-1} + k_t (v_t - decay * S_{t-1} k_t).T β`
   admits a chunked formulation where C consecutive steps run in
   parallel inside a chunk and only the inter-chunk state propagates
   sequentially (T/C chunks). This is exactly the Mamba-style
   parallel scan. Even at C = 64 we'd cut the serial frontier 64×.
   Estimated reach: 4–8× prefill at S ≥ 4 k.
2. **flashinfer-style attention on the FA layers.** Our FA prefill
   today is cuBLAS Q@K, materialized softmax, then attn@V — three
   GEMMs and a kernel between them. flashinfer fuses all of that
   into one tensor-core kernel with online softmax and never
   materializes the score matrix. For 6 FA layers at S = 8 k the
   savings dominate the cuBLAS time; at S = 2 k they're already
   visible. Estimated reach: 2–3× on the FA prefill subtotal.

Decode parity at long S needs flashinfer-decode-style behavior in
the same kernel: chunked KV scan with online softmax instead of
the per-step loop we have today. The persistent-megakernel scaffold
stays — only the FA inner-loop kernel changes.
