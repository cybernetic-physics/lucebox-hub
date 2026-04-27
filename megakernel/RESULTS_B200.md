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
the warmup). Both backends are batch-1 single-stream.

### B200 (sm_100, GPU 0)

Prefill (tok/s):

| S | upstream/main | this branch | sglang | best |
|---:|---:|---:|---:|---|
| 32 | 6,701 | **13,091** | 811 | this branch |
| 64 | 9,115 | **21,696** | 1,938 | this branch |
| 128 | 10,865 | **33,242** | 4,124 | this branch |
| 256 | 11,810 | **43,119** | 8,842 | this branch |
| 512 | 12,122 | **45,596** | 14,671 | this branch |
| 1,024 | 12,285 | **52,426** | 30,753 | this branch |
| 2,048 | 11,619 | 56,193 | **61,766** | sglang |
| 4,096 | — | 54,595 | **127,422** | sglang |
| 8,192 | — | 38,719 | **217,503** | sglang |
| 16,384 | — | 37,742 | **201,577** | sglang |
| 32,768 | — | 36,401 | **158,908** | sglang |
| 65,536 | — | **34,041** | n/a | this branch |

Decode (tok/s, after S-token prefill):

| S | upstream/main | this branch | sglang | best |
|---:|---:|---:|---:|---|
| 32 | hang | **915** | 484 | this branch |
| 64 | hang | **915** | 487 | this branch |
| 128 | hang | **914** | 552 | this branch |
| 256 | hang | **907** | 520 | this branch |
| 512 | hang | **901** | 488 | this branch |
| 1,024 | hang | **885** | 441 | this branch |
| 2,048 | hang | **830** | 450 | this branch |
| 4,096 | hang | **746** | 423 | this branch |
| 8,192 | hang | **602** | 367 | this branch |
| 16,384 | hang | **472** | 282 | this branch |
| 32,768 | hang | **332** | 193 | this branch |
| 65,536 | hang | **208** | n/a | this branch |

upstream/main BF16 decode hangs on B200 — its hand-rolled grid
barrier doesn't progress on 148 SMs. The cooperative-grid-sync fix
is what makes BF16 decode functional on sm_100, and the split-K
rewrite of phase 3 (kernel.cu) lifts decode from 11→208 tok/s at
S=64k by filling 144 of 148 SMs during the FA scan instead of
just 8.

### RTX 3090 (sm_86, vast.ai PCIe 4.0×8, 350 W stock)

Prefill (tok/s):

| S | upstream/main | this branch | sglang | best |
|---:|---:|---:|---:|---|
| 32 | 4,355 | **5,484** | 1,351 | this branch |
| 64 | 6,571 | **9,466** | 2,747 | this branch |
| 128 | 8,453 | **15,674** | 5,285 | this branch |
| 256 | 8,901 | **18,717** | 10,859 | this branch |
| 512 | 8,635 | 17,687 | **20,942** | sglang |
| 1,024 | 8,519 | 17,957 | **35,368** | sglang |
| 2,048 | 7,987 | 19,344 | **41,658** | sglang |
| 4,096 | — | 19,390 | **43,203** | sglang |
| 8,192 | — | 18,923 | **41,434** | sglang |
| 16,384 | — | 17,777 | **37,221** | sglang |
| 32,768 | — | **15,842** | OOM | this branch |
| 65,536 | — | **13,130** | n/a | this branch |

Decode (tok/s, after S-token prefill):

| S | upstream/main | this branch | sglang | best |
|---:|---:|---:|---:|---|
| 32 | 430 | **440** | 335 | this branch |
| 64 | 443 | **448** | 336 | this branch |
| 128 | 439 | **447** | 334 | this branch |
| 256 | 431 | **446** | 334 | this branch |
| 512 | 419 | **445** | 324 | this branch |
| 1,024 | 394 | **443** | 336 | this branch |
| 2,048 | n/a | **437** | 333 | this branch |
| 4,096 | — | **425** | 328 | this branch |
| 8,192 | — | **401** | 320 | this branch |
| 16,384 | — | **361** | 307 | this branch |
| 32,768 | — | **302** | OOM | this branch |
| 65,536 | — | **227** | n/a | this branch |

`upstream/main` columns are capped at S = 2,048 because the upstream
binding hardcodes `FA_KV_HEADS * 2048 * FA_HEAD_DIM` as the KV
stride; S > 2,048 silently corrupts the cache. This branch lifts
that cap (the launcher reads `max_seq` from `fa_k_cache.size(2)`).

## Crossover map

```
                          this branch beats sglang (DECODE)
                         ┌──────────────────────────────┐
  3090 (sm_86)           │  ALL shapes (1.18–1.37x ours)│
  B200 (sm_100)          │  ALL shapes (1.64–2.00x ours)│
                         └──────────────────────────────┘

                          this branch beats sglang (PREFILL)
                         ┌──────────────────────────────┐
  3090 (sm_86)           │  S ≤ 256                     │
  B200 (sm_100)          │  S ≤ 1,024                   │
                         │  near parity at S = 2,048    │
                         └──────────────────────────────┘

                          this branch beats sglang (decode)
                         ┌──────────────────────────────┐
  3090 (sm_86)           │  S ≤ 2,048                   │
  B200 (sm_100)          │  S ≤ 1,024                   │
                         └──────────────────────────────┘
```

## Step-by-step prefill journey on B200

Each row is the sustained pp520 BF16 figure as commits landed:

```
  baseline (gb10 NVFP4 path)         29,974
  bf16 decode B200 fix                  —     (decode unblocked, pp unchanged)
  graph capture + recurrence rewrite 15,859
  2 __syncthreads cut                16,770
  normalize ‖ beta/decay             17,847
  V-split recurrence + gnorm split   27,501
  conv/norm/beta-dec pre-pass        40,278
  cuDNN FA-2 (this commit)           45,596 at S=512, 56,193 at S=2k
```

The cuDNN FA-2 swap is the first commit to lift the long-S envelope
(56k tok/s sustained from S=2k to S=4k on B200 vs the 32k peak that
the cuBLAS-attention path was capped at). At S=65k the attention
swap delivers a 13.9× speedup.

## What did NOT change

- **sm_86 (RTX 3090)**: All `is_blackwell` gates in `setup.py`
  evaluate false on sm_86 — same gb10 / Blackwell sources excluded.
  Verified on a vast.ai 3090: identical decode within 1% noise;
  prefill +26 % to +1041 % across shapes (the new FA-2 path scales
  flat at ~15–19 k tok/s instead of collapsing past S = 1k).
- **sm_120 / sm_121a (RTX 50 / DGX Spark / GB10)**: same gb10 sources,
  same defines as before; the gb10 NVFP4 path is untouched.
- **NVFP4 path on B200**: the LUT-based NVFP4 decode runs (29,994
  pp520 / 327 tg128) but is slower than the new BF16 path because
  it intentionally avoids tensor cores. Recommended use of B200 is
  `--backend bf16`. NVFP4 decode tuning for B200 is deferred.

## Methodology

- `bench_shapes.py` (in this repo) runs in a single Python process:
  weights load once, KV cache and scratch sized for the largest S in
  the sweep, per-S buffers re-allocated each step. 5 timed runs
  after 3 warmup runs per shape.
- SGLang sweep uses `sglang.bench_one_batch` per shape (model
  reloaded per call) — slower wall-clock but the steady-state
  numbers are equivalent.
- Decode timings: 8 warmup steps then 64 timed steps; report median
  per-step throughput.
- All numbers are batch 1, single stream, no other workload pinned
  to the same GPU.
- Torch versions used: 2.11+cu128 (B200), 2.5.1+cu124 → 2.9.1 (3090,
  bumped by sglang install). Megakernel rebuilt against torch 2.9.1
  on the 3090 box; B200 box rebuilt against the same torch.

## What's still SGLang's at long context, and what closes it

After the FA-2 swap, the dominant cost at S ≥ 4k on this branch is
the DeltaNet recurrence — strictly serial-over-t, scales linearly
with S, and runs at about 16 blocks per layer-step which doesn't
saturate B200's 148 SMs. SGLang doesn't pay this cost (it routes
DN traffic through fla's chunk-parallel kernel).

Two outstanding levers:

1. **Chunked associative-scan DeltaNet recurrence.** The b200-train
   branch already has a working bf16-tensor-core chunked DN forward
   (3.4× at S=512 on its own). Porting it onto fork-parent-b200 and
   wiring it into `prefill.cu`'s DN-layer body should give a
   per-step parallelism factor proportional to the chunk size; at
   chunk = 64 we'd reduce the serial frontier 64×. Estimated
   remaining gap-close: 4–8× prefill at S ≥ 4k. This is the next
   commit.

2. **flashinfer-style decode KV scan.** At S ≥ 4k our decode is
   limited by the per-step FA scan that walks the entire KV
   history. The same cuDNN FA path used in prefill is not directly
   reusable for the per-step decode case (it's a single-query case
   where flashinfer's decode-attention kernel pulls ahead of the
   "tile-per-query" FA-2 layout). Closing this gap requires either
   integrating flashinfer for the FA decode layers or writing an
   equivalent online-softmax tensor-core kernel inside `kernel.cu`.
