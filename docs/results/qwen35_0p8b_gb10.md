# Qwen3.5-0.8B on NVIDIA GB10 (DGX Spark) — gb10-train branch

First port of the 3090-train megakernel to GB10 (Grace Blackwell, sm_121a,
compute cap 12.1, aarch64). Build is the unmodified 3090-train tree with
`MEGAKERNEL_CUDA_ARCH=sm_121a`. Backend is selected by
`torch.cuda.get_device_capability()` — cap 12 → NVFP4 decode path
(`kernel_gb10_nvfp4.cu`), otherwise BF16.

## Environment

| Item | Value |
|------|-------|
| GPU | NVIDIA GB10 (sm_121a, cap 12.1, integrated Grace Blackwell) |
| Driver | 580.126.09 (`nvidia-driver-580-open`) |
| CUDA | 13.2.78 (`/usr/local/cuda`) |
| PyTorch | 2.11.0+cu130 (arch_list: sm_80/90/100/110/120) |
| transformers | from `.venv` (HF eager baseline) |
| Build flags | `-arch=sm_121a -O3 --use_fast_math -DBLOCK_SIZE=512 -DLM_BLOCK_SIZE=256` |
| Model | `Qwen/Qwen3.5-0.8B` (HF cache: `/home/sparkz/rl/.hf_cache`) |
| Branch | `gb10-train` (off `3090-train`) |

Power: GB10 reports `power.draw=14W` at idle; the integrated SoC exposes
no `power.limit` rail, so the 3090-style 220 W DVFS sweet-spot table does
not apply.

## Rollout shape sweep (prefill + 32-gen)

Mirrors the canonical 3090-doc shape table. Megakernel path is BF16
(`Decoder.prefill` is BF16-only; NVFP4 prefill is not implemented).
Numbers are best-of-2 runs per shape, one warm pass.

| Prompt | HF (ms) | Ours (ms) | vs HF | Ours pp (tok/s) | Ours tg32 (tok/s) |
|-------:|--------:|----------:|------:|----------------:|------------------:|
|    128 |   529.5 |     233.7 | 2.27× |          7,447 |             143.2 |
|    512 |   543.2 |     258.2 | 2.10× |         14,413 |             139.9 |
|   2048 |   727.4 |     348.7 | 2.09× |         17,464 |             134.0 |
|   8192 | 1,723.2 |     736.8 | 2.34× |         17,115 |             120.1 |
|  16384 | 3,094.4 |   1,297.6 | 2.38× |         16,349 |             104.9 |
|  32768 | 6,255.8 |   2,526.0 | 2.48× |         15,236 |              82.6 |

HF baseline: `AutoModel(input_ids)` for prefill, then `past_key_values`
step loop for 32-gen, `bf16, TF32 on, cuDNN benchmark on`.

For 3090 reference at the same shapes, see `qwen35_0p8b_3090.md`. The
GB10 vs HF gap (2.1–2.5×) is roughly half of the 3090's gap (3–12×) for
the reasons listed below — the megakernel is **slower on GB10 than on
3090** in absolute terms because all of the tile/L2/V-split heuristics
are sm_86-shaped.

Run:

```bash
source /home/sparkz/rl/.venv/bin/activate
export HF_HOME=/home/sparkz/rl/.hf_cache
PYTHONPATH=models/qwen35_0p8b python experiments/bench_gb10_rollout.py \
    --json /tmp/gb10_rollout.json
```

## final_bench.py — pp520 / tg128

| Path | Backend | Prefill mode | pp520 (tok/s) | tg128 (tok/s) |
|------|---------|--------------|--------------:|--------------:|
| **Megakernel** | NVFP4 decode | eager (`prefill_bf16`) | **14,589** | **152** |
| Megakernel | BF16 | eager (`prefill_bf16`) | 14,695 | 141 |
| Megakernel | NVFP4 decode | mega (`prefill_bf16_mega`) | 5,614 | 152 |
| PyTorch HF | BF16 (HF eager) | — | 5,834 | 67 |

Speedup over PyTorch HF eager on the same hardware: **2.5× prefill,
2.3× decode** (NVFP4 path). NVFP4 decode is +8 % over BF16 decode; BF16
slightly edges NVFP4 on prefill (the eager prefill is BF16 in both
cases — the decode kernel is what changes).

### Cross-arch reference (from this branch's other targets)

| Target | Backend | pp520 (tok/s) | tg128 (tok/s) |
|--------|---------|--------------:|--------------:|
| **GB10 (sm_121a)** | NVFP4 | **14,589** | **152** |
| RTX 3090 (sm_86, @220W) | BF16 | 37,800 | 413 |
| RTX 3090 (sm_86, stock) | BF16 | — | 433 |
| B200 (sm_100) | BF16 | 40,278 | 711 |

GB10 lands well below both 3090 and B200 because:

- The persistent megakernel prefill (`prefill_bf16_mega`) is tuned for
  sm_86 WMMA/L2 layout and regresses to **5,614 tok/s** on GB10 — slower
  than the eager `prefill_bf16` path here. `kernel.cu` already swaps to
  `cg::this_grid().sync()` for Blackwell, but prefill block-tile / split-K
  shapes still target Ampere.
- The DeltaNet chunked forward (`dn_chunked_3090.cu`, `V_SPLITS=4, C=32`)
  is hand-tuned for 3090 SM count and L1 capacity.
- NVFP4 decode (`kernel_gb10_nvfp4.cu`) runs but uses bf16 dot-products
  inside `dot8_nvfp4_bf16`/`_f32` rather than the sm_120 tensor-core MMA
  for FP4; tcgen05/`mma.kind::mxf4` paths are not wired up yet.

## Reproduce

```bash
cd /home/sparkz/rl/lucebox-hub
git checkout gb10-train
source /home/sparkz/rl/.venv/bin/activate
export HF_HOME=/home/sparkz/rl/.hf_cache

cd models/qwen35_0p8b
MEGAKERNEL_CUDA_ARCH=sm_121a MAX_JOBS=8 python setup.py build_ext --inplace

python final_bench.py                    # auto → NVFP4 decode
python final_bench.py --backend bf16     # BF16 decode
python final_bench.py --prefill-mode mega --skip-hf   # persistent prefill
```

## Open work (sm_121a tuning)

1. Retune `prefill_bf16_mega` block/tile for Blackwell — current values
   regress >2.5× vs eager prefill on GB10.
2. Replace `dot8_nvfp4_bf16` software path with `mma.kind::mxf4` /
   `tcgen05.mma` on sm_120/121.
3. Investigate DeltaNet chunked forward retune (`V_SPLITS`, `C`) for the
   GB10 SM count and shared-memory budget.
4. Confirm cuDNN FA-2 wrapper picks the Blackwell-optimal heuristic
   (`fa_attn_aten.cpp` calls `at::scaled_dot_product_attention`).
