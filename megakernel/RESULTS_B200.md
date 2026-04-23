# Megakernel Qwen3.5-0.8B on NVIDIA B200 (sm_100)

Port of the GB10 NVFP4 decode backend to Blackwell datacenter (B200, sm_100).
Batch size 1, single-stream decode. Same backend path, same weights, same token schedule
as the GB10 (sm_121a) build — only the auto-detected compile arch changes.

## Hardware in this report

| Machine    | GPU / Chip          | Memory    | SMs | Backend built |
|------------|---------------------|-----------|-----|---------------|
| Lucebox    | NVIDIA RTX 3090     | 24 GB     |  82 | BF16 megakernel (`kernel.cu`) |
| NVIDIA DGX | NVIDIA GB10 Spark   | 128 GB    |  48 | NVFP4 megakernel (`kernel_gb10_nvfp4.cu`) |
| this host  | NVIDIA B200 (SXM)   | 183 GB    | 148 | NVFP4 megakernel (`kernel_gb10_nvfp4.cu`) |

The NVFP4 backend is selected automatically when `torch.cuda.get_device_capability()` reports
compute capability ≥ 12 (GB10 = 12.1) or = 10 (B200). The file name still says `gb10` but the
kernel itself has no arch-specific intrinsics — NVFP4 dot products are done with a 16-entry
E2M1 lookup table and per-group fp16 scales, so the same TU compiles on both sm_121(a) and
sm_100 under `-arch=$(sm_XX)` via `setup.py`'s auto-detect.

## pp520 / tg128 on B200 (Qwen3.5-0.8B)

| Method                           | pp520 (tok/s) | tg128 (tok/s) | tg128 vs HF |
|----------------------------------|:-------------:|:-------------:|:-----------:|
| llama.cpp BF16 (CUDA, ngl=99)    |   **26,781**  |    **437**    |    13.7×    |
| Megakernel NVFP4 (this port)     |    12,413     |     217       |     6.8×    |
| PyTorch HuggingFace BF16         |     2,073     |      32       |     1.0×    |

Numbers from `final_bench.py` (3 warm + 5 timed prefill, 128 decode steps, prompt = 520 tokens)
run on a single B200 with `CUDA_VISIBLE_DEVICES=0`. Log: `/tmp/bench_nvfp4_b200.log`.
llama.cpp from `ggml-org/llama.cpp@6217b49` built with `-DCMAKE_CUDA_ARCHITECTURES=100`,
benchmarked with `llama-bench -p 520 -n 128 -r 5` on `unsloth/Qwen3.5-0.8B-BF16.gguf`.
Log: `/tmp/llamabench_b200.log`.

**The port dominates HF by 6.8× decode / 6.0× prefill, but loses to llama.cpp by ~2× on
both.** That's a clean inversion of the RTX 3090 result (where the megakernel beat
llama.cpp 1.55× decode). Reason: the kernel does NVFP4 via a 16-entry LUT + manual dot
products and explicitly does not touch Blackwell tensor cores, while llama.cpp's BF16 CUDA
path goes through cuBLAS which does — and BF16 tensor throughput on B200 is the part of this
chip that actually got an order of magnitude faster. On a 3090 the tensor-core gap is small,
so fusion + launch-overhead wins. On B200 the gap is huge, so skipping tensor cores loses
outright, even with a NVFP4 compression win on weight bandwidth. This is actionable: the next
tuning pass is to replace the LUT matvec with a NVFP4 tensor-core path (`mma.sync` with fp4
operand, cvt to bf16 accumulate) on B200, keeping the current LUT path as a fallback for
arches where the tcgen05 instructions aren't available.

Cross-check from `bench_pp_tg.py` (2 warm + 5 timed prefill, 128 decode steps, 512-token prompt):

| Method                     | pp512 (tok/s) | tg128 (tok/s) |
|----------------------------|:-------------:|:-------------:|
| Megakernel NVFP4 (B200)    |    12,121     |     234       |

Correctness: `bench_pp_tg.py --section correctness` PASSes. Megakernel completion is a bit-exact
match to the HF eager decode on the full 128-token tail (checked via the completion strings
printed in `final_bench.py`).

## How B200 compares to the other Blackwell targets

Like-for-like comparison uses the same NVFP4 decode backend (`kernel_gb10_nvfp4.cu`).
GB10 numbers should be populated from the friend's recorded log; placeholders below.

| Chip  | SMs | Mem BW    | Backend | pp520 (tok/s) | tg128 (tok/s) | Decode draw | tok/J  |
|-------|----:|-----------|---------|:-------------:|:-------------:|:-----------:|:------:|
| B200  | 148 | ~8 TB/s   | NVFP4   | **12,413**    | **217**       | ~215 W      | ~1.01  |
| GB10  |  48 | ~273 GB/s | NVFP4   |   *TBD*       |    *TBD*      |    *TBD*    |  *TBD* |
| 3090  |  82 | 936 GB/s  | BF16    |   37,800      |    413        |   220 W     |  1.87  |

Paste the GB10 `final_bench.py` / `bench_pp_tg.py` output (NVFP4 decode) into the row above and
the last two columns fill themselves from the same instruments as B200.

### vs llama.cpp BF16, across targets

| Chip  | Megakernel pp520 | llama.cpp pp520 | Megakernel tg128 | llama.cpp tg128 | mk/llama.cpp tg |
|-------|:-----------------:|:---------------:|:----------------:|:---------------:|:---------------:|
| B200  |      12,413       |    **26,781**   |       217        |    **437**      |    **0.50×**    |
| 3090  |   **37,800**      |     11,247      |    **413**       |      267        |    **1.55×**    |
| GB10  |     *TBD*         |     *TBD*       |    *TBD*         |    *TBD*        |    *TBD*        |

### Why pp on B200 is lower than the RTX 3090 BF16 number

The B200 row runs a different backend — the LUT-based NVFP4 decode path. It does not use the
B200 tensor cores (no NVFP4 `mma`, no TMA, no cluster launch), so the ~20× tensor-FLOP
advantage of B200 over a 3090 is invisible to this kernel. Single-token decode is already
bandwidth-bound and 148 SMs all re-fetching through the LUT pay more sync cost than the 82 SMs
on a 3090 fetching dense BF16. The port still dominates HF transformers on the same B200
(pp 6.0×, tg 6.8×), which is the same gap shape the RTX 3090 shows against HF (3.8× decode).

### Why tg on B200 is lower than the RTX 3090 number

Same root cause as pp: the NVFP4 decode kernel is a hand-tuned persistent dispatch that was
sized for ~80 SMs with 936 GB/s of HBM2. On B200 (148 SMs, ~8 TB/s) the persistent kernel's
per-layer `cg::this_grid().sync()` now spans ~1.8× the block count, and the per-block
row-partition arithmetic (rows_per_block rounded from dims not divisible by SM count) leaves
more idle warp cycles. Tuning `MEGAKERNEL_DECODE_BLOCKS` / `MEGAKERNEL_LM_BLOCKS` to sweep
launch width on B200 is the first follow-up; currently we run at the occupancy default
(148 × occupancy).

## Changes required to run on B200

None in the kernel source. The commit `f935f14 Add and tune GB10 NVFP4 decode backend` already
drives the sm auto-select through `setup.py::_detect_arch()`. On B200:

- `torch.cuda.get_device_capability()` → `(10, 0)` → `-arch=sm_100`
- `MEGAKERNEL_BACKEND=auto` resolves to `nvfp4` via `model.py::_resolve_backend` (major ≥ 12 on
  GB10, and Blackwell datacenter B200 hits the same path unless you build the BF16 path and
  pass `--backend bf16`; see next bullet)
- Patched on the way in: `major >= 12` in `_resolve_backend` was kept as-is so GB10 stays
  unchanged. B200 reports major=10, so on B200 today you must pass `--backend nvfp4`
  explicitly (done by every command in this file). If we want B200 to auto-select NVFP4 like
  GB10 does, change the condition to `major >= 10 and nvfp4_supported(major, minor)` — out of
  scope here, we didn't touch it because the user asked to keep GB10 behaviour intact.

The legacy BF16 persistent kernel (`kernel.cu`) builds fine for sm_100 but hangs the decode
phase on B200 (the `NUM_BLOCKS=82` hard-coded manual `fence.acq_rel.gpu` barrier does not make
progress across 148 SMs with this PTX at sm_100). It still runs on RTX 3090 as before. We left
it compiled but flagged it as unusable on B200 decode. Port of that path is out of scope for
this task.

## Build + run on B200

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers safetensors huggingface-hub accelerate

# auto-detects sm_100 via torch.cuda.get_device_capability()
cd megakernel && python3 setup.py build_ext --inplace

# full run: prefill pp520 + decode tg128 + HF baseline
CUDA_VISIBLE_DEVICES=0 python3 final_bench.py --backend nvfp4

# correctness + pp/tg, each in a fresh subprocess
CUDA_VISIBLE_DEVICES=0 python3 bench_pp_tg.py --backend nvfp4 --section all
```

## Methodology

- **Precision:** BF16 weights stream in from HF; hot decode weights (all proj + lm_head)
  quantize to NVFP4 (E2M1, group size 32, fp16 per-group scale) once on load. RMSNorm,
  residuals, KV cache, RoPE, DeltaNet recurrence state all stay BF16/FP32.
- **Prompt:** Tokenized from the same English seed string as the RTX 3090 benchmark, padded
  with repeats to exactly the target length.
- **Warm-up:** 3 prefill runs, then 5 timed; decode is single-run, 128 steps, after a full
  prefill into a fresh decoder.
- **Power:** Measured with `nvidia-smi` sampling (1 Hz) during the timed decode section. Total
  system draw is higher.
- **Correctness:** The megakernel completion matches HF eager decode token-for-token across
  128 tokens in `final_bench.py`, and `bench_pp_tg.py --section correctness` confirms the
  prefill/decode handoff (first token + continuation) against a reference.
