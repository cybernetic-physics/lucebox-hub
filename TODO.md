# Open work — Qwen3.5-0.8B trainer/inference on RTX 3090

State as of `83cccf7` on branch `3090-train`. See
`docs/results/qwen35_0p8b_3090.md` for the perf write-up.

## Final RTX 3090 numbers vs HF generate

Rollout = prefill + 32 generated tokens, ms wall, best of 3 runs:

| P     |  HF  | Ours | Speedup |
|------:|-----:|-----:|--------:|
|   128 |  971 |   80 |   12.1× |
|   512 | 1025 |   92 |   11.1× |
|  2048 | 1126 |  150 |    7.5× |
|  8192 | 1652 |  384 |    4.3× |
| 16384 | 2580 |  813 |    3.2× |
| 32768 | 4798 | 1766 |    2.7× |

## Done this round

| # | Task | Where |
|--:|---|---|
| 13 | Split-K FA decode port from fork-parent-b200~2 | `424d865` |
| 14 | 3090-tuned chunked DN forward (V_SPLITS=4 + C=32) | `dba22e3` |
| 15 | Unified LoRA + frozen base weights (~3 GB freed/instance) | `b16c46e`, `83cccf7` |
| 16 | Training-loop test harness (`grad_harness.py`) | `868081c`, `4d872e3`, `34b82c2` |
|    | Forward graph cache bug + cuDNN FA bwd determinism | `346f4a3`, `690d8c1` |
|3-5 | Arch-aware build + runtime guards on sm_100-only kernels | `354be69` |

## Open

### #17 — gradient corruption in `per_layer_bwd_fa` ✅ shipped (`ebb57dd`)

Root cause: missing `__syncthreads()` between two block-wide reductions
in `bwd_rmsnorm_kernel`. The first reduction wrote `s_red[0] = mean_sq`,
then every thread read it; the second reduction wrote `s_red[warp_id]
= dot` *without a fence first*. A fast warp 0 could overwrite `s_red[0]`
before slow warps finished their mean_sq read → slow warps divided by
H using a stale dot partial → garbage gradient.

The race only triggered at the QKnorm bwd shape (S=240, H=256 for
30 tokens × 8 q-heads × head_dim=256); the K-norm and DN-norm shapes
happened to keep warps in lockstep so the bug stayed hidden across
all the original unit tests.

Compute-sanitizer racecheck found it in seconds once we ran it on the
actual trainer shape. Fix is a one-line `__syncthreads()`.

Verification:
  - racecheck: 0 hazards across the entire kernel-bwd training path
  - stability harness: 20/20 good runs (was 1/20)
  - multi-step training: step-3 rel error 88.6% → 0.29% vs HF+PEFT
  - rollout perf unchanged (S=32K prefill 1197 ms, gen 101 ms)

### #3 / #4 / #5 — sm_86 retarget polish ✅ shipped (`354be69`)

  - `cutlass_train/setup.py` now skips cleanly on SM<100 instead of
    emitting an invalid sm_86a binary.
  - `prefill_bf16_mega`, `dn_bwd`, `dn_chunked_fwd` all query the
    device's per-block opt-in shared-mem cap and refuse to launch
    when the kernel's smem need exceeds it. Surfaces a clear error
    instead of cudaErrorInvalidValue from the hidden FuncSetAttribute.
  - `trainer/setup.py` picks `PM_NUM_BLOCKS` per device cc:
    `(10,0)→148, (12,0)/(9,0)→132, (8,6)→82, (8,9)→76, (8,0)→108`.
    Override via `TRAIN_MEGA_NUM_BLOCKS`.
  - `prefill.cu` cuBLAS workspace scaled to compute capability:
    32 MB on Hopper+, 4 MB on Ampere/Ada.

  Production hot path on RTX 3090 unchanged — uses `prefill_bf16` +
  `dn_chunked_3090` + cuBLAS. The guards are belt-and-braces for
  misconfigured calls.

### #9 — SGLang baseline harness

Stand up the latest SGLang serving Qwen3.5-0.8B on GPU 1, measure
prefill + decode + RL-rollout throughput at S in {128, 512, 2K, 8K, 32K}.
Adds the third reference column to `docs/results/qwen35_0p8b_3090.md`
(today we only compare against HF generate).

  **Acceptance**: a `experiments/bench_3090_sglang.py` that produces
  a table the same shape as `bench_3090_rollout.py`, plus a corresponding
  row in the results doc.

### #10 — tuned HF + PyTorch training baseline

Today the kernel-bwd path uses HF+PEFT autograd as its correctness
reference, but there is no fully-tuned baseline harness for marketing
comparisons. Stand up: HF transformers + PEFT LoRA + flash-linear-attention
+ cuDNN SDPA + `torch.compile` + fused AdamW. Measure RL step (sample +
train) at the same shapes.

  **Acceptance**: a `experiments/bench_3090_hf_train.py` reporting
  ms/step at the canonical shape sweep, with a `--with-compile` flag
  for the torch.compile variant.

## Known infrastructure gotchas

  - The kernel forward's CUDA graph cache (`prefill.cu`) keys on
    pointer values. PyTorch's caching allocator reuses freed addresses,
    so without the `is_training_call` guard a captured graph reads
    stale buffers on replay. Don't remove that guard.
  - Decoder weight tensors are **views** into HF's `state_dict()` after
    `_unify_weights_from_hf`. Keep the HF model alive on the weights
    dict via `weights["_hf_model_keepalive"]` — Python GC of the HF
    model would invalidate the views.
  - `dn_chunked_3090.cu` writes state in `[Dv, Dk]` external layout to
    match the decode/recurrence kernels; internally it's `[Dk, Dv_block]`
    for the matmul wiring. Don't change either side without updating
    both.
  - `wmma_gemm_fp32`'s col_major a_ptr / b_ptr formula was buggy in the
    trainer kernel (M / K swapped). The trainer copy works on B200
    only because the OOB read lands in adjacent shared mem; on 3090
    it crashes. Fixed in `dn_chunked_3090.cu`; the trainer's
    `dn_chunked.cu` still has the original buggy form.

## Where to look first

  - Stability gate: `experiments/grad_harness.py stability --iters 20`
  - Per-op gradient checks: `experiments/test_bwd_kernel_determinism.py`,
    `experiments/test_dn_attn_bwd_determinism.py`,
    `experiments/test_fa_bwd_math.py`
  - Rollout perf: `experiments/bench_3090_rollout.py`,
    `experiments/bench_dn_chunked_3090.py`
  - Results / methodology: `docs/results/qwen35_0p8b_3090.md`
