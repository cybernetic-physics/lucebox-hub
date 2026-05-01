# Open work — Qwen3.5-0.8B trainer/inference on RTX 3090

State as of `9848cb7` on branch `3090-train`. See
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

### #9 — SGLang baseline harness ✅ shipped (`4796320`)

`experiments/bench_3090_sglang.py` runs the canonical shape sweep
against `sglang.Engine(Qwen/Qwen3.5-0.8B, tp_size=1,
mem_fraction_static=0.7, disable_cuda_graph=True,
disable_radix_cache=True)`. Results landed in
`docs/results/qwen35_0p8b_3090.md` as the SGLang column. We're faster
than SGLang up to S=16K (1.14×) and SGLang wins by ~30% at S=32K —
its `fla.chunk_gated_delta_rule` prefill beats our V-split recurrence
at the longest shapes.

### #10 — tuned HF + PyTorch baseline ✅ shipped (`9848cb7`)

`experiments/bench_3090_hf_tuned.py` adds the "HF tuned" rollout column
and a training-step harness. Tuning knobs: bf16, TF32 matmul on, cuDNN
benchmark on, fused AdamW, PEFT LoRA, torchao import-check stub.
`torch.compile(mode="reduce-overhead")` fails Dynamo on the hybrid
Qwen3.5 generate path with `InternalTorchDynamoError: accessing tensor
output` — documented as a known PyTorch limitation, no fix needed.

### #20 — fused mega-bwd, in progress (`f84d034`, `029f1e0`)

Two phases shipped this round:

  - bf16 GEMMs in `layer_mlp_bwd` gate/up recompute (`f84d034`).
    Eliminates the 18 ms / 8% ampere_sgemm_128x64_tn fp32 path at
    P=1024.
  - cuBLAS bf16 GEMMs in `lora_linear_bwd` at S≥512 (`029f1e0`).
    Replaces the 5 SIMT kernels (~33 blocks each on 82 SMs) with
    cutlass_80_tensorop_bf16 calls. Shape-routed: SIMT for S<512
    (1 binding launch wins on overhead), cuBLAS for S≥512.

Bench progress (vs HF+PEFT at P × T=32):

   P=64    1.02× → 1.04×  (parity, both paths small-S SIMT)
   P=256   1.00× → 1.01×  (parity, both paths SIMT)
   P=1024  0.58× → 0.72×  (244 ms → 194 ms = 24% step-time win)

Remaining gap to flip ≥1.0× at P=1024 needs another ~30%. The big
lever is **CUDA graph capture** of `forward_backward` to amortize
the ~7 600 kernel launches per step into a single graph replay.
Estimated 15-20% additional win — would put us at 0.85-0.90×, still
short of flipping. A full flip likely requires reducing the long
tail of small per-op launches (291× / 420× / 505× / 858× elementwise
+ direct_copy at 8-17 µs/launch) by composing them into per-layer
mega-kernels.

Tracked in task #20.

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
