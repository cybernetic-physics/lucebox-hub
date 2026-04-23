# Unified LoRA Training + Inference Engine

Drop-in replacement for `LoRATrainer` in `~/rl/backend`. One CUDA
codebase does LoRA **training** (forward + backward + AdamW) and
**inference** (sampling, log-probs) using the same megakernel
architecture we already have for prefill + decode.

## Why now

Five things converged:

1. **The raw-PTX-only constraint is gone.** `tcgen05_gemm/` on the
   `b200-train` worktree spent weeks and ~20 commits getting a
   hand-rolled Blackwell MMA GEMM to 262 TFLOPs (35 % of B200 peak,
   still below cuBLAS on every shape). cuBLAS + CUTLASS produce that
   in one line. Using them unblocks everything else.
2. **Prefill megakernel is proven.** On B200 we're at 40,278 tok/s
   pp520, 1.50× llama.cpp — that's the target-forward side of
   training already done.
3. **LoRA forward is already wired.** `train_megakernel/kernel.cu`
   applies LoRA residuals on all 13 trainable linears of Qwen3.5-0.8B
   inside the persistent cooperative kernel. Pass `A=B=0` → identical
   output to prefill_megakernel (bit-exact).
4. **Fused AdamW is already wired.** One-kernel AdamW over the flat
   LoRA parameter buffer, correct to bf16 rounding vs torch.optim.AdamW.
5. **The rl/backend seam is clean.** `Runtime`, `Sampler`, `Trainer`,
   `CheckpointStore` protocols in
   `tinker_backend/runtimes/base.py` are narrow enough to drop in a
   new GPU-native implementation without touching the rest of the
   control plane. There's already a scaffolding adapter
   (`lora_megakernel/rl_backend_adapter.py`) for a toy model — we
   just need the real one.

## What exists, where

| Dir (worktree)                      | What it has                                      | Status   |
|-------------------------------------|--------------------------------------------------|----------|
| `megakernel/` (= `models/qwen35_0p8b/`) | BF16 prefill + decode for Qwen3.5-0.8B       | shipped on `b200` |
| `prefill_megakernel/` (b200-train)  | True single-dispatch persistent prefill kernel   | shipped on `b200-train` |
| `train_megakernel/` (b200-train)    | LoRA forward on all 13 linears + fused AdamW     | forward correct, backward partial |
| `lora_megakernel/` (b200-train)     | Toy LoRA-only kernel + `rl_backend_adapter.py`   | working end-to-end against a TinyLoRAModel |
| `tcgen05_gemm/` (b200-train)        | Hand-written Blackwell-native MMA GEMM           | correct but slower than cuBLAS; **retire** |
| `~/rl/backend/`                     | Runtime protocols + `LoRATrainer` (torch CPU)    | control plane in place |

**What's missing**: the backward megakernel (CE + LM head bwd exists;
per-layer reverse doesn't), the cuBLAS/CUTLASS matmul wiring (all
matmuls in `train_megakernel` today are either inside the prefill
megakernel using hand-rolled WMMA tiles, or they're the hand-rolled
`tcgen05_gemm` we're retiring), and the rl/backend integration for a
real (not toy) LoRA session.

## Target architecture

```
  rl/backend (unchanged control plane)
    │
    │  Runtime protocol: sample / forward / forward_backward / optim_step / save / load
    ▼
  models/qwen35_0p8b/trainer/                       ← NEW, lives in lucebox-hub
    ├── kernel.cu            (prefill+backward+AdamW megakernel)
    ├── cublas_gemm.cu       (LoRA A/B matmuls via cuBLAS + CUTLASS wrappers)
    ├── torch_bindings.cpp   (5 ops: sample, forward, fwd_bwd, optim_step, adapter_io)
    ├── trainer.py           (Python: Sampler + Trainer + CheckpointStore impl)
    └── tests/               (gradient check vs HF + PEFT reference)
```

The `trainer.py` implements the three protocols directly. A small shim
in `rl/backend` registers it as one of the available runtimes; no rl
code change beyond that registration.

## Performance target

**3× faster than SGLang-sample + torch-LoRA-train on B200.**

Breakdown of what we're beating:

- SGLang side (inference / sampling): already fast; we won't try to
  beat it on pure sampling throughput. We need to **match** it and
  add training.
- Torch-LoRA-train side (forward + backward + AdamW on GPU): today
  roughly ~500 tokens/s training throughput on B200 for
  Qwen3.5-0.8B with rank-16 LoRA, limited by launch overhead
  (~30 kernels per training step × 24 layers) and autograd graph
  build cost. Our megakernel approach targets **~1,500 tokens/s
  training throughput** on the same setup.
- The x3 comes from: (a) single-dispatch forward (already shipped in
  prefill_megakernel, ~28 % of the win), (b) single-dispatch
  backward fused with AdamW (~25 %), (c) LoRA A/B matmuls pushed
  into the same kernel instead of separate cuBLAS launches (~15 %),
  (d) zero-copy sample path reusing the same weights + LoRA state
  as training (~25 %), (e) fused CE+LM-head+norm-bwd already lands
  in `train_megakernel` latest commit (~7 %).

Measurement plan: `python -m pytest tests/bench_training_throughput.py`
runs one step against the torch-reference trainer (`LoRATrainer` from
rl/backend today), prints tok/s for forward-only, fwd+bwd, and
fwd+bwd+optim. Gate the 3× claim on the last number.

## Implementation plan

### Phase 1 — move & consolidate (2 days)

- Merge the `train_megakernel` + `prefill_megakernel` + `lora_megakernel`
  work from `b200-train` into `models/qwen35_0p8b/trainer/` on the
  `b200` branch.
- Retire `tcgen05_gemm/` (document what we learned, move it under
  `docs/experiments/`). Replace every hand-rolled MMA call site with
  a cuBLAS or CUTLASS call.
- Make `trainer.py` sit directly on `Sampler + Trainer +
  CheckpointStore` protocols from `rl/backend`.

### Phase 2 — finish the backward (3-5 days)

Following the list already in `train_megakernel/README.md`:

1. Activation saving during forward (per-layer normalized inputs, attn
   outputs, gate·up, down outputs). Mechanical — add pointers to the
   existing kernel arg pack.
2. Per-layer reverse kernels:
   - MLP bwd: down → silu_mul → gate/up bwd
   - Post-attn RMSNorm bwd
   - Attention bwd: FlashAttention-style for FA, BPTT for DN
   - QKV bwd → input RMSNorm bwd
3. Fuse each reverse step's LoRA gradient accumulation (A and B grads
   for the active projection) inside the same phase.
4. Gradient check against HF transformers + PEFT LoRA on a 16-token
   sample, tolerance 1e-3 bf16-relative.

### Phase 3 — cuBLAS/CUTLASS wiring  ✅ **done** (2026-04-23)

Instead of fusing LoRA into the megakernel's hand-rolled WMMA
matmuls, we plumbed an optional `LoraPFSet` into the existing
cuBLAS+cudaGraph prefill path (`models/qwen35_0p8b/prefill.cu`'s
`launch_prefill_bf16`). Two cuBLAS `GemmEx` calls per LoRA'd
projection:

```cpp
apply_lora_linear(cublas, X, A_layer, B_layer,
                  Y_base, lora_h_ws,
                  S, N, K_in, lora_rank, lora_scaling);
// internally:
//   lora_h = X @ A_layer                   (GemmEx, beta=0)
//   Y_base += scaling * (lora_h @ B_layer) (GemmEx, beta=1)
```

The B-matmul's `beta=1` accumulates straight into the base projection
output — no residual-add kernel needed. Null pointers disable LoRA on
individual projections, so the inference extension (`qwen35_megakernel_bf16_C`)
reuses the same binary with all-null slots and essentially zero overhead.
New op: `prefill_bf16_with_lora` in the same extension.

**Measured on B200, S=520, rank=16, LoRA active on all 13 linears:**

```
  trainer megakernel (hand-rolled WMMA, all in one dispatch) : 95.54 ms/step
  cuBLAS+graph+LoRA  (prefill_bf16_with_lora, this commit)   : 14.35 ms/step
                                                              →  6.66× faster
```

Correctness (`trainer/test_lora_forward_cublas.py`): zero-LoRA and
nonzero-LoRA cases produce bit-exact token ids matching the existing
trainer megakernel.

Inference path (no LoRA) on the same extension unchanged: pp520
~40,275 tok/s after the change (vs ~40,600 prior). The `LoraPFSet{}`
all-null branch in `launch_prefill_bf16` costs one predictable
null-check per projection per layer — within noise.

The `tcgen05_gemm/` sandbox is retired at `experiments/tcgen05_gemm/`.

### Phase 4 — rl/backend integration (1-2 days)

- Implement `LoraMegakernelTrainer` class in `trainer.py` satisfying
  `Sampler + Trainer + CheckpointStore`.
- Checkpoint format: PEFT-compatible adapter dir
  (`adapter_config.json` + `adapter_model.safetensors`) per the
  existing `LORA_TRAINING_DECISION.md` — so checkpoints load into
  SGLang and HF-PEFT unchanged.
- Optimizer state persist: safetensors of the flat m/v buffers.
- Register in rl/backend's runtime registry; existing
  `LoRATrainer` stays as the CPU-torch fallback for tests.

### Phase 5 — bench + publish (1 day)

- `tests/bench_training_throughput.py` vs the torch reference.
- Update `docs/results/qwen35_0p8b_b200.md` with training numbers.
- Update `models/qwen35_0p8b/README.md` with the training-mode
  section.

### Critical path estimate

8-13 engineering days end to end. Phase 2 dominates. Phase 4 is
fastest because the seam already exists.

## Non-goals

- Multi-GPU training. Single B200, one trainable run at a time, per
  the `LORA_TRAINING_DECISION.md` Phase-1 scope.
- Full-parameter fine-tuning. LoRA rank up to ~64; above that the
  in-kernel activation storage stops fitting and we'd want to spill
  to global.
- Training the 27B target. `models/qwen35_27b/` has its own roadmap
  (`docs/roadmap/qwen35_27b_on_sm100.md`) — the training engine
  here targets `qwen35_0p8b` first, then `qwen36_3b` when that model
  lands.
- Competing with SGLang on pure sampling throughput. We match it and
  add training; SGLang stays the serving runtime.

## Open design questions

- **Optimizer state in bf16 or fp32?** Fused AdamW today keeps m/v
  in fp32 (matches torch). Could pack to bf16 for 2× memory savings
  at the cost of some numerical wobble. Defer to Phase 5 numerics
  pass.
- **Activation recomputation vs save?** For rank-16 LoRA on 0.8B the
  saved activations fit easily (24 layers × 520 tokens × 1024 ×
  2 bytes ≈ 25 MB). At 27B this blows up; pick an "always save at
  rank R ≤ threshold, else recompute" cutoff when we port.
- **Inference weight-sharing with training.** The same LoRA A/B
  buffers should be live for both fwd-only inference and fwd+bwd
  training. Single megakernel with two entry points, or two kernels
  sharing weight pointers? Single kernel with a boolean flag is
  simpler; evaluate cost of the extra branch predicates in Phase 5.

## Success criteria

1. `pytest -x models/qwen35_0p8b/trainer/tests/` passes on B200 with
   gradient check tolerance ≤ 1e-3 bf16-relative.
2. `LoraMegakernelTrainer` registered as a runtime in
   `~/rl/backend` and `bench_training_throughput.py` reports ≥ 3×
   tokens/second over `LoRATrainer` on a rank-16 LoRA over all 13
   projections on Qwen3.5-0.8B.
3. Exported checkpoint loads into SGLang's `--load-lora-adapter`
   unchanged and samples produce identical logits to the trainer
   runtime's own `sample` within bf16 rounding.
4. One row in `docs/results/qwen35_0p8b_b200.md` with training
   tok/s + per-step ms, alongside the existing prefill and decode
   numbers.

## Where to commit

- Main dev on the existing `b200-train` worktree / branch. Merge
  into `b200` when Phase 2 backward is gradient-check-clean.
- Integration work touches `~/rl/backend` — that's a separate repo,
  a PR there will land the registry registration once the lucebox
  side ships to `b200`.
