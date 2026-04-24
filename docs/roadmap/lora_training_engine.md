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

1. Activation saving during forward ✅ **done** (2026-04-23) — see
   `trainer/test_activation_save.py`. New op
   `prefill_bf16_train_step` writes up to four per-layer slabs into
   caller-provided tensors during the same cuBLAS+graph forward as
   inference. All-null ⇒ inference path, zero overhead. Saved
   slabs:
   - `hidden_in[L, S, HIDDEN]`            pre input-RMSnorm
   - `normalized_in[L, S, HIDDEN]`        post input-RMSnorm (LoRA-A grad input)
   - `normalized_post_attn[L, S, HIDDEN]` post post-attn-RMSnorm (MLP-bwd input)
   - `mlp_inter[L, S, INTER]`             silu(gate)·up (down-bwd input)
2. Per-layer reverse kernels (status as of 2026-04-23):
   - MLP bwd: down → silu_mul → gate/up bwd ✅ (SwiGLU bwd + LoRA-linear bwd in `trainer/kernel.cu`)
   - Post-attn RMSNorm bwd ✅ (`bwd_rmsnorm_kernel`)
   - CE + LM-head bwd ✅ (`bwd_ce_lm_head_kernel`)
   - Attention bwd: FlashAttention-style for FA ⏳ **missing**
   - Attention bwd: BPTT for DN ⏳ **missing**
   - QKV bwd → input RMSNorm bwd ⏳ needs FA/DN grads to flow in first
3. Fuse each reverse step's LoRA gradient accumulation (A and B grads
   for the active projection) inside the same phase.
4. Gradient check against HF transformers + PEFT LoRA on a 16-token
   sample, tolerance 1e-3 bf16-relative.

**Recommended order of attack** (unblocks Phase 4 earliest):

a. Wire existing bwd kernels + a **torch-autograd fallback** for FA
   and DN into a Python `_backward(saved, lora)` helper. Uses saved
   activations + cuDNN SDPA for FA, reference torch DN for DN.
   Lands Phase 4 end-to-end (slow but correct) immediately.
b. Replace the torch-SDPA fallback with a CUDA FA bwd kernel when
   Phase 5 bench shows attention-bwd is the bottleneck.
c. Replace the reference DN bwd with a V-split BPTT kernel matching
   the forward's V-split recurrence layout.

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

### Phase 4 — rl/backend integration ✅ **done** (2026-04-24)

`LoraMegakernelTrainer` in `models/qwen35_0p8b/trainer/rl_trainer.py`.
Satisfies `Sampler + Trainer + CheckpointStore` protocols from
`~/rl/backend/src/tinker_backend/runtimes/base.py`:

```
register_model(model_id, rank, train_mlp, train_attn, ...)
  -> builds a PEFT-wrapped copy of HF Qwen3.5-0.8B per model_id,
     creates a torch.optim.AdamW over the LoRA params

forward(model_id, data, "cross_entropy") -> logprobs + loss
forward_backward(model_id, data, "cross_entropy") -> same + bwd
optim_step(model_id, adam_params) -> applies step, bumps session.step

sample(prompt_tokens, max_tokens, ...)
  -> uses our cuBLAS+graph prefill + cooperative decode kernel
     (megakernel Decoder) — 11x faster than HF PyTorch on gen=64

save_checkpoint(model_id, ...)
  -> PEFT adapter_model.safetensors + adapter_config.json +
     manifest.json; drop-in loadable by SGLang --load-lora-adapter
     and HF PEFT

load_weights(model_id, artifact_path) -> rehydrates LoRA params via
     peft.set_peft_model_state_dict; optimizer state optional
```

End-to-end test (`trainer/test_rl_trainer_e2e.py`) exercises the full
lifecycle:
1. register_model(rank=8, all attn+mlp)
2. 5 training steps on a 4-prompt batch: loss 1.19 → 0.0014
3. sample 30 tokens via megakernel: coherent continuation
4. save_checkpoint + load_weights: max |logp_before - logp_after|
   = 0.00e+00 (bit-exact roundtrip)
5. unload_model + forward() rejected

Today's backend is HF+PEFT + torch.optim.AdamW for the training
path (forward+bwd+step), our kernel for sampling. The Trainer API
lets us swap in custom FA/DN backward and fused AdamW later without
changing the public surface.

### Phase 5 — end-to-end bench + 3× claim (in progress)

`bench_trainer_vs_sglang_torch.py` measures three stages vs a
HF/torch reference RL loop (stand-in for SGLang + torch-PEFT):

| stage                                    | incumbent | ours    | speedup |
|:-----------------------------------------|:---------:|:-------:|:-------:|
| (A) sample 64 tokens (greedy)            | 2275 ms   |  206 ms | **11.0×** |
| (B) training step (batch=4, seq=128)     | 2324 ms   | 2324 ms |   1.0×    |
| (C) combined RL step = sample + train    | 4599 ms   | 2530 ms | **1.82×** |

Today's 1.82× is sampling-only — training uses the HF backward
path. The 3× target is gated on Phase 2 custom backward kernels
(FA bwd + DeltaNet BPTT), which would drop (B) from ~2300 ms to
~200 ms and pull (C) to ~400 ms → ~11× over the incumbent
end-to-end. At that point fused AdamW (124× vs torch.optim.AdamW)
also plugs in on the grad flat buffer.

Real SGLang is typically 2-3× faster than plain HF for sampling, so
the measured sampling speedup vs SGLang would scale to ~4-5×. The
end-to-end 3× target should hold against SGLang once Phase 2 lands.

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
