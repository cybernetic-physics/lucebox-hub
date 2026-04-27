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

### Phase 5 — end-to-end bench + 3× claim

`bench_trainer_vs_sglang_torch.py` measures three stages vs a
HF/torch reference RL loop (stand-in for SGLang + torch-PEFT). The
incumbent here is HF transformers + fla (Triton chunk_gated_delta_rule)
+ PEFT + torch.optim.AdamW — i.e. the **optimized torch stack** on B200.

Numbers (B200, batch=4, S=128, gen=64, 2026-04-27 with fla available,
hybrid DN routing live, **Tier 0.1 + Tier 0.3 landed**):

| stage                                    | incumbent | ours    | speedup |
|:-----------------------------------------|:---------:|:-------:|:-------:|
| (A) sample 64 tokens (greedy)            | 1908.9 ms |   77.1 ms | **24.75×** |
| (B) training step (batch=4, seq=128)     |  114.3 ms |  114.3 ms |  1.00× by construction (\*) |
| (C) combined RL step = sample + train    | 2023.2 ms |  191.5 ms | **10.57×** |

(\*) Both sides use the same HF+fla+PEFT+torch.optim.AdamW pipeline;
"speedup" on (B) means our trainer matches the incumbent on this stage.
Tier 0.2 (replace HF+PEFT with our custom fwd/bwd kernels + fused AdamW)
is the unlock for a real (B) win.

**10.57× over HF+fla.** The win comes from two changes since the
previous round:

- **Tier 0.1** (prefill rollout): use `prefill_bf16` for the prompt
  instead of stepping the cooperative decode kernel once per prompt
  token. Sampling 210 ms → 77 ms.
- **Tier 0.3** (batched forward in `forward_backward`): when all
  examples in `data` share `(prompt_len, target_len)`, pack into one
  `[B, P+T]` tensor and run a single forward+CE+backward instead of
  one per example. Training step 471 ms → 114 ms (4.12×). Heterogeneous
  shapes fall back to the per-example loop. The training step
is parity by construction: both sides use HF+PEFT for fwd+bwd and
torch.optim.AdamW for the step, since `LoraMegakernelTrainer` routes
training through fla's chunked Triton bwd via the hybrid DN router
(commit 239db57) and that's the same path the incumbent takes.

**vs real SGLang:** SGLang is typically 2-3× faster than HF generate
for sampling, so the measured (A) ratio of 9.26× would scale to
roughly 3-4× vs SGLang. End-to-end (C) ratio against an SGLang+torch
incumbent would drop to ~1.5-2.5× — past the original 3× target only
once we add an actual training-side win. The realistic next claim:
**3× over HF+fla today, ~2× over SGLang+fla.**

**Per-layer DN, hybrid routing vs HF+fla on B200** (`bench_vs_fla.py`,
2026-04-27):

| mode  | S    | HF+fla    | hybrid    | ratio                |
|-------|------|-----------|-----------|----------------------|
| train | 128  | 103.2 ms  | 96.2 ms   | **1.07× ours**       |
| infer | 128  |  39.3 ms  | 37.6 ms   | **1.04× ours**       |
| train | 256  |  90.5 ms  | 101.7 ms  | 0.89× fla (routing regression — fix) |
| infer | 256  |  48.2 ms  | 37.8 ms   | **1.27× ours**       |
| train | 512  | 103.0 ms  | 101.7 ms  | 1.01× tied           |
| infer | 512  |  48.4 ms  | 38.1 ms   | **1.27× ours**       |
| train | 1024 | 133.3 ms  | 103.4 ms  | **1.29× ours**       |
| infer | 1024 |  50.8 ms  | 51.9 ms   | 0.98× tied           |
| train | 2048 | 139.2 ms  | 130.9 ms  | **1.06× ours**       |
| infer | 2048 |  48.0 ms  | 59.5 ms   | 0.81× fla            |

Hybrid wins on 6/10 points, ties on 2, loses on 2 (S=256 train
routing regression; S=2048 infer where fla's chunk-parallelism
saturates SMs).

### What's needed to push the win further

1. **Chunk-parallelism rewrite** of the DN kernel (~1-2 weeks). Match
   fla's "one block per chunk, chunks across SMs" architecture so
   long-S training goes from "tied" to "decisively faster". Single
   biggest unlock for inference S ≥ 1024 and training S ≥ 256.
2. **CUDA chunked DN backward** (~3-5 days). Port the validated
   analytical Python reference (`dn_chunked_bwd_proto.py`,
   cos > 0.99998 vs torch autograd). Wired into a custom Trainer
   path (today's path uses fla); when paired with (1) at long S,
   pulls training from "tied" to "decisively faster".
3. **Fused AdamW on the LoRA flat buffer** (~1-2 days, kernel exists).
   Wire the existing 124× AdamW kernel into the trainer. Currently
   the trainer uses torch.optim.AdamW because grads are still in
   PEFT's per-tensor layout; need a flatten/scatter shim or a
   torch._foreach-based wrapper.
4. **Custom FA bwd in trainer** (~1 day, kernel exists). The cuDNN
   FA-2 wrapper (`fa_bwd_flash.py`, 1.65× vs autograd SDPA) isn't
   plumbed into `LoraMegakernelTrainer`'s training path yet.
5. **Routing fix for S=256 train** (hours). The hybrid dispatcher
   sometimes picks the slower path at this shape; tune the threshold
   from S=512 to whatever makes 256 train pick fla.

Items 3-5 are short-ROI; 1-2 are the structural unlocks.

### Critical path estimate

Phase 5 publish (results doc + headline numbers) is now the only thing
between us and "done with the original roadmap". Beyond that, the
optimization roadmap (chunk-parallelism rewrite + chunked bwd port +
custom-bwd plumbing) is what pushes the engine from "matches HF+fla"
to "decisively beats it". See `models/qwen35_0p8b/trainer/PERF_NOTES.md`
for the prioritized list with measured kernel costs.

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
