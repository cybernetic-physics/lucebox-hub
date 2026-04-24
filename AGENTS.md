# Agent notes — lucebox-hub (Qwen3.5-0.8B B200 training work)

## GPU selection

**Run all CUDA workloads on GPU 3.** The other B200s (0/1/2) may be
shared with other users / other processes. Set this in every shell
session, via every CUDA entry point the script uses:

```bash
export CUDA_VISIBLE_DEVICES=3
```

All test/bench scripts in `models/qwen35_0p8b/trainer/` are written
against a single CUDA context (device 0 after remapping). Running
them with `CUDA_VISIBLE_DEVICES=3` pins everything to GPU 3 without
code changes.

If you're invoking a tool that reads `CUDA_VISIBLE_DEVICES` late
(e.g. some torch distributed harnesses), set it before `import torch`
or pass `device='cuda:0'` (which is GPU 3 after the remap) explicitly.

For `nsys profile` runs, use `nsys profile --gpu-metrics-device=3 ...`
or equivalently run under `CUDA_VISIBLE_DEVICES=3`.

## Branch layout

- `main` — stable inference stack (pp520 = 40,278 tok/s on B200)
- `b200-train` — training work; LoRA forward + backward, CUTLASS
  scaffold, trainer. Push here.

## Active subprojects

- `models/qwen35_0p8b/trainer/` — LoRA training pipeline
  - `rl_trainer.py` — `LoraMegakernelTrainer` runtime for rl/backend
  - `cutlass_train/` — CUTLASS 3.x bf16 kernels (sm_100a)
- `docs/roadmap/lora_training_engine.md` — live phase status
- `docs/results/qwen35_0p8b_b200.md` — benchmark tables

## House rules

- Don't break `main` / `b200` inference. All training work lives on
  `b200-train`.
- `prefill_bf16_mega` (cooperative megakernel path) is the correctness
  reference — it matches HF-eager. Any change to the cuBLAS path must
  match `prefill_bf16_mega`'s output within bf16 tolerance.
- KV cache stride is currently 32768 — callers must allocate
  `fa_k_cache` / `fa_v_cache` at that row count.
- CUTLASS 3.x builds require `-arch=sm_100a` (the `a` variant).
  Plain `sm_100` compiles but runtime-asserts inside TMA code.
