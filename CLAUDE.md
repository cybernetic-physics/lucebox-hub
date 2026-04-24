# Claude-specific notes

See `AGENTS.md` for the full agent guide; this file is a terse
restatement of the pins.

## Hard rules

1. **GPU 3 only.** Export `CUDA_VISIBLE_DEVICES=3` before any
   torch/CUDA work. Other B200s (0/1/2) may be shared.
2. **Work on the `b200-train` branch.** Do not touch `main` or `b200`
   unless the user asks for a merge.
3. Commit messages and docs: no emojis unless the user asks.
4. Confirm before destructive git operations (force push, branch
   delete, reset --hard).

## Quick starts

```bash
export CUDA_VISIBLE_DEVICES=3

# Rebuild trainer extension
cd models/qwen35_0p8b/trainer && MAX_JOBS=4 python3 setup.py build_ext --inplace

# Rebuild CUTLASS extension (needs /root/cutlass)
cd models/qwen35_0p8b/trainer/cutlass_train && MAX_JOBS=4 python3 setup.py build_ext --inplace

# End-to-end correctness + speed
python3 models/qwen35_0p8b/trainer/bench_lora_e2e.py --ppl-tokens 4096 --ctx-len 512 --ppl-stride 256

# LoraMegakernelTrainer lifecycle test
python3 models/qwen35_0p8b/trainer/test_rl_trainer_e2e.py
```

## Status pointers

Live phase state in `docs/roadmap/lora_training_engine.md`.
Benchmarks in `docs/results/qwen35_0p8b_b200.md`.
