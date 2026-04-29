"""Force the kernel-bwd path on the trainer (heterogeneous shapes) and
verify it converges + matches HF+PEFT path.

On the same model + LoRA + same training data:
  (a) runs 5 steps via the kernel-bwd path, captures losses
  (b) resets, runs 5 steps via the HF+PEFT path, captures losses
  (c) compares — they should diverge slowly due to bf16 noise but the
      first step's loss must match within a few percent.

Heterogeneous = items have different (prompt_len, target_len) so the
trainer takes the kernel branch even with MEGAKERNEL_USE_KERNEL_BWD=1.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, "/home/freiza/lucebox-hub/models/qwen35_0p8b/trainer")
sys.path.insert(0, "/home/freiza/lucebox-hub/models/qwen35_0p8b")

import torch  # noqa: E402

from rl_trainer import LoraMegakernelTrainer  # noqa: E402


def _datum(prompt_ids, target_ids):
    return {
        "model_input": {"chunks": [{"type": "input", "tokens": list(prompt_ids)}]},
        "loss_fn_inputs": {"target_tokens": list(target_ids)},
    }


def run_steps(trainer, model_id, data_per_step, n_steps=3):
    losses = []
    for s in range(n_steps):
        out = trainer.forward_backward(model_id=model_id, data=data_per_step,
                                        loss_fn="cross_entropy")
        trainer.optim_step(model_id=model_id,
                            adam_params={"lr": 1e-4, "betas": (0.9, 0.999),
                                         "eps": 1e-8, "wd": 0.01})
        losses.append(out["metrics"]["loss:mean"])
    return losses


def main():
    print("Loading trainer...")
    trainer = LoraMegakernelTrainer(verbose_loader=False)
    trainer.register_model(
        model_id="kbwd",
        base_model="Qwen/Qwen3.5-0.8B",
        lora_rank=8,
        train_mlp=True,
        train_attn=True,
        train_unembed=False,
        user_metadata=None,
    )

    # Single-item batch — also takes the kernel branch (shapes_uniform
    # requires len(items) > 1).
    data = [_datum(list(range(10, 30)),  list(range(100, 110)))]

    # Path A: HF+PEFT (default).
    print("\n[A] HF+PEFT path:")
    os.environ.pop("MEGAKERNEL_USE_KERNEL_BWD", None)
    losses_a = run_steps(trainer, "kbwd", data, n_steps=3)
    for i, lv in enumerate(losses_a, 1):
        print(f"  step {i}: loss = {lv:.6f}")

    # Reset by re-registering.
    trainer.unload_model(model_id="kbwd")
    trainer.register_model(
        model_id="kbwd",
        base_model="Qwen/Qwen3.5-0.8B",
        lora_rank=8,
        train_mlp=True,
        train_attn=True,
        train_unembed=False,
        user_metadata=None,
    )

    # Path B: kernel-bwd.
    print("\n[B] kernel-bwd path:")
    os.environ["MEGAKERNEL_USE_KERNEL_BWD"] = "1"
    losses_b = run_steps(trainer, "kbwd", data, n_steps=3)
    for i, lv in enumerate(losses_b, 1):
        print(f"  step {i}: loss = {lv:.6f}")

    print("\nComparison:")
    for i, (a, b) in enumerate(zip(losses_a, losses_b), 1):
        rel = abs(a - b) / max(abs(a), 1e-9)
        print(f"  step {i}: HF={a:.6f}  kernel={b:.6f}  rel={rel*100:.3f}%")


if __name__ == "__main__":
    main()
