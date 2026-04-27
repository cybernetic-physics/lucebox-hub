"""Bench the kernel-driven Slice B.3b backward vs HF+PEFT autograd.

Same trainer, same data, same shape — just toggle the env var. Measures
ms/step for forward_backward + optim_step combined.

Goal: a clear baseline so the next round of kernel-bwd optimizations
(replace autograd interiors with hand-rolled FA-bwd / dn_bwd) has a
measured starting point to beat.
"""
from __future__ import annotations

import os
import sys
import time

# We toggle the env var per measurement, so DON'T set it before
# importing the trainer. Just import normally; the trainer reads the
# env var INSIDE forward_backward each call.

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")

import torch  # noqa: E402

from rl_trainer import LoraMegakernelTrainer  # noqa: E402


def make_datum(prompt_tokens, target_tokens):
    return {
        "model_input": {"chunks": [{"type": "input", "tokens": list(prompt_tokens)}]},
        "loss_fn_inputs": {"target_tokens": list(target_tokens)},
    }


def time_step(trainer, model_id, batch, runs=3, warm=1):
    for _ in range(warm):
        trainer.forward_backward(model_id=model_id, data=batch,
                                  loss_fn="cross_entropy", loss_fn_config=None)
        trainer.optim_step(model_id=model_id,
                           adam_params={"learning_rate": 5e-4, "weight_decay": 0.0})
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        trainer.forward_backward(model_id=model_id, data=batch,
                                  loss_fn="cross_entropy", loss_fn_config=None)
        trainer.optim_step(model_id=model_id,
                           adam_params={"learning_rate": 5e-4, "weight_decay": 0.0})
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / runs


def main():
    print("Loading trainer + registering test model...")
    trainer = LoraMegakernelTrainer(verbose_loader=False)
    model_id = "bench-bwd"
    trainer.register_model(
        model_id=model_id,
        base_model="Qwen/Qwen3.5-0.8B",
        lora_rank=8,
        train_mlp=True, train_attn=True,
        train_unembed=False,
    )

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    pairs = [
        ("The quick brown fox jumps over the", " lazy dog."),
        ("Python is a popular", " programming language."),
        ("Machine learning is a subset of", " artificial intelligence."),
        ("The capital of France is", " Paris."),
    ]
    batch = []
    for p, t in pairs:
        pids = tok.encode(p, add_special_tokens=False)
        tids = tok.encode(t, add_special_tokens=False)
        batch.append(make_datum(pids, tids))

    print()
    print(f"{'path':<35} {'ms/step':>10}")
    print("-" * 50)

    # HF+PEFT path (default).
    os.environ.pop("MEGAKERNEL_USE_KERNEL_BWD", None)
    ms_hf = time_step(trainer, model_id, batch, runs=3, warm=2)
    print(f"{'HF+PEFT autograd (default)':<35} {ms_hf:>9.1f}ms")

    # Kernel-driven path.
    os.environ["MEGAKERNEL_USE_KERNEL_BWD"] = "1"
    ms_k = time_step(trainer, model_id, batch, runs=3, warm=2)
    print(f"{'Kernel-driven Slice B.3b':<35} {ms_k:>9.1f}ms")

    print("-" * 50)
    if ms_k < ms_hf:
        print(f"Kernel path is {ms_hf/ms_k:.2f}× FASTER")
    else:
        print(f"Kernel path is {ms_k/ms_hf:.2f}× slower (expected — autograd-based")
        print(f"  per_layer_bwd_dn does HF DN forward+bwd via autograd, ie. recompute)")
    print()
    # Also report how the per-layer reverse walk's two pieces split.
    # (No microbench here — just the overall numbers.)


if __name__ == "__main__":
    main()
