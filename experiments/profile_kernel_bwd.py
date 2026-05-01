"""Per-kernel profile of the kernel-bwd training step at P=1024.

Identifies the actual hotspots in the kernel-bwd path so the fused
mega-bwd design (#20) targets the right composition boundaries.
"""
from __future__ import annotations

import os
import sys
import torch

sys.path.insert(0, "/home/freiza/lucebox-hub/models/qwen35_0p8b/trainer")
sys.path.insert(0, "/home/freiza/lucebox-hub/models/qwen35_0p8b")

# Force kernel-bwd before importing rl_trainer
os.environ["MEGAKERNEL_USE_KERNEL_BWD"] = "1"


def main():
    from rl_trainer import LoraMegakernelTrainer
    P, T = 1024, 32

    trainer = LoraMegakernelTrainer(verbose_loader=False)
    trainer.register_model(
        model_id="t",
        base_model="Qwen/Qwen3.5-0.8B",
        lora_rank=8,
        train_mlp=True, train_attn=True, train_unembed=False,
        user_metadata=None,
    )
    data = [{
        "model_input": {"chunks": [{"type": "input", "tokens": list(range(10, 10 + P))}]},
        "loss_fn_inputs": {"target_tokens": list(range(100, 100 + T))},
    }]
    adam = {"lr": 1e-4, "betas": (0.9, 0.999), "eps": 1e-8, "wd": 0.01}

    # Warm
    trainer.forward_backward(model_id="t", data=data, loss_fn="cross_entropy")
    trainer.optim_step(model_id="t", adam_params=adam)
    trainer.forward_backward(model_id="t", data=data, loss_fn="cross_entropy")
    trainer.optim_step(model_id="t", adam_params=adam)
    torch.cuda.synchronize()

    # Profile
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        trainer.forward_backward(model_id="t", data=data, loss_fn="cross_entropy")
        trainer.optim_step(model_id="t", adam_params=adam)
        torch.cuda.synchronize()

    print()
    print(f"Top CUDA kernels by self-time at P={P}, T={T}, kernel-bwd:")
    print("-" * 92)
    events = prof.key_averages()
    cuda_events = [e for e in events if e.self_device_time_total > 0]
    cuda_events.sort(key=lambda e: e.self_device_time_total, reverse=True)
    total_us = sum(e.self_device_time_total for e in cuda_events)
    for e in cuda_events[:25]:
        ms = e.self_device_time_total / 1000.0
        share = 100.0 * e.self_device_time_total / max(total_us, 1)
        name = (e.key[:70] + "..") if len(e.key) > 72 else e.key
        print(f"{ms:>8.2f} ms  {share:>5.1f}%  ({e.count:>5}×)  {name}")
    print("-" * 92)
    print(f"sum self_device_time = {total_us/1000:.1f} ms across {len(cuda_events)} kernels")
    n_launches = sum(e.count for e in cuda_events)
    print(f"total launches = {n_launches}")


if __name__ == "__main__":
    main()
