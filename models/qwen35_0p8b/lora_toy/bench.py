"""Walltime comparison: fused megakernel vs a pure-torch equivalent step.

Measures per-step latency for the same math:
  forward -> log-softmax/CE -> backward to lora_a/lora_b -> AdamW.
Torch path runs with cuda graphs disabled (same as how rl/backend runs it today).
"""

from __future__ import annotations

import time
import torch

from model import AdamConfig, LoRATrainStepKernel
from test_correctness import TorchReference


def bench(cfg, n_warmup=10, n_iter=50):
    device = torch.device("cuda")
    ref = TorchReference(
        base_model=cfg["base"], vocab_size=cfg["V"], hidden_size=cfg["H"],
        lora_rank=cfg["R"], device=device,
    )
    kernel = LoRATrainStepKernel.from_base_model(
        base_model=cfg["base"], lora_rank=cfg["R"],
        vocab_size=cfg["V"], hidden_size=cfg["H"],
        max_seq_len=max(64, cfg["T"]), device=device,
    )

    adam = AdamConfig(lr=1e-3)
    T = cfg["T"]
    ctx = torch.randint(0, cfg["V"], (T,), device=device, dtype=torch.int32)
    tgt = torch.randint(0, cfg["V"], (T,), device=device, dtype=torch.int32)
    w = torch.ones(T, device=device, dtype=torch.float32)

    for _ in range(n_warmup):
        ref.train_step(ctx, tgt, w, adam)
        kernel.train_step(context_tokens=ctx, target_tokens=tgt, token_weights=w, adam=adam)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        ref.train_step(ctx, tgt, w, adam)
    torch.cuda.synchronize()
    ref_ms = (time.perf_counter() - t0) * 1000 / n_iter

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        kernel.train_step(context_tokens=ctx, target_tokens=tgt, token_weights=w, adam=adam)
    torch.cuda.synchronize()
    k_ms = (time.perf_counter() - t0) * 1000 / n_iter

    return ref_ms, k_ms


def main():
    shapes = [
        {"base": "Llama-3.2-1B", "V": 256, "H": 64, "R": 8, "T": 8},
        {"base": "Llama-3.2-1B", "V": 1024, "H": 128, "R": 16, "T": 16},
        {"base": "Qwen2.5", "V": 4096, "H": 256, "R": 16, "T": 32},
        {"base": "Qwen2.5", "V": 8192, "H": 512, "R": 16, "T": 32},
    ]

    print(f"{'V':>6}  {'H':>5}  {'R':>4}  {'T':>4}  {'torch (ms)':>12}  {'megakernel (ms)':>18}  {'speedup':>8}")
    for cfg in shapes:
        ref_ms, k_ms = bench(cfg)
        print(f"{cfg['V']:>6}  {cfg['H']:>5}  {cfg['R']:>4}  {cfg['T']:>4}  {ref_ms:>12.3f}  {k_ms:>18.3f}  {ref_ms/k_ms:>7.2f}x")


if __name__ == "__main__":
    main()
