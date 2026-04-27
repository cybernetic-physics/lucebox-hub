"""Bench Decoder.prefill-based rollout vs the legacy per-token-step path.

Uses the same shape as bench_trainer_vs_sglang_torch.py: 128-token
prompt, 64 generated tokens. Reports ms/rollout for both.
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")

import torch  # noqa: E402

from rl_trainer import LoraMegakernelTrainer  # noqa: E402


def time_fn(fn, runs=5, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / runs


def step_only_rollout(decoder, prompt_tokens, max_tokens, eos):
    """Replicate the pre-prefill .sample() path: feed each prompt token
    through decoder.step() one at a time, then continue stepping for
    generation."""
    decoder.reset()
    for tid in prompt_tokens[:-1]:
        decoder.step(int(tid))
    pred = decoder.step(int(prompt_tokens[-1]))
    out_ids = []
    for _ in range(max_tokens):
        if pred == eos:
            break
        out_ids.append(int(pred))
        pred = decoder.step(int(pred))
    return out_ids


def main():
    trainer = LoraMegakernelTrainer(verbose_loader=False)
    trainer.register_model(
        model_id="bench",
        base_model="Qwen/Qwen3.5-0.8B",
        lora_rank=8,
        train_mlp=True,
        train_attn=True,
        train_unembed=False,
        user_metadata=None,
    )

    prompt = list(range(2, 130))
    max_tokens = 64

    # First sample call instantiates the decoder.
    trainer.sample(prompt_tokens=prompt, max_tokens=max_tokens,
                   num_samples=1, prompt_logprobs=False, topk_prompt_logprobs=0)
    decoder = trainer._decoder
    eos = trainer._tokenizer.eos_token_id

    def fn_prefill():
        trainer.sample(prompt_tokens=prompt, max_tokens=max_tokens,
                       num_samples=1, prompt_logprobs=False,
                       topk_prompt_logprobs=0)

    def fn_step():
        step_only_rollout(decoder, prompt, max_tokens, eos)

    print(f"Prompt length: {len(prompt)}, gen tokens: {max_tokens}")
    print()
    ms_step = time_fn(fn_step, runs=5, warm=2)
    ms_prefill = time_fn(fn_prefill, runs=5, warm=2)

    print(f"  per-token-step path : {ms_step:8.2f} ms/rollout")
    print(f"  prefill path        : {ms_prefill:8.2f} ms/rollout")
    print(f"  speedup             : {ms_step / ms_prefill:8.2f}x")
    print()


if __name__ == "__main__":
    main()
