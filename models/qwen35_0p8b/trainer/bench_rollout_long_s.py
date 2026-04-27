"""Long-context rollout sweep: how does our megakernel rollout scale
vs HF generate as the prompt grows?

Same config as bench_trainer_vs_sglang_torch.py but sweeps prompt
length up to where the kernel KV cache supports (~32K). Each rollout
generates a fixed number of new tokens (default 32) so the per-step
decode cost is comparable across prompt lengths — only the prefill
cost grows with S.

Reports prompt-tokens/sec for the rollout's prefill phase + decode
tok/s for the generation phase.
"""
from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")

import torch  # noqa: E402

from rl_trainer import LoraMegakernelTrainer  # noqa: E402


def time_fn(fn, runs=3, warm=1):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-lens", type=int, nargs="+",
                    default=[128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
    ap.add_argument("--gen-tokens", type=int, default=32)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warm", type=int, default=1)
    args = ap.parse_args()

    print(f"Loading trainer + HF reference...")
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

    # HF baseline: same model, generate path.
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to("cuda").eval()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")

    print()
    print(f"Rollout shape sweep: gen={args.gen_tokens} tokens, runs={args.runs}")
    print()
    print(f"{'P':>6} | {'HF gen ms':>11} {'ours ms':>11} {'speedup':>8} "
          f"{'HF tok/s':>10} {'our tok/s':>11}")
    print("-" * 70)

    for P in args.prompt_lens:
        prompt = list(range(2, 2 + P))

        # Some shapes may exceed HF's KV cache or just take forever — skip
        # via try/except.
        try:
            def hf_gen():
                ids = torch.tensor(prompt, dtype=torch.long, device="cuda").unsqueeze(0)
                with torch.no_grad():
                    base.generate(
                        ids, max_new_tokens=args.gen_tokens, do_sample=False,
                        pad_token_id=tok.eos_token_id,
                    )

            def our_gen():
                trainer.sample(prompt_tokens=prompt, max_tokens=args.gen_tokens,
                               num_samples=1, prompt_logprobs=False,
                               topk_prompt_logprobs=0)

            # Adaptive: drop runs at long S so the bench finishes.
            r = args.runs if P <= 4096 else max(1, args.runs - 1)
            w = args.warm if P <= 4096 else 0

            ms_hf = time_fn(hf_gen, runs=r, warm=w)
            ms_ours = time_fn(our_gen, runs=r, warm=w)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            print(f"{P:>6} | error: {str(e).splitlines()[0][:50]}")
            torch.cuda.empty_cache()
            continue

        speedup = ms_hf / ms_ours
        # tok/s here = new generated tokens per sec (excludes prompt).
        hf_tps = args.gen_tokens / (ms_hf / 1000.0)
        our_tps = args.gen_tokens / (ms_ours / 1000.0)
        print(f"{P:>6} | {ms_hf:>11.1f} {ms_ours:>11.1f} {speedup:>7.2f}x "
              f"{hf_tps:>10.0f} {our_tps:>11.0f}")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
