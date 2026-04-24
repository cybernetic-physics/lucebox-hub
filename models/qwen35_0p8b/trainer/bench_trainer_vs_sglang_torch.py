"""Phase 5 bench: LoraMegakernelTrainer vs SGLang-sample + torch-LoRA-train.

The Phase 5 target is 3x faster end-to-end than the incumbent RL
runtime (SGLang for sampling + HF+PEFT+torch.optim.AdamW for training).
We can't run SGLang inside this test harness, so we use HF/torch as a
stand-in for BOTH sides of the incumbent — HF as the "SGLang" sampler
(same torch path, same compute) and HF+PEFT+torch AdamW as the trainer.
That's a pessimistic baseline for the incumbent (real SGLang is faster
than torch-HF, so our margin vs SGLang will be smaller than this).

Measures:
  (A) Sample throughput on a single prompt-continuation workload.
  (B) Training-step wall time: fwd + bwd + optim over a batch.
  (C) Combined RL step = sample(N tokens) + train(one mini-batch).

Summary is printed alongside each stage's speedup. We do NOT claim the
full 3x yet — that's gated on Phase 2 backward kernels landing so the
training path can use our fast forward + custom bwd + fused AdamW.
Today the trainer uses the HF path for backward, so training-step
speedup is ~1x (by construction). The sampling column does show the
real speedup our kernel achieves.
"""
from __future__ import annotations

import argparse
import sys
import time

import torch

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")
from rl_trainer import LoraMegakernelTrainer


def make_datum(prompt_tokens, target_tokens):
    return {
        "model_input": {"chunks": [{"type": "input", "tokens": list(prompt_tokens)}]},
        "loss_fn_inputs": {"target_tokens": list(target_tokens)},
    }


def bench_sampling(trainer, prompt_tokens, *, max_tokens=32, warmup=2, runs=5):
    for _ in range(warmup):
        trainer.sample(prompt_tokens=prompt_tokens, max_tokens=max_tokens,
                       num_samples=1, prompt_logprobs=False, topk_prompt_logprobs=0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        trainer.sample(prompt_tokens=prompt_tokens, max_tokens=max_tokens,
                       num_samples=1, prompt_logprobs=False, topk_prompt_logprobs=0)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / runs


def bench_pytorch_sampling(hf_model, tok, prompt_tokens, *, max_tokens=32, warmup=2, runs=5):
    prompt = torch.tensor(prompt_tokens, dtype=torch.long, device="cuda").unsqueeze(0)
    @torch.no_grad()
    def one():
        # Greedy decode via HF .generate.
        _ = hf_model.generate(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            use_cache=True,
        )
    for _ in range(warmup):
        one()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        one()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / runs


def bench_training_step(trainer, model_id, batch, *, warmup=2, runs=5):
    for _ in range(warmup):
        trainer.forward_backward(model_id=model_id, data=batch,
                                 loss_fn="cross_entropy", loss_fn_config=None)
        trainer.optim_step(model_id=model_id,
                           adam_params={"learning_rate": 1e-4, "weight_decay": 0.0})
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        trainer.forward_backward(model_id=model_id, data=batch,
                                 loss_fn="cross_entropy", loss_fn_config=None)
        trainer.optim_step(model_id=model_id,
                           adam_params={"learning_rate": 1e-4, "weight_decay": 0.0})
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    print("=" * 78)
    print("Phase 5 bench: LoraMegakernelTrainer vs HF/torch-reference")
    print("=" * 78)

    trainer = LoraMegakernelTrainer(
        artifact_root="/tmp/lora_phase5", verbose_loader=False)
    model_id = "phase5-bench"
    trainer.register_model(
        model_id=model_id,
        base_model="Qwen/Qwen3.5-0.8B",
        lora_rank=8, train_mlp=True, train_attn=True, train_unembed=False,
        user_metadata=None,
    )

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    hf_ref = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16).to("cuda")
    hf_ref.eval()

    # Build a repeatable batch of datums (prompt_len + target_len = seq_len).
    # Tokenizer compresses repeated text heavily — repeat until we have
    # enough raw tokens to slice batch_size * seq_len from.
    seed = ("Explain in great detail the history of artificial intelligence, "
            "machine learning, deep learning, and neural networks. ")
    text = seed
    prompt_ids = tok.encode(text, add_special_tokens=False)
    need = args.batch_size * args.seq_len + args.seq_len
    while len(prompt_ids) < need:
        text += seed
        prompt_ids = tok.encode(text, add_special_tokens=False)
    half = args.seq_len // 2
    batch = []
    for i in range(args.batch_size):
        off = i * args.seq_len
        p = prompt_ids[off : off + half]
        t = prompt_ids[off + half : off + args.seq_len]
        batch.append(make_datum(p, t))

    sample_prompt = prompt_ids[:args.seq_len]

    print()
    print(f"Config: batch_size={args.batch_size}, seq_len={args.seq_len}, "
          f"gen_tokens={args.gen_tokens}, runs={args.runs}")
    print()

    # ===== (A) Sampling =====
    our_sample_ms = bench_sampling(
        trainer, sample_prompt, max_tokens=args.gen_tokens, runs=args.runs)
    pt_sample_ms = bench_pytorch_sampling(
        hf_ref, tok, sample_prompt, max_tokens=args.gen_tokens, runs=args.runs)
    print(f"(A) Sampling ({args.gen_tokens} tokens, greedy):")
    print(f"    PyTorch HF (incumbent) : {pt_sample_ms:8.2f} ms  ({args.gen_tokens/(pt_sample_ms/1000):>7.0f} tok/s)")
    print(f"    ours (megakernel)      : {our_sample_ms:8.2f} ms  ({args.gen_tokens/(our_sample_ms/1000):>7.0f} tok/s)")
    print(f"    speedup                : {pt_sample_ms/our_sample_ms:>6.2f}x")
    print()

    # ===== (B) Training step =====
    our_train_ms = bench_training_step(trainer, model_id, batch, runs=args.runs)
    # Pure-torch reference: identical HF+PEFT path that the trainer wraps.
    # We reuse the trainer's own session for a fair head-to-head (same
    # HF state, same params). The "incumbent" here is exactly what we
    # delegate to internally, so training speedup is by construction 1x
    # until Phase 2 backward kernels land.
    print(f"(B) Training step (batch={args.batch_size}, seq={args.seq_len}):")
    print(f"    trainer fwd+bwd+optim  : {our_train_ms:8.2f} ms")
    print(f"    (Phase 2 FA+DN bwd kernels pending — training speedup = 1x today)")
    print()

    # ===== (C) Combined RL step =====
    print(f"(C) Combined RL step (sample {args.gen_tokens} tok + 1 train step):")
    combined_ours = our_sample_ms + our_train_ms
    combined_incumbent = pt_sample_ms + our_train_ms   # same train cost
    print(f"    incumbent (HF sample + torch train) : {combined_incumbent:8.2f} ms")
    print(f"    ours (megakernel sample + torch train): {combined_ours:8.2f} ms")
    print(f"    end-to-end speedup                   : {combined_incumbent/combined_ours:>6.2f}x")
    print()
    print("Notes:")
    print("  * Sampling speedup is real and measured against HF/torch on the same")
    print("    weights. SGLang is typically ~2-3x faster than plain HF for sampling,")
    print("    so the measured ratio vs SGLang would be proportionally smaller.")
    print("  * Combined speedup is bottlenecked by training today; Phase 2 custom")
    print("    FA backward + DeltaNet BPTT kernels will unblock the 3x claim.")
    print("  * Fused AdamW (124x vs torch.optim.AdamW) already available — plugged")
    print("    in once grads are in the flat buffer our kernel expects.")


if __name__ == "__main__":
    main()
