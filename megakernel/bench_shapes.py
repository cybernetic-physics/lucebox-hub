"""Sweep prefill / decode throughput across input shapes for the BF16
megakernel path. Single Python process: loads the Qwen3.5-0.8B weights
once, allocates a KV cache and scratch buffers sized for the largest
shape, then re-uses slices of those buffers for each S in the sweep.

Outputs a markdown-style table to stdout and (optionally) a JSON file
with the same data.

Usage:
  python3 bench_shapes.py
  python3 bench_shapes.py --shapes 128,512,2048,8192 --runs 5
  python3 bench_shapes.py --decode-tokens 0   # skip decode timings
"""
import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoTokenizer

from model import (
    DN_CONV_CHANNELS,
    DN_NUM_HEADS,
    DN_V_SIZE,
    FA_KV_SIZE,
    FA_Q_SIZE,
    FA_QPROJ_SIZE,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    Decoder,
)
import qwen35_megakernel_bf16_C  # noqa: F401  (registers torch.ops)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--shapes",
        default="32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536",
        help="Comma-separated list of S (prefill prompt token counts).",
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument(
        "--decode-tokens",
        type=int,
        default=64,
        help="Decode tokens timed after each prefill (0 = skip).",
    )
    p.add_argument("--decode-warmup", type=int, default=8)
    p.add_argument("--model-name", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--out", default="", help="Optional JSON output path.")
    p.add_argument("--label", default="", help="Free-form label written into JSON.")
    return p.parse_args()


def alloc_prefill_buffers(S):
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    f32 = dict(dtype=torch.float32, device="cuda")
    i32 = dict(dtype=torch.int32, device="cuda")
    mx = max(DN_CONV_CHANNELS, FA_QPROJ_SIZE, INTERMEDIATE_SIZE)
    return dict(
        hidden=torch.empty(S * HIDDEN_SIZE, **bf16),
        residual=torch.empty(S * HIDDEN_SIZE, **bf16),
        normalized=torch.empty(S * HIDDEN_SIZE, **bf16),
        proj_buf=torch.empty(S * mx, **bf16),
        proj_buf2=torch.empty(S * mx, **bf16),
        attn_buf=torch.empty(S * max(FA_Q_SIZE, FA_KV_SIZE), **bf16),
        mlp_buf=torch.empty(S * INTERMEDIATE_SIZE, **bf16),
        dn_out_buf=torch.empty(S * DN_V_SIZE, **bf16),
        beta_buf=torch.empty(S * DN_NUM_HEADS, **f32),
        alpha_buf=torch.empty(S * DN_NUM_HEADS, **f32),
        final_normed=torch.empty(HIDDEN_SIZE, **bf16),
        hidden_bf16_out=torch.empty(HIDDEN_SIZE, **bf16),
        lm_bmv=torch.empty(1024, **f32),
        lm_bmi=torch.empty(1024, **i32),
    )


def build_prompt_ids(tokenizer, target_tokens):
    seed = (
        "Explain in great detail the history of artificial intelligence, "
        "machine learning, deep learning, and neural networks. "
    )
    text = seed
    while True:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) >= target_tokens:
            return ids[:target_tokens]
        text += seed * max(1, target_tokens // len(ids))


def main():
    args = parse_args()
    shapes = [int(s) for s in args.shapes.split(",") if s.strip()]
    shapes.sort()
    biggest = shapes[-1]

    # Auto-detect device + capability for the report header.
    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    print(f"# Shape sweep — {name}, sm_{cap[0]}{cap[1]}", flush=True)
    print(f"# shapes: {shapes}", flush=True)
    print(f"# warmup={args.warmup} runs={args.runs} decode_tokens={args.decode_tokens}", flush=True)

    # Load weights once. Allocate KV cache for the biggest S we want to test
    # (plus decode tokens, since decode writes positions S..S+decode_tokens).
    max_seq_len = biggest + args.decode_tokens + args.decode_warmup + 32
    print(f"# max_seq_len = {max_seq_len}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    decoder = Decoder(
        model_name=args.model_name,
        verbose=False,
        max_seq_len=max_seq_len,
    )
    pf = torch.ops.qwen35_megakernel_bf16_C.prefill_bf16

    biggest_ids = build_prompt_ids(tokenizer, biggest)
    print(f"# prompt pool length: {len(biggest_ids)} tokens", flush=True)

    rows = []
    for S in shapes:
        # Slice prompt to length S; allocate buffers sized for S.
        ids_t = torch.tensor(biggest_ids[:S], dtype=torch.int32, device="cuda")
        try:
            buffers = alloc_prefill_buffers(S)
        except torch.cuda.OutOfMemoryError as e:
            print(f"S={S:>6}: OOM during alloc ({e})", flush=True)
            torch.cuda.empty_cache()
            continue

        def run_prefill():
            decoder.reset()
            pf(
                decoder._out_token, ids_t,
                decoder._embed_weight, decoder._layer_weights_packed,
                decoder._final_norm_weight, decoder._lm_head_weight,
                decoder._fa_k_cache, decoder._fa_v_cache,
                decoder._dn_states, decoder._conv_bufs,
                buffers["hidden"], buffers["residual"], buffers["normalized"],
                buffers["proj_buf"], buffers["proj_buf2"],
                buffers["attn_buf"], buffers["mlp_buf"],
                buffers["dn_out_buf"],
                buffers["beta_buf"], buffers["alpha_buf"],
                buffers["final_normed"], buffers["hidden_bf16_out"],
                buffers["lm_bmv"], buffers["lm_bmi"],
            )
            decoder._hidden.copy_(buffers["hidden_bf16_out"])
            decoder._position = S
            return decoder._out_token.item()

        # Prefill warmup + timed.
        for _ in range(args.warmup):
            run_prefill()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.runs):
            run_prefill()
            torch.cuda.synchronize()
        pf_ms = (time.perf_counter() - t0) / args.runs * 1000.0
        pf_tps = S / pf_ms * 1000.0

        # Decode timing — start from a freshly-prefilled state, do warmup
        # decode steps to stabilize, then time the next N steps.
        tg_tps = None
        tg_ms_per_token = None
        if args.decode_tokens > 0:
            try:
                tok_id = run_prefill()
                for _ in range(args.decode_warmup):
                    tok_id = decoder.step(tok_id)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(args.decode_tokens):
                    tok_id = decoder.step(tok_id)
                torch.cuda.synchronize()
                tg_total_s = time.perf_counter() - t0
                tg_ms_per_token = tg_total_s / args.decode_tokens * 1000.0
                tg_tps = args.decode_tokens / tg_total_s
            except RuntimeError as e:
                print(f"S={S:>6}: decode failed ({e})", flush=True)

        row = dict(
            S=S,
            pf_ms=pf_ms,
            pf_tps=pf_tps,
            tg_tps=tg_tps,
            tg_ms_per_token=tg_ms_per_token,
        )
        rows.append(row)
        decode_str = (
            f"  tg {tg_tps:7.1f} t/s  ({tg_ms_per_token:6.2f} ms/tok)"
            if tg_tps is not None
            else "  tg ----"
        )
        print(
            f"S={S:>6}: pf {pf_tps:9.1f} t/s  ({pf_ms:8.2f} ms total)" + decode_str,
            flush=True,
        )

        # Free the per-S buffers before the next iteration.
        del buffers
        torch.cuda.empty_cache()

    # Markdown table summary.
    print()
    print("|       S |   pp tok/s |  pp ms |   tg tok/s |  tg ms/tok |")
    print("|--------:|-----------:|-------:|-----------:|-----------:|")
    for r in rows:
        tg_tps = f"{r['tg_tps']:>10.1f}" if r["tg_tps"] is not None else "          —"
        tg_mpt = f"{r['tg_ms_per_token']:>10.3f}" if r["tg_ms_per_token"] is not None else "          —"
        print(f"| {r['S']:>7} | {r['pf_tps']:>10.1f} | {r['pf_ms']:>6.2f} | {tg_tps} | {tg_mpt} |")

    if args.out:
        payload = dict(
            label=args.label,
            device_name=name,
            cuda_capability=f"{cap[0]}.{cap[1]}",
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            warmup=args.warmup,
            runs=args.runs,
            decode_tokens=args.decode_tokens,
            decode_warmup=args.decode_warmup,
            max_seq_len=max_seq_len,
            rows=rows,
        )
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"# wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
