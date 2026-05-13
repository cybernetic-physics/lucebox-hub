"""Sweep MEGAKERNEL_DECODE_BLOCKS / MEGAKERNEL_LM_BLOCKS on GB10.

The decode kernel is launched cooperatively at `cached_decode_blocks =
active * SM_count` (sm_121a / 48 SMs). With the persistent megakernel
shape, the optimum is often a small multiple of the SM count, not the
cudaOccupancy default. The LM head is a separate non-cooperative
kernel and benefits from oversubscription.

We run each (decode, lm) combo in a fresh subprocess (env-resolved
once, cached for the process lifetime) and report decode tok/s.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap


WORKER = textwrap.dedent("""\
    import os, sys, time, json
    import torch
    sys.path.insert(0, "models/qwen35_0p8b")
    import qwen35_megakernel_bf16_C  # noqa: F401
    from model import Decoder
    backend = sys.argv[1]
    prompt_len = int(sys.argv[2])
    runs = int(sys.argv[3])
    warm = int(sys.argv[4])
    d = Decoder(model_name="Qwen/Qwen3.5-0.8B", backend=backend, verbose=False)
    ids = list(range(2, 2 + prompt_len))
    d.reset()
    if backend == "bf16":
        first = d.prefill(ids)
    else:
        for t in ids[:-1]: d.step(int(t))
        first = d.step(int(ids[-1]))
    cur = int(first)
    for _ in range(warm):
        cur = int(d.step(cur))
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(runs):
        cur = int(d.step(cur))
    e.record(); torch.cuda.synchronize()
    ms = s.elapsed_time(e) / runs
    print("RESULT_JSON " + json.dumps({"backend": backend, "prompt_len": prompt_len,
                                       "us_per_step": ms*1000, "tok_per_s": 1000/ms}))
""")


def run_one(backend, prompt_len, decode_blocks, lm_blocks, runs, warm):
    env = os.environ.copy()
    env["HF_HOME"] = env.get("HF_HOME", "/home/sparkz/rl/.hf_cache")
    if decode_blocks > 0:
        env["MEGAKERNEL_DECODE_BLOCKS"] = str(decode_blocks)
    else:
        env.pop("MEGAKERNEL_DECODE_BLOCKS", None)
    if lm_blocks > 0:
        env["MEGAKERNEL_LM_BLOCKS"] = str(lm_blocks)
    else:
        env.pop("MEGAKERNEL_LM_BLOCKS", None)
    cmd = ["python", "-c", WORKER, backend, str(prompt_len), str(runs), str(warm)]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        return {"error": res.stderr[-500:]}
    for line in res.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            return json.loads(line[len("RESULT_JSON "):])
    return {"error": "no result line"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="bf16", choices=("bf16", "nvfp4"))
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--decode-blocks", type=int, nargs="+",
                    default=[0, 48, 96, 144, 192])
    ap.add_argument("--lm-blocks", type=int, nargs="+",
                    default=[0, 256, 384, 512, 768, 1024])
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--warm", type=int, default=10)
    args = ap.parse_args()

    print(f"backend={args.backend} P={args.prompt_len} runs={args.runs}")
    print(f"{'decode':>8} {'lm':>6} | {'us/step':>8} {'tok/s':>8}")
    print("-" * 40)
    best = (-1, None)
    for db in args.decode_blocks:
        for lm in args.lm_blocks:
            r = run_one(args.backend, args.prompt_len, db, lm,
                        args.runs, args.warm)
            if "error" in r:
                print(f"{db:>8} {lm:>6} | ERROR: {r['error'][:60]}")
                continue
            tps = r["tok_per_s"]
            mark = ""
            if tps > best[0]:
                best = (tps, (db, lm))
                mark = "  *"
            print(f"{db:>8} {lm:>6} | {r['us_per_step']:>7.1f}  {tps:>7.1f}{mark}")
    print(f"\nbest: decode={best[1][0]} lm={best[1][1]} -> {best[0]:.1f} tok/s")


if __name__ == "__main__":
    main()
