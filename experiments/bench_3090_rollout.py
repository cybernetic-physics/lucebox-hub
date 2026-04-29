"""Compare RTX 3090 rollout perf: 3090-train HEAD vs fork-parent-b200~2 vs HF.

Run from the repo root with the venv:
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=models/qwen35_0p8b \
        .venv-3090/bin/python experiments/bench_3090_rollout.py

Each variant is loaded into a fresh subprocess so the C extensions don't
collide in a single Python process.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap


VARIANTS = {
    # name -> (qwen35 dir, decoder import path)
    "head": (
        "/home/freiza/lucebox-hub/models/qwen35_0p8b",
        "head",
    ),
    "fpb200~2": (
        "/home/freiza/lucebox-hub-fpb200/megakernel",
        "fpb200",
    ),
    "fpb200~3": (
        "/home/freiza/lucebox-hub-fpb200-3/megakernel",
        "fpb200",
    ),
    "fpb200~4": (
        "/home/freiza/lucebox-hub-fpb200-4/megakernel",
        "fpb200",
    ),
    "fpb200~5": (
        "/home/freiza/lucebox-hub-fpb200-5/megakernel",
        "fpb200",
    ),
}


WORKER = textwrap.dedent("""\
    import os, sys, time, json
    import torch  # must come before the C extension imports
    BASE = sys.argv[1]
    MODE = sys.argv[2]
    sys.path.insert(0, BASE)
    if MODE == 'head':
        # 3090-train HEAD has model.py in the qwen35 dir
        import qwen35_megakernel_bf16_C  # noqa: F401
        from model import Decoder
    else:
        # fork-parent-b200~2 has model.py in megakernel/
        import qwen35_megakernel_bf16_C  # noqa: F401
        from model import Decoder

    out = {}
    for P in [int(x) for x in sys.argv[3].split(',')]:
        # Construct a fresh decoder per shape so we can size max_seq_len.
        try:
            decoder = Decoder(verbose=False, max_seq_len=max(2048, P + int(sys.argv[4]) + 64))
        except TypeError:
            decoder = Decoder(verbose=False)
        prompt = list(range(2, 2 + P))
        gen = int(sys.argv[4])
        # Warm: reset state and prime once.
        decoder.reset()
        try:
            if hasattr(decoder, 'prefill'):
                _ = decoder.prefill(prompt)
            else:
                for tid in prompt[:-1]:
                    decoder.step(int(tid))
                _ = decoder.step(int(prompt[-1]))
            torch.cuda.synchronize()
        except Exception as e:
            sys.stderr.write(f"[worker] warmup P={P} failed: {repr(e)[:200]}\\n")
            out[P] = {"prefill_ms": float('nan'), "gen_ms": float('nan'), "total_ms": float('nan'), "error": repr(e)[:120]}
            continue
        t_total_runs = []
        t_prefill_runs = []
        t_gen_runs = []
        try:
            for _ in range(int(sys.argv[5])):
                decoder.reset()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                if hasattr(decoder, 'prefill'):
                    pred = decoder.prefill(prompt)
                else:
                    for tid in prompt[:-1]:
                        decoder.step(int(tid))
                    pred = decoder.step(int(prompt[-1]))
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                ids = [int(pred)]
                for _ in range(gen - 1):
                    pred = decoder.step(int(pred))
                    ids.append(int(pred))
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                t_prefill_runs.append((t1 - t0) * 1000)
                t_gen_runs.append((t2 - t1) * 1000)
                t_total_runs.append((t2 - t0) * 1000)
            out[P] = {
                "prefill_ms": min(t_prefill_runs),
                "gen_ms": min(t_gen_runs),
                "total_ms": min(t_total_runs),
            }
        except Exception as e:
            sys.stderr.write(f"[worker] P={P} failed: {repr(e)[:200]}\\n")
            out[P] = {"prefill_ms": float('nan'), "gen_ms": float('nan'), "total_ms": float('nan'), "error": repr(e)[:120]}
    sys.stdout.write("RESULTS:" + json.dumps(out))
""")


def run_variant(name, P_list, gen_tokens, runs):
    base, mode = VARIANTS[name]
    cmd = [
        "/home/freiza/lucebox-hub/.venv-3090/bin/python",
        "-c", WORKER, base, mode,
        ",".join(str(x) for x in P_list), str(gen_tokens), str(runs),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True,
                         env={**__import__('os').environ, "CUDA_VISIBLE_DEVICES": "1"})
    if res.returncode != 0:
        print(f"[{name}] FAILED:\n", res.stderr[-2500:], file=sys.stderr)
        return None
    line = next((l for l in res.stdout.splitlines() if l.startswith("RESULTS:")), None)
    if not line:
        print(f"[{name}] no RESULTS line. stdout tail:\n", res.stdout[-1500:], file=sys.stderr)
        print(f"[{name}] stderr tail:\n", res.stderr[-1500:], file=sys.stderr)
        return None
    return json.loads(line[len("RESULTS:"):])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-lens", type=int, nargs="+",
                    default=[128, 512, 2048, 8192, 16384, 32768])
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--variants", default="head,fpb200")
    args = ap.parse_args()

    variants = args.variants.split(",")
    all_results = {}
    for v in variants:
        print(f"==> running variant {v}")
        all_results[v] = run_variant(v, args.prompt_lens, args.gen, args.runs)

    print()
    hdr = f"{'P':>6} | " + " | ".join(
        f"{v + ' prefill ms':>16} {v + ' gen ms':>13}" for v in variants
    )
    print(hdr)
    print("-" * len(hdr))
    for P in args.prompt_lens:
        cells = []
        for v in variants:
            r = (all_results.get(v) or {}).get(str(P)) or {}
            pref = r.get("prefill_ms", float("nan"))
            gen = r.get("gen_ms", float("nan"))
            cells.append(f"{pref:>16.1f} {gen:>13.1f}")
        print(f"{P:>6} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
