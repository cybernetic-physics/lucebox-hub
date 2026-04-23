"""
10 prompts per dataset, AR + DFlash per prompt.

    python3 scripts/bench_llm.py

Paths resolve from the repo root by default. Override with env vars:
    DFLASH_TARGET   path to target Qwen3.5-27B-Q4_K_M.gguf
    DFLASH_DRAFT    path to draft model.safetensors
    DFLASH_BIN      path to build/test_dflash
    DFLASH_BIN_AR   path to build/test_generate
"""
import json, os, re, struct, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = os.environ.get(
    "DFLASH_TARGET",
    str(ROOT / "models" / "Qwen3.5-27B-Q4_K_M.gguf"),
)
_DRAFT_ROOT = (
    Path.home() / ".cache/huggingface/hub"
    / "models--z-lab--Qwen3.5-27B-DFlash" / "snapshots"
)


def _resolve_draft() -> str:
    env = os.environ.get("DFLASH_DRAFT")
    if env:
        return env
    for st in _DRAFT_ROOT.rglob("model.safetensors"):
        return str(st)
    return str(_DRAFT_ROOT / "model.safetensors")


DRAFT = _resolve_draft()
TEST_DFLASH   = os.environ.get("DFLASH_BIN",    str(ROOT / "build" / "test_dflash"))
TEST_GENERATE = os.environ.get("DFLASH_BIN_AR", str(ROOT / "build" / "test_generate"))

N_GEN = 256
BUDGET = 22
N_SAMPLE = 10

BENCHES = [
    ("HumanEval", "openai_humaneval", None,   "test", lambda x: x["prompt"]),
    ("GSM8K",     "gsm8k",            "main", "test", lambda x: f"Question: {x['question']}\nAnswer: "),
    ("Math500",   "HuggingFaceH4/MATH-500", None, "test", lambda x: f"Problem: {x['problem']}\nSolution: "),
]


def tokenize(tok, p, path):
    ids = tok.encode(p, add_special_tokens=False)
    with open(path, "wb") as f:
        for t in ids: f.write(struct.pack("<i", int(t)))
    return len(ids)


def run_ar(path):
    r = subprocess.run([TEST_GENERATE, TARGET, path, str(N_GEN), "/tmp/ar_out.bin"],
                       capture_output=True, text=True, timeout=300)
    m = re.search(r"(\d+\.\d+)\s+tok/s", r.stdout)
    return float(m.group(1)) if m else 0.0


def _auto_max_ctx(n_prompt):
    # Auto-fit attention budget: prompt + gen + small verify pad, aligned to
    # FATTN_KQ_STRIDE=256. Oversizing max_ctx makes attention stride over
    # unused KV and can cost >20× prefill time (32K prompt + --kv-q4 +
    # max_ctx=131072 → 1035s vs 38s at max_ctx=32768). See scripts/run.py.
    pad = 64  # covers q_len=16 + ddtree budget up to 22 with margin
    return ((n_prompt + N_GEN + pad + 255) // 256) * 256


def run_df(path, n_prompt):
    max_ctx = _auto_max_ctx(n_prompt)
    r = subprocess.run([TEST_DFLASH, TARGET, DRAFT, path, str(N_GEN), "/tmp/df_out.bin",
                        "--fast-rollback", "--ddtree", f"--ddtree-budget={BUDGET}",
                        f"--max-ctx={max_ctx}"],
                       capture_output=True, text=True, timeout=300)
    tps = re.search(r"→\s+(\d+\.\d+)\s+tok/s", r.stdout)
    al  = re.search(r"avg commit/step=(\d+\.\d+)", r.stdout)
    return (float(tps.group(1)) if tps else 0.0,
            float(al.group(1))  if al else 0.0)


def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B", trust_remote_code=True)

    results = {}
    for name, ds_name, cfg, split, extract in BENCHES:
        print(f"\n[bench] ==== {name} (n={N_SAMPLE}) ====", flush=True)
        ds = load_dataset(ds_name, cfg, split=split)
        ds = ds.shuffle(seed=42).select(range(N_SAMPLE))
        ar_tps, df_tps, df_al = [], [], []
        for i, s in enumerate(ds):
            p = extract(s)
            path = f"/tmp/b_{name}_{i:02d}.bin"
            n = tokenize(tok, p, path)
            if n == 0 or n > 3500:
                continue
            ar = run_ar(path)
            df, al = run_df(path, n)
            if ar > 0: ar_tps.append(ar)
            if df > 0: df_tps.append(df); df_al.append(al)
            print(f"  [{i+1:02d}/{N_SAMPLE}] n_tok={n:4d}  AR={ar:6.2f}  DFlash={df:7.2f}  AL={al:5.2f}", flush=True)
        ar_m = sum(ar_tps)/len(ar_tps) if ar_tps else 0
        df_m = sum(df_tps)/len(df_tps) if df_tps else 0
        al_m = sum(df_al)/len(df_al) if df_al else 0
        results[name] = {"ar": ar_m, "dflash": df_m, "al": al_m,
                         "speedup": df_m/ar_m if ar_m else 0}
        print(f"  {name} mean: AR={ar_m:.2f}  DFlash={df_m:.2f}  AL={al_m:.2f}  {results[name]['speedup']:.2f}x", flush=True)

    print("\n[bench] === SUMMARY ===")
    print(f"{'Task':12s}  {'AR':>8s}  {'DFlash':>8s}  {'AL':>6s}  {'Speedup':>8s}")
    for name, r in results.items():
        print(f"{name:12s}  {r['ar']:8.2f}  {r['dflash']:8.2f}  {r['al']:6.2f}  {r['speedup']:7.2f}x")

    with open("/tmp/bench_llm_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
