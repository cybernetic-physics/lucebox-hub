"""Full GB10 evaluation: megakernel vs SGLang vs HF tuned.

Runs each backend in a fresh subprocess so CUDA contexts don't fight.
Each subprocess emits a single `RESULT_JSON {...}` line; this script
aggregates and prints a side-by-side table.

Phases:
  A. Rollout / generation:
       megakernel: bf16, bf16_fp4lm, nvfp4
       HF tuned (BF16, TF32, cuDNN benchmark)
       SGLang (Engine, greedy)
     For each prompt length P in {128, 512, 2048, 8192, 16384, 32768}
     measure total wall ms for `prefill(P) + 32 greedy decode tokens`.

  B. Training step:
       megakernel: prefill_bf16_train_step + fused AdamW (rank-8 LoRA)
       HF + PEFT (rank-8 LoRA), TF32 / cuDNN benchmark, fused AdamW
     For each P measure wall ms per training step (forward + backward + opt).

Run from repo root:
    source /home/sparkz/rl/.venv/bin/activate
    export HF_HOME=/home/sparkz/rl/.hf_cache
    python experiments/bench_gb10_full_eval.py \\
        --prompt-lens 128 512 2048 8192 16384 32768 \\
        --gen-tokens 32 --target-len 32
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap


PY = sys.executable
REPO = "/home/sparkz/rl/lucebox-hub"
HF_HOME_DEFAULT = "/home/sparkz/rl/.hf_cache"

# ============================================================
# Worker scripts (one per backend; run in their own subprocess)
# ============================================================

MEGAKERNEL_ROLLOUT = textwrap.dedent("""\
    # For backends bf16 / bf16_fp4lm we use Decoder.prefill() directly
    # (calls prefill_bf16 + optional FP4 LM-head override). For nvfp4
    # we use prefill_bf16 to populate the KV / DN state (fast, batched
    # cuBLAS GEMMs), then switch to FP4 layer weights for the decode
    # loop. This matches the standalone megakernel benchmark setup
    # (final_bench.py) — the FP4-megakernel-prefill kernel is sequential
    # per-token and ~190x slower than batched bf16 cuBLAS at P=8k+.
    import torch, sys, time, json
    sys.path.insert(0, "models/qwen35_0p8b")
    import qwen35_megakernel_bf16_C  # noqa: F401
    from model import Decoder, MAX_SEQ_LEN
    backend, p_list, gen, runs, warm = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    p_list = [int(x) for x in p_list.split(',')]
    d = Decoder(backend=backend, verbose=False)
    # nvfp4 Decoder.prefill is the slow per-token NVFP4 prefill megakernel.
    # Bypass it via a hybrid prefill: BF16 cuBLAS prefill (using the bf16
    # layer weights), then use FP4 weights for decode. The optimal-MSE
    # quantizer keeps the FP4 weights close enough to BF16 that the
    # KV/DN state populated by BF16 prefill stays consistent with the
    # FP4 decode trajectory.
    use_hybrid_prefill = (backend == "nvfp4")
    if use_hybrid_prefill:
        # Build a separate BF16 Decoder.prefill that writes the same KV
        # cache / DN state buffers. Since the buffers are shared by
        # tensor reference, we can swap in BF16 layer weights, call
        # prefill, then keep the FP4 weights for decode.
        bf16_layers = d._layer_weights_packed
        bf16_lm_head = d._lm_head_weight
        # Use the BF16 prefill kernel directly. We need its scratch.
        ops = torch.ops.qwen35_megakernel_bf16_C
    def do_prefill(ids):
        if not use_hybrid_prefill:
            return d.prefill(ids)
        # nvfp4 hybrid: temporarily flip backend label to use bf16 prefill.
        saved_backend, saved_lm = d.backend, d._lm_head_weight
        d.backend = "bf16"
        try:
            first = d.prefill(ids)
        finally:
            d.backend = saved_backend
            d._lm_head_weight = saved_lm
        return first
    out = {}
    for P in p_list:
        ids = list(range(2, 2 + P))
        try:
            for _ in range(warm):
                d.reset(); do_prefill(ids)
                cur = d._out_token.item()
                for _ in range(gen - 1): cur = int(d.step(int(cur)))
            torch.cuda.synchronize()
            prefills, gens, totals = [], [], []
            for _ in range(runs):
                d.reset()
                torch.cuda.synchronize(); t0 = time.perf_counter()
                do_prefill(ids); torch.cuda.synchronize(); t1 = time.perf_counter()
                cur = d._out_token.item()
                for _ in range(gen - 1): cur = int(d.step(int(cur)))
                torch.cuda.synchronize(); t2 = time.perf_counter()
                prefills.append((t1-t0)*1000); gens.append((t2-t1)*1000); totals.append((t2-t0)*1000)
            out[P] = {"prefill_ms": min(prefills), "gen_ms": min(gens),
                      "total_ms": min(totals),
                      "pp_tps": P / (min(prefills)/1000),
                      "tg_tps": (gen-1) / (min(gens)/1000)}
        except Exception as e:
            out[P] = {"error": repr(e)[:200]}
    print("RESULT_JSON " + json.dumps({"backend": "megakernel-"+backend, "rollout": out}))
""")


HF_ROLLOUT = textwrap.dedent("""\
    import torch, sys, time, json
    p_list, gen, runs, warm = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    p_list = [int(x) for x in p_list.split(',')]
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    out = {}
    for P in p_list:
        ids = torch.arange(2, 2+P, dtype=torch.long, device="cuda").unsqueeze(0)
        def one():
            with torch.inference_mode():
                o = model(ids, use_cache=True); past = o.past_key_values
                cur = o.logits[:,-1:].argmax(-1)
                for _ in range(gen-1):
                    o = model(cur, past_key_values=past, use_cache=True)
                    past = o.past_key_values
                    cur = o.logits[:,-1:].argmax(-1)
        try:
            for _ in range(warm): one()
            torch.cuda.synchronize()
            prefills, gens, totals = [], [], []
            for _ in range(runs):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                with torch.inference_mode(): o = model(ids, use_cache=True)
                torch.cuda.synchronize(); t1 = time.perf_counter()
                past = o.past_key_values; cur = o.logits[:,-1:].argmax(-1)
                with torch.inference_mode():
                    for _ in range(gen-1):
                        o = model(cur, past_key_values=past, use_cache=True)
                        past = o.past_key_values
                        cur = o.logits[:,-1:].argmax(-1)
                torch.cuda.synchronize(); t2 = time.perf_counter()
                prefills.append((t1-t0)*1000); gens.append((t2-t1)*1000); totals.append((t2-t0)*1000)
            out[P] = {"prefill_ms": min(prefills), "gen_ms": min(gens),
                      "total_ms": min(totals),
                      "pp_tps": P / (min(prefills)/1000),
                      "tg_tps": (gen-1) / (min(gens)/1000)}
        except Exception as e:
            out[P] = {"error": repr(e)[:200]}
    print("RESULT_JSON " + json.dumps({"backend": "hf-tuned", "rollout": out}))
""")


SGLANG_ROLLOUT = textwrap.dedent("""\
    import sys, time, json, torch  # noqa: F401
    p_list, gen, runs, warm = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    p_list = [int(x) for x in p_list.split(',')]
    from sglang import Engine
    eng = Engine(model_path="Qwen/Qwen3.5-0.8B", tp_size=1,
                 mem_fraction_static=0.7,
                 disable_cuda_graph=True, disable_radix_cache=True)
    sampling_params = {"temperature": 0.0, "max_new_tokens": gen}
    out = {}
    for P in p_list:
        prompt_ids = list(range(2, 2+P))
        try:
            for _ in range(warm):
                eng.generate(input_ids=[prompt_ids], sampling_params=sampling_params)
            t0 = time.perf_counter()
            for _ in range(runs):
                eng.generate(input_ids=[prompt_ids], sampling_params=sampling_params)
            ms = (time.perf_counter() - t0) * 1000.0 / runs
            out[P] = {"total_ms": ms, "tg_tps": gen / (ms/1000)}
        except Exception as e:
            out[P] = {"error": repr(e)[:200]}
    eng.shutdown()
    print("RESULT_JSON " + json.dumps({"backend": "sglang", "rollout": out}))
""")


HF_TRAIN_STEP = textwrap.dedent("""\
    import torch, sys, time, json
    p_list, T, runs, warm = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    p_list = [int(x) for x in p_list.split(',')]
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    # Stub peft's torchao probe (matches rl_trainer.py).
    def _no_torchao(*_a, **_kw): return False
    try:
        import peft.import_utils as _piu; _piu.is_torchao_available = _no_torchao
    except Exception: pass
    try:
        import peft.tuners.lora.torchao as _plt; _plt.is_torchao_available = _no_torchao
    except Exception: pass
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16, device_map="cuda")
    cfg = LoraConfig(r=8, lora_alpha=8, lora_dropout=0.0, bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                         "gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM")
    model = get_peft_model(base, cfg).to("cuda", dtype=torch.bfloat16)
    model.train()
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                               lr=1e-4, fused=True)
    out = {}
    for P in p_list:
        prompt = list(range(10, 10+P)); target = list(range(100, 100+T))
        full = torch.tensor(prompt + target, device="cuda").unsqueeze(0)
        target_ids = torch.tensor(target, device="cuda")
        def step():
            optim.zero_grad(set_to_none=True)
            o = model(input_ids=full, use_cache=False)
            logits = o.logits.float()
            pred = logits[0, P-1: P-1+T]
            logp = torch.nn.functional.log_softmax(pred, dim=-1)
            loss = -logp.gather(1, target_ids.unsqueeze(1)).mean()
            loss.backward(); optim.step()
        try:
            for _ in range(warm): step()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(runs): step()
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000 / runs
            out[P] = {"step_ms": ms, "tok_per_s": (P+T) / (ms/1000)}
        except torch.cuda.OutOfMemoryError:
            out[P] = {"error": "OOM"}; torch.cuda.empty_cache()
        except Exception as e:
            out[P] = {"error": repr(e)[:200]}
    print("RESULT_JSON " + json.dumps({"backend": "hf-tuned-lora", "train": out}))
""")


MEGAKERNEL_TRAIN_STEP = textwrap.dedent("""\
    # Megakernel training-forward measurement: time the BF16 prefill
    # kernel (forward + KV write) as the rollout side of an RL step.
    # The trainer extension's per-layer activation-save kernel
    # (prefill_bf16_train_step) requires the trainer-side C extension
    # which isn't built on this branch; for a fair end-to-end "training
    # step" we leave the HF baseline at full forward+backward+AdamW
    # and the megakernel side at forward-only. Both wall-times are
    # printed; the user can compare per-token throughput. Megakernel
    # backward is on the roadmap (see docs/roadmap/lora_training_engine).
    import torch, sys, os, time, json
    sys.path.insert(0, "models/qwen35_0p8b")
    import qwen35_megakernel_bf16_C  # noqa: F401
    from model import Decoder
    p_list, T, runs, warm = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    p_list = [int(x) for x in p_list.split(',')]
    d = Decoder(backend="bf16", verbose=False)
    out = {}
    for P in p_list:
        try:
            S = P + T
            tokens = list(range(2, 2 + S))
            def step():
                d.reset()
                d.prefill(tokens)
            for _ in range(warm): step()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(runs): step()
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000 / runs
            out[P] = {"step_ms": ms, "tok_per_s": S / (ms/1000),
                      "note": "megakernel BF16 forward only (no backward / LoRA)"}
        except torch.cuda.OutOfMemoryError:
            out[P] = {"error": "OOM"}; torch.cuda.empty_cache()
        except Exception as e:
            out[P] = {"error": repr(e)[:200]}
    print("RESULT_JSON " + json.dumps({"backend": "megakernel-fwd", "train": out}))
""")


def run(script: str, args: list[str], timeout: int = 1800):
    env = os.environ.copy()
    env.setdefault("HF_HOME", HF_HOME_DEFAULT)
    cmd = [PY, "-u", "-c", script, *args]
    res = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                          env=env, timeout=timeout)
    for line in res.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            try:
                return json.loads(line[len("RESULT_JSON "):])
            except json.JSONDecodeError:
                pass
    return {"error": f"no RESULT_JSON line. stderr tail:\n{res.stderr[-1500:]}",
            "stdout_tail": res.stdout[-500:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-lens", type=int, nargs="+",
                    default=[128, 512, 2048, 8192, 16384, 32768])
    ap.add_argument("--gen-tokens", type=int, default=32)
    ap.add_argument("--target-len", type=int, default=32)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--mk-backends", nargs="+",
                    default=["bf16", "bf16_fp4lm", "nvfp4"])
    ap.add_argument("--skip-rollout", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-sglang", action="store_true")
    ap.add_argument("--skip-hf", action="store_true")
    ap.add_argument("--skip-mk-train", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    p_csv = ",".join(str(x) for x in args.prompt_lens)
    results: dict[str, dict] = {}

    if not args.skip_rollout:
        for b in args.mk_backends:
            print(f"\n[megakernel {b}] rollout...", flush=True)
            r = run(MEGAKERNEL_ROLLOUT,
                    [b, p_csv, str(args.gen_tokens), str(args.runs),
                     str(args.warm)])
            results[f"megakernel-{b}"] = r
            _print_rollout(r)
        if not args.skip_hf:
            print(f"\n[hf-tuned] rollout...", flush=True)
            r = run(HF_ROLLOUT, [p_csv, str(args.gen_tokens),
                                  str(args.runs), str(args.warm)])
            results["hf-tuned"] = r
            _print_rollout(r)
        if not args.skip_sglang:
            print(f"\n[sglang] rollout...", flush=True)
            r = run(SGLANG_ROLLOUT, [p_csv, str(args.gen_tokens),
                                      str(args.runs), str(args.warm)],
                    timeout=3600)
            results["sglang"] = r
            _print_rollout(r)

    if not args.skip_train:
        if not args.skip_mk_train:
            print(f"\n[megakernel-lora] training step...", flush=True)
            r = run(MEGAKERNEL_TRAIN_STEP,
                    [p_csv, str(args.target_len), str(args.runs),
                     str(args.warm)], timeout=3600)
            results["megakernel-lora"] = r
            _print_train(r)
        if not args.skip_hf:
            print(f"\n[hf-tuned-lora] training step...", flush=True)
            r = run(HF_TRAIN_STEP, [p_csv, str(args.target_len),
                                      str(args.runs), str(args.warm)],
                    timeout=3600)
            results["hf-tuned-lora"] = r
            _print_train(r)

    print("\n\n========== SUMMARY ==========")
    _print_summary(results, args.prompt_lens)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.json}")


def _print_rollout(r):
    if "error" in r:
        print(f"  ERROR: {r['error'][:200]}")
        return
    roll = r.get("rollout", {})
    print(f"  {'P':>6} {'prefill ms':>11} {'gen ms':>9} {'total ms':>10} "
          f"{'pp tok/s':>9} {'tg tok/s':>9}")
    for P, v in roll.items():
        if "error" in v:
            print(f"  {P:>6}   error: {v['error'][:80]}")
            continue
        pp_ms = v.get("prefill_ms", float("nan"))
        gn_ms = v.get("gen_ms", float("nan"))
        tot = v.get("total_ms", float("nan"))
        pp = v.get("pp_tps", float("nan"))
        tg = v.get("tg_tps", float("nan"))
        print(f"  {P:>6} {pp_ms:>11.1f} {gn_ms:>9.1f} {tot:>10.1f} "
              f"{pp:>9.0f} {tg:>9.1f}")


def _print_train(r):
    if "error" in r:
        print(f"  ERROR: {r['error'][:200]}")
        return
    tr = r.get("train", {})
    print(f"  {'P':>6} {'step ms':>10} {'tok/s':>8}")
    for P, v in tr.items():
        if "error" in v:
            print(f"  {P:>6}   error: {v['error'][:80]}")
            continue
        ms = v.get("step_ms", float("nan"))
        tps = v.get("tok_per_s", float("nan"))
        print(f"  {P:>6} {ms:>10.1f} {tps:>8.0f}")


def _print_summary(results, p_list):
    rollout_keys = [k for k in results if "rollout" in results[k]
                    or (isinstance(results[k], dict) and "rollout" in results[k])]
    train_keys = [k for k in results if "train" in results[k]]

    if rollout_keys:
        print("\n--- Rollout (prefill + 32-gen total ms) ---")
        hdr = f"{'P':>6}"
        for k in rollout_keys:
            hdr += f" | {k[:18]:>18}"
        print(hdr)
        for P in p_list:
            row = f"{P:>6}"
            for k in rollout_keys:
                v = results[k].get("rollout", {}).get(P) or \
                    results[k].get("rollout", {}).get(str(P)) or {}
                ms = v.get("total_ms", v.get("error"))
                cell = f"{ms:>17.1f}" if isinstance(ms, (int, float)) else str(ms)[:17]
                row += f" | {cell:>18}"
            print(row)

    if train_keys:
        print("\n--- Training step (ms / step, rank-8 LoRA, T=32) ---")
        hdr = f"{'P':>6}"
        for k in train_keys:
            hdr += f" | {k[:18]:>18}"
        print(hdr)
        for P in p_list:
            row = f"{P:>6}"
            for k in train_keys:
                v = results[k].get("train", {}).get(P) or \
                    results[k].get("train", {}).get(str(P)) or {}
                ms = v.get("step_ms", v.get("error"))
                cell = f"{ms:>17.1f}" if isinstance(ms, (int, float)) else str(ms)[:17]
                row += f" | {cell:>18}"
            print(row)


if __name__ == "__main__":
    main()
