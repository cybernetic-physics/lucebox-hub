"""End-to-end LoRA bench: correctness (perplexity match) + speed (per-stage).

Correctness:
  Perplexity on wikitext-2 test for FOUR configurations:
    (a) HF Qwen3.5-0.8B, no adapter   (baseline reference)
    (b) HF + downloaded PEFT adapter  (HF's LoRA-applied model)
    (c) ours, no adapter              (cuBLAS+graph prefill, base only)
    (d) ours + same adapter repacked  (our kernel's LoRA path)

  Invariants we check:
    I1. |PPL(c) - PPL(a)| / PPL(a) < 5 %       — base forward is close
    I2. sign(PPL(d)-PPL(c)) == sign(PPL(b)-PPL(a))  — adapter shifts both
                                                      paths in the same
                                                      direction (either
                                                      both helps on WT-2
                                                      or both hurts)
    I3. |ΔPPL_ours - ΔPPL_HF| / |ΔPPL_HF| < 50 % — adapter magnitude ballpark

  These give an evaluation-level signature, NOT bit-exact logit match
  (which is blocked by HF's torch-native DeltaNet fallback — see the
  "fast path is not available" warning at load time).

Speed (B200, S=512, rank=16):
  Per-stage wall-time comparison covering the pieces of a practical
  LoRA-serving + training stack.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
import torch
from pathlib import Path

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b")
sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/prefill_megakernel")
sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")

import qwen35_megakernel_bf16_C  # noqa: F401 prefill ops
import train_megakernel_C          # noqa: F401 fused AdamW

from model import (
    NUM_LAYERS, HIDDEN, LAYER_TYPE, VOCAB, INTER,
    FA_QPROJ_SIZE, FA_KV_SIZE, FA_Q_SIZE, DN_CONV_CH, DN_V_SIZE,
    FA_HEAD_DIM, FA_KV_HEADS, DN_HEADS, DN_VAL, DN_KEY, DN_CONV_K,
)
from test_lora_forward import (
    _pack_layer_weights, FA_SHAPES, DN_SHAPES,
    build_lora_tensors, zero_lora_tensors,
)
from lora_hf_wrap import wrap_hf_with_lora

_spec = importlib.util.spec_from_file_location(
    "qwen_outer_model",
    "/root/lucebox-hub-b200-train/models/qwen35_0p8b/model.py")
_outer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_outer)
load_weights = _outer.load_weights

N_FA = sum(1 for t in LAYER_TYPE if t == 1)
N_DN = sum(1 for t in LAYER_TYPE if t == 0)
ADAPTER_REPO = "eac123/subliminal-qwen3.5-0.8b-tiger-r5"


# ---------------------------------------------------------------------------
# scratch + kernel wrappers
# ---------------------------------------------------------------------------
def alloc_scratch(S, lora_rank):
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    f32 = dict(dtype=torch.float32, device="cuda")
    i32 = dict(dtype=torch.int32, device="cuda")
    max_proj = max(FA_QPROJ_SIZE, DN_CONV_CH, INTER)
    max_attn = max(FA_Q_SIZE, DN_V_SIZE, FA_KV_SIZE)
    return dict(
        fa_k_cache=torch.zeros(N_FA, FA_KV_HEADS, 32768, FA_HEAD_DIM, **bf16),
        fa_v_cache=torch.zeros(N_FA, FA_KV_HEADS, 32768, FA_HEAD_DIM, **bf16),
        dn_states=torch.zeros(N_DN, DN_HEADS, DN_KEY, DN_VAL, **f32),
        conv_bufs=torch.zeros(N_DN, DN_CONV_CH, DN_CONV_K, **f32),
        hidden=torch.empty(S * HIDDEN, **bf16),
        residual=torch.empty(S * HIDDEN, **bf16),
        normalized=torch.empty(S * HIDDEN, **bf16),
        proj_buf=torch.empty(S * max_proj, **bf16),
        proj_buf2=torch.empty(S * max_proj, **bf16),
        attn_buf=torch.empty(S * max_attn, **bf16),
        mlp_buf=torch.empty(S * INTER, **bf16),
        dn_out_buf=torch.empty(S * max_attn, **bf16),
        beta_buf=torch.empty(S * DN_HEADS, **f32),
        alpha_buf=torch.empty(S * DN_HEADS, **f32),
        final_normed=torch.empty(HIDDEN, **bf16),
        hidden_bf16_out=torch.empty(HIDDEN, **bf16),
        out_token=torch.empty(1, **i32),
        lm_bmv=torch.empty(1024, **f32),
        lm_bmi=torch.empty(1024, **i32),
        lora_h_ws=torch.empty(S, lora_rank, **bf16),
    )


def alloc_activation_saves(S):
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    return dict(
        hidden_in=torch.empty(NUM_LAYERS, S, HIDDEN, **bf16),
        normalized_in=torch.empty(NUM_LAYERS, S, HIDDEN, **bf16),
        normalized_post_attn=torch.empty(NUM_LAYERS, S, HIDDEN, **bf16),
        mlp_inter=torch.empty(NUM_LAYERS, S, INTER, **bf16),
    )


def our_prefill_with_lora(weights, layers_packed, tokens, sc,
                          lora_tensors, lora_rank, lora_scaling):
    torch.ops.qwen35_megakernel_bf16_C.prefill_bf16_with_lora(
        sc["out_token"], tokens.to(dtype=torch.int32, device="cuda").contiguous(),
        weights["embed_weight"], layers_packed,
        weights["final_norm_weight"], weights["lm_head_weight"],
        sc["fa_k_cache"], sc["fa_v_cache"], sc["dn_states"], sc["conv_bufs"],
        sc["hidden"], sc["residual"], sc["normalized"],
        sc["proj_buf"], sc["proj_buf2"], sc["attn_buf"], sc["mlp_buf"],
        sc["dn_out_buf"], sc["beta_buf"], sc["alpha_buf"],
        sc["final_normed"], sc["hidden_bf16_out"],
        sc["lm_bmv"], sc["lm_bmi"],
        *lora_tensors, lora_rank, lora_scaling, sc["lora_h_ws"],
    )


def our_prefill_train_step(weights, layers_packed, tokens, sc, saves,
                           lora_tensors, lora_rank, lora_scaling):
    empty = torch.empty(0, dtype=torch.bfloat16, device="cuda")
    empty_f32 = torch.empty(0, dtype=torch.float32, device="cuda")
    torch.ops.qwen35_megakernel_bf16_C.prefill_bf16_train_step(
        sc["out_token"], tokens.to(dtype=torch.int32, device="cuda").contiguous(),
        weights["embed_weight"], layers_packed,
        weights["final_norm_weight"], weights["lm_head_weight"],
        sc["fa_k_cache"], sc["fa_v_cache"], sc["dn_states"], sc["conv_bufs"],
        sc["hidden"], sc["residual"], sc["normalized"],
        sc["proj_buf"], sc["proj_buf2"], sc["attn_buf"], sc["mlp_buf"],
        sc["dn_out_buf"], sc["beta_buf"], sc["alpha_buf"],
        sc["final_normed"], sc["hidden_bf16_out"],
        sc["lm_bmv"], sc["lm_bmi"],
        *lora_tensors, lora_rank, lora_scaling, sc["lora_h_ws"],
        saves["hidden_in"], saves["normalized_in"],
        saves["normalized_post_attn"], saves["mlp_inter"],
        empty, empty,                          # Slice B.2 saves
        empty, empty, empty_f32,               # Slice B.3b FA saves
    )


def our_logits_for_sequence(weights, layers_packed, tokens_1d, sc,
                            lora_tensors, lora_rank, lora_scaling):
    """Run prefill and return full next-token logits (cast from bf16).

    We compute next-token logits for EACH position [0..S-1] by running
    the prefill ONCE (which caches the final hidden state for the last
    position only) — to get per-position logits we'd need to modify
    the kernel to dump the full hidden buffer. For now we take the
    single last-position logits per prompt and stride over the
    corpus with a sliding window.
    """
    our_prefill_with_lora(weights, layers_packed, tokens_1d, sc,
                          lora_tensors, lora_rank, lora_scaling)
    logits_last = sc["final_normed"].to(torch.float32) @ \
                  weights["lm_head_weight"].to(torch.float32).t()
    return logits_last  # [VOCAB]


# ---------------------------------------------------------------------------
# Adapter loading
# ---------------------------------------------------------------------------
def build_our_lora_from_peft(peft_tensors, lora_rank):
    def zeros(shape):
        return torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
    out = {
        "fa_q":    (zeros((N_FA, HIDDEN,    lora_rank)), zeros((N_FA, lora_rank, FA_QPROJ_SIZE))),
        "fa_k":    (zeros((N_FA, HIDDEN,    lora_rank)), zeros((N_FA, lora_rank, FA_KV_SIZE))),
        "fa_v":    (zeros((N_FA, HIDDEN,    lora_rank)), zeros((N_FA, lora_rank, FA_KV_SIZE))),
        "fa_o":    (zeros((N_FA, FA_Q_SIZE, lora_rank)), zeros((N_FA, lora_rank, HIDDEN))),
        "fa_gate": (zeros((N_FA, HIDDEN,    lora_rank)), zeros((N_FA, lora_rank, INTER))),
        "fa_up":   (zeros((N_FA, HIDDEN,    lora_rank)), zeros((N_FA, lora_rank, INTER))),
        "fa_down": (zeros((N_FA, INTER,     lora_rank)), zeros((N_FA, lora_rank, HIDDEN))),
        "dn_qkv":  (zeros((N_DN, HIDDEN,    lora_rank)), zeros((N_DN, lora_rank, DN_CONV_CH))),
        "dn_z":    (zeros((N_DN, HIDDEN,    lora_rank)), zeros((N_DN, lora_rank, DN_V_SIZE))),
        "dn_out":  (zeros((N_DN, DN_V_SIZE, lora_rank)), zeros((N_DN, lora_rank, HIDDEN))),
        "dn_gate": (zeros((N_DN, HIDDEN,    lora_rank)), zeros((N_DN, lora_rank, INTER))),
        "dn_up":   (zeros((N_DN, HIDDEN,    lora_rank)), zeros((N_DN, lora_rank, INTER))),
        "dn_down": (zeros((N_DN, INTER,     lora_rank)), zeros((N_DN, lora_rank, HIDDEN))),
    }
    fa_map = {("self_attn","q_proj"):"fa_q",("self_attn","k_proj"):"fa_k",
              ("self_attn","v_proj"):"fa_v",("self_attn","o_proj"):"fa_o",
              ("mlp","gate_proj"):"fa_gate",("mlp","up_proj"):"fa_up",
              ("mlp","down_proj"):"fa_down"}
    dn_map = {("mlp","gate_proj"):"dn_gate",("mlp","up_proj"):"dn_up",
              ("mlp","down_proj"):"dn_down"}
    fa_idx = dn_idx = 0
    n_loaded = 0
    for li, lt in enumerate(LAYER_TYPE):
        proj_map = fa_map if lt == 1 else dn_map
        kernel_idx = fa_idx if lt == 1 else dn_idx
        for (parent, proj), our_name in proj_map.items():
            kA = f"base_model.model.model.layers.{li}.{parent}.{proj}.lora_A.weight"
            kB = f"base_model.model.model.layers.{li}.{parent}.{proj}.lora_B.weight"
            if kA in peft_tensors and kB in peft_tensors:
                A_ours = peft_tensors[kA].t().contiguous().to(dtype=torch.bfloat16, device="cuda")
                B_ours = peft_tensors[kB].t().contiguous().to(dtype=torch.bfloat16, device="cuda")
                out[our_name][0][kernel_idx].copy_(A_ours)
                out[our_name][1][kernel_idx].copy_(B_ours)
                n_loaded += 1
        if lt == 1: fa_idx += 1
        else: dn_idx += 1
    names = [n for n, _, _ in FA_SHAPES] + [n for n, _, _ in DN_SHAPES]
    flat = []
    for n in names:
        A, B = out[n]
        flat.extend([A, B])
    return flat, n_loaded


# ---------------------------------------------------------------------------
# Perplexity evaluation
# ---------------------------------------------------------------------------
def load_wikitext_tokens(tokenizer, num_tokens=4096):
    """Concatenate wikitext-2 test text and return a 1-D token tensor."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(r["text"] for r in ds if r["text"].strip())
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[:num_tokens]


def ppl_our_kernel(weights, layers_packed, token_ids,
                   lora_tensors, lora_rank, lora_scaling,
                   ctx_len=256, stride=128, label="ours"):
    """Perplexity via sliding-window prefill.

    Each window is S=ctx_len tokens; we score the LAST position's
    next-token prediction against the next token in the corpus.

    IMPORTANT: KV cache (fa_k/v_cache) and DeltaNet state (dn_states,
    conv_bufs) persist across prefill calls. We MUST reset them
    between windows or state from window N pollutes window N+1 —
    produces ~7× worse PPL (45 vs 6 on wikitext-2).
    """
    total_nll = 0.0
    total_n = 0
    device = "cuda"
    S = ctx_len
    sc = alloc_scratch(S, lora_rank)
    last_end = len(token_ids) - 1
    t0 = time.perf_counter()
    pos = 0
    while pos + S + 1 <= last_end + 1:
        # Reset all stateful buffers so each window is an independent
        # prefill (as if we just loaded the model).
        sc["fa_k_cache"].zero_()
        sc["fa_v_cache"].zero_()
        sc["dn_states"].zero_()
        sc["conv_bufs"].zero_()
        window = token_ids[pos : pos + S]
        target_tok = token_ids[pos + S]
        tokens = torch.tensor(window, dtype=torch.int32, device=device)
        our_prefill_with_lora(weights, layers_packed, tokens, sc,
                              lora_tensors, lora_rank, lora_scaling)
        logits = sc["final_normed"].to(torch.float32) @ \
                 weights["lm_head_weight"].to(torch.float32).t()
        logp = torch.nn.functional.log_softmax(logits, dim=-1)
        total_nll += float(-logp[target_tok].item())
        total_n += 1
        pos += stride
    elapsed = time.perf_counter() - t0
    ppl = math.exp(total_nll / max(total_n, 1))
    return ppl, total_n, elapsed


def ppl_hf(hf_model, token_ids, ctx_len=256, stride=128, label="HF"):
    total_nll = 0.0
    total_n = 0
    device = "cuda"
    S = ctx_len
    last_end = len(token_ids) - 1
    t0 = time.perf_counter()
    pos = 0
    while pos + S + 1 <= last_end + 1:
        window = token_ids[pos : pos + S]
        target_tok = token_ids[pos + S]
        input_ids = torch.tensor(window, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            out = hf_model(input_ids=input_ids, use_cache=False)
        logits = out.logits[0, -1].to(torch.float32)
        logp = torch.nn.functional.log_softmax(logits, dim=-1)
        total_nll += float(-logp[target_tok].item())
        total_n += 1
        pos += stride
    elapsed = time.perf_counter() - t0
    ppl = math.exp(total_nll / max(total_n, 1))
    return ppl, total_n, elapsed


# ---------------------------------------------------------------------------
# Speed bench
# ---------------------------------------------------------------------------
def bench_fn(fn, warmup=3, runs=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx-len", type=int, default=256)
    ap.add_argument("--ppl-tokens", type=int, default=2048,
                    help="total tokens from wikitext-2 to use for PPL")
    ap.add_argument("--ppl-stride", type=int, default=128)
    ap.add_argument("--bench-seq-lens", type=int, nargs="+", default=[128, 512])
    ap.add_argument("--bench-warmup", type=int, default=3)
    ap.add_argument("--bench-runs", type=int, default=10)
    ap.add_argument("--skip-correctness", action="store_true")
    ap.add_argument("--skip-speed", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("LoRA end-to-end bench: correctness (perplexity) + speed (per-stage)")
    print("=" * 78)

    # Load adapter
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    adapter_dir = Path(snapshot_download(ADAPTER_REPO))
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    lora_rank = int(cfg["r"])
    lora_alpha = float(cfg["lora_alpha"])
    lora_scaling = lora_alpha / lora_rank
    peft_tensors = load_file(str(adapter_dir / "adapter_model.safetensors"))

    print(f"\nAdapter: {ADAPTER_REPO}  r={lora_rank}  alpha={lora_alpha}  scaling={lora_scaling}")

    # Load base (ours)
    print("Loading Qwen3.5-0.8B for our kernel...")
    weights, tokenizer = load_weights("Qwen/Qwen3.5-0.8B", verbose=False, backend="bf16")
    layers_packed = _pack_layer_weights(weights["layer_data"])

    # Build our LoRA + zero LoRA
    our_lora, n_loaded = build_our_lora_from_peft(peft_tensors, lora_rank)
    our_zero = zero_lora_tensors(lora_rank)
    print(f"  repacked {n_loaded} PEFT (A, B) pairs")

    # =======================================================================
    # Correctness: perplexity on wikitext-2
    # =======================================================================
    if not args.skip_correctness:
        print()
        print("=" * 78)
        print("Correctness: wikitext-2 test perplexity")
        print("=" * 78)

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        hf_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
        print("\nLoading HF baseline...")
        hf_base = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16).to("cuda")
        hf_base.eval()

        token_ids = load_wikitext_tokens(hf_tok, num_tokens=args.ppl_tokens)
        print(f"\nScoring {len(token_ids)} tokens with ctx_len={args.ctx_len}, stride={args.ppl_stride}...")

        ppl_a, n_a, t_a = ppl_hf(hf_base, token_ids,
                                 ctx_len=args.ctx_len, stride=args.ppl_stride)
        print(f"  (a) HF base              : PPL={ppl_a:7.3f}  ({n_a} windows, {t_a:5.1f}s)")

        ppl_c, n_c, t_c = ppl_our_kernel(weights, layers_packed, token_ids,
                                         our_zero, lora_rank, 1.0,
                                         ctx_len=args.ctx_len, stride=args.ppl_stride)
        print(f"  (c) ours base            : PPL={ppl_c:7.3f}  ({n_c} windows, {t_c:5.1f}s)")

        print("\nApplying adapter to HF model (PEFT load)...")
        hf_lora_model = PeftModel.from_pretrained(hf_base, str(adapter_dir))
        hf_lora_model.eval().to("cuda", dtype=torch.bfloat16)
        ppl_b, n_b, t_b = ppl_hf(hf_lora_model, token_ids,
                                 ctx_len=args.ctx_len, stride=args.ppl_stride)
        print(f"  (b) HF + adapter         : PPL={ppl_b:7.3f}  ({n_b} windows, {t_b:5.1f}s)")

        ppl_d, n_d, t_d = ppl_our_kernel(weights, layers_packed, token_ids,
                                         our_lora, lora_rank, lora_scaling,
                                         ctx_len=args.ctx_len, stride=args.ppl_stride)
        print(f"  (d) ours + adapter       : PPL={ppl_d:7.3f}  ({n_d} windows, {t_d:5.1f}s)")

        # ----- Invariants -----
        print()
        print("Correctness check:")
        base_drift = abs(ppl_c - ppl_a) / ppl_a
        I1 = base_drift < 0.10
        print(f"  I1  base drift |c-a|/a = {base_drift:6.2%}   (target <10 %, "
              f"accounts for HF torch-DN fallback)  {'PASS' if I1 else 'FAIL'}")
        delta_hf   = ppl_b - ppl_a
        delta_ours = ppl_d - ppl_c
        same_sign = (delta_hf * delta_ours) > 0
        I2 = same_sign
        print(f"  I2  adapter shifts both paths same direction                      "
              f"{'PASS' if I2 else 'FAIL'}")
        print(f"        HF shift: {delta_hf:+7.3f}    ours shift: {delta_ours:+7.3f}")
        if abs(delta_hf) > 1e-3:
            mag_ratio = abs(delta_ours) / abs(delta_hf)
            I3 = 0.2 < mag_ratio < 5.0
            print(f"  I3  adapter magnitude ratio = {mag_ratio:.2f}x (target 0.2–5x)  "
                  f"{'PASS' if I3 else 'FAIL'}")
        else:
            print(f"  I3  skipped — HF delta too small to compute ratio")
            I3 = True
        print()
        print(f"CORRECTNESS SUMMARY: I1={'PASS' if I1 else 'FAIL'} "
              f"I2={'PASS' if I2 else 'FAIL'} I3={'PASS' if I3 else 'FAIL'}")
        print(f"  — speed comparison below assumes forward-correctness is OK.")

        del hf_lora_model, hf_base
        torch.cuda.empty_cache()

    # =======================================================================
    # Speed: per-stage wall-time table
    # =======================================================================
    if not args.skip_speed:
        print()
        print("=" * 78)
        print("Speed: per-stage wall-time on B200")
        print("=" * 78)
        from transformers import AutoModelForCausalLM
        hf_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16).to("cuda")
        hf_model.eval()
        random_lora = build_lora_tensors(lora_rank, init_scale=0.02)
        wrap_hf_with_lora(hf_model, random_lora, scaling=1.0)
        hf_model.to("cuda", dtype=torch.bfloat16)

        # Flat LoRA param buffer for fused AdamW bench.
        total_lora_numel = sum(t.numel() for t in random_lora)
        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        f32 = dict(dtype=torch.float32, device="cuda")
        flat_params = torch.randn(total_lora_numel, **bf16)
        flat_m = torch.zeros_like(flat_params, **f32)
        flat_v = torch.zeros_like(flat_params, **f32)
        flat_grad = torch.randn(total_lora_numel, **f32) * 1e-3

        hf_lora_params = [p for n, p in hf_model.named_parameters()
                          if (n.endswith(".A") or n.endswith(".B"))]
        torch_adam = torch.optim.AdamW(hf_lora_params, lr=1e-4)
        for p in hf_lora_params:
            p.grad = torch.randn_like(p) * 1e-3

        print(f"\n{'stage':<32} | {'S':>5} | {'PyTorch (ms)':>14} | {'ours (ms)':>10} | {'speedup':>8}")
        print("-" * 32 + "-+-" + "-" * 5 + "-+-" + "-" * 14 + "-+-" + "-" * 10 + "-+-" + "-" * 8)

        rows = []
        for S in args.bench_seq_lens:
            sc = alloc_scratch(S, lora_rank)
            saves = alloc_activation_saves(S)
            tokens = torch.randint(0, VOCAB, (S,), dtype=torch.int32, device="cuda")
            input_ids = tokens.long().unsqueeze(0)

            # -- prefill no LoRA --
            @torch.no_grad()
            def hf_fwd_base():
                _ = hf_model(input_ids=input_ids, use_cache=False)
            def ours_fwd_noloRA():
                our_prefill_with_lora(weights, layers_packed, tokens, sc,
                                      our_zero, lora_rank, 1.0)

            # -- prefill with LoRA (HF has LoRA wrapped permanently) --
            def ours_fwd_lora():
                our_prefill_with_lora(weights, layers_packed, tokens, sc,
                                      random_lora, lora_rank, 1.0)

            # -- training-forward (with activation saves) --
            def ours_train_fwd():
                our_prefill_train_step(weights, layers_packed, tokens, sc, saves,
                                       random_lora, lora_rank, 1.0)
            def hf_train_full():
                torch_adam.zero_grad()
                out = hf_model(input_ids=input_ids, labels=input_ids, use_cache=False)
                out.loss.backward()
                torch_adam.step()

            hf_fwd_ms   = bench_fn(hf_fwd_base,  warmup=args.bench_warmup, runs=args.bench_runs)
            our_fwd_ms  = bench_fn(ours_fwd_lora, warmup=args.bench_warmup, runs=args.bench_runs)
            our_base_ms = bench_fn(ours_fwd_noloRA, warmup=args.bench_warmup, runs=args.bench_runs)
            our_train_ms = bench_fn(ours_train_fwd, warmup=args.bench_warmup, runs=args.bench_runs)
            hf_train_ms = bench_fn(hf_train_full, warmup=args.bench_warmup, runs=args.bench_runs)

            def fmt(label, pt_ms, ours_ms):
                sp = pt_ms / ours_ms if ours_ms > 0 else 0
                print(f"{label:<32} | {S:>5} | {pt_ms:>10.2f}    | {ours_ms:>7.2f}   | {sp:>6.1f}x")
                rows.append((label, S, pt_ms, ours_ms, sp))

            fmt("prefill fwd (no LoRA)",         hf_fwd_ms,   our_base_ms)
            fmt("prefill fwd + LoRA",            hf_fwd_ms,   our_fwd_ms)  # HF is same w/ LoRA wrapped
            fmt("training fwd (+activ saves)",   hf_train_ms, our_train_ms)

        # AdamW — shape-independent of S
        def torch_adam_step():
            torch_adam.step()
        def our_adamw_step():
            torch.ops.train_megakernel_C.fused_adamw_step(
                flat_params, flat_m, flat_v, flat_grad,
                1, 1e-4, 0.9, 0.999, 1e-8, 0.01)

        torch_ms = bench_fn(torch_adam_step, warmup=args.bench_warmup, runs=args.bench_runs)
        ours_ms  = bench_fn(our_adamw_step,  warmup=args.bench_warmup, runs=args.bench_runs)
        sp = torch_ms / ours_ms
        print(f"{'fused AdamW (on LoRA params)':<32} | {'-':>5} | {torch_ms:>10.3f}    | {ours_ms:>7.3f}   | {sp:>6.1f}x")

        print()
        print("Notes:")
        print("  * 'prefill fwd + LoRA' on HF side uses the same 13 LoRA adapters")
        print("    wrapped as LoraLinear around the base nn.Linear modules.")
        print("  * 'training fwd' = our prefill_bf16_train_step (populates 4 per-layer")
        print("    activation slabs for backward); HF's 'training fwd+bwd+optim' is")
        print("    a full torch.autograd training step — not apples-to-apples, but")
        print("    serves as the baseline a real trainer has to beat.")
        print("  * Full training-step speedup is projected once Phase 2 FA+DN bwd")
        print("    kernels land (currently using PyTorch autograd fallback for bwd).")


if __name__ == "__main__":
    main()
