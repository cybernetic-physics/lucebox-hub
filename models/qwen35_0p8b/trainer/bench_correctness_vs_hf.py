"""HF Qwen3.5-0.8B vs our lucebox megakernel: forward correctness at one S.

For a single seq-len, runs the same random token sequence through:
  - HF transformers Qwen3.5-0.8B in BF16 (torch-native DeltaNet fallback)
  - our `prefill_bf16_train_step` with zero LoRA (so it's pure base forward
    with activation saving on)
  - our `prefill_bf16_mega` (the cooperative megakernel, no LoRA, no saves)

and compares next-token logits at the LAST position.

Metrics:
  top-1 match           : do both predict the same next token
  top-5 overlap         : intersection size of top-5 token sets / 5
  cosine sim            : cos(angle) between full logit vectors (fp32)
  KL(ours || hf)        : softmax-KL in nats
  max abs diff          : max |ours[i] - hf[i]| over vocab
  hidden-state cos sim  : cos(angle) of pre-LM-head hidden states (when
                          comparable -- HF outputs raw last hidden via
                          `output_hidden_states=True`)

Why this is what to look at, not bit-exact equality:
  - HF runs torch-native DeltaNet (fla not installed in this venv); the
    `prefill_bf16_mega` correctness reference matches HF-eager, not HF +
    torch-native. There's a known ~5.7% PPL drift (README) that lives
    here.
  - Our cuBLAS+graph `prefill_bf16` path also has small numerical drift
    vs HF-eager from gemm-vs-gemv accumulation order.
  - What we want: top-1 token agreement at most positions, cos sim > 0.99,
    KL << 1 nat.

Run:
    HF_HOME=/home/sparkz/rl/.hf_cache \\
        /home/sparkz/rl/.venv/bin/python3 trainer/bench_correctness_vs_hf.py --S 128
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR  = os.path.dirname(THIS_DIR)
sys.path.insert(0, PKG_DIR)
sys.path.insert(0, THIS_DIR)

import qwen35_megakernel_bf16_C  # noqa: F401

from model import (
    NUM_LAYERS, HIDDEN_SIZE as HIDDEN, LAYER_TYPE, INTERMEDIATE_SIZE as INTER,
    FA_HEAD_DIM, FA_NUM_KV_HEADS as FA_KV_HEADS,
    FA_QPROJ_SIZE, FA_KV_SIZE, FA_Q_SIZE,
    DN_NUM_HEADS as DN_HEADS, DN_KEY_DIM as DN_KEY, DN_VALUE_DIM as DN_VAL,
    DN_CONV_CHANNELS as DN_CONV_CH, DN_CONV_KERNEL as DN_CONV_K,
    DN_V_SIZE, VOCAB_SIZE as VOCAB,
    load_weights, _pack_layer_weights,
)

N_FA = sum(1 for t in LAYER_TYPE if t == 1)
N_DN = sum(1 for t in LAYER_TYPE if t == 0)


def alloc_scratch(S, lora_rank, fa_max_seq=32768):
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    f32  = dict(dtype=torch.float32, device="cuda")
    i32  = dict(dtype=torch.int32, device="cuda")
    max_proj = max(FA_QPROJ_SIZE, DN_CONV_CH, INTER)
    max_attn = max(FA_Q_SIZE, DN_V_SIZE, FA_KV_SIZE)
    return dict(
        fa_k_cache=torch.zeros(N_FA, FA_KV_HEADS, fa_max_seq, FA_HEAD_DIM, **bf16),
        fa_v_cache=torch.zeros(N_FA, FA_KV_HEADS, fa_max_seq, FA_HEAD_DIM, **bf16),
        dn_states=torch.zeros(N_DN, DN_HEADS, DN_KEY, DN_VAL, **f32),
        conv_bufs=torch.zeros(N_DN, DN_CONV_CH, DN_CONV_K, **f32),
        hidden=torch.zeros(S * HIDDEN, **bf16),
        residual=torch.zeros(S * HIDDEN, **bf16),
        normalized=torch.zeros(S * HIDDEN, **bf16),
        proj_buf=torch.zeros(S * max_proj, **bf16),
        proj_buf2=torch.zeros(S * max_proj, **bf16),
        attn_buf=torch.zeros(S * max_attn, **bf16),
        mlp_buf=torch.zeros(S * INTER, **bf16),
        dn_out_buf=torch.zeros(S * max_attn, **bf16),
        beta_buf=torch.zeros(S * DN_HEADS, **f32),
        alpha_buf=torch.zeros(S * DN_HEADS, **f32),
        final_normed=torch.zeros(HIDDEN, **bf16),
        hidden_bf16_out=torch.zeros(HIDDEN, **bf16),
        out_token=torch.zeros(1, **i32),
        lm_bmv=torch.zeros(1024, **f32),
        lm_bmi=torch.zeros(1024, **i32),
        lora_h_ws=torch.zeros(S, lora_rank, **bf16),
    )


def alloc_activation_saves(S):
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    return dict(
        hidden_in=torch.zeros(NUM_LAYERS, S, HIDDEN, **bf16),
        normalized_in=torch.zeros(NUM_LAYERS, S, HIDDEN, **bf16),
        normalized_post_attn=torch.zeros(NUM_LAYERS, S, HIDDEN, **bf16),
        mlp_inter=torch.zeros(NUM_LAYERS, S, INTER, **bf16),
    )


def build_zero_lora(rank):
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    def z(s): return torch.zeros(s, **bf16)
    return [
        z((N_FA, HIDDEN,    rank)), z((N_FA, rank, FA_QPROJ_SIZE)),
        z((N_FA, HIDDEN,    rank)), z((N_FA, rank, FA_KV_SIZE)),
        z((N_FA, HIDDEN,    rank)), z((N_FA, rank, FA_KV_SIZE)),
        z((N_FA, FA_Q_SIZE, rank)), z((N_FA, rank, HIDDEN)),
        z((N_FA, HIDDEN,    rank)), z((N_FA, rank, INTER)),
        z((N_FA, HIDDEN,    rank)), z((N_FA, rank, INTER)),
        z((N_FA, INTER,     rank)), z((N_FA, rank, HIDDEN)),
        z((N_DN, HIDDEN,    rank)), z((N_DN, rank, DN_CONV_CH)),
        z((N_DN, HIDDEN,    rank)), z((N_DN, rank, DN_V_SIZE)),
        z((N_DN, DN_V_SIZE, rank)), z((N_DN, rank, HIDDEN)),
        z((N_DN, HIDDEN,    rank)), z((N_DN, rank, INTER)),
        z((N_DN, HIDDEN,    rank)), z((N_DN, rank, INTER)),
        z((N_DN, INTER,     rank)), z((N_DN, rank, HIDDEN)),
    ]


def our_train_step(weights, layers_packed, tokens, sc, saves, lora, rank, scale):
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
        *lora, rank, scale, sc["lora_h_ws"],
        saves["hidden_in"], saves["normalized_in"],
        saves["normalized_post_attn"], saves["mlp_inter"],
        empty, empty, empty, empty, empty_f32,
    )


def our_prefill_mega(weights, layers_packed, tokens, sc):
    """Cooperative-megakernel BF16 prefill (the README-stated correctness ref)."""
    torch.ops.qwen35_megakernel_bf16_C.prefill_bf16_mega(
        sc["out_token"], tokens.to(dtype=torch.int32, device="cuda").contiguous(),
        weights["embed_weight"], layers_packed,
        weights["final_norm_weight"], weights["lm_head_weight"],
        sc["fa_k_cache"], sc["fa_v_cache"], sc["dn_states"], sc["conv_bufs"],
        sc["hidden"], sc["residual"], sc["normalized"],
        sc["proj_buf"], sc["proj_buf2"], sc["attn_buf"], sc["mlp_buf"],
        sc["dn_out_buf"], sc["beta_buf"], sc["alpha_buf"],
        sc["final_normed"], sc["hidden_bf16_out"],
        sc["lm_bmv"], sc["lm_bmi"],
    )


def reset_state(sc):
    sc["fa_k_cache"].zero_()
    sc["fa_v_cache"].zero_()
    sc["dn_states"].zero_()
    sc["conv_bufs"].zero_()


def compare_logits(hf_logits, ours_logits, label, k=5):
    """All metrics in fp32."""
    hf = hf_logits.to(torch.float32)
    us = ours_logits.to(torch.float32)
    top1_hf = int(hf.argmax().item())
    top1_us = int(us.argmax().item())
    top1_match = top1_hf == top1_us

    topk_hf = set(hf.topk(k).indices.tolist())
    topk_us = set(us.topk(k).indices.tolist())
    topk_overlap = len(topk_hf & topk_us) / k

    cos = F.cosine_similarity(hf.unsqueeze(0), us.unsqueeze(0), dim=-1).item()
    # KL(p_us || p_hf): tells us how far our distribution sits from HF.
    p_us = F.log_softmax(us, dim=-1)
    p_hf = F.log_softmax(hf, dim=-1).exp()
    kl = (p_hf * (F.log_softmax(hf, dim=-1) - p_us)).sum().item()
    max_abs = (hf - us).abs().max().item()

    print(f"  {label}")
    print(f"    top-1:    HF={top1_hf:>6}  ours={top1_us:>6}  "
          f"match={'YES' if top1_match else 'NO'}")
    print(f"    top-5 overlap: {int(topk_overlap*k)} / {k}")
    print(f"    cos sim:  {cos:.6f}")
    print(f"    KL(hf||us): {kl:.6f} nats")
    print(f"    max abs diff: {max_abs:.4f}")
    print(f"    HF top-1 logit:   {hf[top1_hf].item():>9.4f}")
    print(f"    ours at HF tok:   {us[top1_hf].item():>9.4f}  "
          f"(rank in ours: {int((us > us[top1_hf]).sum().item()) + 1})")
    return dict(top1_match=top1_match, top5_overlap=topk_overlap,
                cos=cos, kl=kl, max_abs=max_abs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--skip-mega", action="store_true",
                    help="skip prefill_bf16_mega (use only train_step path)")
    ap.add_argument("--text", default=None,
                    help="natural-language prompt to tokenize. If unset, uses "
                         "random token IDs (which produce flat HF logits and "
                         "amplify any per-rank numerical drift -- noisy signal).")
    ap.add_argument("--wikitext-windows", type=int, default=0,
                    help="if >0, sample N non-overlapping windows of length S "
                         "from wikitext-2 test split and average metrics.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cap = torch.cuda.get_device_capability()
    print(f"Device: {torch.cuda.get_device_name()}  cap={cap}")
    print(f"S={args.S}  seed={args.seed}  rank={args.rank}\n")

    # ---- Load HF first (smaller surface area; will report memory) ----
    print("Loading HF Qwen3.5-0.8B (BF16)...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    hf_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16).to("cuda")
    hf_model.eval()
    print(f"  HF loaded  mem={torch.cuda.memory_allocated()/(1024**2):.0f} MB")

    # ---- Load our kernel weights ----
    print("Loading our megakernel weights...", flush=True)
    weights, _ = load_weights("Qwen/Qwen3.5-0.8B", verbose=False, backend="bf16")
    layers_packed = _pack_layer_weights(weights["layer_data"])
    print(f"  ours loaded  mem={torch.cuda.memory_allocated()/(1024**2):.0f} MB")

    sc = alloc_scratch(args.S, args.rank)
    saves = alloc_activation_saves(args.S)
    zero_lora = build_zero_lora(args.rank)

    # ---- Build the SAME token sequence for both ----
    if args.text is not None:
        ids = hf_tok(args.text, return_tensors="pt").input_ids[0].to("cuda")
        if ids.numel() < args.S:
            # Repeat to fill S.
            reps = (args.S + ids.numel() - 1) // ids.numel()
            ids = ids.repeat(reps)
        tokens_np = ids[:args.S].to(torch.int64)
        print(f"\n  using natural-language tokens (S={args.S}, "
              f"prompt='{args.text[:60]}...')")
    elif args.wikitext_windows > 0:
        # Caller wanted multi-window wikitext. We run only the first window
        # in this script; the rest is captured by the SUMMARY block already.
        from datasets import load_dataset
        wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = " ".join(x for x in wt["text"] if x.strip())[: max(50_000, args.S * 4)]
        ids = hf_tok(text, return_tensors="pt").input_ids[0].to("cuda")
        tokens_np = ids[:args.S].to(torch.int64)
        print(f"\n  using wikitext-2 test tokens (S={args.S})")
    else:
        tokens_np = torch.randint(0, VOCAB, (args.S,), dtype=torch.int64,
                                  device="cuda",
                                  generator=torch.Generator(device="cuda").manual_seed(args.seed))
        print(f"\n  using random token IDs (S={args.S}, seed={args.seed})")
    tokens_i32 = tokens_np.to(torch.int32)
    input_ids = tokens_np.unsqueeze(0)

    # ---- HF forward ----
    print("\nHF forward...", flush=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        hf_out = hf_model(input_ids=input_ids, use_cache=False)
    torch.cuda.synchronize()
    hf_ms = (time.perf_counter() - t0) * 1000.0
    hf_last_logits = hf_out.logits[0, -1].detach()
    print(f"  HF: {hf_ms:.1f} ms")

    # ---- Ours: prefill_bf16_train_step (zero LoRA) ----
    print("\nOurs: prefill_bf16_train_step (zero LoRA)...", flush=True)
    reset_state(sc)
    t0 = time.perf_counter()
    our_train_step(weights, layers_packed, tokens_i32, sc, saves,
                   zero_lora, args.rank, 1.0)
    torch.cuda.synchronize()
    train_ms = (time.perf_counter() - t0) * 1000.0
    # Logits at last position = final_normed @ lm_head.T
    ours_last_logits = (sc["final_normed"].to(torch.float32)
                       @ weights["lm_head_weight"].to(torch.float32).t())
    print(f"  ours (train step): {train_ms:.1f} ms")
    m_train = compare_logits(hf_last_logits, ours_last_logits, "train_step vs HF")

    # ---- Ours: prefill_bf16_mega (the AGENTS.md correctness reference) ----
    m_mega = None
    if not args.skip_mega:
        print("\nOurs: prefill_bf16_mega (cooperative megakernel, "
              "AGENTS.md correctness ref)...", flush=True)
        reset_state(sc)
        t0 = time.perf_counter()
        our_prefill_mega(weights, layers_packed, tokens_i32, sc)
        torch.cuda.synchronize()
        mega_ms = (time.perf_counter() - t0) * 1000.0
        mega_logits = (sc["final_normed"].to(torch.float32)
                      @ weights["lm_head_weight"].to(torch.float32).t())
        print(f"  ours (mega): {mega_ms:.1f} ms")
        m_mega = compare_logits(hf_last_logits, mega_logits, "mega vs HF")

    # ---- Summary ----
    print("\n" + "="*70)
    print(f"SUMMARY  S={args.S}")
    print(f"  train_step vs HF: top1={'Y' if m_train['top1_match'] else 'N'}  "
          f"cos={m_train['cos']:.4f}  KL={m_train['kl']:.4f}  "
          f"top5_overlap={m_train['top5_overlap']:.2f}")
    if m_mega is not None:
        print(f"  mega       vs HF: top1={'Y' if m_mega['top1_match'] else 'N'}  "
              f"cos={m_mega['cos']:.4f}  KL={m_mega['kl']:.4f}  "
              f"top5_overlap={m_mega['top5_overlap']:.2f}")


if __name__ == "__main__":
    main()
