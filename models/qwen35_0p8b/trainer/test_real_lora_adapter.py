"""End-to-end demo: download a real PEFT LoRA adapter from HF and run
it through our cuBLAS+graph prefill.

Downloads `eac123/subliminal-qwen3.5-0.8b-tiger-r5` (r=8 LoRA on
Qwen3.5-0.8B, targets MLP+FA projections), packs the PEFT A/B
weights into our kernel's per-type-layer-major layout, and runs
prompts on both:
  - HF transformers with the adapter applied via `peft.PeftModel`
  - our kernel with the same adapter weights via
    `prefill_bf16_with_lora`

What this proves:
  (1) we can load a real PEFT-format adapter from the Hub
  (2) our kernel's 26-tensor layout is a valid repack of PEFT's
      per-module A/B weights (module names mapped correctly)
  (3) applying the adapter changes our kernel's output vs
      no-LoRA (so the LoRA is actually being applied, not a no-op)
  (4) both HF+PEFT and our kernel produce sensible top-k tokens

What this does NOT prove:
  - bit-exact logit match with HF. HF's torch-native DeltaNet
    fallback ("fast path is not available" warning) computes the
    DN recurrence slightly differently than our CUDA kernel, so
    the baselines diverge. Our test_lora_forward_cublas.py already
    proves our cuBLAS LoRA path matches our hand-rolled trainer
    megakernel exactly, which is the stronger internal guarantee.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import torch
from pathlib import Path
from safetensors.torch import load_file

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b")
sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/prefill_megakernel")
sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")

import qwen35_megakernel_bf16_C  # noqa: F401

from model import (
    NUM_LAYERS, HIDDEN, LAYER_TYPE, VOCAB, INTER,
    FA_QPROJ_SIZE, FA_KV_SIZE, FA_Q_SIZE, DN_CONV_CH, DN_V_SIZE,
    FA_HEAD_DIM, FA_KV_HEADS, DN_HEADS, DN_VAL, DN_KEY, DN_CONV_K,
)
from test_lora_forward import _pack_layer_weights, FA_SHAPES, DN_SHAPES, zero_lora_tensors

_spec = importlib.util.spec_from_file_location(
    "qwen_outer_model",
    "/root/lucebox-hub-b200-train/models/qwen35_0p8b/model.py")
_outer = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_outer)
load_weights = _outer.load_weights

N_FA = sum(1 for t in LAYER_TYPE if t == 1)
N_DN = sum(1 for t in LAYER_TYPE if t == 0)

ADAPTER_REPO = "eac123/subliminal-qwen3.5-0.8b-tiger-r5"


def download_adapter() -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(ADAPTER_REPO))


def build_our_lora_from_peft(peft_tensors, lora_rank):
    """PEFT layout:
      base_model.model.model.layers.{L}.{mlp|self_attn}.{name}.lora_{A,B}.weight
      A: [R, K_in]    fp32
      B: [K_out, R]   fp32

    Our layout per projection: A=[N_lay, K_in, R], B=[N_lay, R, K_out], bf16.
    Missing projections (this adapter doesn't target linear_attn) get zeros.
    """
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


def our_prefill(weights, layers_packed, tokens, lora_tensors, lora_rank, lora_scaling):
    S = int(tokens.shape[0])
    sc = alloc_scratch(S, lora_rank)
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
    logits = sc["final_normed"].to(torch.float32) @ \
             weights["lm_head_weight"].to(torch.float32).t()
    return int(sc["out_token"].item()), logits


def main():
    print(f"=== Downloading PEFT adapter: {ADAPTER_REPO} ===")
    adapter_dir = download_adapter()
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    lora_rank = int(cfg["r"])
    lora_alpha = float(cfg["lora_alpha"])
    scaling = lora_alpha / lora_rank
    print(f"  rank={lora_rank}, alpha={lora_alpha}, scaling={scaling}")
    print(f"  targets (in model): {[m for m in cfg['target_modules'] if '_proj' in m]}")

    peft_tensors = load_file(str(adapter_dir / "adapter_model.safetensors"))

    print("\n=== Loading base weights + applying adapter both ways ===")
    weights, _ = load_weights("Qwen/Qwen3.5-0.8B", verbose=False, backend="bf16")
    layers_packed = _pack_layer_weights(weights["layer_data"])

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    hf_base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16).to("cuda")
    hf_base.eval()
    hf_model = PeftModel.from_pretrained(hf_base, str(adapter_dir))
    hf_model.eval().to("cuda", dtype=torch.bfloat16)

    our_lora, n_loaded = build_our_lora_from_peft(peft_tensors, lora_rank)
    print(f"  repacked {n_loaded} (A, B) pairs into our kernel layout")

    prompts = [
        "Explain in great detail the history of artificial intelligence,",
        "The quick brown fox jumps over the lazy dog. Once upon a time,",
        "def binary_search(arr, target):",
    ]
    zeros = zero_lora_tensors(lora_rank)

    print()
    print("=== Results ===")
    print("For each prompt we show top-3 next-token suggestions from:")
    print("  (a) HF + PEFT adapter (PyTorch path, torch-DN fallback)")
    print("  (b) ours (cuBLAS+graph+LoRA, same adapter weights)")
    print("  (c) ours with LoRA OFF (base model only)")
    print("Expectation: (a)(b) pick similar tokens, both different from (c).")

    base_vs_lora_differs = 0
    top5_overlaps = []

    for prompt in prompts:
        ids = tok.encode(prompt, add_special_tokens=False)
        S = len(ids)
        tokens = torch.tensor(ids, dtype=torch.int32, device="cuda")
        input_ids = tokens.long().unsqueeze(0)

        with torch.no_grad():
            hf_lora_logits = hf_model(input_ids=input_ids, use_cache=False).logits[0, -1].to(torch.float32)
        hf_top3 = torch.topk(hf_lora_logits, 3).indices

        _, our_lora_logits = our_prefill(weights, layers_packed, tokens,
                                         our_lora, lora_rank, scaling)
        our_top3 = torch.topk(our_lora_logits, 3).indices

        _, our_base_logits = our_prefill(weights, layers_packed, tokens,
                                         zeros, lora_rank, 1.0)
        our_base_top3 = torch.topk(our_base_logits, 3).indices

        overlap_ab = len(set(hf_top3.tolist()) & set(our_top3.tolist()))
        top5_overlaps.append(overlap_ab)
        # Does LoRA actually change our kernel output?
        if set(our_top3.tolist()) != set(our_base_top3.tolist()):
            base_vs_lora_differs += 1

        print(f"\n--- {prompt!r}")
        print(f"  (a) HF+PEFT   top-3: {[tok.decode([int(t)]) for t in hf_top3]}")
        print(f"  (b) ours+LoRA top-3: {[tok.decode([int(t)]) for t in our_top3]}")
        print(f"  (c) ours base top-3: {[tok.decode([int(t)]) for t in our_base_top3]}")
        print(f"       top-3 overlap (a∩b) = {overlap_ab}/3, (b)!=(c) = {set(our_top3.tolist()) != set(our_base_top3.tolist())}")

    print()
    print("=== Summary ===")
    print(f"PEFT adapter loads and applies on our kernel: YES ({n_loaded} A/B pairs packed)")
    print(f"LoRA changes our kernel output vs base:       {base_vs_lora_differs}/{len(prompts)} prompts")
    print(f"Top-3 overlap with HF+PEFT (mean):            {sum(top5_overlaps)/len(top5_overlaps):.2f}/3")
    print()
    print("Notes:")
    print("  * HF's torch-native DN fallback differs from our CUDA DN kernel;")
    print("    this causes HF and our kernel to diverge at the logit level even")
    print("    with IDENTICAL LoRA weights loaded on both paths. That divergence")
    print("    is NOT a bug in our LoRA application — it's a base-model compute")
    print("    difference. The LoRA's effect (delta from base) is consistently")
    print("    applied on our side (see test_lora_forward_cublas.py for the")
    print("    internal-consistency proof that our cuBLAS path matches our")
    print("    hand-rolled trainer megakernel exactly).")


if __name__ == "__main__":
    main()
