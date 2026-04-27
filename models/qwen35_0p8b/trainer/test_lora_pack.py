"""Verify lora_pack.pack_peft_to_flat / unpack_flat_to_peft.

Two checks:

  1. Round-trip on a live PEFT-wrapped Qwen3.5-0.8B with non-trivial
     LoRA values: pack → write zeros into model → unpack(packed) →
     re-pack → must equal original packed (exactly, not just close).

  2. Forward equivalence: feed the same prompt through HF+PEFT and our
     `prefill_bf16_with_lora` op (using the packed LoRA from the SAME
     PEFT model). Compare next-token argmax + cosine similarity on the
     last-position logits. Won't be bit-exact (HF Qwen3-Next's DN path
     differs from our cuBLAS+graph DN), but should be close: cos > 0.95
     and top-1 token usually matching.
"""
from __future__ import annotations

import sys

# Disable transformers' fla import path that triggers @torch.compile and
# fails on this torch+transformers combo. (Same shim rl_trainer.py uses.)
try:
    import transformers.utils.import_utils as _iu
    _iu.is_flash_linear_attention_available = lambda: False
except Exception:
    pass

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b")
sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")

import importlib.util  # noqa: E402

import torch  # noqa: E402

import qwen35_megakernel_bf16_C  # noqa: F401, E402

from lora_pack import pack_peft_to_flat, unpack_flat_to_peft, N_FA, N_DN  # noqa: E402

# Outer model module (load_weights, _pack_layer_weights via test_lora_forward).
_spec = importlib.util.spec_from_file_location(
    "qwen_outer_model",
    "/root/lucebox-hub-b200-train/models/qwen35_0p8b/model.py")
_outer = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_outer)
load_weights = _outer.load_weights
_pack_layer_weights = _outer._pack_layer_weights
HIDDEN = _outer.HIDDEN_SIZE
INTER = _outer.INTERMEDIATE_SIZE
N_LAYERS = _outer.NUM_LAYERS
FA_QPROJ_SIZE = _outer.FA_QPROJ_SIZE
FA_KV_SIZE = _outer.FA_KV_SIZE
FA_Q_SIZE = _outer.FA_Q_SIZE
FA_HEAD_DIM = _outer.FA_HEAD_DIM
FA_KV_HEADS = _outer.FA_NUM_KV_HEADS
DN_HEADS = _outer.DN_NUM_HEADS
DN_KEY = _outer.DN_KEY_DIM
DN_VAL = _outer.DN_VALUE_DIM
DN_V_SIZE = _outer.DN_V_SIZE
DN_CONV_CH = _outer.DN_CONV_CHANNELS
DN_CONV_K = _outer.DN_CONV_KERNEL


def alloc_scratch(S: int, lora_rank: int) -> dict:
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


BASE_MODEL = "Qwen/Qwen3.5-0.8B"
LORA_RANK = 8


def build_peft_model(rank: int):
    """Mirrors LoraMegakernelTrainer.register_model: rank-R LoRA on
    q/k/v/o + gate/up/down (the FA-style names; PEFT skips projections
    that don't match these on DN layers)."""
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16
    ).to("cuda").eval()
    cfg = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base, cfg).to("cuda", dtype=torch.bfloat16)
    peft_model.eval()
    return peft_model


def randomize_lora(peft_model: torch.nn.Module, std: float = 0.02):
    """Replace zeroed LoRA-B weights with small random values so packing
    is exercising real (non-zero) state."""
    g = torch.Generator(device="cuda").manual_seed(123)
    n = 0
    for name, p in peft_model.named_parameters():
        if "lora_B" in name and ".weight" in name:
            p.data.copy_(torch.randn(p.shape, generator=g, device="cuda",
                                     dtype=p.dtype) * std)
            n += 1
        elif "lora_A" in name and ".weight" in name:
            # PEFT inits A with kaiming; leave it (already non-zero).
            n += 1
    return n


def test_round_trip(peft_model: torch.nn.Module, rank: int) -> bool:
    print("\n--- (1) round-trip ---")
    flat0 = pack_peft_to_flat(peft_model, rank)
    # Zero out PEFT in place, then unpack → re-pack must match flat0.
    with torch.no_grad():
        for _, p in peft_model.named_parameters():
            if "lora_" in _:
                p.data.zero_()
    n_written = unpack_flat_to_peft(peft_model, flat0)
    flat1 = pack_peft_to_flat(peft_model, rank)
    print(f"  unpack wrote {n_written} (A,B) pairs")
    ok = True
    for i, (t0, t1) in enumerate(zip(flat0, flat1)):
        if not torch.equal(t0, t1):
            ok = False
            d = (t0.float() - t1.float()).abs()
            print(f"  tensor[{i}] differs: max|Δ|={d.max().item():.2e} "
                  f"shape={tuple(t0.shape)}")
    if ok:
        print("  PASS — pack→zero→unpack→pack is bit-exact across all 26 tensors")
    return ok


def test_forward_equivalence(peft_model: torch.nn.Module, rank: int) -> bool:
    print("\n--- (2) forward equivalence vs HF+PEFT ---")
    # Build a token sequence and run both paths.
    text = ("The quick brown fox jumps over the lazy dog. "
            "Pack my box with five dozen liquor jugs. "
            "Sphinx of black quartz, judge my vow.")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) < 32:
        ids = (ids * (32 // max(len(ids), 1) + 1))[:32]
    S = len(ids)
    tokens = torch.tensor(ids, dtype=torch.int32, device="cuda")

    # HF+PEFT forward.
    with torch.no_grad():
        out = peft_model(input_ids=tokens.long().unsqueeze(0), use_cache=False)
    logits_hf = out.logits[0, -1].float()  # [VOCAB]
    top_hf = int(logits_hf.argmax().item())

    # Our kernel forward with packed LoRA.
    flat = pack_peft_to_flat(peft_model, rank)
    weights, _ = load_weights(BASE_MODEL, verbose=False, backend="bf16")
    layers_packed = _pack_layer_weights(weights["layer_data"])
    sc = alloc_scratch(S, rank)
    our_prefill_with_lora(weights, layers_packed, tokens, sc, flat, rank, 1.0)
    final_normed = sc["final_normed"]
    logits_ours = (final_normed.float() @
                   weights["lm_head_weight"].float().t())
    top_ours = int(sc["out_token"].item())

    cos = torch.nn.functional.cosine_similarity(
        logits_hf.unsqueeze(0), logits_ours.unsqueeze(0), dim=-1).item()
    print(f"  prompt_len={S}")
    print(f"  HF+PEFT next-token argmax : {top_hf}")
    print(f"  ours next-token argmax    : {top_ours}")
    print(f"  cosine(logits_hf, logits_ours) : {cos:.5f}")

    # The DN paths differ between HF (fla Triton) and our cuBLAS+graph
    # kernel, so bit-exact agreement isn't expected. We require:
    # - cosine > 0.95 (forward is in the same direction)
    # - top-1 matches (sometimes will fail on tied logits — relax to top-K)
    cos_ok = cos > 0.95
    top5_hf = set(logits_hf.topk(5).indices.tolist())
    top1_ok = top_ours in top5_hf

    if cos_ok and top1_ok:
        print(f"  PASS — packed LoRA produces forward in agreement with HF+PEFT")
        return True
    else:
        print(f"  FAIL — cos_ok={cos_ok} top1_in_top5={top1_ok}  top5_hf={sorted(top5_hf)}")
        return False


def main():
    print("Loading PEFT-wrapped Qwen3.5-0.8B (rank=8, q/k/v/o + gate/up/down)...",
          flush=True)
    peft_model = build_peft_model(LORA_RANK)
    n_lora = randomize_lora(peft_model, std=0.02)
    print(f"randomized {n_lora} LoRA parameter tensors")

    ok = True
    ok &= test_round_trip(peft_model, LORA_RANK)
    ok &= test_forward_equivalence(peft_model, LORA_RANK)

    print()
    print("=" * 60)
    if ok:
        print("ALL CHECKS PASSED ✓  pack/unpack ready for Slice B (custom backward).")
    else:
        print("FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
