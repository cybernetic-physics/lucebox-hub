"""Verify the two new Slice B.2 activation saves get populated correctly.

Tests:
  1. attn_out_pre_o is non-zero for all layers (= the kernel actually
     wrote into it).
  2. h_post_attn is non-zero for all layers.
  3. h_post_attn[L] = hidden_in[L] + attn_out[L]. We don't have
     attn_out saved, but we can verify the relationship using the post-
     attn RMSnorm: rms_norm(h_post_attn[L]) ≈ normalized_post_attn[L].
     If both saves are correctly populated, this must hold (within bf16
     noise).
  4. The kernel still produces the same final logits as before — adding
     saves doesn't perturb the forward path. We compare next-token from
     a run with saves enabled vs a run with saves nulled.
"""
from __future__ import annotations

import sys

try:
    import transformers.utils.import_utils as _iu
    _iu.is_flash_linear_attention_available = lambda: False
except Exception:
    pass

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b")
sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")

import torch  # noqa: E402

from lora_megakernel_step import (  # noqa: E402
    load_base_model, _train_step_forward, alloc_activation_saves,
    NUM_LAYERS, HIDDEN, DN_V_SIZE, _rms_norm_qwen,
)
from lora_pack import pack_peft_to_flat  # noqa: E402


BASE_MODEL = "Qwen/Qwen3.5-0.8B"
LORA_RANK = 8


def build_peft_model(rank):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16
    ).to("cuda").eval()
    cfg = LoraConfig(
        r=rank, lora_alpha=rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    return get_peft_model(base, cfg).to("cuda", dtype=torch.bfloat16).eval()


def main():
    print("Loading base model + PEFT...", flush=True)
    handle = load_base_model(BASE_MODEL)
    peft_model = build_peft_model(LORA_RANK)
    flat = pack_peft_to_flat(peft_model, LORA_RANK)

    # Recover post-attn RMSnorm weights from packed layer data so we can
    # check rms_norm(h_post_attn[L]) ≈ normalized_post_attn[L] per layer.
    layer_data = handle.weights["layer_data"]

    text = "The quick brown fox jumps over the lazy dog."
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    ids = tok.encode(text, add_special_tokens=False)
    S = len(ids)
    tokens = torch.tensor(ids, dtype=torch.int32, device="cuda")

    # ----- Run with all saves enabled (B.1 + B.2) -----
    saves_b2 = alloc_activation_saves(S, with_b2_saves=True)
    sc_b2, _ = _train_step_forward(handle, tokens, flat, LORA_RANK, 1.0,
                                    saves=saves_b2)
    out_tok_b2 = int(sc_b2["out_token"].item())

    # ----- Run with saves disabled (sanity: forward should be unchanged) -----
    saves_off = alloc_activation_saves(S, with_b2_saves=False)
    # Drop hidden_in too — pure inference path. Actually we still need
    # the 4 B.1 slabs to be empty. Build an empty-tensor version.
    empty_bf16 = torch.empty(0, dtype=torch.bfloat16, device="cuda")
    saves_disabled = {
        "hidden_in": empty_bf16,
        "normalized_in": empty_bf16,
        "normalized_post_attn": empty_bf16,
        "mlp_inter": empty_bf16,
    }  # _train_step_forward will pass empty for B.2 too via .get fallback
    sc_off, _ = _train_step_forward(handle, tokens, flat, LORA_RANK, 1.0,
                                     saves=saves_disabled)
    out_tok_off = int(sc_off["out_token"].item())

    print(f"\nout_token (saves on)  : {out_tok_b2}")
    print(f"out_token (saves off) : {out_tok_off}")

    fails = []
    if out_tok_b2 != out_tok_off:
        fails.append(f"saves perturbed forward: {out_tok_b2} vs {out_tok_off}")
    else:
        print("  ✓ adding saves preserves the forward output")

    # ----- Check attn_out_pre_o is non-zero per layer -----
    aopo = saves_b2["attn_out_pre_o"]  # [NUM_LAYERS, S, DN_V_SIZE]
    n_zero_layers = 0
    for li in range(NUM_LAYERS):
        if aopo[li].abs().max().item() == 0.0:
            n_zero_layers += 1
    print(f"\nattn_out_pre_o: {n_zero_layers} of {NUM_LAYERS} layers all-zero")
    if n_zero_layers > 0:
        fails.append(f"{n_zero_layers} attn_out_pre_o layer slabs are all-zero")
    else:
        print("  ✓ all 24 layers have non-zero attn_out_pre_o")

    # ----- Check h_post_attn is non-zero per layer -----
    hpa = saves_b2["h_post_attn"]  # [NUM_LAYERS, S, HIDDEN]
    n_zero_layers = 0
    for li in range(NUM_LAYERS):
        if hpa[li].abs().max().item() == 0.0:
            n_zero_layers += 1
    print(f"h_post_attn:    {n_zero_layers} of {NUM_LAYERS} layers all-zero")
    if n_zero_layers > 0:
        fails.append(f"{n_zero_layers} h_post_attn layer slabs are all-zero")
    else:
        print("  ✓ all 24 layers have non-zero h_post_attn")

    # ----- Verify h_post_attn → normalized_post_attn via rms_norm -----
    # The per-layer post-attn norm weight lives at ptr index 7 in FA
    # layers, and at a different index in DN layers. Use the layer_data
    # entries directly.
    n_pn_match = 0
    n_pn_check = 0
    LAYER_TYPE = handle.weights.get("LAYER_TYPE", None)
    # outer model exports LAYER_TYPE — fall back to importing it.
    if LAYER_TYPE is None:
        import importlib.util as _u
        _spec = _u.spec_from_file_location(
            "qwen_outer_model",
            "/root/lucebox-hub-b200-train/models/qwen35_0p8b/model.py")
        _outer = _u.module_from_spec(_spec); _spec.loader.exec_module(_outer)
        LAYER_TYPE = _outer.LAYER_TYPE

    npa = saves_b2["normalized_post_attn"]  # [NUM_LAYERS, S, HIDDEN]
    for li in range(NUM_LAYERS):
        # post-attn rmsnorm weight: in layer_data[li]["ptrs"], the
        # layout for FA is: ..., ptrs[7] = post_attn_norm_weight.
        # For DN: ptrs[10] = post_attn_norm. We don't have a clean
        # reverse-map handy, so brute-force: try ptrs[7] first, fall
        # back to ptrs[10] if the result doesn't match.
        ptrs = layer_data[li]["ptrs"]
        h_in_bf = hpa[li]
        npa_kernel = npa[li]
        for ptr_idx in (7, 10, 8, 9):
            if ptr_idx >= len(ptrs):
                continue
            w = ptrs[ptr_idx]
            if w.shape != (HIDDEN,) or w.dtype != torch.bfloat16:
                continue
            try:
                npa_recomputed = _rms_norm_qwen(h_in_bf, w)
                diff = (npa_recomputed.float() - npa_kernel.float()).abs().max().item()
                if diff < 0.05:
                    n_pn_match += 1
                    n_pn_check += 1
                    break
            except Exception:
                continue
        else:
            n_pn_check += 1

    print(f"\nh_post_attn → rms_norm consistency : {n_pn_match}/{n_pn_check} layers match")
    if n_pn_match < n_pn_check:
        fails.append(
            f"only {n_pn_match}/{n_pn_check} layers' h_post_attn passes rms_norm consistency"
        )

    print()
    print("=" * 60)
    if not fails:
        print("ALL CHECKS PASSED ✓  Slice B.2 activation saves are populated correctly.")
    else:
        print("FAIL:")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
