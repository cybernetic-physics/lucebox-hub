"""Verify the new Slice B.3b kernel saves (Q, O, LSE per FA layer) are
populated by prefill_bf16_train_step. After this validates, the
hand-rolled per_layer_bwd_fa can call fa_bwd_flash directly with these
saves instead of recomputing the FA forward under autograd.
"""
from __future__ import annotations

import sys

try:
    import transformers.utils.import_utils as _iu
    _iu.is_flash_linear_attention_available = lambda: False
except Exception:
    pass

# PEFT torchao import-check shim (same as in rl_trainer.py).
def _no_torchao(*_a, **_kw):
    return False
try:
    import peft.import_utils as _peft_iu
    _peft_iu.is_torchao_available = _no_torchao
except Exception:
    pass
try:
    import peft.tuners.lora.torchao as _peft_lora_torchao
    _peft_lora_torchao.is_torchao_available = _no_torchao
except Exception:
    pass

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b")
sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")

import torch  # noqa: E402

from lora_megakernel_step import (  # noqa: E402
    load_base_model, _train_step_forward, alloc_activation_saves,
)
from lora_pack import pack_peft_to_flat, N_FA  # noqa: E402

BASE = "Qwen/Qwen3.5-0.8B"
LORA_RANK = 8
S = 16


def main():
    print("Loading base + PEFT...")
    handle = load_base_model(BASE)
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    base = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16
    ).to("cuda").eval()
    cfg = LoraConfig(r=LORA_RANK, lora_alpha=LORA_RANK,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"],
                     lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    peft_model = get_peft_model(base, cfg).to("cuda", dtype=torch.bfloat16).eval()
    flat = pack_peft_to_flat(peft_model, LORA_RANK)

    tokens = torch.randint(0, 100, (S,), dtype=torch.int32, device="cuda")
    saves = alloc_activation_saves(S, with_b2_saves=True, with_fa_bwd_saves=True)
    sc, _ = _train_step_forward(handle, tokens, flat, LORA_RANK, 1.0, saves=saves)

    fails = []
    for label, key in [
        ("fa_q_save",   "fa_q_save"),
        ("fa_o_save",   "fa_o_save"),
        ("fa_lse_save", "fa_lse_save"),
    ]:
        t = saves[key]
        zero_layers = sum(1 for li in range(N_FA) if t[li].abs().max().item() == 0.0)
        non_zero = N_FA - zero_layers
        print(f"  {label:15s} shape={str(tuple(t.shape)):24s}  populated={non_zero}/{N_FA} layers")
        if zero_layers > 0:
            fails.append(f"{label}: {zero_layers}/{N_FA} layers all-zero")

    print()
    print("=" * 60)
    if not fails:
        print("ALL CHECKS PASSED ✓  FA bwd saves are populated correctly.")
    else:
        print("FAIL:")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
