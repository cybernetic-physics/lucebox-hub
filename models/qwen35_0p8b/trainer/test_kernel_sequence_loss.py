"""Validate kernel_sequence_loss against HF+PEFT.

This is the foundation check for Slice B.2: our kernel-driven sequence
loss must match HF+PEFT's `_sequence_logprobs`-based mean-loss within
bf16 noise (DN routing differences mean we won't be bit-exact).

If this passes:
  - pack_peft_to_flat correctly translates LoRA layouts (already proven)
  - prefill_bf16_train_step + Python final-norm + lm_head reproduces
    HF+PEFT's forward path
  - sc["hidden"] really IS h_out[NUM_LAYERS-1] for all positions
  - We have a stable d(h_pre_norm) entry point for Slice B.2's
    layer-walking backward (autograd through the Python lm_head + norm
    will give us d(sc["hidden"]) for free)
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
    load_base_model, kernel_sequence_loss,
)
from lora_pack import pack_peft_to_flat  # noqa: E402


BASE_MODEL = "Qwen/Qwen3.5-0.8B"
LORA_RANK = 8


def build_peft_model(rank: int):
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
    peft_model = get_peft_model(base, cfg).to("cuda", dtype=torch.bfloat16)
    peft_model.eval()
    return peft_model


def randomize_lora(peft_model, std=0.02):
    g = torch.Generator(device="cuda").manual_seed(7)
    n = 0
    for name, p in peft_model.named_parameters():
        if "lora_B" in name and ".weight" in name:
            p.data.copy_(torch.randn(p.shape, generator=g, device="cuda",
                                     dtype=p.dtype) * std)
            n += 1
    return n


def hf_peft_sequence_loss(peft_model, prompt_tokens, target_tokens):
    """Mirror rl_trainer._sequence_logprobs."""
    full = torch.cat([prompt_tokens.long(), target_tokens.long()], dim=0).unsqueeze(0)
    with torch.no_grad():
        out = peft_model(input_ids=full, use_cache=False)
    logits = out.logits[0].to(torch.float32)
    P = int(prompt_tokens.numel())
    T = int(target_tokens.numel())
    predict_logits = logits[P - 1: P - 1 + T]
    log_probs = torch.nn.functional.log_softmax(predict_logits, dim=-1)
    logp = log_probs.gather(1, target_tokens.long().unsqueeze(1)).squeeze(1)
    return (-logp.mean()).item(), logp.detach()


def main():
    print("Loading PEFT-wrapped Qwen3.5-0.8B...", flush=True)
    peft_model = build_peft_model(LORA_RANK)
    n = randomize_lora(peft_model)
    print(f"randomized {n} LoRA-B tensors")

    print("Loading our base-model handle...", flush=True)
    handle = load_base_model(BASE_MODEL)

    # Build prompt + target.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    prompt = "The capital of France is"
    target = " Paris."
    p_ids = torch.tensor(tok.encode(prompt, add_special_tokens=False),
                         dtype=torch.int32, device="cuda")
    t_ids = torch.tensor(tok.encode(target, add_special_tokens=False),
                         dtype=torch.int32, device="cuda")
    print(f"prompt_len={p_ids.numel()}, target_len={t_ids.numel()}")

    # HF+PEFT loss.
    hf_loss, hf_logp = hf_peft_sequence_loss(peft_model, p_ids, t_ids)
    print(f"\nHF+PEFT mean loss : {hf_loss:.4f}")
    print(f"HF+PEFT per-token logp : {hf_logp.cpu().tolist()}")

    # Our kernel loss.
    flat = pack_peft_to_flat(peft_model, LORA_RANK)
    out = kernel_sequence_loss(
        handle, p_ids, t_ids, flat,
        lora_rank=LORA_RANK, lora_scaling=1.0,
        return_per_position_logp=True,
    )
    our_loss = float(out["loss"].item())
    our_logp = out["logp"]
    print(f"\nour kernel mean loss : {our_loss:.4f}")
    print(f"our per-token logp   : {our_logp.cpu().tolist()}")

    delta = abs(our_loss - hf_loss)
    cos = torch.nn.functional.cosine_similarity(
        our_logp.unsqueeze(0), hf_logp.unsqueeze(0).to(our_logp.dtype), dim=-1
    ).item()
    print(f"\nloss |Δ|        : {delta:.4e}")
    print(f"per-token logp cosine : {cos:.5f}")

    ok = delta < 0.1 and cos > 0.99
    print()
    print("=" * 60)
    if ok:
        print("PASS — kernel sequence loss matches HF+PEFT within bf16 + DN-noise tolerance")
    else:
        print(f"FAIL — loss Δ={delta:.4e}, cos={cos:.5f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
