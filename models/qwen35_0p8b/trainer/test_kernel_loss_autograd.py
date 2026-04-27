"""Validate kernel_loss_autograd's d(h_pre_norm) against HF+PEFT.

The HF+PEFT model exposes the full autograd graph from inputs to loss.
We can hook the residual stream at the output of the last transformer
layer (= input to the final RMSnorm) and read its gradient. That's the
EXACT tensor our kernel produces in `sc["hidden"]`, with the same
forward semantics. Our `kernel_loss_autograd` produces the same
gradient via a separate (kernel-driven) forward.

Test: gradient match within bf16 + DN-noise tolerance.

If this passes, Slice B.3b can confidently consume `grad_h_pre_norm`
as the upstream signal to the per-layer reverse walk.
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
    load_base_model, kernel_loss_autograd,
)
from lora_pack import pack_peft_to_flat  # noqa: E402


BASE_MODEL = "Qwen/Qwen3.5-0.8B"
LORA_RANK = 8


def build_peft_model(rank):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16
    ).to("cuda")
    cfg = LoraConfig(
        r=rank, lora_alpha=rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    return get_peft_model(base, cfg).to("cuda", dtype=torch.bfloat16).train()


def randomize_lora(peft_model, std=0.02):
    g = torch.Generator(device="cuda").manual_seed(13)
    for name, p in peft_model.named_parameters():
        if "lora_B" in name and ".weight" in name:
            p.data.copy_(torch.randn(p.shape, generator=g, device="cuda",
                                     dtype=p.dtype) * std)


def hf_peft_grad_h_pre_norm(peft_model, prompt_tokens, target_tokens):
    """Run HF+PEFT forward+backward, capture d(loss)/d(post-last-layer-residual)
    via a pre-hook on the final RMSnorm.

    The residual stream entering the final norm is the same `h_pre_norm`
    our kernel writes to sc["hidden"] (we proved bit-equality earlier).
    This is the natural ground truth for our gradient.
    """
    full = torch.cat([prompt_tokens.long(), target_tokens.long()], dim=0).unsqueeze(0)
    P = int(prompt_tokens.numel())
    T = int(target_tokens.numel())

    # Hook the input to the final RMSnorm. Qwen3-Next exposes this as
    # `model.norm` (an `Qwen3NextRMSNorm` instance). The pre-hook fires
    # with the input tuple — we capture the input tensor and request
    # autograd on it. We then read its grad after backward.
    inner = peft_model.base_model.model.model
    final_norm = inner.norm

    captured: dict[str, torch.Tensor] = {}

    def pre_hook(module, args):
        x = args[0]
        x.requires_grad_(True)
        x.retain_grad()
        captured["h_pre_norm"] = x
        return None  # don't modify args

    handle = final_norm.register_forward_pre_hook(pre_hook)
    try:
        out = peft_model(input_ids=full, use_cache=False)
        logits = out.logits[0].to(torch.float32)
        predict_logits = logits[P - 1: P - 1 + T]
        log_probs = torch.nn.functional.log_softmax(predict_logits, dim=-1)
        logp = log_probs.gather(1, target_tokens.long().unsqueeze(1)).squeeze(1)
        loss = -logp.mean()
        loss.backward()
    finally:
        handle.remove()

    h = captured["h_pre_norm"]  # [1, S, HIDDEN]
    g = h.grad
    if g is None:
        raise RuntimeError("HF+PEFT didn't accumulate grad on h_pre_norm")
    return loss.detach(), g[0].float().detach()  # [S, HIDDEN]


def main():
    print("Loading PEFT-wrapped Qwen3.5-0.8B...", flush=True)
    peft_model = build_peft_model(LORA_RANK)
    randomize_lora(peft_model)

    print("Loading our base-model handle...", flush=True)
    handle = load_base_model(BASE_MODEL)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    prompt = "The capital of France is"
    target = " Paris."
    p_ids = torch.tensor(tok.encode(prompt, add_special_tokens=False),
                         dtype=torch.int32, device="cuda")
    t_ids = torch.tensor(tok.encode(target, add_special_tokens=False),
                         dtype=torch.int32, device="cuda")
    print(f"prompt_len={p_ids.numel()}, target_len={t_ids.numel()}")

    # HF+PEFT reference.
    print("\nRunning HF+PEFT autograd reference...")
    hf_loss, hf_grad = hf_peft_grad_h_pre_norm(peft_model, p_ids, t_ids)

    # Our kernel-based path.
    print("Running our kernel + autograd...")
    flat = pack_peft_to_flat(peft_model, LORA_RANK)
    out = kernel_loss_autograd(handle, p_ids, t_ids, flat,
                                lora_rank=LORA_RANK, lora_scaling=1.0)

    our_loss = out["loss"]
    our_grad = out["grad_h_pre_norm"].float()

    print(f"\nHF+PEFT loss : {hf_loss.item():.6f}")
    print(f"our loss     : {our_loss.item():.6f}")
    print(f"loss |Δ|     : {abs(our_loss.item() - hf_loss.item()):.4e}")

    # Compare gradients element-wise.
    diff = (hf_grad - our_grad).abs()
    rel = diff / (hf_grad.abs() + 1e-6)
    cos = torch.nn.functional.cosine_similarity(
        hf_grad.flatten().unsqueeze(0),
        our_grad.flatten().unsqueeze(0), dim=-1).item()

    print(f"\ngrad_h_pre_norm comparison:")
    print(f"  shape     : {tuple(hf_grad.shape)}")
    print(f"  max|Δ|    : {diff.max().item():.4e}")
    print(f"  mean|Δ|   : {diff.mean().item():.4e}")
    print(f"  max rel   : {rel.max().item():.4e}")
    print(f"  cosine    : {cos:.6f}")
    print(f"  HF |grad| : {hf_grad.abs().max().item():.4e}")
    print(f"  our |grad|: {our_grad.abs().max().item():.4e}")

    # The two paths use different DN forward routes (HF+PEFT through
    # torch fallback / fla, ours through cuBLAS+graph), so bit-exact
    # equality isn't expected — DN-path noise drives a ~1-3% relative
    # loss difference per sample. The grad direction is what matters
    # for B.3b: cos must be tight, loss within 5% relative.
    rel_loss = abs(our_loss.item() - hf_loss.item()) / max(hf_loss.item(), 1e-3)
    ok = cos >= 0.99 and rel_loss < 0.05
    print()
    print("=" * 60)
    if ok:
        print("PASS — grad_h_pre_norm matches HF+PEFT within bf16 + DN-noise tolerance.")
        print("Slice B.3b can consume this as the entry signal for the per-layer walk.")
    else:
        print(f"FAIL — cos={cos:.5f}, loss Δ={abs(our_loss.item() - hf_loss.item()):.4e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
