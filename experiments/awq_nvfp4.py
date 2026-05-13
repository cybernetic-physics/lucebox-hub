"""AWQ-style per-channel scaling for NVFP4 layer weights.

Idea: the standard FP4 quantizer picks scale = absmax / 6 per group of 32.
Outliers dominate absmax and 50%+ of FP4 precision goes to capturing them
while typical values get snapped to 0 / ±0.5. Cross-channel rebalancing
trades quantization budget from "channels rarely activated" toward
"channels with large activations", preserving the channels that actually
move the output.

Implementation (no kernel changes — pure weight surgery):

  For each rmsnorm -> {q_proj, k_proj, v_proj, ...} block:
    1. Capture activations passing through rmsnorm during a calibration
       forward in BF16. The activation tensor h has shape [tokens, HIDDEN].
    2. Per-input-channel salience: s_j = mean_t |h[t, j]| ** alpha (alpha=0.5).
    3. Renormalize s so its geometric mean is 1 (avoids global scaling).
    4. Fold into the rmsnorm weight: g <- g / s    (math identity below).
    5. Inverse-fold into each downstream projection: W <- W * s
       (W's columns matched to input channels).
    6. Quantize the rescaled W to FP4.

  The pre-rmsnorm tensor x is untouched. After this rewrite:
      h' = (x / rms(x)) * (g / s)        # rmsnorm output is downscaled per channel
      y  = h' @ W'^T = h' @ (W * s)^T    # projection inverse-scales -> identity
  i.e. h' @ W'^T = h @ W^T for any s, so the unquantized math matches HF
  exactly. After FP4 quantization of W', high-salience channels (in s)
  have weight magnitudes inflated by s_j, so they survive FP4 better.

For Qwen3.5-0.8B the rmsnorm -> projection blocks are:
  - input_layernorm    -> q_proj, k_proj, v_proj   (FA layers)
  - input_layernorm    -> qkv_proj, z_proj         (DN layers)
  - post_attn_norm     -> gate_proj, up_proj       (both)

We do NOT touch o_proj or down_proj because they don't follow an
rmsnorm — their input is the attention/MLP output and folding a
diagonal scale there is not a math identity.
"""
from __future__ import annotations

import argparse
import sys

import torch


def gather_calibration_activations(model, tokenizer, prompts, device="cuda"):
    """Hook every relevant rmsnorm output and collect activation magnitudes."""
    saved = {}
    hooks = []

    def make_hook(name):
        def hook(_module, _inp, out):
            # out is the rmsnorm output (post-norm, pre-projection input).
            v = out.detach().float().abs()
            if name not in saved:
                saved[name] = (v.flatten(0, -2).sum(0).double(),
                               torch.tensor(v.flatten(0, -2).shape[0],
                                            device=device, dtype=torch.float64))
            else:
                tot, cnt = saved[name]
                saved[name] = (tot + v.flatten(0, -2).sum(0).double(),
                               cnt + v.flatten(0, -2).shape[0])
        return hook

    # Wire one hook per rmsnorm we care about.
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.input_layernorm.register_forward_hook(
            make_hook(f"L{i}.input_ln")))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(
            make_hook(f"L{i}.post_attn_ln")))

    with torch.inference_mode():
        for p in prompts:
            ids = tokenizer.encode(p, return_tensors="pt").to(device)
            model(ids)

    for h in hooks:
        h.remove()

    return {name: (tot / cnt).to(device).float()
            for name, (tot, cnt) in saved.items()}


def apply_awq_per_channel_scales(model, mean_abs, alpha: float = 0.5):
    """In-place AWQ folding. Returns the diagonal scales applied per layer
    so callers can verify / debug. Equivalent math is preserved."""
    eps = 1e-5
    diagnostics = {}

    for i, layer in enumerate(model.model.layers):
        # Layer type 1 = full attention. q/k/v rows = output channels;
        # cols match HIDDEN_SIZE = input channels. Same for DN.
        # Only AWQ the rmsnorm -> projection blocks where the projection's
        # OUTPUT isn't subsequently renormalized. Qwen3-style q_norm/k_norm
        # would cancel any q_proj/k_proj scaling we apply; DN linear_attn
        # has its own internal norms too. The MLP path is the safe pocket:
        #   post_attn_ln -> gate_proj/up_proj -> SiLU(gate)*up -> down_proj
        # No norm between post_attn_ln and gate/up, so the identity holds.
        for ln_name, proj_names in (
            ("post_attn_ln", ["mlp.gate_proj", "mlp.up_proj"]),
        ):
            key = f"L{i}.{ln_name}"
            if key not in mean_abs:
                continue
            mag = mean_abs[key].clamp(min=eps)
            # Standard AWQ alpha=0.5 -> sqrt(magnitude).
            s = mag.pow(alpha)
            # Renormalize: geometric mean = 1, keeps the global scale of
            # the rmsnorm output unchanged. (Pure scaling identity, but
            # avoids fp16-range issues in downstream scales.)
            s = s / s.log().mean().exp()

            ln_mod = (layer.input_layernorm
                      if ln_name == "input_ln"
                      else layer.post_attention_layernorm)
            with torch.no_grad():
                # Fold into rmsnorm weight: g <- g / s
                ln_mod.weight.div_(s)
                # Inverse-fold into each downstream projection: W <- W * s
                # (W has shape [out, in], s is per input column.)
                for proj_name in proj_names:
                    parts = proj_name.split(".")
                    parent = layer
                    ok = True
                    for part in parts[:-1]:
                        if not hasattr(parent, part):
                            ok = False
                            break
                        parent = getattr(parent, part)
                    if not ok:
                        continue
                    proj = getattr(parent, parts[-1], None)
                    if proj is None or not hasattr(proj, "weight"):
                        continue
                    if proj.weight.shape[1] != s.numel():
                        continue
                    proj.weight.mul_(s.to(proj.weight.dtype))
            diagnostics[key] = {
                "s_min": float(s.min()), "s_max": float(s.max()),
                "s_mean": float(s.mean()),
            }
    return diagnostics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--n-calib", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, "models/qwen35_0p8b")
    import qwen35_megakernel_bf16_C  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from model import (Decoder, _attach_nvfp4_weights, NVFP4_GROUP_SIZE,
                       _unify_weights_from_hf)

    tok = AutoTokenizer.from_pretrained(args.model_name)

    print("Loading HF model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    calib_prompts = [
        "The capital of France is",
        "Once upon a time, in a faraway",
        "import numpy as np\n\ndef fibonacci(n):",
        "Q: What is the meaning of life?\nA:",
    ][: args.n_calib]

    # Baseline top token BEFORE any modification.
    with torch.inference_mode():
        ids = tok.encode(calib_prompts[0], return_tensors="pt").to("cuda")
        orig_logits = model(ids).logits
        orig_top = int(orig_logits[0, -1].argmax().item())
    print(f"BASELINE HF top token: {orig_top} ({tok.decode([orig_top])!r})")

    print(f"Gathering calibration activations from {len(calib_prompts)} prompts...",
          flush=True)
    mean_abs = gather_calibration_activations(model, tok, calib_prompts)
    print(f"  collected magnitudes for {len(mean_abs)} rmsnorms", flush=True)

    print(f"Applying AWQ per-channel scaling (alpha={args.alpha})...", flush=True)
    diag = apply_awq_per_channel_scales(model, mean_abs, alpha=args.alpha)
    for k in list(diag.keys())[:2] + list(diag.keys())[-2:]:
        print(f"  {k}: {diag[k]}")

    # Verify the math identity is preserved on the calibration prompts.
    print("\nSanity: HF forward with AWQ-modified weights should match the "
          "ORIGINAL HF forward (within fp32 rounding)...", flush=True)
    with torch.inference_mode():
        ids = tok.encode(calib_prompts[0], return_tensors="pt").to("cuda")
        new_logits = model(ids).logits
        new_top = int(new_logits[0, -1].argmax().item())
        print(f"  AWQ-modified HF top token: {new_top} ({tok.decode([new_top])!r})")

    # Build a Decoder that wraps the AWQ-modified weights, then quantize.
    print("\nBuilding NVFP4 Decoder from AWQ-scaled weights...", flush=True)
    weights, _ = _unify_weights_from_hf(model)
    weights["_hf_model_keepalive"] = model
    _attach_nvfp4_weights(weights, group_size=NVFP4_GROUP_SIZE, verbose=True)
    d = Decoder(weights=weights, tokenizer=tok, model_name=args.model_name,
                backend="nvfp4", verbose=False)
    ids = tok.encode("The capital of France is", add_special_tokens=False)
    out = d.prefill(ids)
    print(f"\nAWQ NVFP4 first token: {out} ({tok.decode([out])!r})")
    print(f"  HF baseline says ' Paris' (11751)")


if __name__ == "__main__":
    main()
