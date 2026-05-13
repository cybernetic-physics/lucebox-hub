"""Optimal-scale FP4 quantization: instead of `amax/6`, sweep candidate
scales per group and pick the one minimizing squared error.

The default kernel quantizer picks scale = amax / 6, which is the
"saturating" choice — every FP4 magnitude {0, 0.5, 1, 1.5, 2, 3, 4, 6}
can be reached, but outliers dominate and typical values get poor
resolution. With 8 candidate scales we get a near-optimal scalar
quantizer per group, at no kernel cost (offline pass).

We override `_quantize_matrix_nvfp4` to use this scheme, then quantize
all layer weights and run nvfp4 prefill on the test prompt.
"""
from __future__ import annotations

import argparse
import sys

import torch


FP4_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32, device="cuda")
FP4_POS_MAGS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                            dtype=torch.float32, device="cuda")
SCALE_CANDIDATES = torch.tensor(
    [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
     0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00, 1.02, 1.05,
     1.10, 1.15, 1.20, 1.30, 1.40, 1.55, 1.70, 1.85, 2.00, 2.25],
    dtype=torch.float32, device="cuda")


def optimal_fp4_quantize(weight: torch.Tensor, group_size: int = 32):
    """Optimal-MSE FP4 group quantizer.

    For each group of `group_size` consecutive elements in each row:
      1. Compute amax.
      2. Try `len(SCALE_CANDIDATES)` candidate base scales `c_i * amax / 6`.
      3. For each candidate, round each element to the nearest FP4 value
         and compute the squared error vs the original.
      4. Pick the candidate with minimum squared error.
      5. Quantize using that scale, return packed bytes + fp16 scales.

    Returns {'packed', 'scales'} matching the layer kernel's expected
    layout: packed [rows, cols//2] uint8, scales [rows, cols//group_size]
    fp16.
    """
    assert weight.dtype == torch.bfloat16
    assert weight.dim() == 2
    rows, cols = weight.shape
    assert cols % group_size == 0
    g = group_size
    G = cols // g
    Wf = weight.float()  # [rows, cols]
    Wg = Wf.reshape(rows, G, g)  # [rows, G, g]

    amax = Wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)  # [rows, G, 1]
    base_scale = amax / 6.0  # [rows, G, 1]
    # Insert a singleton at dim -2 (will broadcast over the `g` axis below).
    cand_scales = base_scale.unsqueeze(-1) * SCALE_CANDIDATES.view(1, 1, 1, -1)  # [rows, G, 1, C]

    # For each candidate, find best FP4 code for each element.
    # Wg: [rows, G, g, 1]; cand_scales: [rows, G, 1, C]
    Wg4 = Wg.unsqueeze(-1)  # [rows, G, g, 1]
    inv_s = 1.0 / cand_scales  # [rows, G, 1, C]
    norm = Wg4 * inv_s  # [rows, G, g, C]
    sign = norm.sign()
    abs_norm = norm.abs()  # [rows, G, g, C]
    # Round each abs(norm) to nearest pos magnitude.
    # FP4_POS_MAGS: [8]; abs_norm: [rows, G, g, C]
    abs_diff = (abs_norm.unsqueeze(-1) - FP4_POS_MAGS).abs()
    idx = abs_diff.argmin(dim=-1)  # [rows, G, g, C]
    qmag = FP4_POS_MAGS[idx]  # [rows, G, g, C]
    qval = qmag * sign * cand_scales  # [rows, G, g, C] in original units

    # Per-candidate squared error.
    err = (qval - Wg4).pow(2).sum(dim=2)  # [rows, G, C]

    best_c = err.argmin(dim=-1)  # [rows, G]
    # Gather best scale + best quantized values.
    best_scale = cand_scales.squeeze(-2).gather(  # [rows, G, C] -> [rows, G]
        -1, best_c.unsqueeze(-1)).squeeze(-1)
    # For packed output: redo the quantization at best scale.
    inv_best = 1.0 / best_scale.unsqueeze(-1)  # [rows, G, 1]
    norm_best = Wg * inv_best  # [rows, G, g]
    sign_b = norm_best.sign()
    abs_b = norm_best.abs()
    idx_b = (abs_b.unsqueeze(-1) - FP4_POS_MAGS).abs().argmin(dim=-1)  # [rows, G, g]
    # Encode FP4 code: sign bit at bit 3, magnitude bits 0-2.
    is_neg = (sign_b < 0).to(torch.uint8)
    code = (is_neg << 3) | idx_b.to(torch.uint8)  # [rows, G, g] in [0, 16)
    # Pack two codes per byte.
    code_flat = code.reshape(rows, G * g)  # [rows, cols]
    lo = code_flat[:, 0::2]
    hi = code_flat[:, 1::2]
    packed = (hi << 4) | lo  # [rows, cols//2]

    scales_fp16 = best_scale.to(torch.float16)  # [rows, G]
    return {"packed": packed.contiguous(), "scales": scales_fp16.contiguous()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--gen-tokens", type=int, default=8)
    args = ap.parse_args()

    sys.path.insert(0, "models/qwen35_0p8b")
    import qwen35_megakernel_bf16_C  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import model as model_mod
    from model import (Decoder, _unify_weights_from_hf, _attach_nvfp4_weights,
                       NVFP4_GROUP_SIZE)

    # Monkey-patch the layer quantizer to use optimal-MSE scales.
    orig_quant = model_mod._quantize_matrix_nvfp4
    def patched_quant(weight, group_size):
        return optimal_fp4_quantize(weight, group_size)
    model_mod._quantize_matrix_nvfp4 = patched_quant
    print("[patch] _quantize_matrix_nvfp4 -> optimal_fp4_quantize", flush=True)

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    print("Loading HF base...", flush=True)
    hf = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16, device_map="cuda")
    hf.eval()
    weights, _ = _unify_weights_from_hf(hf)
    weights["_hf_model_keepalive"] = hf
    print("Quantizing layer weights with optimal-MSE scales (slow)...", flush=True)
    _attach_nvfp4_weights(weights, group_size=args.group_size, verbose=True)

    d = Decoder(weights=weights, tokenizer=tok, backend="nvfp4", verbose=False)
    pids = tok.encode(args.prompt, add_special_tokens=False)
    first = d.prefill(pids)
    print(f"\nOptimal-FP4 NVFP4 first token: {first} ({tok.decode([first])!r})")

    cur = first
    out_ids = [cur]
    for _ in range(args.gen_tokens - 1):
        cur = int(d.step(cur))
        out_ids.append(cur)
    print(f"Generated: {tok.decode(out_ids, skip_special_tokens=True)!r}")
    print(f"HF baseline says ' Paris' (token 11751)")


if __name__ == "__main__":
    main()
