"""Localize the NVFP4 step-0 divergence on GB10.

Strategy: the BF16 megakernel matches HF 32/32. NVFP4 diverges at step 0.
Two candidate causes:

  (A) FP4 quantization of the LM head loses the argmax token.
  (B) FP4 quantization of layer projections drifts the final hidden,
      so the argmax flips even before the LM head matters.

We isolate (A) and (B) by:

  step 0: Sanity-check the quantize/dequantize kernel on a random
          matrix; report max abs error and check it matches what FP4
          E2M1 group-32 round-trip can achieve.

  step 1: Capture HF's final hidden state h (just before LM head) and
          its BF16 LM head W. Compute:
              y_bf16 = h @ W.T   (argmax = HF top-1)
              y_fp4  = h @ dequant(quantize(W, g=32)).T
          Compare argmax. If y_fp4 picks the same token, the LM head
          quantization is benign; the divergence is in (B).

  step 2: Run BF16 and NVFP4 megakernels on the same prompt; for the
          single decode step, compare:
              MK_bf16 hidden vs HF h            -> kernel BF16 error
              MK_nvfp4 hidden vs HF h           -> aggregate FP4 error
          Both are reconstructed via lm_head_weight argmax (we read
          the argmax token back; full hidden state isn't directly
          exposed, so we exploit the rank of HF-argmax in the FP4
          logits).
"""
from __future__ import annotations

import argparse
import sys

import torch


def quantize_dequant_roundtrip(W: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """Mirror the kernel's quantize_nvfp4_out path via Python so we can
    inspect it independently. FP4 E2M1 codes: ±{0, .5, 1, 1.5, 2, 3, 4, 6}."""
    LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                       dtype=torch.float32, device=W.device)
    rows, cols = W.shape
    Wf = W.float().reshape(rows, cols // group_size, group_size)
    # Per-group scale chosen so |max| maps to 6.0.
    amax = Wf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / 6.0
    # Quantize: find nearest FP4 code in LUT/scale units.
    codes = (Wf / scale)  # in [-6, 6]
    # Each FP4 magnitude level:
    mags = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                        dtype=torch.float32, device=W.device)
    sign = codes.sign()
    absc = codes.abs()
    # Find closest magnitude level.
    diff = (absc.unsqueeze(-1) - mags).abs()
    idx = diff.argmin(dim=-1)
    qmag = mags[idx]
    qval = qmag * sign
    dequant = (qval * scale).reshape(rows, cols)
    return dequant.to(W.dtype), scale.squeeze(-1).to(torch.float16)


def step0_kernel_quant_diag(group_size: int = 32):
    """Check the kernel's own quantize_nvfp4_out vs a reference."""
    sys.path.insert(0, "models/qwen35_0p8b")
    import qwen35_megakernel_bf16_C  # noqa: F401
    torch.manual_seed(0)
    rows, cols = 256, 512
    W = torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")
    packed = torch.empty((rows, cols // 2), dtype=torch.uint8, device="cuda")
    scales = torch.empty((rows, cols // group_size), dtype=torch.float16, device="cuda")
    torch.ops.qwen35_megakernel_bf16_C.quantize_nvfp4_out(
        packed, scales, W.contiguous(), group_size
    )
    # Dequantize the kernel-packed bytes.
    LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                       dtype=torch.float32, device="cuda")
    lo = LUT[(packed & 0xF).long()]
    hi = LUT[(packed >> 4).long()]
    Wd = torch.empty(rows, cols, dtype=torch.float32, device="cuda")
    Wd[:, 0::2] = lo
    Wd[:, 1::2] = hi
    # Scales: per group of `group_size` columns.
    Wd = Wd.reshape(rows, cols // group_size, group_size) * scales.float().unsqueeze(-1)
    Wd = Wd.reshape(rows, cols)
    Wf = W.float()
    err = (Wd - Wf).abs()
    rel = err / Wf.abs().clamp(min=1e-6)
    print("[step 0] kernel quantize_nvfp4 round-trip on a 256x512 N(0,1):")
    print(f"    max abs err = {err.max().item():.6f}")
    print(f"    mean abs err = {err.mean().item():.6f}")
    print(f"    p99 abs err = {err.flatten().kthvalue(int(0.99*err.numel())).values.item():.6f}")
    print(f"    mean rel err = {rel.mean().item():.4%}")
    print(f"    scale dtype = {scales.dtype}, shape = {tuple(scales.shape)}")


def step1_lm_head_argmax_diag(model_name: str, prompt: str, group_size: int = 32):
    """Apply BF16 LM head and FP4-roundtripped LM head to HF's final
    hidden, see if argmax token flips."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    hf = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda"
    )
    hf.eval()
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([prompt_ids], device="cuda")
    with torch.inference_mode():
        out = hf(input_ids, output_hidden_states=True)
        h = out.hidden_states[-1][0, -1].float()  # [HIDDEN]
    # BF16 LM head.
    W = hf.lm_head.weight.detach().to(torch.bfloat16)  # [VOCAB, HIDDEN]
    y_bf16 = (h @ W.float().T)
    hf_top = int(y_bf16.argmax().item())
    # FP4-roundtrip LM head.
    Wd, scales = quantize_dequant_roundtrip(W, group_size=group_size)
    y_fp4 = (h @ Wd.float().T)
    fp4_top = int(y_fp4.argmax().item())
    # Rank of HF top in FP4 logits.
    fp4_sorted = torch.argsort(y_fp4, descending=True)
    rank = int((fp4_sorted == hf_top).nonzero(as_tuple=True)[0].item())
    print(f"\n[step 1] LM head sensitivity to FP4 (group_size={group_size}):")
    print(f"    prompt = {prompt!r} -> last-hidden norm = {h.norm().item():.4f}")
    print(f"    BF16  argmax = {hf_top} ({tok.decode([hf_top])!r})  "
          f"top-logit = {y_bf16.max().item():.4f}")
    print(f"    FP4   argmax = {fp4_top} ({tok.decode([fp4_top])!r})  "
          f"top-logit = {y_fp4.max().item():.4f}")
    print(f"    rank of BF16-top inside FP4 logits = {rank}")
    print(f"    delta between top-2 in BF16 = "
          f"{(y_bf16.topk(2).values[0] - y_bf16.topk(2).values[1]).item():.4f}")
    return hf_top, fp4_top, rank


def step2_megakernel_step0(model_name: str, prompt: str):
    """Drive both backends through one prefill+decode step, report tokens."""
    sys.path.insert(0, "models/qwen35_0p8b")
    import qwen35_megakernel_bf16_C  # noqa: F401
    from model import Decoder
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    pids = tok.encode(prompt, add_special_tokens=False)
    print(f"\n[step 2] Megakernel single-step (prompt={prompt!r}):")
    for backend in ("bf16", "nvfp4"):
        d = Decoder(model_name=model_name, backend=backend, verbose=False)
        d.reset()
        if backend == "bf16":
            first = d.prefill(pids)
        else:
            for t in pids[:-1]:
                d.step(int(t))
            first = d.step(int(pids[-1]))
        print(f"    {backend:>5} -> first token = {int(first)} "
              f"({tok.decode([int(first)])!r})")
        del d
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--group-size", type=int, default=32)
    args = ap.parse_args()

    step0_kernel_quant_diag(args.group_size)
    step1_lm_head_argmax_diag(args.model_name, args.prompt, args.group_size)
    step2_megakernel_step0(args.model_name, args.prompt)


if __name__ == "__main__":
    main()
