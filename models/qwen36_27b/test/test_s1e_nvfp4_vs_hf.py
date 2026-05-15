"""S1e — NVFP4 weight path correctness vs HF (BF16) reference.

Quantize the loaded Qwen3.6-27B BF16 weights to NVFP4 (optimal-MSE
group=32) and run the same prefill through both paths:

  - BF16 megakernel (model_id=1)
  - NVFP4 megakernel (model_id=3)
  - HF reference (BF16) for absolute ground truth

Per-prompt: top-1 match, top-5 overlap, cos sim, KL divergence.

Acceptance bar (per 0.8B sweep): top-1 match on natural text; cos
between BF16 and NVFP4 paths ≥ 0.995 on short prompts; PPL drift on
wikitext ≤ 1%.

Run:
    HF_HOME=/home/sparkz/rl/.hf_cache \
        /home/sparkz/rl/.venv/bin/python3 \
        models/qwen36_27b/test/test_s1e_nvfp4_vs_hf.py

NOTE: quantization runs in PyTorch and takes a few minutes per
projection. Expect ~30-60 minutes per layer of quantization on first
run. Cache the quantized state if iterating.
"""
from __future__ import annotations
import os, sys, time
import torch
import torch.nn.functional as F

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))


def _eval_logits(name, ref, ours):
    cos = F.cosine_similarity(ref.unsqueeze(0).to(torch.float32),
                                ours.unsqueeze(0).to(torch.float32),
                                dim=-1).item()
    r1 = int(ref.argmax().item())
    o1 = int(ours.argmax().item())
    top5_ref = set(ref.topk(5).indices.tolist())
    top5_ours = set(ours.topk(5).indices.tolist())
    top5_ov = len(top5_ref & top5_ours) / 5
    lp_r = F.log_softmax(ref.to(torch.float32), dim=-1)
    lp_o = F.log_softmax(ours.to(torch.float32), dim=-1)
    kl = float((lp_r.exp() * (lp_r - lp_o)).sum().item())
    match = "YES" if r1 == o1 else "NO "
    print(f"  {name:>14}  cos={cos:.4f}  KL={kl:.4f}  top1={r1}/{o1} {match}  "
          f"top5_ov={top5_ov:.2f}")
    return r1 == o1, cos


def main():
    print("Loading HF Qwen3.6-27B (BF16)...")
    t0 = time.perf_counter()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True,
    ).eval()
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    from runtime_megakernel import Qwen36MegakernelDecoder
    print("\nBuilding BF16 megakernel decoder (sharing HF weights)...")
    bf16_dec = Qwen36MegakernelDecoder(max_seq=128, verbose=True,
                                         hf_model=hf, tokenizer=tok,
                                         backend="bf16")

    print("\nBuilding NVFP4 megakernel decoder "
          "(quantizing layer weights — this can take 30+ min)...")
    t1 = time.perf_counter()
    nvfp4_dec = Qwen36MegakernelDecoder(max_seq=128, verbose=True,
                                         hf_model=hf, tokenizer=tok,
                                         backend="nvfp4")
    print(f"  quantization done in {time.perf_counter()-t1:.1f}s")

    prompts = [
        "The capital of France is",
        "Hello, my name is",
        "1+1 =",
    ]
    n_pass = 0
    for prompt in prompts:
        print(f"\nprompt: {prompt!r}")
        ids = tok(prompt, return_tensors="pt").input_ids[0].to(torch.int32).cuda()

        with torch.no_grad():
            hf_out = hf(input_ids=ids.unsqueeze(0).to(torch.long), use_cache=False)
        hf_logits = hf_out.logits[0, -1].cpu()

        bf16_dec.reset(); bf16_dec.prefill(ids); bf16_logits = bf16_dec.logits_for_last().cpu()
        nvfp4_dec.reset(); nvfp4_dec.prefill(ids); nvfp4_logits = nvfp4_dec.logits_for_last().cpu()

        ok_bf16, _ = _eval_logits("BF16 vs HF", hf_logits, bf16_logits)
        ok_nvfp4, _ = _eval_logits("NVFP4 vs HF", hf_logits, nvfp4_logits)
        _eval_logits("NVFP4 vs BF16", bf16_logits, nvfp4_logits)

        if ok_nvfp4: n_pass += 1

    print(f"\nPASS NVFP4-vs-HF on {n_pass}/{len(prompts)} prompts")
    if n_pass != len(prompts):
        sys.exit(1)


if __name__ == "__main__":
    main()
