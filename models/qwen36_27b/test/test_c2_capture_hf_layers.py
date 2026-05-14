"""C2 — capture per-layer hidden states from HF Qwen3.6-27B as the
correctness oracle for layer-by-layer comparison against our kernel.

Saves a .pt file with:
  meta:     prompt token id, prompt text, model name
  layer_0:  hidden state OUT of layer 0   [HIDDEN] fp32
  layer_1:  hidden state OUT of layer 1   [HIDDEN] fp32
  ...
  layer_63: hidden state OUT of layer 63  [HIDDEN] fp32
  final_normed: post final-RMSnorm hidden  [HIDDEN] fp32
  logits:   final logits                   [VOCAB]  fp32

For diff: load this golden, run our kernel with debug capture enabled,
diff per-layer.

Run:
    HF_HOME=/home/sparkz/rl/.hf_cache \\
        /home/sparkz/rl/.venv/bin/python3 \\
        models/qwen36_27b/test/test_c2_capture_hf_layers.py \\
        --out models/qwen36_27b/reference/hf_per_layer.pt
"""
from __future__ import annotations

import argparse
import os, sys, time
from pathlib import Path
import torch

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default=" ",
                    help="single-token (or short) prompt; default a space")
    ap.add_argument("--max-prompt-tokens", type=int, default=1,
                    help="cap prompt length")
    args = ap.parse_args()

    print(f"Loading HF Qwen3.6-27B...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True,
    ).eval()
    print(f"  loaded; GPU alloc={torch.cuda.memory_allocated()/(1024**3):.1f} GB")

    ids = tok(args.prompt, return_tensors="pt").input_ids[0][: args.max_prompt_tokens]
    print(f"\nPrompt: {args.prompt!r}   token_ids: {ids.tolist()}")
    ids = ids.unsqueeze(0).to("cuda")

    print(f"\nRunning forward with output_hidden_states=True...")
    with torch.no_grad():
        out = hf(input_ids=ids, use_cache=False, output_hidden_states=True)

    # `out.hidden_states` is a tuple of (NUM_LAYERS + 1) tensors:
    #   index 0 = embedding output (input to layer 0)
    #   index i+1 = output of layer i
    # We capture from index 1 onward (the layer outputs).
    hs = out.hidden_states
    print(f"  got {len(hs)} hidden-state captures (embedding + 64 layer outputs)")

    payload = {
        "meta": {
            "model_name": "Qwen/Qwen3.6-27B",
            "prompt": args.prompt,
            "prompt_token_ids": ids[0].cpu().tolist(),
            "torch_version": torch.__version__,
        },
        "embedding_out": hs[0][0, -1].to(torch.float32).cpu(),
    }
    for i in range(len(hs) - 1):
        payload[f"layer_{i}_out"] = hs[i + 1][0, -1].to(torch.float32).cpu()

    # Also save the final norm output (post the final RMSnorm before LM head).
    # HF doesn't expose it directly, but we can replicate:
    # final_hidden = last hs; final_norm = model.model.norm(final_hidden)
    final_hidden = hs[-1]
    final_normed = hf.model.norm(final_hidden) if hasattr(hf.model, "norm") else None
    if final_normed is None and hasattr(hf.model, "language_model"):
        # Multimodal wrap.
        final_normed = hf.model.language_model.norm(final_hidden)
    payload["final_normed"] = final_normed[0, -1].to(torch.float32).cpu()
    payload["logits"] = out.logits[0, -1].to(torch.float32).cpu()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(f"\nSaved -> {out_path}  ({out_path.stat().st_size / (1024**2):.1f} MB)")
    print(f"Top-1 token id: {int(payload['logits'].argmax().item())} "
          f"({tok.decode([int(payload['logits'].argmax().item())])!r})")


if __name__ == "__main__":
    main()
