"""C3 — layer-by-layer diff: ours vs HF on a single-token forward.

Workflow:
  1. Load HF Qwen3.6-27B once; share its weight tensors with our kernel
     (no double allocation; total stays at ~50 GB).
  2. HF forward with `output_hidden_states=True`; capture per-layer hidden
     states for our chosen prompt token.
  3. Build Qwen36MegakernelDecoder with the same model; enable per-layer
     capture; run a single decode step at position 0.
  4. For each of the 64 layers, compute cos sim + max abs diff between
     our captured hidden and HF's. Find the FIRST layer where cos < 0.99.
  5. Report that layer + whether it's FA or DN.

The first-divergent layer's type narrows down the bug:
  - DN layer drift starting at layer 0 -> DN forward (most likely V/QK
    GQA indexing or beta/alpha activation order)
  - FA layer drift starting at layer 3 -> FA forward (most likely
    QK-norm placement or RoPE)
  - Drift starting later -> a propagation bug (NaN-tolerant primitive
    or a stale buffer)
"""
from __future__ import annotations

import os, sys, time
import torch
import torch.nn.functional as F

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))

PROMPT_TOKEN_ID = 220  # " " — same single-token prompt as C1


def main():
    print("Loading HF Qwen3.6-27B (single load, shared between paths)...",
          flush=True)
    t0 = time.perf_counter()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True,
    ).eval()
    print(f"  loaded in {time.perf_counter()-t0:.1f}s; "
          f"GPU={torch.cuda.memory_allocated()/(1024**3):.1f} GB")

    # ---- HF reference forward with per-layer capture ----
    ids = torch.tensor([[PROMPT_TOKEN_ID]], dtype=torch.long, device="cuda")
    print(f"\nHF forward with output_hidden_states for token {PROMPT_TOKEN_ID} "
          f"({tok.decode([PROMPT_TOKEN_ID])!r}):")
    with torch.no_grad():
        out = hf(input_ids=ids, use_cache=False, output_hidden_states=True)
    hs = out.hidden_states   # tuple of NUM_LAYERS+1 tensors
    print(f"  captured {len(hs)} hidden states (embedding + 64 layer outputs)")
    ref_per_layer = [h[0, -1].to(torch.float32).cpu() for h in hs[1:]]
    ref_logits = out.logits[0, -1].to(torch.float32).cpu()
    ref_top1 = int(ref_logits.argmax().item())

    # ---- Megakernel forward with capture ----
    print(f"\nBuilding megakernel decoder with shared weights...")
    from runtime_megakernel import Qwen36MegakernelDecoder
    dec = Qwen36MegakernelDecoder(max_seq=128, verbose=True, hf_model=hf, tokenizer=tok)
    dec.enable_layer_capture()
    print(f"  layer_capture buffer: {dec.layer_capture.shape}")

    print(f"\nRunning megakernel decode at position=0 with capture on...")
    next_id = dec.decode(PROMPT_TOKEN_ID)
    ours_logits = dec.logits_for_last().cpu()
    ours_per_layer = [dec.layer_capture[i].to(torch.float32).cpu()
                       for i in range(64)]

    # ---- Diff per layer ----
    print(f"\nPer-layer hidden-state diff (HF reference vs ours):")
    print(f"  {'layer':>5}  {'type':>4}  {'cos':>9}  {'max_abs':>10}  "
          f"{'ref|max|':>9}  {'ours|max|':>10}  {'finite?':>7}")
    first_divergent = None
    for i in range(64):
        layer_type = "FA" if (i + 1) % 4 == 0 else "DN"
        r = ref_per_layer[i]
        u = ours_per_layer[i]
        cos = F.cosine_similarity(r.unsqueeze(0), u.unsqueeze(0), dim=-1).item()
        max_abs = (r - u).abs().max().item()
        finite = bool(torch.isfinite(u).all().item())
        print(f"  {i:>5}  {layer_type:>4}  {cos:>9.6f}  {max_abs:>10.4f}  "
              f"{r.abs().max().item():>9.3f}  {u.abs().max().item():>10.3f}  "
              f"{str(finite):>7}")
        if first_divergent is None and (cos < 0.99 or not finite):
            first_divergent = (i, layer_type)

    # ---- Final logit diff ----
    cos_logits = F.cosine_similarity(
        ref_logits.unsqueeze(0), ours_logits.unsqueeze(0), dim=-1).item()
    print(f"\nFinal logits:  cos={cos_logits:.6f}  "
          f"HF top1={ref_top1} ({tok.decode([ref_top1])!r}) "
          f"ours top1={int(ours_logits.argmax().item())} "
          f"({tok.decode([int(ours_logits.argmax().item())])!r})")

    if first_divergent is None:
        print(f"\nALL 64 LAYERS AGREE — drift must be in final RMSnorm or LM head")
    else:
        idx, kind = first_divergent
        print(f"\nFIRST DIVERGENCE: layer {idx} ({kind})")
        if kind == "DN" and idx == 0:
            print("  -> DN layer 0 is the first thing that runs after the "
                  "embedding lookup. Most likely the DN V/QK GQA "
                  "indexing (V_PER_QK=3 maps each V head to a QK head; "
                  "verify the slicing in dn_layer.cuh matches HF's "
                  "Qwen3_5GatedDeltaNet.forward).")
        elif kind == "FA":
            print("  -> FA layer divergence. Suspects: QK-norm placement, "
                  "RoPE indexing, or attention output gate sigmoid.")
        else:
            print(f"  -> Drift accumulates from layer {idx-1} onward. "
                  "Re-check the previous layer's output.")


if __name__ == "__main__":
    main()
