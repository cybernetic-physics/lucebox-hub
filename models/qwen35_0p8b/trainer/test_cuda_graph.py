"""Validate CUDA Graphs wrap of the HF training step.

Both the inline replay path and the GraphedTrainStep helper. Reports
eager vs graphed ms/step at multiple S, with the speedup.
"""
from __future__ import annotations

import sys
import time

import transformers.utils.import_utils as _iu
_iu.is_flash_linear_attention_available = lambda: False

import torch

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")
from dn_hf_patch import patch_hf_qwen3_deltanet


def time_step_eager(model, ids):
    """Eager (no graph) fwd+bwd."""
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    model.zero_grad(set_to_none=True)


def time_step_graphed(graphed_fn, static_ids, real_ids):
    """Graphed fwd+bwd: copy real ids into static_ids in-place, replay graph."""
    static_ids.copy_(real_ids)
    graphed_fn()


def main():
    torch.manual_seed(0)
    print("Loading Qwen3.5-0.8B...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to("cuda")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")

    cfg = LoraConfig(r=16, lora_alpha=16,
                     target_modules=["q_proj","k_proj","v_proj","o_proj"],
                     lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(base, cfg).to("cuda", dtype=torch.bfloat16)
    model.train()
    n = patch_hf_qwen3_deltanet(model)
    print(f"Patched {n} GatedDeltaNet layers")

    for S in (128, 256, 512):
        ids = torch.randint(0, tok.vocab_size, (1, S), device="cuda", dtype=torch.long)

        # Eager baseline
        for _ in range(2): time_step_eager(model, ids)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5): time_step_eager(model, ids)
        torch.cuda.synchronize()
        eager_ms = (time.perf_counter() - t0) * 1000.0 / 5

        # CUDA-graph capture + replay
        # Use a STATIC input tensor; copy data into it before each replay.
        static_ids = ids.clone()
        # Warmup outside graph (PyTorch requirement: capture must follow
        # at least 3 warmup steps on a side stream).
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                out = model(input_ids=static_ids, labels=static_ids)
                out.loss.backward()
                model.zero_grad(set_to_none=True)
        torch.cuda.current_stream().wait_stream(s)

        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = model(input_ids=static_ids, labels=static_ids)
                out.loss.backward()
                model.zero_grad(set_to_none=True)
        except Exception as e:
            print(f"  S={S}: graph capture failed: {str(e)[:80]}")
            continue

        # Time graph replay.
        for _ in range(2):
            static_ids.copy_(ids); g.replay()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            static_ids.copy_(ids); g.replay()
        torch.cuda.synchronize()
        graph_ms = (time.perf_counter() - t0) * 1000.0 / 5

        print(f"  S={S}: eager {eager_ms:6.1f} ms   graph {graph_ms:6.1f} ms   "
              f"speedup {eager_ms/graph_ms:.2f}x")

    # Also exercise the production helper.
    print()
    print("GraphedTrainStep helper:")
    from cuda_graph_train import GraphedTrainStep
    for S in (128, 256, 512):
        ids = torch.randint(0, tok.vocab_size, (1, S), device="cuda", dtype=torch.long)
        runner = GraphedTrainStep(model, batch_size=1, seq_len=S)
        for _ in range(2): runner.step(ids)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5): runner.step(ids)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0 / 5
        loss = runner.step(ids).item()
        print(f"  S={S}: helper.step() {ms:6.1f} ms/iter  (last loss={loss:.3f})")


if __name__ == "__main__":
    main()
