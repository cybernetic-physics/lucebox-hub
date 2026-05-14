"""Python runtime that drives the Qwen3.x templated megakernel.

End-to-end inference path:
    HF weights -> weight_packer -> packed layer blob
    scratch + KV cache allocated via weight_packer.alloc_scratch
    prefill via torch.ops.qwen3x_C.prefill_qwen3x_naive
    decode  via torch.ops.qwen3x_C.decode_qwen3x

Same API surface as `runtime_hf.Qwen36Runtime` so the OpenAI server +
correctness harness can swap one for the other.

Validation status (as of commit time):
  - Build: both Cfg specializations compile + link for sm_121a.
  - MLP smoke: cos=1.000 vs torch reference for both Cfg shapes.
  - Weight key mapping: validated against the real Qwen3.6-27B
    safetensors index (1199 keys, 16 FA + 48 DN layers).
  - End-to-end ours-vs-HF logits: needs the running HF model side by
    side; first run is gated on the test_correctness_vs_hf.py loop.

Known follow-ups (in order of likely surface area):
  - The cooperative-grid launch may need a different grid-size heuristic
    on GB10 (default 64-96 blocks works for sm_121a; tune later).
  - DN V/QK split: algorithm follows the fla / Qwen3 spec but real-
    weight drift may still surface on the first forward — primary
    suspects are beta/alpha activation order and the per-V-head
    `a_log` indexing.
  - The HF chat template wires through to apply_chat_template — same
    code path as runtime_hf.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(THIS_DIR, "megakernel"))

# Load extension by path so users don't need to install qwen3x_C.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "qwen3x_C",
    os.path.join(THIS_DIR, "megakernel",
                 "qwen3x_C.cpython-312-aarch64-linux-gnu.so"),
)
qwen3x_C = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(qwen3x_C)

from weight_packer import (
    load_27b_weights, pack_layer_weights, alloc_scratch,
    NUM_LAYERS, HIDDEN_SIZE, VOCAB_SIZE,
)

# The native context length for Qwen3.6-27B is 262144. YaRN is *off*
# by default per the config.json's `rope_type: "default"`.
DEFAULT_YARN = dict(
    scale=1.0, beta_fast=32.0, beta_slow=1.0,
    orig_ctx=262144, enabled=False,
)


@dataclass
class MegakernelGenConfig:
    max_tokens: int = 128
    temperature: float = 0.0    # greedy only for now
    stop: list[int] = None      # token ids; if any sampled, break


class Qwen36MegakernelDecoder:
    """Loads HF Qwen/Qwen3.6-27B weights, packs them, and serves
    prefill + decode through `torch.ops.qwen3x_C`."""

    MODEL_ID = 1  # Cfg_27B in the kernel

    def __init__(self,
                 model_name: str = "Qwen/Qwen3.6-27B",
                 max_seq: int = 32768,
                 yarn: dict | None = None,
                 num_blocks: int = 0,
                 verbose: bool = True,
                 hf_model=None,        # pre-loaded HF model to share weights
                 tokenizer=None):
        self.max_seq = max_seq
        self.yarn = yarn or DEFAULT_YARN
        self.num_blocks = num_blocks
        self.position = 0
        self.verbose = verbose

        if hf_model is not None:
            if verbose: print("[megakernel] using pre-loaded HF model "
                              "(sharing weight tensors)", flush=True)
            # Extract the same weights structure load_27b_weights would build,
            # but from the existing HF model — no extra allocation.
            from weight_packer import _unify_from_hf_model
            self.weights = _unify_from_hf_model(hf_model)
            self.tokenizer = tokenizer
            if self.tokenizer is None:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=True)
        else:
            if verbose: print(f"[megakernel] loading HF weights from {model_name}", flush=True)
            t0 = time.perf_counter()
            self.weights, self.tokenizer = load_27b_weights(model_name, verbose=verbose)
            if verbose: print(f"[megakernel]   ...{time.perf_counter()-t0:.1f}s", flush=True)

        if verbose: print("[megakernel] packing layer pointers...", flush=True)
        self.layer_blob = pack_layer_weights(self.weights["layer_data"])

        if verbose: print(f"[megakernel] allocating scratch (max_seq={max_seq})...", flush=True)
        self.sc = alloc_scratch(max_seq=max_seq)

        if verbose:
            gpu = torch.cuda.memory_allocated() / (1024 ** 3)
            print(f"[megakernel] ready. GPU alloc: {gpu:.1f} GB")

    def reset(self):
        self.position = 0
        self.sc.fa_k_cache.zero_()
        self.sc.fa_v_cache.zero_()
        self.sc.dn_states.zero_()
        self.sc.conv_bufs.zero_()

    def prefill(self, prompt_ids: torch.Tensor) -> int:
        """Run prefill over `prompt_ids` (CPU/CUDA long or int32 1-D).
        Updates the position counter; returns the next-token argmax."""
        if not prompt_ids.is_cuda:
            prompt_ids = prompt_ids.cuda()
        if prompt_ids.dtype != torch.int32:
            prompt_ids = prompt_ids.to(torch.int32)
        prompt_ids = prompt_ids.contiguous()

        ops = torch.ops.qwen3x_C
        ops.prefill_qwen3x_naive(
            self.MODEL_ID, prompt_ids,
            self.weights["embed_weight"], self.weights["final_norm_weight"],
            self.layer_blob,
            self.sc.fa_k_cache, self.sc.fa_v_cache,
            self.sc.dn_states, self.sc.conv_bufs,
            self.sc.hidden_buffer, self.sc.g_residual,
            self.sc.g_qkv_scratch, self.sc.g_kv_scratch,
            self.sc.g_attn_out, self.sc.g_mlp_inter,
            self.sc.g_z_scratch, self.sc.g_beta_scratch, self.sc.g_alpha_scratch,
            self.sc.g_normalized, self.sc.g_fa_partials, self.sc.g_rope_inv_freq,
            self.max_seq,
            float(self.yarn["scale"]), float(self.yarn["beta_fast"]),
            float(self.yarn["beta_slow"]),
            int(self.yarn["orig_ctx"]), bool(self.yarn["enabled"]),
            int(self.num_blocks),
        )
        self.position = int(prompt_ids.numel())
        return self._argmax_from_normalized()

    def decode(self, token_id: int) -> int:
        """One decode step. Advances position by 1."""
        ops = torch.ops.qwen3x_C
        ops.decode_qwen3x(
            self.MODEL_ID,
            self.weights["embed_weight"], self.weights["final_norm_weight"],
            self.layer_blob,
            self.sc.fa_k_cache, self.sc.fa_v_cache,
            self.sc.dn_states, self.sc.conv_bufs,
            self.sc.hidden_buffer, self.sc.g_residual,
            self.sc.g_qkv_scratch, self.sc.g_kv_scratch,
            self.sc.g_attn_out, self.sc.g_mlp_inter,
            self.sc.g_z_scratch, self.sc.g_beta_scratch, self.sc.g_alpha_scratch,
            self.sc.g_normalized, self.sc.g_fa_partials, self.sc.g_rope_inv_freq,
            int(token_id), self.position, 0, 0, self.max_seq,
            float(self.yarn["scale"]), float(self.yarn["beta_fast"]),
            float(self.yarn["beta_slow"]),
            int(self.yarn["orig_ctx"]), bool(self.yarn["enabled"]),
            int(self.num_blocks),
        )
        self.position += 1
        return self._argmax_from_normalized()

    def _argmax_from_normalized(self) -> int:
        """LM head argmax over g_normalized @ lm_head.T. fp32 matmul."""
        hidden = self.sc.g_normalized.to(torch.float32)
        lm_head = self.weights["lm_head_weight"].to(torch.float32)
        # logits shape: [VOCAB]
        logits = hidden @ lm_head.t()
        return int(logits.argmax().item())

    def logits_for_last(self) -> torch.Tensor:
        """Return fp32 [VOCAB] logits at the last-processed position."""
        hidden = self.sc.g_normalized.to(torch.float32)
        lm_head = self.weights["lm_head_weight"].to(torch.float32)
        return hidden @ lm_head.t()


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of the United States is")
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--max-seq", type=int, default=512)
    args = ap.parse_args()

    dec = Qwen36MegakernelDecoder(max_seq=args.max_seq)
    print(f"[smoke] prompt: {args.prompt!r}")
    ids = dec.tokenizer(args.prompt, return_tensors="pt").input_ids[0]
    next_id = dec.prefill(ids)
    out = [next_id]
    for _ in range(args.max_tokens - 1):
        next_id = dec.decode(next_id)
        out.append(next_id)
    text = dec.tokenizer.decode(out, skip_special_tokens=True)
    print(f"[smoke] generated: {text!r}")
