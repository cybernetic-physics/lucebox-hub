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
    prefill + decode through `torch.ops.qwen3x_C`.

    `backend` selects the kernel variant:
      "bf16"  -> MODEL_ID = 1  (BF16 weights, ~50 GB)
      "nvfp4" -> MODEL_ID = 3  (NVFP4 weights, ~14 GB; runs the
                                _nvfp4 layer functions and dispatches
                                to launch_decode_27b_nvfp4).
    """

    def __init__(self,
                 model_name: str = "Qwen/Qwen3.6-27B",
                 max_seq: int = 32768,
                 yarn: dict | None = None,
                 num_blocks: int = 0,
                 verbose: bool = True,
                 hf_model=None,        # pre-loaded HF model to share weights
                 tokenizer=None,
                 backend: str = "bf16"):
        self.max_seq = max_seq
        self.yarn = yarn or DEFAULT_YARN
        self.num_blocks = num_blocks
        self.position = 0
        self.verbose = verbose
        self.backend = backend
        if backend == "bf16":
            self.MODEL_ID = 1
        elif backend == "nvfp4":
            self.MODEL_ID = 3
        else:
            raise ValueError(f"backend must be 'bf16' or 'nvfp4', got {backend!r}")

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

        if backend == "nvfp4":
            if verbose: print("[megakernel] quantizing layer weights to NVFP4...",
                              flush=True)
            from nvfp4_27b import quantize_27b_weights
            quantized = quantize_27b_weights(self.weights["layer_data"], verbose=verbose)
            # Remap type: 0 (DN_bf16) -> 2 (DN_nvfp4); 1 (FA_bf16) -> 3 (FA_nvfp4).
            for ld in quantized:
                ld["type"] = int(ld["type"]) + 2
            self.weights["layer_data"] = quantized

        if verbose: print("[megakernel] packing layer pointers...", flush=True)
        self.layer_blob = pack_layer_weights(self.weights["layer_data"])

        if verbose: print(f"[megakernel] allocating scratch (max_seq={max_seq})...", flush=True)
        self.sc = alloc_scratch(max_seq=max_seq, verbose=verbose)

        # Optional layer-by-layer hidden-state capture buffer. Allocated on
        # demand via enable_layer_capture(). When set, decode_qwen3x writes
        # each layer's `hidden_buffer` into this [NUM_LAYERS, HIDDEN] slab.
        self.layer_capture: torch.Tensor | None = None

        if verbose:
            gpu = torch.cuda.memory_allocated() / (1024 ** 3)
            print(f"[megakernel] ready. GPU alloc: {gpu:.1f} GB")

    def enable_layer_capture(self):
        """Allocate the [NUM_LAYERS, HIDDEN] capture buffer."""
        self.layer_capture = torch.zeros(NUM_LAYERS, HIDDEN_SIZE,
                                          dtype=torch.bfloat16, device="cuda")

    def reset(self):
        self.position = 0
        self.sc.fa_k_cache.zero_()
        self.sc.fa_v_cache.zero_()
        self.sc.dn_states.zero_()
        self.sc.conv_bufs.zero_()

    def prefill(self, prompt_ids: torch.Tensor, start_position: int = 0) -> int:
        """Run prefill over `prompt_ids` (CPU/CUDA long or int32 1-D).
        Sets the position counter to `start_position + len(prompt_ids)`;
        returns the next-token argmax.

        For multi-turn KV cache reuse, pass `start_position=self.position`
        and only the *new* tokens in `prompt_ids` — the cached prefix
        stays untouched (F2)."""
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
            self.layer_capture,
            int(start_position),
        )
        self.position = start_position + int(prompt_ids.numel())
        return self._argmax_from_normalized()

    def prefill_via_hf(self, prompt_ids: torch.Tensor) -> int:
        """Fast prefill using the HF model's batched forward.

        Runs the HF Qwen3.6-27B forward with `use_cache=True` on the
        full prompt (cuBLAS-batched matmuls — ~500 ms at S=256 vs our
        host-loop prefill's 54 s). Then copies HF's KV / DN state
        into our scratch so subsequent `decode()` calls continue
        seamlessly via the megakernel.

        Requires that this decoder was built with `hf_model=hf` so
        weights are shared. Returns the next-token argmax.
        """
        hf = self.weights.get("_hf_keepalive")
        if hf is None:
            raise RuntimeError(
                "prefill_via_hf requires the runtime to have been built "
                "with `hf_model=hf` so HF weights stay loaded.")

        if not prompt_ids.is_cuda:
            prompt_ids = prompt_ids.cuda()
        if prompt_ids.dtype != torch.long:
            prompt_ids = prompt_ids.to(torch.long)
        S = prompt_ids.numel()
        if S > self.max_seq:
            raise ValueError(f"prefill length {S} > max_seq {self.max_seq}")

        with torch.no_grad():
            out = hf(input_ids=prompt_ids.unsqueeze(0), use_cache=True)

        # Walk HF's past_key_values + cache_params (the Qwen3.6 cache
        # exposes FA K/V via past_key_values and DN state via
        # cache_params; both are on `out.past_key_values` for this
        # HF release — verified empirically against modeling_qwen3_5.py).
        cache = out.past_key_values
        # FA mapping: cache.layers[layer_idx] -> our fa_k_cache[fa_idx]
        from weight_packer import LAYER_TYPE, N_FA, N_DN
        fa_idx = 0; dn_idx = 0
        for layer_idx in range(len(LAYER_TYPE)):
            cl = cache.layers[layer_idx]
            if LAYER_TYPE[layer_idx] == 1:    # FA layer
                # HF K/V: [1, num_kv_heads, S, head_dim]
                k = cl.keys   if hasattr(cl, "keys")   else cl.key_cache
                v = cl.values if hasattr(cl, "values") else cl.value_cache
                # Squeeze batch dim and copy into our [KV_H, max_seq, head_dim]
                self.sc.fa_k_cache[fa_idx, :, :S, :].copy_(
                    k[0].to(torch.bfloat16))
                self.sc.fa_v_cache[fa_idx, :, :S, :].copy_(
                    v[0].to(torch.bfloat16))
                fa_idx += 1
            else:                              # DN layer
                conv = getattr(cl, "conv_states", None)
                rec  = getattr(cl, "recurrent_states", None)
                if conv is not None:
                    # HF: [1, CONV_CH, CONV_K] -> ours: [CONV_CH, CONV_K]
                    self.sc.conv_bufs[dn_idx].copy_(conv[0].to(torch.float32))
                if rec is not None:
                    # HF: [1, V_H, k_head_dim, v_head_dim] = [V_H, KEY, VAL]
                    # Ours: state[j * KEY + i], i.e. [V_H, VAL, KEY] —
                    # transpose the last two dims (KEY = VAL = 128 so the
                    # shape matches without transpose, but the data layout
                    # would be wrong → top-1 mismatch in the very next
                    # decode step). Don't drop this transpose.
                    self.sc.dn_states[dn_idx].copy_(
                        rec[0].transpose(-1, -2).contiguous().to(torch.float32))
                dn_idx += 1

        self.position = S
        # Fill g_normalized + argmax for the next token from HF's last
        # logits so subsequent decode() calls start from a consistent state.
        last_logits = out.logits[0, -1].to(torch.float32)
        return int(last_logits.argmax().item())

    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int,
                  eos_id: int | None = None,
                  use_hf_prefill: bool | None = None) -> torch.Tensor:
        """Sequential prefill + decode wrapper. EOS-aware.

        If `use_hf_prefill=True`, uses the HF model's batched forward
        for prefill (~50× faster at S=256) and our megakernel for
        decode. Requires `hf_model=hf` was passed at init time.

        Default OFF: prefill_via_hf's KV/DN-state copy from HF's cache
        has a known correctness issue — first decoded token matches
        HF, but subsequent decodes diverge (out-of-range token ids).
        Likely a remaining DN cache-layout mismatch beyond the
        transpose fix in commit 8e8d225. Opt-in only until F9
        regression test catches the exact layout difference.
        """
        if use_hf_prefill is None:
            use_hf_prefill = False  # see docstring
        if use_hf_prefill:
            next_id = self.prefill_via_hf(prompt_ids)
        else:
            next_id = self.prefill(prompt_ids)
        new_tokens = [next_id]
        if eos_id is not None and next_id == eos_id:
            return torch.tensor(new_tokens, dtype=torch.int32)
        for _ in range(max_new_tokens - 1):
            next_id = self.decode(next_id)
            new_tokens.append(next_id)
            if eos_id is not None and next_id == eos_id: break
        return torch.tensor(new_tokens, dtype=torch.int32)

    def generate_sample(self, prompt_ids: torch.Tensor, max_new_tokens: int,
                         temperature: float = 1.0,
                         top_k: int = 0, top_p: float = 1.0,
                         eos_id: int | None = None,
                         seed: int | None = None) -> torch.Tensor:
        """Prefill + sampling decode. Use temperature=0 for greedy.
        Returns the int32 list of generated token ids."""
        rng = None
        if seed is not None:
            rng = torch.Generator(device="cuda").manual_seed(seed)
        next_id = self.prefill(prompt_ids)
        if temperature > 0.0 or top_k > 0 or top_p < 1.0:
            # Replace the prefill's argmax with a sampled first token by
            # running sample() on the existing g_normalized. We DON'T
            # need to re-run the decode kernel; prefill already left
            # g_normalized populated.
            logits = self.logits_for_last()
            if temperature > 0.0 and temperature != 1.0:
                logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k); cutoff = v[-1]
                logits = torch.where(logits < cutoff,
                                       torch.full_like(logits, -float("inf")), logits)
            if top_p < 1.0:
                sl, si = torch.sort(logits, descending=True)
                cp = torch.cumsum(torch.softmax(sl, dim=-1), dim=-1)
                km = cp <= top_p; km[0] = True
                mask = torch.zeros_like(logits, dtype=torch.bool)
                mask.scatter_(0, si[km], True)
                logits = torch.where(mask, logits, torch.full_like(logits, -float("inf")))
            probs = torch.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probs, 1, generator=rng).item())
        new_tokens = [next_id]
        if eos_id is not None and next_id == eos_id:
            return torch.tensor(new_tokens, dtype=torch.int32)
        for _ in range(max_new_tokens - 1):
            next_id = self.sample(next_id, temperature=temperature,
                                    top_k=top_k, top_p=top_p, rng=rng)
            new_tokens.append(next_id)
            if eos_id is not None and next_id == eos_id: break
        return torch.tensor(new_tokens, dtype=torch.int32)

    def decode_chain(self, n_tokens: int) -> torch.Tensor:
        """Async decode that keeps the next-token id on-device between
        steps. After a prefill, calls decode_qwen3x N times consecutively
        without any host syncs — the LM head argmax writes the token id
        into self._lm_head_scratch["out"] (int32 device tensor), and the
        next decode_qwen3x reads it from there.

        Returns a CPU int32 tensor of the N decoded token ids. Only one
        host sync at the end (the final .cpu() copy).

        Saves ~50 us of host-sync overhead per token, but more
        importantly is CUDA-Graph-capturable (subsequent work).
        """
        # First touch primes _lm_head_scratch (via decode then ignore).
        if not hasattr(self, "_lm_head_scratch"):
            _ = self._argmax_from_normalized()
        s = self._lm_head_scratch
        ops = torch.ops.qwen3x_C
        device_token = s["out"]  # int32 [1], on device

        # Collect on-device per-step output by capturing each into a slot
        # of a host-staged buffer. Avoid item() per iteration.
        out_history = torch.empty(n_tokens, dtype=torch.int32, device="cuda")

        for i in range(n_tokens):
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
                -1, self.position, self.position, self.position,
                self.max_seq,
                float(self.yarn["scale"]), float(self.yarn["beta_fast"]),
                float(self.yarn["beta_slow"]),
                int(self.yarn["orig_ctx"]), bool(self.yarn["enabled"]),
                int(self.num_blocks),
                self.layer_capture,
                device_token,                  # NEW: read token from device
            )
            ops.lm_head_argmax(
                self.MODEL_ID, self.sc.g_normalized,
                self.weights["lm_head_weight"],
                device_token, s["block_max_vals"], s["block_max_idxs"],
                s["num_blocks"])
            # Copy device_token into our history buffer (still no host sync).
            out_history[i].copy_(device_token[0])
            self.position += 1
        return out_history.cpu()

    def _run_decode_kernel(self, token_id: int) -> None:
        """Run the megakernel decode for one token, advance position.
        Leaves g_normalized populated with the post-final-norm hidden
        state; caller chooses argmax vs sample."""
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
            int(token_id), self.position, self.position, self.position,
            self.max_seq,
            float(self.yarn["scale"]), float(self.yarn["beta_fast"]),
            float(self.yarn["beta_slow"]),
            int(self.yarn["orig_ctx"]), bool(self.yarn["enabled"]),
            int(self.num_blocks),
            self.layer_capture,
        )
        self.position += 1

    def sample(self, token_id: int, temperature: float = 1.0,
                top_k: int = 0, top_p: float = 1.0,
                rng: torch.Generator | None = None) -> int:
        """One decode step + sample from the resulting logits.

        Args:
          token_id: previous token (input to this step).
          temperature: logit scale; 0 falls back to greedy argmax.
          top_k:  keep only top-K logits; 0 = no filter.
          top_p:  nucleus filter; 1.0 = no filter.
          rng:    optional torch.Generator for determinism.

        Returns the sampled token id. Advances position by 1.
        """
        self._run_decode_kernel(token_id)
        if temperature <= 0.0 and top_k == 0 and top_p >= 1.0:
            return self._argmax_from_normalized()
        # Compute fp32 logits via the slow path (fast kernel only does
        # argmax; full-logits requires the matmul). Cost is ~50 ms;
        # negligible vs sampling tail latency on a chat workload.
        logits = self.logits_for_last()
        if temperature > 0.0 and temperature != 1.0:
            logits = logits / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            cutoff = v[-1]
            logits = torch.where(logits < cutoff,
                                   torch.full_like(logits, -float("inf")),
                                   logits)
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            keep_mask = cumulative <= top_p
            keep_mask[0] = True   # always keep top-1 even if its prob > p
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask.scatter_(0, sorted_idx[keep_mask], True)
            logits = torch.where(mask, logits, torch.full_like(logits, -float("inf")))
        probs = torch.softmax(logits, dim=-1)
        next_id = int(torch.multinomial(probs, num_samples=1, generator=rng).item())
        return next_id

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
            # For text-only inference HF's MRoPE position_ids has all 3 axes
            # set to the temporal position. Passing 0 for pos_h/pos_w would
            # make the height/width sections of the rotary_dim static, which
            # is NOT equivalent to standard RoPE.
            int(token_id), self.position, self.position, self.position,
            self.max_seq,
            float(self.yarn["scale"]), float(self.yarn["beta_fast"]),
            float(self.yarn["beta_slow"]),
            int(self.yarn["orig_ctx"]), bool(self.yarn["enabled"]),
            int(self.num_blocks),
            self.layer_capture,
        )
        self.position += 1
        return self._argmax_from_normalized()

    def _argmax_from_normalized(self) -> int:
        """LM head argmax over g_normalized @ lm_head.T.

        Uses the fast bf16 kernel `lm_head_argmax` whenever lm_head is
        BF16 (which it always is — only the layer projections get
        quantized, not the LM head embedding). The kernel's model_id
        only selects HIDDEN/VOCAB shape via Cfg dispatch (model_id 0
        for 0.8B, 1 for 27B); NVFP4 backend (model_id=3) shares the
        Cfg_27B shape, so we map both 1 and 3 to the 27B kernel.
        """
        lm_head = self.weights["lm_head_weight"]
        if (lm_head.dtype == torch.bfloat16
                and not getattr(self, "_force_slow_argmax", False)):
            if not hasattr(self, "_lm_head_scratch"):
                # 2 blocks per SM (each 256 threads × 20 KB shmem ≈ 40 KB
                # per SM, well under the 102 KB limit). Empirically the
                # difference between 32/48/96 blocks is <1 % (the kernel
                # is HBM-bound at ~238 GB/s reading the 2.5 GB lm_head),
                # but 2/SM gives the best occupancy headroom.
                sm_count = torch.cuda.get_device_properties(0).multi_processor_count
                num_blocks = sm_count * 2
                self._lm_head_scratch = dict(
                    num_blocks=num_blocks,
                    out=torch.zeros(1, dtype=torch.int32, device="cuda"),
                    block_max_vals=torch.zeros(num_blocks, dtype=torch.float32, device="cuda"),
                    block_max_idxs=torch.zeros(num_blocks, dtype=torch.int32, device="cuda"),
                )
            s = self._lm_head_scratch
            # lm_head_argmax is keyed on (HIDDEN, VOCAB) shape only; 27B
            # bf16 (MODEL_ID=1) and 27B nvfp4 (MODEL_ID=3) share dims,
            # so always pass the bf16-kernel model id.
            lm_model_id = 1 if self.MODEL_ID in (1, 3) else 0
            torch.ops.qwen3x_C.lm_head_argmax(
                lm_model_id, self.sc.g_normalized, lm_head,
                s["out"], s["block_max_vals"], s["block_max_idxs"],
                s["num_blocks"])
            return int(s["out"].item())

        # Fallback: fp32 matmul.
        hidden = self.sc.g_normalized.to(torch.float32)
        lm_head_f32 = lm_head.to(torch.float32)
        logits = hidden @ lm_head_f32.t()
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
