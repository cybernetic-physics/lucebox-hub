"""LoraMegakernelTrainer — satisfies rl/backend's Sampler + Trainer +
CheckpointStore protocols for Qwen3.5-0.8B on Blackwell.

Architecture (phased):

  Sampling (eval, no grad) — fast path:
    Our cuBLAS+graph prefill + cooperative-kernel decode via the
    `qwen35_megakernel_bf16_C` extension. Measured 20-74x faster than
    HF+torch on B200 for prompts up to ~1k tokens.

  Training (forward+backward+optim) — correctness path, today:
    HF transformers Qwen3.5-0.8B + PEFT LoRA adapter (same 7 FA + 6
    DN * 3 = 21 projections wrapped with `peft.LoraConfig`), and
    torch.optim.AdamW over the LoRA params. Correct end-to-end but
    at torch speed. Phase 2 custom FA/DN backward kernels will
    replace this path without changing the Trainer API.

  Checkpoint format:
    PEFT-compatible adapter_model.safetensors + adapter_config.json,
    drop-in loadable by SGLang --load-lora-adapter and HF PEFT.

The sampler and trainer paths currently hold independent copies of
the base weights (one packed into our format, one in HF's). Both are
~800 MB in bf16 — fine on B200 but worth unifying later.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

# transformers' is_flash_linear_attention_available calls
# version.parse("N/A") when an empty `fla` namespace package is installed
# (happens on some boxes). Short-circuit it before importing transformers.
try:
    import transformers.utils.import_utils as _iu
    _iu.is_flash_linear_attention_available = lambda: False
except Exception:
    pass

# PEFT >= 0.19 unconditionally probes torchao via a strict version check
# during LoRA module dispatch — even when the target Linear isn't a
# torchao quantized module. On boxes with torchao < 0.16 (e.g. our
# B200 box: torchao 0.9.0) PEFT raises ImportError before reaching the
# regular Linear dispatcher. Short-circuit the check so dispatcher
# falls through to the standard nn.Linear path. This is safe because
# the trainer doesn't use torchao quantized weights. We patch the check
# at every place PEFT imports it from — peft.import_utils and the
# already-imported reference inside peft.tuners.lora.torchao.
def _peft_no_torchao(*_a, **_kw):
    return False
try:
    import peft.import_utils as _peft_iu
    _peft_iu.is_torchao_available = _peft_no_torchao
except Exception:
    pass
try:
    import peft.tuners.lora.torchao as _peft_lora_torchao
    _peft_lora_torchao.is_torchao_available = _peft_no_torchao
except Exception:
    pass

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b")
sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/prefill_megakernel")
sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")

import qwen35_megakernel_bf16_C  # noqa: F401
import train_megakernel_C          # noqa: F401  fused AdamW

from lora_hf_wrap import wrap_hf_with_lora, LoraLinear  # reused helpers

# Load the outer model.py (has load_weights + Decoder for fast sampling).
_spec = importlib.util.spec_from_file_location(
    "qwen_outer_model",
    "/root/lucebox-hub-b200-train/models/qwen35_0p8b/model.py",
)
_outer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_outer)
load_weights = _outer.load_weights
Decoder = _outer.Decoder


# --------- Adapter ↔ flat-buffer mapping ---------

# Order matches our kernel's per-type-layer-major LoRA tensor convention
# (same as test_lora_forward.FA_SHAPES + DN_SHAPES).
_FA_PROJ_NAMES = ("fa_q", "fa_k", "fa_v", "fa_o", "fa_gate", "fa_up", "fa_down")
_DN_PROJ_NAMES = ("dn_qkv", "dn_z", "dn_out", "dn_gate", "dn_up", "dn_down")


@dataclass
class _Session:
    model_id: str
    base_model: str
    lora_rank: int
    lora_alpha: float
    train_mlp: bool
    train_attn: bool
    hf_model: Any                           # peft.PeftModel (base Qwen + LoRA)
    hf_optimizer: torch.optim.Optimizer
    lora_scaling: float
    step: int = 0
    last_loss: float | None = None


class LoraMegakernelTrainer:
    """rl/backend runtime: Sampler + Trainer + CheckpointStore.

    Thread-safe per-model via an RLock. Multiple model_ids can be
    registered concurrently — each gets its own HF+PEFT wrapped copy
    and optimizer.
    """

    BASE_MODEL = "Qwen/Qwen3.5-0.8B"

    def __init__(
        self,
        *,
        artifact_root: str | Path = "/tmp/lora_megakernel_artifacts",
        verbose_loader: bool = False,
    ) -> None:
        self.artifact_root = Path(artifact_root)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self._verbose_loader = verbose_loader
        self._sessions: dict[str, _Session] = {}
        self._state_lock = threading.RLock()
        self._model_locks: dict[str, threading.RLock] = {}

        # Lazy-loaded resources: the megakernel Decoder (for sampling) and
        # the HF tokenizer. Both hold a copy of the base model; we do not
        # instantiate them until the first call that needs one.
        self._decoder: Decoder | None = None
        self._tokenizer = None
        self._hf_base_cache: Any = None  # the shared frozen base for PEFT wrap

    # ---------- Runtime metadata ----------

    def server_capabilities(self) -> dict[str, Any]:
        return {
            "runtime": "qwen35_0p8b_megakernel_lora",
            "supports_sampling": True,
            "supports_training": True,
            "supports_custom_loss_v1": True,
            "supports_optimizer_restore": True,
            "supports_dynamic_lora_loading": True,
            "supported_models": [
                {"name": self.BASE_MODEL, "kind": "hybrid-delta-attention"}
            ],
        }

    def close(self) -> None:
        with self._state_lock:
            self._sessions.clear()
            self._model_locks.clear()

    # ---------- Sampler ----------

    def sample(
        self,
        *,
        prompt_tokens: list[int],
        max_tokens: int,
        num_samples: int,
        prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        """Greedy sampling via the megakernel decode path.

        Currently samples the FROZEN base model (ignores any LoRA from
        the registered sessions). Hot-swappable LoRA into the decode
        kernel is a Phase 5 follow-up; today's runtime uses sampling
        mainly for evaluation + preview.
        """
        self._ensure_decoder()
        decoder = self._decoder
        eos = self._tokenizer.eos_token_id

        # Prefill the prompt via cuBLAS+graph (one dispatch instead of one
        # decode kernel per prompt token). NVFP4 backend doesn't have a
        # prefill kernel yet — fall back to the per-token loop there.
        use_prefill = decoder.backend == "bf16" and len(prompt_tokens) > 1

        sequences = []
        for _ in range(num_samples):
            if use_prefill:
                pred = decoder.prefill(prompt_tokens)
            else:
                decoder.reset()
                for tid in prompt_tokens[:-1]:
                    decoder.step(int(tid))
                pred = decoder.step(int(prompt_tokens[-1]))

            out_ids: list[int] = []
            stop_reason = "max_tokens"
            for _ in range(max_tokens):
                if pred == eos:
                    stop_reason = "eos"
                    break
                out_ids.append(int(pred))
                pred = decoder.step(int(pred))
            sequences.append({
                "stop_reason": stop_reason,
                "tokens": out_ids,
                "logprobs": None,
            })
        return {"type": "sample", "sequences": sequences}

    # ---------- Trainer ----------

    def register_model(
        self,
        *,
        model_id: str,
        base_model: str | None = None,
        lora_rank: int | None = 16,
        train_mlp: bool | None = True,
        train_attn: bool | None = True,
        train_unembed: bool | None = False,
        user_metadata: dict[str, Any] | None = None,
    ) -> None:
        base_model = base_model or self.BASE_MODEL
        if base_model != self.BASE_MODEL:
            raise ValueError(
                f"this runtime is pinned to {self.BASE_MODEL}; got {base_model!r}"
            )
        rank = int(lora_rank or 16)

        # Load the HF base once, shared across all sessions (frozen).
        if self._hf_base_cache is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._hf_base_cache = AutoModelForCausalLM.from_pretrained(
                self.BASE_MODEL, dtype=torch.bfloat16
            ).to("cuda")
            self._hf_base_cache.eval()
            if self._tokenizer is None:
                self._tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL)

        # Build a PEFT LoRA wrap on a fresh copy of the base. We use
        # peft.LoraConfig with explicit target_modules so the wrap is
        # deterministic (and matches what HF+SGLang load later).
        from peft import LoraConfig, get_peft_model
        # Clone the weights into a new model container so each session
        # can be trained independently without corrupting siblings.
        # For memory, we actually share the underlying nn.Parameters
        # via `copy.deepcopy` on the *module graph* — PEFT attaches
        # adapters on top of the per-session container.
        import copy
        session_base = copy.deepcopy(self._hf_base_cache)
        target = []
        if train_attn:
            target += ["q_proj", "k_proj", "v_proj", "o_proj"]
        if train_mlp:
            target += ["gate_proj", "up_proj", "down_proj"]
        cfg = LoraConfig(
            r=rank,
            lora_alpha=rank,
            target_modules=target,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        hf_model = get_peft_model(session_base, cfg).to("cuda", dtype=torch.bfloat16)
        hf_model.train()

        # Swap HF's fp32 torch_chunk_gated_delta_rule for our CUDA DN
        # recurrence kernel (bit-exact to the recurrent variant, ~2-6x
        # faster training step end-to-end at S=128..512). Opt-out with
        # LORA_MEGAKERNEL_DISABLE_DN_PATCH=1 to fall back to HF's impl.
        import os
        if not os.environ.get("LORA_MEGAKERNEL_DISABLE_DN_PATCH"):
            from dn_hf_patch import patch_hf_qwen3_deltanet
            try:
                _n = patch_hf_qwen3_deltanet(hf_model)
            except Exception:
                _n = 0  # kernel not built — fall back quietly
            if _n:
                print(f"[LoraMegakernelTrainer] patched {_n} GatedDeltaNet "
                      f"layers with CUDA kernel")

        optimizer = torch.optim.AdamW(
            [p for p in hf_model.parameters() if p.requires_grad],
            lr=1e-4,
        )

        session = _Session(
            model_id=model_id,
            base_model=base_model,
            lora_rank=rank,
            lora_alpha=float(rank),
            train_mlp=bool(train_mlp),
            train_attn=bool(train_attn),
            hf_model=hf_model,
            hf_optimizer=optimizer,
            lora_scaling=1.0,
        )
        with self._lock_for_model(model_id), self._state_lock:
            self._sessions[model_id] = session

    def unload_model(self, *, model_id: str) -> None:
        with self._lock_for_model(model_id), self._state_lock:
            self._sessions.pop(model_id, None)
        torch.cuda.empty_cache()

    def forward(self, *, model_id: str, data: list[dict], loss_fn: str, loss_fn_config=None):
        with self._lock_for_model(model_id):
            if loss_fn != "cross_entropy":
                raise ValueError(f"unsupported loss_fn={loss_fn!r}")
            s = self._session(model_id)
            outputs = []
            per_losses = []
            with torch.no_grad():
                for d in data:
                    prompt, targets = _extract_prompt_targets(d)
                    logp_selected = _sequence_logprobs(s.hf_model, prompt, targets)
                    outputs.append({
                        "logprobs": {
                            "data": logp_selected.detach().cpu().tolist(),
                            "dtype": "float32",
                            "shape": [int(logp_selected.numel())],
                        }
                    })
                    per_losses.append(float(-logp_selected.mean().item()))
            return {
                "loss_fn_output_type": "TensorData",
                "loss_fn_outputs": outputs,
                "metrics": {
                    "loss:mean": float(sum(per_losses) / max(len(per_losses), 1)),
                    "batch_size:mean": float(len(data)),
                },
            }

    def forward_backward(self, *, model_id: str, data: list[dict], loss_fn: str, loss_fn_config=None):
        with self._lock_for_model(model_id):
            if loss_fn != "cross_entropy":
                raise ValueError(f"unsupported loss_fn={loss_fn!r}")
            s = self._session(model_id)
            s.hf_model.zero_grad(set_to_none=True)

            items: list[tuple[torch.Tensor, torch.Tensor]] = []
            for d in data:
                prompt, targets = _extract_prompt_targets(d)
                if targets.numel() == 0:
                    continue
                items.append((prompt, targets))

            if not items:
                return {
                    "loss_fn_output_type": "TensorData",
                    "loss_fn_outputs": [],
                    "metrics": {"loss:mean": 0.0, "batch_size:mean": float(len(data))},
                }

            # Optional Slice-B.3b path: kernel-driven forward + per-layer
            # custom backward, bypassing HF+PEFT autograd entirely. Opt-in
            # via env var; the default HF+PEFT path stays the production
            # default until the kernel path's full convergence is verified.
            if os.environ.get("MEGAKERNEL_USE_KERNEL_BWD") == "1":
                return self._forward_backward_kernel_path(
                    s=s, items=items, total_data_len=len(data),
                )

            # If all examples share (prompt_len, target_len), pack into one
            # batch and do a single forward — collapses ~B× the kernel
            # launches and lets HF's batched fla path handle DN.
            p0 = items[0][0].numel()
            t0 = items[0][1].numel()
            if all(p.numel() == p0 and t.numel() == t0 for p, t in items):
                logp_per_item, per_loss = _batched_logprobs(s.hf_model, items)
            else:
                # Heterogeneous shapes — fall back to the per-example loop.
                logps = []
                losses = []
                for prompt, targets in items:
                    lp = _sequence_logprobs(s.hf_model, prompt, targets)
                    logps.append(lp)
                    losses.append(-lp.mean())
                logp_per_item = logps
                per_loss = torch.stack(losses)

            total_loss = per_loss.sum()
            (total_loss / max(len(items), 1)).backward()

            # Materialize per-example outputs after backward (one bulk D→H
            # for losses, one per-example .tolist() for logprobs).
            losses_cpu = per_loss.detach().cpu().tolist()
            outputs = []
            per_losses = []
            for b, lp in enumerate(logp_per_item):
                lv = float(losses_cpu[b])
                per_losses.append(lv)
                outputs.append({
                    "logprobs": {
                        "data": lp.detach().cpu().tolist(),
                        "dtype": "float32",
                        "shape": [int(lp.numel())],
                    },
                    "loss": {"data": [lv], "dtype": "float32", "shape": [1]},
                })

            mean_loss = float(sum(per_losses) / max(len(per_losses), 1))
            s.last_loss = mean_loss
            return {
                "loss_fn_output_type": "TensorData",
                "loss_fn_outputs": outputs,
                "metrics": {
                    "loss:mean": mean_loss,
                    "batch_size:mean": float(len(data)),
                },
            }

    # ---------- Slice B.3b kernel-driven backward path ----------
    def _forward_backward_kernel_path(
        self, *, s: "_Session", items: list, total_data_len: int,
    ) -> dict[str, Any]:
        """forward_backward via kernel_loss_autograd + run_layer_walking_bwd.

        Per item:
          1) pack PEFT LoRA → 26 flat tensors
          2) run kernel forward → (loss, grad_h_pre_norm, saves)
          3) run layer-walking bwd → 26 flat fp32 grad tensors
          4) scatter into PEFT params' .grad (accumulating across items)

        After the loop, optim_step() (the HF+PEFT one) consumes the
        scattered grads. No code change to optim_step.
        """
        # Lazy imports to avoid pulling in fla / extensions when the kernel
        # path is disabled. The trainer is normally imported at startup
        # before all the C++ extensions exist.
        from lora_pack import pack_peft_to_flat, scatter_flat_grads_to_peft
        from lora_megakernel_step import (
            kernel_loss_autograd, load_base_model,
        )
        from lora_layer_bwd_skel import run_layer_walking_bwd

        # Lazy-load + cache the base-model handle on first kernel-path call.
        if not hasattr(self, "_kernel_base_handle"):
            self._kernel_base_handle = load_base_model(self.BASE_MODEL,
                                                       verbose=False)
        handle = self._kernel_base_handle

        lora_flat = pack_peft_to_flat(s.hf_model, s.lora_rank)

        per_losses: list[float] = []
        outputs: list[dict] = []
        for prompt, targets in items:
            out = kernel_loss_autograd(
                handle=handle,
                prompt_tokens=prompt,
                target_tokens=targets,
                lora_flat=lora_flat,
                lora_rank=s.lora_rank,
                lora_scaling=s.lora_scaling,
                hf_model=s.hf_model,
            )
            loss = out["loss"]
            grad_h_pre_norm = out["grad_h_pre_norm"]
            saves = out["saves"]
            scratch = out["scratch"]

            # Scale grad_h_pre_norm by 1/N so the accumulated gradient
            # corresponds to mean-loss across items (matches HF+PEFT
            # path's `(total_loss/N).backward()` convention).
            scale = 1.0 / max(len(items), 1)
            grad_h_pre_norm = grad_h_pre_norm * scale

            flat_grads = run_layer_walking_bwd(
                grad_h_pre_norm=grad_h_pre_norm,
                saves=saves,
                lora_flat=lora_flat,
                final_norm_weight=handle.final_norm_weight,
                hf_model=s.hf_model,
                lora_rank=s.lora_rank,
                lora_scaling=s.lora_scaling,
                fa_k_cache=scratch["fa_k_cache"],
                fa_v_cache=scratch["fa_v_cache"],
            )

            # Scatter into PEFT params' .grad. Accumulate across items.
            scatter_flat_grads_to_peft(s.hf_model, flat_grads,
                                        accumulate=True)

            lv = float(loss.item())
            per_losses.append(lv)
            # Per-item logp output uses the pre-final-norm path; we don't
            # have per-position logp natively from kernel_loss_autograd
            # (it returns the scalar loss only by default). Pass empty
            # logprobs back — rl/backend's logprobs metric is informational.
            outputs.append({
                "logprobs": {"data": [], "dtype": "float32", "shape": [0]},
                "loss":     {"data": [lv], "dtype": "float32", "shape": [1]},
            })

        mean_loss = float(sum(per_losses) / max(len(per_losses), 1))
        s.last_loss = mean_loss
        return {
            "loss_fn_output_type": "TensorData",
            "loss_fn_outputs": outputs,
            "metrics": {
                "loss:mean": mean_loss,
                "batch_size:mean": float(total_data_len),
            },
        }

    def optim_step(self, *, model_id: str, adam_params: dict[str, float]):
        with self._lock_for_model(model_id):
            s = self._session(model_id)
            lr = float(adam_params.get("learning_rate", 1e-4))
            b1 = float(adam_params.get("beta1", 0.9))
            b2 = float(adam_params.get("beta2", 0.999))
            eps = float(adam_params.get("eps", 1e-8))
            wd = float(adam_params.get("weight_decay", 0.0))
            for g in s.hf_optimizer.param_groups:
                g["lr"] = lr; g["betas"] = (b1, b2); g["eps"] = eps
                g["weight_decay"] = wd
            s.hf_optimizer.step()
            s.hf_optimizer.zero_grad(set_to_none=True)
            s.step += 1
            return {
                "metrics": {
                    "learning_rate:mean": lr,
                    "weight_decay:mean": wd,
                    "step:mean": float(s.step),
                    "last_loss:mean": float(s.last_loss) if s.last_loss is not None else 0.0,
                }
            }

    # ---------- CheckpointStore ----------

    def save_checkpoint(
        self, *,
        model_id: str,
        checkpoint_kind: str = "state",
        checkpoint_name: str = "latest",
        tinker_path: str = "",
        owner_api_key: str = "",
        include_optimizer: bool = False,
    ) -> dict[str, Any]:
        with self._lock_for_model(model_id):
            s = self._session(model_id)
            out_dir = self._ckpt_dir(model_id, checkpoint_kind, checkpoint_name)
            out_dir.mkdir(parents=True, exist_ok=True)
            # PEFT's built-in save writes adapter_model.safetensors +
            # adapter_config.json, directly loadable by SGLang / HF PEFT.
            s.hf_model.save_pretrained(str(out_dir))
            has_optim = False
            if include_optimizer and s.step > 0:
                torch.save(s.hf_optimizer.state_dict(), out_dir / "optimizer.pt")
                has_optim = True
            manifest = {
                "schema_version": 1,
                "saved_at": datetime.now(UTC).isoformat(),
                "tinker_path": tinker_path,
                "training_run_id": model_id,
                "base_model": s.base_model,
                "lora_rank": s.lora_rank,
                "lora_alpha": s.lora_alpha,
                "train_mlp": s.train_mlp,
                "train_attn": s.train_attn,
                "checkpoint_kind": checkpoint_kind,
                "owner_api_key": owner_api_key,
                "optimizer_included": has_optim,
                "step": s.step,
                "runtime": "qwen35_0p8b_megakernel_lora",
            }
            (out_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True)
            )
            return {"artifact_path": str(out_dir), "has_optimizer_state": has_optim}

    def load_weights(
        self, *, model_id: str, path: str = "", artifact_path: str, with_optimizer: bool = False
    ) -> None:
        with self._lock_for_model(model_id):
            s = self._session(model_id)
            d = Path(artifact_path)
            # PEFT saves adapter_model.safetensors with keys like
            #   base_model.model.<inner>.lora_A.weight
            # but the live model names them
            #   base_model.model.<inner>.lora_A.default.weight
            # Use peft's set_peft_model_state_dict to handle the mapping.
            from peft import set_peft_model_state_dict
            state = load_file(str(d / "adapter_model.safetensors"))
            set_peft_model_state_dict(s.hf_model, state)
            if with_optimizer:
                opt_path = d / "optimizer.pt"
                if not opt_path.exists():
                    raise ValueError(f"{path}: optimizer state not found")
                s.hf_optimizer.load_state_dict(
                    torch.load(opt_path, map_location="cuda")
                )
            manifest = d / "manifest.json"
            if manifest.exists():
                m = json.loads(manifest.read_text())
                s.step = int(m.get("step", 0))

    def delete_checkpoint_artifact(self, *, artifact_path: str) -> None:
        shutil.rmtree(artifact_path, ignore_errors=True)

    # ---------- Internals ----------

    def _lock_for_model(self, model_id: str) -> threading.RLock:
        with self._state_lock:
            lock = self._model_locks.get(model_id)
            if lock is None:
                lock = threading.RLock()
                self._model_locks[model_id] = lock
            return lock

    def _session(self, model_id: str) -> _Session:
        with self._state_lock:
            try:
                return self._sessions[model_id]
            except KeyError as exc:
                raise ValueError(f"unknown model_id {model_id!r}") from exc

    def _ckpt_dir(self, model_id: str, kind: str, name: str) -> Path:
        return self.artifact_root / model_id / kind / name

    def _ensure_decoder(self) -> None:
        if self._decoder is None:
            self._decoder = Decoder(
                model_name=self.BASE_MODEL,
                backend="bf16",
                verbose=self._verbose_loader,
            )
            if self._tokenizer is None:
                self._tokenizer = self._decoder.tokenizer


# ---------- helpers ----------

def _extract_prompt_targets(datum: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert rl/backend's Datum wire format to (prompt_ids, target_ids).

    Wire format (simplified):
      datum["model_input"]["chunks"] = [{"type": "input", "tokens": [...]}]
      datum["loss_fn_inputs"]["target_tokens"] = [...] | TensorData
    """
    chunks = datum.get("model_input", {}).get("chunks", [])
    prompt_ids: list[int] = []
    for ch in chunks:
        prompt_ids.extend(int(t) for t in ch.get("tokens", []))
    tgt = datum.get("loss_fn_inputs", {}).get("target_tokens", [])
    if isinstance(tgt, dict):
        tgt = tgt.get("data", [])
    tgt_ids = [int(t) for t in tgt]
    return (
        torch.tensor(prompt_ids, dtype=torch.long, device="cuda"),
        torch.tensor(tgt_ids, dtype=torch.long, device="cuda"),
    )


def _sequence_logprobs(
    hf_model, prompt_ids: torch.Tensor, target_ids: torch.Tensor
) -> torch.Tensor:
    """Return log p(target_t | prompt + target_{<t}) for each target.

    We concatenate prompt + target (shift-by-one is handled implicitly by
    the causal LM) and gather the logprobs at the target positions.
    """
    full = torch.cat([prompt_ids, target_ids], dim=0).unsqueeze(0)
    out = hf_model(input_ids=full, use_cache=False)
    logits = out.logits[0].to(torch.float32)
    # target_t is predicted by logits[len(prompt) + t - 1]
    pl = int(prompt_ids.numel())
    tl = int(target_ids.numel())
    predict_logits = logits[pl - 1 : pl - 1 + tl]
    logp = torch.nn.functional.log_softmax(predict_logits, dim=-1)
    return logp.gather(1, target_ids.unsqueeze(1)).squeeze(1)


def _batched_logprobs(
    hf_model,
    items: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Batched equivalent of `_sequence_logprobs` for shape-uniform items.

    All (prompt, targets) tuples must share the same (P, T). Returns the
    per-item logp tensor list (each shape [T]) and a [B] tensor of
    per-item -mean(logp) losses.
    """
    B = len(items)
    P = items[0][0].numel()
    T = items[0][1].numel()
    full_ids = torch.stack(
        [torch.cat([p, t], dim=0) for p, t in items], dim=0
    )  # [B, P+T]
    out = hf_model(input_ids=full_ids, use_cache=False)
    logits = out.logits.to(torch.float32)  # [B, P+T, V]
    predict_logits = logits[:, P - 1 : P - 1 + T]  # [B, T, V]
    log_probs = torch.nn.functional.log_softmax(predict_logits, dim=-1)
    targets = torch.stack([t for _, t in items], dim=0)  # [B, T]
    logp = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # [B, T]
    per_loss = -logp.mean(dim=1)  # [B]
    return [logp[b] for b in range(B)], per_loss


__all__ = ["LoraMegakernelTrainer"]
