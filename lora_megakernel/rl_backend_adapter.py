"""Adapter: plug the LoRA training megakernel into rl/backend's LoRATrainer.

rl/backend's `tinker_backend/runtimes/lora_trainer.py` exposes three calls:

    forward(data)            -> log-probs (no grad)
    forward_backward(data)   -> log-probs + populates .grad on lora_a/lora_b
    optim_step(adam_params)  -> reads .grad, AdamW update

This adapter presents the same shape while doing the actual work with one
fused CUDA dispatch. It defers the kernel launch until `optim_step` so that
the whole forward+backward+AdamW chain is a single cooperative grid launch.

To wire it in, replace the `TinyLoRAModel` + `torch.optim.AdamW` pair
inside `_TrainingSession` with a `MegakernelSession` instance and route
the three `LoRATrainer` methods to it.

The adapter speaks the same `Datum` dict shape as the rest of the
runtime, so no changes to wire/Datum types are needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from .model import AdamConfig, LoRATrainStepKernel


def _prompt_tokens(datum: dict) -> list[int]:
    chunks = datum.get("model_input", {}).get("chunks", [])
    toks: list[int] = []
    for ch in chunks:
        toks.extend(ch.get("tokens", []))
    return toks


def _target_tokens(datum: dict, vocab_size: int) -> list[int]:
    t: Any = datum.get("loss_fn_inputs", {}).get("target_tokens")
    if isinstance(t, dict):
        data = t.get("data") or []
    elif isinstance(t, list):
        data = t
    else:
        data = []
    return [int(x) % vocab_size for x in data]


def _weights(datum: dict, target_len: int) -> list[float]:
    w: Any = datum.get("loss_fn_inputs", {}).get("weights")
    if isinstance(w, dict):
        data = w.get("data") or []
    elif isinstance(w, list):
        data = w
    else:
        data = []
    if not data:
        return [1.0] * target_len
    return [float(x) for x in data]


@dataclass
class _PendingStep:
    ctx: torch.Tensor
    tgt: torch.Tensor
    w: torch.Tensor
    selected: Optional[torch.Tensor] = None
    loss: Optional[float] = None


class MegakernelSession:
    """Replacement for TinyLoRAModel + torch.optim.AdamW for one training run.

    Keeps deferred per-datum state until `optim_step` flushes a fused kernel
    call for each accumulated datum.
    """

    def __init__(
        self,
        *,
        base_model: str,
        vocab_size: int,
        hidden_size: int,
        lora_rank: int,
        max_seq_len: int = 128,
        device: Optional[torch.device] = None,
    ) -> None:
        self.kernel = LoRATrainStepKernel.from_base_model(
            base_model=base_model,
            lora_rank=lora_rank,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_seq_len=max_seq_len,
            device=device or torch.device("cuda"),
        )
        self.vocab_size = vocab_size
        self._pending: list[_PendingStep] = []

    @property
    def step(self) -> int:
        return self.kernel.step

    @property
    def lora_a(self) -> torch.Tensor:
        return self.kernel.lora_a

    @property
    def lora_b(self) -> torch.Tensor:
        return self.kernel.lora_b

    def _context_for(self, prompt_tokens: list[int], target_len: int) -> list[int]:
        if not prompt_tokens:
            prompt_tokens = [0]
        return [prompt_tokens[-1]] * max(target_len, 1)

    def forward(self, data: list[dict]) -> list[dict]:
        """Forward only: run kernel with do_update=0, capture log-probs, discard grads."""
        out_rows: list[dict] = []
        for datum in data:
            tgt = _target_tokens(datum, self.vocab_size)
            if not tgt:
                out_rows.append({"logprobs": {"data": [], "dtype": "float32", "shape": [0]}})
                continue
            ctx = self._context_for(_prompt_tokens(datum), len(tgt))
            w = _weights(datum, len(tgt))
            ctx_t = torch.tensor(ctx, dtype=torch.int32, device=self.kernel.device)
            tgt_t = torch.tensor(tgt, dtype=torch.int32, device=self.kernel.device)
            w_t = torch.tensor(w, dtype=torch.float32, device=self.kernel.device)
            out = self.kernel.train_step(
                context_tokens=ctx_t, target_tokens=tgt_t, token_weights=w_t,
                adam=AdamConfig(), do_update=False,
            )
            out_rows.append({
                "logprobs": {
                    "data": out["selected_log_probs"].cpu().tolist(),
                    "dtype": "float32",
                    "shape": [len(tgt)],
                }
            })
        return out_rows

    def forward_backward(self, data: list[dict]) -> tuple[list[dict], float]:
        """Stage inputs; the fused kernel runs in `optim_step`."""
        out_rows: list[dict] = []
        self._pending = []
        total_loss = 0.0
        for datum in data:
            tgt = _target_tokens(datum, self.vocab_size)
            if not tgt:
                out_rows.append({"logprobs": {"data": [], "dtype": "float32", "shape": [0]},
                                 "loss": {"data": [0.0], "dtype": "float32", "shape": [1]}})
                continue
            ctx = self._context_for(_prompt_tokens(datum), len(tgt))
            w = _weights(datum, len(tgt))
            ctx_t = torch.tensor(ctx, dtype=torch.int32, device=self.kernel.device)
            tgt_t = torch.tensor(tgt, dtype=torch.int32, device=self.kernel.device)
            w_t = torch.tensor(w, dtype=torch.float32, device=self.kernel.device)

            # Eager forward through kernel with do_update=0 so we can report
            # the forward stats now; gradients persist in kernel workspace.
            out = self.kernel.train_step(
                context_tokens=ctx_t, target_tokens=tgt_t, token_weights=w_t,
                adam=AdamConfig(), do_update=False,
            )
            self._pending.append(_PendingStep(ctx_t, tgt_t, w_t,
                                              out["selected_log_probs"].clone(), out["loss"]))
            total_loss += out["loss"]
            out_rows.append({
                "logprobs": {
                    "data": out["selected_log_probs"].cpu().tolist(),
                    "dtype": "float32",
                    "shape": [len(tgt)],
                },
                "loss": {"data": [out["loss"]], "dtype": "float32", "shape": [1]},
            })
        mean_loss = total_loss / max(len(data), 1)
        return out_rows, mean_loss

    def optim_step(self, adam: AdamConfig) -> None:
        """Re-run each pending datum with do_update=1 so AdamW fires in-kernel."""
        if not self._pending:
            return
        last = self._pending[-1]
        # One fused launch per pending datum — keeps grads fresh per step.
        for pending in self._pending:
            self.kernel.train_step(
                context_tokens=pending.ctx,
                target_tokens=pending.tgt,
                token_weights=pending.w,
                adam=adam,
                do_update=True,
            )
        self._pending = []
        _ = last  # keep last alive until after loop

    def export_lora(self) -> dict[str, torch.Tensor]:
        return {
            "lora_a": self.kernel.lora_a.detach().cpu(),
            "lora_b": self.kernel.lora_b.detach().cpu(),
        }

    def load_lora(self, state: dict[str, torch.Tensor]) -> None:
        self.kernel.lora_a.copy_(state["lora_a"].to(self.kernel.device, dtype=torch.bfloat16))
        self.kernel.lora_b.copy_(state["lora_b"].to(self.kernel.device, dtype=torch.bfloat16))
