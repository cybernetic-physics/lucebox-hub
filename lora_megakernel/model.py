"""Python wrapper for the fused LoRA training megakernel.

`LoRATrainStepKernel` owns GPU-resident weights plus Adam state and exposes a
single `train_step(context_tokens, target_tokens, weights)` entrypoint that
delegates to one fused CUDA dispatch doing forward + backward + AdamW.

Shape and init conventions mirror the `TinyLoRAModel` in the rl/backend
tinker_backend runtime so callers can swap this in without touching the
training loop.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

import torch


def _seed_from_name(name: str) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def _load_op():
    import lora_megakernel_C  # noqa: F401
    return torch.ops.lora_megakernel_C.lora_train_step


@dataclass
class AdamConfig:
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0


@dataclass
class LoRATrainStepKernel:
    """Persistent LoRA training state backed by one fused CUDA kernel.

    `train_step` runs the complete forward + backward + AdamW update as a
    single cooperative grid launch. No torch autograd involvement.
    """

    base_model: str
    vocab_size: int
    hidden_size: int
    lora_rank: int
    max_seq_len: int
    device: torch.device
    _step: int = 0

    embedding: torch.Tensor = field(init=False)
    output_weight: torch.Tensor = field(init=False)
    lora_a: torch.Tensor = field(init=False)
    lora_b: torch.Tensor = field(init=False)

    m_a: torch.Tensor = field(init=False)
    v_a: torch.Tensor = field(init=False)
    m_b: torch.Tensor = field(init=False)
    v_b: torch.Tensor = field(init=False)

    hidden_ws: torch.Tensor = field(init=False)
    lora_h_ws: torch.Tensor = field(init=False)
    logits_ws: torch.Tensor = field(init=False)
    grad_logits_ws: torch.Tensor = field(init=False)
    grad_lora_a_ws: torch.Tensor = field(init=False)
    grad_lora_b_ws: torch.Tensor = field(init=False)
    grad_lora_h_ws: torch.Tensor = field(init=False)
    selected_ws: torch.Tensor = field(init=False)
    loss_ws: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        dev = self.device
        V, H, R, T_max = self.vocab_size, self.hidden_size, self.lora_rank, self.max_seq_len

        # Deterministic init matching TinyLoRAModel's seed scheme.
        gen = torch.Generator(device="cpu").manual_seed(_seed_from_name(self.base_model))
        emb_cpu = torch.randn(V, H, generator=gen) * 0.05
        out_cpu = torch.randn(V, H, generator=gen) * 0.05
        a_cpu = torch.randn(H, R, generator=gen) * 0.01
        b_cpu = torch.zeros(R, V)

        self.embedding = emb_cpu.to(device=dev, dtype=torch.bfloat16).contiguous()
        self.output_weight = out_cpu.to(device=dev, dtype=torch.bfloat16).contiguous()
        self.lora_a = a_cpu.to(device=dev, dtype=torch.bfloat16).contiguous()
        self.lora_b = b_cpu.to(device=dev, dtype=torch.bfloat16).contiguous()

        self.m_a = torch.zeros(H, R, device=dev, dtype=torch.float32)
        self.v_a = torch.zeros(H, R, device=dev, dtype=torch.float32)
        self.m_b = torch.zeros(R, V, device=dev, dtype=torch.float32)
        self.v_b = torch.zeros(R, V, device=dev, dtype=torch.float32)

        self.hidden_ws = torch.empty(T_max, H, device=dev, dtype=torch.bfloat16)
        self.lora_h_ws = torch.empty(T_max, R, device=dev, dtype=torch.float32)
        self.logits_ws = torch.empty(T_max, V, device=dev, dtype=torch.float32)
        self.grad_logits_ws = torch.empty(T_max, V, device=dev, dtype=torch.float32)
        self.grad_lora_a_ws = torch.empty(H, R, device=dev, dtype=torch.float32)
        self.grad_lora_b_ws = torch.empty(R, V, device=dev, dtype=torch.float32)
        self.grad_lora_h_ws = torch.empty(T_max, R, device=dev, dtype=torch.float32)
        self.selected_ws = torch.empty(T_max, device=dev, dtype=torch.float32)
        self.loss_ws = torch.zeros(1, device=dev, dtype=torch.float32)

    @property
    def step(self) -> int:
        return self._step

    @classmethod
    def from_base_model(
        cls,
        *,
        base_model: str,
        lora_rank: int,
        vocab_size: int,
        hidden_size: int,
        max_seq_len: int = 128,
        device: Optional[torch.device] = None,
    ) -> "LoRATrainStepKernel":
        dev = device or torch.device("cuda")
        return cls(
            base_model=base_model,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            lora_rank=lora_rank,
            max_seq_len=max_seq_len,
            device=dev,
        )

    def train_step(
        self,
        *,
        context_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        token_weights: Optional[torch.Tensor] = None,
        adam: AdamConfig = AdamConfig(),
        do_update: bool = True,
    ) -> dict:
        """Run one fused forward+backward+AdamW step. Returns loss and per-token log-probs."""
        assert context_tokens.dim() == 1 and target_tokens.dim() == 1
        T = int(context_tokens.shape[0])
        assert target_tokens.shape[0] == T, "context and target must have equal length"
        assert T <= self.max_seq_len, f"T={T} exceeds max_seq_len={self.max_seq_len}"

        dev = self.device
        ctx = context_tokens.to(device=dev, dtype=torch.int32).contiguous()
        tgt = target_tokens.to(device=dev, dtype=torch.int32).contiguous()
        if token_weights is None:
            w = torch.ones(T, device=dev, dtype=torch.float32)
        else:
            assert token_weights.shape[0] == T
            w = token_weights.to(device=dev, dtype=torch.float32).contiguous()

        next_step = self._step + 1 if do_update else max(self._step, 1)
        bc1 = 1.0 - (adam.beta1 ** next_step)
        bc2 = 1.0 - (adam.beta2 ** next_step)

        op = _load_op()
        op(
            ctx, tgt, w,
            self.embedding, self.output_weight,
            self.lora_a, self.lora_b,
            self.m_a, self.v_a, self.m_b, self.v_b,
            self.hidden_ws, self.lora_h_ws,
            self.logits_ws, self.grad_logits_ws,
            self.grad_lora_a_ws, self.grad_lora_b_ws, self.grad_lora_h_ws,
            self.selected_ws, self.loss_ws,
            self.vocab_size, self.lora_rank,
            adam.lr, adam.beta1, adam.beta2, adam.eps, adam.weight_decay,
            bc1, bc2,
            1 if do_update else 0,
        )

        if do_update:
            self._step += 1

        return {
            "loss": float(self.loss_ws.item()),
            "selected_log_probs": self.selected_ws[:T].detach().clone(),
        }

    def state_dict(self) -> dict:
        return {
            "lora_a": self.lora_a.detach().cpu(),
            "lora_b": self.lora_b.detach().cpu(),
            "m_a": self.m_a.detach().cpu(),
            "v_a": self.v_a.detach().cpu(),
            "m_b": self.m_b.detach().cpu(),
            "v_b": self.v_b.detach().cpu(),
            "step": self._step,
        }

    def load_state_dict(self, state: dict) -> None:
        self.lora_a.copy_(state["lora_a"].to(self.device, dtype=torch.bfloat16))
        self.lora_b.copy_(state["lora_b"].to(self.device, dtype=torch.bfloat16))
        self.m_a.copy_(state["m_a"].to(self.device, dtype=torch.float32))
        self.v_a.copy_(state["v_a"].to(self.device, dtype=torch.float32))
        self.m_b.copy_(state["m_b"].to(self.device, dtype=torch.float32))
        self.v_b.copy_(state["v_b"].to(self.device, dtype=torch.float32))
        self._step = int(state.get("step", 0))
