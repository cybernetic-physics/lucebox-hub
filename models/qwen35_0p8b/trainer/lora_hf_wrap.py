"""Shared utilities for benchmarking our cuBLAS+graph LoRA forward
against an HF-transformers Qwen3.5-0.8B with the same 13 LoRA adapters.

We use the same A/B tensor layout as our kernel (`test_lora_forward.py`
`build_lora_tensors`) on both sides. Absolute output tokens won't match
because random LoRA init perturbs the base model — we care about
per-step wall-time and (for training) loss decrease per step, not
logit-bit-match.
"""
from __future__ import annotations

import sys
import torch
import torch.nn as nn

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b")
sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")
sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/prefill_megakernel")

from model import LAYER_TYPE


class LoraLinear(nn.Module):
    """Wraps an nn.Linear with a rank-R LoRA adapter.

    Forward:  y = base(x) + scaling * (x @ A) @ B
    Shapes:   x:[*, K_in], A:[K_in, R], B:[R, K_out], base.weight:[K_out, K_in]

    The base linear is frozen (`requires_grad_(False)` on its params);
    A and B are trainable.
    """

    def __init__(self, base: nn.Linear, A: torch.Tensor, B: torch.Tensor,
                 scaling: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        assert A.dim() == 2 and B.dim() == 2
        assert A.shape[1] == B.shape[0], f"rank mismatch: A={A.shape} B={B.shape}"
        self.A = nn.Parameter(A.clone().contiguous())
        self.B = nn.Parameter(B.clone().contiguous())
        self.scaling = float(scaling)

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.A) @ self.B
        return base_out + self.scaling * lora_out


# Projection order matches test_lora_forward.FA_SHAPES / DN_SHAPES, and
# also matches the order in which lora_tensors are passed to our kernel.
# For FA layers:   indices 0..6  in lora_tensors (pair of A, B per index)
# For DN layers:   indices 7..12 in lora_tensors
_FA_PROJ_NAMES = ["fa_q", "fa_k", "fa_v", "fa_o", "fa_gate", "fa_up", "fa_down"]
_DN_PROJ_NAMES = ["dn_qkv", "dn_z", "dn_out", "dn_gate", "dn_up", "dn_down"]


def wrap_hf_with_lora(hf_model, lora_tensors: list[torch.Tensor],
                      scaling: float = 1.0) -> None:
    """Replace the 13 trainable Linear modules on each Qwen layer with a
    LoraLinear, using the same A/B tensors as our kernel consumes.

    ``lora_tensors`` is the 26-element list returned by
    `test_lora_forward.build_lora_tensors` — A0, B0, A1, B1, ... in the
    order (7 FA projections, then 6 DN projections). Each A is
    [n_fa_or_dn, K_in, R] and each B is [n_fa_or_dn, R, K_out].
    """
    # Unpack to {name: (A_all_layers, B_all_layers)}.
    by_name: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    it = iter(lora_tensors)
    for n in _FA_PROJ_NAMES + _DN_PROJ_NAMES:
        A = next(it); B = next(it)
        by_name[n] = (A, B)

    fa_idx = 0
    dn_idx = 0
    for li, lt in enumerate(LAYER_TYPE):
        layer = hf_model.model.layers[li]
        if lt == 1:  # Full attention
            sa, mlp = layer.self_attn, layer.mlp
            patches = [
                (sa, "q_proj",    "fa_q"),
                (sa, "k_proj",    "fa_k"),
                (sa, "v_proj",    "fa_v"),
                (sa, "o_proj",    "fa_o"),
                (mlp, "gate_proj", "fa_gate"),
                (mlp, "up_proj",   "fa_up"),
                (mlp, "down_proj", "fa_down"),
            ]
            for parent, attr, name in patches:
                A_all, B_all = by_name[name]
                A, B = A_all[fa_idx], B_all[fa_idx]
                base = getattr(parent, attr)
                setattr(parent, attr, LoraLinear(base, A, B, scaling))
            fa_idx += 1
        else:  # DeltaNet
            la, mlp = layer.linear_attn, layer.mlp
            patches = [
                (la, "in_proj_qkv", "dn_qkv"),
                (la, "in_proj_z",   "dn_z"),
                (la, "out_proj",    "dn_out"),
                (mlp, "gate_proj", "dn_gate"),
                (mlp, "up_proj",   "dn_up"),
                (mlp, "down_proj", "dn_down"),
            ]
            for parent, attr, name in patches:
                A_all, B_all = by_name[name]
                A, B = A_all[dn_idx], B_all[dn_idx]
                base = getattr(parent, attr)
                setattr(parent, attr, LoraLinear(base, A, B, scaling))
            dn_idx += 1

    # Also freeze the embed + lm_head + norms — they stay non-trainable
    # in a LoRA adapter.
    for n, p in hf_model.named_parameters():
        if "lora" not in n.lower() and not any(
            f"layers.{li}." in n and (".A" in n or ".B" in n)
            for li in range(len(LAYER_TYPE))
        ):
            # Parameters ending in .A or .B are our injected LoRA params.
            if n.endswith(".A") or n.endswith(".B"):
                continue
            p.requires_grad_(False)


def lora_parameters(hf_model) -> list[torch.Tensor]:
    """Return the flat list of (A, B) Parameters from all LoraLinear
    modules — same order as lora_tensors input (26 tensors)."""
    params = []
    fa_idx = 0
    dn_idx = 0
    for li, lt in enumerate(LAYER_TYPE):
        layer = hf_model.model.layers[li]
        if lt == 1:
            sa, mlp = layer.self_attn, layer.mlp
            modules = [sa.q_proj, sa.k_proj, sa.v_proj, sa.o_proj,
                       mlp.gate_proj, mlp.up_proj, mlp.down_proj]
            fa_idx += 1
        else:
            la, mlp = layer.linear_attn, layer.mlp
            modules = [la.in_proj_qkv, la.in_proj_z, la.out_proj,
                       mlp.gate_proj, mlp.up_proj, mlp.down_proj]
            dn_idx += 1
        for m in modules:
            assert isinstance(m, LoraLinear), f"layer {li} module not wrapped"
            params.append(m.A)
            params.append(m.B)
    return params
