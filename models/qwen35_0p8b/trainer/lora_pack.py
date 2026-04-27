"""Pack/unpack helpers between PEFT's per-projection LoRA modules and the
flat 26-tensor layout our cuBLAS+graph kernel consumes.

PEFT layout (per projection, fp32 / bf16 once cast on the live model):
    .lora_A.default.weight  : [R, K_in]
    .lora_B.default.weight  : [K_out, R]

Our kernel layout (per projection, bf16, layer-major):
    A_all : [n_layers_of_type, K_in, R]
    B_all : [n_layers_of_type, R, K_out]

Order across the 13 projections (canonical, matches torch_bindings.cpp):
    FA (n_layers=N_FA): q, k, v, o, gate, up, down       — 7 (A,B) pairs
    DN (n_layers=N_DN): qkv, z, out, gate, up, down      — 6 (A,B) pairs
    Flattened: [A0, B0, A1, B1, ..., A12, B12]           — 26 tensors

Projections not present on the live PEFT model (e.g. when the trainer's
LoRA config doesn't target dn_qkv / dn_z / dn_out) are left as zeros,
which the kernel treats as a no-op for that projection.
"""
from __future__ import annotations

import importlib.util
import sys

from typing import Any

import torch

# Load constants from the OUTER model.py via importlib to avoid the
# prefill_megakernel/model.py sibling shadowing it on sys.path. The
# outer model uses HIDDEN_SIZE / INTERMEDIATE_SIZE / DN_CONV_CHANNELS
# names; the prefill_megakernel sibling uses shorter HIDDEN / INTER /
# DN_CONV_CH and would fail this from-import.
_spec = importlib.util.spec_from_file_location(
    "qwen_outer_model_for_pack",
    "/root/lucebox-hub-b200-train/models/qwen35_0p8b/model.py")
_outer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_outer)
LAYER_TYPE     = _outer.LAYER_TYPE
HIDDEN         = _outer.HIDDEN_SIZE
INTER          = _outer.INTERMEDIATE_SIZE
FA_QPROJ_SIZE  = _outer.FA_QPROJ_SIZE
FA_KV_SIZE     = _outer.FA_KV_SIZE
FA_Q_SIZE      = _outer.FA_Q_SIZE
DN_CONV_CH     = _outer.DN_CONV_CHANNELS
DN_V_SIZE      = _outer.DN_V_SIZE


N_FA = sum(1 for t in LAYER_TYPE if t == 1)
N_DN = sum(1 for t in LAYER_TYPE if t == 0)


# (parent attribute, projection attribute) → kernel-layout name.
# The trainer's LoraConfig targets the projection by attribute name; PEFT's
# replacement modules live at the same attribute path.
_FA_PROJ_MAP: dict[tuple[str, str], str] = {
    ("self_attn", "q_proj"):     "fa_q",
    ("self_attn", "k_proj"):     "fa_k",
    ("self_attn", "v_proj"):     "fa_v",
    ("self_attn", "o_proj"):     "fa_o",
    ("mlp", "gate_proj"):        "fa_gate",
    ("mlp", "up_proj"):          "fa_up",
    ("mlp", "down_proj"):        "fa_down",
}
_DN_PROJ_MAP: dict[tuple[str, str], str] = {
    ("linear_attn", "in_proj_qkv"): "dn_qkv",
    ("linear_attn", "in_proj_z"):   "dn_z",
    ("linear_attn", "out_proj"):    "dn_out",
    ("mlp", "gate_proj"):           "dn_gate",
    ("mlp", "up_proj"):             "dn_up",
    ("mlp", "down_proj"):           "dn_down",
}

# Canonical kernel order — must match torch_bindings.cpp's SET(idx, name) table.
_KERNEL_ORDER: list[str] = [
    "fa_q", "fa_k", "fa_v", "fa_o", "fa_gate", "fa_up", "fa_down",
    "dn_qkv", "dn_z", "dn_out", "dn_gate", "dn_up", "dn_down",
]


def _shape_table(rank: int) -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Per-projection (A_shape, B_shape) for the kernel's batched layout."""
    return {
        "fa_q":    ((N_FA, HIDDEN,    rank), (N_FA, rank, FA_QPROJ_SIZE)),
        "fa_k":    ((N_FA, HIDDEN,    rank), (N_FA, rank, FA_KV_SIZE)),
        "fa_v":    ((N_FA, HIDDEN,    rank), (N_FA, rank, FA_KV_SIZE)),
        "fa_o":    ((N_FA, FA_Q_SIZE, rank), (N_FA, rank, HIDDEN)),
        "fa_gate": ((N_FA, HIDDEN,    rank), (N_FA, rank, INTER)),
        "fa_up":   ((N_FA, HIDDEN,    rank), (N_FA, rank, INTER)),
        "fa_down": ((N_FA, INTER,     rank), (N_FA, rank, HIDDEN)),
        "dn_qkv":  ((N_DN, HIDDEN,    rank), (N_DN, rank, DN_CONV_CH)),
        "dn_z":    ((N_DN, HIDDEN,    rank), (N_DN, rank, DN_V_SIZE)),
        "dn_out":  ((N_DN, DN_V_SIZE, rank), (N_DN, rank, HIDDEN)),
        "dn_gate": ((N_DN, HIDDEN,    rank), (N_DN, rank, INTER)),
        "dn_up":   ((N_DN, HIDDEN,    rank), (N_DN, rank, INTER)),
        "dn_down": ((N_DN, INTER,     rank), (N_DN, rank, HIDDEN)),
    }


def _peft_underlying_layers(peft_model: Any):
    """Return the list of HF transformer layers under a PEFT wrapper.

    PEFT wraps the model as ``base_model.model.<original>``; the layer
    list ``hf_model.model.layers`` is reachable as
    ``peft_model.base_model.model.model.layers``.
    """
    return peft_model.base_model.model.model.layers


def _get_lora_module(parent_module, proj_attr: str):
    """Return the PEFT LoRA wrapper at parent_module.<proj_attr>, or None
    if that projection isn't LoRA-wrapped (the trainer config didn't
    target it)."""
    proj = getattr(parent_module, proj_attr, None)
    if proj is None:
        return None
    if not (hasattr(proj, "lora_A") and hasattr(proj, "lora_B")):
        return None
    return proj


def _read_AB(lora_module) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Read PEFT's lora_A/lora_B Parameters (default adapter) on a wrapped
    projection; returns (A, B) as fp-or-bf16 tensors in PEFT's layout
    (A:[R,K_in], B:[K_out,R]). Returns None if missing."""
    A_dict = lora_module.lora_A
    B_dict = lora_module.lora_B
    name = "default" if "default" in A_dict else next(iter(A_dict.keys()))
    if name not in A_dict or name not in B_dict:
        return None
    A = A_dict[name].weight  # [R, K_in]
    B = B_dict[name].weight  # [K_out, R]
    return A, B


def pack_peft_to_flat(peft_model: Any, lora_rank: int) -> list[torch.Tensor]:
    """Walk a live PEFT-wrapped HF Qwen3.5-0.8B and pack its LoRA adapters
    into the kernel's 26-tensor flat layout.

    Returns a list of 26 bf16 cuda tensors, in canonical kernel order.
    Projections not present on the model are left as zeros. The returned
    tensors are **fresh allocations** — no aliasing into PEFT's parameters.
    """
    shapes = _shape_table(lora_rank)
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name in _KERNEL_ORDER:
        Ash, Bsh = shapes[name]
        out[name] = (torch.zeros(Ash, **bf16), torch.zeros(Bsh, **bf16))

    layers = _peft_underlying_layers(peft_model)
    fa_idx = dn_idx = 0
    for li, lt in enumerate(LAYER_TYPE):
        layer = layers[li]
        proj_map = _FA_PROJ_MAP if lt == 1 else _DN_PROJ_MAP
        kernel_idx = fa_idx if lt == 1 else dn_idx
        for (parent_attr, proj_attr), kernel_name in proj_map.items():
            parent = getattr(layer, parent_attr, None)
            if parent is None:
                continue
            mod = _get_lora_module(parent, proj_attr)
            if mod is None:
                continue
            ab = _read_AB(mod)
            if ab is None:
                continue
            A, B = ab
            # PEFT (R, K_in) → ours (K_in, R).
            A_ours = A.detach().t().contiguous().to(**bf16)
            # PEFT (K_out, R) → ours (R, K_out).
            B_ours = B.detach().t().contiguous().to(**bf16)
            out[kernel_name][0][kernel_idx].copy_(A_ours)
            out[kernel_name][1][kernel_idx].copy_(B_ours)
        if lt == 1:
            fa_idx += 1
        else:
            dn_idx += 1

    flat: list[torch.Tensor] = []
    for name in _KERNEL_ORDER:
        A, B = out[name]
        flat.append(A)
        flat.append(B)
    return flat


def unpack_flat_to_peft(peft_model: Any, flat: list[torch.Tensor]) -> int:
    """Write the 26 flat tensors back into a PEFT-wrapped model's LoRA
    parameters (in-place via ``param.data.copy_``).

    Counterpart to :func:`pack_peft_to_flat`. Returns the number of
    (A, B) pairs written. Projections not present on the live model are
    silently skipped.
    """
    if len(flat) != 26:
        raise ValueError(f"expected 26 flat tensors, got {len(flat)}")
    by_name: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for i, name in enumerate(_KERNEL_ORDER):
        by_name[name] = (flat[2 * i], flat[2 * i + 1])

    layers = _peft_underlying_layers(peft_model)
    fa_idx = dn_idx = 0
    n_written = 0
    for li, lt in enumerate(LAYER_TYPE):
        layer = layers[li]
        proj_map = _FA_PROJ_MAP if lt == 1 else _DN_PROJ_MAP
        kernel_idx = fa_idx if lt == 1 else dn_idx
        for (parent_attr, proj_attr), kernel_name in proj_map.items():
            parent = getattr(layer, parent_attr, None)
            if parent is None:
                continue
            mod = _get_lora_module(parent, proj_attr)
            if mod is None:
                continue
            ab = _read_AB(mod)
            if ab is None:
                continue
            A_ours, B_ours = by_name[kernel_name]
            A_peft, B_peft = ab
            # ours (K_in, R) → PEFT (R, K_in) = transpose.
            A_peft.data.copy_(A_ours[kernel_idx].t().contiguous().to(A_peft.dtype))
            B_peft.data.copy_(B_ours[kernel_idx].t().contiguous().to(B_peft.dtype))
            n_written += 1
        if lt == 1:
            fa_idx += 1
        else:
            dn_idx += 1
    return n_written


def scatter_flat_grads_to_peft(
    peft_model: Any,
    flat_grads: list[torch.Tensor],
    *,
    accumulate: bool = False,
) -> int:
    """Scatter the 26-tensor flat-format LoRA gradient buffer (output of
    `run_layer_walking_bwd`) into PEFT's per-projection ``.grad``.

    Layout transform mirrors :func:`unpack_flat_to_peft`: ours
    (K_in, R) / (R, K_out) → PEFT (R, K_in) / (K_out, R) via transpose.
    Dtypes are preserved on PEFT's params (typically bf16).

    If ``accumulate=True``, the new grad is added to any existing
    ``.grad``; otherwise it overwrites. Returns the number of (A, B)
    pairs scattered.
    """
    if len(flat_grads) != 26:
        raise ValueError(f"expected 26 flat grad tensors, got {len(flat_grads)}")
    by_name: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for i, name in enumerate(_KERNEL_ORDER):
        by_name[name] = (flat_grads[2 * i], flat_grads[2 * i + 1])

    layers = _peft_underlying_layers(peft_model)
    fa_idx = dn_idx = 0
    n_written = 0
    for li, lt in enumerate(LAYER_TYPE):
        layer = layers[li]
        proj_map = _FA_PROJ_MAP if lt == 1 else _DN_PROJ_MAP
        kernel_idx = fa_idx if lt == 1 else dn_idx
        for (parent_attr, proj_attr), kernel_name in proj_map.items():
            parent = getattr(layer, parent_attr, None)
            if parent is None:
                continue
            mod = _get_lora_module(parent, proj_attr)
            if mod is None:
                continue
            ab = _read_AB(mod)
            if ab is None:
                continue
            A_grads_ours, B_grads_ours = by_name[kernel_name]
            A_peft, B_peft = ab

            # Layer-slice of our flat grads.
            gA_ours = A_grads_ours[kernel_idx]                  # [K_in, R]
            gB_ours = B_grads_ours[kernel_idx]                  # [R, K_out]

            # Transpose to PEFT layout, cast to PEFT param dtype.
            gA_peft = gA_ours.t().contiguous().to(A_peft.dtype)  # [R, K_in]
            gB_peft = gB_ours.t().contiguous().to(B_peft.dtype)  # [K_out, R]

            if accumulate and A_peft.grad is not None:
                A_peft.grad += gA_peft
            else:
                A_peft.grad = gA_peft.clone()
            if accumulate and B_peft.grad is not None:
                B_peft.grad += gB_peft
            else:
                B_peft.grad = gB_peft.clone()

            n_written += 1
        if lt == 1:
            fa_idx += 1
        else:
            dn_idx += 1
    return n_written


__all__ = [
    "pack_peft_to_flat", "unpack_flat_to_peft",
    "scatter_flat_grads_to_peft",
    "N_FA", "N_DN",
]
