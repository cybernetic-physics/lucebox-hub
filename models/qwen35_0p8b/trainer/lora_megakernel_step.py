"""Custom autograd.Function for the LoRA training step.

This file is the foundation for Tier 0.2 of the optimization roadmap:
replace HF+PEFT's autograd path (442 ms wall, 19,500 launches/step) with
a single-dispatch forward through our cuBLAS+graph kernel plus a
hand-rolled backward that calls the per-layer bwd kernels we already
have. See `models/qwen35_0p8b/trainer/PERF_NOTES.md`.

Status (Slice B.1, 2026-04-27): forward + per-position sequence loss is
fully implemented and validated against HF+PEFT (loss Δ ≈ 5e-3, per-
position logp cosine ≥ 0.99998). The backward stub raises
NotImplementedError; Slice B.2 will fill it in.

The 4 activation slabs `prefill_bf16_train_step` already saves cover
half of what the per-layer reverse walk needs:

  hidden_in[L, S, HIDDEN]          → input to layer-L input RMSnorm
  normalized_in[L, S, HIDDEN]      → input to QKV projections
  normalized_post_attn[L, S, HIDDEN] → input to gate/up projections
  mlp_inter[L, S, INTER]           → input to down projection

The other half — attn_out_pre_o[L,S,*] (input to o_proj), h_post_attn[L]
(input to post-attn RMSnorm), and the DN state history needed by
dn_bwd — must be added to the kernel before Slice B.2 can land. Slice
B.2's CUDA work is therefore:

  1. Extend prefill_bf16_train_step to take additional optional save
     buffers and write them during the forward (~1 day):
       saves["attn_out_pre_o"][L, S, max(FA_Q_SIZE, DN_V_SIZE)]  bf16
       saves["h_post_attn"][L, S, HIDDEN]                        bf16
       saves["dn_state_history"][N_DN, S+1, DN_HEADS, DN_KEY, DN_VAL] f32
  2. With saves in place, write the layer-walking backward (~3 days):
       autograd through python (lm_head + log_softmax + nll + final_norm)
       gives d(sc["hidden"]) for free. Then per layer in reverse:
         residual split: dh_post_attn += dh_out, dmlp_out = dh_out
         down LoRA bwd       (bwd_lora_linear with x=mlp_inter[L])
         + base down dx       (cuBLAS GemmEx on base down weight)
         swiglu bwd          (bwd_swiglu, recompute gate/up via cuBLAS)
         gate, up LoRA bwd   (bwd_lora_linear; sum dx into dnorm_post_attn)
         post-attn rmsnorm bwd (bwd_rmsnorm with x=h_post_attn[L])
         residual split: dh_in += dh_post_attn, dattn_out = dh_post_attn
         o_proj LoRA bwd     (bwd_lora_linear with x=attn_out_pre_o[L])
         attention bwd:
           FA → fa_bwd_flash (cuDNN FA-2)
           DN → dn_bwd (recurrent CUDA), uses dn_state_history
         q,k,v (FA) or qkv,z (DN) LoRA bwd → dnormalized_in
         input rmsnorm bwd   (bwd_rmsnorm with x=hidden_in[L])
         dh_in += dnormalized_in_through_input_rmsnorm
       Output dh_in is layer L-1's dh_out.
  3. Wire LoRA grads into a flat fp32 buffer for fused_adamw_step (~1 day).
  4. Trainer integration in rl_trainer.forward_backward (~1 day).
  5. Convergence test in test_rl_trainer_e2e.py (~1 day).

Total: ~7-8 engineer-days for the full Slice B.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from typing import Any

import torch

# Guard the same `fla` import path we disable in rl_trainer + tests.
try:
    import transformers.utils.import_utils as _iu
    _iu.is_flash_linear_attention_available = lambda: False
except Exception:
    pass

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b")

import qwen35_megakernel_bf16_C  # noqa: F401

# Outer model module (constants + helpers).
_spec = importlib.util.spec_from_file_location(
    "qwen_outer_model",
    "/root/lucebox-hub-b200-train/models/qwen35_0p8b/model.py")
_outer = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_outer)
_load_weights = _outer.load_weights
_pack_layer_weights = _outer._pack_layer_weights

NUM_LAYERS = _outer.NUM_LAYERS
HIDDEN = _outer.HIDDEN_SIZE
INTER = _outer.INTERMEDIATE_SIZE
VOCAB = _outer.VOCAB_SIZE
FA_QPROJ_SIZE = _outer.FA_QPROJ_SIZE
FA_KV_SIZE = _outer.FA_KV_SIZE
FA_Q_SIZE = _outer.FA_Q_SIZE
FA_HEAD_DIM = _outer.FA_HEAD_DIM
FA_KV_HEADS = _outer.FA_NUM_KV_HEADS
DN_HEADS = _outer.DN_NUM_HEADS
DN_KEY = _outer.DN_KEY_DIM
DN_VAL = _outer.DN_VALUE_DIM
DN_V_SIZE = _outer.DN_V_SIZE
DN_CONV_CH = _outer.DN_CONV_CHANNELS
DN_CONV_K = _outer.DN_CONV_KERNEL
LAYER_TYPE = _outer.LAYER_TYPE

N_FA = sum(1 for t in LAYER_TYPE if t == 1)
N_DN = sum(1 for t in LAYER_TYPE if t == 0)


# ---------------------------------------------------------------------------
# Base-model handle: groups the things that live for the lifetime of a
# training session (frozen weights, packed-pointer struct). Built once,
# reused per training step. Not part of autograd.
# ---------------------------------------------------------------------------

@dataclass
class BaseModelHandle:
    """Frozen base-model state needed to call our kernel.

    Built once via :func:`load_base_model` and shared across many training
    steps. The kernel reads weights through ``layers_packed`` (a packed
    pointer struct), so Python only has to keep the underlying tensors
    alive — it never touches their data on the hot path.
    """
    weights: dict[str, Any]          # outer model.load_weights(...) output
    layers_packed: torch.Tensor      # packed LayerWeights struct on cuda
    embed_weight: torch.Tensor       # bf16, [VOCAB, HIDDEN]
    final_norm_weight: torch.Tensor  # bf16, [HIDDEN]
    lm_head_weight: torch.Tensor     # bf16, [VOCAB, HIDDEN]


def load_base_model(model_name: str = "Qwen/Qwen3.5-0.8B",
                    *, verbose: bool = False) -> BaseModelHandle:
    """Load Qwen3.5-0.8B's frozen base weights for our kernel."""
    weights, _tokenizer = _load_weights(model_name, verbose=verbose, backend="bf16")
    layers_packed = _pack_layer_weights(weights["layer_data"])
    return BaseModelHandle(
        weights=weights,
        layers_packed=layers_packed,
        embed_weight=weights["embed_weight"],
        final_norm_weight=weights["final_norm_weight"],
        lm_head_weight=weights["lm_head_weight"],
    )


# ---------------------------------------------------------------------------
# Scratch buffers — one set per training step shape (S, lora_rank). Cached
# on the handle to avoid re-allocation when the trainer's batches share a
# shape (the common case after Tier 0.3's batched forward).
# ---------------------------------------------------------------------------

def alloc_scratch(S: int, lora_rank: int) -> dict:
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    f32 = dict(dtype=torch.float32, device="cuda")
    i32 = dict(dtype=torch.int32, device="cuda")
    max_proj = max(FA_QPROJ_SIZE, DN_CONV_CH, INTER)
    max_attn = max(FA_Q_SIZE, DN_V_SIZE, FA_KV_SIZE)
    return dict(
        fa_k_cache=torch.zeros(N_FA, FA_KV_HEADS, 32768, FA_HEAD_DIM, **bf16),
        fa_v_cache=torch.zeros(N_FA, FA_KV_HEADS, 32768, FA_HEAD_DIM, **bf16),
        dn_states=torch.zeros(N_DN, DN_HEADS, DN_KEY, DN_VAL, **f32),
        conv_bufs=torch.zeros(N_DN, DN_CONV_CH, DN_CONV_K, **f32),
        hidden=torch.empty(S * HIDDEN, **bf16),
        residual=torch.empty(S * HIDDEN, **bf16),
        normalized=torch.empty(S * HIDDEN, **bf16),
        proj_buf=torch.empty(S * max_proj, **bf16),
        proj_buf2=torch.empty(S * max_proj, **bf16),
        attn_buf=torch.empty(S * max_attn, **bf16),
        mlp_buf=torch.empty(S * INTER, **bf16),
        dn_out_buf=torch.empty(S * max_attn, **bf16),
        beta_buf=torch.empty(S * DN_HEADS, **f32),
        alpha_buf=torch.empty(S * DN_HEADS, **f32),
        final_normed=torch.empty(HIDDEN, **bf16),
        hidden_bf16_out=torch.empty(HIDDEN, **bf16),
        out_token=torch.empty(1, **i32),
        lm_bmv=torch.empty(1024, **f32),
        lm_bmi=torch.empty(1024, **i32),
        lora_h_ws=torch.empty(S, lora_rank, **bf16),
    )


def alloc_activation_saves(S: int, *, with_b2_saves: bool = True,
                            with_fa_bwd_saves: bool = True) -> dict:
    """The per-layer activation slabs `prefill_bf16_train_step` writes
    during fwd.

    ``with_b2_saves`` — the two Slice B.2 slabs the layer-walking
                        backward needs (attn_out_pre_o, h_post_attn).
    ``with_fa_bwd_saves`` — the three Slice B.3b FA-only slabs that
                        let per_layer_bwd_fa skip the autograd
                        recompute (fa_q_save, fa_o_save, fa_lse_save).
    """
    from lora_pack import N_FA  # local import to avoid circular dep
    FA_HEAD_DIM = 256
    FA_Q_HEADS  = 8
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    f32  = dict(dtype=torch.float32,  device="cuda")
    out = dict(
        hidden_in=torch.empty(NUM_LAYERS, S, HIDDEN, **bf16),
        normalized_in=torch.empty(NUM_LAYERS, S, HIDDEN, **bf16),
        normalized_post_attn=torch.empty(NUM_LAYERS, S, HIDDEN, **bf16),
        mlp_inter=torch.empty(NUM_LAYERS, S, INTER, **bf16),
    )
    if with_b2_saves:
        out["attn_out_pre_o"] = torch.empty(NUM_LAYERS, S, DN_V_SIZE, **bf16)
        out["h_post_attn"] = torch.empty(NUM_LAYERS, S, HIDDEN, **bf16)
    if with_fa_bwd_saves:
        out["fa_q_save"]   = torch.empty(N_FA, S, FA_Q_HEADS, FA_HEAD_DIM, **bf16)
        out["fa_o_save"]   = torch.empty(N_FA, S, FA_Q_HEADS, FA_HEAD_DIM, **bf16)
        out["fa_lse_save"] = torch.empty(N_FA, FA_Q_HEADS, S, **f32)
    return out


# ---------------------------------------------------------------------------
# Forward kernel call — a thin wrapper that returns the kernel's outputs +
# the four activation slabs needed by the per-layer backward kernels.
# ---------------------------------------------------------------------------

def _train_step_forward(
    handle: BaseModelHandle,
    tokens: torch.Tensor,            # [S], int32 cuda
    lora_flat: list[torch.Tensor],   # 26 packed bf16 tensors
    lora_rank: int,
    lora_scaling: float,
    *,
    sc: dict | None = None,
    saves: dict | None = None,
) -> tuple[dict, dict]:
    """Run prefill_bf16_train_step, return (scratch, saves).

    `scratch["final_normed"]` is the bf16 hidden-state at the LAST prompt
    position post final RMSnorm — useful to compute the next-token logits
    for that position via `final_normed @ lm_head.T`.

    The 4 activation slabs in `saves` are populated for every layer and
    every position in the sequence. Slice B.2 will consume them in the
    backward kernel sequence.
    """
    S = int(tokens.numel())
    if sc is None:
        sc = alloc_scratch(S, lora_rank)
    if saves is None:
        saves = alloc_activation_saves(S)
    empty_bf16 = torch.empty(0, dtype=torch.bfloat16, device="cuda")
    empty_f32  = torch.empty(0, dtype=torch.float32,  device="cuda")
    torch.ops.qwen35_megakernel_bf16_C.prefill_bf16_train_step(
        sc["out_token"], tokens.to(dtype=torch.int32, device="cuda").contiguous(),
        handle.embed_weight, handle.layers_packed,
        handle.final_norm_weight, handle.lm_head_weight,
        sc["fa_k_cache"], sc["fa_v_cache"], sc["dn_states"], sc["conv_bufs"],
        sc["hidden"], sc["residual"], sc["normalized"],
        sc["proj_buf"], sc["proj_buf2"], sc["attn_buf"], sc["mlp_buf"],
        sc["dn_out_buf"], sc["beta_buf"], sc["alpha_buf"],
        sc["final_normed"], sc["hidden_bf16_out"],
        sc["lm_bmv"], sc["lm_bmi"],
        *lora_flat, lora_rank, lora_scaling, sc["lora_h_ws"],
        saves["hidden_in"], saves["normalized_in"],
        saves["normalized_post_attn"], saves["mlp_inter"],
        saves.get("attn_out_pre_o", empty_bf16),
        saves.get("h_post_attn", empty_bf16),
        saves.get("fa_q_save",   empty_bf16),
        saves.get("fa_o_save",   empty_bf16),
        saves.get("fa_lse_save", empty_f32),
    )
    return sc, saves


# ---------------------------------------------------------------------------
# Per-position post-final-norm hidden state.
#
# Empirically (probe_hidden_buffer.py — see git log), `sc["hidden"]` post-
# kernel contains the pre-final-norm residual stream h_out[NUM_LAYERS-1]
# for ALL S positions, bit-exactly matching what the kernel applied final
# RMSNorm to for the last position. We can therefore compute per-position
# final_normed in Python (one cuBLAS-equivalent op) without modifying the
# kernel — unlocks per-position CE loss for sequence training.
# ---------------------------------------------------------------------------

def _rms_norm_qwen(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Qwen3-Next's RMSNorm formula (matches the kernel bit-exactly):
       y = x * rsqrt(mean(x*x) + eps) * (1.0 + w)
    Uses fp32 intermediate, casts back to x.dtype.
    """
    x_f = x.float()
    rstd = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_f * rstd * (1.0 + w.float())).to(x.dtype)


def per_position_final_normed(
    sc: dict, handle: BaseModelHandle, S: int
) -> torch.Tensor:
    """Recover post-final-norm hidden states for all S positions.

    Returns a [S, HIDDEN] bf16 tensor.
    """
    h_out_last_layer = sc["hidden"].view(S, HIDDEN)
    return _rms_norm_qwen(h_out_last_layer, handle.final_norm_weight)


# ---------------------------------------------------------------------------
# Sequence cross-entropy loss via our kernel — autograd-tracked through the
# Python-side final-norm + lm_head + log_softmax. The base-model weights
# are frozen, but `sc["hidden"]` is a leaf tensor for autograd (because
# the kernel call is opaque), so gradients flowing back from this loss
# stop at sc["hidden"] — they tell us d(h_out[NUM_LAYERS-1]) at all
# positions, which is the input to the per-layer reverse walk in
# Slice B.2's backward.
# ---------------------------------------------------------------------------

def kernel_sequence_loss(
    handle: BaseModelHandle,
    prompt_tokens: torch.Tensor,     # [P]
    target_tokens: torch.Tensor,     # [T]
    lora_flat: list[torch.Tensor],
    lora_rank: int,
    lora_scaling: float,
    *,
    return_per_position_logp: bool = False,
) -> dict:
    """Run our kernel forward, compute -log p(target_t | prompt + target_<t)
    averaged over t∈[0, T).

    Returns a dict with:
      "loss"          : scalar fp32 cuda tensor (mean -logp over targets)
      "logp"          : [T] fp32 cuda tensor of per-target logp (if asked)
      "h_pre_norm"    : [S, HIDDEN] bf16, pre-final-norm hidden for all
                        positions S = P + T - 1 (the kernel was run on
                        the concatenated prompt+target[:-1] sequence).
                        This is the input to Slice B.2's reverse walk.
      "saves"         : 4-slab activation save dict
      "scratch"       : kernel scratch buffers (kept alive for backward)
      "predict_pos"   : torch.LongTensor of positions in `h_pre_norm` whose
                        predictions correspond to target_tokens — i.e.
                        for target_t in [0, T), predict_pos[t] = P-1+t.

    The path mirrors `rl_trainer._sequence_logprobs` so we can drop this
    in as a 1:1 replacement once the backward is wired.
    """
    P = int(prompt_tokens.numel())
    T = int(target_tokens.numel())
    if T == 0:
        raise ValueError("target_tokens must be non-empty")
    full = torch.cat([prompt_tokens, target_tokens], dim=0)
    S = int(full.numel())

    sc, saves = _train_step_forward(
        handle, full, lora_flat, lora_rank, lora_scaling,
    )

    # Pre-final-norm hidden, all positions.
    h_pre = sc["hidden"].view(S, HIDDEN)
    final_normed = _rms_norm_qwen(h_pre, handle.final_norm_weight)  # [S, HIDDEN]

    # logits at predict_pos = P-1, P, ..., P+T-2  (T positions; predicts targets).
    predict_pos = torch.arange(P - 1, P - 1 + T, device="cuda")
    predict_logits = (final_normed[predict_pos].float()
                      @ handle.lm_head_weight.float().t())  # [T, VOCAB]
    log_probs = torch.nn.functional.log_softmax(predict_logits, dim=-1)
    logp = log_probs.gather(1, target_tokens.long().unsqueeze(1)).squeeze(1)  # [T]
    loss = -logp.mean()

    out = {
        "loss": loss,
        "h_pre_norm": h_pre,
        "saves": saves,
        "scratch": sc,
        "predict_pos": predict_pos,
    }
    if return_per_position_logp:
        out["logp"] = logp
    return out


# ---------------------------------------------------------------------------
# autograd.Function — forward done, backward stub. Slice B.2 fills in the
# layer-walking backward path using:
#
#   bwd_ce_lm_head        → grad on final_normed at the target position
#   bwd_rmsnorm           → grad through the final RMSNorm
#   for layer in reverse:
#     down LoRA bwd        (bwd_lora_linear with x=mlp_inter)
#     swiglu bwd           (bwd_swiglu)
#     gate / up LoRA bwd   (bwd_lora_linear, accumulate dx → grad_post_attn_norm)
#     post-attn rmsnorm bwd
#     o_proj LoRA bwd
#     attention bwd:
#       FA layers → fa_bwd_flash (cuDNN FA-2)
#       DN layers → dn_bwd (recurrent CUDA) or fla.chunk_gated_delta_rule_bwd
#     q/k/v (FA) or qkv/z (DN) LoRA bwd → grad_norm_in
#     input rmsnorm bwd
#     accumulate: grad_hidden_in = norm_in_dx + residual paths
#
# Per-layer state needs (from the activation saves):
#   FA: hidden_in[i], normalized_in[i], normalized_post_attn[i], mlp_inter[i]
#   DN: same, plus the saved DN state_history (need to add to the kernel
#       output, currently the kernel doesn't emit per-layer DN state for bwd).
#
# Per-projection LoRA grads scatter into a flat [num_params] fp32 buffer
# the fused AdamW kernel consumes.
# ---------------------------------------------------------------------------

class MegakernelTrainStep(torch.autograd.Function):
    """LoRA training step through our cuBLAS+graph kernel.

    Forward: full prefill+LoRA forward in one cuBLAS+graph dispatch,
    saves the 4 per-layer activation slabs, returns last-position
    final_normed for the caller to project to logits.

    Backward (Slice B.2): NotImplementedError. See module docstring for
    the per-layer bwd plan.
    """

    @staticmethod
    def forward(
        ctx,
        lora_flat_concat: torch.Tensor,  # placeholder; see note below
        prompt_tokens: torch.Tensor,
        handle: BaseModelHandle,
        lora_rank: int,
        lora_scaling: float,
        lora_flat: list[torch.Tensor],
    ) -> torch.Tensor:
        """Run forward, save state needed by backward.

        Note: torch.autograd.Function has restrictions on what can be
        non-Tensor positional args. We pass `handle`, `lora_rank`,
        `lora_scaling`, and `lora_flat` as Python objects via ctx, and
        carry a placeholder lora_flat_concat tensor as the first arg
        purely so autograd has a Tensor handle for grad tracking. Slice
        B.2 will replace this with a single flat concatenated bf16 LoRA
        buffer (and matching grad buffer) so the autograd plumbing is
        cleaner and the fused AdamW can read/write it directly.
        """
        sc, saves = _train_step_forward(
            handle, prompt_tokens, lora_flat, lora_rank, lora_scaling,
        )
        ctx.save_for_backward(
            prompt_tokens,
            saves["hidden_in"], saves["normalized_in"],
            saves["normalized_post_attn"], saves["mlp_inter"],
            sc["fa_k_cache"], sc["fa_v_cache"],
            sc["dn_states"], sc["conv_bufs"],
            sc["final_normed"],
            *lora_flat,
        )
        ctx.handle = handle
        ctx.lora_rank = lora_rank
        ctx.lora_scaling = lora_scaling
        return sc["final_normed"].clone()

    @staticmethod
    def backward(ctx, grad_final_normed: torch.Tensor):
        raise NotImplementedError(
            "MegakernelTrainStep.backward is Slice B.2 work. The forward "
            "and activation saves are wired and validated; the backward "
            "needs the layer-walking bwd-kernel sequence per the module "
            "docstring."
        )


def precompute_dn_saves(
    hf_model,
    saves: dict,
) -> list[dict | None]:
    """Run dn_attn_forward for each DN layer at forward time, storing the
    saves dict the bwd path will consume. Indexed by ABSOLUTE layer index
    (so FA layers get None entries; DN layers get a saves dict).

    This eliminates the bwd-time dn_attn_forward recompute. The work
    moves from bwd to fwd; net wall time is reduced because fwd-time
    Python ops can be issued without contending with bwd-time critical
    path, and the saves dict is reused as-is by dn_attn_backward.
    """
    from dn_attn_handrolled import dn_attn_forward
    layers = hf_model.base_model.model.model.layers
    out: list[dict | None] = [None] * NUM_LAYERS
    for L in range(NUM_LAYERS):
        if LAYER_TYPE[L] != 0:
            continue
        hf_layer = layers[L]
        input_norm = hf_layer.input_layernorm
        dn = hf_layer.linear_attn
        rms_eps_in = getattr(input_norm, "eps",
                              getattr(input_norm, "variance_epsilon", 1e-6))
        rms_eps_dn = getattr(dn.norm, "eps",
                              getattr(dn.norm, "variance_epsilon", 1e-6))
        h_in_b = saves["hidden_in"][L].unsqueeze(0).contiguous()
        npa_b  = saves["normalized_in"][L].unsqueeze(0).contiguous()
        _, layer_saves = dn_attn_forward(
            h_in_b,
            input_norm_w=input_norm.weight,
            in_proj_qkv_W=dn.in_proj_qkv.weight,
            in_proj_z_W=dn.in_proj_z.weight,
            in_proj_b_W=dn.in_proj_b.weight,
            in_proj_a_W=dn.in_proj_a.weight,
            conv1d_W=dn.conv1d.weight,
            A_log=dn.A_log, dt_bias=dn.dt_bias,
            dn_norm_W=dn.norm.weight,
            out_proj_W=dn.out_proj.weight,
            rms_eps=rms_eps_in, layer_norm_eps=rms_eps_dn,
            npa_precomputed=npa_b,
        )
        out[L] = layer_saves
    return out


def kernel_loss_autograd(
    handle: BaseModelHandle,
    prompt_tokens: torch.Tensor,     # [P]
    target_tokens: torch.Tensor,     # [T]
    lora_flat: list[torch.Tensor],
    lora_rank: int,
    lora_scaling: float,
    *,
    hf_model=None,
) -> dict:
    """Run kernel forward, then backward to d(h_pre_norm) via torch autograd.

    This is the entry point Slice B.3 builds on. Layout:

        kernel forward  -> h_pre_norm     (LEAF in autograd; opaque kernel)
        h_pre_norm      -> rms_norm()     (autograd-tracked python)
                        -> @ lm_head.T    (autograd-tracked python)
                        -> log_softmax    (autograd-tracked)
                        -> nll mean       (autograd-tracked)
        loss            -> loss.backward()
        h_pre_norm.grad = d(loss)/d(h_pre_norm) — the gradient flowing
                          into the per-layer reverse walk.

    For Slice B.3b the layer-walking backward consumes h_pre_norm.grad
    + the saved activations and produces grads on the LoRA params. This
    function is the foundation: validates that the upstream-of-layers
    autograd is set up correctly.

    Returns:
        loss              : scalar fp32 cuda tensor (autograd-tracked)
        grad_h_pre_norm   : [S, HIDDEN] fp32 cuda tensor — the gradient
                            d(loss)/d(h_pre_norm) for all S positions.
                            For positions outside `predict_pos` this is
                            zero (they don't contribute to the loss).
        h_pre_norm        : [S, HIDDEN] bf16 cuda tensor — the kernel's
                            pre-final-norm hidden, kept alive so the
                            caller can pass it to the layer-walking bwd.
        saves             : 6-slab activation save dict (B.1 + B.2).
        scratch           : kernel scratch buffers (kept alive).
        predict_pos       : torch.LongTensor of positions whose
                            predictions correspond to target_tokens.
    """
    P = int(prompt_tokens.numel())
    T = int(target_tokens.numel())
    if T == 0:
        raise ValueError("target_tokens must be non-empty")
    full = torch.cat([prompt_tokens, target_tokens], dim=0)
    S = int(full.numel())

    sc, saves = _train_step_forward(
        handle, full, lora_flat, lora_rank, lora_scaling,
    )

    # Detach and clone sc["hidden"] into a leaf tensor with requires_grad.
    # The kernel's own buffer is opaque to autograd; we want grad routing
    # to STOP here so we can read it explicitly.
    h_pre_norm = sc["hidden"].view(S, HIDDEN).detach().clone()
    h_pre_norm.requires_grad_(True)

    # Autograd-tracked python from h_pre_norm to loss.
    final_normed = _rms_norm_qwen(h_pre_norm, handle.final_norm_weight)
    predict_pos = torch.arange(P - 1, P - 1 + T, device="cuda")
    predict_logits = (final_normed[predict_pos].float()
                      @ handle.lm_head_weight.float().t())  # [T, VOCAB]
    log_probs = torch.nn.functional.log_softmax(predict_logits, dim=-1)
    logp = log_probs.gather(1, target_tokens.long().unsqueeze(1)).squeeze(1)
    loss = -logp.mean()

    grad_h_pre_norm, = torch.autograd.grad(loss, h_pre_norm,
                                            retain_graph=False,
                                            create_graph=False)

    # Pre-compute DN attention saves at forward time (eliminates the
    # bwd-side dn_attn_forward recompute). Per-layer Python forward
    # using fla directly. Saves are keyed by absolute layer index.
    if hf_model is not None:
        saves["dn_attn_saves"] = precompute_dn_saves(hf_model, saves)
    return {
        "loss": loss.detach(),
        "grad_h_pre_norm": grad_h_pre_norm,    # [S, HIDDEN] fp32
        "h_pre_norm": h_pre_norm.detach(),     # [S, HIDDEN] bf16
        "saves": saves,
        "scratch": sc,
        "predict_pos": predict_pos,
    }


__all__ = [
    "BaseModelHandle",
    "load_base_model",
    "alloc_scratch",
    "alloc_activation_saves",
    "kernel_sequence_loss",
    "kernel_loss_autograd",
    "per_position_final_normed",
    "MegakernelTrainStep",
]
