"""Run the kernel-bwd path twice with identical inputs.
Capture every intermediate tensor inside layer_attn_bwd_fa_handrolled
at layer 15. Find the first op whose output differs between trials.

The forward saves are bit-identical across trials (already verified).
The dh entering the layer is bit-identical. The dh exiting differs.
Therefore some op between Step 1 and Step 9 produces different output
on the second run with the same inputs.

We monkey-patch lora_layer_bwd_skel to record all intermediates per call.
"""
from __future__ import annotations

import sys
import torch

sys.path.insert(0, "/home/freiza/lucebox-hub/models/qwen35_0p8b/trainer")
sys.path.insert(0, "/home/freiza/lucebox-hub/models/qwen35_0p8b")

import lora_layer_bwd_skel as bwd_skel
from rl_trainer import LoraMegakernelTrainer
from lora_megakernel_step import kernel_loss_autograd, load_base_model
from lora_pack import pack_peft_to_flat


# ------------- Instrument layer_attn_bwd_fa_handrolled -------------

_orig = bwd_skel.layer_attn_bwd_fa_handrolled
_capture_layer = None
_captures = []   # list of dicts, one per call


def instrumented(**kw):
    saved = {}
    # Step 1: residual split — trivial
    saved["dh_post_attn"] = kw["dh_post_attn"].clone()

    # Pull frequently-used.
    npa = kw["normalized_in"]
    saved["normalized_in"] = npa.clone()

    fa_o_save = kw["fa_o_save"]
    fa_q_save = kw["fa_q_save"]
    fa_lse_save = kw["fa_lse_save"]
    saved["fa_q_save"] = fa_q_save.clone()
    saved["fa_o_save"] = fa_o_save.clone()
    saved["fa_lse_save"] = fa_lse_save.clone()

    # Step 2: o_proj LoRA bwd.
    d_attn_pre_o_flat, grad_o_A, grad_o_B = bwd_skel.lora_linear_bwd(
        x=kw["attn_out_pre_o"], A=kw["o_A"], B=kw["o_B"], base_W=kw["o_W"],
        grad_y=kw["dh_post_attn"], scaling=kw["lora_scaling"],
    )
    torch.cuda.synchronize()
    saved["s2_d_attn_pre_o"] = d_attn_pre_o_flat.clone()
    saved["s2_grad_o_A"] = grad_o_A.clone()
    saved["s2_grad_o_B"] = grad_o_B.clone()

    S = kw["hidden_in"].shape[0]
    Hq = bwd_skel._FA_Q_HEADS
    D = bwd_skel._FA_HEAD_DIM
    Hk = bwd_skel._FA_KV_HEADS
    H_R = bwd_skel._FA_GQA

    d_attn_pre_o = d_attn_pre_o_flat.view(S, Hq, D)

    # Step 3: gate sigmoid.
    q_W = kw["q_W"]; q_A = kw["q_A"]; q_B = kw["q_B"]
    q_raw_recomp = npa @ q_W.t() + kw["lora_scaling"] * (npa @ q_A) @ q_B
    q_packed_recomp = q_raw_recomp.view(S, Hq, 2, D)
    Q_h_recomp = q_packed_recomp[:, :, 0, :].contiguous()
    Gate_recomp = q_packed_recomp[:, :, 1, :]
    sig = torch.sigmoid(Gate_recomp.float())
    attn_unfold = fa_o_save.float()
    d_attn_unfold = sig * d_attn_pre_o
    d_Gate = (sig * (1.0 - sig)) * attn_unfold * d_attn_pre_o
    torch.cuda.synchronize()
    saved["s3_q_raw_recomp"] = q_raw_recomp.clone()
    saved["s3_Q_h_recomp"] = Q_h_recomp.clone()
    saved["s3_sig"] = sig.clone()
    saved["s3_d_attn_unfold"] = d_attn_unfold.clone()
    saved["s3_d_Gate"] = d_Gate.clone()

    # Step 4: FA bwd.
    Q_bhsd = fa_q_save.permute(1, 0, 2).unsqueeze(0).contiguous()
    O_bhsd = fa_o_save.permute(1, 0, 2).unsqueeze(0).contiguous()
    dO_bhsd = d_attn_unfold.to(torch.bfloat16).permute(1, 0, 2).unsqueeze(0).contiguous()
    K_bhsd = kw["k_cache_layer_S"].unsqueeze(0).contiguous()
    V_bhsd = kw["v_cache_layer_S"].unsqueeze(0).contiguous()
    K_e = K_bhsd.repeat_interleave(H_R, dim=1)
    V_e = V_bhsd.repeat_interleave(H_R, dim=1)
    LSE_bhs = fa_lse_save.unsqueeze(0).contiguous()
    import math
    scale = 1.0 / math.sqrt(D)
    fa_backward_flash = bwd_skel._fa_bwd_flash()
    dQ_bhsd, dK_bhsd, dV_bhsd = fa_backward_flash(
        dO_bhsd, Q_bhsd, K_e, V_e, O_bhsd, LSE_bhs,
        is_causal=True, scale=scale, num_kv_heads=Hk,
    )
    torch.cuda.synchronize()
    saved["s4_Q_bhsd"] = Q_bhsd.clone()
    saved["s4_dO_bhsd"] = dO_bhsd.clone()
    saved["s4_dQ_bhsd"] = dQ_bhsd.clone()
    saved["s4_dK_bhsd"] = dK_bhsd.clone()
    saved["s4_dV_bhsd"] = dV_bhsd.clone()

    dQ_post = dQ_bhsd.squeeze(0).permute(1, 0, 2).contiguous()
    dK_post = dK_bhsd.squeeze(0).permute(1, 0, 2).contiguous()
    dV_h = dV_bhsd.squeeze(0).permute(1, 0, 2).contiguous()

    # Step 5-6: rope_bwd + qknorm bwd.
    dQ_normed = bwd_skel._qwen_rope_bwd(dQ_post)
    dK_normed = bwd_skel._qwen_rope_bwd(dK_post)
    Q_h_flat = Q_h_recomp.view(S * Hq, D).contiguous()
    dQ_normed_flat = dQ_normed.contiguous().view(S * Hq, D)
    dQ_h_flat = bwd_skel.rmsnorm_bwd(Q_h_flat, kw["q_nw"], dQ_normed_flat,
                                       eps=kw["rms_eps"])
    dQ_h = dQ_h_flat.view(S, Hq, D)

    k_W = kw["k_W"]; k_A = kw["k_A"]; k_B = kw["k_B"]
    k_raw_recomp = npa @ k_W.t() + kw["lora_scaling"] * (npa @ k_A) @ k_B
    K_h_recomp = k_raw_recomp.view(S, Hk, D).contiguous()
    K_h_flat = K_h_recomp.view(S * Hk, D)
    dK_normed_flat = dK_normed.contiguous().view(S * Hk, D)
    dK_h_flat = bwd_skel.rmsnorm_bwd(K_h_flat, kw["k_nw"], dK_normed_flat,
                                       eps=kw["rms_eps"])
    dK_h = dK_h_flat.view(S, Hk, D)
    torch.cuda.synchronize()
    saved["s5_dQ_normed"] = dQ_normed.clone()
    saved["s5_dK_normed"] = dK_normed.clone()
    saved["s6_dQ_h"] = dQ_h.clone()
    saved["s6_dK_h"] = dK_h.clone()
    saved["s6_K_h_recomp"] = K_h_recomp.clone()
    saved["s6_k_raw_recomp"] = k_raw_recomp.clone()

    # Step 7: pack dq_raw.
    dq_packed = torch.empty(S, Hq, 2, D, dtype=torch.float32, device="cuda")
    dq_packed[:, :, 0, :] = dQ_h
    dq_packed[:, :, 1, :] = d_Gate
    dq_raw = dq_packed.reshape(S, bwd_skel._FA_QPROJ_SIZE).contiguous()
    dk_raw = dK_h.reshape(S, bwd_skel._FA_KV_SIZE).contiguous()
    dv_raw = dV_h.reshape(S, bwd_skel._FA_KV_SIZE).to(torch.float32).contiguous()
    torch.cuda.synchronize()
    saved["s7_dq_raw"] = dq_raw.clone()
    saved["s7_dk_raw"] = dk_raw.clone()
    saved["s7_dv_raw"] = dv_raw.clone()

    # Step 8: q/k/v LoRA bwd.
    v_W = kw["v_W"]; v_A = kw["v_A"]; v_B = kw["v_B"]
    dnpa_q, grad_q_A, grad_q_B = bwd_skel.lora_linear_bwd(
        x=npa, A=q_A, B=q_B, base_W=q_W, grad_y=dq_raw,
        scaling=kw["lora_scaling"])
    dnpa_k, grad_k_A, grad_k_B = bwd_skel.lora_linear_bwd(
        x=npa, A=k_A, B=k_B, base_W=k_W, grad_y=dk_raw,
        scaling=kw["lora_scaling"])
    dnpa_v, grad_v_A, grad_v_B = bwd_skel.lora_linear_bwd(
        x=npa, A=v_A, B=v_B, base_W=v_W, grad_y=dv_raw,
        scaling=kw["lora_scaling"])
    dnpa = dnpa_q + dnpa_k + dnpa_v
    torch.cuda.synchronize()
    saved["s8_dnpa_q"] = dnpa_q.clone()
    saved["s8_dnpa_k"] = dnpa_k.clone()
    saved["s8_dnpa_v"] = dnpa_v.clone()
    saved["s8_dnpa"] = dnpa.clone()

    # Step 9: input rmsnorm bwd.
    dh_in_through_norm = bwd_skel.rmsnorm_bwd(
        x=kw["hidden_in"], w=kw["input_norm_w"],
        dy=dnpa, eps=kw["rms_eps"])
    dh_in = kw["dh_post_attn"] + dh_in_through_norm
    torch.cuda.synchronize()
    saved["s9_dh_in_through_norm"] = dh_in_through_norm.clone()
    saved["s9_dh_in"] = dh_in.clone()

    _captures.append(saved)
    return {
        "dh_in":      dh_in,
        "grad_q_A":   grad_q_A,   "grad_q_B":   grad_q_B,
        "grad_k_A":   grad_k_A,   "grad_k_B":   grad_k_B,
        "grad_v_A":   grad_v_A,   "grad_v_B":   grad_v_B,
        "grad_o_A":   grad_o_A,   "grad_o_B":   grad_o_B,
    }


def main():
    bwd_skel.layer_attn_bwd_fa_handrolled = instrumented

    trainer = LoraMegakernelTrainer(verbose_loader=False)
    trainer.register_model(
        model_id="t",
        base_model="Qwen/Qwen3.5-0.8B",
        lora_rank=8,
        train_mlp=True, train_attn=True, train_unembed=False,
        user_metadata=None,
    )
    handle = load_base_model("Qwen/Qwen3.5-0.8B", verbose=False)
    s = trainer._sessions["t"]
    lora_flat = pack_peft_to_flat(s.hf_model, 8)
    prompt_t = torch.tensor(list(range(10, 30)), dtype=torch.long, device="cuda")
    target_t = torch.tensor(list(range(100, 110)), dtype=torch.long, device="cuda")

    # We process layers in reverse order: 23, 22, 21, ..., 0.
    # Layer 23 (FA, fa_idx=5) is the first FA-handrolled call per run.
    # Layer 19 fa_idx=4, 15 fa_idx=3, 11 fa_idx=2, 7 fa_idx=1, 3 fa_idx=0.
    # Each run calls instrumented() 6 times. We want trials 1's calls
    # vs trial 2's.

    for trial in range(2):
        out = kernel_loss_autograd(
            handle=handle, prompt_tokens=prompt_t, target_tokens=target_t,
            lora_flat=lora_flat, lora_rank=8, lora_scaling=2.0,
            hf_model=s.hf_model,
        )
        flat_grads = bwd_skel.run_layer_walking_bwd(
            grad_h_pre_norm=out["grad_h_pre_norm"], saves=out["saves"],
            lora_flat=lora_flat,
            final_norm_weight=handle.final_norm_weight,
            hf_model=s.hf_model, lora_rank=8, lora_scaling=2.0,
            fa_k_cache=out["scratch"]["fa_k_cache"],
            fa_v_cache=out["scratch"]["fa_v_cache"],
        )
        torch.cuda.synchronize()
    print(f"captured {len(_captures)} layer-FA-bwd calls "
          f"(expected 12 = 6 FA layers × 2 trials)")

    # Compare trial 1 vs trial 2 for each FA-layer position.
    # layout: trial1[0..5] are calls 0..5, trial2[6..11] are same indices.
    n_per_trial = 6
    for fa_call_idx in range(n_per_trial):
        a = _captures[fa_call_idx]
        b = _captures[n_per_trial + fa_call_idx]
        diverged = []
        first_divergence = None
        for k in a:
            ta, tb = a[k], b[k]
            if not torch.is_tensor(ta) or not torch.is_tensor(tb):
                continue
            if ta.shape != tb.shape:
                diverged.append((k, "shape mismatch"))
                continue
            if ta.dtype != tb.dtype:
                diverged.append((k, "dtype mismatch"))
                continue
            if ta.float().equal(tb.float()):
                continue
            mx = float((ta.float() - tb.float()).abs().max())
            diverged.append((k, mx))
            if first_divergence is None:
                first_divergence = (k, mx)
        # Layer order: call_idx 0 = layer 23 (last), 5 = layer 3 (first).
        layer_id = [23, 19, 15, 11, 7, 3][fa_call_idx]
        if not diverged:
            print(f"FA call {fa_call_idx} (layer {layer_id}): bit-identical across trials")
        else:
            keys_in_order = [
                "dh_post_attn", "normalized_in", "fa_q_save", "fa_o_save",
                "fa_lse_save",
                "s2_d_attn_pre_o", "s2_grad_o_A", "s2_grad_o_B",
                "s3_q_raw_recomp", "s3_Q_h_recomp", "s3_sig",
                "s3_d_attn_unfold", "s3_d_Gate",
                "s4_Q_bhsd", "s4_dO_bhsd", "s4_dQ_bhsd", "s4_dK_bhsd", "s4_dV_bhsd",
                "s5_dQ_normed", "s5_dK_normed",
                "s6_dQ_h", "s6_dK_h", "s6_K_h_recomp", "s6_k_raw_recomp",
                "s7_dq_raw", "s7_dk_raw", "s7_dv_raw",
                "s8_dnpa_q", "s8_dnpa_k", "s8_dnpa_v", "s8_dnpa",
                "s9_dh_in_through_norm", "s9_dh_in",
            ]
            d_dict = dict(diverged)
            print(f"FA call {fa_call_idx} (layer {layer_id}): divergent intermediates")
            for k in keys_in_order:
                if k in d_dict:
                    delta = d_dict[k]
                    print(f"    *** {k}: max|Δ|={delta:.4e}")
                    break
            else:
                # No known key matched
                pass


if __name__ == "__main__":
    main()
