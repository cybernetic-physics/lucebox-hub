"""Decode latency breakdown using CUDA events.

For each decode step, times:
  - decode_qwen3x kernel (the megakernel that does all 64 layers)
  - lm_head_argmax kernel
  - Python overhead between them

Helps localize where the 121 ms / token actually goes after the
matvec inner-loop optimization frontier closed at 1.84x.

Run:
    HF_HOME=/home/sparkz/rl/.hf_cache \
        /home/sparkz/rl/.venv/bin/python3 \
        models/qwen36_27b/bench_decode_breakdown.py [--backend bf16|nvfp4]
"""
from __future__ import annotations
import argparse, os, sys, time
import torch

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="nvfp4", choices=("bf16", "nvfp4"))
    ap.add_argument("--n-steps", type=int, default=20)
    args = ap.parse_args()

    print("Loading HF Qwen3.6-27B...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B",
                                          trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True).eval()

    from runtime_megakernel import Qwen36MegakernelDecoder
    dec = Qwen36MegakernelDecoder(max_seq=128, verbose=False, hf_model=hf,
                                    tokenizer=tok, backend=args.backend)

    prompt = "The capital of France is"
    ids = tok(prompt, return_tensors="pt").input_ids[0].to(torch.int32).cuda()
    dec.prefill(ids)
    next_id = dec._argmax_from_normalized()

    # Warmup.
    for _ in range(3):
        next_id = dec.decode(next_id)
    torch.cuda.synchronize()

    # Time each step with CUDA events.
    print(f"\n[{args.backend}] timing {args.n_steps} decode steps...")
    ev_start = [torch.cuda.Event(enable_timing=True) for _ in range(args.n_steps)]
    ev_after_decode = [torch.cuda.Event(enable_timing=True) for _ in range(args.n_steps)]
    ev_after_lm = [torch.cuda.Event(enable_timing=True) for _ in range(args.n_steps)]

    ops = torch.ops.qwen3x_C

    for i in range(args.n_steps):
        ev_start[i].record()

        ops.decode_qwen3x(
            dec.MODEL_ID,
            dec.weights["embed_weight"], dec.weights["final_norm_weight"],
            dec.layer_blob,
            dec.sc.fa_k_cache, dec.sc.fa_v_cache,
            dec.sc.dn_states, dec.sc.conv_bufs,
            dec.sc.hidden_buffer, dec.sc.g_residual,
            dec.sc.g_qkv_scratch, dec.sc.g_kv_scratch,
            dec.sc.g_attn_out, dec.sc.g_mlp_inter,
            dec.sc.g_z_scratch, dec.sc.g_beta_scratch, dec.sc.g_alpha_scratch,
            dec.sc.g_normalized, dec.sc.g_fa_partials, dec.sc.g_rope_inv_freq,
            int(next_id), dec.position, dec.position, dec.position,
            dec.max_seq,
            float(dec.yarn["scale"]), float(dec.yarn["beta_fast"]),
            float(dec.yarn["beta_slow"]),
            int(dec.yarn["orig_ctx"]), bool(dec.yarn["enabled"]),
            int(dec.num_blocks),
            dec.layer_capture,
        )
        ev_after_decode[i].record()

        next_id = dec._argmax_from_normalized()
        ev_after_lm[i].record()

        dec.position += 1

    torch.cuda.synchronize()

    decode_ms = []
    lm_ms = []
    for i in range(args.n_steps):
        decode_ms.append(ev_start[i].elapsed_time(ev_after_decode[i]))
        lm_ms.append(ev_after_decode[i].elapsed_time(ev_after_lm[i]))

    d_mean = sum(decode_ms) / len(decode_ms)
    l_mean = sum(lm_ms) / len(lm_ms)
    total = d_mean + l_mean
    print(f"\n  decode_qwen3x kernel: {d_mean:6.2f} ms  ({d_mean/total*100:5.1f}%)")
    print(f"  lm_head_argmax + .item(): {l_mean:6.2f} ms  ({l_mean/total*100:5.1f}%)")
    print(f"  total per step:        {total:6.2f} ms  ({1000/total:5.2f} tok/s)")
    print(f"\n  decode_qwen3x stddev: {(max(decode_ms)-min(decode_ms)):.2f} ms range")


if __name__ == "__main__":
    main()
