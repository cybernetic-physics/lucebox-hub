"""Smoke test: random weights, run the megakernel, check we get an int back.

This doesn't validate numerical correctness (that needs real weights or a
diff against prefill.cu); it catches launch/sync/OOB bugs that would hang
or crash the kernel.
"""

from __future__ import annotations

import sys
import torch

from model import (
    NUM_LAYERS, HIDDEN, VOCAB, LAYER_TYPE,
    FA_QPROJ_SIZE, FA_KV_SIZE, FA_HEAD_DIM,
    DN_CONV_CH, DN_V_SIZE, DN_HEADS, DN_VAL, DN_CONV_K,
    INTER,
    PrefillMegakernel,
)


def _rand_bf16(*shape):
    return (torch.randn(*shape) * 0.02).to(device="cuda", dtype=torch.bfloat16)


def random_weights():
    layer_data = []
    for i in range(NUM_LAYERS):
        if LAYER_TYPE[i] == 1:
            ptrs = [
                _rand_bf16(HIDDEN),             # input_norm
                _rand_bf16(FA_QPROJ_SIZE, HIDDEN),
                _rand_bf16(FA_KV_SIZE, HIDDEN),
                _rand_bf16(FA_KV_SIZE, HIDDEN),
                _rand_bf16(FA_HEAD_DIM),        # q_norm
                _rand_bf16(FA_HEAD_DIM),        # k_norm
                _rand_bf16(HIDDEN, 8 * FA_HEAD_DIM),
                _rand_bf16(HIDDEN),             # post_norm
                _rand_bf16(INTER, HIDDEN),
                _rand_bf16(INTER, HIDDEN),
                _rand_bf16(HIDDEN, INTER),
            ]
            layer_data.append({"type": 1, "ptrs": ptrs})
        else:
            ptrs = [
                _rand_bf16(HIDDEN),              # input_norm
                _rand_bf16(DN_CONV_CH, HIDDEN),  # qkv
                _rand_bf16(DN_V_SIZE, HIDDEN),   # z
                _rand_bf16(DN_HEADS, HIDDEN),    # beta proj
                _rand_bf16(DN_HEADS, HIDDEN),    # alpha proj
                _rand_bf16(DN_CONV_CH, DN_CONV_K),
                _rand_bf16(DN_HEADS),            # A_log
                _rand_bf16(DN_HEADS),            # dt_bias
                _rand_bf16(DN_VAL),              # dn norm
                _rand_bf16(HIDDEN, DN_V_SIZE),   # out
                _rand_bf16(HIDDEN),              # post_norm
                _rand_bf16(INTER, HIDDEN),
                _rand_bf16(INTER, HIDDEN),
                _rand_bf16(HIDDEN, INTER),
            ]
            layer_data.append({"type": 0, "ptrs": ptrs})

    return {
        "embed_weight": _rand_bf16(VOCAB, HIDDEN),
        "final_norm_weight": _rand_bf16(HIDDEN),
        "lm_head_weight": _rand_bf16(VOCAB, HIDDEN),
        "layer_data": layer_data,
    }


def main():
    torch.manual_seed(0)
    S = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    print(f"S = {S}")
    w = random_weights()
    m = PrefillMegakernel(w)
    tokens = torch.randint(0, VOCAB, (S,), dtype=torch.int32, device="cuda")
    torch.cuda.synchronize()
    print("launching...")
    nxt = m.forward(tokens)
    torch.cuda.synchronize()
    print(f"next token: {nxt}")
    assert 0 <= nxt < VOCAB, "token out of range"
    print("OK")


if __name__ == "__main__":
    main()
