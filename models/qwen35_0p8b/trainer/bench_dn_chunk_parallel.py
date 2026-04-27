"""Bench dn_chunk_parallel_fwd vs dn_chunked_fwd vs fla.chunk_gated_delta_rule.

The chunk-parallel kernel splits the existing one-block-per-head
architecture into one block per (head, chunk), giving us min(SMs,
H × n_chunks) blocks in flight instead of H = 16. At S ≥ 1K we expect
the new kernel to win because B200 has 148 SMs of which only 16 were
busy in the old kernel.
"""
from __future__ import annotations

import argparse
import sys
import time

# Shim so fla can be imported without breaking transformers' fla detect.
sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")
import _fla_torch_compile_shim  # noqa: F401

import torch  # noqa: E402

import train_megakernel_C  # noqa: F401, E402

DN_HEADS = 16
DN_KEY = 128
DN_VAL = 128
CHUNK_SIZE = 64


def time_fn(fn, runs, warm):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / runs


def make_inputs(S):
    g = torch.Generator(device="cuda").manual_seed(S)
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    f32 = dict(dtype=torch.float32, device="cuda")
    q = torch.randn(S, DN_HEADS, DN_KEY, generator=g, **bf16) * 0.5
    k = torch.randn(S, DN_HEADS, DN_KEY, generator=g, **bf16) * 0.5
    v = torch.randn(S, DN_HEADS, DN_VAL, generator=g, **bf16) * 0.5
    beta = torch.rand(S, DN_HEADS, generator=g, **f32) * 0.1
    g_log = torch.randn(S, DN_HEADS, generator=g, **f32) * 0.05
    state_in = torch.zeros(DN_HEADS, DN_KEY, DN_VAL, **f32)
    return q, k, v, beta, g_log, state_in


def make_chunked_call(q, k, v, beta, g_log, state_in):
    S = q.shape[0]
    H = q.shape[1]
    n_chunks = (S + CHUNK_SIZE - 1) // CHUNK_SIZE
    y = torch.empty(S, H, DN_VAL, dtype=torch.bfloat16, device="cuda")
    state_out = torch.empty(H, DN_KEY, DN_VAL, dtype=torch.float32, device="cuda")
    state_chunks = torch.empty(H, n_chunks + 1, DN_KEY, DN_VAL,
                                dtype=torch.float32, device="cuda")
    def call():
        torch.ops.train_megakernel_C.dn_chunked_fwd(
            q, k, v, beta, g_log, state_in, y, state_out, state_chunks)
    return call


def make_parallel_call(q, k, v, beta, g_log, state_in):
    S = q.shape[0]
    H = q.shape[1]
    n_chunks = (S + CHUNK_SIZE - 1) // CHUNK_SIZE
    y = torch.empty(S, H, DN_VAL, dtype=torch.bfloat16, device="cuda")
    state_out = torch.empty(H, DN_KEY, DN_VAL, dtype=torch.float32, device="cuda")
    state_chunks = torch.empty(H, n_chunks + 1, DN_KEY, DN_VAL,
                                dtype=torch.float32, device="cuda")
    chunk_counter = torch.zeros(H, dtype=torch.uint32, device="cuda")
    def call():
        # Per-call setup: counter must be zeroed, state_chunks[:, 0] = state_in.
        chunk_counter.zero_()
        state_chunks[:, 0].copy_(state_in)
        torch.ops.train_megakernel_C.dn_chunk_parallel_fwd(
            q, k, v, beta, g_log, y, state_out, state_chunks, chunk_counter)
    return call


def make_fla_call(q, k, v, beta, g_log, state_in):
    """fla expects q/k/v shape [B, H, S, D] and beta/g shape [B, H, S]."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    S = q.shape[0]
    H = q.shape[1]
    qf = q.permute(1, 0, 2).unsqueeze(0).contiguous()  # [1, H, S, Dk]
    kf = k.permute(1, 0, 2).unsqueeze(0).contiguous()
    vf = v.permute(1, 0, 2).unsqueeze(0).contiguous()
    bf = beta.permute(1, 0).unsqueeze(0).contiguous()  # [1, H, S]
    gf = g_log.permute(1, 0).unsqueeze(0).contiguous()
    sf = state_in.unsqueeze(0).contiguous()             # [1, H, Dk, Dv]
    def call():
        with torch.no_grad():
            chunk_gated_delta_rule(qf, kf, vf, gf, bf,
                                    initial_state=sf,
                                    output_final_state=False,
                                    use_qk_l2norm_in_kernel=False)
    return call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-lens", type=int, nargs="+",
                    default=[128, 256, 512, 1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--include-fla", dest="include_fla", action="store_true", default=False)
    ap.add_argument("--no-fla", dest="include_fla", action="store_false")
    args = ap.parse_args()

    print(f"{'S':>6} | {'chunked':>10} {'parallel':>10} {'speedup':>8}"
          + (f" | {'fla':>10} {'par/fla':>9}" if args.include_fla else ""))
    print("-" * (70 if args.include_fla else 42))

    for S in args.seq_lens:
        q, k, v, beta, g_log, state_in = make_inputs(S)
        runs = 5 if S <= 1024 else 3 if S <= 8192 else 2
        warm = 2 if S <= 1024 else 1

        ms_chunked = time_fn(make_chunked_call(q, k, v, beta, g_log, state_in),
                              runs=runs, warm=warm)
        ms_parallel = time_fn(make_parallel_call(q, k, v, beta, g_log, state_in),
                               runs=runs, warm=warm)

        speedup = ms_chunked / ms_parallel
        line = f"{S:>6} | {ms_chunked:>9.3f}ms {ms_parallel:>9.3f}ms {speedup:>7.2f}x"
        if args.include_fla:
            try:
                ms_fla = time_fn(make_fla_call(q, k, v, beta, g_log, state_in),
                                  runs=runs, warm=warm)
                par_vs_fla = ms_fla / ms_parallel
                line += f" | {ms_fla:>9.3f}ms {par_vs_fla:>8.2f}x"
            except Exception as e:
                line += f" | (fla error: {str(e)[:30]})"
        print(line)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
