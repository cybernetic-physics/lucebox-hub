"""Validate dn_chunk_parallel_fwd against dn_chunked_fwd.

Both kernels implement the same chunked-delta-rule algorithm. The
existing kernel processes all chunks of one head sequentially in a
single block (low SM occupancy at 16 heads). The new kernel processes
one chunk per block with state passed through global memory and an
atomic counter enforcing per-head sequential order — should produce
bit-identical output (modulo FP non-associativity in cuBLAS-tuned
WMMA tiles, which is bf16-noise level).

We require:
  - max|Δ y|  < 1e-3 (bf16 unit roundoff scale)
  - cosine(y_chunked, y_parallel) > 0.99999
  - state_out match (both end-of-sequence states)
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/root/lucebox-hub-b200-train/models/qwen35_0p8b/trainer")

import torch  # noqa: E402

import train_megakernel_C  # noqa: F401, E402


# Match dn_chunked.cu defaults.
DN_HEADS  = 16
DN_KEY    = 128
DN_VAL    = 128
CHUNK_SIZE = 64


def make_inputs(S: int, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    H = DN_HEADS
    Dk = DN_KEY
    Dv = DN_VAL
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    f32 = dict(dtype=torch.float32, device="cuda")
    q = torch.randn(S, H, Dk, generator=g, **bf16) * 0.5
    k = torch.randn(S, H, Dk, generator=g, **bf16) * 0.5
    v = torch.randn(S, H, Dv, generator=g, **bf16) * 0.5
    beta = torch.rand(S, H, generator=g, **f32) * 0.1
    g_log = torch.randn(S, H, generator=g, **f32) * 0.05
    state_in = torch.zeros(H, Dk, Dv, **f32)
    return q, k, v, beta, g_log, state_in


def run_dn_chunked_fwd(q, k, v, beta, g_log, state_in):
    S, H, Dk = q.shape
    Dv = v.shape[2]
    n_chunks = (S + CHUNK_SIZE - 1) // CHUNK_SIZE
    y = torch.zeros(S, H, Dv, dtype=torch.bfloat16, device="cuda")
    state_out = torch.zeros(H, Dk, Dv, dtype=torch.float32, device="cuda")
    state_chunks = torch.empty(H, n_chunks + 1, Dk, Dv,
                               dtype=torch.float32, device="cuda")
    torch.ops.train_megakernel_C.dn_chunked_fwd(
        q, k, v, beta, g_log, state_in, y, state_out, state_chunks,
    )
    return y, state_out, state_chunks


def run_dn_chunk_parallel_fwd(q, k, v, beta, g_log, state_in):
    S, H, Dk = q.shape
    Dv = v.shape[2]
    n_chunks = (S + CHUNK_SIZE - 1) // CHUNK_SIZE
    y = torch.zeros(S, H, Dv, dtype=torch.bfloat16, device="cuda")
    state_out = torch.zeros(H, Dk, Dv, dtype=torch.float32, device="cuda")

    # Caller-managed: state_chunks pre-filled at [:, 0] with state_in,
    # chunk_counter pre-zeroed.
    state_chunks = torch.empty(H, n_chunks + 1, Dk, Dv,
                               dtype=torch.float32, device="cuda")
    state_chunks[:, 0].copy_(state_in)
    chunk_counter = torch.zeros(H, dtype=torch.uint32, device="cuda")

    torch.ops.train_megakernel_C.dn_chunk_parallel_fwd(
        q, k, v, beta, g_log, y, state_out, state_chunks, chunk_counter,
    )
    return y, state_out, state_chunks


def compare(y_a, y_b, label):
    diff = (y_a.float() - y_b.float()).abs()
    cos = torch.nn.functional.cosine_similarity(
        y_a.float().flatten().unsqueeze(0),
        y_b.float().flatten().unsqueeze(0), dim=-1).item()
    print(f"  {label:20s} max|Δ|={diff.max().item():.4e}  "
          f"mean|Δ|={diff.mean().item():.4e}  cos={cos:.6f}")
    return diff.max().item(), cos


def main():
    fails = []
    for S in [64, 128, 256, 512, 1024, 2048, 4096]:
        print(f"\n--- S={S} (n_chunks={(S+CHUNK_SIZE-1)//CHUNK_SIZE}) ---")
        q, k, v, beta, g_log, state_in = make_inputs(S, seed=S)

        # Reference: existing chunked kernel.
        y_ref, state_ref, _ = run_dn_chunked_fwd(q, k, v, beta, g_log, state_in)

        # New: chunk-parallel kernel.
        y_par, state_par, _ = run_dn_chunk_parallel_fwd(q, k, v, beta, g_log, state_in)

        max_y, cos_y = compare(y_ref, y_par, "y")
        max_s, cos_s = compare(state_ref, state_par, "state_out")

        ok = max_y < 1e-3 and cos_y > 0.99999 and max_s < 1e-3 and cos_s > 0.99999
        if not ok:
            fails.append(f"S={S}: y max|Δ|={max_y:.2e} cos={cos_y:.5f}, "
                         f"state max|Δ|={max_s:.2e} cos={cos_s:.5f}")
        else:
            print(f"  PASS")

    print()
    print("=" * 60)
    if not fails:
        print("ALL CHECKS PASSED ✓  dn_chunk_parallel_fwd == dn_chunked_fwd")
    else:
        print("FAIL:")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
