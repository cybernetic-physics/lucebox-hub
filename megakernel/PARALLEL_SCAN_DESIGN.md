# Parallel-scan DeltaNet — design notes

## Why we need this

The post-FA-2 prefill profile on B200 has the DN recurrence as the
binding constraint at long S. SGLang routes DN through fla's
chunked + parallel-scan kernel and is 5–6× faster than our V-split
recurrence at S ≥ 8k. Our chunked port (`dn_chunked.cu`) is also
behind because it sequentially propagates state across chunks —
`n_chunks` deep, where `n_chunks = S / 64`.

The mathematically clean fix is a *parallel scan* across chunks,
turning a strictly serial dependency of depth `n_chunks` into a
tree-shaped dependency of depth `log2(n_chunks)`.

## The math (locked in `dn_parallel_scan_proto.py`)

The gated delta rule

```
state_{t+1} = decay_t * state_t
              + beta_t * k_t @ (v_t - decay_t * (k_t.T @ state_t)).T
```

is a *linear* update on the state:

```
state_{t+1} = A_t @ state_t + B_t
A_t = decay_t * (I - beta_t * k_t @ k_t.T)        # [Dk, Dk]
B_t = beta_t * k_t @ v_t.T                         # [Dk, Dv]
```

The composition

```
(A2, B2) ∘ (A1, B1) = (A2 @ A1, A2 @ B1 + B2)
```

is associative, so a parallel prefix scan over `(A_t, B_t)` lets
each step `t` recover its true `state_t` in O(log T) depth.

`dn_parallel_scan_proto.py` validates this against the sequential
recurrence to 3e-15 in fp64 on randomized input.

## Naive implementation cost

Per-step operator storage:
- `A_t` : [Dk, Dk] = 16384 fp32 = 64 KB
- `B_t` : [Dk, Dv] = 16384 fp32 = 64 KB

For T=8192 and 18 DN layers: 1 GB × 18 = 18 GB of operator state
just for the linear form. Not viable.

## Chunked parallel scan (the real plan)

Compute scan operators at *chunk* granularity, not per-token. With
chunks of C=64 tokens:

- `n_chunks = S / 64` (128 chunks at S=8k).
- Per chunk: compute `(A_chunk, B_chunk)` representing the chunk's
  cumulative operator.
- Parallel scan across `n_chunks` operators: depth `log2(n_chunks)`
  = 7 at S=8k.
- Per chunk in parallel: read prefix state, compute outputs `y[chunk]`.

Storage: 128 chunks × 128 KB = 16 MB per layer × 18 = 288 MB.
Acceptable.

### Chunk operator structure

A naive chunk operator is `(A_chunk, B_chunk)` of size
[Dk, Dk] + [Dk, Dv]. But the chunked formulation allows a more
compact form. Inside a chunk:

```
state_after = exp_g_total * state_before + k_chunk.T @ T_chunk @ v_chunk_modified
```

where
- `exp_g_total` is a scalar (cumulative decay across the chunk).
- `T_chunk = (I - tril(k_beta @ k.T * decay_mask))^{-1}` is a
  `[C, C]` lower-triangular matrix that depends only on the chunk's
  own `k`, `beta`, `g`.
- `v_chunk_modified` already incorporates the chunk-internal updates.

The chunk operator can therefore be encoded as
`(exp_g_total, T_chunk, k_chunk, v_chunk_modified)` — far smaller
than the full `(A_chunk, B_chunk)`.

### Composing two chunk operators

For two adjacent chunks `c1` then `c2`, the combined operator's
effect on a starting state `s0` is:

```
s_after_c2 = exp_g_total_2 * (exp_g_total_1 * s0
                              + k1.T @ T1 @ v1_mod)
             + k2.T @ T2 @ v2_mod
```

The composition requires a single matmul per composition (the
`k2 @ k1.T` cross-term that updates `T_chunk` for the merged
operator). This is the per-stage cost of the parallel scan.

## Implementation plan

The full implementation is multi-day kernel work. Sketched stages:

1. **Phase A: per-chunk operator extraction kernel** *(few hours)*
   Compute `(exp_g_total, T_chunk, k_chunk, v_chunk_modified)` for
   each chunk in parallel. Grid: `(H, n_chunks)`. Per-block work is
   exactly what the existing `dn_chunked.cu` does inside one chunk.
   Output: arrays in global memory, sized for log-tree storage.

2. **Phase B: parallel scan kernel** *(1–2 days)*
   Hillis-Steele or Blelloch over the chunk operators.
   `log2(n_chunks)` cooperative-grid-sync stages. Each stage's
   composition uses tensor-core matmuls (WMMA on sm_86, ideally
   tcgen05.mma on sm_100 for further speedup). The composition's
   cross-term `k2 @ k1.T` is a `[C, C]` matmul — perfectly sized
   for two WMMA tiles.

3. **Phase C: per-chunk output kernel** *(few hours)*
   Each chunk reads its prefix state from the scanned operators and
   computes y outputs in parallel. Grid: `(H, n_chunks)`. Per-block
   this is the existing chunked-DN output computation.

4. **Wiring + correctness** *(1 day)*
   Replace `pf_deltanet_recurrence_vsplit_prepped` call in
   `prefill.cu`'s DN body with the three-phase parallel-scan
   pipeline. Validate bit-exact against the recurrence path on the
   pp520 reference prompt.

5. **Bench + tune** *(1 day)*
   At S ≥ 4k the parallel scan should beat the serial chunked DN
   significantly. The log-depth path means scaling to S=64k stays
   linear in n_chunks but log in serial wall time. Tune chunk size
   C and parallel-scan grid for each arch.

## Expected reach

At S=8k on B200, the DN portion would drop from a ~50% prefill
share with serial chunk propagation to a ~15% share with log-depth
scan. Total prefill at S=8k: ~80–120 k tok/s (vs current 38k vs
SGLang's 217k) — closing the gap by a factor of 2–3×.

To fully match or beat SGLang at long S still likely needs
`tcgen05.mma` GEMMs in the scan composition (the matmuls are small
enough that WMMA's 16×16×16 tile is suboptimal for utilization; the
warp-group `tcgen05.mma` shapes on sm_100 are designed for exactly
this regime).

## Status

- `dn_parallel_scan_proto.py`: math reference, validated.
- CUDA kernel: queued. Multi-day focused engineering.
