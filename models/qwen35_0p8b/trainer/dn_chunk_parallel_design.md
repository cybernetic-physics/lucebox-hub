# DN chunk-parallel rewrite — design

## Why

`dn_chunked.cu`'s forward kernel uses Grid:(H,) — one block per head,
processing all S/C chunks sequentially inside the block. That occupies
H = 16 SMs out of B200's 148. fla's chunk-parallel architecture
launches one block per (chunk, head) — at S=2K, 32 chunks × 16 heads =
512 blocks, comfortably saturating the 148 SMs.

The bench data (`bench_vs_fla.py` 2026-04-27 sweep) shows we tie fla
at S ≥ 1K because the hybrid router routes long-S inference through
fla. To beat fla on long-S DN we need our own chunk-parallel kernel.

## Algorithmic invariants (preserved from dn_chunked.cu)

The chunked-delta-rule algorithm is unchanged. Per chunk c:

```
chunk inputs:
  q_c   [C, Dk]      = q [t_c .. t_c+C)
  k_c   [C, Dk]
  v_c   [C, Dv]
  beta_c [C]
  g_c    [C]         (log decay per step)

state in: state_c   [Dk, Dv]   fp32 (residual)
state out: state_c+1 [Dk, Dv]  fp32

per-chunk math (fla / dn_chunked_proto, line for line):
  k_beta = k * beta
  v_beta = v * beta
  attn0  = -(k_beta @ k.T) * decay_mask        (lower triangular, diag zeroed)
  T      = (I - tril(attn0))^(-1)               (sequential row update)
  v_new  = T @ v_beta - (T @ (k_beta * exp_g_cs)) @ state_c
  attn_in = q @ k.T * decay_mask                 (lower-and-diag)
  attn_int = (q * exp_g_cs) @ state_c
  y_chunk = attn_int + attn_in @ v_new

state_c+1 = state_c * exp(g_c_total)
            + (k * exp(g_c_total - g_cs))^T @ v_new
```

## What changes vs the existing dn_chunked.cu

| dimension                | dn_chunked.cu                  | chunk-parallel rewrite               |
|--------------------------|--------------------------------|--------------------------------------|
| Grid                     | (H,)                           | (H, n_chunks_max)                    |
| State location           | shared memory across chunks    | global memory between chunks         |
| Sync between chunks      | __syncthreads (same block)     | atomic barrier on per-head counter   |
| Per-block work           | All chunks of 1 head           | 1 chunk of 1 head                    |
| Shared mem per block     | 217 KB (state + working)       | ~150 KB (no persistent state slab)   |
| Concurrent SMs at S=2K   | 16                             | min(SM_count, H × n_chunks)          |
| Launches per fwd         | 1                              | 1 (persistent kernel)                |

## Concrete kernel layout

```cuda
__global__ __launch_bounds__(256, 1)
void dn_chunk_parallel_fwd(
    const __nv_bfloat16 *q, *k, *v,             // [S, H, Dk] / [S, H, Dv]
    const float *beta, *g,                       // [S, H]
    const float *state_in,                        // [H, Dk, Dv] fp32
    __nv_bfloat16 *y,                             // [S, H, Dv]
    float *state_out,                             // [H, Dk, Dv]
    float *state_chunks,                          // [H, n_chunks+1, Dk, Dv]
    unsigned int *chunk_counter,                  // [H] — per-head atomic counter
    int S, int H, int C)
{
    int head        = blockIdx.x;
    int my_chunk    = blockIdx.y;
    int n_chunks    = (S + C - 1) / C;
    if (head >= H || my_chunk >= n_chunks) return;

    // Wait for chunk_counter[head] == my_chunk (i.e. previous chunk finished).
    if (threadIdx.x == 0) {
        while (atomicAdd(&chunk_counter[head], 0u) < (unsigned)my_chunk) { /* spin */ }
    }
    __syncthreads();

    // Load state from global into shared.
    // (Index from state_chunks[head, my_chunk] which the previous chunk wrote.)
    __shared__ float state_smem[Dk * Dv];
    float *state_src = state_chunks + ((size_t)head * (n_chunks+1) + my_chunk) * Dk * Dv;
    for (int i = threadIdx.x; i < Dk*Dv; i += blockDim.x) state_smem[i] = state_src[i];
    __syncthreads();

    // ── Per-chunk math (unchanged from dn_chunked.cu, lines 240-540 ish) ──
    // load q/k/v/beta/g for [t_c, t_c+C)
    // compute attn0, T, v_new, attn_in, attn_int, y_chunk, state_next
    // write y_chunk to y[t_c:t_c+C, head, :]
    // …

    // Write state_next to global at state_chunks[head, my_chunk+1].
    float *state_dst = state_chunks + ((size_t)head * (n_chunks+1) + (my_chunk+1)) * Dk * Dv;
    for (int i = threadIdx.x; i < Dk*Dv; i += blockDim.x) state_dst[i] = state_smem[i];
    __threadfence();
    __syncthreads();

    // Signal next chunk.
    if (threadIdx.x == 0) atomicAdd(&chunk_counter[head], 1u);

    // Last chunk also writes state_out (consumed by next decode session).
    if (my_chunk == n_chunks - 1) {
        float *state_final = state_out + (size_t)head * Dk * Dv;
        for (int i = threadIdx.x; i < Dk*Dv; i += blockDim.x) state_final[i] = state_smem[i];
    }
}
```

## Memory cost

- `state_chunks`: [H, n_chunks+1, Dk, Dv] fp32. At S=2K, n_chunks=32:
  16 × 33 × 128 × 128 × 4 = 32 MB per layer × 18 DN layers = **576 MB**.
  At S=8K: 128 × 16 × 64 KB = 256 MB per layer × 18 = 4.6 GB.
  At S=32K: 4× larger = 18 GB.
- `chunk_counter`: [H] u32 = 64 bytes. Trivial.

For long-S training the state_chunks buffer is large but still fits on
B200. Inference doesn't need this buffer to be retained — could free
after forward.

## Sync correctness

The atomic `chunk_counter[head]` starts at 0. Block (head, my_chunk=0)
sees counter ≥ 0 immediately, runs, increments to 1. Block
(head, my_chunk=1) was spin-waiting for counter ≥ 1, now proceeds,
runs, increments to 2. And so on.

Risk: SM scheduler may schedule (head, 1) before (head, 0). The
spin-wait handles this — block 1 just waits for block 0 to finish.
The B200 scheduler is fair enough that this won't deadlock as long
as we have enough SMs for at least one block per head to run
concurrently (we do: 148 SMs ≥ 16 heads).

If we wanted to avoid spin-waits, we could use the cooperative groups
`grid_group::sync()` API, but that requires `cudaLaunchCooperativeKernel`
and limits grid size to what fits in residency simultaneously
(`cudaOccupancyMaxActiveBlocksPerMultiprocessor * SM_count`). At
H × n_chunks_max for any practical S, we don't fit, so spin-wait is
the right choice.

## Implementation steps

1. **Refactor existing dn_chunked.cu** — extract per-chunk body
   (lines ~240-540) into a `__device__` helper that takes pointers
   into shared memory.
2. **Write `dn_chunk_parallel_fwd` shell** — counter spin-wait, state
   load/store, calls the helper.
3. **Allocate state_chunks tensor** — caller (Python) allocates fp32
   buffer of size `[H, n_chunks+1, Dk, Dv]`.
4. **Launch kernel** with `dim3(H, n_chunks_max)` blocks. n_chunks_max
   = ceil_div(S, C); blocks where blockIdx.y >= actual n_chunks return
   immediately.
5. **Validation**: same as `dn_chunked.cu` — compare against fla and
   against `dn_chunked_proto.py` for cos > 0.999 on output.
6. **Bench** at S=128..32K vs fla. Should show parity at short S, win
   at long S where fla's per-call constants dominate over our extra
   per-chunk launch overhead in fla.

## Effort estimate

- Refactor + write parallel shell:           1-2 days
- Validation against fla / proto:            0.5-1 day
- Performance tuning (shared mem layout,
  bank-conflict audit, occupancy):           1-2 days
- Wire into `dn_hf_patch.py` hybrid router:  0.5 day
- Bench + results doc:                       0.5 day

**Total: 4-6 engineer-days** for a clean correct + fast version.

## Next steps when picked up

The starting point is to extract the existing per-chunk body in
`dn_chunked.cu` into a `__device__` function. Once that's done, the
chunk-parallel shell wraps it with the spin-wait + global state I/O.

After this lands, the chunk-parallel BACKWARD (Tier 1.6) can follow
the same pattern using `dn_chunked_bwd_proto.py` as the algorithm
spec — which is itself the next thing the trainer's full Tier 0.2
custom backward depends on for the DN bwd path.

## Algorithmic findings from fla source (2026-04-30)

To inform a 3090-targeted rewrite, audited fla's
`chunk_gated_delta_rule_fwd` decomposition. The 30% perf advantage at
S=32K vs our `dn_chunked_3090` traces to three structural choices we
should adopt:

1. **`BT=64` chunk size with `BC=BT/4=16` sub-chunks.** fla never
   materializes a 64×64 K@K^T matrix — it computes the lower
   triangle of `BT×BT` as 10 separate `BC×BC = 16×16` tile products
   (`chunk_fwd.py:117-120`: "all 10 lower-triangular [BC, BC] blocks
   of K @ K^T"). That avoids the smem blowup we hit when trying
   C=64 directly: 4 of our `C×Dk` bf16 buffers go from 8 KB → 16 KB
   each (+32 KB) and our `C×C` fp32 buffers go from 4 KB → 16 KB
   each (+24 KB), pushing the kernel from ~76 KB to ~140 KB at
   V_SPLITS=4 — well over 99 KB. fla's sub-chunk decomposition
   keeps the per-tile smem footprint at BC=16 sizes.
2. **Register-resident state.** fla holds the recurrence state
   in registers as `b_h1..b_h4 = tl.zeros([BV, 64], fp32)`
   (`chunk_delta_h.py:81-95`), accumulating across BK splits.
   `BV=32` and `BK=64` mean each register slab is 2048 fp32 = 8 KB
   in registers; at K=128 (our Dk) there are 2 BK splits, so 16 KB
   register state per block. Our kernel keeps state in shared mem
   (16 KB at V_SPLITS=4) and pays a `state_fp32 → state_bf16` cast
   every chunk to feed the wmma matmul. Going register-resident
   eliminates the cast traffic and the bf16 mirror of the state
   buffer (8 KB smem savings).
3. **TF32 matmul instead of bf16 cast-and-wmma.** fla uses
   `SOLVE_TRIL_DOT_PRECISION = tf32` on Ampere
   (`chunk_fwd.py:17-20`). TF32 is 1.5× slower per-op than bf16
   tensor cores on sm_86, but it eliminates the per-chunk fp32→bf16
   cast and works directly on register-resident state. For inputs
   that arrive as bf16 (q, k, v) we still cast on load, but the
   STATE accumulation stays fp32 throughout. Compared to our path
   (fp32 state in smem → bf16 cast in smem → wmma fp32 accumulator
   → write back fp32 in smem), fla saves 2 round-trip casts per
   chunk.

### Implications for the 3090 rewrite

The "parallel-scan" framing in the original design (sequential chunk
ordering with global-state spin-wait sync) is orthogonal to the
30% gap — fla's design is also sequential-per-head with the same
recurrence structure. The win is the per-chunk inner loop, not the
chunk parallelism. So:

- **Phase A (high impact):** rewrite `dn_chunked_3090.cu`'s per-chunk
  body using fla's sub-chunk + register-state + TF32 design. Keep
  the existing `(H × V_SPLITS=4)` grid layout (fits 82 SMs in one
  wave). Target: parity with fla on S=32K.
- **Phase B (lower impact):** ALSO add chunk-parallel grid scheme.
  Useful only on hardware where (H × V_SPLITS) doesn't already
  saturate the SM count — i.e. NOT a 3090 win, but plausibly a
  B200 win where 64 blocks leaves 84 SMs idle.

Phase A is the right thing for 3090 specifically. Phase B is
deferred until B200 work resumes.

### Effort estimate (revised)

- Phase A: refactor per-chunk body to register-resident state +
  TF32, with sub-chunk decomposition for K@K^T:
  3-4 engineer-days (kernel rewrite, validation against fla, tuning).
- Phase B (deferred): 1-2 days on top.

## Bench: actual gap to fla at long S (2026-04-30)

Updated bench (`experiments/bench_dn_chunked_3090.py` at H=16, Dk=Dv=128):

|     S |   fla ms |  ours ms | fla/ours |
|------:|---------:|---------:|---------:|
|   128 |     0.54 |     0.14 |    3.88x |
|   256 |     0.48 |     0.28 |    1.68x |
|   512 |     0.49 |     0.56 |    0.88x |
|  1024 |     0.47 |     1.10 |    0.43x |
|  2048 |     0.48 |     2.18 |    0.22x |
|  4096 |     0.53 |     4.35 |    0.12x |
|  8192 |     0.94 |     8.67 |    0.11x |
| 16384 |     1.77 |    17.30 |    0.10x |
| 32768 |     3.44 |    34.56 |    0.10x |

Note: fla's time is essentially CONSTANT for S ≤ 4K (~0.5 ms — launch-
bound) and only starts scaling at S ≥ 8K. Our kernel scales linearly
throughout (per-chunk overhead dominates).

At S=32K we're **10x slower per call** (34.6 vs 3.4 ms). Per layer
that's a 31 ms gap; across 18 DN layers that's 558 ms — slightly
larger than the 486 ms total prefill gap to SGLang at S=32K
(1740 ms vs 1254 ms). Closing this entirely would put us ahead of
SGLang at every shape.

## fla architecture audit (deeper than the chunk_fwd.py audit)

Reading `chunk_delta_h.py:chunk_gated_delta_rule_fwd_kernel_h_blockdim64`
revealed fla splits the work across **three kernels**, not one:

1. **`chunk_gated_delta_rule_fwd_intra`** (`chunk_fwd.py`) — per-chunk
   WY representation: produces `w`, `u`, `A` from `k`, `v`, `beta`, `g`.
   PARALLELIZED across chunks AND heads (no inter-chunk dependency).
   This is where the 10-block sub-chunked K@K^T + solve_tril runs.
2. **`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`** (`chunk_delta_h.py`)
   — the sequential recurrence over chunks. Reads pre-computed `w`,
   `u`; produces `h` (per-chunk states) and `v_new`. Per block holds
   state in registers as `b_h1..b_h4 = tl.zeros([BV, 64], fp32)`,
   one slab per BK split (K=128 → 2 slabs).
3. **`chunk_fwd_o`** (`chunk_o.py`) — output projection per chunk:
   `o_chunk = q @ k.T * mask @ v_new + (q * exp_g_cs) @ h_chunk`.
   PARALLELIZED across chunks.

Our `dn_chunked_3090` does ALL three phases in one big sequential
loop. That means our per-chunk inner loop carries:

  - the per-chunk K@K^T + solve_tril (~30% of per-chunk time)
  - the recurrence step (~30%)
  - the output projection step (~40%)

…all serialized inside the chunk dependence chain. fla pulls phases
1 and 3 OUT of the dependence chain — they only need to be done once
per chunk and don't depend on the inter-chunk state. With ~16K total
chunks at S=32K (n_chunks × H), Phase 1 and Phase 3 run as massively
parallel kernels saturating all 82 SMs, while only Phase 2 runs at
the 16-block-per-head bottleneck.

Memory: fla pays for the intermediates. At S=32K, H=16, K=V=128:
`A`: 128 MB, `w`: 128 MB, `u`: 128 MB, `h`: 540 MB. Total ~924 MB.
For inference this is large but fits in 24 GB. For training it adds
to the activation budget but is offset by avoiding the bwd recompute.

## Revised effort estimate (after deeper audit)

The Phase A "single-kernel rewrite" approach is INSUFFICIENT to
match fla's perf because the dependency chain is wrong. The right
target is fla's 3-kernel split:

- 3-kernel rewrite: 5-7 engineer-days
  - Phase 1 kernel (per-chunk WY): 2 days
  - Phase 2 kernel (sequential recurrence with register state): 2 days
  - Phase 3 kernel (per-chunk output projection): 1-2 days
  - Wiring + validation + tuning: 1 day
- Memory budget audit on the 24GB constraint: 0.5 day

Pragmatic alternative: **route DN inference at S ≥ 1024 to
fla.chunk_gated_delta_rule** via a Python-side hook in the prefill
flow. This requires breaking the C++ prefill into "before DN" /
"DN" / "after DN" phases at the Python boundary, but reuses fla's
already-tuned kernels and avoids the 5-7 day rewrite. Estimate
1-2 days for the Python integration + graph capture rework.

## Build-spec for the 3-kernel rewrite (next session)

When picking this up, the work is structured as follows:

### Files to create

1. `models/qwen35_0p8b/dn_chunked_3090_v2.cu` — new file with three
   `__global__` kernels and one `launch_dn_chunked_3090_v2` entry
   that orchestrates them. Keep the original `dn_chunked_3090.cu`
   intact; gate v2 behind `MEGAKERNEL_DN_USE_PARALLEL_SCAN=1`.
2. `experiments/test_dn_chunked_3090_v2.py` — bit-correctness test
   vs `dn_chunked_3090` (original) AND vs fla.chunk_gated_delta_rule
   at S in {128, 1024, 8192, 32768}, H=16, Dk=Dv=128.
3. `experiments/bench_dn_chunked_3090_v2.py` — perf harness
   (existing `bench_dn_chunked_3090.py` extended to compare v1 / v2 /
   fla side-by-side).

### Kernel 1: `dn3090v2_intra_kernel` (per-chunk WY rep)

Computes `w[chunk, head, t, k]`, `u[chunk, head, t, v]` from
`k`, `v`, `beta`, `g`. PARALLEL across (head, chunk). Grid:
`(n_chunks, H)`. Per block: BT=64 chunk, processes 1 head's
[BT, Dk] K and [BT, Dv] V slices.

Internal layout: 4 sub-chunks of BC=16 each. Computes 10
lower-triangular [BC, BC] tiles of `K @ K.T`, applies beta and
gate scaling, runs sequential forward substitution PER tile (just
16 iterations, fits in registers), then block-merges the 4 diagonal
tiles into the full [BT, BT] (I+A)^-1.

Output: `A[n_chunks, H, BT, BT]` fp32 (kept for bwd) and `w`, `u`
bf16 for the recurrence.

### Kernel 2: `dn3090v2_recurrence_kernel` (sequential per head)

Sequential walk of the recurrence. Grid: `(K/BK, H)` where BK=64
splits the K axis. Per block: 1 head, 1 K-slice. Holds state in
REGISTERS as `b_h1, b_h2 = float[BV=32, 64]` slabs (one per BK
split).

Per chunk in the (sequential) chunk loop:
  - Load `w[chunk, h, :, k_slice]` from kernel 1's output
  - `b_v = b_w @ b_h` (one BV column at a time)
  - Apply gate: `b_v = u - b_v` (residual), then scale by exp(g_last - g_t) per row
  - Scale `b_h *= exp(g_last)`
  - Load k.T from chunk's K range
  - `b_h += b_k @ b_v.T`

Output: `h[n_chunks, H, K, V]` fp32 (per-chunk states for kernel 3
and for bwd) and final state.

### Kernel 3: `dn3090v2_o_kernel` (per-chunk output projection)

Computes `y[chunk, t, h, v] = q[t] @ k.T * mask @ v_new[chunk] +
(q[t] * exp_g_cs[t]) @ h[chunk]`. PARALLEL across (chunk, head).

Grid: `(n_chunks, H, V/BV)` where BV=32. Per block: 1 chunk, 1 head,
1 V-slice.

Output: `y[S, H, V]` bf16 — the final per-token output.

### Memory budget (S=32K, H=16, Dk=Dv=128, BT=64)

  - `A`: [n_chunks=512, H=16, BT=64, BT=64] fp32 = 128 MB
  - `w`: [S=32K, H=16, K=128] bf16 = 128 MB
  - `u`: [S=32K, H=16, V=128] bf16 = 128 MB
  - `h`: [n_chunks=512, H=16, K=128, V=128] fp32 = 540 MB

Total: 924 MB temporary. Fits in 24 GB but is significant. For
inference we can free `A` after kernel 2 (only needed for bwd).
For the prefill graph, allocate these as scratch buffers and
reuse across layers.

### Validation gates

  1. `dn3090v2_intra_kernel` output `A` matches fla's `chunk_fwd_o`
     intermediate (compare via `chunk_gated_delta_rule_fwd_intra`'s
     return value `A`) — bf16 tolerance.
  2. `dn3090v2_recurrence_kernel` output `h[chunk]` matches fla's
     `chunk_gated_delta_rule_fwd_h` per-chunk state — fp32 1e-5
     tolerance.
  3. End-to-end y[S, H, V] matches fla's chunk_gated_delta_rule
     output — bf16 tolerance.
  4. Bench ≥ 0.8× of fla at S=32K (within 25%). Stretch goal:
     ≥ 1.0× via better autotuning.

### Effort

  - Kernel 1 (intra): 1.5 days
  - Kernel 2 (recurrence with register state): 2 days  ← hardest
  - Kernel 3 (output projection): 1 day
  - Validation suite: 0.5 day
  - Bench + tuning: 1 day
  - Wiring into prefill.cu: 0.5 day
  - Total: 6.5 engineer-days

This is genuine kernel-engineering work, not a chat-bot session
deliverable. The design doc above (and the fla source under
`/home/freiza/lucebox-hub/.venv-3090/lib/python3.10/site-packages/fla/ops/`)
is the spec.
