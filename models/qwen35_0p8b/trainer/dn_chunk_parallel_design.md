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
