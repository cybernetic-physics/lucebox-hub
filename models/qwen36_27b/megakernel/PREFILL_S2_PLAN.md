# S2 — Parallel-S prefill kernel (port from 0.8B)

The 0.8B has a real parallel-S prefill in `models/qwen35_0p8b/prefill_megakernel.cu`
(~1100 lines). The 27B path is currently host-looped (`prefill_qwen3x_naive`,
~215 ms × S decode steps), which makes any S ≥ 1024 unusable in wall-clock.

This file outlines the port. Estimated effort: **3-5 days**.

## What the 0.8B version does (one cooperative-grid kernel)

1. Embed lookup → hidden[S, HIDDEN] BF16.
2. For each layer (24 layers, hybrid 3 DN + 1 FA):
   - Pre-attn RMSNorm.
   - QKV projection via WMMA matmul tiled across persistent blocks.
   - For FA:
     - Per-position head-norm + RoPE.
     - Batched FlashAttention over S × S triangle (causal mask).
     - O-proj + residual.
   - For DN:
     - 16-block-per-layer recurrent scan (serial over t, parallel over
       V heads). The 0.8B uses chunked scan with chunk size 64.
     - Out-proj + residual.
   - Post-attn RMSNorm.
   - MLP (gate + up + SiLU + down + residual) via WMMA.
3. Final RMSnorm.
4. (Optional) LM head matmul over the last position.

## What changes for 27B

| | 0.8B | 27B |
|---|---:|---:|
| NUM_LAYERS | 24 | 64 |
| HIDDEN | 1024 | 5120 |
| INTER | 3584 | 17408 |
| FA Q heads | 8 | 24 |
| FA KV heads | 2 | 4 |
| FA head_dim | 256 | 256 |
| DN V heads | 16 | 48 |
| DN QK heads | 16 | 16 |
| **DN V_PER_QK** | **1** | **3** ← only real algorithmic difference |
| Native context | 32k | 262k |
| RoPE theta | 1e6 | 1e7 |
| Use YaRN | optional | yes (for >32k) |
| Use MRoPE | no | yes |

## Port checklist

1. **Template the file on Cfg.**
   - Replace all the `constexpr int HIDDEN = 1024;` lines with `Cfg::HIDDEN`.
   - This makes the same source compile for both Cfg_0p8B and Cfg_27B.
   - The 0.8B path can switch over to the templated kernel once
     correctness vs the existing prefill_megakernel.cu is verified.

2. **DN V/QK split** (new for 27B):
   ```cpp
   int v_head = block_id;
   int qk_head = v_head / Cfg::DN_V_PER_QK;
   ```
   Each V head's recurrent state lives at `dn_state[v_head]`; the Q/K
   projections are shared across V_PER_QK siblings.

3. **MRoPE** (new for 27B):
   - Replace the standard RoPE in head_norm_rope with the MRoPE
     section-interleaved variant from `rope.cuh`.
   - For text-only, pass `pos_t == pos_h == pos_w` so all sections rotate.

4. **YaRN scaling** (new for 27B beyond 32k):
   - Insert YarnParams into the kernel signature; apply scale to
     `inv_freq` per the formula in `rope.cuh:compute_yarn_inv_freq`.

5. **WMMA tile sizes** for the 5120 → 17408 GEMM:
   - Block tile [BTM=32, BTN=128] is fine but check tile count vs the
     wider INTER. Likely just more blocks, no tile-size change.

6. **Shared memory budget**:
   - 17408 × 4 bytes (mlp_inter) = 68 KB per block. Already at the
     limit for static shmem on Blackwell. Use dynamic shmem with
     `cudaFuncSetAttribute(MaxDynamicSharedMemorySize, ...)`.

7. **DN chunked scan** with V_PER_QK=3:
   - The 0.8B chunked scan iterates over S in chunks of 64. For
     V_PER_QK=3, each chunk processes 64 V-head × 3-Q/K-sibling × KEY
     dot products before advancing state. The serialization is per
     V head, so V_PER_QK does NOT increase the serial depth — just
     the per-chunk compute.

## Acceptance

- `prefill_qwen3x_parallel` correctness: outputs match prefill_naive
  output within fp32 accumulation noise (cos > 0.999, top-1 match)
  on the same wikitext prompts as C7.
- Speed: at least 8× faster than prefill_naive at S=2048.
- Memory: peak alloc stays under prefill_naive's footprint
  (the chunked scan is the same per-V-head cost; the WMMA tiles add
  ~256 KB shmem total).

## See also

- 0.8B reference: `models/qwen35_0p8b/prefill_megakernel.cu`
- DN chunked scan algorithm: `fla.modules.gated_delta_rule.chunk_gated_delta_rule`
- Working host-loop: `prefill_megakernel.cu:prefill_naive_impl`
