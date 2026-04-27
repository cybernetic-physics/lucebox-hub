# Training Megakernel (LoRA forward, all 13 trainable linears)

Fork of `../prefill_megakernel` extended with LoRA residuals applied
inside the single cooperative dispatch, plus a fused multi-param AdamW
optimizer kernel.

## Shipped this directory

### Forward pass

LoRA forward is wired into **all 13 trainable linears** of Qwen 3.5-0.8B:

| Layer type | LoRA on |
|---|---|
| Full Attention (6 layers) | q, k, v, o, gate, up, down |
| DeltaNet (18 layers) | qkv, z, out, gate, up, down |

Per projection, the kernel calls two phases: `phase_lora_h` (x @ A) then
`phase_lora_b_add` (adds scaling · (lora_h @ B) to the base GEMM output).
Each LoRA pair is nullable in the kernel signature — passing zero-sized
tensors disables that specific linear's LoRA.

Kernel arg packs all 26 LoRA pointer ptrs into a `LoraSet` struct (208
bytes). Memory layout per pointer:

- `A_all[layer_idx, K_in, LORA_R]`
- `B_all[layer_idx, LORA_R, K_out]`

Three-way sanity (`test_lora_forward.py`, S=32, random bf16 weights):

```
baseline prefill_megakernel          next_token = 159560
train_mega A=B=0  (all 13 linears)   next_token = 159560   ✓ matches baseline
train_mega A,B≠0  (all 13 linears)   next_token = 163951   ✓ diverges
```

### Fused AdamW

`torch.ops.train_megakernel_C.fused_adamw_step(params, m, v, grad, step, lr, beta1, beta2, eps, wd)`
runs the standard bias-corrected AdamW in one kernel dispatch over a flat
contiguous bf16 parameter buffer with fp32 m/v state.

Correctness (`test_adamw.py` vs `torch.optim.AdamW` on fp32 master weights
downcast per step):

```
step 1: max|Δ| = 1.95e-03  mean|Δ| = 3.51e-05
step 2: max|Δ| = 2.93e-03  mean|Δ| = 9.83e-05
step 3: max|Δ| = 3.91e-03  mean|Δ| = 1.46e-04
CORRECTNESS OK
```

Deltas are bf16 rounding at step-3 scale (accumulated 1e-3 · 3 = 3e-3).

### Backward kernels (Phase 2)

**FA backward** — `fa_bwd_flash.py`. Wraps cuDNN FlashAttention-2 via
`torch.ops.aten._scaled_dot_product_flash_attention(_backward)`. Bit-exact
(cos=1.0, max|Δ|=0), 1.65× faster than autograd sdpa at S=512. GQA-aware
(sums dK/dV over the head-replication axis when num_kv_heads != num_q_heads).

**DN BPTT backward** — `dn_bwd.cu` (CUDA) + `dn_autograd.py` (autograd
wrapper) + `dn_hf_patch.py` (drops into HF Qwen3-Next/Qwen3.5 layers).

  - `dn_fwd_with_delta_save_kernel`: forward recurrence, saves per-step
    delta + full state history `[H, S+1, Dk, Dv]`.
  - `dn_bwd_kernel`: reverse walk reading state from history. Computes
    dq, dk, dv, dbeta, ddecay, dstate_init.
  - Grid is `(H,)` — one block per head, 256 threads each cooperating on
    the [Dk=128, Dv=128] state matrix in shared memory.

  Bit-exact (cos=1.0 every gradient) at S=32..1024 (`test_dn_bwd_shapes.py`).
  Finite (no NaN/Inf) at S=2048, 4096, 8192, 16384, 32768.

  Speed @ S=512, one DN layer fwd+bwd: **15.9 ms**
   (was 261.6 ms with per-head host loop — 16.4× from grid=(H,) parallelism).

  Sweep on B200 (one DN layer fwd+bwd, scalar fp32):
  ```
  S      fwd ms    bwd ms    fwd+bwd ms   state_history
  32     0.20      0.61      0.81         0.03 GB
  128    0.83      3.11      3.94         0.13 GB
  512    3.34      12.66     16.00        0.50 GB
  1024   6.67      25.44     32.11        1.00 GB
  2048   13.34     51.01     64.35        2.00 GB
  4096   26.66     102.22    128.88       4.00 GB
  8192   53.30     204.60    257.90       8.00 GB
  16384  106.59    409.33    515.92       16.00 GB
  32768  213.16    818.74    1031.90      32.00 GB
  ```
  Per-step cost is constant ~31.5 µs — perfectly linear scaling for the
  sequential recurrence. State_history allocates [H, S+1, Dk, Dv] fp32 =
  ~1 MB × S; at S=32K that's 32 GB per layer (fits B200's 192 GB but
  bounds full 18-layer training to S≈4K without chunking).

  vs HF's `torch_chunk_gated_delta_rule` (the fp32 Python fallback used
  when fla isn't available): 7.4× at S=128, 2.3× at S=512.

  **Long-context shape sweep vs production HF + fla** (`bench_vs_fla.py`,
  hybrid routing on, B200, 2026-04-27):

  | mode  | S      | HF + fla  | hybrid    | ratio          | tok/s ours |
  |-------|--------|-----------|-----------|----------------|------------|
  | train | 128    | 110.7 ms  | 98.5 ms   | **1.12× ours** | 1,299      |
  | infer | 128    |  43.2 ms  | 33.4 ms   | **1.29× ours** | 3,830      |
  | train | 256    |  97.7 ms  | 97.4 ms   | 1.00× tied     | 2,629      |
  | infer | 256    |  43.3 ms  | 33.2 ms   | **1.30× ours** | 7,704      |
  | train | 512    | 101.8 ms  | 103.3 ms  | 0.99× tied     | 4,958      |
  | infer | 512    |  43.3 ms  | 34.0 ms   | **1.28× ours** | 15,077     |
  | train | 1024   |  98.2 ms  | 99.7 ms   | 0.98× tied     | 10,270     |
  | infer | 1024   |  43.3 ms  | 43.8 ms   | 0.99× tied     | 23,392     |
  | train | 2048   |  99.7 ms  | 101.9 ms  | 0.98× tied     | 20,091     |
  | infer | 2048   |  43.9 ms  | 44.0 ms   | 1.00× tied     | 46,489     |
  | train | 4096   | 101.0 ms  | 106.7 ms  | 0.95× fla      | 38,390     |
  | infer | 4096   |  49.8 ms  | 44.4 ms   | **1.12× ours** | 92,232     |
  | train | 8192   | 159.4 ms  | 159.5 ms  | 1.00× tied     | 51,357     |
  | infer | 8192   |  59.2 ms  | 59.6 ms   | 0.99× tied     | 137,398    |
  | infer | 16384  | 121.3 ms  | 122.2 ms  | 0.99× tied     | 134,130    |
  | infer | 32768  | 268.9 ms  | 269.5 ms  | 1.00× tied     | 121,598    |

  (Training mode skipped at S>8192 — HF's autograd graph exhausts HBM.)

  Hybrid routing (`dn_hf_patch.cuda_chunk_gated_delta_rule`, commit
  239db57) routes per (mode, shape):

    * training (autograd needed)         → `fla.chunk_gated_delta_rule`
    * inference S ≤ 512                  → our chunked CUDA kernel
    * inference S > 512 + fla available  → `fla.chunk_gated_delta_rule`
    * training, no fla                   → our recurrent CUDA + autograd

  Read: we win **1.28-1.30× on inference at S ≤ 512** (where our
  chunked DN kernel beats fla's per-call constants); essentially tied
  with fla everywhere else because the hybrid router picks fla at
  long S. Peak inference throughput is **137K tok/s at S=8K**. The
  per-step training ratio is ~1.0× across all S — meaning HF+fla's
  autograd is essentially as fast as us per training step today; the
  "training step" win in `bench_trainer_vs_sglang_torch.py` (10.57×
  combined RL step) comes from sampling and forward-batching, not
  from beating fla's per-layer kernel.

  Loss parity: 18-layer-compounded logit cos = 0.96 vs HF, |Δloss| ≈ 7.4e-2
  (bf16 noise across recurrent vs chunked accumulation orders).

### Tensor-core chunked forward (shipped)

`dn_chunked.cu` is a fused tensor-core forward kernel implementing the
chunked-delta-rule algorithm (matches HF's `torch_chunk_gated_delta_rule`
output within bf16 noise, cos > 0.99996 across S=64..1024). All matmuls
go through `nvcuda::wmma` m16n16k16 with bf16 inputs + fp32 accumulator,
one fused kernel per head — ~225 KB shared mem, fits B200's 228 KB
per-block limit.

Speed @ S=512 (one DN layer, forward only):
  | path | ms | x vs recurrent |
  |---|---|---|
  | scalar recurrent | 3.36 | 1.0 |
  | chunked tensor-core | **0.999** | **3.37×** |

For 18-layer Qwen3.5-0.8B forward at S=512: ~18 ms via chunked vs ~60 ms
recurrent — about 30× HF's torch_chunk fallback (~684 ms for 18 layers).

### Chunked backward — foundation in place, kernel pending

Two pieces shipped toward the chunked tensor-core backward:

  1. `dn_chunked_fwd_kernel` now optionally saves chunk-boundary state
     into a `state_chunks[H, n_chunks+1, Dk, Dv]` fp32 tensor. Memory:
     1 MB per chunk per layer (vs 1 MB per *step* for the recurrent
     kernel) — 64× less.
  2. `dn_chunked_bwd_proto.py` — analytical Python reference for the
     chunked backward, validated against torch.autograd at cos > 0.99998
     for every gradient at S = 64, 128, 512.

Key trick used in the analytical bwd: instead of differentiating through
the sequential row-update inner loop, use the matrix-inverse formula

    T = (I - tril(attn0))^{-1}   =>   dA = -T.T @ dT @ T.T

then mask the result to the strict lower triangle. Avoids saving
per-iteration intermediates from the inner loop.

The Python reference at S=512 takes ~40 ms/layer (3-4× slower than the
scalar CUDA bwd at 12.7 ms/layer), so it's not viable as a runtime
replacement; the CUDA port is what unlocks the win. Target after CUDA
port: ~2 ms/layer at S=512.

### What's still pending

- CUDA chunked backward kernel — port the validated Python reference.
  Buffer scheduling under B200's 228 KB shared-mem budget is the main
  engineering challenge (the kernel needs ~16 working buffers of
  [C, C], [C, Dk], [C, Dv] alongside the persistent state).
- Flash-attn long-context (S≥8k): cuDNN FA-2 is fast at small S; for
  long context our own kernel could drop further.

## Files

- `kernel.cu` — prefill megakernel + all 13 LoRA applies + fused AdamW
- `dn_bwd.cu` — DeltaNet BPTT CUDA kernel (forward+save and backward)
- `dn_autograd.py` — torch.autograd.Function wrapping the CUDA kernel
- `dn_hf_patch.py` — drop-in replacement for HF's `chunk_gated_delta_rule`
- `fa_bwd_flash.py` — cuDNN FA-2 backward via torch low-level ops
- `torch_bindings.cpp` — torch ops: `train_mega_forward`, `dn_fwd_save`,
  `dn_bwd`, `fused_adamw_step`
- `setup.py` — CUDAExtension
- `test_lora_forward.py` — three-way diff exercising all 13 linears
- `test_adamw.py` — fused AdamW vs torch reference
- `test_dn_bwd_cuda.py` — DN bwd CUDA correctness vs torch autograd ref
- `test_dn_autograd.py` — autograd-wrapped DN matches torch ref under autograd
- `test_dn_hf_patch.py` — `cuda_chunk_gated_delta_rule` matches HF reference
- `bench_phase2_live.py` — per-layer DN/FA bwd timings with CUDA path
- `bench_hf_dn_patch.py` — end-to-end HF training step with vs without patch

## Remaining roadmap

1. ✅ Save per-layer activations during forward so backward can consume them.
2. ✅ FA backward (cuDNN FA-2; bit-exact, 1.65×).
3. ✅ DN BPTT backward (CUDA kernel; bit-exact, 17–20× vs torch ref;
      end-to-end HF training step 2.4–5.9× faster).
4. ⬜ Tensor-core (CUTLASS/WMMA) chunked DN backward — target ~5 ms/layer
      at S=512, decisive long-S win over HF's chunk-fp32 path.
5. ⬜ Flash-attn long-context (S≥8k) — beat HF cuDNN by 20× target.
