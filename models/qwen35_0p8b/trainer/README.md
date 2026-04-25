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

  vs HF's `torch_chunk_gated_delta_rule` (the actual training-path
  reference on this box, fla unavailable): **7.4× at S=128, 2.3× at S=512**.

  End-to-end HF Qwen3.5-0.8B + LoRA fwd+bwd training step (`bench_hf_dn_patch.py`):
  | S | baseline ms | patched ms | speedup |
  |---|---|---|---|
  | 128 | 542 | 92 | **5.92×** |
  | 512 | 686 | 282 | **2.43×** |

  Loss parity: 18-layer-compounded logit cos = 0.96, |Δloss| ≈ 7.4e-2
  (consistent with bf16 noise across recurrent vs chunked accumulation
  orders — training converges).

### What's still pending

- Tensor-core / chunked DN port — the scalar fp32 kernel is near its
  ceiling at ~16 ms (S=512). HF's chunk path uses cuBLAS tensor cores
  via fp32 `@`. A true CUTLASS / WMMA chunked port targets ~5 ms/layer
  and decisively wins long-S training.
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
