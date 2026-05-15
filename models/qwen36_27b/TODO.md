# TODO — Qwen3.6-27B on GB10 sm_121a

## Mission

Run **Qwen/Qwen3.6-27B** on GB10 (Grace Blackwell sm_121a, 128 GB unified
memory) with:

1. **Correctness == HF Qwen3_5TextModel reference**. Top-1 match on
   natural text and wikitext-2 windows up to S=32k. Cos sim ≥ 0.999
   vs HF on in-distribution prompts.
2. **Speed**: usable at S=32k (target: ≥ 20 tok/s decode, ≥ 4k tok/s
   prefill on GB10 with NVFP4 weights and MTP speculative).
3. **Full feature surface** of the HF reference: chat template,
   thinking mode, native tool calls, grammar-constrained sampling.
   Optional: vision tower.
4. **Memory**: inference fits on GB10 with room for ≥ 32k context;
   NVFP4-quantized version fits on 24 GB consumer cards.

This file is the actionable backlog. Each item has:
- **Status**: TODO / WIP / DONE / BLOCKED
- **Priority**: P0 (blocks downstream) / P1 / P2
- **Effort**: S (≤ 1 day) / M (2-5 days) / L (1-2 weeks) / XL (> 2 weeks)
- **Depends on**: which other items must complete first
- **Acceptance**: what "done" means concretely

Items are grouped by workstream and ordered within each group by
dependency.

---

## Critical path — correctness gate (unblocks everything else)

The templated megakernel is built and the wiring is verified statically.
**Nothing past this point is real until a megakernel forward produces
logits that match HF on at least one prompt.** First-real-forward almost
certainly surfaces bugs; the next ~5 items are the tight debug loop.

### C1. First megakernel forward on real Qwen3.6-27B weights
- **Status**: TODO  | **Prio**: P0  | **Effort**: S  | **Deps**: —
- Run `Qwen36MegakernelDecoder.prefill(prompt_ids)` for a 1-token prompt
  on the real HF-loaded weights. Just confirm it doesn't crash and emits
  any output.
- **Acceptance**: returns a token id without CUDA error.
- **Likely failure modes**:
  - `cudaLaunchCooperativeKernel` rejects the launch because static
    shmem per block (~68 KB at 27B) > per-SM allowance. Fix: switch the
    inner `__shared__ char shmem_raw[...]` to dynamic shmem + opt in
    via `cudaFuncSetAttribute(MaxDynamicSharedMemorySize, ...)`.
  - `LayerWeights<Cfg>` field ordering differs between Python pack and
    C struct (already audited but easy to break).
  - Kernel exits cleanly but writes NaNs everywhere — DN state init or
    conv ring buffer indexing wrong.

### C2. Layer-by-layer hidden-state capture vs HF
- **Status**: TODO  | **Prio**: P0  | **Effort**: S  | **Deps**: C1
- Add hooks on HF Qwen3_5TextModel that capture the hidden state out of
  every transformer layer for a fixed prompt; save as a .pt golden.
  Mirror the same capture in our kernel by writing each layer's output
  to a debug buffer.
- **Acceptance**: a 64-element list of `[HIDDEN]` tensors per side,
  saved as `reference/per_layer_hf.pt` and `reference/per_layer_ours.pt`.

### C3. First-divergence detector
- **Status**: TODO  | **Prio**: P0  | **Effort**: S  | **Deps**: C2
- Script that loads both .pt files, iterates layers 0..63, prints
  cos-sim + max-abs-diff at each layer. Identify the first layer where
  drift exceeds noise floor (cos < 0.999).
- **Acceptance**: report names a specific layer index and whether it's
  FA or DN.

### C4. Fix first-divergence cause
- **Status**: TODO  | **Prio**: P0  | **Effort**: M  | **Deps**: C3
- Top suspects, ordered by probability based on the code:
  1. **DN V/QK GQA indexing**. The new code maps `v_head -> qk_head =
     v_head / V_PER_QK`. If HF instead expects `qk_head = v_head %
     QK_HEADS` (interleaved vs grouped), the kernel reads the wrong Q/K
     projection per V head. Sub-fix: instrument the QKV slicing in
     `dn_layer.cuh` and compare against `linear_attn.in_proj_qkv` slices
     in HF's `Qwen3_5GatedDeltaNet.forward`.
  2. **Beta/alpha activation order**. Our code applies `sigmoid` to
     beta and `exp(-exp(a_log) * softplus(alpha + dt_bias))` to alpha
     once. If HF order differs (e.g. activations applied per-step in
     the chunked scan), state evolves differently from step 1.
  3. **Per-head L2 normalization of Q with `1/sqrt(128)` factor**. The
     scaling factor is Qwen-specific; if HF removed it in 3.6 the
     attention scores rescale.
  4. **MRoPE section split**. We index `i < sections.t / i < t+h /
     else` for the rotary pair index; verify against HF's
     `mrope_section` interpretation (the meta repos vary).
  5. **Conv1d ring buffer**. The first step at position=0 reads zeros
     from the ring; correct, but the shift-by-1 logic at later positions
     could index off-by-one.
- **Acceptance**: cos ≥ 0.999 at the previously-divergent layer.

### C5. Loop C2-C4 until all 64 layers agree [DONE]
**ROOT CAUSE FOUND**: FA layer was missing post-attn-norm + MLP +
residual entirely. The weight struct declared post_attn_layernorm_weight,
gate_proj_weight, up_proj_weight, down_proj_weight but they weren't
USED. Skipping the MLP every 4th layer (16 times total) compounded
to catastrophic drift.

Fix in fa_layer.cuh: append the standard post-attn-norm + SwiGLU +
down + residual sequence after the FA O-proj. Reuses g_attn_out as
the mlp_inter f32 scratch.

After fix:
  All 64 layers: cos > 0.999 (DN: ~0.99998; FA: 0.99993-0.99999)
  Final logits:  cos = 0.999640
  Top-1 match:   YES (HF=16, ours=16)

This closes C5 for single-token forward at position 0.

### C5z. (archived investigation log)
- **Status**: WIP  | **Prio**: P0  | **Effort**: M  | **Deps**: C4
- Each fix may surface the next divergence point. Bound the loop by
  the layer count.

**Current state from C3 diff** (`tests/test_c3_layer_diff.py` on
real Qwen3.6-27B weights, single space token at position 0):

```
DN layers 0..2  : cos > 0.9999   (essentially perfect)
FA layer 3      : cos = 0.997    (first FA, ~0.003 drift)
FA layers 7-43  : cos ~ 0.997-0.999  (compounds slowly)
FA layer 47+    : cos ~ 0.98 -> 0.28  (catastrophic by L63)
final logits    : cos = 0.87, top-1 mismatch (HF=16 ours=220)
```

**Diagnosis so far**:
- The DN V/QK split (the new code we were most worried about) is
  CORRECT. DN layers match HF essentially bit-exactly.
- Bug is localized to the FA layer; produces ~0.003 cosine drift
  per FA pass, compounds over 16 FA layers to 0 cos.

**Tested + ruled out**:
- ✗ Output gate = silu/swish (cos=0.81 at L3, much worse than sigmoid)
- ✗ Q-proj output split layout `[Q_all, gate_all]` (cos=0.948, worse
  than interleaved's 0.997; magnitudes blew up 3×)
- ✓ Per-head interleaved layout `[Q_h, gate_h]` is correct
- ✓ Output gate = sigmoid is correct (despite config saying "swish")

**Remaining suspects, in priority**:
1. **Per-head q_norm/k_norm weight shape**. I assume `[head_dim=256]`
   shared across heads. If actually `[num_heads*head_dim=6144]` per-
   head distinct, my code is silently using head-0's weights for all
   heads. Need to inspect actual safetensors shape. (My
   `_check_shape` asserts on `[FA_HEAD_DIM,]` so this would have
   failed at pack time — unless HF reshapes on load.)
2. **RMSNorm formula in q_norm/k_norm**: my `head_norm_rope` uses
   `(1+w)` like 0.8B. If Qwen3.6 q_norm uses standard `w` (no +1),
   my Q,K are over-scaled by ~2×. But softmax over 1 position is
   trivial; magnitude difference must come from V or gate which
   don't use this norm.
3. **Attention scale**: I use `1/sqrt(head_dim=256)` = 1/16. Some
   impls use `1/sqrt(rotary_dim=64)` = 1/8. The attn output at
   position 0 doesn't depend on the score (softmax-of-one is 1),
   so this isn't it for the L3 drift but matters at S>1.
4. **gate's interaction with V**: maybe HF applies gate as
   `output = silu(gate * attn_out)` (gate inside silu, applied to
   the product) rather than `output = silu(gate) * attn_out`.

**Next debug step**: add per-FA-step intermediate capture to the
kernel. Save Q-proj row, post-Q-norm Q, gate slice, V output, attn-
out, post-gate attn-out, and O-proj output to a debug global. Hook
the same in HF via `forward_pre_hook` on `model.layers[3].self_attn.*`.
Diff each step element-wise. ~1 day of plumbing.

**Next debug step**: add intermediate-value capture buffers to the FA
kernel (Q-proj output, gate slice, V output, attn_out, post-gate
attn_out) and compare element-by-element with HF using a hook on
the corresponding modules. ~1 day of scratch-buffer plumbing.

- **Acceptance**: top-1 + cos ≥ 0.999 against HF on a single-prompt
  forward at S=1.

### C5b. Intermediate-value capture in FA layer
- **Status**: TODO  | **Prio**: P0  | **Effort**: M  | **Deps**: C5
- Add scratch buffers + capture hooks to fa_layer.cuh that, when
  enabled, write the following to a debug global:
    1. Q-proj output (first row) [post-matvec]
    2. Q after RMSNorm + RoPE [per-head]
    3. K after norm + RoPE
    4. V after projection
    5. attn_out after softmax * V
    6. attn_out * sigmoid(gate)
    7. O-proj output
- Add corresponding `forward_pre_hook` on HF `model.layers[3].self_attn`
  module to capture the same 7 values.
- Diff element-wise; the FIRST one that differs is the bug.

### C5c. Q-norm/K-norm weight shape verification
- **Status**: DONE  | **Prio**: P0  | **Effort**: S  | **Deps**: —
- Inspected via safetensors metadata. Results:
    q_norm.weight: shape (256,), mean=0.22 std=0.07
    k_norm.weight: shape (256,), mean=0.21
    input_layernorm.weight: shape (5120,), mean=0.24
- All weights centered around 0.2 (not 1.0), confirming the model
  was trained with `(1 + weight)` scaling. My code's `(1.0f + w)`
  form is correct for all three norms.

### C6. Long-context correctness sweep
- **Status**: WIP  | **Prio**: P0  | **Effort**: S  | **Deps**: C5
- Test scaffold: `test/test_c6_multitoken.py`. Three prompts of
  varying lengths (S=1, ~6, ~17), runs both HF and our prefill,
  compares last-position logits + per-layer cos snapshot.
- Run with `HF_HOME=/home/sparkz/rl/.hf_cache python3 test/test_c6_multitoken.py`
- After test passes for short prompts, extend to wikitext windows at
  S ∈ {32, 128, 512, 1024, 4096, 16384, 32768}.
- **Acceptance**: top-1 match on natural text + cos ≥ 0.99 throughout.

---

## Speed optimization (after C5 lands)

These workstreams parallelize. Order by speed-per-effort ratio.

### S1. NVFP4 weight quantization wired into the megakernel
- **Status**: TODO  | **Prio**: P0 (memory-gated on consumer cards)  | **Effort**: M  | **Deps**: C5
- Python plumbing exists (`nvfp4_27b.py`). Missing: an NVFP4-aware
  matvec in `matvec.cuh` that takes the packed `(data, scales)` tensor
  pair. Pattern is in `models/qwen35_0p8b/kernel_gb10_nvfp4.cu:matvec_nvfp4`.
- Add `Cfg::USE_NVFP4_WEIGHTS` trait; specialize the FA/DN/MLP layers
  to call `matvec_nvfp4<Cfg>` when set.
- **Acceptance**: 27B weights load at ~14 GB, end-to-end inference
  matches BF16 path within 1% PPL drift on wikitext.

### S2. Parallel-S prefill kernel
- **Status**: TODO  | **Prio**: P1  | **Effort**: L  | **Deps**: C5
- Replace `prefill_qwen3x_naive` (host-loop over S decode calls) with a
  `prefill_megakernel<Cfg>(S, ...)` that processes the S-dim in
  parallel. Structure:
  - matvec over [S, *] tiled by cuBLAS or hand-rolled CUTLASS tile.
  - Per-position head-norm + RoPE.
  - Batched FA over the S×S causal triangle.
  - DN sequential recurrence (unparallelizable over S; chunked).
  - out-proj + residual.
  - post-attn norm + MLP.
- The DN chunked scan is the algorithmic bulk. Reference:
  `models/qwen35_0p8b/prefill_megakernel.cu` for the existing 0.8B
  pattern, plus the `fla.modules.gated_delta_rule.chunk_gated_delta_rule`
  for the math.
- **Acceptance**: prefill at S=2048 matches naive output within
  fp32-accumulation noise, **at least 8× faster**.

### S3. NVFP4 KV cache wired into the megakernel FA path
- **Status**: TODO  | **Prio**: P1  | **Effort**: S  | **Deps**: S1
- `nvfp4_kv.cuh` helpers already work standalone; need to wire them
  into `fa_layer.cuh`'s K/V read/write sites. Add `Cfg::USE_NVFP4_KV`
  trait; conditional inclusion.
- **Acceptance**: KV memory drops from 1 GB → 280 MB at S=32k; PPL drift
  ≤ 5% on wikitext (matches the 0.8B NVFP4-KV sweep we already verified).

### S4. Optimize the FA scan for 32k context
- **Status**: TODO  | **Prio**: P1  | **Effort**: M  | **Deps**: S3
- The current FA scan uses split-K with num_splits = num_blocks / Q_H.
  At S=32k with 24 Q heads on 64 SMs, num_splits = 2. That's not enough
  parallelism. Either:
  - Use more blocks per SM (oversubscribe but match block grain to
    cooperative-grid limit). Requires moving away from cooperative
    grid for the FA scan and using a separate kernel for the FA stage.
  - Tile the K dimension finer (head_dim=256 -> blocks of 64).
- **Acceptance**: FA scan at S=32k is < 50% of decode wall time
  (currently expected to be > 80%).

### S5. Multi-Token Prediction (MTP) wire-up
- **Status**: TODO  | **Prio**: P1  | **Effort**: M  | **Deps**: S1
- Qwen3.6-27B ships a native NEXTN head: a single transformer layer at
  `mtp.layers.0.*` in the safetensors. Layout matches a standard FA
  layer.
- Wire `mtp_speculative.MTPDecoder._mtp_predict` to call this head
  instead of the LM-head approximation it uses now.
- **Acceptance**: chain MTP achieves AL ≥ 3 on natural text.

### S6. Tree-verify parallel forward kernel
- **Status**: TODO  | **Prio**: P1  | **Effort**: L  | **Deps**: S5
- `tree_verify.cuh` primitives are in place. Missing: a
  `prefill_megakernel_tree<Cfg>` kernel that does one forward over a
  TreeNode descriptor with tree-aware attention masking. Reference:
  `models/qwen35_27b/src/dflash_decode.cpp` for the host-side
  control flow (ggml-based; port the algorithm, rewrite the inner loop
  against our kernels).
- **Acceptance**: tree-verify achieves AL ≥ 6 at budget=22, matching
  the qwen35_27b DFlash + DDTree results.

### S7. cuBLASLt FP4 LM head
- **Status**: TODO  | **Prio**: P2  | **Effort**: S  | **Deps**: S1
- Re-use the 0.8B cuBLASLt FP4-LM-head plan from
  `models/qwen35_0p8b/kernel_gb10_nvfp4.cu:lm_head_plan()` at the 27B
  vocab=248320 / hidden=5120 shapes.
- **Acceptance**: LM head GEMV at < 1 ms at FP4.

---

## Memory optimization for training (separate concern, post-inference)

Training at 27B/S=32k has a different memory profile than inference.
Activation saves dominate. Inference works without these; **only block on
training**.

### T1. Activation-save memory budget audit at 27B/S=32k
- **Status**: TODO  | **Prio**: P1 (gates training)  | **Effort**: S  | **Deps**: C5
- Compute exact bytes for `prefill_bf16_train_step` at 27B dims:
  - 4 slabs × 64 layers × 32768 × max(5120, 17408) × 2 bytes ≈ **292 GB**
  - Plus FA-bwd saves, etc.
- This **does not fit** on GB10 (121 GB). Mitigations needed before any
  27B training is possible:
  1. Gradient checkpointing (recompute instead of save). ~5× memory cut.
  2. NVFP4 activation saves. ~4× memory cut.
  3. Sequence-chunked training (split S into shards of 4k, train each).
- **Acceptance**: exact GB/save buffer reported; mitigation plan picked.

### T2. NVFP4 activation save format
- **Status**: TODO  | **Prio**: P1  | **Effort**: M  | **Deps**: T1
- The NVFP4 KV helpers can quantize the activation slabs too — same
  head_dim, same E4M3 scaling. Modify `prefill_bf16_train_step` to
  write NVFP4-packed activations; backward kernel reads dequant on
  the fly.
- **Acceptance**: training fwd at S=32k fits on GB10 (≤ 100 GB
  including all scratch).

### T3. Backward kernels at 27B dims
- **Status**: TODO  | **Prio**: P1  | **Effort**: L  | **Deps**: T1
- Port `models/qwen35_0p8b/prefill_bw.cu` to template on Cfg. Same
  structure as the inference port. The DN backward in particular needs
  the V/QK split handling for grads.
- **Acceptance**: kernel-driven backward matches HF autograd within
  bf16 noise on a tiny rank-16 LoRA train step.

---

## Application-level features (parallel to optimization)

These are needed for production but don't gate correctness.

### F1. Streaming /v1/chat/completions
- **Status**: TODO  | **Prio**: P1  | **Effort**: S  | **Deps**: C5
- Current `runtime_hf` and (future) megakernel runtime return the full
  generation. Wire streaming via SSE; emit each decoded token.
- **Acceptance**: `curl --no-buffer http://.../v1/chat/completions
  -d '{"stream": true, ...}'` emits incremental chunks.

### F2. Multi-turn KV cache reuse
- **Status**: TODO  | **Prio**: P1  | **Effort**: S  | **Deps**: C5
- Today every request resets state. For chat, keep the KV cache for
  the prefix that's already been processed.
- **Acceptance**: 2-turn chat run reuses position from turn 1.

### F3. Concurrent request batching
- **Status**: TODO  | **Prio**: P2  | **Effort**: L  | **Deps**: S2
- The kernel currently handles batch=1. For a multi-user serve, need
  either dynamic batching (request scheduler stitches multiple prompts
  into one kernel) or per-request CUDA streams.
- **Acceptance**: 4 concurrent /v1/chat/completions requests stay
  within 2× their individual latency.

### F4. Vision tower
- **Status**: TODO  | **Prio**: P2  | **Effort**: M  | **Deps**: C5
- Qwen3.6-27B's `vision_config` has depth=27, hidden_size=1152, an
  out_hidden_size=5120 projection back into the LM. Run vision as a
  separate stage via HF (no perf sensitivity — single image is < 100ms
  on GB10). Splice vision tokens into the prompt.
- **Acceptance**: a chat request with an image attachment runs end-to-
  end; output references image content.

### F5. Native Qwen3 tool-call parser polish
- **Status**: WIP  | **Prio**: P2  | **Effort**: S  | **Deps**: —
- Current parser handles `<tool_call>{...}</tool_call>` blocks. The
  Qwen3 "coder" variant uses a different format with `<function=name>`
  tags. Add a `tool_call_parser` enum.
- **Acceptance**: passes the published Qwen3 tool-call test suite.

### F6. Thinking-mode "preserve" toggle wired through OpenAI API
- **Status**: WIP  | **Prio**: P2  | **Effort**: S  | **Deps**: —
- `preserve_thinking` exists in GenerationConfig and the server but
  needs end-to-end test against the OpenAI client semantics — the
  `reasoning_content` field should show on responses when preserving.
- **Acceptance**: integration test passes.

---

## Production hardening

### P1. End-to-end correctness harness on a real eval
- **Status**: TODO  | **Prio**: P1  | **Effort**: M  | **Deps**: C5
- Run wikitext-2 perplexity on our runtime vs HF; verify within 1%.
- Run a small chunk of MMLU on both; verify top-1 within 2pp.
- **Acceptance**: report committed to `docs/results/qwen36_27b_correctness.md`.

### P2. Speed benchmark suite
- **Status**: TODO  | **Prio**: P1  | **Effort**: S  | **Deps**: C5, S2
- Adapt `models/qwen35_0p8b/bench_pp_tg.py` for 27B. Sweep S ∈
  {128, 512, 2048, 8192, 32768} for prefill and decode.
- **Acceptance**: numbers committed to `docs/results/qwen36_27b_speed.md`.

### P3. Memory regression tests
- **Status**: TODO  | **Prio**: P2  | **Effort**: S  | **Deps**: —
- A bench that runs decode at S=32k and asserts peak alloc ≤ a fixed
  budget. Catches future regressions in the activation/scratch sizing.
- **Acceptance**: CI script in `test/test_memory_budget.py`.

### P4. Numerical hardening checklist
- **Status**: TODO  | **Prio**: P2  | **Effort**: S  | **Deps**: C5
- Audit NaN/Inf handling in:
  - DN recurrence at long S (state can grow if alpha < 1 narrow)
  - FA softmax at very small logit magnitudes (current `fast_exp` may
    underflow)
  - RMSnorm at near-zero inputs (the EPS=1e-6 floor)
- **Acceptance**: bench that runs 32k random tokens through inference
  and asserts no NaNs.

### P5. OOM-safe load path
- **Status**: TODO  | **Prio**: P2  | **Effort**: S  | **Deps**: —
- `weight_packer.load_27b_weights` currently allocates 50 GB BF16 then
  another 2 GB of scratch. If GPU mem is tight (e.g. browser also using
  it), the load fails with cryptic OOM. Detect this and fall back to
  CPU staging + `.to(device, non_blocking=True)` per layer.
- **Acceptance**: load succeeds with `CUDA_VISIBLE_DEVICES_MEMORY=60GB`
  pressure.

### P6. Pin transformers version
- **Status**: TODO  | **Prio**: P2  | **Effort**: S  | **Deps**: —
- We're on transformers 5.6.0; Qwen3.6 config requires 4.57.1 nominally
  but our discovery showed 5.6.0 works after `trust_remote_code=True`.
  Pin the working combination in `setup.py` / requirements file.
- **Acceptance**: `pip install -r requirements.txt` reproduces the
  working environment.

---

## Documentation

### D1. Per-(model, arch) results doc
- **Status**: TODO  | **Prio**: P2  | **Effort**: S  | **Deps**: P1, P2
- Add `docs/results/qwen36_27b_gb10.md` matching the existing 0.8B
  results page format. Include the speed table, the correctness
  invariants, the memory budget table.

### D2. Migration writeup
- **Status**: TODO  | **Prio**: P2  | **Effort**: S  | **Deps**: C5
- `docs/roadmap/qwen36_27b_migration.md` capturing the lessons: the
  Cfg-template pattern that let 0.8B and 27B share source; the DN V/QK
  split being the only real architectural addition; the bugs we found
  (sizeof + state_dict prefix); the debug strategy (layer-by-layer
  capture).

### D3. README updates
- **Status**: TODO  | **Prio**: P2  | **Effort**: S  | **Deps**: C5
- Update root `README.md` to add a "03 · Qwen3.6-27B megakernel"
  section paralleling the existing 01 and 02 entries.

---

## What's done already (for context)

- BF16 + NVFP4 KV cache helpers at head_dim=256 (works on both 0.8B
  and 27B).
- `Cfg_0p8B` + `Cfg_27B` tag structs; templated rmsnorm, matvec, MLP,
  FA layer, DN layer (with V/QK split), RoPE+YaRN+MRoPE, decode
  kernel orchestrator, naive prefill loop.
- Compile-verified for sm_121a on both Cfgs.
- MLP smoke test cos=1.000 vs torch reference on both shapes.
- HF-backed `runtime_hf.Qwen36Runtime` with chat template, thinking,
  tool calls, XGrammar.
- OpenAI-compatible serve.
- 54 GB BF16 weights downloaded + verified.
- Weight packer loads + packs successfully against real safetensors.
- Two ABI bugs (LayerWeights sizeof, state_dict prefix) caught and
  fixed before reaching kernel runtime.
- `decode_qwen3x` and `prefill_qwen3x_naive` torch ops registered and
  invokable from Python.
- MTP chain decoder + tree-verify host driver.
- NVFP4 27B Python plumbing (footprint: 13.9 GB at 32k ctx — fits on
  24 GB consumer cards).

---

## Honest schedule

Inference parity (C1-C6): **3-5 days** focused debugging. Risk is C4
which can take longer if multiple sources of drift compound.

Optimization to chat-grade speed (S1, S5, S6): **2 weeks** after parity.
Tree-verify forward kernel is the longest single item.

Training enablement (T1-T3): **2-3 weeks** independent of inference
work.

Application features (F1-F6): **1-2 weeks**, mostly independent of
kernel work.

Full production-ready system with everything: **6-8 weeks** of focused
engineering from here.
