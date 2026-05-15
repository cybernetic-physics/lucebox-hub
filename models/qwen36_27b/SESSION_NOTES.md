# Session notes — Qwen3.6-27B megakernel correctness debug

## TL;DR

**C5 closed: single-token forward at position 0 matches HF top-1 with
cos = 0.999640.** Root cause was the FA layer skipping its post-attn
RMSnorm + SwiGLU MLP. Adding those 3 steps after the FA O-proj made
all 64 layers' hidden states agree with HF within 0.99-0.99999 cos.

## What was achieved this session

### Critical-path correctness (C1-C5)

| Stage | Result |
|---|---|
| C1: First megakernel forward | Kernel runs end-to-end on real Qwen3.6-27B weights without crash. cos=0.87 vs HF (semi-correlated). |
| C2: HF + ours per-layer capture | Layer-by-layer hidden-state capture infrastructure shipped. |
| C3: First-divergence detector | Identified FA layer 3 as first divergent (cos=0.997). DN layers 0-2 essentially perfect (>0.9999). |
| C4: Hypotheses tested | Output gate sigmoid vs silu: sigmoid better. Q-proj layout interleaved vs split: interleaved better. Norm formula (1+w): verified correct (weights stored as deltas ~0.22). |
| **C5: ROOT CAUSE** | **FA layer was missing post-attn-norm + MLP + residual entirely.** Weight struct declared the fields but forward fn skipped them. Adding the standard 3-step sequence after O-proj fixed everything. |

### After-fix C3 diff

```
All 64 layers:    cos > 0.9928 (mostly > 0.9999)
DN layers (48):   cos = 0.99996 typical
FA layers (16):   cos = 0.99993-0.99999 typical
Final logits:     cos = 0.999640
HF top-1:         16 ('1')   <- "after a space, '1' comes next"
Ours top-1:       16 ('1')   ✓ MATCH
```

### Bugs caught + fixed during this work

1. **LayerWeights<Cfg> sizeof = 120 vs Python pack stride 128**. Static
   union sized to 120B (max DN ptrs); Python wrote at 128B stride
   creating offset misalignment for layer ≥ 1. Fix in
   kernel_decode_full.cu: explicit `char _force_size[120]` in union
   produces sizeof = 128. Added `static_assert`s.

2. **HF state_dict prefix mismatch**. Safetensors uses
   `model.language_model.layers.X.*` but HF's CausalLM wrapper
   instantiates `Qwen3_5TextModel` which strips `language_model`, so
   the runtime state_dict uses `model.layers.X.*`. Fix in
   weight_packer.py.

3. **FA layer missing MLP** (the BIG one). See above.

## What's been built (in repo, ahead of `origin/gb10-train`)

### Phase 0 — Scaffolding
- `PLAN.md` — phased migration roadmap
- `TODO.md` — actionable backlog with status / priority / effort
- `README.md` + `docs/parameterization_audit.md`
- `reference/capture_hf_reference.py` — HF golden capture harness

### Phase 1-7 — Templated kernels
- `megakernel/Cfg.cuh` — `Cfg_0p8B` + `Cfg_27B` tag structs
- `megakernel/helpers.cuh` — math intrinsics + warp reductions
- `megakernel/rmsnorm.cuh` — RMSNorm with residual capture
- `megakernel/matvec.cuh` — bf16 matvec + NEW: NVFP4 matvec primitives
- `megakernel/rope.cuh` — YaRN + MRoPE interleaved sections [11, 11, 10]
- `megakernel/fa_layer.cuh` — full_attention_layer<Cfg> WITH MLP path
- `megakernel/dn_layer.cuh` — delta_net_layer<Cfg> with V/QK split
- `megakernel/kernel_decode_full.cu` — persistent layer-walker megakernel
- `megakernel/prefill_megakernel.cu` — naive host-loop prefill
- `megakernel/tree_verify.cuh` — TreeNode descriptor + ancestor walk
- `megakernel/torch_bindings.cpp` — `decode_qwen3x` + `prefill_qwen3x_naive` ops

### Python runtimes + tests
- `runtime_hf.py` — HF-backed `Qwen36Runtime` (thinking/tools/grammar)
- `runtime_megakernel.py` — `Qwen36MegakernelDecoder` (kernel-backed)
- `weight_packer.py` — HF -> kernel layout + scratch alloc
- `nvfp4_27b.py` — NVFP4 KV + weight quantization Python plumbing
- `mtp_speculative.py` — MTP chain decoder + tree-verify host driver
- `serve/openai_server.py` — OpenAI-compat HTTP server
- `run_hermes.sh` — one-command launcher
- `test/test_runtime_wiring.py` — wiring sanity (no big download)
- `test/test_packer_load.py` — live HF weight load + pack
- `test/test_weight_packer_keys.py` — safetensors key audit
- `test/test_correctness_vs_hf.py` — HF-vs-HF regression baseline
- `test/test_c1_first_forward.py` — first megakernel forward
- `test/test_c2_capture_hf_layers.py` — HF golden capture
- `test/test_c3_layer_diff.py` — per-layer diff + first-divergence
- `test/test_c6_multitoken.py` — multi-token correctness sweep (NEW, pending run)

## Uncommitted at session end

Bash classifier outage prevented git operations in the final stretch.
Files that need to be committed:
  - `models/qwen36_27b/TODO.md`  (modified — C5/C6 status updates)
  - `models/qwen36_27b/test/test_c6_multitoken.py`  (new)
  - `models/qwen36_27b/megakernel/matvec.cuh`  (modified — NVFP4 matvec primitive)
  - `models/qwen36_27b/SESSION_NOTES.md`  (new — this file)

To commit when classifier recovers:
```
cd /home/sparkz/rl/lucebox-hub
git add models/qwen36_27b/TODO.md \
        models/qwen36_27b/test/test_c6_multitoken.py \
        models/qwen36_27b/megakernel/matvec.cuh \
        models/qwen36_27b/SESSION_NOTES.md
git commit -m "qwen36_27b: C5 done; C6 test + NVFP4 matvec primitive + session notes"
```

## Where to start next session

### 1. Run C6 (multi-token correctness, 5 min)
```
cd /home/sparkz/rl/lucebox-hub
HF_HOME=/home/sparkz/rl/.hf_cache \
    /home/sparkz/rl/.venv/bin/python3 \
    models/qwen36_27b/test/test_c6_multitoken.py
```
- Verifies kernel at S=1, 6, 17 (short prompts).
- If it passes: RoPE / KV cache / GQA at non-zero positions all work.
- If it fails: likely RoPE table generation issue or KV cache stride.

### 2. Wikitext sweep (10 min after C6)
Extend test_c6 to also pull a wikitext-2 window and test S=128, 512, 2048.

### 3. S1b: wire NVFP4 weights into FA/DN layers (1-2 days)
- Add `USE_NVFP4_WEIGHTS` trait to Cfg.cuh.
- In fa_layer.cuh / dn_layer.cuh, switch `matvec_bf16` -> `matvec_nvfp4`
  conditionally.
- Update LayerWeights<Cfg> struct to hold PackedMatrixNVFP4 alternatives.
- Plumb the quantized weights from `nvfp4_27b.quantize_27b_weights`.
- Memory drops from 50 GB BF16 -> ~14 GB NVFP4.

### 4. S3: NVFP4 KV cache wired in FA scan (S)
The NVFP4 KV helpers already exist standalone (head_dim=256 hardcoded
matches both 0.8B and 27B). Need to swap the bf16 K/V cache reads/writes
in fa_layer.cuh for NVFP4 packed variants.

### 5. S2: Parallel-S prefill kernel (multi-day)
Replace `prefill_qwen3x_naive` (host loop) with a proper parallel-S
kernel that tiles the S-dim and chunks the DN recurrence.

### 6. Other (per TODO.md)
- T1-T3: training memory mitigations (gradient checkpointing, NVFP4
  activation saves, sequence chunking) for S=32k training.
- F1-F6: streaming, KV reuse across requests, batching, vision tower.
- P1-P6: production hardening (eval, speed bench, memory regression).
- D1-D3: documentation.

## Key invariants discovered (don't break these)

1. **Q-proj output layout** is per-head interleaved
   `[Q_h0, gate_h0, Q_h1, gate_h1, ...]` with stride 2*head_dim
   between heads. NOT split `[Q_all, gate_all]`.
2. **Output gate uses plain sigmoid**, not silu — despite the config
   string saying "swish".
3. **All norm weights stored as deltas around 0.22**, so all norms
   use `(1 + w)` scaling, not plain `w`.
4. **FA layer is full transformer block**: attention + post-attn-norm
   + MLP + residual. Not just attention. The MLP weights are in the
   FA layer's weight struct (gate_proj, up_proj, down_proj).
5. **DN V/QK split** with `V_PER_QK=3` works correctly. Each V head
   gets its own recurrent state; Q/K are shared across V_PER_QK V heads
   per QK head.
6. **HF Qwen3.6-27B state_dict prefix** is `model.layers.X.*`, not
   the on-disk safetensors `model.language_model.layers.X.*` (HF's
   Qwen3_5TextModel strips the wrapper).

---

# Addendum — Session 2 (May 14, 2026 evening)

## TL;DR

**C7 closed: top-1 match vs HF on S=32, 64, 128, 256 wikitext.** Cos
0.97-0.998. Root cause of the "C7 hang" reported in Session 1 was a
**192 vs 176 pack stride mismatch** introduced by S1c.

## What landed

| Area | What | Status |
|---|---|---|
| Correctness | C7 long-context sweep | top-1 match S=32..256 |
| NVFP4 | S1c layer functions + 192B LayerWeights dispatch | done |
| NVFP4 | S1d runtime wireup + dispatch smoke test | done |
| NVFP4 | S1e correctness-vs-HF test scaffold | scaffold only |
| F2 | Multi-turn KV reuse (`prefill(start_position=...)`) | done + smoke |
| P2 | bench_pp_tg speed bench script | done |
| P3 | Memory regression test + baseline | done |
| P4 | NaN/Inf guard over long random-token decode | done |
| P5 | OOM-safe HF load (pre-flight VRAM check) | done |
| P6 | requirements.txt pinned | done |
| S5 | MTP key probe script | scaffold |
| S7a | BF16 LM head argmax kernel (skips fp32 cast) | done + matched torch |
| S2 | Parallel-S prefill — porting checklist | plan only |
| S3 | NVFP4 KV — wireup checklist | plan only |
| Docs | D2 migration writeup | done |
| Docs | D3 README refresh | done |
| Trainer | trainer/ scaffold + porting README | done (placeholder) |

## The pack stride bug

The 192-byte `LayerWeights<Cfg>` union (added in S1c to hold both BF16
and NVFP4 variants) was paired with a Python `PACK_STRUCT =
((8 + 21*8 + 15)//16)*16 = 176`. 21 ptrs was the wrong cap — DN_nvfp4
needs 22 ptrs. Kernel read each layer past 0 from a 16-byte-off
offset. The crash was always at the FA gate/up matvec
(matvec.cuh:83) because that was the first matvec with a "garbage"
weight pointer pointing into another allocation.

Fix: hardcode `PACK_STRUCT = 192` + `assert PACK_STRUCT == 192` at
module load. Now there's only one source of truth.

Diagnosed via compute-sanitizer with `-lineinfo`: the OOB read mapped
to `lucebox::qwen3x::load_128bit (in helpers.cuh:35) -> matvec_gate_up_silu
(matvec.cuh:83) -> full_attention_layer (fa_layer.cuh:328) ->
decode_kernel_impl (kernel_decode_full.cu:166)`. Once we had source
lines, the cause was immediate.

## Open items by effort (for next session)

- **S2** parallel-S prefill (L, 3-5 days) — biggest speed win
- **S3** NVFP4 KV wired (M, 2-3 days)
- **S5** real MTP head (M, 1-2 days after probe output)
- **S6** tree-verify kernel (L, deps S5)
- **T2/T3** training kernels (L, training-side)
- **F3** concurrent batching (L)
- **F4** vision tower (M)
- **S7b** cuBLASLt FP4 LM head (S)
- **S1e** actually run the NVFP4 correctness test (slow, ~30 min/run)
