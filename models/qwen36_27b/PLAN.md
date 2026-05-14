# Qwen3.6-27B megakernel — migration plan

Target: `Qwen/Qwen3.6-27B` (BF16, dense 27B, hybrid Gated DeltaNet + Gated
Attention), released 2026-04-22. HF Transformers reference is the
correctness oracle.

## Why this ports cleanly

Qwen3.6-27B is **the same architecture family** as Qwen3.5-0.8B that the
megakernel in `models/qwen35_0p8b/` already targets. The hybrid layer
pattern is preserved: `N × (3 × DeltaNet → 1 × Gated Attention)`. Most
critically, the **head_dim, RoPE rotary dim, RoPE theta, and vocab are
identical** across the two models. Only the layer count, hidden /
intermediate dimensions, and head multiplicities change.

|                    | 0.8B (have)        | 27B (target)       | ratio |
|--------------------|--------------------|--------------------|-------|
| layers             | 24 (6 × 4)         | 64 (16 × 4)        | 2.67× |
| hidden             | 1,024              | 5,120              | 5×    |
| intermediate       | 3,584              | 17,408             | 4.86× |
| FA Q heads         | 8                  | 24                 | 3×    |
| FA KV heads        | 2                  | 4                  | 2×    |
| FA head_dim        | 256                | **256**            | 1×    |
| DN V heads         | 16                 | 48                 | 3×    |
| DN QK heads        | 16                 | **16**             | 1×    |
| DN head_dim        | 128                | **128**            | 1×    |
| RoPE rotary dim    | 64                 | **64**             | 1×    |
| RoPE theta         | 1e7                | **1e7**            | 1×    |
| vocab              | 248,320            | **248,320**        | 1×    |
| native context     | (n/a)              | 262,144 (YaRN)     | new   |
| BF16 weights       | 1.6 GB             | 54 GB              | 34×   |

`MRoPE` (interleaved, sections `[11, 11, 10]`) is new — the rotary
position encoding splits across (T, H, W) for multimodal. For text-only,
all three sections concatenate on the temporal axis.

## Hardware budget on GB10 (sm_121a, 128 GB unified)

| component                              | BF16    | NVFP4 weights |
|----------------------------------------|--------:|--------------:|
| model weights                          |  54 GB  |  ~15 GB       |
| FA KV cache @ 32k ctx, NVFP4 (helpers shipped) | — | ~1 GB |
| FA KV cache @ 256k ctx, NVFP4          |   —     |  ~8 GB        |
| DN state (per-layer recurrent)         |  tiny   |  tiny         |
| training activation saves @ S=32k      | ~30 GB  |  (FP8/NVFP4-able) |
| **total inference @ 32k**              | ~57 GB  | **~16 GB**    |
| **total train fwd @ S=32k**            | ~85 GB  | ~30 GB        |

Inference fits BF16 today on GB10. NVFP4 weights unlock 24 GB-class
cards (RTX 4090, 5090) and free headroom for longer-context training.

## Phased deliverables

### Phase 0 — Foundation [DONE]

- [x] Directory scaffold (`models/qwen36_27b/`).
- [x] HF reference harness (`reference/capture_hf_reference.py`).
- [x] Parameterization audit (`docs/parameterization_audit.md`).
- [x] HF-backed `runtime_hf.Qwen36Runtime` with thinking/tools/grammar/
      chat-template support. Working correctness baseline today.
- [x] OpenAI-compatible server (`serve/openai_server.py`).
- [x] Launcher script (`run_hermes.sh`).
- [x] Wiring tests (`test/test_runtime_wiring.py`) — 10/10 pass.
- [x] Correctness regression harness (`test/test_correctness_vs_hf.py`)
      — skip-with-message until 54 GB weights are pulled.
- [x] Megakernel Phase-1 scaffold (`megakernel/`) — `Cfg_0p8B` and
      `Cfg_27B` tag structs, templated `mlp_forward<Cfg>`, both
      specializations compile + execute on GB10 (`test_mlp_smoke.py`
      passes with cos=1.000).

### Phase 1 — Parameterize 0.8B kernels (1–2 days)

Make the existing kernels work for both 0.8B and 27B. Strategy: template
the per-model constants on a `ModelConfig` struct passed as a `__constant__`
mem block; per-call kernels read it. Keep the existing 0.8B constexpr
inlining as a `static constexpr ModelConfig CFG_0P8B` so the compiler
specializes when the model is known at compile time.

Files touched:
- `kernel.cu` (decode FA + DN + MLP) — ~1000 lines, all constexpr
- `prefill.cu` (multi-token prefill) — ~1700 lines
- `prefill_megakernel.cu` — ~1100 lines
- `prefill_bw.cu` (training fwd with activation saves) — ~1500 lines
- `kernel_gb10_nvfp4.cu` (NVFP4 weights path) — ~2300 lines

Regression: existing 0.8B benches must produce identical output (bit
exact for `prefill_bf16_mega`, ≤bf16-noise for `prefill_bf16`).

### Phase 2 — 27B BF16 forward parity (3–5 days)

Build per-layer weight packer for 27B's shapes. Load BF16 weights from
HF, pack into kernel layout, run `prefill_bf16_mega` on natural text and
wikitext-2 windows, compare to HF reference captured in Phase 0.

Acceptance: top-1 match ≥ 95% over 1024 wikitext windows at S=512,
cos sim ≥ 0.999, KL ≤ 0.01 nats. Same bar as `prefill_bf16_mega` on
0.8B vs HF-eager.

### Phase 3 — YaRN + MRoPE (2–3 days)

Add YaRN scaling factor + MRoPE-interleaved positional encoding. Affects
the per-token RoPE table in `kernel.cu`'s FA layer. Test at S=32k vs HF.

### Phase 4 — NVFP4 KV @ 27B (1 day)

Re-use `nvfp4_kv.cuh` helpers (already shipped). Allocate
`fa_k_cache_data/scales`, `fa_v_cache_data/scales` at 27B's shapes,
swap the FA cache reads/writes. KV memory: 32k context, 16 FA layers, 4
KV heads, head_dim=256 → bf16=1 GB, nvfp4=288 MB (3.5× cut).

### Phase 5 — NVFP4 weights @ 27B (3–5 days)

Port the existing 0.8B NVFP4 weight quantizer to 27B's projection
shapes. Hot path: cuBLASLt FP4 LM head (already validated), per-token
NVFP4 GEMV in decode. Validate top-1 stability at <1% PPL drift.

### Phase 6 — MTP speculative decode (1 week)

Qwen3.6 ships with a native multi-token-prediction head (NEXTN). Re-use
the existing DFlash + DDTree infra in `models/qwen35_27b/` (currently
ggml-based) — port the tree-verify algorithm to our megakernel runtime.
Expect 2–3× decode throughput at AL≈6 on chat workloads.

### Phase 7 — Grammar (XGrammar) (2 days)

Logit-mask hook in `decode` between `lm_head` and `argmax`. Use
`xgrammar` (Outlines-compatible) for the FSM. Apply mask to bf16 logits
before final argmax. Compatible with tool-call structured output.

### Phase 8 — Tool calls (1 day)

Application-level. `serve/hermes_agent.py` already does Hermes-style
`<tool_call>{json}</tool_call>` parsing. Add `qwen3_coder` parser (the
native Qwen tool format) as an alternative.

### Phase 9 — Thinking mode (1 day)

Application-level. Chat template injects `<think>...</think>` markers.
Stop-on-`</think>` is just a stop-token rule. The model is trained to
emit a final answer after the thinking block.

### Phase 10 — Vision tower (3–5 days)

ViT preprocessor as a separate stage (HF Transformers). Image tokens
splice into the LM input. Multimodal path is optional — `--language-
model-only` flag mirrors HF behavior.

## Correctness oracle

`reference/capture_hf_reference.py` captures fixed reference outputs
from `Qwen/Qwen3.6-27B` HF transformers. Two modes:

1. **Loose oracle**: top-1 / top-5 / cos / KL on natural-text and
   wikitext-2 prompts. Bar: cos ≥ 0.999 on natural text at S ≤ 2k.
2. **Strict oracle** (when `fla` is installed in HF's env): bit-exact
   logits at S ≤ 2k. Stretch.

## Build target

GB10 sm_121a (primary). The kernel structure should remain
sm_86/sm_100/sm_120/sm_121-clean via the existing arch gates; the NVFP4
paths drop out on pre-Blackwell.

## Risks

- **YaRN/MRoPE** is genuinely new code, not a port. Risk: subtle
  RoPE bugs corrupt long-context accuracy without surfacing at S ≤ 2k.
  Mitigation: capture HF reference logits at S=4k, 8k, 16k, 32k.
- **MTP head** weight layout is undocumented for the public release.
  May require reverse-engineering from the safetensors.
- **27B BF16 weight load** is 54 GB and slow to download from HF —
  budget ~1 hour for the first pull on a typical connection.
- **HF correctness oracle drift**: HF's torch-native DeltaNet fallback
  is what we see in the 0.8B path today (cos ~0.95 at long S). Bit-
  exact requires installing `fla` in the HF env, which has its own
  build requirements on aarch64.
