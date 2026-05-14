# Qwen3.x templated megakernel — Phase 1 scaffold

This directory holds the on-ramp for the Qwen3.6-27B megakernel port.
Strategy: **template every device function on a `Cfg` tag struct** so
both Qwen3.5-0.8B and Qwen3.6-27B compile from the same source, with
nvcc fully specializing each cubin per model.

## What works today

- `Cfg.cuh` — `FamilyInvariants` + `Cfg_0p8B` + `Cfg_27B` tag structs.
- `kernel_decode.cu` — templated `rmsnorm<Cfg>`, `fused_silu_mul<Cfg>`,
  `matvec_bf16_row<Cfg>`, `mlp_forward<Cfg>`. Two extern-C entry points
  per Cfg.
- `torch_bindings.cpp` — `torch.ops.qwen3x_C.mlp_smoke_0p8b/27b`.
- `setup.py` — builds the extension for sm_121a (or whatever the
  current GB10 driver reports).
- `test_mlp_smoke.py` — runs both Cfg specializations end-to-end vs a
  PyTorch reference. Both pass (cos=1.000, rel_max<0.01).

Build + test:
```
/home/sparkz/rl/.venv/bin/python3 setup.py build_ext --inplace
/home/sparkz/rl/.venv/bin/python3 test_mlp_smoke.py
```

## What's left (clearly scoped)

| component | status | notes |
|---|---|---|
| `mlp_forward<Cfg>` | done | RMSNorm + SwiGLU + down + residual |
| `full_attention_layer<Cfg>` | TODO | port from `models/qwen35_0p8b/kernel.cu:409`; mostly parameterization (GQA ratio 24/4 vs 8/2 just scales) |
| `delta_net_layer<Cfg>` | **TODO new code** | DN V/QK split (`DN_V_PER_QK`) — 0.8B is 1, 27B is 3. Each QK head feeds 3 V heads via GQA-style replication |
| `yarn_rope<Cfg>` | TODO | Qwen3.6 RoPE extension to 262k; 0.8B doesn't use it |
| `mrope_interleaved<Cfg>` | TODO | multimodal RoPE; text-only path concatenates sections on the temporal axis |
| `decode_kernel<Cfg>` | TODO | the persistent megakernel that walks all `NUM_LAYERS` layers per token (cooperative-groups grid sync) |
| `prefill_megakernel<Cfg>` | TODO | multi-token prefill — port `models/qwen35_0p8b/prefill_megakernel.cu` |
| 27B weight packer | TODO | python; mirror `models/qwen35_0p8b/model.py:_pack_layer_weights` for the 27B shapes |
| NVFP4 KV at 27B | TODO | re-use `models/qwen35_0p8b/nvfp4_kv.cuh` — same head_dim=256 |
| NVFP4 weights at 27B | TODO | port the existing 0.8B optimal-MSE quantizer |
| MTP speculative decode | TODO | Qwen3.6 ships a native NEXTN draft head |

## Phase ordering recommendation

1. **`full_attention_layer<Cfg>`** — port verbatim from 0.8B, just
   parameterize the constants. Validate 0.8B path stays bit-equal to
   `prefill_bf16_mega` reference.
2. **`delta_net_layer<Cfg>`** — implement the V/QK split. This is the
   one real architectural change. Compare against HF reference logits
   captured by `models/qwen36_27b/reference/capture_hf_reference.py`.
3. **`decode_kernel<Cfg>` + `prefill_megakernel<Cfg>`** — wire everything
   together, port the existing 0.8B mega/prefill structure.
4. **YaRN + MRoPE** — needed for context > 32k. Validate at 32k, 64k.
5. **NVFP4 KV + weights** — apply the existing 0.8B quantizers to 27B.
6. **MTP speculative head** — port the existing DFlash+DDTree from
   `models/qwen35_27b/` (currently ggml-based, port to our runtime).

## Why this is the right shape

The Qwen3.x family shares a lot more than the parameter count suggests:
head_dim=256 on both FA and DN-QK, DN_key_dim=DN_value_dim=128,
rotary_dim=64, rope_theta=1e7, vocab=248320, layer pattern (3 DN + 1
FA), unified BF16+NVFP4 quantization layout. **Only seven dimensions
actually differ between 0.8B and 27B** (see
`../docs/parameterization_audit.md`).

Once the Phase-1 refactor is done, future Qwen3.x releases (e.g. a
Qwen3.7-50B) become a one-struct addition: write `Cfg_50B`, add a
specialization point, recompile. No new kernel logic.
