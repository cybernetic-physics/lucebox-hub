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

## Status update (post second session)

| component | status | notes |
|---|---|---|
| `mlp_forward<Cfg>` | **done, tested** | cos=1.000 vs torch ref on both 0.8B and 27B shapes |
| `full_attention_layer<Cfg>` | **done, compiles** | Cfg-templated RMSNorm + Q/K/V proj + per-head QK-norm + RoPE + split-K online softmax + O-proj. End-to-end test pending the 54 GB HF weight pull |
| `delta_net_layer<Cfg>` | **done, compiles** | with the DN V/QK split (`V_PER_QK`). Per-V-head recurrent state of shape [V_HEADS, VAL, KEY]. End-to-end test pending HF weights |
| `rope.cuh` (YaRN + MRoPE) | **done** | NTK-aware ramp blend + MRoPE-interleaved sections `{11, 11, 10}`. Text-only path concatenates on temporal axis (h=w=0) |
| `decode_kernel<Cfg>` | **done, compiles** | persistent megakernel layer walker — both Cfg specializations build cleanly for sm_121a |
| `prefill_megakernel<Cfg>` | TODO | multi-token; same primitives, different shmem layout |
| 27B weight packer | **done** | `../weight_packer.py` with HF state-dict mapping + shape checks per layer |
| NVFP4 KV at 27B | **done (Python plumbing)** | `../nvfp4_27b.py` — re-uses the 0.8B helpers (head_dim=256 identical). Footprint: 0.56 GB at 32k ctx, 4.5 GB at 262k ctx |
| NVFP4 weights at 27B | **done (Python plumbing)** | applies the existing optimal-MSE quantizer. Footprint: 13.3 GB total. **Fits on a 24 GB consumer card** |
| MTP speculative decode | **chain version implemented**, tree-verify TODO | `../mtp_speculative.py`. Chain MTP works against any runtime; tree-verify needs `prefill_megakernel_tree<Cfg>` (~3 days kernel work) |

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
