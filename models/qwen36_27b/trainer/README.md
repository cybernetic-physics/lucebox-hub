# Qwen3.6-27B trainer (placeholder)

This directory is reserved for the 27B training pipeline. **Nothing
runs from here yet.** The 0.8B trainer at `models/qwen35_0p8b/trainer/`
has the full reference: LoRA wrap, FA/DN forward+backward kernels,
chunked DN scan, NVFP4 activation save, CUDA Graphs, end-to-end PPL
test against HF.

## What needs to be ported

### Kernel side (the bulk of the work)

| 0.8B file | 27B equivalent | Cfg changes |
|---|---|---|
| `kernel.cu` | done — see `../megakernel/*` | Cfg-templated |
| `dn_chunked.cu` | TODO — DN chunk-parallel forward | DN V/QK split (V_PER_QK=3) |
| `dn_bwd.cu` | TODO — DN chunked backward | DN V/QK split |
| `fa_bwd_flash.py` | TODO — FA backward | 24 Q heads, 4 KV heads |
| (none yet) | TODO — NVFP4 activation save kernels (T2) | matches 0.8B `_fwd_save_act_nvfp4` |

The DN backward is the algorithmic bulk: it inverts the chunked
forward scan, which already exists for V_PER_QK=1 in the 0.8B path.
Cfg-templating it should make most of the code reusable; the per-V-
head loops just need the V_PER_QK indexing.

### Python side

| 0.8B file | What 27B needs |
|---|---|
| `dn_autograd.py` | wrap the kernel as torch.autograd.Function |
| `lora_hf_wrap.py` | apply LoRA to the HF Qwen3.6 model attrs |
| `cuda_graph_train.py` | capture train step under cudaGraph for speedups |
| `bench_lora_e2e.py` | end-to-end PPL vs HF after k optimizer steps |

### Memory budgets (per T1 audit)

At S=8192, LoRA training fits in 121 GB with NVFP4 weights + plain
activation save (no checkpointing). At S=32768, need NVFP4 acts and/or
gradient checkpointing — without those, activations alone are 136 GB.

See `../training_memory_audit.py` for the table.

## Suggested first step

1. Port `bench_dn_chunk_parallel.py` (correctness vs HF on a single
   DN forward+backward at 27B dims). That validates the V/QK split
   in the backward path before wiring full training.
2. Port `dn_bwd.cu` with Cfg templating.
3. Port `fa_bwd_flash.py` for 27B FA dims.
4. Put it all together via `dn_autograd.py` + `lora_hf_wrap.py`.

Estimated effort: 1-2 weeks for an experienced kernel author.

## What works today (no trainer needed)

- Inference via `../runtime_hf.py` (HF backend, BF16, all features)
- Inference via `../runtime_megakernel.py` (BF16 megakernel — top-1
  vs HF on wikitext S=32..256)
- Chat / tools / thinking / grammar / streaming through
  `../serve/openai_server.py`

For RL training at scale, the 0.8B path is the reference until this
directory is filled in.
