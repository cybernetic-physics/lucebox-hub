<p align="center">
  <strong>Open LLM inference, rewritten by hand for one specific chip at a time.</strong><br/>
  Kernels, speculative decoding, quantization, and LoRA training, tailored per target.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-e8e8ed?style=for-the-badge&labelColor=090909" alt="MIT"></a>
  <a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-12%2B-76b900?style=for-the-badge&logo=nvidia&logoColor=76b900&labelColor=090909" alt="CUDA 12+"></a>
  <a href="https://isocpp.org"><img src="https://img.shields.io/badge/C%2B%2B-17-e8e8ed?style=for-the-badge&logo=cplusplus&logoColor=e8e8ed&labelColor=090909" alt="C++17"></a>
</p>

---

## Inside the repo

- **`models/qwen35_0p8b/`** — fused megakernel forward pass for Qwen3.5-0.8B (BF16 + NVFP4), plus LoRA training + inference on B200 (sm_100) and RTX 3090 (sm_86).
- **`models/qwen35_27b/`** — DFlash + DDTree speculative decoding for Qwen3.5-27B (Q4_K_M target + BF16 draft).
- **`docs/roadmap/`** — per-(model, arch) plans for in-progress work.
- **`docs/results/`** — per-(model, arch) benchmark reports.

---

## 01 · Megakernel Qwen3.5-0.8B

**The first megakernel for hybrid DeltaNet / Attention LLMs.** All 24 layers of Qwen3.5-0.8B in a single CUDA dispatch.

```bash
# 1. clone + enter
git clone https://github.com/cybernetic-physics/lucebox-hub && cd lucebox-hub/models/qwen35_0p8b

# 2. install (Python 3.10+, CUDA 12+, PyTorch 2.0+). Weights stream from HF on first run.
pip install -e .

# 3. run the benchmark (prefill pp520 + decode tg128 vs llama.cpp BF16 + PyTorch HF)
python final_bench.py
```

### RTX 3090 (sm_86)

| Method | Prefill pp520 | Decode tg128 | tok/J |
|--------|:-------------:|:------------:|:-----:|
| **Megakernel** `@220W` | **37,800** | **413** | **1.87** |
| llama.cpp BF16 `@350W` | 11,247 | 267 | 0.76 |
| PyTorch HF | 7,578 | 108 | n/a |

### NVIDIA B200 (sm_100)

| Method | Prefill pp520 (tok/s) | Decode tg128 (tok/s) | tg128 vs llama.cpp |
|:-------|:-:|:-:|:-:|
| **Megakernel BF16** | **40,278** | **711** | **1.63×** |
| llama.cpp BF16 (CUDA, ngl=99) | 26,781 | 437 | 1.00× |
| PyTorch HuggingFace BF16 | 1,797 | 27 | 0.06× |

[Full writeup →](models/qwen35_0p8b/README.md) · [B200 benchmarks →](docs/results/qwen35_0p8b_b200.md)

---

## 02 · LoRA training + inference on B200

Rank-16 LoRA adapters on all 13 trainable Qwen3.5-0.8B projections (FA q/k/v/o + gate/up/down, DN qkv/z/out + gate/up/down), plumbed into the existing cuBLAS + cudaGraph prefill body via `prefill_bf16_with_lora`. Two cuBLAS `GemmEx` calls per projection (`beta=1` accumulates directly into the base output — no residual-add kernel needed). Activation saving during the training forward (`prefill_bf16_train_step`) populates four per-layer bf16 slabs (`hidden_in`, `normalized_in`, `normalized_post_attn`, `mlp_inter`) consumed by the backward pass. All LoRA-disabled branches collapse to one null-check per projection per layer — inference overhead is within noise.

### Correctness: wikitext-2 perplexity vs HF PyTorch

Downloaded PEFT adapter `eac123/subliminal-qwen3.5-0.8b-tiger-r5` (r=8), repacked into our kernel's per-type-layer-major A/B layout. Scored on 4096 tokens of wikitext-2 test, 512-token windows stride 256:

| path                          |  PPL   |  +adapter | adapter Δ |
|:------------------------------|:------:|:---------:|:---------:|
| HF Qwen3.5-0.8B (torch-DN)    | 12.15  |  15.02    |  +2.86    |
| **ours (cuBLAS+graph)**       | 12.84  |  15.62    |  +2.78    |
| base drift                    | 5.7 %  |           |           |
| adapter magnitude ratio       |        |           | **0.97×** |

Base perplexity matches HF within 5.7 % and the adapter's effect is reproduced to within 3 % of HF on the same weights — end-to-end correctness verified against PyTorch.

### Speed vs PyTorch (B200, rank-16 LoRA on 13 projections)

Baseline: HF transformers + matching LoRA wrapped as `LoraLinear` around each target `nn.Linear`, fed the same tokens.

| stage                          |   S   |  PyTorch HF (ms) |  ours (ms) |  speedup |
|:-------------------------------|:-----:|-----------------:|-----------:|---------:|
| prefill fwd (no LoRA)          |  128  |  254.2           |   6.54     | **38.9×** |
| prefill fwd + LoRA             |  128  |  254.2           |   6.55     | **38.8×** |
| training fwd (+activ saves)    |  128  |  602.9           |   6.95     | **86.7×** |
| prefill fwd (no LoRA)          |  512  |  262.2           |  15.32     | **17.1×** |
| prefill fwd + LoRA             |  512  |  262.2           |  15.32     | **17.1×** |
| training fwd (+activ saves)    |  512  |  750.5           |  15.70     | **47.8×** |
| prefill fwd (no LoRA)          | 1024  |  294.2           |  28.92     | **10.2×** |
| prefill fwd + LoRA             | 1024  |  294.2           |  28.94     | **10.2×** |
| training fwd (+activ saves)    | 1024  |  898.0           |  29.34     | **30.6×** |
| fused AdamW (on LoRA params)   |  —    |    3.36          |   0.027    | **123.6×** |

Reproduce the full end-to-end check (correctness + speed):

```bash
cd models/qwen35_0p8b
pip install -e .
python trainer/bench_lora_e2e.py --ppl-tokens 4096 --ctx-len 512 --ppl-stride 256 --bench-seq-lens 128 512 1024
```

LoRA adds no measurable overhead over base prefill (the two extra cuBLAS GEMMs per projection are buried in the 24-layer fold). Activation saving for the training forward costs ~2-4 % on top of pure inference.

**Status:** Forward and fused-AdamW are shipped and numerically matched against HF. Per-layer backward (FA bwd + DeltaNet BPTT) is Phase 2 WIP — see [`docs/roadmap/lora_training_engine.md`](docs/roadmap/lora_training_engine.md) for the plan.

[LoRA design →](docs/roadmap/lora_training_engine.md) · [Full B200 writeup →](docs/results/qwen35_0p8b_b200.md)

---

## 03 · DFlash + DDTree Qwen3.5-27B on RTX 3090

**First GGUF port of DFlash speculative decoding.** Qwen3.5-27B on a single RTX 3090, Q4_K_M target + BF16 draft, DDTree budget=22.

- **Up to 207 tok/s** in the demo (207.6 tok/s DFlash vs 38.0 tok/s AR, 5.46×)
- **129.5 tok/s mean** on the HumanEval 10-prompt bench
- **3.43× faster than autoregressive** (+15% over chain speculative decoding)
- **2.8× faster than SGLang AWQ** on the same hardware
- **128K context in 24 GB** (134.78 tok/s at ctx=131072)

```bash
# 1. clone with submodules (pulls the pinned llama.cpp fork)
git clone --recurse-submodules https://github.com/cybernetic-physics/lucebox-hub && cd lucebox-hub/models/qwen35_27b

# 2. build the C++/CUDA decoder (~3 min on sm_86, CUDA 12+, CMake 3.18+)
cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cmake --build build --target test_dflash -j

# 3. fetch weights: ~16 GB Q4_K_M target + 3.46 GB bf16 draft
huggingface-cli download unsloth/Qwen3.5-27B-GGUF Qwen3.5-27B-Q4_K_M.gguf --local-dir models/
huggingface-cli download z-lab/Qwen3.5-27B-DFlash model.safetensors --local-dir models/draft/

# 4a. one-shot streaming generate
python3 scripts/run.py --prompt "def fibonacci(n):"

# 4b. or reproduce the paper-style bench (HumanEval + GSM8K + Math500, ~15 min)
python3 scripts/bench_llm.py
```

| Benchmark | AR (tok/s) | DFlash+DDTree (tok/s) | Speedup |
|-----------|:----------:|:---------------------:|:-------:|
| **HumanEval** | 37.8 | **129.5** | **3.43×** |
| Math500 | 37.7 | 110.5 | 2.93× |
| GSM8K | 37.7 | 96.2 | 2.55× |

**The constraint that shaped the project.** AWQ INT4 of Qwen3.5-27B plus the BF16 draft doesn't leave room for the DDTree verify state on a 24 GB card. Q4_K_M GGUF (~16 GB target) is the largest format that fits target + 3.46 GB draft + budget=22 tree state + KV cache in 24 GB on the RTX 3090. Picking it forced a new port on top of ggml, since no public DFlash runtime supports a GGUF target.

**What we built vs what we didn't.** The algorithms are not ours:
- [**DFlash**](https://arxiv.org/abs/2602.06036) (z-lab, 2026): block-diffusion draft conditioned on target hidden states.
- [**DDTree**](https://arxiv.org/abs/2604.12989) (Ringel et al., 2026): tree-structured verify that beats chain verify at the same compute budget.

What we ported and tuned:
- C++/CUDA decode engine on top of ggml (no libllama, no Python runtime, Q4_K_M target path).
- Three custom CUDA kernels for tree-aware SSM state rollback: `ggml_ssm_conv_tree`, `ggml_gated_delta_net_tree`, `ggml_gated_delta_net_tree_persist`.
- DDTree budget swept for RTX 3090 + Q4_K_M target: **budget=22** is the sweet spot.
- Q4_0 KV cache + sliding `target_feat` ring to fit 128K context in 24 GB with ~3% AL hit.

[Full writeup →](models/qwen35_27b/README.md) · [Benchmarks →](models/qwen35_27b/RESULTS.md)

> **Qwen3.6-27B (experimental):** same `qwen35` architecture, so the 3.6 Q4_K_M GGUF loads as a drop-in target. With the 3.5-trained draft, throughput lands around ~74 tok/s on HumanEval (vs 129.5 on 3.5). Details in [models/qwen35_27b/README.md](models/qwen35_27b/README.md#qwen36-27b-target-experimental).

---

## Why this exists

Local AI should be a default, not a privilege: private data, no per-token bill, no vendor lock-in. The hardware to run capable models already sits on desks. The software to run those chips well doesn't.

General-purpose frameworks dominated the last decade because hand-tuning kernels per chip was too expensive to justify. One stack, decent on everything, great on nothing. Most of the silicon's capability stays on the floor.

AI-assisted development flips that calculus. Rewrites that took a quarter now fit in a release cycle. This repo publishes them, one chip and one model family at a time. MIT source, full writeup, reproducible benchmarks.

---

## Requirements

Built and benchmarked on NVIDIA RTX 3090 (sm_86), NVIDIA B200 (sm_100), and NVIDIA GB10 (sm_121a); portable to other Ampere+ NVIDIA GPUs with minor tuning. CUDA 12+, PyTorch 2.0+.

DFlash needs CMake 3.18+ and `--recurse-submodules` for the pinned `llama.cpp` fork (three tree-mode ggml ops).

**Optional, find your GPU's sweet spot:** `sudo nvidia-smi -pl 220` (megakernel hits best tok/J at 220 W on RTX 3090).

---

## Inspired by

- [Hazy Research](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles): megakernel idea and the intelligence-per-watt methodology.
- [z-lab/DFlash](https://arxiv.org/abs/2602.06036) (Wang et al., 2026): block-diffusion speculative decoding algorithm. We use their published Qwen3.5-27B-DFlash draft weights as-is.
- [DDTree](https://arxiv.org/abs/2604.12989) (Ringel & Romano, 2026): tree-structured verify that DFlash 27B uses for its 3.5× speedup over chain spec decoding. [liranringel/ddtree](https://github.com/liranringel/ddtree).
- [AlpinDale/qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel), [Infatoshi/MegaQwen](https://github.com/Infatoshi/MegaQwen): prior art on fused Qwen kernels.

---

<p align="center">
  <sub><a href="LICENSE">MIT</a></sub>
</p>
