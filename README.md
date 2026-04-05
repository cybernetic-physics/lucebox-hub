<p align="center">
  <img src="hero.png" width="600" />
</p>

<h1 align="center">Luce Megakernel</h1>

<p align="center">
  The first megakernel for hybrid DeltaNet/Attention LLMs.<br/>
  All 24 layers of Qwen 3.5-0.8B in a single CUDA dispatch.<br/>
  <a href="https://lucebox.com/blog/megakernel">Blog post</a> · <a href="RESULTS.md">Benchmarks</a> · <a href="https://discord.gg/NjweHTtTVj">Discord</a> · <a href="https://lucebox.com">lucebox.com</a>
</p>

---

```
                        Prefill      Decode      tok/J
Megakernel (RTX 3090)   37,800       413         1.87  @220W
llama.cpp  (RTX 3090)   11,247       267         0.76
Apple M5 Max               -         229         1.76
```

## What this is

A persistent CUDA kernel that processes the entire Qwen 3.5-0.8B forward pass in one dispatch. 18 DeltaNet layers + 6 attention layers, no CPU round-trips between them.

Qwen 3.5-0.8B uses a hybrid DeltaNet + Attention architecture (linear attention interleaved with standard attention). No fused kernel existed for this pattern. This is the first.

Inspired by [Hazy Research's megakernel work on Llama-1B](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles), we asked: can the same idea work for hybrid DeltaNet/Attention models on consumer GPUs?

## Run

```bash
pip install -e .
python bench_pp_tg.py
```

Requires NVIDIA Ampere+ GPU, CUDA 12+, PyTorch 2.0+. Tested on RTX 3090.

## How it works

Each token goes through all 24 layers inside one kernel (82 blocks x 512 threads). DeltaNet recurrence stays in F32 registers. KV cache updates in-kernel. Layers sync via cooperative grid instead of separate launches.

BF16 weights, BF16 activations, FP32 accumulation. Weights loaded directly from HuggingFace.

## Scope and limitations

This is a **research proof-of-concept**, not a production inference server.

- **Batch size 1 only.** This targets single-user local inference (the llama.cpp/Ollama use case), not multi-tenant serving. If you need batched throughput, use vLLM or SGLang.
- **Single model, single architecture.** The kernel is hand-written for Qwen 3.5-0.8B's specific layer pattern (18 DeltaNet + 6 Attention). It does not generalize to other models without rewriting.
- **BF16 only.** No quantization support (GGUF/GPTQ/AWQ). We benchmark at BF16 to isolate kernel-level efficiency from quantization tradeoffs.
- **0.8B parameters.** This is a small model. Megakernel fusion benefits shrink as model size grows and compute begins to dominate over launch overhead. We chose 0.8B because it's the first hybrid DeltaNet model available, not because it's representative of all workloads.
- **Power methodology.** Efficiency numbers measure accelerator power only (NVML for NVIDIA, `powermetrics` for Apple), following [Hazy Research's Intelligence Per Watt](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) methodology. Total system draw is higher for both platforms.
- **Correctness.** The benchmark includes an end-to-end correctness check comparing megakernel output against a reference decode path. See `bench_pp_tg.py`.

The goal is to demonstrate that architecture-specific kernel fusion eliminates a real efficiency gap on consumer hardware, and to do it in the open so others can reproduce, critique, and extend the work.

## Files

```
kernel.cu            Decode megakernel
prefill.cu           Prefill (cuBLAS + standalone kernels)
torch_bindings.cpp   PyTorch C++ bindings
model.py             Weight loading + Decoder
setup.py             Build
bench_pp_tg.py       Benchmark
```

## Why llama.cpp as a baseline?

llama.cpp is the most widely used local inference engine. It's what most people actually run on consumer GPUs. We also include PyTorch HuggingFace numbers (3.8x slower) for a second reference point. This is not a critique of llama.cpp, it's an excellent project. The comparison shows what architecture-specific optimization can unlock on top of a generic framework.

## Why an RTX 3090?

Deliberately chosen as the "worst case" for NVIDIA: a 2020 GPU, widely dismissed as power-hungry, available for ~$900-1,000 used. If the software gap is real on old hardware, it's even larger on newer cards.
## Community

Questions, ideas, or want to see what others are building? Join the [Luce Discord](https://discord.gg/NjweHTtTVj).

---

MIT · [Lucebox](https://lucebox.com)

Built with [Claude](https://claude.ai)
