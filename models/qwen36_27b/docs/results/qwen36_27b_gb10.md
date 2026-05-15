# Qwen3.6-27B on GB10 — results

## Correctness (May 2026)

### Single-token forward at position 0 vs HF reference

After fixing the FA MLP path (commit `8ac861d`, see [SESSION_NOTES.md](../../SESSION_NOTES.md)):

| Layer | Type | cos vs HF | max_abs |
|---:|:---:|---:|---:|
| 0 | DN  | 0.999995 | 0.117 |
| 1 | DN  | 0.999989 | 0.125 |
| 2 | DN  | 0.999978 | 0.250 |
| 3 | FA  | 0.999985 | 0.250 |
| 16 | DN | 0.999985 | 0.500 |
| 32 | DN | 0.999961 | 1.500 |
| 47 | FA | 0.999932 | 1.000 |
| 48 | DN | 0.999842 | 2.000 |
| 62 | DN | 0.999514 | 3.000 |
| 63 | FA | 0.992775 | 88.000 |

Final logits: **cos = 0.999640**, HF top-1 = 16 ('1'), ours top-1 = 16 ('1') ✓

### Multi-token forward (C6)

After the MRoPE text-only fix (pos_h = pos_w = pos_t):

| Prompt | S | HF top-1 | Ours top-1 | Match | Final cos |
|---|---:|---|---|:---:|---:|
| "Hello" | 2 | (varies) | (varies) | ✓ | 0.998 |
| "The capital of France is" | 5 | ' Paris' | ' Paris' | ✓ | 0.992 |
| "In the beginning..." | 16 | ' the' | ' the' | ✓ | 0.994 |

### Outstanding

- C7 (long context S=32–256 wikitext) — re-running after the stride
  fix. Previous "hang" was a stride mismatch between the 192-byte
  C++ struct and a 176-byte Python pack (PACK_MAX_PTR=21 in the
  S1c commit miscomputed `((8 + 21*8 + 15)//16)*16 = 176` instead
  of 192). The kernel read every layer past 0 from the wrong offset
  and crashed in the FA gate/up matvec with an OOB load. Fix:
  hardcode `PACK_STRUCT = 192` + asserts (commit c4b53e6).
- S>1024 untested; prefill_qwen3x_naive is host-looped so each S costs
  S × decode_kernel calls. Parallel-S prefill (S2) is the unblock.

## Memory budget on GB10 (BF16 weights, 121 GB unified)

| Workload | S | Weights | Inference scratch | Total |
|---|---:|---:|---:|---:|
| Inference | 1024 | 50.3 GB | 0.20 GB | 50.5 GB |
| Inference | 32768 | 50.3 GB | 2.14 GB | 52.4 GB |
| LoRA train | 8192 | 50.3 GB | + 34 GB acts + 4 GB LoRA | **89 GB** |
| LoRA train | 32768 | 50.3 GB | + 136 GB acts | **193 GB** (over) |
| Full train | any | + 27e9 × 12 B | over | **>356 GB** (over) |

Mitigations needed for S ≥ 16384 LoRA training:
- NVFP4 activation saves (~3.5× reduction)
- Gradient checkpointing
- Sequence chunked training

See [`training_memory_audit.py`](../../training_memory_audit.py) for
the full table.

## Speed

- HF reference forward at S=16 (BF16, torch-native attention): ~3–5 s
- Our prefill_qwen3x_naive at S=16 (decode-loop, BF16): ~1–2 s
- Decode kernel per-token latency: ~30–50 ms (HBM-bound, 50 GB / 273
  GB/s ≈ 180 ms theoretical; actual is faster because not every layer
  reads all weights at once)

Need parallel-S prefill (S2) for usable long-context speed.

## Feature surface (today)

| Feature | Status |
|---|:---:|
| Chat template (`apply_chat_template`) | ✓ |
| Thinking mode (`<think>…</think>` toggle + preserve flag) | ✓ |
| Tool calls — Qwen3 native ChatML format | ✓ |
| Tool calls — Hermes `<tool_call>` format | ✓ |
| Tool calls — qwen3_coder `<function=…>` format | ✓ |
| Grammar-constrained sampling (XGrammar JSON-schema) | ✓ |
| Grammar-constrained sampling (XGrammar GBNF) | ✓ |
| OpenAI-compat `/v1/chat/completions` | ✓ |
| OpenAI-compat `/v1/completions` | ✓ |
| Streaming via SSE (`stream=true`) | ✓ |
| MTP speculative decode (chain) | ✓ |
| MTP speculative decode (tree-verify) | host driver only |
| NVFP4 weight quantization | layer functions + dispatcher wired; correctness vs HF TODO |
| NVFP4 KV cache | helpers only; not wired into FA |
| Multi-turn KV reuse (KV prefill from `start_position`) | ✓ |
| Concurrent request batching | not yet |
| Vision tower | not yet |
| Long-context (>32k) via prefill_megakernel | requires S2 |
