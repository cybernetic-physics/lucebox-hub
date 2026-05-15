# Qwen3.6-27B megakernel (in progress)

Migration target: extend the hybrid-DeltaNet megakernel pattern from
`models/qwen35_0p8b/` to Qwen3.6-27B (dense 27B, multimodal, 262K
native context).

**Status**: HF backend + megakernel BF16 path both operational.
BF16 single- and multi-token forward passes top-1 vs HF on natural
text (cos > 0.998, top-1 match — see
[docs/results/qwen36_27b_gb10.md](docs/results/qwen36_27b_gb10.md)).
NVFP4 weight path is wired through the kernel (model_id=3) with a
dispatch smoke test passing; correctness vs HF on real weights is the
next gate. Parallel-S prefill (S2) is the open speed item.

Quick state:
- BF16 inference at ~50 GB GPU on GB10 (max_seq=1024)
- Multi-turn KV reuse via `prefill(prompt_ids, start_position=...)`
- OpenAI-compat server with SSE streaming
- Thinking mode, three tool-call formats, XGrammar — all working

See [PLAN.md](PLAN.md) for the phased migration roadmap.

## Quick start (HF-backed runtime, today)

```bash
# 1. Pull weights (~54 GB BF16; one-time, ~1 hour)
HF_HOME=/home/sparkz/rl/.hf_cache /home/sparkz/rl/.venv/bin/python3 -c \
    "from transformers import AutoModelForCausalLM; \
     AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.6-27B', dtype='bfloat16')"

# 2. Wiring tests (no model download required)
HF_HOME=/home/sparkz/rl/.hf_cache /home/sparkz/rl/.venv/bin/python3 \
    test/test_runtime_wiring.py

# 3. Serve OpenAI-compat
./run_hermes.sh                            # boot server, exit
./run_hermes.sh "Reply with one sentence." # one-shot prompt
./run_hermes.sh --interactive              # REPL

# 4. Correctness vs HF (regression bar for the future megakernel)
HF_HOME=/home/sparkz/rl/.hf_cache /home/sparkz/rl/.venv/bin/python3 \
    test/test_correctness_vs_hf.py
```

## Megakernel scaffold (Phase 1 on-ramp)

```bash
cd megakernel
/home/sparkz/rl/.venv/bin/python3 setup.py build_ext --inplace
/home/sparkz/rl/.venv/bin/python3 test_mlp_smoke.py
# expect: BOTH Cfg SPECIALIZATIONS PASS
```

## Layout

```
.
├── PLAN.md                            phased migration plan
├── runtime_hf.py                      HF-backed Decoder (working today)
├── run_hermes.sh                      one-command launcher
├── reference/
│   └── capture_hf_reference.py        HF golden capture
├── docs/
│   └── parameterization_audit.md      constexpr inventory
├── serve/
│   └── openai_server.py               OpenAI-compat HTTP server
├── test/
│   ├── test_runtime_wiring.py         wiring tests (no big download)
│   └── test_correctness_vs_hf.py      regression harness vs HF
├── megakernel/
│   ├── Cfg.cuh                        Cfg_0p8B + Cfg_27B tag structs
│   ├── kernel_decode.cu               templated MLP (Phase-1 scaffold)
│   ├── torch_bindings.cpp             qwen3x_C ops
│   ├── setup.py                       build script
│   ├── test_mlp_smoke.py              both Cfgs validated
│   └── README.md                      scaffold notes
└── trainer/                           (empty; populated in Phase 5+)
```

## Capabilities supported today (via HF backend)

| feature | status | notes |
|---|:---:|---|
| greedy + sampled decode | OK | via `runtime_hf.complete` / `chat` |
| chat template (multi-turn, system) | OK | from HF tokenizer's template |
| thinking mode `<think>...</think>` | OK | toggle via `GenerationConfig.enable_thinking` |
| tool calls (Qwen3 native + Hermes) | OK | parsed post-hoc; OpenAI-style response |
| grammar via XGrammar (JSON schema) | OK | `response_format.json_schema=` |
| grammar via GBNF/EBNF | OK | `response_format.grammar=` |
| 32k context | OK (slow) | HF runtime; megakernel needed for fast |
| streaming | OK | SSE via `serve/openai_server.py` (`stream=true`) |
| vision tower | not yet | text-only path for now |
| MTP speculative decode (chain) | OK | placeholder MTP head; real head TODO |
| MTP tree-verify | host driver | parallel-forward kernel TODO |
| NVFP4 weight quantization | dispatch ready | correctness vs HF TODO |
| Multi-turn KV reuse | OK | `prefill(start_position=...)` |

## Quickstart (once Phase 2 lands)

```bash
# 1. Download Qwen3.6-27B BF16 weights (54 GB, one-time)
HF_HOME=/home/sparkz/rl/.hf_cache \
    /home/sparkz/rl/.venv/bin/python3 -c \
    "from transformers import AutoModelForCausalLM; \
     AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.6-27B', \
         dtype='bfloat16')"

# 2. Capture HF reference logits (run once, ~few minutes)
HF_HOME=/home/sparkz/rl/.hf_cache \
    /home/sparkz/rl/.venv/bin/python3 \
    reference/capture_hf_reference.py --out reference/hf_golden.pt

# 3. (Phase 2) Compare our kernel to the golden:
#    /home/sparkz/rl/.venv/bin/python3 trainer/test_correctness_vs_hf.py
```

## Architecture pointer

Identical hybrid pattern to 0.8B but scaled up. See PLAN.md for the
exact dimension delta. Key constants the kernels need to learn:

```
NUM_LAYERS         = 64       # 0.8B: 24
HIDDEN_SIZE        = 5120     # 0.8B: 1024
INTERMEDIATE_SIZE  = 17408    # 0.8B: 3584
FA_NUM_Q_HEADS     = 24       # 0.8B: 8
FA_NUM_KV_HEADS    = 4        # 0.8B: 2
FA_HEAD_DIM        = 256      # 0.8B: 256  (unchanged — good)
DN_NUM_V_HEADS     = 48       # 0.8B: 16
DN_NUM_QK_HEADS    = 16       # 0.8B: 16  (unchanged)
DN_HEAD_DIM        = 128      # 0.8B: 128  (unchanged)
FA_ROTARY_DIM      = 64       # 0.8B: 64   (unchanged)
FA_ROPE_THETA      = 1e7      # 0.8B: 1e7  (unchanged)
VOCAB_SIZE         = 248320   # 0.8B: 248320 (unchanged)
MAX_CONTEXT        = 262144   # 0.8B: 65536  (4× longer)
```
