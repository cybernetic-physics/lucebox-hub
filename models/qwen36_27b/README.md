# Qwen3.6-27B megakernel (in progress)

Migration target: extend the hybrid-DeltaNet megakernel pattern from
`models/qwen35_0p8b/` to Qwen3.6-27B (dense 27B, multimodal, 262K
native context).

**Status**: Phase 0 done. You can serve Qwen3.6-27B today via the HF
backend (`runtime_hf` + `serve/openai_server.py`) — correctness baseline
with thinking-mode, native Qwen3 tool calls, and XGrammar-constrained
sampling. The megakernel speed path is Phase 1+; the templated `Cfg`
scaffold in `megakernel/` compiles cleanly for both 0.8B and 27B
specializations.

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
| streaming | not yet | future addition |
| vision tower | not yet | text-only path for now |
| MTP speculative decode | not yet | Phase 6 (megakernel-side) |

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
