# Qwen3.6-27B megakernel (in progress)

Migration target: extend the hybrid-DeltaNet megakernel pattern from
`models/qwen35_0p8b/` to Qwen3.6-27B (dense 27B, multimodal, 262K
native context).

**Status**: Phase 0 — scaffolding + HF reference correctness harness +
parameterization audit. The actual kernels still live in
`models/qwen35_0p8b/`; this directory holds the planning, reference
capture, and forthcoming 27B-specific code.

See [PLAN.md](PLAN.md) for the phased migration roadmap.

## Layout

```
.
├── PLAN.md                            phased migration plan
├── reference/
│   ├── capture_hf_reference.py        load HF Qwen3.6-27B, capture
│   │                                  reference logits + hidden states
│   │                                  for fixed prompts
│   └── (golden tensors land here once captured)
├── docs/
│   └── parameterization_audit.md      inventory of constexpr to
│                                      template for 0.8B → 27B
└── trainer/                           (empty; populated in Phase 2+)
```

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
