# Hermes agent over the megakernel — quickstart

One-command launcher that puts the qwen35_0p8b megakernel decoder
behind an OpenAI-compatible HTTP endpoint and drives it with a
NousResearch-style `<tool_call>` agent loop.

## Commands

```bash
cd models/qwen35_0p8b

# default demo task (uses the echo tool — known-good wiring test)
./run_hermes.sh "Use the echo tool with text='hi', then summarize."

# interactive REPL — Ctrl-D to exit
./run_hermes.sh --interactive

# NVFP4 decode path (currently approximate — see Correctness note)
BACKEND=nvfp4 ./run_hermes.sh "..."

# leave the server running after the agent exits
KEEP_SERVER=1 ./run_hermes.sh "..."

# attach the agent to an already-running server (skip boot)
ATTACH=http://127.0.0.1:8765 ./run_hermes.sh "..."
```

Env knobs picked up by `run_hermes.sh`:

| Var          | Default                       | Purpose                                  |
|--------------|-------------------------------|------------------------------------------|
| `VENV`       | `/home/sparkz/rl/.venv`       | Python venv with torch + the C extension |
| `HF_HOME`    | `/home/sparkz/rl/.hf_cache`   | HuggingFace cache for Qwen weights       |
| `BACKEND`    | `bf16`                        | `bf16` / `nvfp4` / `auto`                |
| `PORT`       | `8765`                        | HTTP port for the server                 |
| `MODEL`      | `Qwen/Qwen3.5-0.8B`           | HF model name                            |
| `KEEP_SERVER`| `0`                           | If `1`, don't kill server on exit        |
| `ATTACH`     | unset                         | If set, skip boot and reuse this URL     |

## Components

| Path                                       | What it does                                                                                     |
|--------------------------------------------|--------------------------------------------------------------------------------------------------|
| `models/qwen35_0p8b/run_hermes.sh`         | Boots server, waits for `/healthz`, runs the agent, tears the server down on exit.               |
| `models/qwen35_0p8b/serve/openai_server.py`| FastAPI on `:8765`: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/healthz`. Greedy-only — `temperature`/`top_p` are ignored with a warning. Single `Decoder` instance behind a `Lock`. |
| `models/qwen35_0p8b/serve/hermes_agent.py` | Hermes `<tool_call>{json}</tool_call>` loop. Ships `calculator`, `get_time`, `echo` tools.       |
| `models/qwen35_0p8b/serve/README.md`       | Per-component reference (HTTP examples, model caveat).                                           |

## Direct HTTP

```bash
curl -s http://127.0.0.1:8765/v1/chat/completions \
  -H 'Content-Type: application/json' -d '{
    "messages":[{"role":"user","content":"hello"}],
    "max_tokens": 32
  }'
```

The endpoint is OpenAI-compatible enough for any client with a
`base_url`. Tool-calling happens at the prompt level (Hermes
`<tool_call>` tags), not via OpenAI's `tools=` field — clients should
not pass `tools=` in the request.

## Correctness notes

| Backend     | Greedy top-1 vs HF eager (32 steps) | First divergence |
|-------------|------------------------------------:|------------------|
| BF16        | **32/32 (100 %)**                  | none             |
| NVFP4 decode| 3/32 (9.4 %)                       | step 0           |

BF16 megakernel decode is bit-exact against HF on the test prompt.
NVFP4 diverges from step 0 because the LM head and all projections are
FP4-quantized; output stays coherent but greedy argmax flips. Use
`BF16` for evaluations.

## Model caveat

**Qwen3.5-0.8B is too small for reliable tool calling.** The wiring is
exercised — `echo` round-trips cleanly — but harder multi-step tasks
(e.g. calculator with multi-digit math) will loop or hallucinate. The
problem is model size, not the agent code. Swap in a stronger model by
keeping the megakernel's shape constants (`HIDDEN_SIZE`,
`INTERMEDIATE_SIZE`, FA_*/DN_*) compatible — see `model.py` — or jump
to the 27B target in `models/qwen35_27b/`.

## Verifying it works end to end

```bash
cd models/qwen35_0p8b
./run_hermes.sh "Use the echo tool with text='hi', then summarize."
```

Expected: server boots in ~15-20s, model emits a `<tool_call>` for
`echo`, the agent dispatches it, the model returns a one-line summary.
A non-empty `FINAL` block at the end means the full
`HTTP -> kernel -> tokenizer -> tool dispatch -> kernel` loop is alive.
