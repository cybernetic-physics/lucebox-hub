# Megakernel OpenAI server + Hermes agent

Two scripts that put the qwen35_0p8b megakernel decoder behind an
OpenAI-compatible HTTP endpoint and drive it with a Hermes-style
tool-calling agent loop.

## Quick start

```bash
# from models/qwen35_0p8b/
./run_hermes.sh "Use the echo tool with text='hi', then summarize."
```

That command starts `serve/openai_server.py` on port 8765, waits for
`/healthz`, and runs the agent against it. Server is torn down on exit;
set `KEEP_SERVER=1` to leave it running.

## Components

| File | What |
|------|------|
| `serve/openai_server.py` | FastAPI server. `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/healthz`. Greedy-only — `temperature`/`top_p` are ignored with a warning. |
| `serve/hermes_agent.py` | Hermes function-calling agent. Bundles `calculator`, `get_time`, `echo` tools. Parses `<tool_call>{json}</tool_call>` blocks from assistant turns. |
| `run_hermes.sh` | Boot + wait + invoke wrapper. Reads `BACKEND`, `PORT`, `MODEL`, `KEEP_SERVER`, `ATTACH` env. |

## CLI examples

```bash
# One-shot demo:
./run_hermes.sh "What time is it?"

# Interactive REPL:
./run_hermes.sh --interactive

# Attach to an already-running server (skip the boot):
KEEP_SERVER=1 ./run_hermes.sh "..."         # leaves the server up
ATTACH=http://127.0.0.1:8765 ./run_hermes.sh "..."

# NVFP4 decode path (greedy, lower throughput on GB10 right now):
BACKEND=nvfp4 ./run_hermes.sh "..."
```

## Direct HTTP

```bash
curl -s http://127.0.0.1:8765/v1/chat/completions \
  -H 'Content-Type: application/json' -d '{
    "messages":[{"role":"user","content":"hello"}],
    "max_tokens": 32
  }'
```

The endpoint is OpenAI-compatible enough to plug into any client library
that accepts a `base_url`. Tool-calling is done at the prompt level
(Hermes `<tool_call>` tags) rather than via OpenAI's `tools=` field, so
clients should not pass `tools=` in the request.

## Model caveat

Qwen3.5-0.8B is **way under-sized for reliable tool calling**. The
plumbing is exercised — `echo` round-trips cleanly — but complex
multi-step agent tasks (calculator with non-trivial math, planning) will
often loop or hallucinate. Swap in a stronger model by editing
`Decoder` / `model.py` only when the weights are compatible with the
megakernel's hard-coded Qwen3.5-0.8B shape constants (HIDDEN_SIZE,
INTERMEDIATE_SIZE, FA_*, DN_*). For larger models, target the 27B
variant in `models/qwen35_27b/`.
