#!/usr/bin/env bash
# One-command launcher for the Qwen3.6-27B HF runtime + OpenAI server.
#
# Mirrors models/qwen35_0p8b/run_hermes.sh but points at the 27B runtime.
# Boots the server, waits for /healthz, optionally fires a chat request,
# and tears the server down on exit.
#
# Env knobs (with defaults):
#   VENV        : python venv  (default: /home/sparkz/rl/.venv)
#   HF_HOME     : HF cache dir (default: /home/sparkz/rl/.hf_cache)
#   BACKEND     : bf16 | fp8 | bnb-4bit (default: bf16)
#   PORT        : HTTP port (default: 8765)
#   MODEL       : HF repo (default: Qwen/Qwen3.6-27B)
#   KEEP_SERVER : 1 to leave server running after the agent exits
#   ATTACH      : if set, skip boot and use this URL
#
# Usage:
#   ./run_hermes.sh                          # boot server, exit
#   ./run_hermes.sh "What is 7*11?"          # one-shot prompt
#   ./run_hermes.sh --interactive            # REPL (Ctrl-D to exit)
set -euo pipefail

VENV=${VENV:-/home/sparkz/rl/.venv}
HF_HOME=${HF_HOME:-/home/sparkz/rl/.hf_cache}
BACKEND=${BACKEND:-bf16}
PORT=${PORT:-8765}
MODEL=${MODEL:-Qwen/Qwen3.6-27B}
KEEP_SERVER=${KEEP_SERVER:-0}

THIS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 ; pwd -P)"
cd "$THIS_DIR"

server_pid=""
cleanup() {
  if [[ "$KEEP_SERVER" != "1" && -n "$server_pid" ]]; then
    echo "[run_hermes] shutting down server pid=$server_pid"
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ -z "${ATTACH:-}" ]]; then
  echo "[run_hermes] booting server  backend=$BACKEND  port=$PORT  model=$MODEL"
  HF_HOME="$HF_HOME" "$VENV/bin/python3" serve/openai_server.py \
      --backend "$BACKEND" --model "$MODEL" --port "$PORT" &
  server_pid=$!
  echo "[run_hermes] server pid=$server_pid; waiting for /healthz..."
  for i in $(seq 1 300); do
    if curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then
      echo "[run_hermes] server ready"
      break
    fi
    sleep 1
    if (( i == 300 )); then
      echo "[run_hermes] server did not come up in 5 minutes" >&2
      exit 1
    fi
  done
  BASE_URL="http://127.0.0.1:$PORT"
else
  BASE_URL="$ATTACH"
  echo "[run_hermes] attaching to $BASE_URL"
fi

if [[ $# -gt 0 ]]; then
  if [[ "$1" == "--interactive" ]]; then
    echo "[run_hermes] interactive mode (Ctrl-D to exit)"
    while IFS= read -r -p "> " line; do
      [[ -z "$line" ]] && continue
      curl -s "$BASE_URL/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$(jq -nc --arg c "$line" '{
              messages: [{role:"user", content:$c}],
              max_tokens: 512
            }')" | jq -r '.choices[0].message.content // (.choices[0].message.tool_calls | tostring)'
    done
  else
    prompt="$*"
    echo "[run_hermes] prompt: $prompt"
    curl -s "$BASE_URL/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "$(jq -nc --arg c "$prompt" '{
            messages: [{role:"user", content:$c}],
            max_tokens: 512
          }')" | jq -r '.choices[0].message.content // (.choices[0].message.tool_calls | tostring)'
  fi
fi
