#!/usr/bin/env bash
# Boot the megakernel OpenAI server and drive it with the Hermes agent.
#
# Usage:
#   ./run_hermes.sh                              # one-shot demo task
#   ./run_hermes.sh "What is 23 * 41?"           # one-shot task
#   ./run_hermes.sh --interactive                # interactive REPL
#   BACKEND=nvfp4 ./run_hermes.sh "..."          # NVFP4 decode path
#   KEEP_SERVER=1 ./run_hermes.sh                # don't kill server after run
#   ATTACH=http://x:8765 ./run_hermes.sh "..."   # use an already-running server
#
# Env:
#   VENV       (default: /home/sparkz/rl/.venv)
#   HF_HOME    (default: /home/sparkz/rl/.hf_cache)
#   BACKEND    (default: bf16; alternatives: nvfp4, auto)
#   PORT       (default: 8765)
#   MODEL      (default: Qwen/Qwen3.5-0.8B)

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${VENV:=/home/sparkz/rl/.venv}"
: "${HF_HOME:=/home/sparkz/rl/.hf_cache}"
: "${BACKEND:=bf16}"
: "${PORT:=8765}"
: "${MODEL:=Qwen/Qwen3.5-0.8B}"

export HF_HOME
# shellcheck source=/dev/null
source "${VENV}/bin/activate"
export PYTHONPATH="${HERE}:${PYTHONPATH:-}"

cleanup_pid=""
trap '[[ -n "${cleanup_pid}" ]] && kill "${cleanup_pid}" 2>/dev/null || true' EXIT

if [[ -n "${ATTACH:-}" ]]; then
  BASE="${ATTACH}"
  echo "[run_hermes] attaching to ${BASE}"
else
  BASE="http://127.0.0.1:${PORT}"
  LOG="/tmp/megakernel_server.${PORT}.log"
  echo "[run_hermes] starting server backend=${BACKEND} port=${PORT}"
  python -u -m serve.openai_server \
      --backend "${BACKEND}" --port "${PORT}" --model-name "${MODEL}" \
      > "${LOG}" 2>&1 &
  cleanup_pid="$!"
  if [[ "${KEEP_SERVER:-0}" == "1" ]]; then
    trap - EXIT
    echo "[run_hermes] KEEP_SERVER=1 -> server pid=${cleanup_pid} log=${LOG}"
  fi

  echo -n "[run_hermes] waiting for /healthz "
  for i in $(seq 1 120); do
    if curl -sf "${BASE}/healthz" >/dev/null 2>&1; then
      echo " ready (${i}s)"
      break
    fi
    sleep 1
    if (( i == 120 )); then
      echo " TIMEOUT"
      echo "--- server log tail ---"
      tail -40 "${LOG}"
      exit 1
    fi
  done
  echo "[run_hermes] log: ${LOG}"
fi

# Default demo task exercises tool dispatch.
if [[ $# -eq 0 ]]; then
  set -- "Use the calculator tool to compute 17 * 23, then tell me the answer in one sentence."
fi

# Pass --interactive through as a sentinel.
if [[ "$1" == "--interactive" ]]; then
  shift
  exec python -m serve.hermes_agent --base "${BASE}" "$@"
else
  python -m serve.hermes_agent --base "${BASE}" --task "$1" "${@:2}"
fi
