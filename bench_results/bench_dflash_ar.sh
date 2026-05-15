#!/usr/bin/env bash
# Bench dflash test_generate (AR-only, target-only) on Qwen3.6-27B Q4_K_M GGUF
# across many S values. Records prefill wall time + decode tok/s.
set -euo pipefail

DFLASH_ROOT=/home/sparkz/lucbox-gb10-parent/dflash
BIN="$DFLASH_ROOT/build/test_generate"
TARGET=/home/sparkz/rl/.hf_cache/dflash_models/Qwen3.6-27B-Q4_K_M.gguf
TMPDIR=$(mktemp -d /tmp/dflash_bench.XXXXXX)
RESULTS=/home/sparkz/rl/lucebox-hub/bench_results/bench_dflash_ar.json

ALL_IDS="$TMPDIR/all_ids.bin"
# Tokenize 2200 tokens of wikitext into one int32 bin
HF_HOME=/home/sparkz/rl/.hf_cache /home/sparkz/rl/.venv/bin/python3 - <<PY
import struct, os
from datasets import load_dataset
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B", trust_remote_code=True)
wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = " ".join(x for x in wt["text"] if x.strip())[:50_000]
ids = tok.encode(text, add_special_tokens=False)[:2200]
print(f"tokenized {len(ids)} tokens")
with open("$ALL_IDS", "wb") as f:
    for i in ids:
        f.write(struct.pack("<i", int(i)))
PY

N_ALL=$(/home/sparkz/rl/.venv/bin/python3 -c "import os; print(os.path.getsize('$ALL_IDS')//4)")
echo "[bench] total tokens: $N_ALL"

GEN=128
SHAPES=(16 64 256 1024 2048)
echo '{"runs":[' > "$RESULTS"
FIRST=1
for S in "${SHAPES[@]}"; do
    if [ "$S" -gt "$N_ALL" ]; then echo "skip S=$S"; continue; fi
    PROMPT="$TMPDIR/prompt_${S}.bin"
    OUT="$TMPDIR/out_${S}.bin"
    head -c $((S * 4)) "$ALL_IDS" > "$PROMPT"

    LOG="$TMPDIR/run_${S}.log"
    echo "=== S=$S gen=$GEN ==="
    T0=$(date +%s.%N)
    "$BIN" "$TARGET" "$PROMPT" "$GEN" "$OUT" > "$LOG" 2>&1 || { echo "FAIL S=$S"; tail -5 "$LOG"; continue; }
    T1=$(date +%s.%N)
    WALL=$(/home/sparkz/rl/.venv/bin/python3 -c "print($T1 - $T0)")

    DEC_LINE=$(grep '\[gen\]' "$LOG" | head -1)
    DEC_TPS=$(echo "$DEC_LINE" | sed -nE 's/.*-> +([0-9.]+) tok\/s.*/\1/p')
    DEC_SEC=$(echo "$DEC_LINE" | sed -nE 's/.* in +([0-9.]+) s.*/\1/p')
    PP_EST=$(/home/sparkz/rl/.venv/bin/python3 -c "print($WALL - ${DEC_SEC:-0})")

    [ $FIRST -eq 1 ] && FIRST=0 || echo ',' >> "$RESULTS"
    cat >> "$RESULTS" <<EOF
{"S":$S,"gen":$GEN,"wall_s":$WALL,"decode_s":${DEC_SEC:-0},"decode_tok_s":${DEC_TPS:-0},"prefill_plus_init_s":$PP_EST}
EOF
    echo "  wall=${WALL}s  decode=${DEC_SEC}s (${DEC_TPS} tok/s)  pp+init=${PP_EST}s"
done
echo "]}" >> "$RESULTS"
echo "[bench] wrote $RESULTS"
echo "[bench] tmpdir kept: $TMPDIR"
