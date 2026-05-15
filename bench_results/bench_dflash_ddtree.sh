#!/usr/bin/env bash
# Bench dflash DDTree spec-decode (target + draft + tree verify) on GB10.
# This is the "production" upstream runtime path.
set -euo pipefail

DFLASH_ROOT=/home/sparkz/lucbox-gb10-parent/dflash
BIN="$DFLASH_ROOT/build/test_dflash"
TARGET=/home/sparkz/rl/.hf_cache/dflash_models/Qwen3.6-27B-Q4_K_M.gguf
DRAFT=/home/sparkz/rl/.hf_cache/dflash_models/dflash-draft-3.6-q8_0.gguf
TMPDIR=$(mktemp -d /tmp/dflash_ddtree.XXXXXX)
RESULTS=/home/sparkz/rl/lucebox-hub/bench_results/bench_dflash_ddtree.json

ALL_IDS="$TMPDIR/all_ids.bin"
HF_HOME=/home/sparkz/rl/.hf_cache /home/sparkz/rl/.venv/bin/python3 - <<PY
import struct, os
from datasets import load_dataset
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B", trust_remote_code=True)
wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = " ".join(x for x in wt["text"] if x.strip())[:50_000]
ids = tok.encode(text, add_special_tokens=False)[:2200]
with open("$ALL_IDS", "wb") as f:
    for i in ids: f.write(struct.pack("<i", int(i)))
print(f"tokenized {len(ids)} tokens")
PY

GEN=128
SHAPES=(16 64 256 1024 2048)
echo '{"runs":[' > "$RESULTS"
FIRST=1
for S in "${SHAPES[@]}"; do
    PROMPT="$TMPDIR/prompt_${S}.bin"
    OUT="$TMPDIR/out_${S}.bin"
    head -c $((S * 4)) "$ALL_IDS" > "$PROMPT"
    LOG="$TMPDIR/run_${S}.log"
    echo "=== S=$S gen=$GEN ==="
    T0=$(date +%s.%N)
    # --ddtree --ddtree-budget=22 is the default DFlash spec-decode mode.
    # --draft-swa=2048 needed for unsloth Qwen3.6 target per README.
    "$BIN" "$TARGET" "$DRAFT" "$PROMPT" "$GEN" "$OUT" \
        --ddtree --ddtree-budget=22 --draft-swa=2048 \
        > "$LOG" 2>&1 || { echo "FAIL S=$S"; tail -10 "$LOG"; continue; }
    T1=$(date +%s.%N)
    WALL=$(/home/sparkz/rl/.venv/bin/python3 -c "print($T1 - $T0)")

    # test_dflash prints various timing lines; grep tok/s
    DEC=$(grep -E 'tok/s|tokens? in|committed' "$LOG" | tail -20)
    echo "$DEC"

    [ $FIRST -eq 1 ] && FIRST=0 || echo ',' >> "$RESULTS"
    cat >> "$RESULTS" <<EOF
{"S":$S,"gen":$GEN,"wall_s":$WALL,"log_tail":"$(tail -3 "$LOG" | tr '\n' '|' | sed 's/"/\\"/g')"}
EOF
done
echo "]}" >> "$RESULTS"
echo "[bench] wrote $RESULTS  tmpdir=$TMPDIR"
