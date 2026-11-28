#!/bin/bash
# usage: e14watch.sh <out-dir>
# Every 15 s: if any settled call names a model other than deepseek-flash (a failed call names none), record the
# offending ledger lines, write the STOP marker and stop this user's campaign processes on this host.
ROOT="$HOME/e14-10-campaign-v3"; LEDGER="$ROOT/$1/llm_ledger.jsonl"
while true; do
  if [ -f "$LEDGER" ]; then
    bad=$(grep '"op": "settle"' "$LEDGER" | grep -v '"model": "deepseek-flash"' | grep -v '"model": null')
    if [ -n "$bad" ]; then
      echo "$bad" > "$ROOT/logs/model-mismatch.jsonl"
      touch "$ROOT/STOP"
      pkill -u "$(id -u)" -f 'e14-10-campaign-v3/e14jobs.sh'
      pkill -u "$(id -u)" -f 'benchmarks/bench_e14_end_to_end.py run'
      echo "model mismatch: stopped $(date -u +%FT%TZ) host $(hostname)" >> "$ROOT/logs/watch.log"
      exit 1
    fi
  fi
  [ -f "$ROOT/WATCH_OFF" ] && exit 0
  sleep 15
done
