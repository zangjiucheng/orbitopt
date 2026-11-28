#!/bin/bash
# E14-10 campaign v3 launcher on j7zang.
# usage: e14run.sh <code-dir> <out-dir> <label> <bench_e14_end_to_end.py args...>
# Loads the provider key from ~/.llm_env (mode 600, never echoed), exports the frozen provider, pins threads.
set -u
CODE="$1"; OUT="$2"; LABEL="$3"; shift 3
ROOT="$HOME/e14-10-campaign-v3"
mkdir -p "$ROOT/logs"
if [ -f "$ROOT/STOP" ]; then
  echo "STOP marker present, not starting $(date -u +%FT%TZ)" >> "$ROOT/logs/$LABEL.log"; exit 4
fi
if [ ! -f "$HOME/.llm_env" ]; then
  echo "no ~/.llm_env: refusing to run (a provider shard would be written not_run) $(date -u +%FT%TZ)" \
    >> "$ROOT/logs/$LABEL.log"
  exit 5
fi
set -a; . "$HOME/.llm_env"; set +a                 # the key, never echoed
if [ -z "${LLM_API_KEY:-}" ]; then echo "LLM_API_KEY empty: refusing to run" >> "$ROOT/logs/$LABEL.log"; exit 5; fi
export LLM_KIND=deepseek LLM_MODEL=deepseek-flash LLM_BASE_URL=https://api.deepseek.com
export LLM_CAMPAIGN_MAX_CALLS=14210 LLM_CAMPAIGN_MAX_USD=5
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0
cd "$ROOT/$CODE" || exit 1
{
  echo "start $(date -u +%FT%TZ) host $(hostname) pid $$ code $CODE args $*"
  PYTHONPATH=src nice -n 10 "$HOME/e11env/bin/python" benchmarks/bench_e14_end_to_end.py "$@" --out "$ROOT/$OUT"
  rc=$?
  echo "exit $rc $(date -u +%FT%TZ)"
  exit $rc
} >> "$ROOT/logs/$LABEL.log" 2>&1
