#!/bin/bash
# usage: e14jobs.sh <code-dir> <out-dir> <label> <max-usd> <extra run args> <family> [<family> ...]
# Each family's shards in turn (every arm in plan order, then its faults), one process at a time.
# A run that dies without finishing (killed, exit not 0/3/4/5) is retried: shards already written are skipped and
# every call a cassette holds is served from it. Exit 3 (hard stop), 4 (STOP marker) or 5 (no key) ends the job.
CODE="$1"; OUT="$2"; LABEL="$3"; USD="$4"; EXTRA="$5"; shift 5
ROOT="$HOME/e14-10-campaign-v3"
for FAMILY in "$@"; do
  for attempt in 1 2 3 4; do
    # shellcheck disable=SC2086
    "$ROOT/e14run.sh" "$CODE" "$OUT" "$LABEL" run --all --family "$FAMILY" $EXTRA \
      --max-llm-usd "$USD" --max-llm-calls 14210
    rc=$?
    [ $rc -eq 0 ] && break
    if [ $rc -eq 3 ] || [ $rc -eq 4 ] || [ $rc -eq 5 ]; then
      echo "jobs stopped rc $rc family $FAMILY $(date -u +%FT%TZ)" >> "$ROOT/logs/$LABEL.log"; exit $rc
    fi
    echo "retry family $FAMILY after rc $rc (attempt $attempt) $(date -u +%FT%TZ)" >> "$ROOT/logs/$LABEL.log"
    sleep 30
  done
done
echo "jobs done $(date -u +%FT%TZ)" >> "$ROOT/logs/$LABEL.log"
