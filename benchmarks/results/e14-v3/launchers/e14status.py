"""Campaign v3 status: ledger totals and cassette usage, by arm. usage: python3 e14status.py <out-dir>

Prints counts only: no request or reply text."""
import json
import sys
from collections import defaultdict
from pathlib import Path

PRICE = {"cache_hit": 0.006, "cache_miss": 0.30, "output": 1.20}
out = Path(sys.argv[1])
led = out / "llm_ledger.jsonl"
open_res, settled = {}, defaultdict(float)
models = defaultdict(int)
for line in led.read_text().splitlines() if led.exists() else []:
    e = json.loads(line)
    if e["op"] == "reserve":
        open_res[e["id"]] = e["usd"]
    else:
        open_res.pop(e["id"], None)
        for k in ("calls", "usd", "attempts", "tokens"):
            settled[k] += e[k]
        models[str(e.get("model"))] += 1
print(f"ledger settled: calls {int(settled['calls'])} attempts {int(settled['attempts'])}"
      f" tokens {int(settled['tokens'])} usd {settled['usd']:.4f}")
print(f"ledger open: {len(open_res)} reserved usd {sum(open_res.values()):.4f};"
      f" total usd {settled['usd'] + sum(open_res.values()):.4f}")
print(f"ledger models: {dict(models)}")
agg = defaultdict(lambda: defaultdict(float))
for cas in out.glob("runs/**/*.cassette.jsonl"):
    parts = cas.relative_to(out / "runs").parts
    shard = parts[0]
    arm = "formulation" if shard.startswith("formulation") else shard.split(".")[2]
    a = agg[arm]
    a["cells"] += 1
    for line in cas.read_text().splitlines()[1:]:
        r = json.loads(line)["response"]
        u = r.get("usage") or {}
        a["calls"] += 1
        hit, miss = u.get("cache_hit_tokens") or 0, u.get("cache_miss_tokens") or 0
        a["hit"] += hit
        a["miss"] += miss
        a["out"] += u.get("completion_tokens") or 0
        a["usd"] += (hit * PRICE["cache_hit"] + miss * PRICE["cache_miss"]
                     + (u.get("completion_tokens") or 0) * PRICE["output"]) / 1e6
        a["latency"] += r.get("latency_s") or 0
        if r.get("model") != "deepseek-flash":
            a["other_model"] += 1
for arm, a in sorted(agg.items()):
    tin = a["hit"] + a["miss"]
    print(f"{arm:11s} cas {int(a['cells'])} calls {int(a['calls'])} in {int(tin)} hit {a['hit'] / tin if tin else 0:.2f}"
          f" out {int(a['out'])}")
    print(f"{'':11s} usd {a['usd']:.4f} per-cas {a['usd'] / a['cells'] if a['cells'] else 0:.5f}"
          f" lat_s {a['latency']:.0f} other_model {int(a['other_model'])}")
shards = sorted(p.stem for p in (out / "shards").glob("*.json")) if (out / "shards").exists() else []
print(f"shards written {len(shards)}")
