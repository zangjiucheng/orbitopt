"""Project the scored campaign's spend from the pilot cassettes. usage: python3 e14project.py <out-dir> <code-dir>

Per provider arm, the list-price cost of every finished pilot cell (every cassette but the newest of its shard, which
may be a cell still in flight) from its recorded usage. Remaining cells are charged:
  D, E-LLM: the highest per-instance mean seen (prompts are within 4 % in size across instances);
  C: by dimension, cells of d <= 5 at the mean of finished d <= 5 cells, cells of d > 5 at the mean of finished
     d > 5 cells (the 20-D cells), and both at least the overall mean; C is the arm that scales with the problem.
Prints counts and dollars only."""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PRICE = {"cache_hit": 0.006, "cache_miss": 0.30, "output": 1.20}
out, code = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(code / "benchmarks"))
sys.path.insert(0, str(code / "src"))
import e14_common as C  # noqa: E402

dims = {i.id: i.dimension for i in C.P.INSTANCES if i.split == "scoring"}


def cost(cas: Path) -> tuple[float, int]:
    usd, calls = 0.0, 0
    for line in cas.read_text().splitlines()[1:]:
        u = json.loads(line)["response"].get("usage") or {}
        usd += ((u.get("cache_hit_tokens") or 0) * PRICE["cache_hit"] + (u.get("cache_miss_tokens") or 0)
                * PRICE["cache_miss"] + (u.get("completion_tokens") or 0) * PRICE["output"]) / 1e6
        calls += 1
    return usd, calls


done = defaultdict(dict)  # arm -> {(inst, seed): (usd, calls)}
for shard in sorted(out.glob("runs/cells.scoring.*")):
    arm = shard.name.split(".")[2]
    cas = sorted(shard.glob("*/*/seed-*.cassette.jsonl"), key=lambda p: p.stat().st_mtime)
    written = (out / "shards" / f"{shard.name}.json").exists()
    keep = cas if written else cas[:-1]
    for c in keep:
        done[arm][(c.parts[-2], int(c.name.split(".")[0].removeprefix("seed-")))] = cost(c)
ledger = {"usd": 0.0}
open_ = {}
for line in (out / "llm_ledger.jsonl").read_text().splitlines():
    e = json.loads(line)
    if e["op"] == "reserve":
        open_[e["id"]] = e["usd"]
    else:
        open_.pop(e["id"], None)
        ledger["usd"] += e["usd"]
spent = ledger["usd"] + sum(open_.values())
print(f"ledger now (settled + open reservations): ${spent:.4f}")
total_remaining = 0.0
for arm in ("C", "D", "E-LLM"):
    cells = done.get(arm, {})
    if not cells:
        print(f"{arm}: no finished pilot cell; cannot project")
        total_remaining = float("inf")
        continue
    usd = [v[0] for v in cells.values()]
    calls = [v[1] for v in cells.values()]
    by_inst = defaultdict(list)
    for (inst, _), v in cells.items():
        by_inst[inst].append(v[0])
    print(f"{arm}: finished pilot cells {len(cells)}, calls {sum(calls)} (max {max(calls)} per cell), "
          f"usd {sum(usd):.4f}, mean/cell {statistics.mean(usd):.5f}, max/cell {max(usd):.5f}")
    for inst, v in sorted(by_inst.items()):
        print(f"   {inst} (d={dims[inst]}): {len(v)} cells, mean ${statistics.mean(v):.5f}, max ${max(v):.5f}")
    remaining = [(i, s) for i in dims for s in C.M.SCORING_SEEDS if (i, s) not in cells]
    if arm == "C":
        lo = [v[0] for (i, _), v in cells.items() if dims[i] <= 5]
        hi = [v[0] for (i, _), v in cells.items() if dims[i] > 5]
        m_all = statistics.mean(usd)
        m_lo = max(statistics.mean(lo), m_all) if lo else m_all
        m_hi = max(statistics.mean(hi), m_all) if hi else float("nan")
        proj = sum(m_lo if dims[i] <= 5 else m_hi for i, _ in remaining)
        print(f"   C rule: d<=5 ${m_lo:.5f}/cell, d>5 ${m_hi:.5f}/cell ({len(hi)} finished d>5 cells)")
    else:
        proj = len(remaining) * max(statistics.mean(v) for v in by_inst.values())
    print(f"   remaining cells {len(remaining)}, projected ${proj:.4f}")
    total_remaining += proj
print(f"projected campaign total: ${spent + total_remaining:.4f} (threshold to continue: $4.00, hard stop $5.00)")
