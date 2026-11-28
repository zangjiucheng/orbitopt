"""Calls, tokens and USD of the scored provider arms, from the real prompts and a tuning dry run's cassettes.

    PYTHONPATH=src python benchmarks/e14_llm_estimate.py --dry-run DIR [--json OUT]

For every scoring instance and formulation case it builds the exact first request each arm sends (compiler,
planner, proposer) and measures it in characters on the wire, the way DeepSeekProvider sends it. The dry run's
cassettes turn characters into tokens (prompt tokens per character, per kind of call) and give the measured reply
lengths, cache-hit shares, calls per compile, planner calls per E-LLM cell and calls per arm-C cell. Three
scenarios follow:

- low: one call per compile, the fewest planner and proposer calls the dry run saw, the measured cache-hit share;
- expected: the dry run's means;
- high: as expected, but every arm-C and E-LLM cell makes every call its cap allows (C 20, the planner 19 or
  until the runtime has no evaluations left), each cell stopped at 200,000 tokens;
- worst: every cap reached (D 3 calls, C and E-LLM 20, each cell stopped at 200,000 tokens), every reply at its
  max_tokens, every prompt token a cache miss.

Prices are the protocol's peak list prices; off-peak is half. Not part of the code digest: it reads, never runs, a
provider.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e14_common as C  # noqa: E402
import e14_llm as L  # noqa: E402

PRICE = L.PRICES_USD_PER_MTOK["deepseek-flash"]
COMPILE_MAX, PLANNER_MAX = 4096, 512
CELL_TOKENS = C.A.BUDGET.llm_tokens_max
SEEDS = len(C.M.SCORING_SEEDS)


def wire_chars(request) -> int:
    """Characters of every message the DeepSeek preset puts on the wire for ``request``."""
    from orbitopt.agent.providers import DeepSeekProvider
    body = DeepSeekProvider("deepseek-flash", "http://unused").request_body(request)
    return sum(len(m["content"] or "") for m in body["messages"])


def first_compile_request(text: str, problem_id: str):
    import bench_e14_formulation as FM
    from orbitopt.agent.providers import MockProvider
    mock = MockProvider(["not json"] * 2)
    FM.compile_statement(text, mock, 0, problem_id)
    return mock.requests[0]


def planner_requests(inst, solves: int):
    """The planner's requests on the reference problem when it asks for a solve every time: the prompt of each
    iteration, up to the call cap or until the runtime has no evaluations left for another decision."""
    from orbitopt.agent.providers import MockProvider
    from orbitopt.agent.runtime import LLMPlanner, solve
    from orbitopt.optimization import Budget

    def reply(request):
        state = json.loads(request.messages[-1].content)
        done = sum(h.get("kind") == "solve" for h in state["history"])
        if done >= solves or not state["solvers"]:
            return json.dumps({"action": "stop", "rationale": "estimate"})
        return json.dumps({"action": "solve", "solver_id": state["solvers"][done % len(state["solvers"])]
                           ["solver_id"], "rationale": "estimate"})

    mock = MockProvider(reply)
    budget = Budget(max_evaluations=C.A.BUDGET.evaluations(inst.dimension), max_wall_s=600, max_tokens=10**9,
                    max_iterations=solves + 1)
    solve(C.instance_ir(inst), planner=LLMPlanner(mock), budget=budget, tools=C.tools_for(inst), seed=0)
    return mock.requests


def proposer_first_request(inst):
    import bench_e14_end_to_end as E2E
    from orbitopt.agent.providers import ChatRequest, Message
    budget = C.A.BUDGET.evaluations(inst.dimension)
    msgs = (Message("system", E2E.PROPOSER_SYSTEM),
            Message("user", f"{inst.statement}\n\nBudget: {E2E.ARM_CALLS['C']} replies and {budget} evaluations."))
    return ChatRequest(msgs, E2E.PROPOSER_MAX_TOKENS, 0.0, 0, E2E.PROPOSER_SCHEMA)


def kind_of(request: dict) -> str:
    system = request["messages"][0]["content"] or ""
    return ("compile" if system.startswith("You turn a natural-language") else
            "planner" if system.startswith("You plan the next step") else "proposer")


def calibrate(dry: Path) -> dict:
    """Per kind of call: prompt tokens per wire character, reply tokens, cache-hit share; per cell: calls."""
    from orbitopt.agent.providers import ChatRequest, Message, ToolSpec
    calls: dict[str, list] = {"compile": [], "planner": [], "proposer": []}
    cells = []
    for path in sorted(dry.rglob("*.cassette.jsonl")):
        lines = [json.loads(x) for x in path.read_text().splitlines()[1:]]
        kinds = []
        for entry in lines:
            if "response" not in entry:
                continue
            r = entry["request"]
            req = ChatRequest(tuple(Message(m["role"], m["content"]) for m in r["messages"]), r["max_tokens"],
                              r["temperature"], r["seed"], r["response_schema"],
                              tuple(ToolSpec(**t) for t in r["tools"]))
            u = entry["response"]["usage"]
            k = kind_of(r)
            kinds.append(k)
            calls[k].append({"chars": wire_chars(req), "prompt": u["prompt_tokens"], "out": u["completion_tokens"],
                             "hit": u.get("cache_hit_tokens") or 0, "model": entry["response"]["model"]})
        arm = path.parts[-3]  # runs/<shard>/<arm>/<instance>/seed-<k>.cassette.jsonl
        cells.append({"arm": arm, "cassette": str(path.relative_to(dry)), "calls": len(lines),
                      "kinds": {k: kinds.count(k) for k in set(kinds)}})
    out = {}
    for k, xs in calls.items():
        if xs:
            out[k] = {"n": len(xs), "tokens_per_char": sum(x["prompt"] for x in xs) / sum(x["chars"] for x in xs),
                      "out_mean": statistics.fmean(x["out"] for x in xs), "out_max": max(x["out"] for x in xs),
                      "hit_share": sum(x["hit"] for x in xs) / sum(x["prompt"] for x in xs),
                      "models": sorted({x["model"] for x in xs})}
    out["cells"] = cells
    return out


def usd(prompt: float, hit_share: float, out: float) -> float:
    return (prompt * hit_share * PRICE["cache_hit"] + prompt * (1 - hit_share) * PRICE["cache_miss"]
            + out * PRICE["output"]) / 1e6


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", type=Path, required=True, help="the tuning dry run's output directory")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    import bench_e14_formulation as FM
    cal = calibrate(args.dry_run)
    print(json.dumps({k: v for k, v in cal.items() if k != "cells"}, indent=1))
    scoring = C.instances("scoring")
    compile_chars = {i.id: wire_chars(first_compile_request(i.statement, i.id)) for i in scoring}
    case_chars = {c["id"]: wire_chars(first_compile_request(c["prompt"], c["id"])) for c in FM.load_cases()}
    proposer_chars = {i.id: wire_chars(proposer_first_request(i)) for i in scoring}
    planner_worst = {}
    for i in scoring:  # every iteration a solve, up to E-LLM's 20 calls less one compile call
        planner_worst[i.id] = [wire_chars(r) for r in planner_requests(i, C.LLM_PROTOCOL["per_cell"]["calls"]
                                                                       ["E-LLM"] - 1)]

    def tok(kind, chars):
        return chars * cal[kind]["tokens_per_char"]

    e_cells = [c for c in cal["cells"] if c["arm"] == "E-LLM"]
    c_cells = [c for c in cal["cells"] if c["arm"] == "C"]
    d_cells = [c for c in cal["cells"] if c["arm"] in ("D",)]
    compile_calls = [c["kinds"].get("compile", 0) for c in d_cells + e_cells if c["kinds"].get("compile")]
    planner_calls = [c["kinds"].get("planner", 0) for c in e_cells]
    proposer_calls = [c["kinds"].get("proposer", 0) for c in c_cells]
    k_compile = {"low": 1.0, "expected": statistics.fmean(compile_calls) if compile_calls else 1.0}
    k_compile["high"] = k_compile["expected"]
    k_planner = {"low": min(planner_calls, default=1), "expected": statistics.fmean(planner_calls or [2]),
                 "high": C.LLM_PROTOCOL["per_cell"]["calls"]["E-LLM"] - 1}
    k_proposer = {"low": min(proposer_calls, default=1), "expected": statistics.fmean(proposer_calls or [3]),
                  "high": C.LLM_PROTOCOL["per_cell"]["calls"]["C"]}

    def capped(prompts_out):
        """Calls of one cell until the cell's tokens pass 200,000 (the crossing call is the last)."""
        used = p_sum = o_sum = n = 0
        for p, o in prompts_out:
            n += 1
            p_sum, o_sum, used = p_sum + p, o_sum + o, used + p + o
            if used > CELL_TOKENS:
                break
        return n, p_sum, o_sum

    rows = []
    for scen in ("low", "expected", "high"):
        hit = {k: cal[k]["hit_share"] for k in ("compile", "planner", "proposer") if k in cal}
        out = {k: cal[k]["out_mean"] for k in hit}
        total = {"calls": 0.0, "prompt": 0.0, "out": 0.0, "usd": 0.0}
        parts = {}

        def add(name, calls, prompt, outt, share):
            t = parts.setdefault(name, {"calls": 0.0, "prompt": 0.0, "out": 0.0, "usd": 0.0})
            for target in (t, total):
                target["calls"] += calls
                target["prompt"] += prompt
                target["out"] += outt
                target["usd"] += usd(prompt, share, outt)

        kc = k_compile[scen]
        for i in scoring:
            p = tok("compile", compile_chars[i.id])
            # a repair call resends the prompt plus the reply and the repair turn
            add("D", SEEDS * kc, SEEDS * (kc * p + (kc - 1) * out["compile"]), SEEDS * kc * out["compile"],
                hit["compile"])
            add("E-LLM compile", SEEDS * kc, SEEDS * (kc * p + (kc - 1) * out["compile"]),
                SEEDS * kc * out["compile"], hit["compile"])
            # planner prompts as the runtime builds them, one per iteration (the runtime stops early when no
            # evaluations are left for another decision, so "high" can stay below the call cap)
            n = int(round(k_planner[scen]))
            seq = [(tok("planner", c), out.get("planner", 60)) for c in planner_worst[i.id][:n]]
            calls, prompt, outt = capped(seq)
            scale = k_planner[scen] / n if scen != "high" else 1.0
            add("E-LLM planner", SEEDS * calls * scale, SEEDS * prompt * scale, SEEDS * outt * scale,
                hit.get("planner", 0))
            # arm C resends the conversation: each turn adds the reply and its feedback (about 1.5x the reply)
            m = k_proposer[scen]
            p0 = tok("proposer", proposer_chars[i.id])
            per_turn = out.get("proposer", 200) * 2.5
            seq = [(p0 + j * per_turn, out.get("proposer", 200)) for j in range(int(round(m)))]
            calls, prompt, outt = capped(seq)
            scale = m / round(m) if scen != "high" else 1.0
            add("C", SEEDS * calls * scale, SEEDS * prompt * scale, SEEDS * outt * scale, hit.get("proposer", 0))
        for cid, chars in case_chars.items():
            p = tok("compile", chars)
            add("formulation", SEEDS * kc, SEEDS * (kc * p + (kc - 1) * out["compile"]),
                SEEDS * kc * out["compile"], hit["compile"])
        rows.append({"scenario": scen, "total": total, "parts": parts})

    # worst case: every cap and every max_tokens, all prompt tokens at the miss price, 200,000 tokens per cell
    total = {"calls": 0.0, "prompt": 0.0, "out": 0.0, "usd": 0.0}
    parts = {}

    def add_worst(name, calls, prompt, outt):
        t = parts.setdefault(name, {"calls": 0.0, "prompt": 0.0, "out": 0.0, "usd": 0.0})
        for target in (t, total):
            target["calls"] += calls
            target["prompt"] += prompt
            target["out"] += outt
            target["usd"] += usd(prompt, 0.0, outt)

    for i in scoring:
        p = tok("compile", compile_chars[i.id])
        seq = [(p + j * (COMPILE_MAX + 200), COMPILE_MAX) for j in range(3)]
        add_worst("D", *(SEEDS * v for v in capped(seq)))
        pl = [tok("planner", c) for c in planner_worst[i.id]]
        pl = pl + [pl[-1]] * (19 - len(pl))  # the runtime may stop early; worst assumes every call is made
        add_worst("E-LLM", *(SEEDS * v for v in capped(seq + [(x, PLANNER_MAX) for x in pl[:17]])))
        p0 = tok("proposer", proposer_chars[i.id])
        import bench_e14_end_to_end as E2E
        seqc = [(p0 + j * 2.5 * E2E.PROPOSER_MAX_TOKENS, E2E.PROPOSER_MAX_TOKENS) for j in range(20)]
        add_worst("C", *(SEEDS * v for v in capped(seqc)))
    for cid, chars in case_chars.items():
        p = tok("compile", chars)
        add_worst("formulation", *(SEEDS * v for v in capped([(p + j * (COMPILE_MAX + 200), COMPILE_MAX)
                                                               for j in range(3)])))
    rows.append({"scenario": "worst", "total": total, "parts": parts})
    for r in rows:
        t = r["total"]
        print(f"{r['scenario']:9s} calls {t['calls']:9.0f}  prompt {t['prompt'] / 1e6:8.2f} M  out "
              f"{t['out'] / 1e6:7.2f} M  USD {t['usd']:8.2f} (off-peak {t['usd'] / 2:7.2f})")
        for name, p in r["parts"].items():
            print(f"    {name:14s} calls {p['calls']:9.0f}  prompt {p['prompt'] / 1e6:8.2f} M  out "
                  f"{p['out'] / 1e6:7.2f} M  USD {p['usd']:8.2f}")
    if args.json:
        C.write_json_atomic(args.json, C.jsonable({"calibration": cal, "scenarios": rows,
                                                   "prompt_chars": {"compile": compile_chars,
                                                                    "formulation": case_chars,
                                                                    "proposer": proposer_chars,
                                                                    "planner": planner_worst},
                                                   "price": PRICE, "price_source": L.PRICE_SOURCE}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
