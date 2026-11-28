"""Arms C, D and E-LLM end to end through the campaign CLI against a local DeepSeek look-alike: no network.

What the provider arms rest on: every call is metered against the spec's per-cell caps and recorded in a cassette,
the row names the model the server says answered and prices its usage, a replay from the cassette reproduces the
row, a rerun never repeats a recorded call, and the campaign-wide hard stop holds across processes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fake_deepseek import FakeDeepSeek

BENCH = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCH))

import bench_e14_end_to_end as E2E  # noqa: E402
import bench_e14_formulation as FM  # noqa: E402
import e14_llm as L  # noqa: E402

from orbitopt.agent.providers import ChatRequest, Message, MockProvider, Usage  # noqa: E402

KEY = "sk-test-arms-not-a-key"
INSTANCE, SEED = "lp-diet-d2", 1000
COMPILE_REPLY = next(c["replies"][0] for c in FM.load_cases() if c["id"] == "FC-016")


def answer(body: dict) -> str:
    """The fake model: the FC-016 compile for the compiler, one solve then stop for the planner, (3, 1) for C."""
    system = body["messages"][0]["content"]
    if system.startswith("You turn a natural-language"):
        return COMPILE_REPLY
    if system.startswith("You plan the next step"):
        state = json.loads(body["messages"][-1]["content"])
        if any(h.get("kind") == "solve" for h in state["history"]):
            return json.dumps({"action": "stop", "rationale": "done"})
        return json.dumps({"action": "solve", "solver_id": state["solvers"][0]["solver_id"], "rationale": "go"})
    return json.dumps({"points": [[3.0, 1.0]], "stop": True})


@pytest.fixture
def deepseek(monkeypatch):
    fake = FakeDeepSeek(answer, KEY)
    fake.models["deepseek-flash"] = "deepseek-v4-pro"  # the server, not the request, names who answered
    monkeypatch.setenv("NO_PROXY", "*")
    for name, value in {"LLM_KIND": "deepseek", "LLM_MODEL": "deepseek-flash", "LLM_BASE_URL": fake.base_url,
                        "LLM_API_KEY": KEY}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv(L.ENV_MAX_CALLS, raising=False)
    monkeypatch.delenv(L.ENV_MAX_USD, raising=False)
    yield fake
    fake.close()


def run(out: Path, arm: str, *extra: str) -> int:
    return E2E.main(["run", "--all", "--split", "tuning", "--arm", arm, "--instances", INSTANCE, "--seeds", str(SEED),
                     "--out", str(out), "--freeze-path", str(out / "no-freeze.json"), *extra])


def rows(out: Path, arm: str) -> list[dict]:
    return json.loads((out / "shards" / f"cells.tuning.{arm}.linear_program.json").read_text())["rows"]


@pytest.mark.parametrize("arm, calls", [("C", 1), ("D", 1), ("E-LLM", 3)])
def test_a_live_cell_records_meters_prices_and_replays(deepseek, tmp_path, arm, calls):
    assert run(tmp_path, arm) == 0
    [row] = rows(tmp_path, arm)
    assert row["feasible"] and row["error"] is None, row  # D solves with arm A's DE: feasible, maybe not optimal
    assert row["llm_calls"] == calls == len(deepseek.requests)
    assert row["llm_models"] == ["deepseek-v4-pro"]  # the model that answered, priced as such
    assert row["tokens_in"] > 0 and row["tokens_out"] > 0 and row["cost_priced"]
    assert row["cost_usd"] == pytest.approx(sum(
        L.cost_usd(Usage(u["prompt_tokens"], u["completion_tokens"], u["total"], None, u["cache_hit_tokens"],
                         u["cache_miss_tokens"]), "deepseek-v4-pro")
        for u in (json.loads(line)["response"]["usage"]
                  for line in next(tmp_path.glob(f"runs/**/{arm}/**/seed-{SEED}.cassette.jsonl"))
                  .read_text().splitlines()[1:])))
    assert row["replay"]["equal"] is True, row["replay"]
    assert row["transcript_digest"].startswith("sha256:")
    ledger = L.Ledger(tmp_path / "llm_ledger.jsonl", max_calls=1, max_usd=1.0, model="deepseek-flash").totals()
    assert ledger["calls"] == calls and ledger["open"] == 0 and ledger["usd"] == pytest.approx(row["cost_usd"])
    sent = deepseek.requests[0]["body"]
    assert "seed" not in sent and sent["thinking"] == {"type": "disabled"} and sent["temperature"] == 0.0
    assert KEY not in json.dumps([r["body"] for r in deepseek.requests])
    for path in tmp_path.rglob("*"):
        assert path.is_dir() or KEY.encode() not in path.read_bytes()


def test_a_rerun_serves_every_recorded_call_and_reproduces_the_row(deepseek, tmp_path):
    assert run(tmp_path, "E-LLM") == 0
    first = rows(tmp_path, "E-LLM")[0]
    sent = len(deepseek.requests)
    next((tmp_path / "shards").glob("cells.tuning.E-LLM.*.json")).unlink()
    assert run(tmp_path, "E-LLM") == 0
    assert len(deepseek.requests) == sent  # nothing new reached the server
    assert E2E.reexecution_digest(rows(tmp_path, "E-LLM")[0]) == E2E.reexecution_digest(first)


def test_the_hard_stop_leaves_the_shard_unwritten_and_a_rerun_resumes_without_repeating(deepseek, tmp_path, capsys):
    assert run(tmp_path, "E-LLM", "--max-llm-calls", "2") == 3
    assert "STOPPED" in capsys.readouterr().out
    assert not list((tmp_path / "shards").glob("cells.tuning.E-LLM.*.json"))
    assert len(deepseek.requests) == 2
    assert run(tmp_path, "E-LLM", "--max-llm-calls", "10") == 0
    assert len(deepseek.requests) == 3  # the two recorded calls were served from the cassette
    assert rows(tmp_path, "E-LLM")[0]["status"] == "success"


def test_the_spend_cap_is_checked_before_the_call(deepseek, tmp_path):
    assert run(tmp_path, "D", "--max-llm-usd", "0.000001") == 3
    assert deepseek.requests == []


def test_an_account_refusal_stops_the_campaign(deepseek, tmp_path, capsys):
    deepseek.fail(402)
    assert run(tmp_path, "D") == 3
    assert "refused the account" in capsys.readouterr().out
    assert not list((tmp_path / "shards").glob("*.json"))


def test_without_a_provider_the_arms_are_not_run(tmp_path, monkeypatch):
    for name in ("LLM_KIND", "LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert run(tmp_path, "C") == 0
    assert [r["not_run"] for r in rows(tmp_path, "C")] == ["no provider"]


def test_narrowing_the_plan_is_refused_outside_the_tuning_split(capsys):
    with pytest.raises(SystemExit):
        E2E.main(["run", "--all", "--split", "scoring", "--instances", INSTANCE])
    assert "tuning split" in capsys.readouterr().err


# ------------------------------------------------------------------------------------------- per-cell caps
def cell(script, cassette: Path, max_calls: int, max_tokens: int = 200_000):
    return L.LLMContext(MockProvider(script), None).cell(cassette, max_calls, max_tokens)


def test_arm_d_stops_at_its_three_calls(tmp_path):
    """Call 1 is invalid JSON and call 2 its schema repair, a draft the compiler rejects; call 3 is the compiler's
    repair round, rejected again; the compiler's next round would be call 4."""
    from e14_common import P
    bad = json.loads(COMPILE_REPLY)
    bad["problem"]["objectives"][0]["expression"] = "undeclared_name * 2"
    provider = cell(["not json"] + [json.dumps(bad)] * 10, tmp_path / "c.jsonl", E2E.ARM_CALLS["D"])
    row = FM.run_cell_d(P.by_id(INSTANCE), SEED, provider, mode="tuning")
    assert row["llm_calls"] == 3 and row["llm_cut"] == "calls"
    assert row["status"] == "budget_exhausted" and row["stop"] == "budget_exhausted:llm_calls"
    assert row["compile_outcome"] == "budget_exhausted" and row["success"] is False


def test_arm_c_bought_past_the_token_budget_is_a_failure(tmp_path):
    from e14_common import P
    optimum = json.dumps({"points": [[3.0, 1.0]], "stop": True})
    provider = cell([optimum], tmp_path / "c.jsonl", E2E.ARM_CALLS["C"], max_tokens=10)
    row = E2E.run_cell_c(P.by_id(INSTANCE), SEED, provider, mode="tuning")
    assert row["llm_over_budget"] and row["stop"] == "budget_exhausted:llm_tokens"
    assert row["n_evaluations"] == 0  # the optimum in the reply that crossed the budget is never scored
    assert row["success"] is False and row["within_budget"] is False


def test_arm_c_stops_at_twenty_calls(tmp_path):
    from e14_common import P
    provider = cell([json.dumps({"points": [[3.0, 1.0 + k]]}) for k in range(30)], tmp_path / "c.jsonl",
                    E2E.ARM_CALLS["C"])
    row = E2E.run_cell_c(P.by_id(INSTANCE), SEED, provider, mode="tuning")
    assert row["llm_calls"] == 20 and row["stop"] == "budget_exhausted:llm_calls" and row["n_evaluations"] == 20


# ------------------------------------------------------------------------------------------------ the ledger
def request(chars: int = 100, max_tokens: int = 1000) -> ChatRequest:
    return ChatRequest((Message("user", "x" * chars),), max_tokens)


def test_the_ledger_is_shared_by_every_process_on_one_file(tmp_path):
    path = tmp_path / "llm_ledger.jsonl"
    one = L.Ledger(path, max_calls=3, max_usd=1.0, model="deepseek-flash")
    two = L.Ledger(path, max_calls=3, max_usd=1.0, model="deepseek-flash")
    a = one.reserve(request())
    two.reserve(request())  # a reservation another process never settles stays counted
    one.settle(a)
    assert two.totals()["calls"] == 2
    one.reserve(request())
    with pytest.raises(L.CampaignHardStop, match="3 calls"):
        two.reserve(request())


def test_the_reservation_over_states_and_the_settlement_prices_the_usage(tmp_path):
    ledger = L.Ledger(tmp_path / "l.jsonl", max_calls=10, max_usd=1.0, model="deepseek-flash")
    big = request(chars=40_000, max_tokens=4096)
    assert ledger.worst_case_usd(big) == pytest.approx((20_000 * 0.30 + 4096 * 1.20) / 1e6, rel=0.01)
    rid = ledger.reserve(big)
    from orbitopt.agent.providers import ProviderResponse
    usage = Usage(10_000, 500, 10_500, None, 8_000, 2_000)
    ledger.settle(rid, ProviderResponse("{}", None, (), usage, "deepseek-flash", "stop", 0.1, "id"))
    assert ledger.totals()["usd"] == pytest.approx((8_000 * 0.006 + 2_000 * 0.30 + 500 * 1.20) / 1e6)
    with pytest.raises(ValueError, match="no pinned price"):
        L.Ledger(tmp_path / "x.jsonl", max_calls=1, max_usd=1.0, model="some-model")


def test_limits_come_from_the_argument_then_the_environment_then_the_default():
    assert L.limits_from(None, None, {}) == (L.DEFAULT_MAX_CALLS, L.DEFAULT_MAX_USD)
    assert L.limits_from(None, None, {L.ENV_MAX_CALLS: "7", L.ENV_MAX_USD: "2.5"}) == (7, 2.5)
    assert L.limits_from(5, 1.0, {L.ENV_MAX_CALLS: "7"}) == (5, 1.0)


def test_the_formulation_suite_runs_live_under_arm_d_caps_and_replays(deepseek, tmp_path, monkeypatch):
    cases = [c for c in FM.load_cases() if c["id"] in ("FC-016", "FC-002")]
    monkeypatch.setattr(FM, "load_cases", lambda: cases)
    llm = L.context_from_env(tmp_path, max_calls=10, max_usd=1.0)
    doc = FM.run_suite(llm, [0], "tuning", tmp_path / "runs")
    by_case = {r["case_id"]: r for r in doc["rows"]}
    assert by_case["FC-016"]["correct"] and by_case["FC-016"]["llm_models"] == ["deepseek-v4-pro"]
    assert by_case["FC-016"]["variable_matching"] == "case aliases"
    assert not by_case["FC-002"]["correct"]  # the fake compiles a request it should have questioned
    assert all(r["replay"]["equal"] and r["calls"] == 1 and r["cost_usd"] > 0 for r in doc["rows"])
    assert doc["ir_schema_validity"] == 1.0 and len(deepseek.requests) == 2
    assert (tmp_path / "runs" / "FC-016" / "seed-0.cassette.jsonl").exists()


def test_a_compiled_problem_is_matched_by_declared_order_when_its_names_are_its_own():
    from orbitopt.optimization import parse_problem
    case = next(c for c in FM.load_cases() if c["id"] == "FC-016")
    reference = parse_problem(case["reference_ir"])
    renamed = json.loads(json.dumps(case["reference_ir"]).replace('"x0"', '"a"').replace('"x1"', '"b"')
                         .replace("x0", "a").replace("x1", "b"))
    aliases, how = FM.variable_matching(parse_problem(renamed), reference, case["aliases"])
    assert how == "declared order" and aliases == {"a": "x0", "b": "x1"}


def test_a_scored_provider_shard_refuses_a_provider_that_is_not_the_frozen_one(deepseek, tmp_path):
    import e14_common as C
    freeze = tmp_path / "freeze.json"
    assert E2E.main(["freeze", "--freeze-path", str(freeze)]) == 0
    shard = next(s for s in E2E.plan(["lp-wyndor-d2"], [0]) if s["arm"] == "D" and s["split"] == "scoring")
    llm = L.context_from_env(tmp_path)  # deepseek-flash, but at the local fake's URL
    with pytest.raises(C.FreezeError, match="base_url"):
        E2E.run_shard(shard, tmp_path, tmp_path / "runs", "scored", freeze, ["test"], llm)
    assert deepseek.requests == []
    C.check_provider({"kind": "deepseek", "model": "deepseek-flash"})  # the frozen provider at its default URL


def test_the_protocol_pins_the_caps_the_runner_enforces_and_the_price_it_charges():
    import e14_common as C
    assert C.LLM_PROTOCOL["per_cell"]["calls"] == E2E.ARM_CALLS == {"C": 20, "D": 3, "E-LLM": 20}
    assert C.LLM_PROTOCOL["price_table"]["usd_per_mtok"] is L.PRICES_USD_PER_MTOK
    assert C.PROTOCOL["llm_provider_env"][:4] == ["LLM_KIND", "LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY"]


def test_the_committed_v3_freeze_is_sealed_and_holds_this_protocol():
    """The protocol may not move after the freeze; the code digest is checked by every scored shard instead."""
    import e14_common as C
    frozen = C.load_freeze(C.FREEZE_PATH)
    assert C.FREEZE_PATH.parent.name == "e14-v3" and frozen["protocol"]["version"] == "e14-10-campaign/3"
    assert frozen["protocol_digest"] == C.protocol_digest() and frozen["spec_digest"] == C.SPEC.spec_digest()
