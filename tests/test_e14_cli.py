"""E14 command line: the five-command workflow, run end to end through ``orbitopt.cli.main``.

``agent compile`` uses the mock provider with the formulation cases'
recorded replies, so no test needs a model, a network or an API key. Three
catalogue problems go through compile, validate, solve, verify and replay;
the rest checks refusals, the LLM planner, cassettes, tampering and secrets.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from orbitopt.cli import main
from orbitopt.optimization import parse_problem

REPO = Path(__file__).resolve().parents[1]
CASES = {c["id"]: c for c in json.loads((REPO / "tests/data/agent/formulation_cases.json").read_text())["cases"]}
CATALOGUE = {  # formulation case, catalogue class, whether the rule planner can verify it independently
    "FC-001": ("convex_qp", False),  # SLSQP is the only solver for an open-bounded QP: no alternate solver
    "FC-017": ("lp", True),
    "FC-018": ("milp", True),
}


def run(capsys, *argv):
    """``orbitopt <argv>``: the exit code and stdout, parsed as JSON when ``--json`` was passed."""
    with pytest.raises(SystemExit) as stop:
        main([str(a) for a in argv])
    out = capsys.readouterr().out
    return stop.value.code, json.loads(out) if "--json" in argv else out


def case_files(tmp_path, case_id):
    case = CASES[case_id]
    prompt, replies = tmp_path / f"{case_id}.txt", tmp_path / f"{case_id}.replies.json"
    prompt.write_text(case["prompt"], encoding="utf-8")
    replies.write_text(json.dumps(case["replies"]), encoding="utf-8")
    return prompt, replies


@pytest.mark.parametrize("case_id", sorted(CATALOGUE))
def test_five_command_workflow_on_a_catalogue_problem(tmp_path, capsys, case_id):
    problem_class, verifiable = CATALOGUE[case_id]
    prompt, replies = case_files(tmp_path, case_id)
    problem_file, run_dir = tmp_path / "problem.json", tmp_path / "runs/experiment-001"

    code, report = run(capsys, "agent", "compile", "--prompt-file", prompt, "-o", problem_file,
                       "--provider", "mock", "--mock-replies", replies, "--json")
    assert code == 0 and report["accepted"] and report["outcome"] == "compile"
    problem = parse_problem(json.loads(problem_file.read_text()))
    assert report["problem_digest"] == problem.digest()
    assert json.loads((tmp_path / "problem.compile.json").read_text())["problem_digest"] == problem.digest()

    code, summary = run(capsys, "optimization", "validate", problem_file, "--json")
    assert code == 0 and summary["valid"] and summary["digest"] == problem.digest()
    assert summary["structure"]["class"] == problem_class and summary["compatible_solvers"]

    code, solved = run(capsys, "agent", "solve", problem_file, "--max-iterations", 10, "--max-evaluations", 4000,
                       "--seed", 1, "-o", run_dir, "--json")
    assert solved["verified"] is verifiable and code == (0 if verifiable else 1)
    assert set(solved["statuses"]) == {"solver_converged", "constraints_satisfied", "independently_verified",
                                       "optimality_certified"}
    assert solved["statuses"]["constraints_satisfied"] == "passed" and solved["usage"]["tokens"] == 0

    code, graded = run(capsys, "optimization", "verify", run_dir, "--expect-digest", solved["manifest_digest"],
                       "--json")
    assert (code, graded["verified"], graded["integrity"]) == ((0, True, []) if verifiable else (1, False, []))
    assert graded["recorded_status"] == solved["status"]

    code, replayed = run(capsys, "agent", "replay", run_dir, "--expect-digest", solved["manifest_digest"], "--json")
    assert code == 0 and replayed["equal"] and replayed["recorded_digest"] == replayed["replayed_digest"]

    code, text = run(capsys, "optimization", "verify", run_dir)  # the human form names the four statuses
    assert all(name in text for name in solved["statuses"])


def test_a_refused_compile_writes_no_problem_unless_a_draft_is_allowed(tmp_path, capsys):
    prompt, replies = case_files(tmp_path, "FC-002")
    out = tmp_path / "bag.json"
    code, report = run(capsys, "agent", "compile", "--prompt-file", prompt, "-o", out, "--provider", "mock",
                       "--mock-replies", replies, "--json")
    assert code == 1 and not out.exists() and report["outcome"] == "needs_clarification"
    assert report["missing_parameters"] and report["ambiguities"] and report["problem_path"] is None

    prompt, replies = case_files(tmp_path, "FC-032")  # a missing parameter value, with a draft around it
    code, report = run(capsys, "agent", "compile", "--prompt-file", prompt, "-o", out, "--provider", "mock",
                       "--mock-replies", replies, "--allow-draft", "--json")
    draft = json.loads(out.read_text())
    assert code == 1 and report["draft_path"] == str(out) and draft["draft"] is True
    code, summary = run(capsys, "optimization", "validate", out, "--json")
    assert code == 1 and not summary["valid"]  # a draft is never mistaken for a problem


def test_llm_planner_records_a_cassette_and_replays_without_a_model(tmp_path, capsys):
    prompt, replies = case_files(tmp_path, "FC-017")
    problem_file = tmp_path / "lp.json"
    assert run(capsys, "agent", "compile", "--prompt-file", prompt, "-o", problem_file, "--provider", "mock",
               "--mock-replies", replies)[0] == 0
    plan, cassette = tmp_path / "plan.json", tmp_path / "plan.jsonl"
    plan.write_text(json.dumps([{"action": "solve", "solver_id": "scipy.highs", "rationale": "an LP"},
                                {"action": "stop", "rationale": "verified"}]))
    budget = ("--max-iterations", 4, "--max-evaluations", 2000, "--max-tokens", 50_000)
    code, first = run(capsys, "agent", "solve", problem_file, "--planner", "llm", "--provider", "mock",
                      "--mock-replies", plan, "--cassette", cassette, *budget, "-o", tmp_path / "a", "--json")
    assert code == 0 and first["verified"] and first["usage"]["tokens"] > 0 and cassette.exists()
    code, again = run(capsys, "agent", "solve", problem_file, "--planner", "llm", "--provider", "replay",
                      "--cassette", cassette, *budget, "-o", tmp_path / "b", "--json")
    trace = [json.loads((tmp_path / d / "trace.json").read_text())["digest"] for d in ("a", "b")]
    assert code == 0 and again["objective_values"] == first["objective_values"] and trace[0] == trace[1]
    code, replayed = run(capsys, "agent", "replay", tmp_path / "a", "--json")
    assert code == 0 and replayed["equal"]
    code, _ = run(capsys, "agent", "solve", problem_file, *budget, "-o", tmp_path / "a")
    assert code == 2  # a directory that holds a run is never overwritten


def test_tampering_and_a_wrong_digest_fail_verify_and_replay(tmp_path, capsys):
    prompt, replies = case_files(tmp_path, "FC-018")
    problem_file, run_dir = tmp_path / "knapsack.json", tmp_path / "run"
    run(capsys, "agent", "compile", "--prompt-file", prompt, "-o", problem_file, "--provider", "mock",
        "--mock-replies", replies)
    code, solved = run(capsys, "agent", "solve", problem_file, "--max-iterations", 6, "--max-evaluations", 3000,
                       "-o", run_dir, "--json")
    assert code == 0
    code, sealed = run(capsys, "optimization", "verify", run_dir / "sealed", "--json")
    assert code == 0 and sealed["kind"] == "sealed" and sealed["verified"]

    assert run(capsys, "agent", "replay", run_dir, "--expect-digest", "sha256:" + "0" * 64)[0] == 1
    assert run(capsys, "optimization", "verify", run_dir, "--expect-digest", "sha256:" + "0" * 64)[0] == 1
    result = run_dir / "sealed/result.json"
    data = json.loads(result.read_text())
    data["objective_values"] = {k: v + 1.0 for k, v in data["objective_values"].items()}
    result.write_text(json.dumps(data))
    code, graded = run(capsys, "optimization", "verify", run_dir, "--json")
    assert code == 1 and not graded["verified"] and graded["integrity"]
    code, replayed = run(capsys, "agent", "replay", run_dir, "--json")
    assert code == 1 and not replayed["equal"]


def test_validate_reports_every_error_with_a_nonzero_exit(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schemaVersion": "1.1", "id": "p", "variables": [], "objectives": []}))
    code, summary = run(capsys, "optimization", "validate", bad, "--json")
    assert code == 1 and not summary["valid"] and summary["errors"]
    garbled = tmp_path / "garbled.json"
    garbled.write_text("{not json")
    code, summary = run(capsys, "optimization", "validate", garbled, "--json")
    assert code == 1 and summary["errors"][0]["code"] == "schema"
    with pytest.raises(SystemExit) as stop:
        main(["optimization", "validate", str(tmp_path / "missing.json")])
    assert stop.value.code == 2 and "cannot read" in capsys.readouterr().err


def test_credentials_come_from_the_environment_only_and_are_never_printed(tmp_path, capsys, monkeypatch):
    from orbitopt.cli import build_parser

    options = {opt for action in build_parser()._subparsers._group_actions[0].choices["agent"]._subparsers
               ._group_actions[0].choices["compile"]._actions for opt in action.option_strings}
    assert {o for o in options if "key" in o} == {"--api-key-env"}  # a name, never a value
    prompt, _ = case_files(tmp_path, "FC-001")
    monkeypatch.delenv("ORBITOPT_TEST_KEY", raising=False)
    with pytest.raises(SystemExit) as stop:
        main(["agent", "compile", "--prompt-file", str(prompt), "-o", str(tmp_path / "p.json"),
              "--provider", "openai-compatible", "--base-url", "http://127.0.0.1:9/v1", "--model", "m",
              "--api-key-env", "ORBITOPT_TEST_KEY"])
    captured = capsys.readouterr()
    assert stop.value.code == 2 and "ORBITOPT_TEST_KEY is not set" in captured.err
    assert not (tmp_path / "p.json").exists()


CORE_HELP_PROBE = r"""
import json, sys, contextlib, io
import orbitopt, orbitopt.cli
for argv in ([], ["optimization"], ["agent"], ["agent", "solve"]):
    with contextlib.suppress(SystemExit), contextlib.redirect_stdout(io.StringIO()):
        orbitopt.cli.main(argv + ["--help"])
heavy = ("numpy", "scipy", "cma", "pygmo", "openai", "anthropic", "httpx", "requests", "tiktoken", "litellm")
print(json.dumps(sorted(m for m in sys.modules if m.split(".")[0] in heavy or m.startswith("orbitopt.agent")
                        or m.startswith("orbitopt.optimization"))))
"""


ROOTS_PROBE = r"""
import importlib, json, sys
import orbitopt.optimization, orbitopt.agent
before = sorted(m for m in sys.modules if m.startswith(("orbitopt.agent.", "orbitopt.optimization.solvers",
                                                        "orbitopt.optimization.verification")))
for root in (orbitopt.optimization, orbitopt.agent):
    for name in root.__all__:
        getattr(root, name)
heavy = sorted(m for m in sys.modules if m.split(".")[0] in ("numpy", "scipy", "cma", "pygmo"))
print(json.dumps([before, heavy]))
"""


def probe(source):
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    result = subprocess.run([sys.executable, "-c", source], env=env, cwd=REPO, capture_output=True, text=True,
                            check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_cli_help_imports_no_llm_sdk_solver_backend_or_e14_package():
    assert probe(CORE_HELP_PROBE) == []


def test_package_roots_reexport_the_public_api_lazily_and_without_numpy():
    import orbitopt.agent as agent
    import orbitopt.optimization as optimization
    from orbitopt.agent import compiler, providers, runtime, tools
    from orbitopt.optimization import solvers, verification

    for name in agent.__all__:
        assert any(getattr(part, name, None) is getattr(agent, name) for part in (providers, compiler, runtime, tools))
    assert (optimization.default_registry, optimization.OptionSpec) == (solvers.default_registry, solvers.OptionSpec)
    assert (optimization.verify_result, optimization.seal_run) == (verification.verify_result, verification.seal_run)
    assert agent.replay is runtime.replay and agent.solve is runtime.solve  # the run-level names, not the tool log's
    for root in (agent, optimization):
        assert len(set(root.__all__)) == len(root.__all__)
    assert probe(ROOTS_PROBE) == [[], []]  # nothing loaded on import, and no numpy once every name resolved


def test_the_agent_extra_is_the_solvers_extra_and_names_no_llm_sdk():
    tomllib = pytest.importorskip("tomllib")
    extras = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    names = {extra: {req.split(">")[0].split("=")[0].split("[")[0] for req in reqs} for extra, reqs in extras.items()}
    assert extras["agent"] == ["orbitopt[solvers]"] and names["solvers"] == {"numpy", "scipy", "cma"}
    sdks = {"openai", "anthropic", "litellm", "langchain", "httpx", "requests", "tiktoken"}
    assert not any(names[extra] & sdks for extra in names)
