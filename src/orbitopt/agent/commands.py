"""What the ``orbitopt agent`` commands and ``orbitopt optimization verify`` compute: paths in, summaries out.

The CLI only parses arguments and prints. Provider credentials come from
the environment only: a command takes the *name* of the variable that holds
an API key, never the key, so a key never lands in shell history, a
cassette or a report.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from orbitopt.agent.providers import ChatProvider, make_provider
from orbitopt.optimization import Budget, OptimizationProblem, ToolCall
from orbitopt.optimization.commands import CommandError, error_dict, load_problem, read_json, verify_sealed_run
from orbitopt.optimization.verification import RUN_FORMAT

PROVIDERS = ("mock", "openai-compatible", "local", "replay")
"""``--provider`` values. ``replay`` serves a recorded cassette and calls no model."""
PLANNERS = ("rule", "llm")


def build_provider(kind: str, *, model: str | None = None, base_url: str | None = None,
                   api_key_env: str | None = None, mock_replies: str | Path | None = None,
                   cassette: str | Path | None = None) -> ChatProvider:
    """The provider a command talks to; with ``cassette`` every call is recorded there (served, for ``replay``).

    ``mock`` replies from ``mock_replies``, a JSON list whose items are reply
    strings or JSON objects (sent as their JSON text). ``openai-compatible``
    needs ``base_url`` and ``model`` and reads its key from the environment
    variable ``api_key_env`` (default ``OPENAI_API_KEY``) at call time.
    ``local`` needs ``model`` and takes no key.
    """
    if kind == "replay":
        if cassette is None:
            raise CommandError("--provider replay needs --cassette")
        return make_provider({"kind": "replay", "cassette": cassette})
    if kind == "mock":
        if mock_replies is None:
            raise CommandError("--provider mock needs --mock-replies (a JSON list of replies)")
        replies = read_json(mock_replies)
        if not isinstance(replies, list):
            raise CommandError(f"{mock_replies}: the mock replies must be a JSON list")
        spec: dict[str, Any] = {"kind": "mock", "script": [r if isinstance(r, str) else json.dumps(r, sort_keys=True)
                                                           for r in replies]}
    elif kind == "openai-compatible":
        if not base_url or not model:
            raise CommandError("--provider openai-compatible needs --base-url and --model")
        spec = {"kind": "openai_compat", "base_url": base_url, "model": model,
                "api_key_env": api_key_env or "OPENAI_API_KEY"}
    elif kind == "local":
        if not model:
            raise CommandError("--provider local needs --model")
        spec = {"kind": "local", "model": model, **({"base_url": base_url} if base_url else {})}
    else:
        raise CommandError(f"--provider must be one of {list(PROVIDERS)}, not {kind!r}")
    if cassette is not None:
        spec["record"] = cassette
    return make_provider(spec)


# ------------------------------------------------------------------------------------------------------- compile
def compile_prompt(prompt_file: str | Path, output: str | Path, provider: ChatProvider, *,
                   report: str | Path | None = None, allow_draft: bool = False) -> dict:
    """Compile the statement in ``prompt_file`` and write the problem to ``output`` and the report beside it.

    Only a ``compile`` outcome writes ``output``, unless ``allow_draft``: then
    a refused compile writes the model's raw draft there instead, which is
    not an IR problem and says so. The report (default
    ``<output stem>.compile.json``) is always written and returned. It lists
    the assumptions, every question split into missing parameters,
    ambiguities and other questions, the refusal reason, unauthorised
    changes or compile errors, and the tokens spent. A problem may call the
    built-in tools, the ones ``solve_file`` can resolve. A provider failure
    (unreachable, no key, budget, a cassette that does not match) is a
    ``CommandError``.
    """
    from orbitopt.agent.compiler import (CompileFailed, Compiled, NeedsClarification, RejectedUnauthorizedChange,
                                         Unsupported, compile_problem)
    from orbitopt.agent.providers import ProviderError, ReplayMismatchError
    from orbitopt.agent.tools import builtin_registry

    try:
        prompt = Path(prompt_file).read_text(encoding="utf-8")
    except OSError as exc:
        raise CommandError(f"{prompt_file}: cannot read: {exc.strerror or exc}") from None
    output = Path(output)
    report_path = Path(report) if report is not None else output.with_name(output.stem + ".compile.json")
    try:
        outcome = compile_problem(prompt, provider, known_tool_ids=[t.tool_id for t in builtin_registry().list()])
    except (ProviderError, ReplayMismatchError) as exc:
        raise CommandError(f"the provider failed: {exc}") from None
    questions = list(outcome.questions) if isinstance(outcome, NeedsClarification) else []
    summary: dict[str, Any] = {
        "outcome": outcome.outcome, "accepted": isinstance(outcome, Compiled), "attempts": outcome.attempts,
        "usage": asdict(outcome.usage), "problem_path": None, "problem_digest": None, "draft_path": None,
        "assumptions": [asdict(a) for a in outcome.assumptions] if isinstance(outcome, Compiled) else [],
        "missing_parameters": [asdict(q) for q in questions if q.kind == "missing_parameter"],
        "ambiguities": [asdict(q) for q in questions if q.kind == "ambiguity"],
        "questions": [asdict(q) for q in questions if q.kind not in ("missing_parameter", "ambiguity")],
        "unsupported": asdict(outcome.reason) if isinstance(outcome, Unsupported) else None,
        "violations": [asdict(v) for v in outcome.violations] if isinstance(outcome, RejectedUnauthorizedChange)
        else [],
        "errors": [error_dict(e) for e in outcome.errors] if isinstance(outcome, CompileFailed) else [],
    }
    draft = None
    if isinstance(outcome, Compiled):
        _write_json(output, outcome.problem.to_dict())
        summary.update(problem_path=str(output), problem_digest=outcome.problem.digest())
    elif allow_draft:
        draft = outcome.partial if isinstance(outcome, NeedsClarification) else \
            _json_or_none(outcome.raw) if isinstance(outcome, CompileFailed) else None
    if draft is not None:
        _write_json(output, {"draft": True, "outcome": outcome.outcome, "compiler_output": draft})
        summary["draft_path"] = str(output)
    _write_json(report_path, summary)
    summary["report_path"] = str(report_path)
    return summary


def _write_json(path: Path, document: Any) -> None:
    """Readable JSON with sorted keys; ``ValueError`` for NaN or infinity, which no document may hold."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def _json_or_none(text: str | None) -> Any:
    try:
        return None if text is None else json.loads(text)
    except ValueError:
        return None


# --------------------------------------------------------------------------------------------------------- solve
def tools_for(problem: OptimizationProblem):
    """A logged view of the built-in tools the problem's ``ToolCall`` items name; ``None`` when it names none.

    ``CommandError`` for a tool id that is not built in: a command line has
    no way to pass a resolver.
    """
    named = {item.expression.tool_id for item in (*problem.objectives, *problem.constraints)
             if isinstance(item.expression, ToolCall)}
    if not named:
        return None
    from orbitopt.agent.tools import builtin_registry

    registry = builtin_registry()
    unknown = sorted(t for t in named if t not in registry)
    if unknown:
        raise CommandError(f"the problem calls tools that are not built in: {unknown}")
    return registry.view(named)


def solve_file(problem_file: str | Path, output: str | Path, *, budget: Budget, seed: int = 0,
               planner: str = "rule", provider: ChatProvider | None = None) -> dict:
    """Run the agent loop on the problem at ``problem_file``, save the run directory at ``output``, and summarise.

    ``budget`` overrides the problem's own limits field by field. The ``llm``
    planner asks ``provider``; the ``rule`` planner asks nothing. ``output``
    must not hold a run already. The summary's ``manifest_digest`` is what
    ``replay`` and ``verify`` take as the expected digest; keep it out of band.
    """
    from orbitopt.agent.runtime import LLMPlanner, RuleBasedPlanner, save_run, solve

    if planner not in PLANNERS:
        raise CommandError(f"--planner must be one of {list(PLANNERS)}, not {planner!r}")
    if (planner == "llm") != (provider is not None):
        raise CommandError("--planner llm needs --provider, and only it takes one")
    output = Path(output)
    if (output / "manifest.json").exists():
        raise CommandError(f"{output} already holds a run; choose a new directory")
    problem = load_problem(problem_file)
    chosen = LLMPlanner(provider) if planner == "llm" else RuleBasedPlanner()
    outcome = solve(problem, planner=chosen, budget=budget, seed=seed, tools=tools_for(problem))
    digest = save_run(outcome, output)
    result = outcome.result
    return {
        "run_dir": str(output), "manifest_digest": digest, "status": outcome.status.value,
        "verified": outcome.verified, "stop_reason": outcome.stop_reason.value, "statuses": outcome.statuses,
        "solver_id": None if result is None else result.solver_id,
        "x": None if result is None else dict(result.x),
        "objective_values": None if result is None else dict(result.objective_values),
        "usage": asdict(outcome.trace.usage), "iterations": len(outcome.trace.iterations),
    }


# ------------------------------------------------------------------------------------------- replay and verify
def _run_problem(run_dir: Path) -> OptimizationProblem | None:
    try:
        return load_problem(run_dir / "problem.json")
    except CommandError:
        return None  # the runtime reports an unreadable problem as an integrity finding


def replay_dir(run_dir: str | Path, *, expected_manifest_digest: str | None = None) -> dict:
    """Re-execute a saved agent run from its recorded proposals, with no planner or LLM call."""
    from orbitopt.agent.runtime import replay

    run_dir = Path(run_dir)
    problem = _run_problem(run_dir)
    report = replay(run_dir, tools=None if problem is None else tools_for(problem),
                    expected_manifest_digest=expected_manifest_digest)
    return {"run_dir": str(run_dir), "status": report.status, "equal": report.equal,
            "integrity": list(report.integrity), "differences": list(report.differences),
            "recorded_digest": report.recorded_digest, "replayed_digest": report.replayed_digest,
            "sealed_verdict": report.sealed_verdict}


def verify_run_dir(run_dir: str | Path, *, expected_manifest_digest: str | None = None) -> dict:
    """Re-grade a saved run with the guard: an agent run directory, or a bare sealed run.

    An agent run (``orbitopt.agent.run/1``) has every byte checked and its
    sealed best result re-graded, with the final verification's alternate
    solvers run again; the loop is not re-run (that is ``replay``). A sealed
    run (``orbitopt.optimization.run/2``) is graded from its own files.
    """
    from orbitopt.agent.runtime import RUN_MANIFEST_FORMAT, regrade

    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "manifest.json")
    kind = manifest.get("format") if isinstance(manifest, dict) else None
    if kind == RUN_FORMAT:
        return verify_sealed_run(run_dir, expected_manifest_digest=expected_manifest_digest)
    if kind != RUN_MANIFEST_FORMAT:
        raise CommandError(f"{run_dir}: manifest format {kind!r} is neither {RUN_MANIFEST_FORMAT} nor {RUN_FORMAT}")
    problem = _run_problem(run_dir)
    graded = regrade(run_dir, tools=None if problem is None else tools_for(problem),
                     expected_manifest_digest=expected_manifest_digest)
    return {"run_dir": str(run_dir), "kind": "agent", "verified": graded.verified,
            "integrity": list(graded.integrity), "recorded_status": graded.recorded_status,
            "report": None if graded.report is None else graded.report.to_dict()}
