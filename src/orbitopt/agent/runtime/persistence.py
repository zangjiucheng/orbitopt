"""Run directories: write a run, and replay it without any LLM call.

A run directory holds:

| File | Content |
|---|---|
| ``problem.json`` | The problem asked. |
| ``trace.json`` | The ``agent-trace-1.0`` trace. |
| ``results/<label>.json`` | Every result of the run, full, by the label the trace uses. |
| ``sealed/`` | The E14-06 sealed run (``orbitopt.optimization.run/2``) of the best result and its final evidence. |
| ``tool-calls.json`` | The hash-chained tool call log, when the run evaluated tools through a ``ToolView``. |
| ``manifest.json`` | ``orbitopt.agent.run/1``: the SHA-256 of every file above and the sealed run's manifest digest. |

The sealed run's manifest digest lives here, outside ``sealed/``, as
``seal_run`` asks. ``save_run`` returns this manifest's own digest; keep it
out of band, since whoever can rewrite the directory can rewrite the
manifest too. ``replay`` checks every byte, then re-executes the trace's
recorded proposals with a ``RecordedPlanner`` and compares the new trace
with the recorded one. ``regrade`` checks every byte and re-grades only the
sealed best result with the guard, without re-running the loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from orbitopt.agent.tools import Caller, ToolView, bind_problem
from orbitopt.optimization import Budget, InvalidProblemError, OptimizationProblem, parse_problem, schema_errors
from orbitopt.optimization.solvers import SolverRegistry, default_registry
from orbitopt.optimization.verification import (RUN_FORMAT, Outcome, VerificationReport, canonical_json, file_digest,
                                                seal_run, verify_run)

from .loop import Tools, solve
from .model import TRACE_SCHEMA, Policy, Proposal, RunOutcome, RunStatus, stable, trace_digest
from .planners import RecordedPlanner

RUN_MANIFEST_FORMAT = "orbitopt.agent.run/1"
MANIFEST = "manifest.json"
SEALED = "sealed"
MAX_DIFFERENCES = 20


def save_run(outcome: RunOutcome, run_dir: str | Path) -> str:
    """Write ``outcome`` as a run directory and return the ``sha256:`` digest of its manifest."""
    run_dir = Path(run_dir)
    files: dict[str, bytes] = {
        "problem.json": canonical_json(outcome.problem.to_dict()),
        "trace.json": canonical_json(outcome.trace.to_dict()),
        **{f"results/{label}.json": canonical_json(result.to_dict()) for label, result in outcome.results.items()},
    }
    if outcome.tool_log is not None:
        files["tool-calls.json"] = outcome.tool_log.to_json().encode("utf-8")
    for name, data in files.items():
        path = run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    sealed = None
    if outcome.result is not None:
        digest = seal_run(outcome.problem, outcome.result, outcome.evidence, run_dir / SEALED)
        sealed = {"dir": SEALED, "format": RUN_FORMAT, "manifest_digest": digest}
    manifest = canonical_json({"format": RUN_MANIFEST_FORMAT, "files": {n: file_digest(b) for n, b in files.items()},
                               "problem_digest": outcome.problem.digest(), "trace_digest": outcome.trace.digest(),
                               "sealed": sealed})
    (run_dir / MANIFEST).write_bytes(manifest)
    return file_digest(manifest)


def load_trace(run_dir: str | Path) -> dict:
    """The run's trace as a dict, checked against ``agent-trace-1.0.json``; ``ValueError`` otherwise."""
    data = json.loads((Path(run_dir) / "trace.json").read_text(encoding="utf-8"))
    errors = schema_errors(data, TRACE_SCHEMA)
    if errors:
        raise ValueError(f"trace.json does not match {TRACE_SCHEMA}: " + "; ".join(f"{e.path}: {e.message}"
                                                                              for e in errors[:5]))
    return data


@dataclass(frozen=True)
class ReplayReport:
    """``equal`` when every file is intact and the re-executed trace matches the recorded one.

    ``integrity`` lists changed, missing or inconsistent files. ``differences``
    lists the trace paths where the re-execution differs (wall-clock fields
    are not compared). ``sealed_verdict`` is the guard's verdict on the sealed
    run, re-graded from its files.
    """

    integrity: tuple[str, ...]
    differences: tuple[str, ...]
    recorded_digest: str | None
    replayed_digest: str | None
    sealed_verdict: str | None

    @property
    def equal(self) -> bool:
        return not self.integrity and not self.differences

    @property
    def status(self) -> str:
        return "equal" if self.equal else "diverged"


def replay(run_dir: str | Path, *, tools: Tools | None = None, registry: SolverRegistry | None = None,
           expected_manifest_digest: str | None = None) -> ReplayReport:
    """Check a run directory and re-execute its recorded proposals; no planner or LLM is called.

    The re-execution uses the recorded seed, limits and policy, the problem
    from ``problem.json`` and the solvers of ``registry``. A deterministic
    solver recomputes the same results, so any edit to a recorded result,
    decision or outcome shows up as a difference even when the editor also
    recomputed the trace digest. Pass ``tools`` as the original run did.
    """
    registry = registry or default_registry()
    integrity: list[str] = []
    try:
        manifest, trace, problem = _open(Path(run_dir), expected_manifest_digest, integrity)
    except (OSError, ValueError, InvalidProblemError) as exc:
        return ReplayReport(tuple(integrity) + (f"cannot load the run: {exc}",), (), None, None, None)
    proposals = [Proposal.from_dict(r["proposal"]) for r in trace["iterations"]]
    outcome = solve(problem, planner=RecordedPlanner(proposals, name=trace["planner"]),
                    budget=Budget(**trace["limits"]), policy=Policy.from_dict(trace["policy"]), tools=tools,
                    seed=trace["seed"], registry=registry)
    replayed = outcome.trace.to_dict()
    differences = _differences(stable(trace), stable(replayed), "")
    report = _regrade_sealed(Path(run_dir), manifest, problem, trace, tools, registry, integrity)
    sealed = None if report is None else report.verdict.value
    verified = trace["outcome"]["status"] == RunStatus.VERIFIED.value
    if sealed is not None and (sealed == Outcome.PASSED.value) != verified:
        differences.append(f"sealed/: the guard re-grades the sealed run {sealed!r}, but the trace says "
                           f"{trace['outcome']['status']!r}")
    return ReplayReport(tuple(integrity), tuple(differences[:MAX_DIFFERENCES]), trace["digest"], replayed["digest"],
                        sealed)


@dataclass(frozen=True)
class RegradeReport:
    """``verified`` when every file is intact and the guard passes the sealed best result again.

    ``integrity`` lists changed, missing or inconsistent files. ``report`` is
    the guard's fresh grade of ``sealed/``, with the final verification's
    alternate solvers run again; ``None`` when the run sealed no result.
    ``recorded_status`` is the run status the trace claims.
    """

    integrity: tuple[str, ...]
    report: VerificationReport | None
    recorded_status: str | None

    @property
    def verified(self) -> bool:
        return not self.integrity and self.report is not None and self.report.verdict is Outcome.PASSED


def regrade(run_dir: str | Path, *, tools: Tools | None = None, registry: SolverRegistry | None = None,
            expected_manifest_digest: str | None = None) -> RegradeReport:
    """Check a run directory's bytes and re-grade its sealed best result; no planner, LLM or loop runs.

    A trace that says ``verified`` while the guard does not pass the sealed
    result, or the other way round, is an integrity finding. Pass ``tools``
    as the original run did.
    """
    run_dir = Path(run_dir)
    integrity: list[str] = []
    try:
        manifest, trace, problem = _open(run_dir, expected_manifest_digest, integrity)
    except (OSError, ValueError, InvalidProblemError) as exc:
        return RegradeReport(tuple(integrity) + (f"cannot load the run: {exc}",), None, None)
    report = _regrade_sealed(run_dir, manifest, problem, trace, tools, registry or default_registry(), integrity)
    status = trace["outcome"]["status"]
    if report is not None and (report.verdict is Outcome.PASSED) != (status == RunStatus.VERIFIED.value):
        integrity.append(f"sealed/: the guard re-grades the sealed run {report.verdict.value!r}, but the trace says "
                         f"{status!r}")
    return RegradeReport(tuple(integrity), report, status)


def _open(run_dir: Path, expected: str | None, integrity: list[str]):
    """The run's manifest (``None`` when unreadable), trace and problem, with every consistency failure in ``integrity``.

    Raises ``OSError``, ``ValueError`` or ``InvalidProblemError`` when the trace or the problem cannot be loaded.
    """
    manifest = _manifest(run_dir, expected, integrity)
    trace = load_trace(run_dir)
    problem = parse_problem(json.loads((run_dir / "problem.json").read_text(encoding="utf-8")))
    if trace["digest"] != trace_digest(trace):
        integrity.append("trace.json: the recorded digest does not match the trace's content")
    if trace["problem_digest"] != problem.digest():
        integrity.append("problem.json: its digest is not the problem the trace names")
    if manifest is not None and manifest.get("trace_digest") != trace["digest"]:
        integrity.append("manifest.json: the trace digest does not match trace.json")
    return manifest, trace, problem


def _manifest(run_dir: Path, expected: str | None, integrity: list[str]) -> dict | None:
    try:
        raw = (run_dir / MANIFEST).read_bytes()
        manifest = json.loads(raw)
    except (OSError, ValueError) as exc:
        integrity.append(f"{MANIFEST}: cannot read: {exc}")
        return None
    if expected is not None and file_digest(raw) != expected:
        integrity.append(f"{MANIFEST}: digest {file_digest(raw)} is not the expected {expected}")
    if not isinstance(manifest, dict) or manifest.get("format") != RUN_MANIFEST_FORMAT:
        integrity.append(f"{MANIFEST}: not an {RUN_MANIFEST_FORMAT} manifest")
        return None
    for name, recorded in sorted(manifest.get("files", {}).items()):
        try:
            actual = file_digest((run_dir / name).read_bytes())
        except OSError:
            integrity.append(f"{name}: missing")
            continue
        if actual != recorded:
            integrity.append(f"{name}: content digest {actual} does not match the manifest")
    return manifest


def _regrade_sealed(run_dir: Path, manifest: dict | None, problem: OptimizationProblem, trace: dict,
                    tools: Tools | None, registry: SolverRegistry,
                    integrity: list[str]) -> VerificationReport | None:
    """The guard's report on the sealed run, with the final verification's alternate solvers run again."""
    sealed = None if manifest is None else manifest.get("sealed")
    if not sealed:
        return None
    best = trace["outcome"]["best"]
    attempts = [v for v in trace["verifications"] if v["candidate"] == best]
    runs = attempts[-1]["alternates"] if attempts else []

    def alternate(run: Mapping[str, Any]):
        solver = registry.get(run["solver_id"])
        return lambda graded: solver.solve(graded, budget=Budget(max_evaluations=run["allocation"]), seed=run["seed"],
                                           options=run["options"] or None, tools=resolvers)

    resolvers = bind_problem(problem, tools, Caller("agent-replay")) if isinstance(tools, ToolView) else tools
    report = verify_run(run_dir / sealed["dir"], problem=problem, tools=resolvers,
                        alternate_solvers={run["solver_id"]: alternate(run) for run in runs},
                        expected_manifest_digest=sealed["manifest_digest"])
    integrity.extend(f"sealed/: {f.code.value} at {f.path}" for f in report.findings
                     if f.code.value in ("digest_mismatch", "missing_file", "malformed_artifact"))
    return report


def _differences(recorded: Any, replayed: Any, path: str) -> list[str]:
    if isinstance(recorded, dict) and isinstance(replayed, dict):
        out = []
        for key in sorted(set(recorded) | set(replayed)):
            if key == "digest" and path == "":
                continue
            if key not in recorded or key not in replayed:
                out.append(f"{path}/{key}: present on one side only")
            else:
                out.extend(_differences(recorded[key], replayed[key], f"{path}/{key}"))
        return out
    if isinstance(recorded, list) and isinstance(replayed, list):
        out = [f"{path}: {len(recorded)} recorded items, {len(replayed)} replayed"] if len(recorded) != len(replayed) \
            else []
        for i, (a, b) in enumerate(zip(recorded, replayed)):
            out.extend(_differences(a, b, f"{path}/{i}"))
        return out
    return [] if _same(recorded, replayed) else [f"{path}: recorded {recorded!r}, replayed {replayed!r}"]


def _same(a: Any, b: Any) -> bool:
    """Equal, with numbers compared at the 12 significant digits the digests use."""
    numbers = (int, float)
    if isinstance(a, numbers) and isinstance(b, numbers) and not isinstance(a, bool) and not isinstance(b, bool):
        return float(f"{a:.12g}") == float(f"{b:.12g}")
    return a == b
