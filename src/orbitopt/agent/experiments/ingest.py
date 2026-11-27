"""Record a saved run directory, and write a recorded run back out as a directory.

``record_run`` trusts nothing in the directory. By default it runs the
runtime's ``check_run``: every file's bytes against the manifest, file names
that stay inside the directory, the trace digest, the trace against the
saved results, and the guard's re-grade of the sealed run, with no solve run
again. ``verify="replay"`` has the runtime's ``replay`` re-execute the
recorded proposals as well. The record stores exactly the bytes the check
read.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import cache
from importlib import metadata
from pathlib import Path
from typing import Any, Literal, Mapping

from orbitopt.agent.runtime import BudgetUse, Limits, RunStatus, StopReason, check_run, replay, safe_relative
from orbitopt.optimization import OptimizationProblem
from orbitopt.optimization.solvers import SolverRegistry
from orbitopt.optimization.verification import file_digest

from .index import ExperimentDB
from .model import (STATUS_NAMES, ExperimentRecord, Integrity, RunIntegrityError, SolverStep, VersionPolicy,
                    cache_key, check_run_id)

MANIFEST = "manifest.json"
SEALED_FILES = ("problem.json", "result.json", "evidence.json", MANIFEST)
Verify = Literal["check", "replay"]


@cache
def orbitopt_version() -> str:
    """The installed orbitopt version (read once: the metadata lookup costs about 2 ms)."""
    try:
        return metadata.version("orbitopt")
    except metadata.PackageNotFoundError:
        return "0+unknown"


def record_run(db: ExperimentDB, run_dir: str | Path, *, expected_manifest_digest: str | None = None,
               verify: Verify = "check", tools: Any = None, registry: SolverRegistry | None = None,
               provider: str | None = None, model: str | None = None, family: str | None = None,
               parent_id: str | None = None, resumed_from: str | None = None) -> ExperimentRecord:
    """Verify the run directory ``run_dir``, store its files and index it; return its record.

    Idempotent: the same run (the same ``manifest.json`` bytes) gives one
    record, and recording it again returns the first. A changed, forged or
    unsafe run raises ``RunIntegrityError`` (``UnsafePathError`` for a name
    outside the directory) and records nothing. ``verify="check"`` runs
    ``check_run`` and records ``integrity`` ``checked``; ``verify="replay"``
    re-executes the run as well and records ``replayed``, which only a cache
    entry needs. Pass ``tools`` and ``registry`` as the run used them: the
    guard re-evaluates points, and replay re-runs the solvers.
    ``provider`` and ``model`` name the LLM behind the planner, which
    the trace does not hold. ``family`` defaults to ``problem.metadata["family"]``,
    then to the problem id. ``parent_id`` and ``resumed_from`` must name
    recorded runs.
    """
    if verify not in ("check", "replay"):
        raise ValueError(f"verify must be 'check' or 'replay', not {verify!r}")
    for link in (parent_id, resumed_from):
        if link is not None:
            check_run_id(link)
    run_dir = Path(run_dir)
    try:
        raw = (run_dir / MANIFEST).read_bytes()
    except OSError as exc:
        raise RunIntegrityError(f"{run_dir} is not a run directory", (f"{MANIFEST}: {exc.strerror or exc}",)) from None
    run_id = file_digest(raw)
    if expected_manifest_digest is not None and run_id != expected_manifest_digest:
        raise RunIntegrityError("manifest.json is not the expected run", (f"{run_id} != {expected_manifest_digest}",))
    existing = db.find(run_id)
    if existing is not None:
        return existing

    if verify == "check":
        check, differences = check_run(run_dir, tools=tools, registry=registry, expected_manifest_digest=run_id), None
    else:
        replayed = replay(run_dir, tools=tools, registry=registry, expected_manifest_digest=run_id)
        check, differences = replayed.check, replayed.differences
    if check.integrity:
        raise RunIntegrityError("the run directory was changed", check.integrity)
    if differences:
        raise RunIntegrityError("replay does not reproduce the run", differences)
    trace, files = check.trace, check.files
    integrity = Integrity.CHECKED if differences is None else Integrity.REPLAYED
    record = _record(run_id, _json(files[MANIFEST], MANIFEST), trace, check.problem, files, integrity,
                     provider=provider, model=model, family=family, parent_id=parent_id, resumed_from=resumed_from)
    for data in files.values():
        db.store.put(data)
    return db.add(record, _cache_keys(record))


def export_run(db: ExperimentDB, run_id: str, dest: str | Path) -> Path:
    """Write the recorded run ``run_id`` from the artifact store into ``dest``, so ``replay`` can run on it.

    Every file is read through the store's integrity check and written at its
    recorded path, which never leaves ``dest``.
    """
    record = db.get(run_id)
    dest = Path(dest)
    for name, digest in sorted(record.files.items()):
        path = dest / safe_relative(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(db.store.get(digest))
    return dest


# ------------------------------------------------------------------------------------------------ checks
def _json(data: bytes, name: str) -> Any:
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise RunIntegrityError(f"{name} is not JSON", (str(exc),)) from None


# ------------------------------------------------------------------------------------------------ the record
def _record(run_id: str, manifest: Mapping, trace: Mapping, problem: OptimizationProblem,
            files: Mapping[str, bytes], integrity: Integrity, **links: Any) -> ExperimentRecord:
    steps, summaries = [], {}
    for item in trace["iterations"]:
        decision = item["decision"]
        if item["kind"] != "solve" or decision is None:
            continue
        label = f"iteration-{item['index']:04d}"
        steps.append(SolverStep(label, decision["solver_id"], decision["options"], decision["seed"], decision["x0"],
                                decision["evaluations"], decision["max_iterations"]))
        summaries[label] = item["result"]
    for attempt in trace["verifications"]:
        for run in attempt["alternates"]:
            label = f"verify-{attempt['attempt']}-{run['solver_id']}"
            steps.append(SolverStep(label, run["solver_id"], run["options"], run["seed"], None, run["allocation"]))
            summaries[label] = run["result"]
    best = trace["outcome"]["best"]
    summary = summaries.get(best)
    sealed = manifest.get("sealed")
    objective = problem.objectives[0]
    value = None if summary is None else summary["objective_values"].get(objective.name)
    family = links.pop("family") or problem.metadata.get("family")
    return ExperimentRecord(
        run_id=run_id, problem_id=problem.id, family=family if isinstance(family, str) and family else problem.id,
        problem_digest=trace["problem_digest"], fingerprint=trace["fingerprint"], trace_digest=trace["digest"],
        result_digest=None if sealed is None else _sealed_result_digest(files, sealed["dir"]),
        sealed_digest=None if sealed is None else sealed["manifest_digest"], planner=trace["planner"],
        seed=trace["seed"], steps=tuple(steps), best=best, objective_name=objective.name, sense=objective.sense,
        objective=value, max_violation=None if summary is None else _max_violation(problem, summary),
        statuses=_statuses(trace, best), status=RunStatus(trace["outcome"]["status"]),
        stop_reason=StopReason(trace["outcome"]["stop_reason"]), limits=Limits(**trace["limits"]),
        usage=BudgetUse(**trace["usage"]), created_at=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        orbitopt_version=orbitopt_version(), integrity=integrity,
        files={name: file_digest(data) for name, data in files.items()}, **links)


def _sealed_result_digest(files: Mapping[str, bytes], folder: str) -> str:
    return _json(files[f"{folder}/{MANIFEST}"], "sealed manifest.json")["result_digest"]


def _max_violation(problem: OptimizationProblem, summary: Mapping) -> float | None:
    residuals = summary["constraint_residuals"]
    if any(c.name not in residuals for c in problem.constraints):
        return None
    return max([0.0, *(float(c.violation(residuals[c.name])) for c in problem.constraints)])


def _statuses(trace: Mapping, best: str | None) -> dict[str, str]:
    attempts = [v for v in trace["verifications"] if v["candidate"] == best]
    if attempts:
        report = attempts[-1]["report"]
    else:
        screens = [i["screen"] for i in trace["iterations"] if f"iteration-{i['index']:04d}" == best and i["screen"]]
        report = screens[0] if screens else {"statuses": {}}
    return {name: s["outcome"] for name, s in report["statuses"].items() if name in STATUS_NAMES}


def _cache_keys(record: ExperimentRecord) -> list[str]:
    """One key per version policy for the step that produced a verified, fully replayed best."""
    step = record.best_step
    if not record.verified or record.integrity is not Integrity.REPLAYED or step is None:
        return []
    return [cache_key(record.problem_digest, step.solver_id, step.options, step.seed, step.x0, step.evaluations,
                      version=record.orbitopt_version, policy=policy) for policy in VersionPolicy]
