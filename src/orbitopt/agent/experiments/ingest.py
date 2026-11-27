"""Record a saved run directory, and write a recorded run back out as a directory.

``record_run`` trusts nothing in the directory. It checks every file's bytes
against the manifest, refuses a file name that would leave the directory,
and then has the runtime's ``replay`` re-execute the recorded proposals and
re-grade the sealed run. Only a run that replays is recorded. The numbers
in the record come from the replayed trace, and the sealed result must
agree with them.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import cache
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

from orbitopt.agent.runtime import RUN_MANIFEST_FORMAT, BudgetUse, Limits, RunStatus, StopReason, replay
from orbitopt.optimization import OptimizationProblem, canonical_digest, parse_problem
from orbitopt.optimization.solvers import SolverRegistry
from orbitopt.optimization.verification import RUN_FORMAT, file_digest

from .index import ExperimentDB
from .model import (STATUS_NAMES, ExperimentRecord, Integrity, RunIntegrityError, SolverStep, VersionPolicy,
                    cache_key, check_run_id)
from .store import safe_relative

MANIFEST = "manifest.json"
SEALED_FILES = ("problem.json", "result.json", "evidence.json", MANIFEST)
OUTCOME_ONLY = ("/outcome/", "/usage/", "/verifications")
"""Where the replay of a cancelled run may differ: a cancellation cannot be re-executed, its iterations can."""
COMPARED = ("solver_id", "x", "objective_values", "constraint_residuals")


@cache
def orbitopt_version() -> str:
    """The installed orbitopt version (read once: the metadata lookup costs about 2 ms)."""
    try:
        return metadata.version("orbitopt")
    except metadata.PackageNotFoundError:
        return "0+unknown"


def record_run(db: ExperimentDB, run_dir: str | Path, *, expected_manifest_digest: str | None = None,
               tools: Any = None, registry: SolverRegistry | None = None, provider: str | None = None,
               model: str | None = None, family: str | None = None, parent_id: str | None = None,
               resumed_from: str | None = None) -> ExperimentRecord:
    """Verify the run directory ``run_dir``, store its files and index it; return its record.

    Idempotent: the same run (the same ``manifest.json`` bytes) gives one
    record, and recording it again returns the first. A changed, forged or
    unsafe run raises ``RunIntegrityError`` and records nothing. Pass
    ``tools`` and ``registry`` as the run used them, since replay re-executes
    the run. ``provider`` and ``model`` name the LLM behind the planner, which
    the trace does not hold. ``family`` defaults to ``problem.metadata["family"]``,
    then to the problem id. ``parent_id`` and ``resumed_from`` must name
    recorded runs.
    """
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

    manifest = _json(raw, MANIFEST)
    files = _read_files(run_dir, manifest)
    report = replay(run_dir, tools=tools, registry=registry, expected_manifest_digest=run_id)
    if report.integrity:
        raise RunIntegrityError("the run directory was changed", report.integrity)
    trace = _json(files["trace.json"], "trace.json")
    integrity = _integrity(report.differences, trace)
    problem = parse_problem(_json(files["problem.json"], "problem.json"))  # the bytes replay parsed
    record = _record(run_id, manifest, trace, problem, files, integrity, provider=provider, model=model,
                     family=family, parent_id=parent_id, resumed_from=resumed_from)
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


def _read_files(run_dir: Path, manifest: Any) -> dict[str, bytes]:
    """Every file the manifest seals, with its bytes checked; the sealed run's files under ``sealed/``."""
    if not isinstance(manifest, dict) or manifest.get("format") != RUN_MANIFEST_FORMAT \
            or not isinstance(manifest.get("files"), dict):
        raise RunIntegrityError(f"{MANIFEST} is not an {RUN_MANIFEST_FORMAT} manifest")
    missing = {"problem.json", "trace.json"} - set(manifest["files"])
    if missing:
        raise RunIntegrityError(f"{MANIFEST} does not seal {sorted(missing)}")
    problems: list[str] = []
    files = {MANIFEST: (run_dir / MANIFEST).read_bytes()}
    sealed = manifest.get("sealed")
    expected = dict(manifest["files"])
    if sealed is not None:
        if not isinstance(sealed, dict) or sealed.get("format") != RUN_FORMAT:
            raise RunIntegrityError(f"{MANIFEST}: the sealed run is not {RUN_FORMAT}")
        folder = str(safe_relative(sealed.get("dir")))
        seal = _checked(run_dir, f"{folder}/{MANIFEST}", sealed.get("manifest_digest"), problems)
        inner = {} if seal is None else _json(seal, f"{folder}/{MANIFEST}")
        inner_files = inner.get("files") if isinstance(inner, dict) else None
        if seal is not None and not isinstance(inner_files, dict):
            raise RunIntegrityError(f"{folder}/{MANIFEST} is not a {RUN_FORMAT} manifest")
        if seal is not None:
            files[f"{folder}/{MANIFEST}"] = seal
            expected.update({f"{folder}/{name}": digest for name, digest in inner_files.items()})
    for name in expected:
        safe_relative(name)
    for name, digest in sorted(expected.items()):
        data = _checked(run_dir, name, digest, problems)
        if data is not None:
            files[name] = data
    if problems:
        raise RunIntegrityError("the run directory was changed", tuple(problems))
    return files


def _checked(run_dir: Path, name: str, digest: Any, problems: list[str]) -> bytes | None:
    path = run_dir / safe_relative(name)
    try:
        data = path.read_bytes()
    except OSError:
        problems.append(f"{name}: missing")
        return None
    if file_digest(data) != digest:
        problems.append(f"{name}: content digest {file_digest(data)} does not match the manifest")
        return None
    return data


def _integrity(differences: tuple[str, ...], trace: Mapping) -> Integrity:
    if not differences:
        return Integrity.REPLAYED
    if trace["outcome"]["stop_reason"] == StopReason.CANCELLED.value and not trace["verifications"] \
            and all(d.startswith(OUTCOME_ONLY) for d in differences):
        return Integrity.ITERATIONS_REPLAYED
    raise RunIntegrityError("replay does not reproduce the run", differences)


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
    sealed_result = None
    if sealed is not None:
        sealed_result = _json(files[f"{sealed['dir']}/result.json"], "sealed result.json")
        if summary is None or _identity(sealed_result) != _identity(summary):
            raise RunIntegrityError("the sealed result is not the best result the replayed trace names", (str(best),))
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


def _identity(result: Mapping) -> str:
    return canonical_digest({key: result.get(key) for key in COMPARED})


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
