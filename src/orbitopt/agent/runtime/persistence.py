"""Run directories: write a run, and replay it without any LLM call.

A run directory holds:

| File | Content |
|---|---|
| ``problem.json`` | The problem asked. |
| ``trace.json`` | The ``agent-trace-1.2`` trace (a 1.0 or 1.1 trace still loads). |
| ``results/<label>.json`` | Every result of the run, full, by the label the trace uses. |
| ``sealed/`` | The E14-06 sealed run (``orbitopt.optimization.run/2``) of the best result and its final evidence. |
| ``tool-calls.json`` | The hash-chained tool call log, when the run evaluated tools through a ``ToolView``. |
| ``manifest.json`` | ``orbitopt.agent.run/1``: the SHA-256 of every file above and the sealed run's manifest digest. |

The sealed run's manifest digest lives here, outside ``sealed/``, as
``seal_run`` asks. ``save_run`` returns this manifest's own digest; keep it
out of band, since whoever can rewrite the directory can rewrite the
manifest too.

``check_run`` is the integrity check: every byte against the manifest, the
trace digest, the trace against the full results, and the guard's re-grade
of the sealed best result. It runs no solver: by default the final
verification's alternates are given their recorded results. ``replay`` does
the same check with the alternates run again, then re-executes the trace's
recorded proposals with a ``RecordedPlanner`` and compares the new trace
with the recorded one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping

from orbitopt.agent.tools import Caller, ToolView, bind_problem
from orbitopt.optimization import (Budget, InvalidProblemError, OptimizationProblem, OptimizationResult, parse_problem,
                                   schema_errors)
from orbitopt.optimization.solvers import SolverRegistry, default_registry
from orbitopt.optimization.verification import (RUN_FORMAT, Outcome, VerificationReport, canonical_json, file_digest,
                                                seal_run, verify_run)

from .loop import Tools, solve
from .model import (ENVIRONMENT_KEYS, TRACE_SCHEMA, TRACE_SCHEMAS, CancelPoint, Policy, Proposal, RunOutcome,
                    RunStatus, result_summary, stable, trace_digest, upgrade_trace)
from .planners import RecordedPlanner

RUN_MANIFEST_FORMAT = "orbitopt.agent.run/1"
MANIFEST = "manifest.json"
SEALED = "sealed"
MAX_DIFFERENCES = 20
SAFE_PART = re.compile(r"^[A-Za-z0-9_.-]+$")
Alternates = Literal["recorded", "rerun"]
UNCOMPARED = frozenset({"digest", "schemaVersion", *ENVIRONMENT_KEYS})
"""Top-level trace keys replay does not compare: the digest of a 1.0 trace covers its 1.0 form, and the version
that replays may differ from the one that ran."""


class UnsafePathError(ValueError):
    """A file name from a manifest would leave its directory (``..``, an absolute path, a drive, a backslash)."""


def safe_relative(name: Any) -> PurePosixPath:
    """``name`` as a relative POSIX path that stays inside its directory, else ``UnsafePathError``.

    Every part must be ``[A-Za-z0-9_.-]+`` and neither ``.`` nor ``..``, so an
    absolute path, a drive letter, a backslash or an empty part is refused.
    """
    if not isinstance(name, str) or not name:
        raise UnsafePathError(f"not a file name: {name!r:.80}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") or not SAFE_PART.fullmatch(part) for part in name.split("/")):
        raise UnsafePathError(f"unsafe file name: {name!r:.80}")
    return path


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
    """The run's trace as recorded, checked against the schema its ``schemaVersion`` names; ``ValueError`` otherwise.

    ``upgrade_trace`` turns a 1.0 or 1.1 trace into the 1.2 shape.
    """
    return _parse_trace((Path(run_dir) / "trace.json").read_bytes())


def _parse_trace(data: bytes) -> dict:
    trace = json.loads(data.decode("utf-8"))
    schema = TRACE_SCHEMAS.get(trace.get("schemaVersion"), TRACE_SCHEMA) if isinstance(trace, dict) else TRACE_SCHEMA
    errors = schema_errors(trace, schema)
    if errors:
        raise ValueError(f"trace.json does not match {schema}: " + "; ".join(f"{e.path}: {e.message}"
                                                                         for e in errors[:5]))
    return trace


@dataclass(frozen=True)
class CheckReport:
    """The integrity check of a run directory, with no solve run again.

    ``integrity`` lists changed, missing or inconsistent files: a byte that
    does not match the manifest, a trace whose digest does not match its
    content, a trace entry that is not the full result saved beside it, a
    sealed result that is not the trace's best, and a sealed re-grade that
    contradicts the trace's status. ``report`` is the guard's fresh grade of
    ``sealed/``, ``None`` when the run sealed no result. ``alternates`` says
    whether the final verification's alternate solvers were given their
    recorded results (``recorded``) or run again (``rerun``). ``trace``,
    ``problem`` and ``files`` are what was checked: the trace in the newest
    schema's shape (``upgrade_trace``; its ``digest`` is the recorded one),
    the problem it names, and the bytes of every file by its path in the run
    directory, the sealed run's included.
    """

    integrity: tuple[str, ...]
    report: VerificationReport | None
    recorded_status: str | None
    alternates: Alternates
    trace: Mapping[str, Any] | None = None
    problem: OptimizationProblem | None = None
    files: Mapping[str, bytes] = field(default_factory=dict)

    @property
    def intact(self) -> bool:
        return not self.integrity

    @property
    def verified(self) -> bool:
        """Intact, and the guard passes the sealed best result again."""
        return self.intact and self.report is not None and self.report.verdict is Outcome.PASSED

    @property
    def sealed_verdict(self) -> str | None:
        return None if self.report is None else self.report.verdict.value


def check_run(run_dir: str | Path, *, tools: Tools | None = None, registry: SolverRegistry | None = None,
              expected_manifest_digest: str | None = None, alternates: Alternates = "recorded") -> CheckReport:
    """Check a run directory without re-executing it: bytes, trace digest, results, and the sealed re-grade.

    No planner, LLM or loop runs. With ``alternates="recorded"`` no solver
    runs either: the guard re-grades the sealed result with each alternate's
    recorded result as that alternate's answer, so it re-checks the
    alternate's point and the comparison but not that the solver produces
    it. ``"rerun"`` runs the alternates again with their recorded
    allocation, seed and options from ``registry``. The guard still
    re-evaluates points, so a tool-evaluated problem needs ``tools`` as the
    original run had them. Raises ``UnsafePathError`` for a file name that
    would leave ``run_dir``: the manifest's names before any file but
    ``manifest.json`` is read, the sealed manifest's before any sealed file.
    """
    if alternates not in ("recorded", "rerun"):
        raise ValueError(f"alternates must be 'recorded' or 'rerun', not {alternates!r}")
    run_dir = Path(run_dir)
    integrity: list[str] = []
    files: dict[str, bytes] = {}
    try:
        manifest, trace, problem = _open(run_dir, expected_manifest_digest, integrity, files)
    except UnsafePathError:
        raise
    except (OSError, ValueError, InvalidProblemError) as exc:
        return CheckReport(tuple(integrity) + (f"cannot load the run: {exc}",), None, None, alternates, files=files)
    _check_results(trace, files, None if manifest is None else manifest["sealed"], integrity)
    report = _regrade_sealed(run_dir, manifest, problem, trace, files, tools, registry or default_registry(),
                             alternates, integrity)
    status = trace["outcome"]["status"]
    if report is not None and (report.verdict is Outcome.PASSED) != (status == RunStatus.VERIFIED.value):
        integrity.append(f"sealed/: the guard re-grades the sealed run {report.verdict.value!r}, but the trace says "
                         f"{status!r}")
    return CheckReport(tuple(integrity), report, status, alternates, upgrade_trace(trace), problem, files)


@dataclass(frozen=True)
class ReplayReport:
    """``equal`` when the run passed ``check_run`` and the re-executed trace matches the recorded one.

    ``check`` is the integrity check, with the alternates run again.
    ``differences`` lists the trace paths where the re-execution differs
    (wall-clock fields are not compared).
    """

    check: CheckReport
    differences: tuple[str, ...]
    recorded_digest: str | None
    replayed_digest: str | None

    @property
    def integrity(self) -> tuple[str, ...]:
        return self.check.integrity

    @property
    def sealed_verdict(self) -> str | None:
        """The guard's verdict on the sealed run, re-graded from its files."""
        return self.check.sealed_verdict

    @property
    def equal(self) -> bool:
        return not self.integrity and not self.differences

    @property
    def status(self) -> str:
        return "equal" if self.equal else "diverged"


def replay(run_dir: str | Path, *, tools: Tools | None = None, registry: SolverRegistry | None = None,
           expected_manifest_digest: str | None = None) -> ReplayReport:
    """``check_run`` with the alternates run again, then re-execute the recorded proposals; no LLM is called.

    The re-execution uses the recorded seed, limits, policy and cancel point,
    the problem from ``problem.json`` and the solvers of ``registry``. A
    trace without a cancel point (1.0) replays without a cancel. The
    comparison leaves out ``schemaVersion``, so a 1.0 trace that the current
    runtime reproduces is equal. A deterministic
    solver recomputes the same results, so any edit to a recorded result,
    decision or outcome shows up as a difference even when the editor also
    recomputed the trace digest. Pass ``tools`` as the original run did.
    """
    registry = registry or default_registry()
    check = check_run(run_dir, tools=tools, registry=registry, expected_manifest_digest=expected_manifest_digest,
                      alternates="rerun")
    trace, problem = check.trace, check.problem
    if trace is None or problem is None:
        return ReplayReport(check, (), None, None)
    proposals = [Proposal.from_dict(r["proposal"]) for r in trace["iterations"]]
    cancel = trace["outcome"]["cancel"]
    outcome = solve(problem, planner=RecordedPlanner(proposals, name=trace["planner"], provider=trace["provider"],
                                                     model=trace["model"]),
                    budget=Budget(**trace["limits"]), policy=Policy.from_dict(trace["policy"]), tools=tools,
                    seed=trace["seed"], registry=registry,
                    cancel=None if cancel is None else CancelPoint.from_dict(cancel))
    replayed = outcome.trace.to_dict()
    differences = _differences(stable(trace), stable(replayed), "")
    return ReplayReport(check, tuple(differences[:MAX_DIFFERENCES]), trace["digest"], replayed["digest"])


def _open(run_dir: Path, expected: str | None, integrity: list[str], files: dict[str, bytes]):
    """The run's manifest (``None`` when unreadable), trace and problem; every inconsistency goes into ``integrity``.

    ``files`` receives the bytes of every file that matches its digest.
    Raises ``OSError``, ``ValueError`` or ``InvalidProblemError`` when the
    trace or the problem cannot be loaded.
    """
    manifest = _manifest(run_dir, expected, integrity, files)
    trace = _parse_trace(files.get("trace.json") or (run_dir / "trace.json").read_bytes())
    problem = parse_problem(json.loads((files.get("problem.json") or (run_dir / "problem.json").read_bytes())
                                       .decode("utf-8")))
    if trace["digest"] != trace_digest(trace):
        integrity.append("trace.json: the recorded digest does not match the trace's content")
    if trace["problem_digest"] != problem.digest():
        integrity.append("problem.json: its digest is not the problem the trace names")
    if manifest is not None and manifest.get("trace_digest") != trace["digest"]:
        integrity.append("manifest.json: the trace digest does not match trace.json")
    return manifest, trace, problem


def _manifest(run_dir: Path, expected: str | None, integrity: list[str], files: dict[str, bytes]) -> dict | None:
    try:
        raw = (run_dir / MANIFEST).read_bytes()
        manifest = json.loads(raw)
    except (OSError, ValueError) as exc:
        integrity.append(f"{MANIFEST}: cannot read: {exc}")
        return None
    if expected is not None and file_digest(raw) != expected:
        integrity.append(f"{MANIFEST}: digest {file_digest(raw)} is not the expected {expected}")
    if not isinstance(manifest, dict) or manifest.get("format") != RUN_MANIFEST_FORMAT \
            or not isinstance(manifest.get("files"), dict) \
            or not isinstance(manifest.get("sealed"), (dict, type(None))):
        integrity.append(f"{MANIFEST}: not an {RUN_MANIFEST_FORMAT} manifest")
        return None
    files[MANIFEST] = raw
    unsealed = sorted({"problem.json", "trace.json"} - set(manifest["files"]))
    if unsealed:
        integrity.append(f"{MANIFEST}: does not seal {unsealed}")
    sealed = manifest["sealed"]
    for name in (*manifest["files"], *([sealed.get("dir")] if sealed is not None else [])):
        safe_relative(name)  # every name, before any file is read
    for name, recorded in sorted(manifest["files"].items()):
        _read(run_dir / name, name, recorded, integrity, files)
    if sealed is not None:
        _read_sealed(run_dir / sealed["dir"], sealed, files)
    return manifest


def _read(path: Path, name: str, recorded: Any, integrity: list[str], files: dict[str, bytes]) -> None:
    try:
        data = path.read_bytes()
    except OSError:
        integrity.append(f"{name}: missing")
        return
    if file_digest(data) != recorded:
        integrity.append(f"{name}: content digest {file_digest(data)} does not match the manifest")
        return
    files[name] = data


def _read_sealed(folder: Path, sealed: Mapping[str, Any], files: dict[str, bytes]) -> None:
    """The sealed run's manifest and files into ``files``, where they match; ``verify_run`` reports the rest."""
    try:
        raw = (folder / MANIFEST).read_bytes()
        inner = json.loads(raw)
    except (OSError, ValueError):
        return
    if file_digest(raw) != sealed.get("manifest_digest") or not isinstance(inner, dict) \
            or not isinstance(inner.get("files"), dict):
        return
    for name in inner["files"]:
        safe_relative(name)
    files[f"{sealed['dir']}/{MANIFEST}"] = raw
    for name, recorded in sorted(inner["files"].items()):
        _read(folder / name, f"{sealed['dir']}/{name}", recorded, [], files)


def _labelled(trace: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Every result summary in the trace, by the label ``save_run`` files its full result under."""
    out = {f"iteration-{item['index']:04d}": item["result"] for item in trace["iterations"] if item["result"]}
    for attempt in trace["verifications"]:
        for run in attempt["alternates"]:
            if run["result"] is not None:
                out[f"verify-{attempt['attempt']}-{run['solver_id']}"] = run["result"]
    return out


def _check_results(trace: Mapping[str, Any], files: Mapping[str, bytes], sealed: Mapping[str, Any] | None,
                   integrity: list[str]) -> None:
    """Each trace entry summarises the full result saved under its label, and the sealed result is the best."""
    for label, summary in _labelled(trace).items():
        name = f"results/{label}.json"
        if name not in files:
            integrity.append(f"{name}: the trace names this result, but the manifest does not seal it")
            continue
        try:
            full = result_summary(OptimizationResult.from_dict(json.loads(files[name].decode("utf-8"))))
        except Exception as exc:  # noqa: BLE001 - an unreadable result is a finding, never a crash
            integrity.append(f"{name}: not a result: {exc!r:.200}")
            continue
        changed = _differences(stable(summary), stable(full), "")
        if changed:
            integrity.append(f"trace.json: {label} is not the result in {name}: {changed[0]}")
    best = trace["outcome"]["best"]
    name = None if sealed is None else f"{sealed['dir']}/result.json"
    if name in files and files[name] != files.get(f"results/{best}.json"):
        integrity.append(f"{name}: not the result the trace names as best ({best})")


def _regrade_sealed(run_dir: Path, manifest: dict | None, problem: OptimizationProblem, trace: dict,
                    files: Mapping[str, bytes], tools: Tools | None, registry: SolverRegistry, alternates: Alternates,
                    integrity: list[str]) -> VerificationReport | None:
    """The guard's report on the sealed run, each final-verification alternate recorded or run again."""
    sealed = None if manifest is None else manifest.get("sealed")
    if not sealed:
        return None
    best = trace["outcome"]["best"]
    attempts = [v for v in trace["verifications"] if v["candidate"] == best]
    attempt = attempts[-1] if attempts else None
    resolvers = bind_problem(problem, tools, Caller("agent-replay")) if isinstance(tools, ToolView) else tools

    def rerun(run: Mapping[str, Any]):
        solver = registry.get(run["solver_id"])
        return lambda graded: solver.solve(graded, budget=Budget(max_evaluations=run["allocation"]), seed=run["seed"],
                                           options=run["options"] or None, tools=resolvers)

    def recorded(run: Mapping[str, Any]):
        data = files.get(f"results/verify-{attempt['attempt']}-{run['solver_id']}.json")
        # parsed inside the guard's call, so a result that does not parse is the guard's finding, not a crash
        return lambda graded: None if data is None else OptimizationResult.from_dict(json.loads(data.decode("utf-8")))

    answer = rerun if alternates == "rerun" else recorded
    report = verify_run(run_dir / sealed["dir"], problem=problem, tools=resolvers,
                        alternate_solvers={run["solver_id"]: answer(run) for run in attempt["alternates"]}
                        if attempt else None,
                        expected_manifest_digest=sealed["manifest_digest"])
    integrity.extend(f"sealed/: {f.code.value} at {f.path}" for f in report.findings
                     if f.code.value in ("digest_mismatch", "missing_file", "malformed_artifact"))
    return report


def _differences(recorded: Any, replayed: Any, path: str) -> list[str]:
    if isinstance(recorded, dict) and isinstance(replayed, dict):
        out = []
        for key in sorted(set(recorded) | set(replayed)):
            if key in UNCOMPARED and path == "":
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
