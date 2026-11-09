"""The unified mission pipeline: one stage lifecycle and one standardized
result document for every config-driven mission kind, shared by the CLI
(``orbitopt run``), the GUI (the built-in "goes"/"mars" missions in
orbitopt.missions) and the Python API.

    MissionConfig -> Constraints -> Search -> Optimize -> Select -> Refine -> Verify -> Export

    from orbitopt.mission_config import load_mission_config
    from orbitopt.pipeline import run_pipeline

    result = run_pipeline(load_mission_config("goes.yaml"))
    result.scene                # SceneData dict (verification reports removed)
    result.verification_reports # machine-readable reports (orbitopt.verify.report)
    result.to_dict()            # pipeline-result-1.0 document, same shape for every kind
    result.success              # True only if the exported candidate passed verification
    result.gate                 # Verification Gate: per-candidate verdicts + warnings
    print(result.summary())

Every stage reports the same fields (``StageResult``): candidate
trajectories, convergence status, constraint violations, the physical model
used, numerical error estimates and wall-clock time. A stage a mission kind
doesn't have is ``skipped``; a stage that never ran because an earlier one
raised is ``not_run``.

Candidate Selection and the Verification Gate (task #171, see
orbitopt.candidate_selection): the ``select`` stage dedupes, ranks and
diversity-filters the optimizer's candidates into the short list refinement
walks through, recording why each candidate was kept or dropped. The gate
(``StageRecorder.gate``) records each candidate's verdict -- rejected by
refinement, accepted (verification passed) or degraded (verification failed
or not_checked) -- and which one was exported. Only an accepted candidate
makes the run a ``success``; a best-effort export (nothing passed) carries a
prominent warning in the pipeline result, the scene and the CLI output.
Builders that don't drive the gate get a generic one: the exported scene's
verification reports decide.

Migration is incremental, not a one-shot rewrite: a mission kind's existing
``build_from_config(config, recorder=None)`` builder keeps doing the work
and just records what each stage produced through the ``StageRecorder`` it
is handed (``with recorder.stage("search") as st: ...``). The pipeline
itself owns the ``mission_config`` stage (schema validation), fills in a
``verify`` summary from the verification reports when the builder didn't
record one, and owns ``export`` (scene validation and, optionally, writing
the scene / verification report / pipeline result files).

This module only needs stdlib + jsonschema + pyyaml at import time; a mission
kind's compute stack is imported when that kind is actually run.
"""
from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cache
from importlib import resources
from pathlib import Path

SCHEMA_VERSION = "1.0"
SCHEMA_FILE = "pipeline-result-1.0.json"

MISSION_CONFIG = "mission_config"
CONSTRAINTS = "constraints"
SEARCH = "search"
OPTIMIZE = "optimize"
SELECT = "select"
REFINE = "refine"
VERIFY = "verify"
EXPORT = "export"
STAGES = (MISSION_CONFIG, CONSTRAINTS, SEARCH, OPTIMIZE, SELECT, REFINE, VERIFY, EXPORT)

STATUS_COMPLETED = "completed"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"
STATUS_NOT_RUN = "not_run"
STAGE_STATUSES = (STATUS_COMPLETED, STATUS_SKIPPED, STATUS_FAILED, STATUS_NOT_RUN)

# physical_model["fidelity"] vocabulary, lowest to highest.
FIDELITIES = ("none", "analytic", "patched-conic", "two-body", "high-fidelity")


def _jsonable(obj):
    """numpy scalars/arrays and tuples -> plain JSON types; NaN/inf -> None
    (the document is written with allow_nan=False)."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "tolist"):  # numpy array / scalar
        return _jsonable(obj.tolist())
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return str(obj)


@dataclass
class StageResult:
    """Standardized outcome of one pipeline stage."""

    stage: str
    status: str = STATUS_NOT_RUN
    elapsed_s: float = 0.0
    # Candidate trajectories this stage produced/evaluated, best first. Free
    # form per kind; ``selected: True`` marks the one passed downstream.
    candidates: list = field(default_factory=list)
    # ``converged``: True/False, or None when the notion doesn't apply.
    convergence: dict = field(default_factory=lambda: {"converged": None})
    # constraint name -> violation magnitude (>= 0; 0 = satisfied).
    constraint_violations: dict = field(default_factory=dict)
    # ``name`` (+ ``fidelity`` from FIDELITIES, free-form extras).
    physical_model: dict = field(default_factory=dict)
    # error-estimate name (with unit suffix) -> value.
    numerical_error: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    error: str | None = None

    @property
    def converged(self):
        return self.convergence.get("converged")

    def to_dict(self) -> dict:
        doc = {
            "stage": self.stage,
            "status": self.status,
            "elapsed_s": round(float(self.elapsed_s), 6),
            "candidates": _jsonable(self.candidates),
            "convergence": _jsonable({"converged": None, **self.convergence}),
            "constraint_violations": _jsonable(self.constraint_violations),
            "physical_model": _jsonable(self.physical_model),
            "numerical_error": _jsonable(self.numerical_error),
            "notes": [str(n) for n in self.notes],
        }
        if self.error is not None:
            doc["error"] = self.error
        return doc


class StageRecorder:
    """Collects StageResults while a mission builder runs. Builders that are
    called outside the pipeline (e.g. the zero-argument
    compute_and_export_* functions) simply get a fresh recorder nobody reads.

    ``stage()`` is re-entrant per stage name: entering the same stage again
    (e.g. one refine attempt per candidate) accumulates its elapsed time
    into the one StageResult.
    """

    def __init__(self, on_stage: Callable[[StageResult], None] | None = None):
        from orbitopt.candidate_selection import VerificationGate

        self.results = {name: StageResult(name) for name in STAGES}
        self.gate = VerificationGate()
        self._on_stage = on_stage

    def __getitem__(self, name: str) -> StageResult:
        return self.results[name]

    @contextmanager
    def stage(self, name: str):
        result = self.results[name]
        t0 = time.perf_counter()
        try:
            yield result
        except BaseException as exc:
            result.status = STATUS_FAILED
            result.error = f"{type(exc).__name__}: {exc}"
            raise
        else:
            if result.status in (STATUS_NOT_RUN, STATUS_FAILED):
                # A re-entered stage that failed before (e.g. a rejected
                # candidate, retried with the next one) recovers here.
                result.status = STATUS_COMPLETED
                result.error = None
        finally:
            result.elapsed_s += time.perf_counter() - t0
            if self._on_stage is not None:
                self._on_stage(result)

    def add_time(self, name: str, seconds: float) -> StageResult:
        """Attribute separately-measured wall time to a stage (and mark it
        completed), for work that can't be wrapped in ``stage()`` because
        which stage it belongs to is only known afterwards."""
        result = self.results[name]
        result.elapsed_s += float(seconds)
        if result.status == STATUS_NOT_RUN:
            result.status = STATUS_COMPLETED
        return result

    def skip(self, name: str, reason: str) -> StageResult:
        result = self.results[name]
        result.status = STATUS_SKIPPED
        result.notes.append(reason)
        return result


@dataclass
class PipelineResult:
    """Standardized, kind-independent result of one pipeline run."""

    mission: dict
    stages: list
    scene: dict | None = None
    verification_reports: list = field(default_factory=list)
    outputs: dict = field(default_factory=dict)
    error: str | None = None
    gate: object = None  # orbitopt.candidate_selection.VerificationGate

    def __post_init__(self):
        if self.gate is None:
            from orbitopt.candidate_selection import VerificationGate

            self.gate = VerificationGate()

    @property
    def success(self) -> bool:
        """The run completed *and* the exported candidate passed the
        Verification Gate (failed / not_checked / best-effort never count)."""
        return self.status == STATUS_COMPLETED and self.gate.success

    @property
    def warnings(self) -> list:
        return self.gate.warnings()

    @property
    def status(self) -> str:
        return STATUS_FAILED if self.error is not None or any(
            s.status == STATUS_FAILED for s in self.stages) else STATUS_COMPLETED

    @property
    def elapsed_s(self) -> float:
        return float(sum(s.elapsed_s for s in self.stages))

    def stage(self, name: str) -> StageResult:
        for s in self.stages:
            if s.stage == name:
                return s
        raise KeyError(name)

    @property
    def verification_result(self) -> str:
        from orbitopt.verify.report import RESULT_NOT_CHECKED, bundle, status_result

        if not self.verification_reports:
            return RESULT_NOT_CHECKED
        if len(self.verification_reports) == 1:
            return status_result(self.verification_reports[0]["status"])
        return bundle(self.verification_reports, label="pipeline")["status"]["result"]

    def verification_document(self) -> dict | None:
        """One report (or a bundle of several) for writing next to the scene."""
        from orbitopt.verify.report import bundle

        if not self.verification_reports:
            return None
        if len(self.verification_reports) == 1:
            return self.verification_reports[0]
        return bundle(self.verification_reports, label=self.mission.get("id", "mission"))

    def to_dict(self) -> dict:
        doc = {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "pipeline-result",
            "mission": dict(self.mission),
            "status": self.status,
            "success": self.success,
            "elapsed_s": round(self.elapsed_s, 6),
            "stages": [s.to_dict() for s in self.stages],
            "verification": {
                "result": self.verification_result,
                "reports": [f"{r.get('kind', '?')}:{r.get('label', '?')}" for r in self.verification_reports],
            },
            "gate": self.gate.to_dict(),
            "outputs": dict(self.outputs),
        }
        if self.error is not None:
            doc["error"] = self.error
        return doc

    def summary(self) -> str:
        header = (f"Pipeline {self.mission.get('id')} ({self.mission.get('kind')}): "
                  f"{self.status.upper()} in {self.elapsed_s:.1f} s")
        lines = [header]
        for s in self.stages:
            conv = {True: "converged", False: "NOT converged", None: "-"}[s.converged]
            worst = max(s.constraint_violations.values(), default=0.0) or 0.0
            lines.append(f"  {s.stage:<15} {s.status:<10} {s.elapsed_s:8.2f} s  "
                         f"{len(s.candidates):3d} cand.  {conv:<14} max violation {worst:.3g}")
        g = self.gate
        if g.exported is not None:
            lines.append(f"  verification gate: {'SUCCESS' if self.success else 'NOT SUCCESSFUL'} -- exported "
                         f"candidate #{g.exported} ({g.result}); "
                         + ", ".join(f"{n} {v}" for v, n in g.to_dict()["counts"].items() if n))
        lines.extend(f"  WARNING: {w}" for w in self.warnings)
        return "\n".join(lines)


@cache
def schema() -> dict:
    text = resources.files("orbitopt").joinpath("schemas", SCHEMA_FILE).read_text(encoding="utf-8")
    return json.loads(text)


def validate_pipeline_result(doc: dict) -> None:
    """Raise ``jsonschema.ValidationError`` if ``doc`` (a
    ``PipelineResult.to_dict()``) doesn't match pipeline-result-1.0."""
    import jsonschema

    jsonschema.validate(instance=doc, schema=schema())


def write_pipeline_result(result: PipelineResult, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def default_output_path(scene_path, suffix: str) -> str:
    p = Path(scene_path)
    return str(p.with_name(p.stem + suffix))


def _summarize_verify_from_reports(st: StageResult, reports: list) -> None:
    """Generic verify stage contents derived from verification reports (used
    when the builder didn't record its own verify convergence)."""
    from orbitopt.verify.report import RESULT_NOT_CHECKED, RESULT_PASSED, status_result

    results = [status_result(r.get("status") or {}) for r in reports]
    checks = {}
    for r in reports:
        for name, ok in ((r.get("status") or {}).get("checks") or {}).items():
            checks[f"{r.get('kind')}:{r.get('label')}:{name}"] = bool(ok)
        for res in r.get("residuals") or []:
            if res.get("position_residual_m") is not None:
                st.numerical_error.setdefault(
                    f"{r.get('label')}.{res['name']}.position_residual_m", res["position_residual_m"])
    if st.convergence.get("converged") is None and results:
        checked = [res for res in results if res != RESULT_NOT_CHECKED]
        st.convergence["converged"] = bool(checked) and all(res == RESULT_PASSED for res in checked)
    st.convergence.setdefault("report_results", results)
    st.convergence.setdefault("checks_passed", sum(checks.values()))
    st.convergence.setdefault("checks_total", len(checks))


def run_pipeline(
    config: dict,
    *,
    write_outputs: bool = False,
    scene_path=None,
    on_stage: Callable[[StageResult], None] | None = None,
    raise_on_error: bool = True,
) -> PipelineResult:
    """Run a mission config through the full stage lifecycle.

    ``write_outputs`` writes the scene (to ``scene_path`` or the config's
    ``output.scene_file``), its verification report (``output.
    verification_report`` or ``<scene>.verification.json``) and this
    pipeline result (``output.pipeline_result`` or ``<scene>.pipeline.json``).

    ``on_stage`` is called with each StageResult as it finishes (progress
    reporting for the GUI/CLI). With ``raise_on_error=False`` a failure is
    recorded in the result (``status == "failed"``) instead of raised; by
    default the original exception propagates with the partial result
    attached as ``exc.pipeline_result``.
    """
    import importlib

    from orbitopt import mission_config as mc

    recorder = StageRecorder(on_stage=on_stage)
    mission = dict(config.get("mission") or {})
    result = PipelineResult(
        mission={k: mission.get(k) for k in ("id", "title", "kind")},
        stages=[recorder[name] for name in STAGES],
        gate=recorder.gate,
    )
    try:
        with recorder.stage(MISSION_CONFIG) as st:
            mc.validate_mission_config(config)
            st.physical_model = {"name": "configuration", "fidelity": "none"}
            st.notes.append(f"schema {mc._KIND_SCHEMAS[mission['kind']]}")

        module_name, attr = mc._KIND_BUILDERS[mission["kind"]]
        builder = getattr(importlib.import_module(module_name), attr)
        scene = builder(config, recorder=recorder)

        from orbitopt.verify.report import pop_from_scene

        reports = pop_from_scene(scene)
        result.verification_reports = reports
        verify = recorder[VERIFY]
        if reports:
            if verify.status == STATUS_NOT_RUN:
                verify.status = STATUS_COMPLETED
            _summarize_verify_from_reports(verify, reports)
        elif verify.status == STATUS_NOT_RUN:
            recorder.skip(VERIFY, "no verification report produced")
        _apply_gate(recorder.gate, reports, scene)

        with recorder.stage(EXPORT) as st:
            from orbitopt.scene_format import validate_scene

            validate_scene(scene)
            result.scene = scene
            st.physical_model = {"name": "SceneData export", "fidelity": "none"}
            st.candidates = [{"scene_id": scene.get("id"),
                              "bodies": len(scene.get("bodies", [])), "selected": True}]
            result.outputs["scene_id"] = scene.get("id")
            if write_outputs:
                _write_outputs(result, config, scene_path)
    except BaseException as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        if raise_on_error:
            try:
                exc.pipeline_result = result
            except AttributeError:
                pass
            raise
    # Stages a kind doesn't have are explicitly skipped, never left
    # ambiguous -- unless the run failed, in which case later stages are
    # genuinely not_run.
    if result.error is None:
        for st in result.stages:
            if st.status == STATUS_NOT_RUN:
                st.status = STATUS_SKIPPED
                st.notes.append(f"{mission.get('kind')} has no {st.stage} stage")
    if write_outputs and "pipeline_result" in result.outputs:
        write_pipeline_result(result, result.outputs["pipeline_result"])
    return result


def _apply_gate(gate, reports: list, scene: dict) -> None:
    """Close the Verification Gate for the exported scene. A builder that
    drove the gate itself (``gate.exported`` set) is taken as is; otherwise
    the exported scene is the one candidate and its reports decide."""
    from orbitopt.candidate_selection import mark_scene_best_effort
    from orbitopt.verify.report import RESULT_FAILED

    if gate.exported is None:
        c = gate.evaluate(0, reports, source="exported scene")
        gate.export(0, best_effort=c["verification"] == RESULT_FAILED)
    mark_scene_best_effort(scene, gate)


def _write_outputs(result: PipelineResult, config: dict, scene_path) -> None:
    from orbitopt.scene_format import write_scene
    from orbitopt.verify.report import write_report

    output = config.get("output") or {}
    out = str(scene_path or output["scene_file"])
    write_scene(result.scene, out, validate=False)
    result.outputs["scene_file"] = out
    doc = result.verification_document()
    if doc is not None:
        report_out = output.get("verification_report") or default_output_path(out, ".verification.json")
        write_report(doc, report_out)
        result.outputs["verification_report"] = str(report_out)
    result.outputs["pipeline_result"] = str(
        output.get("pipeline_result") or default_output_path(out, ".pipeline.json"))
