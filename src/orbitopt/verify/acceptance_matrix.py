"""V-06 (task #262): cross-backend acceptance matrix.

Epic P1's completion bar is that a *passed* verdict says which evidence
backed it, over which model scope, against which tolerance, for which
stage -- and what it cannot prove. V-02 (:mod:`orbitopt.verify.cross_model`)
produces the numbers (pairwise backend differences); V-03
(:mod:`orbitopt.verify.integrator_convergence`) produces the numerical-error
bounds. This module ties them together into one **configuration file**: the
acceptance matrix *backend x stage x metric x tolerance*, where every cell
carries

* its **stage(s)** -- ``dynamics`` / ``control`` / ``propagation`` /
  ``handoff``, so each phase gets its own convergence line instead of one
  global threshold;
* a **basis** per row (`convergence_analysis` | `physical_reasoning` |
  `measured_agreement` | `ill_conditioned`): the tolerance is anchored to a
  V-03 bound, a physical argument, or a measured spread -- never to taste,
  and a row whose basis says *ill-conditioned* is recorded, never gated;
* a **cause classification** per row (`numerical_integration` |
  `model_formulation` | `attitude_control` | `propellant_accounting` |
  `state_handoff` | `frame_bridge`): when a row fails, the report says which
  kind of problem a violation indicates and what to look at first; rows that
  could not run are classified too (``backend_environment``,
  ``backend_runtime_error``, ``backend_no_track``,
  ``backend_not_requested``, ``channel_unavailable``, ``no_matrix_cell``).

The matrix is data, not constants: the shipped default is
``orbitopt/data/acceptance_matrix.yaml`` (schema
``schemas/acceptance-matrix-1.0.json``), overridden per run with
``load_matrix(path)`` or ``--matrix``. The loader enforces what the
acceptance asks for: every cell declares a row for **every** harness metric,
every row has a basis, every gated row names its failure cause, and
``stage_budgets`` records the convergence study / measured spread the lines
were built on (a test ties the copied numbers back to their sources, so the
anchors cannot silently drift).

Gate semantics are V-02's / task #243's, unchanged: a row without a resolved
limit is *recorded*, a pair without a matrix cell is *unmatriculated*
(recorded, never gated), and evidence that did not run is ``not_checked``
with its cause code -- never a pass and never a fake failure. The run's
``acceptance-matrix`` verification report has one residual per matrix row
that the harness measured; feeding it to the pipeline gate grades a candidate
exactly like any other report::

    gate.register(0, scenario="acceptance-matrix")
    gate.evaluate(0, [res.report])

    from orbitopt.verify.acceptance_matrix import run_acceptance_matrix
    res = run_acceptance_matrix(thresholds={"ascent.position_m": 1000})
    print(res.result)                     # passed | failed | not_checked

Run from the command line (exit code 1 if a gated row fails; a cause-class
summary prints with the table)::

    python -m orbitopt.verify.acceptance_matrix [--matrix PATH]
        [--backends ...] [--threshold KEY=VALUE] [--json]
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping, Sequence

from orbitopt.verify.cross_model import (
    MIN_ECCENTRICITY_FOR_ANGLES,
    MIN_INCLINATION_FOR_ANGLES_DEG,
    STATUS_FAILED,
    STATUS_UNAVAILABLE,
    THRESHOLD_METRICS,
    CrossModelScenario,
    CrossModelVerification,
    parse_thresholds,
    run_cross_model,
)
from orbitopt.verify.report import (
    REPORT_DIR_ENV,
    RESULT_FAILED,
    RESULT_NOT_CHECKED,
    build_report,
    default_report_path,
    finalize_report,
    format_status,
    status_result,
)

__all__ = [
    "AcceptanceMatrix",
    "AcceptanceMatrixVerification",
    "FAILURE_CLASSES",
    "MatrixCell",
    "MatrixRow",
    "FailureClass",
    "RowOutcome",
    "default_matrix",
    "evaluate_matrix",
    "load_matrix",
    "matrix_schema_path",
    "run_acceptance_matrix",
    "validate_matrix",
]

#: Result vocabulary of one matrix row (finer than the report tri-state:
#: "recorded" and "unmatriculated" are explicitly *not* passes).
ROW_PASSED = "passed"
ROW_FAILED = "failed"
ROW_RECORDED = "recorded"
ROW_NOT_CHECKED = "not_checked"
ROW_UNMATRICULATED = "unmatriculated"
ROW_RESULTS = (ROW_PASSED, ROW_FAILED, ROW_RECORDED, ROW_NOT_CHECKED, ROW_UNMATRICULATED)

ROW_NOT_RUN = (ROW_NOT_CHECKED, ROW_UNMATRICULATED)


# ---------------------------------------------------------------------------
# Failure cause catalog (任务验收: 不满足项给出原因分类)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FailureClass:
    """One cause classification a not-satisfied matrix row can carry:
    ``category`` groups the cause for the report rollup, ``next_step`` says
    what to look at first."""

    code: str
    category: str
    title: str
    explanation: str
    next_step: str
    #: ``violated``: the row ran and exceeded its tolerance; ``missing``:
    #: the row could not be judged (absent evidence is never a pass).
    severity: str = "violated"

    def text(self) -> str:
        return f"{self.title} — {self.explanation} Next: {self.next_step}"


FAILURE_CLASSES: dict[str, FailureClass] = {c.code: c for c in (
    FailureClass(
        "numerical_integration", "numerics",
        "Same-model pair exceeds its numerical tolerance",
        "Both backends fly the same force model, so the difference is "
        "integration / resampling numerics, not physics.",
        "Re-run the V-03 convergence study for the affected scenario and "
        "compare against the budget this row's basis cites."),
    FailureClass(
        "model_formulation", "physics",
        "Independent models disagree beyond the formulation budget",
        "The pair compares different force / vehicle models and the residual "
        "exceeded the line derived from the measured or argued model spread.",
        "Check the scenario's declared force models and vehicle numbers on "
        "both sides before touching the tolerance."),
    FailureClass(
        "attitude_control", "control",
        "Closed-loop 6-DOF tracking exceeds the control tolerance",
        "The ascent state difference is dominated by the attitude controller "
        "tracking the reference program, not by propagation.",
        "Inspect the FL-403 PID tracking margins (gimbal lag, pitch-over "
        "kick) and the reference-program resampling."),
    FailureClass(
        "propellant_accounting", "plumbing",
        "Mass bookkeeping disagrees between backends",
        "Both models burn the same total propellant; a mass difference far "
        "above the integration-lag bound means a broken mass model path.",
        "Check tank drain rates, staging mass and the mass channel of each "
        "backend's output."),
    FailureClass(
        "state_handoff", "plumbing",
        "Shared handoff state does not survive the copy",
        "Coast backends re-use the identical handoff constant, so any "
        "difference above float round-off is plumbing.",
        "Check the MECO handoff construction and the mass/state plumbing "
        "in orbitopt.verify.cross_model and the mission contract."),
    FailureClass(
        "frame_bridge", "convention",
        "Frame / epoch convention leak between implementations",
        "Independent propagators of one declared force model should agree "
        "far below this line; metre-per-second scale differences are the "
        "signature of a frame or epoch bridging bug.",
        "Check the GCRS/ECEF bridge, the scenario epoch and the GM/radius "
        "triple each backend used."),
    FailureClass(
        "backend_environment", "environment",
        "Backend not installed in this environment",
        "The comparison could not run because a required backend "
        "(e.g. jsbsim, tudatpy) is missing.",
        "Install the optional extra (envs/sixdof-requirements.txt or the "
        "conda tudatpy stack), or run the matrix over the installed subset."),
    FailureClass(
        "backend_runtime_error", "execution",
        "Backend raised while flying the scenario",
        "The backend was available but failed during the run; the harness "
        "recorded its error and continued.",
        "Read the backend's detail line in the report inputs and fix the "
        "reported error."),
    FailureClass(
        "backend_no_track", "execution",
        "Backend ran but produced no track for this phase",
        "The pair has no shared trajectory window because one side did not "
        "produce a track for the cell's phase.",
        "Check the backend producer: it must return a track for every phase "
        "the matrix gates for it."),
    FailureClass(
        "backend_not_requested", "execution",
        "Backend not part of this run's backend list",
        "The cell needs a backend that this run did not ask for.",
        "Add the backend to --backends, or accept the row as not_checked."),
    FailureClass(
        "channel_unavailable", "data",
        "Metric channel not produced by both backends",
        "The pair flew but one side does not report this channel (e.g. one "
        "backend has no mass history).",
        "Record the row without a gate in the matrix, or extend the backend "
        "to produce the channel."),
    FailureClass(
        "ungated_override", "matrix",
        "Row gate switched off by a threshold override",
        "The operator un-gated this row (threshold value 'none'); the metric "
        "is still measured and recorded, it just cannot vote.",
        "Decide whether the row should carry a justified limit again; never "
        "ship an acceptance line away for good."),
    FailureClass(
        "no_matrix_cell", "matrix",
        "Backend pair has no matrix cell",
        "The harness compared a pair the loaded matrix does not cover, so "
        "there is no justified tolerance for it and it is never gated.",
        "Add a cell (with a basis for every metric) to the matrix config."),
)}
"""Cause classifications for not-satisfied rows, keyed by code."""


# ---------------------------------------------------------------------------
# Matrix configuration
# ---------------------------------------------------------------------------

MATRIX_SCHEMA_FILE = "acceptance-matrix-1.0.json"
DEFAULT_MATRIX_RESOURCE = ("data", "acceptance_matrix.yaml")


def matrix_schema_path() -> Path:
    """Path of the packaged JSON Schema for matrix configuration files."""
    return Path(__file__).resolve().parent.parent / "schemas" / MATRIX_SCHEMA_FILE


def default_matrix_path() -> Path:
    """Path of the shipped default matrix (``orbitopt/data/acceptance_matrix.yaml``)."""
    return Path(__file__).resolve().parent.parent / DEFAULT_MATRIX_RESOURCE[0] \
        / DEFAULT_MATRIX_RESOURCE[1]


@dataclass(frozen=True)
class MatrixRow:
    """One (cell, metric) comparison item: the tolerance (``limit``; ``None``
    records without gating), its basis, and the failure cause a violation
    carries (``on_fail``; required exactly when a limit is set)."""

    metric: str
    limit: float | None
    basis: str
    basis_kind: str
    on_fail: str | None = None

    def to_dict(self) -> dict:
        return {"metric": self.metric, "limit": self.limit, "basis_kind": self.basis_kind,
                "basis": self.basis, "on_fail": self.on_fail}


@dataclass(frozen=True)
class MatrixCell:
    """One backend pair and phase: the stages it guards, the evidence
    category, the model scope, what a pass does *not* prove, and the rows."""

    pair: str
    phase: str
    stages: tuple[str, ...]
    category: str
    scope: str
    not_proven: str
    rows: Mapping[str, MatrixRow]

    def to_dict(self) -> dict:
        return {"pair": self.pair, "phase": self.phase, "stages": list(self.stages),
                "category": self.category, "scope": self.scope,
                "not_proven": self.not_proven,
                "tolerances": {m: self.rows[m].to_dict() for m in THRESHOLD_METRICS}}


@dataclass(frozen=True)
class AcceptanceMatrix:
    """The acceptance matrix: backend x stage x metric x tolerance, with a
    basis for every row and a cause class for every violation."""

    name: str
    stages: tuple[str, ...]
    cells: tuple[MatrixCell, ...]
    description: str = ""
    harness: str = "orbitopt.verify.cross_model"
    stage_budgets: Mapping = field(default_factory=dict)
    config_digest: str = ""

    def cell(self, pair: str, phase: str) -> MatrixCell | None:
        for c in self.cells:
            if c.pair == pair and c.phase == phase:
                return c
        return None

    def to_dict(self) -> dict:
        return {"schemaVersion": "1.0", "name": self.name, "description": self.description,
                "harness": self.harness, "stages": list(self.stages),
                "stage_budgets": json.loads(json.dumps(self.stage_budgets)),
                "config_digest": self.config_digest,
                "cells": [c.to_dict() for c in self.cells]}

    def harness_thresholds(self, overrides: Mapping | None = None) -> dict:
        """Threshold overrides for the V-02 harness: every matrix row as a
        full ``<pair>.<phase>.<metric>`` key with its *resolved* limit (user
        overrides applied), plus any user key that addresses a pair the
        matrix does not cover. Injecting resolved full-form keys is what
        makes it impossible for the cross-model report and this matrix to
        disagree about one number."""
        out: dict[str, float | None] = {}
        for c in self.cells:
            for m, row in c.rows.items():
                out[f"{c.pair}.{c.phase}.{m}"] = resolve_limit(overrides, c, row)
        for key, value in (overrides or {}).items():
            out.setdefault(key, value)
        return out


def resolve_limit(overrides: Mapping | None, cell: "MatrixCell", row: "MatrixRow") -> float | None:
    """The limit one row is graded with: a user ``--threshold`` / ``thresholds``
    override wins in the V-02 precedence (``<pair>.<phase>.<metric>`` >
    ``<phase>.<metric>`` > ``<metric>``), otherwise the matrix's own limit."""
    for key in (f"{cell.pair}.{cell.phase}.{row.metric}", f"{cell.phase}.{row.metric}",
                row.metric):
        if overrides is not None and key in overrides:
            return overrides[key]
    return row.limit


def validate_matrix(document: Mapping) -> dict:
    """Validate a matrix configuration document against the packaged schema
    plus the cross-checks the schema cannot express (known phases/metrics,
    duplicate cells, positive limits). Raises on the first problem; returns
    the document as a plain dict."""
    import jsonschema

    try:
        jsonschema.validate(instance=json.loads(json.dumps(dict(document))),
                            schema=json.loads(matrix_schema_path().read_text(encoding="utf-8")))
    except jsonschema.ValidationError as exc:
        raise ValueError(f"matrix config rejected by {MATRIX_SCHEMA_FILE}: {exc.message}") from None
    cells = []
    seen: set[tuple] = set()
    for raw in document["cells"]:
        key = (raw["pair"], raw["phase"])
        if key in seen:
            raise ValueError(f"matrix {document['name']}: duplicate cell {key[0]} @ {key[1]}")
        seen.add(key)
        labels = raw["pair"].split("_vs_")
        if len(labels) != 2 or any(not lbl for lbl in labels):
            raise ValueError(f"matrix cell {raw['pair']!r}: pair must be <label>_vs_<label>")
        for metric, row in raw["tolerances"].items():
            limit = row.get("limit")
            if limit is not None and (not math.isfinite(float(limit)) or float(limit) <= 0.0):
                raise ValueError(f"matrix {key[0]} @ {key[1]} {metric}: limit must be > 0 or null")
            if limit is not None and row["basis_kind"] == "ill_conditioned":
                raise ValueError(f"matrix {key[0]} @ {key[1]} {metric}: an ill-conditioned row "
                                 "must not carry a gate (limit: null)")
            if limit is None and row["basis_kind"] != "ill_conditioned":
                raise ValueError(f"matrix {key[0]} @ {key[1]} {metric}: a row without a limit "
                                 "must say why (basis_kind: ill_conditioned)")
            if row.get("on_fail") is not None and row["on_fail"] not in FAILURE_CLASSES:
                raise ValueError(f"matrix {key[0]} @ {key[1]} {metric}: unknown failure class "
                                 f"{row['on_fail']!r}")
        cells.append(raw)
    unknown = sorted({s for c in document["cells"] for s in c["stages"]}
                     - set(document["stages"]))
    if unknown:
        raise ValueError(f"matrix uses stages outside its vocabulary: {unknown}")
    return json.loads(json.dumps(dict(document)))


def load_matrix(path=None) -> AcceptanceMatrix:
    """Load a matrix configuration from a YAML/JSON file (``None`` = the
    shipped default). Validated by :func:`validate_matrix`."""
    import yaml

    source = default_matrix_path() if path is None else Path(path)
    document = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{source}: expected a mapping at the top level")
    return _matrix_from_document(validate_matrix(document))


def default_matrix() -> AcceptanceMatrix:
    """The shipped default matrix (no arguments)."""
    return load_matrix(None)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RowOutcome:
    """One evaluated matrix row: the measured value (or why nothing was
    measured), the verdict and, when the row is not satisfied, its cause
    classification."""

    pair: str
    phase: str
    stages: tuple[str, ...]
    category: str
    metric: str
    limit: float | None
    value: float | None
    max_value: float | None
    result: str
    failure_class: str | None = None
    basis: str = ""
    basis_kind: str = ""

    @property
    def satisfied(self) -> bool:
        return self.result == ROW_PASSED

    def to_dict(self) -> dict:
        return {"pair": self.pair, "phase": self.phase, "stages": list(self.stages),
                "category": self.category, "metric": self.metric, "limit": self.limit,
                "value": self.value, "max_value": self.max_value, "result": self.result,
                "failure_class": self.failure_class, "basis_kind": self.basis_kind,
                "basis": self.basis}


_MAX_KEYS = {"position_m": "max_position_m", "velocity_m_s": "max_velocity_m_s",
             "mass_kg": "max_mass_kg"}


def _row_value(metric: str, metrics: Mapping) -> float | None:
    if metric == "position_m":
        return abs(float(metrics["final_position_m"]))
    if metric == "velocity_m_s":
        return abs(float(metrics["final_velocity_m_s"]))
    if metric == "mass_kg":
        v = metrics.get("final_mass_kg")
        return None if v is None else abs(float(v))
    return abs(float(metrics["elements"][metric]))


def _angles_gated(metrics: Mapping, metric: str) -> bool:
    """The V-02 conditioning rule: RAAN/argP are only judgable on a plane,
    well-conditioned orbit (recorded regardless; task #243)."""
    if metric not in ("raan_deg", "argument_of_perigee_deg"):
        return True
    incl_a, incl_b = metrics["final_inclination_deg"]
    if min(float(incl_a), float(incl_b)) <= MIN_INCLINATION_FOR_ANGLES_DEG:
        return False
    if metric == "argument_of_perigee_deg":
        ecc_a, ecc_b = metrics.get("final_eccentricity", [0.0, 0.0])
        return min(float(ecc_a), float(ecc_b)) > MIN_ECCENTRICITY_FOR_ANGLES
    return True


def _missing_pair_cause(cell: MatrixCell, statuses: Mapping) -> FailureClass:
    """Why a cell could not be evaluated, classified from the harness's own
    per-backend outcomes (never guessed): not requested > not installed >
    raised > no track."""
    labels = cell.pair.split("_vs_", 1)
    outcomes = [statuses.get(label) for label in labels]
    if any(o is None for o in outcomes):
        return FAILURE_CLASSES["backend_not_requested"]
    if any(o.status == STATUS_UNAVAILABLE for o in outcomes):
        return FAILURE_CLASSES["backend_environment"]
    if any(o.status == STATUS_FAILED for o in outcomes):
        return FAILURE_CLASSES["backend_runtime_error"]
    return FAILURE_CLASSES["backend_no_track"]


def evaluate_matrix(matrix: AcceptanceMatrix, harness: CrossModelVerification,
                    overrides: Mapping | None = None) -> list:
    """Grade one V-02 harness run against ``matrix``. ``overrides`` is a
    validated dotted threshold mapping (``<metric>`` / ``<phase>.<metric>`` /
    ``<pair>.<phase>.<metric>``) that wins over the matrix limits. Returns
    the row outcomes sorted by (pair, phase, metric): every matrix cell,
    whether or not its pair flew, plus one unmatriculated row per harness
    pair the matrix does not cover."""
    user = dict(overrides or {})
    statuses = {o.label: o for o in harness.backends}
    by_pair = {(p["pair"], p["phase"]): p for p in harness.pair_metrics}

    rows: list[RowOutcome] = []
    for cell in matrix.cells:
        pair = by_pair.get((cell.pair, cell.phase))
        cause: FailureClass | None = None
        if pair is None:
            cause = _missing_pair_cause(cell, statuses)
        elif pair["category"] != cell.category:
            raise ValueError(
                f"matrix cell {cell.pair} @ {cell.phase} declares category "
                f"{cell.category!r} but the harness categorises this pair "
                f"{pair['category']!r} -- fix the matrix, not the harness")
        for metric in THRESHOLD_METRICS:
            row = cell.rows[metric]
            limit = resolve_limit(user, cell, row)
            common = dict(pair=cell.pair, phase=cell.phase, stages=cell.stages,
                          category=cell.category, metric=metric, limit=limit,
                          basis=row.basis, basis_kind=row.basis_kind)
            if pair is None:
                rows.append(RowOutcome(**common, value=None, max_value=None,
                                       result=ROW_NOT_CHECKED, failure_class=cause.code))
                continue
            value = _row_value(metric, pair["metrics"])
            if value is None:
                rows.append(RowOutcome(**common, value=None, max_value=None,
                                       result=ROW_NOT_CHECKED,
                                       failure_class="channel_unavailable"))
                continue
            max_key = _MAX_KEYS.get(metric)
            max_value = None if max_key is None else pair["metrics"].get(max_key)
            if limit is None:
                rows.append(RowOutcome(**common, value=value, max_value=max_value,
                                       result=ROW_RECORDED,
                                       failure_class=None if row.basis_kind == "ill_conditioned"
                                       else "ungated_override"))
                continue
            if not _angles_gated(pair["metrics"], metric):
                rows.append(RowOutcome(**common, value=value, max_value=max_value,
                                       result=ROW_RECORDED, failure_class=None))
                continue
            failed = value > limit
            rows.append(RowOutcome(**common, value=value, max_value=max_value,
                                   result=ROW_FAILED if failed else ROW_PASSED,
                                   failure_class=row.on_fail if failed else None))

    covered = {(c.pair, c.phase) for c in matrix.cells}
    for p in harness.pair_metrics:
        if (p["pair"], p["phase"]) in covered:
            continue
        for metric in THRESHOLD_METRICS:
            value = _row_value(metric, p["metrics"])
            rows.append(RowOutcome(pair=p["pair"], phase=p["phase"], stages=(),
                                   category=p["category"], metric=metric, limit=None,
                                   value=value,
                                   max_value=(None if value is None else
                                              p["metrics"].get(_MAX_KEYS.get(metric, ""), None)),
                                   result=ROW_UNMATRICULATED, failure_class="no_matrix_cell",
                                   basis="no matrix cell for this pair; recorded, never gated",
                                   basis_kind="no_cell"))
    rows.sort(key=lambda r: (r.pair, r.phase, THRESHOLD_METRICS.index(r.metric)))
    return rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _residual_from_row(row: RowOutcome, matrix: AcceptanceMatrix) -> dict:
    """One verification-report residual per evaluated matrix row: value,
    tolerance, basis and (for a not-satisfied row) its cause classification.
    Only gated rows carry ``within_tolerance`` -- recorded / not_checked /
    unmatriculated rows are visible but never vote in the status."""
    cell = matrix.cell(row.pair, row.phase)
    residual = {
        "name": f"{row.pair}.{row.phase}.{row.metric}",
        "reference": (f"acceptance-matrix {matrix.name}: {row.pair.replace('_vs_', ' vs ')} "
                      f"@ {row.phase} [{', '.join(row.stages)}]"),
        "category": row.category,
        "position_residual_m": row.value if row.metric == "position_m" else None,
        "velocity_residual_m_s": row.value if row.metric == "velocity_m_s" else None,
        "stage": list(row.stages),
        "basis": row.basis,
        "basis_kind": row.basis_kind,
        "matrix_result": row.result,
    }
    if row.limit is not None:
        residual["tolerance"] = {row.metric: row.limit}
    if row.value is not None:
        residual["observed"] = {row.metric: row.value}
    if row.max_value is not None:
        residual["max_along_arc"] = {row.metric: row.max_value}
    if row.result == ROW_PASSED:
        residual["within_tolerance"] = True
    elif row.result == ROW_FAILED:
        residual["within_tolerance"] = False
    if row.failure_class is not None:
        cause = FAILURE_CLASSES[row.failure_class]
        residual["failure_class"] = cause.code
        residual["failure_category"] = cause.category
        residual["failure_severity"] = cause.severity
        residual["failure_detail"] = cause.text()
    if cell is not None:
        residual["scope"] = cell.scope
        residual["not_proven"] = cell.not_proven
    return residual


def _summary(rows: Sequence[RowOutcome]) -> dict:
    """Rollups the acceptance review reads first: per result, per cause and
    per stage."""
    by_result = {k: 0 for k in ROW_RESULTS}
    by_cause: dict[str, int] = {}
    by_stage = {s: {k: 0 for k in ROW_RESULTS} for s in ("dynamics", "control", "propagation",
                                                         "handoff")}
    for r in rows:
        by_result[r.result] += 1
        if r.failure_class:
            by_cause[r.failure_class] = by_cause.get(r.failure_class, 0) + 1
        for s in r.stages:
            if s in by_stage:
                by_stage[s][r.result] += 1
    return {"rows": len(rows), "by_result": by_result, "by_failure_class": by_cause,
            "by_stage": by_stage}


@dataclass
class AcceptanceMatrixVerification:
    """One matrix run: the configuration, the harness outcome it graded, the
    row outcomes and the digested ``acceptance-matrix`` report."""

    matrix: AcceptanceMatrix
    harness: CrossModelVerification
    rows: tuple[RowOutcome, ...]
    report: dict
    report_path: object = None

    @property
    def result(self) -> str:
        return status_result(self.report["status"])

    def failed_rows(self) -> list:
        return [r for r in self.rows if r.result in (ROW_FAILED,) + ROW_NOT_RUN]

    def to_dict(self) -> dict:
        return {"matrix": self.matrix.to_dict(),
                "rows": [r.to_dict() for r in self.rows],
                "summary": _summary(self.rows),
                "status": self.report["status"],
                "report_file": str(self.report_path) if self.report_path else None}


def run_acceptance_matrix(matrix: "AcceptanceMatrix | Mapping | str | Path | None" = None,
                          scenario: CrossModelScenario | None = None,
                          backends: Sequence[str] | None = None,
                          thresholds: Mapping | None = None,
                          report_path=None) -> AcceptanceMatrixVerification:
    """Fly the V-02 scenario on the requested backends and grade it against
    the acceptance matrix. ``matrix`` accepts a loaded
    :class:`AcceptanceMatrix`, a path (str/Path) or ``None`` (the shipped
    default); ``thresholds`` is a dotted override mapping that wins over the
    matrix limits and is also injected into the harness, so the two reports
    can never disagree. The resulting report honours
    ``$ORBITOPT_VERIFICATION_REPORT_DIR`` like every other verify entry
    point and grades through ``VerificationGate.evaluate``."""
    if isinstance(matrix, AcceptanceMatrix):
        mat = matrix
    elif isinstance(matrix, (str, Path)):
        mat = load_matrix(matrix)
    elif isinstance(matrix, Mapping):
        mat = load_matrix_from_dict(matrix)
    else:
        mat = default_matrix()
    overrides = parse_thresholds(thresholds) if thresholds else {}
    harness = run_cross_model(scenario, backends=backends,
                              thresholds=mat.harness_thresholds(overrides))
    rows = evaluate_matrix(mat, harness, overrides)
    residuals = [_residual_from_row(r, mat) for r in rows]
    report = build_report(
        "acceptance-matrix", mat.name, propagations=[], residuals=residuals,
        inputs={"matrix": mat.to_dict(),
                "threshold_overrides": {k: v for k, v in sorted(overrides.items())},
                "backends": [o.to_dict() for o in harness.backends],
                "scenario": scenario.to_dict() if scenario is not None
                else harness.scenario.to_dict(),
                "summary": _summary(rows),
                "matrix_result": {f"{r.pair}.{r.phase}.{r.metric}": r.result for r in rows}})
    report = finalize_report(report, report_path)
    path = report_path
    if path is None:
        import os

        if os.environ.get(REPORT_DIR_ENV):
            path = default_report_path(report, os.environ[REPORT_DIR_ENV])
    return AcceptanceMatrixVerification(matrix=mat, harness=harness, rows=tuple(rows),
                                        report=report, report_path=path)


def load_matrix_from_dict(document: Mapping) -> AcceptanceMatrix:
    """Validate and build a matrix from an in-memory document (same rules as
    :func:`load_matrix`)."""
    return _matrix_from_document(validate_matrix(dict(document)))


def _matrix_from_document(document: dict) -> AcceptanceMatrix:
    digest = hashlib.sha256(
        json.dumps(document, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()
    cells = tuple(
        MatrixCell(pair=c["pair"], phase=c["phase"], stages=tuple(c["stages"]),
                   category=c["category"], scope=c["scope"], not_proven=c["not_proven"],
                   rows={m: MatrixRow(metric=m,
                                      limit=(None if r.get("limit") is None else float(r["limit"])),
                                      basis=r["basis"], basis_kind=r["basis_kind"],
                                      on_fail=r.get("on_fail"))
                          for m, r in sorted(c["tolerances"].items())})
        for c in sorted(document["cells"], key=lambda c: (c["pair"], c["phase"])))
    return AcceptanceMatrix(name=str(document["name"]), stages=tuple(document["stages"]),
                            cells=cells, description=str(document.get("description", "")).strip(),
                            harness=str(document.get("harness", "orbitopt.verify.cross_model")),
                            stage_budgets=dict(document.get("stage_budgets") or {}),
                            config_digest=digest)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    """Run the acceptance matrix; print the table with each unsatisfied
    row's cause code; exit 1 if a gated row failed (not_checked exits 0 with
    its cause listed: absence of evidence is neither pass nor failure)."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m orbitopt.verify.acceptance_matrix",
        description="V-06: grade the V-02 cross-backend harness against the configurable "
                    "acceptance matrix (backend x stage x metric x tolerance, every row "
                    "with a basis, every violation with a cause class).")
    parser.add_argument("--matrix", default=None, metavar="PATH",
                        help="matrix config (YAML/JSON); default: the shipped "
                             "orbitopt/data/acceptance_matrix.yaml")
    parser.add_argument("--backends", default=None,
                        help="comma list passed to the V-02 harness (default: its full set)")
    parser.add_argument("--threshold", action="append", default=[], metavar="KEY=VALUE",
                        help="override one row: KEY is <metric>, <phase>.<metric> or "
                             "<pair>.<phase>.<metric>; VALUE 'none' records without gating")
    parser.add_argument("--coast-duration", type=float, default=None, metavar="S",
                        help="override the harness scenario coast duration")
    parser.add_argument("-o", "--output", default=None, metavar="PATH",
                        help="report path (default: $ORBITOPT_VERIFICATION_REPORT_DIR)")
    parser.add_argument("--json", action="store_true",
                        help="print the acceptance-matrix verification report instead of the table")
    args = parser.parse_args(argv)

    overrides = {}
    for item in args.threshold:
        key, _, value = item.partition("=")
        overrides[key] = value
    scenario = None
    if args.coast_duration is not None:
        scenario = replace(CrossModelScenario(), coast_duration_s=float(args.coast_duration))
    res = run_acceptance_matrix(args.matrix, scenario=scenario,
                                backends=None if args.backends is None else args.backends.split(","),
                                thresholds=overrides, report_path=args.output)
    if args.json:
        print(json.dumps(res.report, indent=2, sort_keys=True))
        return 1 if res.result == RESULT_FAILED else 0

    for o in res.harness.backends:
        line = f"[{o.status:>11}] {o.label}"
        if o.detail:
            line += f" -- {o.detail}"
        print(line)
    print()
    header = (f"{'result':>14}  {'stages':<22} {'cell':<46} {'metric':<24} "
              f"{'value':>12}  {'limit':>12}  basis / cause")
    print(header)
    for r in res.rows:
        value = "n/a" if r.value is None else f"{r.value:.3g}"
        limit = "--" if r.limit is None else f"{r.limit:.3g}"
        tail = r.basis_kind + (f" [{r.failure_class}]" if r.failure_class else "")
        print(f"{r.result:>14}  {','.join(r.stages):<22} {r.pair + ' @ ' + r.phase:<46} "
              f"{r.metric:<24} {value:>12}  {limit:>12}  {tail}")
    summary = _summary(res.rows)
    print()
    for code, count in sorted(summary["by_failure_class"].items()):
        print(f"  cause {code:<24} {FAILURE_CLASSES[code].title}  x{count}")
    print(f"matrix {res.matrix.name} ({res.matrix.config_digest[:12]}): "
          f"{format_status(res.report['status'])}"
          + (f" -> {res.report_path}" if res.report_path else ""))
    if res.result == RESULT_NOT_CHECKED:
        print("NOT CHECKED: no matrix row produced gateable evidence (see cause lines above) "
              "-- absence of evidence is not a pass.")
    return 1 if res.result == RESULT_FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
