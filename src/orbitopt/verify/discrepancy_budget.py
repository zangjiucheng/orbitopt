"""E8-02 (task #282): model discrepancy budget -- decompose, don't just total.

V-02 (:mod:`orbitopt.verify.cross_model`) reports the *total* pairwise
difference between backends; V-06 (:mod:`orbitopt.verify.acceptance_matrix`)
gates that total against justified tolerances. A total alone cannot say what
to fix. This module takes every comparison the harness measured and splits
it into a per-source **error budget** over five sources:

``frame_time``
    Frame / epoch convention differences. Every V-02 backend shares one
    declared epoch and one GCRS bridge, so the *applied* desync is exactly
    zero; the line records the bridge-desync sensitivity at the IERS
    convention bound ``|UT1-UTC| < 0.9 s`` as the budget for the case where
    an independent backend ever rotates on its own time convention
    (``frame_bridge`` in the matrix's cause vocabulary).
``numerical_integration``
    Measured by a tolerance-relaxation probe (the 3-DOF reference re-run at
    rtol/atol x 1000); the JSBSim fixed-step side is carried by the
    published E6-04 step-convergence study (RK4 truncation at dt = 0.01 s,
    ~2.5e-8 m -- docs/benchmarks/e604_step_convergence.json).
``engine``
    Based on the E6-01 (#273) *aligned* propulsion law: the perturbation is
    the live parity residual between the two implementations of that one law
    (``parity-stage1-thrust-vs-jsbsim``, currently ~3e-8 relative), applied
    as a thrust-scale probe on the reference ascent -- not a guessed engine
    margin; a failed parity check is refused, not budgeted. The mass channel
    carries the disclosed one-frame bookkeeping lag ``mdot * dt`` as its
    bound (ADR flightlab-0009).
``aero``
    Zero by declaration for the zero-aero V-02 scenario (asserted by the
    ``parity-jsbsim-aero-zero`` parity check); the line additionally records
    the *potential* -- what the same closed-loop flight differs by with the
    production ``spike_stage_aero`` coefficients in -- so switching aero on
    cannot silently inherit the zero. Potential is recorded, not summed.
``controller``
    Measured by a loop-stiffening probe: the 6-DOF plant re-flown with 2x
    PID gains; the share of the pair difference a stiffer loop removes is
    attributed to the attitude controller. The rest stays in the residual,
    honestly.

Every line records how its number was obtained (``status`` / ``method``) and
its probe payload; the component sum and the **residual** (total minus the
sum) are recorded alongside, never hidden. A large residual is a statement
that the disagreement is dominated by plant *formulation* (rigid-body vs
point-mass, FCS/actuator dynamics, atmosphere tables) -- exactly what an
unsplit total could not say. A negative residual means the recorded bounds
already cover the total (conservative), also honest.

The acceptance matrix is *reused*, not duplicated: the harness runs with the
matrix's resolved tolerances, ``evaluate_matrix`` grades the totals, and each
budget residual inherits ``within_tolerance`` from its matrix row -- so this
report's status is the matrix's status for the evidence that flew, while the
decomposition itself is recorded, never gate-fudged. A backend that did not
fly yields ``not_measured`` lines carrying the matrix vocabulary's cause
code (plus two budget-only codes, ``probes_disabled`` and
``no_decomposable_recipe``) -- never a fake zero, never a pass.

    from orbitopt.verify.discrepancy_budget import run_discrepancy_budget
    res = run_discrepancy_budget(backends=["three_dof", "six_dof"])
    for b in res.budgets:
        print(b.pair, b.phase, b.channel, b.total, b.components, b.residual)

Run from the command line (exit code 1 if a gated matrix row failed;
``--no-probes`` records the structure with every probe line ``not_measured``
and the full total in the residual):

    python -m orbitopt.verify.discrepancy_budget [--matrix PATH]
        [--backends ...] [--threshold KEY=VALUE] [--no-probes] [--json]
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, replace
from datetime import timedelta
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from orbitopt.verify.acceptance_matrix import (
    FAILURE_CLASSES,
    ROW_FAILED,
    ROW_NOT_CHECKED,
    ROW_PASSED,
    ROW_UNMATRICULATED,
    AcceptanceMatrix,
    RowOutcome,
    _missing_pair_cause,
    _row_value,
    default_matrix,
    evaluate_matrix,
    load_matrix,
    load_matrix_from_dict,
)
from orbitopt.verify.cross_model import (
    STATUS_RAN,
    CrossModelScenario,
    CrossModelVerification,
    ModelTrack,
    compare_tracks,
    parse_thresholds,
    run_cross_model,
    six_dof_ascent,
    three_dof_ascent,
)
from orbitopt.verify.report import (
    REPORT_DIR_ENV,
    build_report,
    default_report_path,
    finalize_report,
    format_status,
    status_result,
)

__all__ = [
    "BUDGET_CHANNELS",
    "BUDGET_SOURCES",
    "ChannelBudget",
    "DiscrepancyBudgetVerification",
    "SourceLine",
    "run_discrepancy_budget",
]

#: The five discrepancy sources the task demands a line for, in report order.
BUDGET_SOURCES = ("frame_time", "numerical_integration", "engine", "aero",
                  "controller")

#: Channels a pair difference is budgeted over -- the gateable metric names
#: of the V-02 threshold space, so a budget line's matrix row lookup is the
#: matrix's own row lookup.
BUDGET_CHANNELS = ("position_m", "velocity_m_s", "mass_kg")
CHANNEL_UNIT = {"position_m": "m", "velocity_m_s": "m/s", "mass_kg": "kg"}

#: Which acceptance-matrix failure class a source's violation would be
#: classified under -- the budget speaks the matrix's cause vocabulary.
SOURCE_FAILURE_CLASS = {
    "frame_time": "frame_bridge",
    "numerical_integration": "numerical_integration",
    "engine": "model_formulation",
    "aero": "model_formulation",
    "controller": "attitude_control",
}

# Line statuses: how the number on a line was obtained.
STATUS_MEASURED = "measured"              # a probe measured this share
STATUS_MEASURED_TOTAL = "measured_total"  # the whole total is this source
STATUS_PROBE = "probe"                    # directional probe (may be partial)
STATUS_BOUND = "bound"                    # conservative recorded bound
STATUS_DECLARED_ZERO = "declared_zero"    # structurally zero, with the proof
STATUS_NOT_MEASURED = "not_measured"      # could not run / not decomposable
BUDGET_STATUSES = (STATUS_MEASURED, STATUS_MEASURED_TOTAL, STATUS_PROBE,
                   STATUS_BOUND, STATUS_DECLARED_ZERO, STATUS_NOT_MEASURED)

#: IERS conventions keep |UT1-UTC| < 0.9 s -- the desync sensitivity the
#: frame/time line records against a convention leak between implementations.
DESYNC_DT_S = 0.9

#: Probes (deliberate, documented perturbations of one model input each).
_RELAX_RTOL, _RELAX_ATOL = 1e-7, 1e-2       # DOP853 tolerances x 1000
_STIFF_GAINS = (6.0, 0.02, 8.0)             # 2x the FL-403 PID
AERO_PROBE_MODEL = "spike_stage_aero"       # production-coefficient airframe
_JSBSIM_DT_S = 0.01                         # the 6-DOF executor step


# ---------------------------------------------------------------------------
# Budget lines
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceLine:
    """One source's contribution to one pair/phase/channel difference. The
    contribution is in channel units; ``status`` says how it was obtained,
    ``method`` is the one-sentence accounting, ``probe`` carries the raw
    probe payload, ``cause`` (a matrix FAILURE_CLASSES code or one of the
    two budget-only codes) explains a ``not_measured``."""

    source: str
    contribution: float
    status: str
    method: str
    cause: str | None = None
    matrix_class: str = ""
    probe: Mapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source not in BUDGET_SOURCES:
            raise ValueError(f"unknown budget source {self.source!r}")
        if self.status not in BUDGET_STATUSES:
            raise ValueError(f"unknown budget line status {self.status!r}")
        if self.status == STATUS_NOT_MEASURED and self.contribution != 0.0:
            raise ValueError("a not_measured line cannot carry a contribution")

    def to_dict(self) -> dict:
        return {"source": self.source, "contribution": self.contribution,
                "status": self.status, "method": self.method, "cause": self.cause,
                "matrix_failure_class": self.matrix_class,
                "probe": json.loads(json.dumps(dict(self.probe), default=float))}


@dataclass(frozen=True)
class ChannelBudget:
    """The full budget for one (pair, phase, channel): the measured total
    (``None``: the pair did not fly -- recorded, not zero), one line per
    source (always all five, in :data:`BUDGET_SOURCES` order), and -- the
    point of the exercise -- the residual between the component sum and the
    total, recorded, with a coverage ratio."""

    pair: str
    phase: str
    channel: str
    unit: str
    total: float | None
    lines: tuple[SourceLine, ...]
    matrix_result: str | None = None
    matrix_limit: float | None = None

    def __post_init__(self) -> None:
        sources = tuple(l.source for l in self.lines)
        if sources != BUDGET_SOURCES:
            raise ValueError(f"{self.pair}@{self.phase}/{self.channel}: budget "
                             f"lines must be exactly {BUDGET_SOURCES}, got {sources}")

    @property
    def components(self) -> dict[str, float]:
        return {l.source: l.contribution for l in self.lines}

    @property
    def component_sum(self) -> float:
        return sum(l.contribution for l in self.lines)

    @property
    def residual(self) -> float | None:
        """total - sum(components). Negative = the recorded bounds already
        cover the total (conservative); positive = unattributed remainder
        (plant formulation residual). Recorded either way, never zeroed."""
        return None if self.total is None else self.total - self.component_sum

    @property
    def coverage(self) -> float | None:
        """component_sum / total -- how much of the difference the budget
        accounts for (can exceed 1 when lines are conservative bounds)."""
        if self.total is None or self.total <= 0.0:
            return None
        return self.component_sum / self.total

    def to_dict(self) -> dict:
        return {"pair": self.pair, "phase": self.phase, "channel": self.channel,
                "unit": self.unit, "total": self.total,
                "lines": [l.to_dict() for l in self.lines],
                "component_sum": None if self.total is None else self.component_sum,
                "residual": self.residual, "coverage": self.coverage,
                "matrix_result": self.matrix_result, "matrix_limit": self.matrix_limit}


def _line(source: str, contribution: float, status: str, method: str, *,
          cause: str | None = None, probe: Mapping | None = None,
          matrix_class: str | None = None) -> SourceLine:
    if status == STATUS_NOT_MEASURED:
        contribution = 0.0
    return SourceLine(source=source, contribution=float(contribution), status=status,
                      method=method, cause=cause,
                      matrix_class=matrix_class or SOURCE_FAILURE_CLASS[source],
                      probe=dict(probe or {}))


def _not_measured(source: str, method: str, cause: str, probe: Mapping | None = None,
                  matrix_class: str | None = None) -> SourceLine:
    return _line(source, 0.0, STATUS_NOT_MEASURED, method, cause=cause,
                 probe=probe, matrix_class=matrix_class)


def _value(channel: str, metrics: Mapping) -> float:
    """Channel value of a compare_tracks metrics dict -- the same absolute
    final-state value the matrix row grades."""
    v = _row_value(channel, metrics)
    if v is None:
        raise KeyError(f"metrics carry no {channel} channel")
    return float(v)


def _summary_diff(metrics: Mapping) -> dict:
    out = {"final_position_m": float(metrics["final_position_m"]),
           "final_velocity_m_s": float(metrics["final_velocity_m_s"])}
    if "final_mass_kg" in metrics:
        out["final_mass_kg"] = float(metrics["final_mass_kg"])
    return out


# ---------------------------------------------------------------------------
# Probes (deliberate single-input perturbations, at most once per budget run)
# ---------------------------------------------------------------------------

class _Probes:
    """Runs every probe at most once per budget run and caches the track (or
    the error) it produced. A probe that cannot run -- backend missing,
    ``probes=False``, probe raised -- yields ``None`` and the caller records
    a ``not_measured`` line: the absence, not a zero, is the evidence."""

    def __init__(self, scenario: CrossModelScenario,
                 ran: Mapping[str, object], enabled: bool):
        self.scenario = scenario
        self.ran = dict(ran)
        self.enabled = enabled
        self.errors: dict[str, str] = {}
        self._cache: dict[str, object] = {}
        self._engine_residuals: Mapping | None = None

    @property
    def reference_ascent(self) -> ModelTrack | None:
        three = self.ran.get("three_dof")
        return None if three is None else three.ascent

    @property
    def _six_dof_flown(self) -> bool:
        return "six_dof" in self.ran

    def _guard(self, key: str, base_ok: bool) -> bool:
        if key in self._cache:
            return False
        if not self.enabled or not base_ok:
            self._cache[key] = None
            return False
        return True

    def _six_dof_probe(self, key: str, **spike_kwargs) -> ModelTrack | None:
        if not self._guard(key, self._six_dof_flown and self.reference_ascent is not None):
            return self._cache.get(key)
        try:
            track = six_dof_ascent(self.scenario, reference=self.reference_ascent,
                                   **spike_kwargs)
        except Exception as exc:  # a failed probe is recorded, never faked
            self.errors[key] = f"{type(exc).__name__}: {exc}"
            track = None
        self._cache[key] = track
        return track

    def controller(self) -> ModelTrack | None:
        """6-DOF ascent with 2x PID gains (loop-stiffening probe)."""
        from orbitopt.flightlab.sixdof.tracking import PidGains

        return self._six_dof_probe("controller",
                                   gains=PidGains(kp=_STIFF_GAINS[0], ki=_STIFF_GAINS[1],
                                                  kd=_STIFF_GAINS[2]))

    def aero(self) -> ModelTrack | None:
        """6-DOF ascent on the production aero airframe (potential probe --
        the V-02 scenario itself is zero-aero on every side)."""
        if "aero" in self._cache:
            return self._cache["aero"]
        sim = None
        if self.enabled and self._six_dof_flown and self.reference_ascent is not None:
            from orbitopt.flightlab.sixdof.jsbsim_driver import SixDofSim

            sim = SixDofSim(_JSBSIM_DT_S, model=AERO_PROBE_MODEL)
        return self._six_dof_probe("aero", sim=sim)

    def _three_dof_probe(self, key: str, **kwargs) -> ModelTrack | None:
        if not self._guard(key, self.reference_ascent is not None):
            return self._cache.get(key)
        try:
            track = three_dof_ascent(self.scenario, **kwargs)
        except Exception as exc:
            self.errors[key] = f"{type(exc).__name__}: {exc}"
            track = None
        self._cache[key] = track
        return track

    def numerics(self) -> ModelTrack | None:
        """3-DOF ascent at rtol/atol x 1000 (tolerance-relaxation probe)."""
        return self._three_dof_probe("numerics", rtol=_RELAX_RTOL, atol=_RELAX_ATOL)

    def engine(self) -> ModelTrack | None:
        """3-DOF ascent with thrust scaled by (1 + the live E6-01 parity
        residual) -- the pair shift the two aligned engine laws' residual
        actually produces (choked mass flow does not follow the scaling, as
        the E6-01 law demands)."""
        resid = self.engine_residuals()
        if resid is None:
            return None
        return self._three_dof_probe("engine",
                                     thrust_factor=1.0 + resid["thrust_rel_residual"])

    def engine_residuals(self) -> Mapping | None:
        """The live E6-01 (#273) engine-law residuals -- the parity checks
        that the two implementations of the ONE aligned propulsion law agree.
        The engine line is built from these numbers (the acceptance
        requirement 'engine term based on the E6-01 aligned model' is
        literally the input); a failed alignment check returns ``None``, so
        an unaligned engine is refused, not silently budgeted."""
        if "engine_residuals" in self._cache:
            return self._cache["engine_residuals"]
        from orbitopt.flightlab.parity import parity_report

        out = None
        if self.enabled:
            try:
                report = parity_report()
                checks = {c["id"]: c for c in report["checks"]}
                thrust = checks.get("parity-stage1-thrust-vs-jsbsim")
                flow = checks.get("parity-jsbsim-flow-vs-source")
                pad = checks.get("parity-engine-pad-thrust-vs-jsbsim")
                if report.get("all_passed") and thrust and flow \
                        and thrust["passed"] and flow["passed"]:
                    out = {"thrust_rel_residual": abs(float(thrust["value"])),
                           "flow_rel_residual": abs(float(flow["value"])),
                           "pad_thrust_rel_residual": (
                               None if pad is None or pad.get("value") is None
                               else abs(float(pad["value"]))),
                           "basis": "orbitopt.flightlab.parity.check_parity, E6-01 (#273) "
                                    "aligned propulsion law (ADR flightlab-0009)"}
            except Exception as exc:
                self.errors["engine_residuals"] = f"{type(exc).__name__}: {exc}"
        self._cache["engine_residuals"] = out
        return out

    def desync(self) -> Mapping | None:
        """GCRS shift of the reference MECO state under a |UT1-UTC| desync
        at the IERS bound -- the frame/time sensitivity the single shared
        bridge applies zero of, recorded as the budget."""
        if "desync" in self._cache:
            return self._cache["desync"]
        from orbitopt.flightlab.frames import from_gcrs, to_gcrs

        out = None
        ref = self.reference_ascent
        if ref is not None:
            t = np.array([ref.t_end])
            state = ref.states_gcrs[-1]
            r_e, v_e = from_gcrs(state[None, :3], state[None, 3:], "ECEF",
                                 self.scenario.epoch_utc(), t)
            r_g, v_g = to_gcrs(r_e, v_e, "ECEF",
                               self.scenario.epoch_utc() + timedelta(seconds=DESYNC_DT_S), t)
            out = {"dt_desync_s": DESYNC_DT_S,
                   "position_shift_m": float(np.linalg.norm(r_g[0] - state[:3])),
                   "velocity_shift_m_s": float(np.linalg.norm(v_g[0] - state[3:]))}
        self._cache["desync"] = out
        return out


# ---------------------------------------------------------------------------
# Per-pair recipes
# ---------------------------------------------------------------------------

def _not_measured_reason(probes: _Probes, key: str,
                         ran_ok: bool) -> str:
    """Cause code for a probe line that could not run: the probe was asked
    off, the backend is missing, or the probe itself raised (matrix
    vocabulary where it fits)."""
    if not probes.enabled:
        return "probes_disabled"
    if not ran_ok:
        return "backend_no_track"
    return "backend_runtime_error" if key in probes.errors else "backend_no_track"


def _ascent_lines(channel: str, total: float, scenario: CrossModelScenario,
                  probes: _Probes) -> list[SourceLine]:
    """six_dof_vs_three_dof @ ascent: every source gets a probe, a bound or
    a declared zero -- and what is left over is the formulation residual."""
    u = CHANNEL_UNIT[channel]
    desync = probes.desync()
    lines: list[SourceLine] = []

    # frame/time: one declared epoch, one shared bridge -> applied desync 0.
    sens = None
    if desync is not None and channel != "mass_kg":
        sens = desync["position_shift_m"] if channel == "position_m" \
            else desync["velocity_shift_m_s"]
    if sens is None:
        frame_method = (
            "both backends bridge through ONE to_gcrs at ONE declared epoch: "
            "the applied frame/time desync is exactly 0"
            + ("; mass is invariant to the frame/time convention"
               if channel == "mass_kg" else ""))
    else:
        frame_method = (
            "both backends bridge through ONE to_gcrs at ONE declared epoch: "
            "the applied frame/time desync is exactly 0; the budget records "
            "the bridge sensitivity at the IERS |UT1-UTC| < 0.9 s bound "
            f"({sens:.3g} {u}) in case a future backend rotates on its own "
            "time convention")
    lines.append(_line("frame_time", 0.0, STATUS_DECLARED_ZERO, frame_method,
                       probe=dict(desync or {})))

    # numerical integration: tolerance-relaxation probe on the 3-DOF side.
    relaxed = probes.numerics()
    if relaxed is None:
        lines.append(_not_measured(
            "numerical_integration",
            "tolerance-relaxation probe (DOP853 rtol/atol x 1000) not run",
            _not_measured_reason(probes, "numerics",
                                 probes.reference_ascent is not None),
            probe={"rtol": _RELAX_RTOL, "atol": _RELAX_ATOL},
            matrix_class="numerical_integration"))
    else:
        d = compare_tracks(probes.reference_ascent, relaxed)
        share = _value(channel, d)
        lines.append(_line(
            "numerical_integration", share, STATUS_MEASURED,
            "3-DOF reference re-run at rtol 1e-7 / atol 1e-2 (x1000 the "
            "harness tolerances): the final-state shift is the integration "
            f"share (>= {share:.3g} {u}); the JSBSim RK4 fixed-step side is "
            "the published E6-04 bound (~2.5e-8 m at dt=0.01 s, "
            "docs/benchmarks/e604_step_convergence.json)",
            probe={"rtol": _RELAX_RTOL, "atol": _RELAX_ATOL,
                   "shift": _summary_diff(d)}))

    # engine: the E6-01 aligned law's live parity residual, probed.
    resid = probes.engine_residuals()
    if resid is None:
        lines.append(_not_measured(
            "engine",
            "the E6-01 parity report is unavailable or the thrust/flow "
            "alignment checks did not pass -- an unaligned engine cannot be "
            "budgeted against the aligned model",
            "probes_disabled" if not probes.enabled else "backend_runtime_error",
            matrix_class="propellant_accounting" if channel == "mass_kg" else None))
    elif channel == "mass_kg":
        bound = scenario.mdot_kg_s * _JSBSIM_DT_S
        lines.append(_line(
            "engine", bound, STATUS_BOUND,
            f"mass channel: the JSBSim tank bookkeeping runs one FDM frame "
            f"late (mdot * dt = {bound:.4g} kg) -- the disclosed "
            "implementation residual of the aligned law (ADR flightlab-0009), "
            "while the thrust/flow law itself matches to ~3e-8",
            matrix_class="propellant_accounting",
            probe={"mdot_kg_s": scenario.mdot_kg_s, "dt_s": _JSBSIM_DT_S,
                   "thrust_rel_residual": resid["thrust_rel_residual"],
                   "flow_rel_residual": resid["flow_rel_residual"]}))
    else:
        engine_track = probes.engine()
        if engine_track is None:
            lines.append(_not_measured(
                "engine", "engine-law probe (thrust x (1 + parity residual)) "
                "did not run",
                _not_measured_reason(probes, "engine",
                                     probes.reference_ascent is not None)))
        else:
            d = compare_tracks(probes.reference_ascent, engine_track)
            share = _value(channel, d)
            lines.append(_line(
                "engine", share, STATUS_MEASURED,
                "thrust perturbed by (1 + the live parity residual "
                f"{resid['thrust_rel_residual']:.3g} relative) -- the "
                "residual of the TWO implementations of the one E6-01 "
                "aligned law (choked flow held fixed); E6-01 closed the "
                "pre-alignment ~12% back-pressure gap, and this near-zero "
                "share is exactly what that alignment means",
                probe=dict(resid, shift=_summary_diff(d))))

    # aero: zero by declaration, potential recorded.
    aero_track = probes.aero()
    potential = None if aero_track is None else _summary_diff(
        compare_tracks(probes.reference_ascent, aero_track))
    lines.append(_line(
        "aero", 0.0, STATUS_DECLARED_ZERO,
        "the scenario declares zero aerodynamics on every side and the "
        "parity check parity-jsbsim-aero-zero asserts the packaged airframe "
        "flies none -- the contribution is 0"
        + (f"; POTENTIAL with the production {AERO_PROBE_MODEL} coefficients "
           f"in: {potential['final_position_m']:.3g} m @ MECO (recorded, not "
           "summed)" if potential is not None else ""),
        probe={"aero_model": AERO_PROBE_MODEL, "potential": potential}))

    # controller: loop-stiffening probe share.
    stiff = probes.controller()
    if stiff is None:
        lines.append(_not_measured(
            "controller", "loop-stiffening probe (2x PID gains) not run",
            _not_measured_reason(probes, "controller", probes._six_dof_flown)))
    else:
        after = _value(channel, compare_tracks(probes.reference_ascent, stiff))
        share = max(total - after, 0.0)
        lines.append(_line(
            "controller", share, STATUS_PROBE,
            "6-DOF plant re-flown at 2x PID gains (kp/ki/kd = 6/0.02/8, "
            "FL-403 baseline 3/0.01/4): the pair difference against the same "
            f"reference falls from {total:.3g} to {after:.3g} {u}; the "
            "removable share is attributed here, the rest stays in the "
            "residual (plant formulation, not loop gain)",
            matrix_class="attitude_control" if channel != "mass_kg"
            else "propellant_accounting",
            probe={"gains": list(_STIFF_GAINS), "baseline_total": total,
                   "stiff_pair_diff": after}))
    return lines


def _coast_lines(channel: str, total: float, scenario: CrossModelScenario,
                 probes: _Probes) -> list[SourceLine]:
    """Every coast pair coasts the SAME point-mass (+ optional J2) handoff
    state from ONE declared GCRS epoch -- the disagreement can only be
    integration numerics and (only with J2 on) a frame/time convention
    difference; with a time-invariant point-mass field the epoch cannot move
    the arc at all, so the numerical line carries the total."""
    u = CHANNEL_UNIT[channel]
    desync = probes.desync()
    sens = 0.0
    if desync is not None:
        sens = desync["position_shift_m"] if channel == "position_m" \
            else desync["velocity_shift_m_s"] if channel == "velocity_m_s" else 0.0
    if scenario.coast_j2:
        frame = _line("frame_time", sens, STATUS_BOUND,
                      "the J2 figure is Earth-fixed, so the epoch convention "
                      "does matter once coast_j2 is on: the desync budget is "
                      "the IERS 0.9 s sensitivity (handoff itself is ONE "
                      "declared state)",
                      probe=dict(desync or {}))
    else:
        frame = _line(
            "frame_time", 0.0, STATUS_DECLARED_ZERO,
            "handoff is ONE declared GCRS state at ONE declared epoch and "
            "the point-mass coast is epoch-invariant: a time-convention "
            "leak cannot move the arc -- structurally 0 "
            f"(0.9 s desync sensitivity would be {sens:.3g} {u})",
            probe=dict(desync or {}))
    numerics = _line(
        "numerical_integration", total, STATUS_MEASURED_TOTAL,
        "both backends coast the identical force model from the identical "
        "handoff constant -- closed form vs DOP853 vs independent solver: "
        f"the whole {total:.3g} {u} difference IS integration numerics; the "
        "matrix row's basis cites the V-03 convergence bound it sits under")
    engine = _line(
        "engine", 0.0, STATUS_DECLARED_ZERO,
        "coast: no thrust on any side, one shared handoff mass constant"
        if channel != "mass_kg" else
        "coast: mass is one constant (the handoff value) copied to every "
        "backend -- no engine term exists to disagree about",
        matrix_class="propellant_accounting" if channel == "mass_kg" else None)
    aero = _line("aero", 0.0, STATUS_DECLARED_ZERO,
                 "coast above Karman with zero aero declared on every side")
    controller = _line("controller", 0.0, STATUS_DECLARED_ZERO,
                       "coast: nothing to control on any backend")
    return [frame, numerics, engine, aero, controller]


#: The coast backends that all coast the SAME point-mass (+J2) handoff, so
#: a difference between two of them is numerics by construction; a coast
#: pair outside this set gets no decomposition recipe.
SAME_MODEL_COAST_BACKENDS = frozenset({"three_dof", "two_body_analytic", "tudatpy"})


def _lines_for_pair(pair: str, phase: str, channel: str, total: float,
                    scenario: CrossModelScenario, probes: _Probes,
                    row: RowOutcome | None) -> list[SourceLine]:
    labels = pair.split("_vs_", 1)
    if phase == "ascent" and set(labels) == {"three_dof", "six_dof"}:
        return _ascent_lines(channel, total, scenario, probes)
    if phase == "coast" and set(labels) <= SAME_MODEL_COAST_BACKENDS:
        return _coast_lines(channel, total, scenario, probes)
    return [_not_measured(
        s, "no decomposition recipe for this pair/phase: the full total "
           "stays in the residual, recorded",
        "no_matrix_cell" if row is not None and row.result == ROW_UNMATRICULATED
        else "no_decomposable_recipe") for s in BUDGET_SOURCES]


# ---------------------------------------------------------------------------
# Harness / matrix plumbing
# ---------------------------------------------------------------------------

@dataclass
class DiscrepancyBudgetVerification:
    """One budget run: the matrix it reused, the harness run it budgets, the
    per-channel budgets and the digested ``model-discrepancy-budget``
    report."""

    matrix: AcceptanceMatrix
    harness: CrossModelVerification
    rows: tuple[RowOutcome, ...]
    budgets: tuple[ChannelBudget, ...]
    report: dict
    report_path: object = None

    @property
    def result(self) -> str:
        return status_result(self.report["status"])

    def failed_rows(self) -> list:
        return [r for r in self.rows
                if r.result in (ROW_FAILED, ROW_NOT_CHECKED, ROW_UNMATRICULATED)]

    def to_dict(self) -> dict:
        return {"matrix": {"name": self.matrix.name,
                           "config_digest": self.matrix.config_digest},
                "budgets": [b.to_dict() for b in self.budgets],
                "status": self.report["status"],
                "report_file": str(self.report_path) if self.report_path else None}


def _residual_from_budget(b: ChannelBudget, row: RowOutcome) -> dict:
    """One verification-report residual per flown budget: the total in the
    standard fields, the components / sum / residual alongside, and
    ``within_tolerance`` mirrored from the matrix row -- the matrix votes on
    the total, the budget records the split of the same number."""
    residual = {
        "name": f"{b.pair}.{b.phase}.{b.channel}",
        "reference": (f"model-discrepancy-budget: {b.pair.replace('_vs_', ' vs ')} "
                      f"@ {b.phase}, {b.channel} channel"),
        "category": row.category,
        "stage": list(row.stages),
        "position_residual_m": b.total if b.channel == "position_m" else None,
        "velocity_residual_m_s": (b.total if b.channel == "velocity_m_s" else None),
        "scope": ("per-source decomposition of the pair difference (frame/time, "
                  "numerical integration, engine, aero, controller)"),
        "not_proven": ("the residual is the recorded unattributed remainder -- "
                       "plant formulation differences (rigid-body vs point-mass, "
                       "FCS/actuator dynamics, atmosphere tables), not zero-padding"),
        "components": b.components,
        "component_sum": b.component_sum,
        "component_residual": b.residual,
        "coverage": b.coverage,
        "budget_status": {l.source: l.status for l in b.lines},
        "matrix_result": b.matrix_result,
    }
    if b.matrix_limit is not None:
        residual["tolerance"] = {b.channel: b.matrix_limit}
        residual["observed"] = {b.channel: b.total}
    if row.result == ROW_PASSED:
        residual["within_tolerance"] = True
    elif row.result == ROW_FAILED:
        residual["within_tolerance"] = False
    if row.basis:
        residual["matrix_basis"] = row.basis
    return residual


def run_discrepancy_budget(matrix: "AcceptanceMatrix | Mapping | str | Path | None" = None,
                           scenario: CrossModelScenario | None = None,
                           backends: Sequence[str] | None = None,
                           thresholds: Mapping | None = None,
                           probes: bool = True,
                           report_path=None) -> DiscrepancyBudgetVerification:
    """Fly the V-02 scenario once (with the acceptance matrix's resolved
    tolerances -- the matrix is *reused*, not re-implemented), grade it with
    ``evaluate_matrix``, and build a per-source discrepancy budget for every
    pair that flew. ``probes=False`` records the structure with every probe
    line ``not_measured`` (full total in the residual) -- for environments
    where the JSBSim probe flights are not wanted. The report's status
    mirrors the matrix verdicts: budget lines are recorded accounting, and
    only the matrix rows vote."""
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
    row_map = {(r.pair, r.phase, r.metric): r for r in rows}
    ran = {o.label: o for o in harness.backends if o.status == STATUS_RAN}

    P = _Probes(harness.scenario, ran, bool(probes))
    budgets: list[ChannelBudget] = []
    for p in harness.pair_metrics:
        for channel in BUDGET_CHANNELS:
            total = _row_value(channel, p["metrics"])
            if total is None:
                continue
            row = row_map[(p["pair"], p["phase"], channel)]
            budgets.append(ChannelBudget(
                pair=p["pair"], phase=p["phase"], channel=channel,
                unit=CHANNEL_UNIT[channel], total=float(total),
                lines=tuple(_lines_for_pair(p["pair"], p["phase"], channel,
                                            float(total), harness.scenario, P, row)),
                matrix_result=row.result, matrix_limit=row.limit))

    # Matrix cells whose pair did not fly: record all-five lines not_measured
    # with the matrix's own missing-pair cause code -- never a fake budget,
    # never a pass.
    flown = {(p["pair"], p["phase"]) for p in harness.pair_metrics}
    statuses = {o.label: o for o in harness.backends}
    for cell in mat.cells:
        if (cell.pair, cell.phase) in flown:
            continue
        cause = _missing_pair_cause(cell, statuses).code
        for channel in BUDGET_CHANNELS:
            row = row_map[(cell.pair, cell.phase, channel)]
            budgets.append(ChannelBudget(
                pair=cell.pair, phase=cell.phase, channel=channel,
                unit=CHANNEL_UNIT[channel], total=None,
                lines=tuple(_not_measured(
                    s, f"pair did not fly: {_cause_text(cause)}", cause)
                    for s in BUDGET_SOURCES),
                matrix_result=row.result, matrix_limit=row.limit))

    budgets.sort(key=lambda b: (b.pair, b.phase, BUDGET_CHANNELS.index(b.channel)))
    residuals = [_residual_from_budget(b, row_map[(b.pair, b.phase, b.channel)])
                 for b in budgets if b.total is not None]

    report = build_report(
        "model-discrepancy-budget", f"{mat.name}-budget", propagations=[],
        residuals=residuals,
        inputs={"matrix": {"name": mat.name, "config_digest": mat.config_digest,
                           "stages": list(mat.stages)},
                "scenario": harness.scenario.to_dict(),
                "probes_enabled": bool(probes),
                "probe_errors": dict(P.errors),
                "sources": list(BUDGET_SOURCES),
                "budgets": [b.to_dict() for b in budgets],
                "backends": [o.to_dict() for o in harness.backends],
                "matrix_result": {f"{r.pair}.{r.phase}.{r.metric}": r.result
                                  for r in rows}})
    report = finalize_report(report, report_path)
    path = report_path
    if path is None:
        import os

        if os.environ.get(REPORT_DIR_ENV):
            path = default_report_path(report, os.environ[REPORT_DIR_ENV])
    return DiscrepancyBudgetVerification(
        matrix=mat, harness=harness, rows=tuple(rows), budgets=tuple(budgets),
        report=report, report_path=path)


def _cause_text(code: str) -> str:
    """One-line text for a matrix cause code (for not_run budget lines)."""
    fc = FAILURE_CLASSES.get(code)
    return fc.title if fc is not None else code


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_STATUS_TAGS = {STATUS_MEASURED: "measured", STATUS_MEASURED_TOTAL: "total",
                STATUS_PROBE: "probe", STATUS_BOUND: "bound",
                STATUS_DECLARED_ZERO: "zero*", STATUS_NOT_MEASURED: "--"}


def _fmt(x: float | None, digits: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{digits}g}"


def main(argv=None) -> int:
    """Print the per-source budget per pair/phase/channel with the recorded
    residual, then the matrix verdict. Exit 1 iff a gated matrix row failed
    -- the budget records, the matrix gates."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m orbitopt.verify.discrepancy_budget",
        description="E8-02: split every V-02 cross-backend difference into a "
                    "per-source error budget (frame/time, numerical integration, "
                    "engine, aero, controller) with the component-sum residual "
                    "recorded, reusing the acceptance matrix for tolerances.")
    parser.add_argument("--matrix", default=None, metavar="PATH",
                        help="acceptance-matrix config (default: the shipped one)")
    parser.add_argument("--backends", default=None,
                        help="comma list passed to the V-02 harness (default: all)")
    parser.add_argument("--threshold", action="append", default=[], metavar="KEY=VALUE",
                        help="matrix threshold override, same syntax as the matrix CLI")
    parser.add_argument("--coast-duration", type=float, default=None, metavar="S",
                        help="override the harness scenario coast duration")
    parser.add_argument("--no-probes", action="store_true",
                        help="skip the probe flights (record the structure; the "
                             "full totals stay in the residual)")
    parser.add_argument("-o", "--output", default=None, metavar="PATH",
                        help="report path (default: $ORBITOPT_VERIFICATION_REPORT_DIR)")
    parser.add_argument("--json", action="store_true",
                        help="print the verification report document instead of the table")
    args = parser.parse_args(argv)

    overrides = {}
    for item in args.threshold:
        key, _, value = item.partition("=")
        overrides[key] = value
    scen = None
    if args.coast_duration is not None:
        scen = replace(CrossModelScenario(), coast_duration_s=float(args.coast_duration))
    res = run_discrepancy_budget(args.matrix, scenario=scen,
                                 backends=None if args.backends is None else args.backends.split(","),
                                 thresholds=overrides, probes=not args.no_probes,
                                 report_path=args.output)
    if args.json:
        print(json.dumps(res.report, indent=2, sort_keys=True))
        return 1 if res.result == "failed" else 0

    for o in res.harness.backends:
        line = f"[{o.status:>11}] {o.label}"
        if o.detail:
            line += f" -- {o.detail}"
        print(line)
    print()
    for b in res.budgets:
        if b.total is None:
            print(f"[not_run  ] {b.pair} @ {b.phase} {b.channel}: "
                  f"{b.lines[0].cause} -- {b.lines[0].method}")
            continue
        cov = "" if b.coverage is None else f"; coverage {100.0 * b.coverage:.1f}%"
        tags = {l.source: _STATUS_TAGS[l.status] for l in b.lines}
        parts = " + ".join(f"{s}={_fmt(b.components[s])}({tags[s]})"
                           for s in BUDGET_SOURCES)
        print(f"[budget   ] {b.pair} @ {b.phase} {b.channel}: total {_fmt(b.total)} {b.unit}")
        print(f"            = {parts}")
        print(f"            = {b.component_sum:.4g}; residual {_fmt(b.residual)} {b.unit} "
              f"(recorded){cov}")
        if b.matrix_limit is not None:
            print(f"            matrix {b.matrix_result} (limit {b.matrix_limit:g} {b.unit})")
    for key, detail in res.report["inputs"]["probe_errors"].items():
        print(f"probe error {key}: {detail}")
    print(f"\nbudget {res.matrix.name} ({res.matrix.config_digest[:12]}): "
          f"{format_status(res.report['status'])}"
          + (f" -> {res.report_path}" if res.report_path else ""))
    return 1 if res.result == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
