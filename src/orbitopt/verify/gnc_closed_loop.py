"""E13-07 (task #424): independent closed-loop verification of a sealed GNC run.

An E13-01 run log proves the run agrees with *itself* (causal replay,
schedule recomputation, digest). Self-consistency is not correctness: the
same mis-implemented propagator that wrote the Truth stream would re-fly it
identically. This module attacks the record from outside, with nothing but
the sealed log as input. Four error classes are measured and listed
SEPARATELY in ``report["inputs"]["error_classes"]`` -- never blended into
one flag (task #395's discipline, task #424's acceptance):

``numerical_error``
    The recorded ``ControlEvent`` sequence (realized delta-v at
    ``t_exec_s``) re-integrated through
    :func:`orbitopt.relative.propagation.propagate_pair` on a step-halving
    grid and compared against the recorded Truth stream: integrator error
    under the commands the loop actually flew, measured, not assumed
    (category ``numerical_consistency``, evidence ``numerical_consistency``).

``model_error``
    The same maneuver sequence re-flown in **TudatPy**
    (:func:`orbitopt.verify.tudat_propagate.propagate_multi_arc`, coast /
    impulsive arcs behind an independently formulated C++ propagator), with
    the run's exact GM and J2 forced through ``central_gravity``, in
    ECLIPJ2000 -- the SPICE pool tudatpy loads defines no GCRS frame, and a
    rejected frame aborts the propagation at t=0 *without raising* (the
    guard E12-05 learned). States of any other log frame are rotated by the
    constant J2000.0 obliquity; every graded quantity is a relative state
    or a norm, invariant under that one rotation (category
    ``model_agreement``, evidence ``independent_model``). **No backend, no
    fake pass**: without tudatpy the line is recorded ``not_checked`` with
    reason :data:`REFLIGHT_BACKEND_MISSING`, never skipped or passed.

``navigation_inconsistency``
    Analytical review of the navigation record against the truth record:
    every Observation normalised by its OWN declared covariance at its
    sampling instant (NIS), every Estimate by its declared 6x6 covariance
    at its update instant (NEES), truth interpolated by the C1 Hermite
    dense output through the truth stream. A filter lying about its
    uncertainty -- or about the state -- moves these numbers (category
    ``model_agreement``, evidence ``analytical``).

``optimizer_control_nonconvergence``
    What the actuator delivered against what the law asked, cross-checked
    with the run's declared execution model (``config["execution"]``: the
    E13-01 reference model is realized = commanded * (scale + N(0, rtol))),
    plus a settle-window read of the command stream, cross-checked against
    a producer convergence assertion (``config["convergence"]["converged"]``,
    written by the closed-loop runner, e.g. E13-04). Absent assertion ->
    ``not_checked``, never a pass; a claim the records contradict fails
    (category ``numerical_consistency``, evidence ``analytical``).

Evidence binding follows the E13-00 layering (``report["inputs"]["binding"]``):
the run's ``run_digest``, its optional ``plan_digest``, the raw artifact
digest of the run-log document, a ``verification_digest`` binding THIS
verdict to that run, and the ``environment_digest`` of the environment that
checked (provenance; kept out of the machine-independent digests).
:func:`verify_evidence_binding` recomputes the block and fails closed.

``status.independent_validation`` is exactly the TudatPy line: a run
re-flown only by its own model's numerics is internally consistent, not
independently validated. Grading a *sample set* (success, each failure
class, boundaries) is this task's final gate and needs the E13-04
closed-loop results; this module is the tool that will grade them, built
and proven against the E13-01 reference run in the meantime. It depends
only on the E13-01 contract layer and pre-existing verify tooling, so it
develops in parallel with (not on top of) E13-02/E13-03.

    from orbitopt.verify.gnc_closed_loop import verify_gnc_closed_loop
    report = verify_gnc_closed_loop("runs/e13-run.json", out_path="runs/e13-07.json")
    report["status"]["independent_validation"]   # the re-flight verdict
    report["inputs"]["error_classes"]            # four lines, separately
"""
from __future__ import annotations

import importlib.util
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from orbitopt.gnc.recording import GncRunLog
from orbitopt.rendezvous.canonical import raw_sha256
from orbitopt.relative.propagation import EARTH_RE, EARTH_MU, propagate_pair
from orbitopt.verify.events import HermiteDenseOutput
from orbitopt.verify.report import (
    CHECK_MODEL, CHECK_NUMERICAL, EVIDENCE_ANALYTICAL, EVIDENCE_INDEPENDENT_MODEL,
    EVIDENCE_NUMERICAL, EVIDENCE_QUALIFICATION, RESULT_NOT_CHECKED,
    build_report, report_digest, to_jsonable, write_report)

#: verification-report-1.0 ``kind`` of the report this module assembles.
REPORT_KIND = "gnc-closed-loop"
#: document tag inside the E13-00-style verification digest binding a verdict to a run.
BINDING_KIND = "orbitopt.verify.gnc-closed-loop.verification-digest"

#: missing-independent-backend marker (tri-state policy, task #261: absent
#: evidence is recorded, never scored as a pass).
REFLIGHT_BACKEND_MISSING = "REFLIGHT_BACKEND_MISSING"
#: a convergence verdict is the producer's to state; without one there is
#: nothing to cross-check, so the assertion line is not_checked, not passed.
CONVERGENCE_ASSERTION_ABSENT = "PRODUCER_CONVERGENCE_ASSERTION_ABSENT"
#: a truth stream too short to interpolate or replay against is an honest
#: absence, recorded as such.
TRUTH_TRACK_TOO_SHORT = "TRUTH_TRACK_TOO_SHORT"

#: The four error classes task #424 requires listed separately.
ERROR_CLASSES = ("numerical_error", "model_error", "navigation_inconsistency",
                 "optimizer_control_nonconvergence")

#: default upper-tail probability of the per-record chi-square gate
NIS_GATE_PROB = 0.9995
#: replay-vs-recorded-truth and convergence-sweep floors (LEO closed loop at
#: sub-second RK4 steps: anything above these is a bug, not a step effect)
REPLAY_TOL_POSITION_M = 0.1
REPLAY_TOL_VELOCITY_M_S = 1.0e-4
#: float-resolution floor for a ~7000 km orbit: re-chained RK4 sums differ at
#: ~1e-6 m even when they integrate the identical model, so below this the
#: convergence sweep measures bookkeeping, not integration error.
REPLAY_FLOOR_POSITION_M = 1.0e-5
#: truth-reference refinement for the navigation review [s]: finer than any
#: plausible sensor cadence, coarse enough to stay cheap; its own residual
#: (Kepler curvature at h=0.5 s) is orders below the declared sigmas.
REFERENCE_REFINE_DT_S = 0.5
#: re-flight tolerances on relative position / velocity at truth instants;
#: sized for a J2-formulation difference over a ~90-minute run -- tighten
#: per campaign, they are arguments, not constants of nature.
REFLIGHT_TOL_POSITION_M = 100.0
REFLIGHT_TOL_VELOCITY_M_S = 0.05


# ---------------------------------------------------------------------------
# chi-square gate, closed form (no scipy): Wilson-Hilferty on a rational
# normal quantile -- analytical, documented, bounded.
# ---------------------------------------------------------------------------

def _normal_ppf(p: float) -> float:
    """Acklam's rational approximation of the standard normal quantile
    (|error| < 1.2e-9), ``p`` in (0, 1)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"_normal_ppf: p must be in (0, 1), got {p!r}")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    p_low = 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > 1.0 - p_low:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def chi2_upper_quantile(prob: float, dof: int) -> float:
    """Upper-tail chi-square quantile ``q`` with P(X <= q) = prob via the
    Wilson-Hilferty cube-root normal approximation, well inside the slack
    of the 99.95%-class gate used here. ``dof`` >= 1."""
    dof = int(dof)
    if dof < 1:
        raise ValueError(f"chi2_upper_quantile: dof must be >= 1, got {dof!r}")
    z = _normal_ppf(float(prob))
    k3 = 2.0 / (9.0 * dof)
    return dof * (1.0 - k3 + z * math.sqrt(k3)) ** 3


def nis_quadratic_form(delta, covariance) -> tuple[float, int]:
    """chi-square contribution of ``delta`` under a symmetric PSD
    ``covariance``: whiten by eigen-decomposition and sum over the
    numerically positive spectrum. The returned dof is the effective rank:
    a covariance declaring zero variance on a channel makes that channel
    UNSCORED, it never inflates the statistic. A fully rank-deficient
    covariance gives ``(nan, 0)`` -- the caller records the row unscoreable,
    it does not pass it. A one-sided PSD check: a negative eigenvalue beyond
    the rank tolerance is a contract violation raised here."""
    d = np.asarray(delta, dtype=np.float64).reshape(-1)
    cov = np.asarray(covariance, dtype=np.float64).reshape(d.size, d.size)
    cov = 0.5 * (cov + cov.T)
    evals, evecs = np.linalg.eigh(cov)
    cap = float(evals[-1]) if evals.size else 0.0
    if cap <= 0.0:
        return math.nan, 0
    tol = 1.0e-12 * cap
    if float(evals[0]) < -tol:
        raise ValueError(f"nis_quadratic_form: covariance not PSD (min eigenvalue "
                         f"{float(evals[0]):.3e} vs max {cap:.3e})")
    keep = evals > tol
    dof = int(np.count_nonzero(keep))
    if dof == 0:
        return math.nan, 0
    q = evecs[:, keep].T @ d
    return float(np.sum(q * q / evals[keep])), dof


# ---------------------------------------------------------------------------
# truth track: the verification reference (read-only; feeds nothing)
# ---------------------------------------------------------------------------

def truth_tracks(log: GncRunLog) -> dict[str, HermiteDenseOutput]:
    """Per-spacecraft C1 Hermite dense output through the Truth stream
    (position derivative = the recorded velocity, so interpolant error is
    O(h^4) on a uniform dynamics grid -- the verification reference is at
    least as sharp as what the sensors saw)."""
    frame = log.config.get("frame")
    by_sc: dict[str, list] = {}
    for tr in log.records.get("truth", ()):
        if frame is not None and tr.frame != frame:
            raise ValueError(f"run {log.run_id}: truth {tr.record_id} frame {tr.frame!r} "
                             f"disagrees with config frame {frame!r}")
        by_sc.setdefault(str(tr.spacecraft_id), []).append(tr)
    tracks: dict[str, HermiteDenseOutput] = {}
    for sc_id, recs in by_sc.items():
        recs = sorted(recs, key=lambda r: r.t_s)
        epochs = np.array([r.t_s for r in recs], dtype=np.float64)
        states = np.array([np.concatenate([r.r_m, r.v_m_s]) for r in recs], dtype=np.float64)
        tracks[sc_id] = HermiteDenseOutput(epochs, states)
    return tracks


class _TruthRel:
    """Truth relative state (chaser minus target) over elapsed seconds, off
    the truth reference tracks: one frame (the run's single declared one),
    one clock, both crafts evaluated at the same instant.

    The reference is NOT a plain Hermite through the raw truth samples: the
    dynamics grid can be coarser than the sensor grid (a 5 s plant cadence
    scored against 2 s observations would charge the INTERPOLANT's error to
    the filter). The recorded truth stream is instead refined through the
    run's own declared dynamics -- every recorded inter-sample interval is
    re-propagated from its own endpoint states on a refined grid under the
    same event-splitting as :func:`closed_loop_replay` -- and the dense
    track is a Hermite through the refined samples. Endpoints still sit on
    the recorded truth exactly; the model is the one the log itself
    declares, not one this verifier chose."""

    def __init__(self, log: GncRunLog, *, refine_dt_s: float = REFERENCE_REFINE_DT_S):
        self.log = log
        self.frame = str(log.config.get("frame"))
        self.target_id = str(log.config["target_id"])
        self.chaser_id = str(log.config["chaser_id"])
        self.refine_dt_s = float(refine_dt_s)
        self._tracks: dict[str, HermiteDenseOutput] | None = None

    @property
    def tracks(self) -> dict[str, HermiteDenseOutput]:
        if self._tracks is None:
            self._tracks = replay_reference_tracks(self.log, refine_dt_s=self.refine_dt_s)
        return self._tracks

    @property
    def span(self) -> tuple[float, float] | None:
        try:
            t_t = self.tracks[self.target_id]
            t_c = self.tracks[self.chaser_id]
        except (KeyError, ValueError):
            return None
        return max(t_t.t0, t_c.t0), min(t_t.t1, t_c.t1)

    def __call__(self, t_s: float) -> np.ndarray | None:
        """(r_rel, v_rel) at ``t_s``, or ``None`` outside the truth span."""
        span = self.span
        t_s = float(t_s)
        if span is None or not span[0] - 1e-9 <= t_s <= span[1] + 1e-9:
            return None
        st_t = self.tracks[self.target_id](t_s)
        st_c = self.tracks[self.chaser_id](t_s)
        return np.concatenate([st_c[:3] - st_t[:3], st_c[3:] - st_t[3:]])


# ---------------------------------------------------------------------------
# navigation_inconsistency
# ---------------------------------------------------------------------------

def navigation_consistency(log: GncRunLog, *, gate_prob: float = NIS_GATE_PROB,
                           max_gate_violation_fraction: float = 0.05,
                           max_mean_ratio: float = 3.0) -> dict:
    """Independent review of the navigation record from the record itself:
    every Observation against the interpolated truth at its sampling
    instant, normalised by its own declared covariance (NIS; velocity block
    scored when declared), every Estimate against truth at its update
    instant through its 6x6 covariance (NEES). Gates (tri-state -- an absent
    stream is not_checked, never a pass): gate violations within
    ``max_gate_violation_fraction`` of scored records AND mean
    statistic-per-dof within ``max_mean_ratio`` (a filter claiming tighter
    sigma than the data supports moves the mean; a liar about the state
    moves the gates)."""
    rel = _TruthRel(log)
    gates: dict[int, float] = {}

    def gate_for(dof: int) -> float:
        if dof not in gates:
            gates[dof] = chi2_upper_quantile(gate_prob, dof)
        return gates[dof]

    def score_stream(rows_iter):
        rows, ratios = [], []
        violations = unscoreable = 0
        max_nis = 0.0
        for row in rows_iter:
            nis, dof = row["nis"], row["dof"]
            if dof == 0 or not math.isfinite(nis):
                unscoreable += 1
                row["gate_passed"] = None
                rows.append(row)
                continue
            ratios.append(nis / dof)
            max_nis = max(max_nis, nis)
            ok = bool(nis <= gate_for(dof))
            row["gate_passed"] = ok
            violations += 0 if ok else 1
            rows.append(row)
        return {"rows": rows, "scored": len(ratios), "gate_violations": violations,
                "unscoreable": unscoreable, "max_nis": max_nis,
                "mean_nis_per_dof": float(np.mean(ratios)) if ratios else None}

    def obs_rows():
        for ob in log.records.get("observation", ()):
            truth = rel(ob.t_sample_s)
            if truth is None:
                continue
            rs = ob.relative_state
            delta_r = np.asarray(rs.r_rel_m, dtype=np.float64) - truth[:3]
            delta_v = np.asarray(rs.v_rel_m_s, dtype=np.float64) - truth[3:]
            nis_p, dof_p = nis_quadratic_form(delta_r, ob.covariance_position_m2)
            if ob.covariance_velocity_m2_s2:
                nis_v, dof_v = nis_quadratic_form(delta_v, ob.covariance_velocity_m2_s2)
            else:
                nis_v, dof_v = (math.nan, 0)
            if dof_v == 0:
                nis, dof = nis_p, dof_p
            elif dof_p == 0:
                nis, dof = nis_v, dof_v
            else:
                nis, dof = nis_p + nis_v, dof_p + dof_v
            yield {"record_id": ob.record_id, "t_s": float(ob.t_sample_s), "nis": nis, "dof": dof,
                   "position_error_m": float(np.linalg.norm(delta_r)),
                   "velocity_error_m_s": float(np.linalg.norm(delta_v))}

    def est_rows():
        for es in log.records.get("estimate", ()):
            truth = rel(es.t_update_s)
            if truth is None:
                continue
            delta = np.concatenate([np.asarray(es.relative_state.r_rel_m, dtype=np.float64) - truth[:3],
                                    np.asarray(es.relative_state.v_rel_m_s, dtype=np.float64) - truth[3:]])
            nis, dof = nis_quadratic_form(delta, es.covariance_rel_m2)
            yield {"record_id": es.record_id, "t_s": float(es.t_update_s), "nis": nis, "dof": dof,
                   "position_error_m": float(np.linalg.norm(delta[:3])),
                   "velocity_error_m_s": float(np.linalg.norm(delta[3:]))}

    obs = score_stream(obs_rows())
    est = score_stream(est_rows())

    def gate(stream: dict) -> bool | None:
        if stream["scored"] == 0:
            return None
        frac_ok = stream["gate_violations"] <= max_gate_violation_fraction * stream["scored"]
        ratio = stream["mean_nis_per_dof"]
        return bool(frac_ok and ratio is not None and ratio <= max_mean_ratio)

    return {"checks": {"navigation_inconsistency.observation_nis": gate(obs),
                       "navigation_inconsistency.estimate_nees": gate(est)},
            "observations": obs, "estimates": est,
            "gate_prob": float(gate_prob),
            "max_gate_violation_fraction": float(max_gate_violation_fraction),
            "max_mean_ratio": float(max_mean_ratio)}


# ---------------------------------------------------------------------------
# optimizer_control_nonconvergence
# ---------------------------------------------------------------------------

def maneuver_execution(log: GncRunLog, *, settle_fraction: float = 0.25,
                       settle_tol_m_s: float = 1.0e-4, margin: float = 8.0) -> dict:
    """Commanded-vs-realized execution and the settle-window read.

    * with a declared execution model (``config["execution"]`` =
      ``{"scale", "rtol"}``; the E13-01 reference model is
      realized_i = commanded_i * (scale + N(0, rtol)) per axis) every
      executed ControlEvent must sit within ``margin`` rtol-sigma of the
      declared model -- a swapped, re-signed or mis-calibrated actuator
      record fails here, whatever its digest says;
    * the settle window (last ``settle_fraction`` of the actuation
      timeline) is read for commands above ``settle_tol_m_s``: the loop is
      *settled* when none is left. A producer convergence assertion
      (``config["convergence"]["converged"]``) is cross-checked against it
      -- agreement passes, contradiction fails, absence is ``not_checked``.
    """
    events = sorted(log.records.get("control_event", ()), key=lambda e: e.t_exec_s)
    execution = log.config.get("execution")
    execution = dict(execution) if isinstance(execution, Mapping) else None
    dev_max = 0.0
    total_realized = np.zeros(3)
    declared_ok = None
    per_event = []
    for ev in events:
        cmd = np.asarray(ev.commanded_delta_v_m_s, dtype=np.float64)
        real = np.asarray(ev.realized_delta_v_m_s, dtype=np.float64)
        dev = real - cmd
        dev_max = max(dev_max, float(np.linalg.norm(dev)))
        if ev.executed:
            total_realized += real
        if execution is not None:
            scale = float(execution.get("scale", 1.0))
            rtol = float(execution.get("rtol", 0.0))
            expected = scale * cmd
            cmd_norm = float(np.linalg.norm(cmd))
            floor = 1.0e-9 * max(cmd_norm, 1.0)
            if float(np.linalg.norm(real - expected)) > margin * rtol * cmd_norm + floor:
                declared_ok = False
            elif declared_ok is None:
                declared_ok = True
        per_event.append({"record_id": ev.record_id, "t_exec_s": float(ev.t_exec_s),
                          "executed": bool(ev.executed),
                          "commanded_norm_m_s": float(np.linalg.norm(cmd)),
                          "execution_error_m_s": float(np.linalg.norm(dev))})

    t_last = float(events[-1].t_exec_s) if events else 0.0
    window = [ev for ev in events if ev.t_exec_s >= t_last * (1.0 - settle_fraction)]
    nonzero = [ev for ev in window
               if float(np.linalg.norm(ev.commanded_delta_v_m_s)) > settle_tol_m_s]
    settled = bool(events) and not nonzero

    assertion = log.config.get("convergence")
    assertion_value = bool(assertion["converged"]) if isinstance(assertion, Mapping) \
        and "converged" in assertion else None
    if assertion_value is None:
        assertion_check: bool | None = None
        assertion_reason: str | None = CONVERGENCE_ASSERTION_ABSENT
    else:
        assertion_check = bool(assertion_value == settled)
        assertion_reason = None

    return {"checks": {
                "optimizer_control_nonconvergence.execution_within_declared_bounds":
                    declared_ok,
                "optimizer_control_nonconvergence.convergence_assertion": assertion_check},
            "events": len(events), "execution_error_max_m_s": dev_max,
            "delta_v_realized_total_m_s": [float(x) for x in total_realized],
            "settle_window_events": len(window), "settle_window_nonzero": len(nonzero),
            "settled": settled, "declared_execution": execution,
            "convergence_assertion": assertion_value,
            "convergence_assertion_reason": assertion_reason,
            "command_max_m_s": max((e["commanded_norm_m_s"] for e in per_event), default=0.0),
            "margin": float(margin)}


# ---------------------------------------------------------------------------
# numerical_error: closed-loop replay under the recorded events, step sweep
# ---------------------------------------------------------------------------

def _pair_from_truth(log: GncRunLog):
    """Initial pair state + per-craft truth streams (target, chaser), both
    on one shared time grid (the contract's single clock makes this so; a
    log that broke it is refused here, not interpolated around)."""
    truth = sorted(log.records.get("truth", ()), key=lambda r: r.t_s)
    if not truth or abs(truth[0].t_s) > 1e-9:
        raise ValueError("closed-loop replay: truth stream must start at t=0")
    target_id = str(log.config["target_id"])
    chaser_id = str(log.config["chaser_id"])
    t_t = [r for r in truth if r.spacecraft_id == target_id]
    t_c = [r for r in truth if r.spacecraft_id == chaser_id]
    if len(t_t) < 2 or len(t_c) < 2:
        return None
    if [r.t_s for r in t_t] != [r.t_s for r in t_c]:
        raise ValueError("closed-loop replay: truth streams of target and chaser sit on "
                         "different time grids")
    return t_t, t_c


def closed_loop_replay(log: GncRunLog, *, sub_dt_s: float | None = None) -> dict:
    """Re-integrate the pair under the RECORDED control events: coast with
    ``propagate_pair`` between actuation instants, add each realized delta-v
    to the chaser at its ``t_exec_s`` (matching the run's declared event
    priority: a truth sample at an actuation instant is the PRE-impulse
    state). Returns the re-flown states at the recorded truth instants and
    the max errors against the recorded truth."""
    pair = _pair_from_truth(log)
    if pair is None:
        return {"available": False, "reason": TRUTH_TRACK_TOO_SHORT}
    t_t, t_c = pair
    times = [r.t_s for r in t_t]
    truth_t = np.array([[*r.r_m, *r.v_m_s] for r in t_t], dtype=np.float64)
    truth_c = np.array([[*r.r_m, *r.v_m_s] for r in t_c], dtype=np.float64)

    frame = str(log.config["frame"])
    mu = float(log.config.get("mu_m3_s2", EARTH_MU))
    j2 = float(log.config.get("j2", 0.0))
    impulse: dict[float, np.ndarray] = {}
    for ev in log.records.get("control_event", ()):
        if not ev.executed or float(np.linalg.norm(ev.realized_delta_v_m_s)) == 0.0:
            continue
        t = float(ev.t_exec_s)
        if t > times[-1] + 1e-9:
            continue  # an actuation after the last truth instant moves no recorded sample
        key = round(t, 6)
        impulse[key] = impulse.get(key, np.zeros(3)) + np.asarray(ev.realized_delta_v_m_s,
                                                                  dtype=np.float64)

    bounds = sorted({0.0, times[-1]} | set(impulse))
    state = np.concatenate([truth_t[0], truth_c[0]])
    samples: dict[float, np.ndarray] = {0.0: state.copy()}
    imp0 = impulse.get(0.0)
    if imp0 is not None:  # a burn at t=0: the recorded sample there is the pre-impulse state
        state = state.copy()
        state[9:12] += imp0
    for a, b in zip(bounds, bounds[1:]):
        want = [t for t in times if a < t < b]
        grid = np.array([0.0] + [t - a for t in want] + [b - a])
        traj = propagate_pair(state[:3], state[3:6], state[6:9], state[9:12], grid,
                              frame=frame, mu=mu, j2=j2, sub_dt=sub_dt_s,
                              where="gnc_closed_loop.closed_loop_replay")
        for j, t in enumerate(grid):
            key = round(a + float(t), 6)
            if j == 0 and key in samples:
                continue  # do not overwrite the pre-impulse state stored at an actuation instant
            samples[key] = np.concatenate(
                [traj.chief_r[0, j], traj.chief_v[0, j],
                 traj.deputy_r[0, j], traj.deputy_v[0, j]])
        state = samples[round(b, 6)].copy()
        imp = impulse.get(round(b, 6))
        if imp is not None:
            state[9:12] += imp  # chaser velocity, applied after the pre-impulse sample at b

    rep_t = np.array([samples[round(t, 6)][:3] for t in times])
    rep_tv = np.array([samples[round(t, 6)][3:6] for t in times])
    rep_c = np.array([samples[round(t, 6)][6:9] for t in times])
    rep_cv = np.array([samples[round(t, 6)][9:12] for t in times])
    err_pos = np.maximum(np.linalg.norm(rep_t - truth_t[:, :3], axis=1),
                         np.linalg.norm(rep_c - truth_c[:, :3], axis=1))
    err_vel = np.maximum(np.linalg.norm(rep_tv - truth_t[:, 3:], axis=1),
                         np.linalg.norm(rep_cv - truth_c[:, 3:], axis=1))
    err_rel = np.linalg.norm((rep_c - rep_t) - (truth_c[:, :3] - truth_t[:, :3]), axis=1)
    return {"available": True, "n_samples": len(times),
            "max_position_error_m": float(err_pos.max()),
            "max_velocity_error_m_s": float(err_vel.max()),
            "max_relative_position_error_m": float(err_rel.max())}


def replay_reference_tracks(log: GncRunLog, *,
                            refine_dt_s: float = REFERENCE_REFINE_DT_S,
                            ) -> dict[str, HermiteDenseOutput]:
    """Dense truth reference: the recorded Truth stream refined through the
    run's own declared dynamics, with the same event-splitting as
    :func:`closed_loop_replay`. Every inter-sample interval is re-propagated
    FROM ITS OWN RECORDED ENDPOINT STATES (endpoints stay exact; no
    drift accumulates against the record), sub-gridded at ``refine_dt_s``.
    At an actuation instant the node carries the PRE-impulse state (the
    run's event priority; E13-01's reference sim verifies the same
    convention sample-for-sample). Returns {} when the truth pair is
    missing or too short."""
    pair = _pair_from_truth(log)
    if pair is None:
        return {}
    t_t, t_c = pair
    times = [r.t_s for r in t_t]
    truth_t = np.array([[*r.r_m, *r.v_m_s] for r in t_t], dtype=np.float64)
    truth_c = np.array([[*r.r_m, *r.v_m_s] for r in t_c], dtype=np.float64)
    frame = str(log.config["frame"])
    mu = float(log.config.get("mu_m3_s2", EARTH_MU))
    j2 = float(log.config.get("j2", 0.0))
    impulse: dict[float, np.ndarray] = {}
    for ev in log.records.get("control_event", ()):
        if not ev.executed or float(np.linalg.norm(ev.realized_delta_v_m_s)) == 0.0:
            continue
        t = float(ev.t_exec_s)
        if t > times[-1] + 1e-9:
            continue
        key = round(t, 6)
        impulse[key] = impulse.get(key, np.zeros(3)) + np.asarray(ev.realized_delta_v_m_s,
                                                                  dtype=np.float64)
    h = max(float(refine_dt_s), 1.0e-6)
    bounds = sorted({0.0, float(times[-1])} | set(impulse)
                    | {round(float(t), 6) for t in times})
    state = np.concatenate([truth_t[0], truth_c[0]])
    nodes: dict[float, np.ndarray] = {0.0: state.copy()}
    imp0 = impulse.get(0.0)
    if imp0 is not None:
        state = state.copy()
        state[9:12] += imp0
    for a, b in zip(bounds, bounds[1:]):
        n = max(1, math.ceil((b - a) / h))
        grid = np.linspace(0.0, b - a, n + 1)
        traj = propagate_pair(state[:3], state[3:6], state[6:9], state[9:12], grid,
                              frame=frame, mu=mu, j2=j2,
                              where="gnc_closed_loop.replay_reference_tracks")
        for j, t in enumerate(grid):
            key = round(a + float(t), 6)
            if j == 0 and key in nodes:
                continue
            nodes[key] = np.concatenate([traj.chief_r[0, j], traj.chief_v[0, j],
                                         traj.deputy_r[0, j], traj.deputy_v[0, j]])
        state = nodes[round(b, 6)].copy()
        imp = impulse.get(round(b, 6))
        if imp is not None:
            state = state.copy()
            state[9:12] += imp
    epochs = np.array(sorted(nodes), dtype=np.float64)
    states = np.array([nodes[float(t)] for t in epochs], dtype=np.float64)
    if len(epochs) < 2:
        return {}
    return {str(log.config["target_id"]): HermiteDenseOutput(epochs, states[:, 0:6]),
            str(log.config["chaser_id"]): HermiteDenseOutput(epochs, states[:, 6:12])}


def numerical_convergence(log: GncRunLog, *, period_s: float | None = None,
                          step_divisors: Sequence[float] = (20.0, 40.0, 80.0)) -> dict:
    """Step refinement of the closed-loop replay against the recorded Truth:
    each coarse-to-fine doubling must not increase the error, the observed
    order between the two coarsest levels must reach 2 (RK4's asymptotic 4
    is not demanded -- the replay re-segments at actuation instants), and the
    tightest grid must match the recorded truth within
    :data:`REPLAY_TOL_POSITION_M` / :data:`REPLAY_TOL_VELOCITY_M_S`. Where a
    log's truth was produced on the very grid the replay uses (the E13-01
    reference sim), every level sits at the float-resolution floor and the
    order measurement is degenerate -- reported as ``at_float_floor`` and
    passed on that branch, never silently. An integrator error that matters
    (wrong mu, wrong frame, wrong force model) shows up far above the floor
    and fails the monotone/order/tolerance gates."""
    if period_s is None:
        truth = sorted(log.records.get("truth", ()), key=lambda r: r.t_s)
        if not truth:
            return {"checks": {"numerical_error.replay_step_convergence": None,
                               "numerical_error.replay_matches_recorded_truth": None},
                    "reason": TRUTH_TRACK_TOO_SHORT, "errors": []}
        period_s = 2.0 * math.pi * math.sqrt(float(np.linalg.norm(truth[0].r_m)) ** 3
                                             / float(log.config.get("mu_m3_s2", EARTH_MU)))
    if len(step_divisors) < 2:
        raise ValueError("numerical_convergence: step_divisors needs at least two levels")
    errors = []
    for n in step_divisors:
        rep = closed_loop_replay(log, sub_dt_s=float(period_s) / float(n))
        if not rep.get("available"):
            return {"checks": {"numerical_error.replay_step_convergence": None,
                               "numerical_error.replay_matches_recorded_truth": None},
                    "reason": rep.get("reason", TRUTH_TRACK_TOO_SHORT), "errors": []}
        errors.append({"step_s": float(period_s) / float(n), **rep})
    e = [x["max_position_error_m"] for x in errors]
    floor = REPLAY_FLOOR_POSITION_M
    at_floor = bool(e[-1] <= floor)
    monotone = all(e[i + 1] <= max(e[i], floor) * 1.05 for i in range(len(e) - 1))
    order = math.log(e[0] / e[1]) / math.log(2.0) if e[1] > floor and e[0] > e[1] else math.inf
    converged = bool(monotone and (order >= 2.0 or at_floor))
    return {"checks": {
                "numerical_error.replay_step_convergence": converged,
                "numerical_error.replay_matches_recorded_truth":
                    bool(e[-1] <= REPLAY_TOL_POSITION_M
                         and errors[-1]["max_velocity_error_m_s"] <= REPLAY_TOL_VELOCITY_M_S)},
            "errors": errors, "observed_order": None if order == math.inf else order,
            "at_float_floor": at_floor,
            "period_s": float(period_s), "step_divisors": [float(n) for n in step_divisors]}


# ---------------------------------------------------------------------------
# model_error: TudatPy closed-loop re-flight
# ---------------------------------------------------------------------------

_OBLIQUITY_J2000 = math.radians(84381.448 / 3600.0)  # IAU76: ECLIPJ2000 is GCRS about x


def _frame_rotation(log_frame: str) -> np.ndarray:
    """Constant GCRS -> ECLIPJ2000 rotation (E12-05's guard: the SPICE pool
    tudatpy loads defines no GCRS/ICRS frame at all, and a rejected frame
    makes tudat abort at t=0 WITHOUT raising, so the flight frame is chosen,
    not hoped for). Every graded quantity is a relative state or a norm,
    invariant under this one rotation of both crafts and every burn."""
    if log_frame == "ECLIPJ2000":
        return np.eye(3)
    if log_frame == "GCRS":
        c, s = math.cos(_OBLIQUITY_J2000), math.sin(_OBLIQUITY_J2000)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])
    raise ValueError(f"independent_reflight: log frame {log_frame!r} has no declared constant "
                     "rotation into ECLIPJ2000 (supported: GCRS, ECLIPJ2000)")


def _state_before_impulse(dense: HermiteDenseOutput, t: float) -> np.ndarray:
    """Dense state at ``t``, taken from the interval that ENDS at ``t`` when ``t`` sits on a node.

    The run records a truth sample at an actuation instant BEFORE the impulse
    (:func:`closed_loop_replay` states the same priority). ``HermiteDenseOutput``
    evaluates a seam epoch on the interval that STARTS there, i.e. after the
    burn, so asking it for ``t`` unpinned would compare the recorded pre-burn
    sample with a post-burn state and report the burn's own delta-v as an error.
    At a node that is not a seam both sides agree, so this is exact there too.
    """
    i = int(np.searchsorted(dense.epochs, t - 1e-9, side="left")) - 1
    on_node = 0 <= i < dense.epochs.size - 1 and abs(float(dense.epochs[i + 1]) - t) <= 1e-9
    return dense(t, interval=i) if on_node else dense(t)


def independent_reflight(log: GncRunLog, *, tol_position_m: float = REFLIGHT_TOL_POSITION_M,
                         tol_velocity_m_s: float = REFLIGHT_TOL_VELOCITY_M_S) -> dict:
    """Re-fly the Truth initial state + recorded ControlEvents in TudatPy.
    Without tudatpy in this interpreter: ``not_checked`` with
    :data:`REFLIGHT_BACKEND_MISSING` -- the independent line is recorded
    absent, never imputed (task #261's rule, the E12-05 policy)."""
    if importlib.util.find_spec("tudatpy") is None:
        return {"available": False, "reason": REFLIGHT_BACKEND_MISSING}
    pair = _pair_from_truth(log)
    if pair is None:
        return {"available": False, "reason": TRUTH_TRACK_TOO_SHORT}
    t_t, t_c = pair
    from orbitopt.verify.tudat_propagate import propagate_multi_arc
    from orbitopt.verify.tolerances import VERIFY_RTOL, VERIFY_ATOL

    frame = str(log.config["frame"])
    mu = float(log.config.get("mu_m3_s2", EARTH_MU))
    j2 = float(log.config.get("j2", 0.0))
    rot = _frame_rotation(frame)
    st0_t = rot @ np.asarray(t_t[0].r_m, dtype=np.float64), rot @ np.asarray(t_t[0].v_m_s,
                                                                              dtype=np.float64)
    st0_c = rot @ np.asarray(t_c[0].r_m, dtype=np.float64), rot @ np.asarray(t_c[0].v_m_s,
                                                                             dtype=np.float64)
    t_fly = float(max(float(log.config.get("t_end_s", 0.0)), t_t[-1].t_s))

    gravity = {"gravitational_parameter": mu, "reference_radius": EARTH_RE, "j2": j2}
    kw = dict(central_body="Earth", perturbing_bodies=("Earth",),
              earth_spherical_harmonic=(2, 0) if j2 else None,
              integrator="dop853", rtol=VERIFY_RTOL, atol=VERIFY_ATOL,
              central_gravity=gravity, frame="ECLIPJ2000")
    target = propagate_multi_arc(st0_t[0], st0_t[1], 0.0,
                                 [{"type": "coast", "duration": t_fly}], **kw)
    events = sorted((e for e in log.records.get("control_event", ()) if e.executed),
                    key=lambda e: e.t_exec_s)
    arcs: list[dict] = []
    t_prev = 0.0
    for ev in events:
        dv = rot @ np.asarray(ev.realized_delta_v_m_s, dtype=np.float64)
        if float(np.linalg.norm(dv)) == 0.0 or ev.t_exec_s > t_fly + 1e-9:
            continue
        if ev.t_exec_s > t_prev:
            arcs.append({"type": "coast", "duration": float(ev.t_exec_s) - t_prev})
            t_prev = float(ev.t_exec_s)
        arcs.append({"type": "impulsive_burn", "delta_v": dv})
    if t_fly > t_prev:
        arcs.append({"type": "coast", "duration": t_fly - t_prev})
    elif not any(a["type"] == "coast" for a in arcs):
        arcs.append({"type": "coast", "duration": 1.0e-9})
    chaser = propagate_multi_arc(st0_c[0], st0_c[1], 0.0, arcs, **kw)

    dense_t = HermiteDenseOutput(np.asarray(target.epochs, dtype=np.float64),
                                 np.asarray(target.states, dtype=np.float64))
    dense_c = HermiteDenseOutput(np.asarray(chaser.epochs, dtype=np.float64),
                                 np.asarray(chaser.states, dtype=np.float64))
    d = dict(target_pos=0.0, target_vel=0.0, chaser_pos=0.0, chaser_vel=0.0, rel_pos=0.0,
             rel_vel=0.0)
    for rec_t, rec_c in zip(t_t, t_c):
        t = float(rec_t.t_s)
        if t > dense_t.t1 + 1e-9 or t > dense_c.t1 + 1e-9:
            return {"available": False, "reason": "REFLIGHT_HISTORY_SHORT",
                    "t_last_truth": t}
        s_t, s_c = _state_before_impulse(dense_t, t), _state_before_impulse(dense_c, t)
        r_t = rot @ np.asarray(rec_t.r_m, dtype=np.float64)
        v_t = rot @ np.asarray(rec_t.v_m_s, dtype=np.float64)
        r_c = rot @ np.asarray(rec_c.r_m, dtype=np.float64)
        v_c = rot @ np.asarray(rec_c.v_m_s, dtype=np.float64)
        d["target_pos"] = max(d["target_pos"], float(np.linalg.norm(s_t[:3] - r_t)))
        d["target_vel"] = max(d["target_vel"], float(np.linalg.norm(s_t[3:] - v_t)))
        d["chaser_pos"] = max(d["chaser_pos"], float(np.linalg.norm(s_c[:3] - r_c)))
        d["chaser_vel"] = max(d["chaser_vel"], float(np.linalg.norm(s_c[3:] - v_c)))
        d["rel_pos"] = max(d["rel_pos"], float(np.linalg.norm((s_c[:3] - s_t[:3]) - (r_c - r_t))))
        d["rel_vel"] = max(d["rel_vel"], float(np.linalg.norm((s_c[3:] - s_t[3:]) - (v_c - v_t))))
    return {"available": True,
            "max_target_position_error_m": d["target_pos"],
            "max_target_velocity_error_m_s": d["target_vel"],
            "max_chaser_position_error_m": d["chaser_pos"],
            "max_chaser_velocity_error_m_s": d["chaser_vel"],
            "max_relative_position_error_m": d["rel_pos"],
            "max_relative_velocity_error_m_s": d["rel_vel"],
            "within_tolerance": bool(d["rel_pos"] <= tol_position_m
                                     and d["rel_vel"] <= tol_velocity_m_s),
            "tolerance": {"position_m": float(tol_position_m),
                          "velocity_m_s": float(tol_velocity_m_s)}}


# ---------------------------------------------------------------------------
# report assembly
# ---------------------------------------------------------------------------

def _as_log(log: GncRunLog | str | Path | Mapping) -> GncRunLog:
    if isinstance(log, GncRunLog):
        return log
    if isinstance(log, (str, Path)):
        from orbitopt.gnc.recording import read_run_log
        return read_run_log(log)
    from orbitopt.gnc.recording import run_log_from_document
    return run_log_from_document(log)


def _residual(name: str, category: str, evidence_type: str, passed: bool,
               reference: str, **extra) -> dict:
    """A hand-written residual for the report (the shape report.py accepts
    next to :func:`~orbitopt.verify.report.state_residual`): carries its
    category, its evidence qualification block and its verdict, so the
    status rollup counts it without pretending it is a state residual."""
    qual = EVIDENCE_QUALIFICATION[evidence_type]
    out = {"name": name, "reference": reference, "category": category,
           "evidence": {"type": evidence_type, "independent": bool(qual["independent"]),
                        "cannot_prove": qual["cannot_prove"]},
           "within_tolerance": bool(passed)}
    out.update(extra)
    return out


def _check_label(ok: bool | None) -> str:
    if ok is None:
        return RESULT_NOT_CHECKED
    return "passed" if ok else "failed"


def verification_digest_of(status: Mapping, error_classes: Mapping, run_digest: str) -> str:
    """E13-00-policy binding: sha256 over the EXACT verdict (the status
    checks + the four error-class lines) pinned to the run digest it graded.
    Flipping a verdict, relabelling an error class or swapping the run
    changes it."""
    return raw_sha256({"kind": BINDING_KIND, "run_digest": run_digest,
                       "checks": {k: bool(v) for k, v in status.get("checks", {}).items()},
                       "error_classes": error_classes})


def evidence_binding(log: GncRunLog, *, environment_label: str = "gnc-verify") -> dict:
    """Everything but ``verification_digest`` (needs the assembled status):
    run identity, optional sealed-plan binding, raw artifact digest of the
    canonical run-log document, environment provenance."""
    from orbitopt.crossenv.envelope import environment_digest
    doc = log.to_document(log.digest())
    return {"policy": "E13-00",
            "run_id": log.run_id,
            "run_digest": doc["run_digest"],
            "plan_digest": log.plan_digest,
            "artifact_sha256": raw_sha256(doc),
            "environment": str(environment_label),
            "environment_digest": environment_digest(environment_label)}


def verify_evidence_binding(report: Mapping, log: GncRunLog | str | Path | Mapping) -> None:
    """Fail closed if ``report``'s binding block does not recompute against
    ``log``: a report whose verdict was earned on another run (or a run-log
    edited after the report) raises."""
    log = _as_log(log)
    binding = report.get("inputs", {}).get("binding", {})
    fresh = evidence_binding(log, environment_label=binding.get("environment", "gnc-verify"))
    for field in ("run_digest", "plan_digest", "artifact_sha256"):
        if binding.get(field) != fresh[field]:
            raise ValueError(f"evidence binding mismatch: report {field} "
                             f"{binding.get(field)!r} != recomputed {fresh[field]!r}")
    want = verification_digest_of(report["status"], report["inputs"]["error_classes"],
                                  fresh["run_digest"])
    if binding.get("verification_digest") != want:
        raise ValueError("evidence binding mismatch: verification_digest does not bind "
                         "this verdict to this run")


def verify_gnc_closed_loop(log: GncRunLog | str | Path | Mapping, *,
                           out_path: str | Path | None = None,
                           gate_prob: float = NIS_GATE_PROB,
                           step_divisors: Sequence[float] = (20.0, 40.0, 80.0),
                           reflight_tol_position_m: float = REFLIGHT_TOL_POSITION_M,
                           reflight_tol_velocity_m_s: float = REFLIGHT_TOL_VELOCITY_M_S,
                           environment_label: str = "gnc-verify") -> dict:
    """Grade one sealed GNC run against itself-from-outside and return the
    digested verification-report-1.0 (``kind`` ``gnc-closed-loop``); with
    ``out_path`` also write it there. The four error classes are listed
    separately under ``inputs["error_classes"]``, each with per-check
    tri-state results, measured values and any not_checked reason."""
    log = _as_log(log)
    residuals: list[dict] = []
    checks: dict[str, bool | None] = {}
    reasons: dict[str, str | None] = {cls: None for cls in ERROR_CLASSES}

    # -- numerical_error -----------------------------------------------------
    num = numerical_convergence(log, step_divisors=step_divisors)
    checks.update(num["checks"])
    if num.get("reason"):
        reasons["numerical_error"] = num["reason"]
    else:
        residuals.append(_residual(
            "numerical_error.replay_step_convergence", CHECK_NUMERICAL, EVIDENCE_NUMERICAL,
            bool(num["checks"]["numerical_error.replay_step_convergence"]),
            "recorded truth (same model, independent step grid)",
            position_residual_m=num["errors"][-1]["max_position_error_m"],
            observed_order=num["observed_order"], step_sweep=num["errors"]))
        residuals.append(_residual(
            "numerical_error.replay_matches_recorded_truth", CHECK_NUMERICAL, EVIDENCE_NUMERICAL,
            bool(num["checks"]["numerical_error.replay_matches_recorded_truth"]),
            "recorded truth at the tightest replay grid",
            position_residual_m=num["errors"][-1]["max_position_error_m"],
            velocity_residual_m_s=num["errors"][-1]["max_velocity_error_m_s"],
            tolerance={"position_m": REPLAY_TOL_POSITION_M,
                       "velocity_m_s": REPLAY_TOL_VELOCITY_M_S}))

    # -- model_error (independent propagator; absent backend, never a pass) --
    fly = independent_reflight(log, tol_position_m=reflight_tol_position_m,
                               tol_velocity_m_s=reflight_tol_velocity_m_s)
    if fly.get("available"):
        checks["model_error.reflight_vs_recorded_truth"] = bool(fly["within_tolerance"])
        residuals.append(_residual(
            "model_error.reflight_vs_recorded_truth", CHECK_MODEL, EVIDENCE_INDEPENDENT_MODEL,
            bool(fly["within_tolerance"]),
            "recorded truth, TudatPy re-flight of the recorded ControlEvents",
            position_residual_m=fly["max_relative_position_error_m"],
            velocity_residual_m_s=fly["max_relative_velocity_error_m_s"],
            tolerance=fly["tolerance"], scope="full run, truth sample instants"))
    else:
        checks["model_error.reflight_vs_recorded_truth"] = None
        reasons["model_error"] = fly.get("reason", REFLIGHT_BACKEND_MISSING)

    # -- navigation_inconsistency --------------------------------------------
    nav = navigation_consistency(log, gate_prob=gate_prob)
    checks.update(nav["checks"])
    for stream, name, label in (("observations", "navigation_inconsistency.observation_nis",
                                 "NIS vs declared measurement covariance"),
                                ("estimates", "navigation_inconsistency.estimate_nees",
                                 "NEES vs declared estimate covariance")):
        ok = nav["checks"][name]
        s = nav[stream]
        if ok is None:
            reasons["navigation_inconsistency"] = reasons["navigation_inconsistency"] or \
                "STREAM_EMPTY"
            continue
        residuals.append(_residual(name, CHECK_MODEL, EVIDENCE_ANALYTICAL, bool(ok),
                                   f"interpolated truth stream ({label})",
                                   max_nis=s["max_nis"], mean_nis_per_dof=s["mean_nis_per_dof"],
                                   gate_violations=s["gate_violations"], scored=s["scored"]))

    # -- optimizer_control_nonconvergence -------------------------------------
    ctl = maneuver_execution(log)
    checks.update(ctl["checks"])
    for name in ("optimizer_control_nonconvergence.execution_within_declared_bounds",
                 "optimizer_control_nonconvergence.convergence_assertion"):
        ok = ctl["checks"][name]
        if ok is None:
            if name.endswith("convergence_assertion"):
                reasons["optimizer_control_nonconvergence"] = CONVERGENCE_ASSERTION_ABSENT
            continue
        residuals.append(_residual(name, CHECK_NUMERICAL, EVIDENCE_ANALYTICAL, bool(ok),
                                   "recorded command/control-event pair "
                                   + "(execution model / settle window)",
                                   execution_error_max_m_s=ctl["execution_error_max_m_s"],
                                   settle_window_nonzero=ctl["settle_window_nonzero"],
                                   settled=ctl["settled"]))

    # -- error classes, separately --------------------------------------------
    def line_of(prefix: str, reason_key: str) -> dict:
        names = {n: _check_label(v) for n, v in checks.items() if n.startswith(prefix)}
        ran = [r for r in names.values() if r != RESULT_NOT_CHECKED]
        result = (RESULT_NOT_CHECKED if not ran
                  else "failed" if "failed" in ran else "passed")
        line = {"result": result, "checks": names}
        if result == RESULT_NOT_CHECKED and reasons.get(reason_key):
            line["reason"] = reasons[reason_key]
        return line

    error_classes = {
        "numerical_error": line_of("numerical_error.", "numerical_error"),
        "model_error": line_of("model_error.", "model_error"),
        "navigation_inconsistency": line_of("navigation_inconsistency.", "navigation_inconsistency"),
        "optimizer_control_nonconvergence":
            line_of("optimizer_control_nonconvergence.", "optimizer_control_nonconvergence"),
    }

    def _summary(stream: dict) -> dict:
        return {"scored": stream["scored"], "gate_violations": stream["gate_violations"],
                "unscoreable": stream["unscoreable"], "max_nis": stream["max_nis"],
                "mean_nis_per_dof": stream["mean_nis_per_dof"]}

    inputs: dict[str, Any] = {
        "run": {"run_id": log.run_id, "frame": log.config.get("frame"),
                "target_id": log.config.get("target_id"), "chaser_id": log.config.get("chaser_id"),
                "mu_m3_s2": log.config.get("mu_m3_s2"), "j2": log.config.get("j2"),
                "t_end_s": log.config.get("t_end_s"),
                "record_counts": {k: len(v) for k, v in log.records.items()}},
        "error_classes": error_classes,
        "sections": {
            "numerical": {"errors": num["errors"], "observed_order": num.get("observed_order"),
                          "at_float_floor": num.get("at_float_floor"), "reason": num.get("reason")},
            "model": {k: v for k, v in fly.items() if k != "rows"},
            "navigation": {"gate_prob": nav["gate_prob"], "observations": _summary(nav["observations"]),
                           "estimates": _summary(nav["estimates"])},
            "control": {k: v for k, v in ctl.items() if k != "checks"},
        },
        "binding": evidence_binding(log, environment_label=environment_label),
    }
    report = build_report(REPORT_KIND, log.run_id, [], residuals, inputs=inputs)
    report["inputs"] = to_jsonable(dict(
        inputs, binding=dict(inputs["binding"], verification_digest=verification_digest_of(
            report["status"], error_classes, inputs["binding"]["run_digest"]))))
    report["digest"] = report_digest(report)
    if out_path is not None:
        write_report(report, out_path)
    return report
