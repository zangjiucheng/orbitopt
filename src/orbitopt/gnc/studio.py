"""GNC Mission Studio bridge: replay a sealed closed-loop run inside Flight Lab (E13-08).

One entry point, :func:`build_gnc_studio_run`, turns a sealed
:class:`orbitopt.gnc.recording.GncRunLog` (plus optionally the campaign
assessment that flew it and the E13-07 ``verification-report-1.0`` that graded
it) into the artefacts Flight Lab replays:

* a **FlightResult v1 run directory** (``orbitopt.flightlab.contract``) with
  three tracks on the run's own clock -- ``target`` and ``chaser`` truth and a
  third body ``chaser_est``, the navigation-filter's absolute view of the
  chaser (truth chief at the update instant + the record's relative state,
  exactly the anchor ``make_estimate`` used) -- so the replay shows truth vs
  estimate side by side;
* a ``gnc.json`` **sidecar** (schema ``gnc-studio-1.0``): measurement events
  with the frame-gap and dropout accounting, control events with the
  commanded-vs-realized pair, supervision state transitions, the estimation
  error / covariance / NEES series, sampled 3-sigma covariance ellipsoids, and
  two SEPARATE evidence blocks -- what the local replay shows versus what an
  independent verification report concluded, each with its source and digest.

Honesty rules this bridge keeps (E12-06 precedent, E13-07 evidence discipline):

* every curve is produced by the SAME evaluator the verification report uses
  (:func:`orbitopt.verify.gnc_closed_loop.navigation_consistency` rows -- so
  the plotted estimation error and NEES are the numbers the report grades, not
  a reimplementation); when a report is supplied the bridge compares the
  sidecar's summary numbers against the report's own sections and refuses to
  write anything if they disagree, and binds the report to the log through
  :func:`orbitopt.verify.gnc_closed_loop.verify_evidence_binding`;
* the report's ``not_checked`` is never rendered as a pass: a missing
  independent backend (model_error) sets ``physics_verified=None`` and the
  reason is recorded in the sidecar, exactly as E12-09 demanded;
* the local-replay block records what the log ITSELF proves (its digest,
  counts, dropouts) and is labelled as such -- conclusions drawn from the
  record being replayed are not independent of it; the independent block only
  ever mirrors the consumed report, never an inference of this bridge;
* ``chaser_est`` is display-derived (the record's relative state anchored on
  truth chief, linear between truth samples -- sub-centimetre at the truth
  cadence, stated in the manifest's assumptions); the auditable estimate is
  the record, not the track;
* masses are a constant illustrative 5000 kg and ``git_sha``/``created_utc``
  are fixed markers, so regenerating the directory reproduces it.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from orbitopt.flightlab.contract import (
    BodyTrack,
    Check,
    FlightRun,
    Verification,
    write_run,
)
from orbitopt.gnc.recording import GncRunLog, run_digest_of
from orbitopt.gnc.closed_loop import tracking_error
from orbitopt.timescales import mjd2000_to_utc_datetime

#: Sidecar schema tag (this module's own document, not a frozen v1 schema).
SIDECAR_SCHEMA_VERSION = "gnc-studio-1.0"

#: Constant mass on every track -- the GNC contract has no mass dynamics;
#: recorded as a manifest assumption, never as flown data (E12-06 precedent).
TRACK_MASS_KG = 5000.0

#: Fixed marker digests (E12-06 precedent): this run has no FlightTarget and
#: no vehicle file, so the identity fields are digests of what they stand for.
NO_VEHICLE_SHA256 = hashlib.sha256(b"orbitopt:e13-08:gnc-studio:no-vehicle").hexdigest()
MARKER_GIT_SHA = "0" * 40
MARKER_CREATED_UTC = "2026-01-01T00:00:00Z"

#: 3-sigma multiplier on the covariance eigen-axes for the displayed ellipsoid.
ELLIPSOID_SIGMA = 3.0

#: Every N-th estimate gets an ellipsoid sample in the sidecar/scene, and the
#: list is capped so a long run cannot bloat the artefact (the curves carry
#: the full history; the ellipsoids are sampled display metadata).
ELLIPSOID_EVERY = 4
ELLIPSOID_MAX = 60


@dataclass(frozen=True)
class GncStudioRun:
    """Everything the bridge produced: the Flight Lab run, the ``gnc.json``
    sidecar document, and the verification report document it consumed
    (``None`` when the replay carries no independent report)."""

    run: FlightRun
    sidecar: dict
    verification_report: dict | None


# ---------------------------------------------------------------------------
# truth extraction
# ---------------------------------------------------------------------------


def _truth_streams(log: GncRunLog) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t_s, target rows, chaser rows) on the run's shared truth grid."""
    target_id = str(log.config["target_id"])
    chaser_id = str(log.config["chaser_id"])
    by_t: dict[float, dict[str, object]] = {}
    for r in log.records.get("truth", ()):
        by_t.setdefault(float(r.t_s), {})[r.spacecraft_id] = r
    times = sorted(t for t, row in by_t.items() if target_id in row and chaser_id in row)
    if len(times) < 2:
        raise ValueError("gnc studio: the log carries no aligned truth grid")
    tgt = np.array([[*by_t[t][target_id].r_m, *by_t[t][target_id].v_m_s] for t in times],
                   dtype=np.float64)
    chs = np.array([[*by_t[t][chaser_id].r_m, *by_t[t][chaser_id].v_m_s] for t in times],
                   dtype=np.float64)
    return np.asarray(times, dtype=np.float64), tgt, chs


def _epoch_utc(log: GncRunLog) -> str:
    truth = sorted(log.records.get("truth", ()), key=lambda r: r.t_s)
    dt = mjd2000_to_utc_datetime(truth[0].epoch.to_utc_mjd2000())
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# series: the verification evaluator's own rows, on the scored-estimate schedule
# ---------------------------------------------------------------------------


def _cov_blocks(estimates) -> tuple[np.ndarray, np.ndarray]:
    """(trace of the 3x3 position block, sqrt of its largest diagonal) per
    estimate record -- the two scalar views of the position covariance."""
    trace, sig = [], []
    for e in estimates:
        c = np.asarray(e.covariance_rel_m2, dtype=np.float64).reshape(6, 6)
        pos = c[:3, :3]
        trace.append(float(np.trace(pos)))
        sig.append(float(math.sqrt(max(float(np.max(np.diag(pos))), 0.0))))
    return np.asarray(trace), np.asarray(sig)


# ---------------------------------------------------------------------------
# ellipsoids
# ---------------------------------------------------------------------------


def _ellipsoid_samples(log: GncRunLog, t_truth: np.ndarray, tgt: np.ndarray) -> list[dict]:
    """Sampled 3-sigma position-covariance ellipsoids in the run's inertial
    frame: centre = truth chief at the update instant (linear between truth
    samples) + the record's relative position; axes = ELLIPSOID_SIGMA * the
    square roots of the position block's eigenvalues; orientationBasis columns
    = the matching eigenvectors (already inertial: the record's covariance is
    expressed in the record's frame)."""
    out: list[dict] = []
    estimates = log.records.get("estimate", ())
    step = max(1, int(ELLIPSOID_EVERY))
    for i, e in enumerate(estimates):
        if i % step or len(out) >= ELLIPSOID_MAX:
            continue
        c = np.asarray(e.covariance_rel_m2, dtype=np.float64).reshape(6, 6)[:3, :3]
        vals, vecs = np.linalg.eigh(0.5 * (c + c.T))
        vals = np.clip(vals, 0.0, None)
        axes = ELLIPSOID_SIGMA * np.sqrt(vals)[::-1]
        vecs = vecs[:, ::-1]
        if np.linalg.det(vecs) < 0.0:
            vecs = vecs * -1.0
        chief = np.array([np.interp(e.t_update_s, t_truth, tgt[:, 0]),
                          np.interp(e.t_update_s, t_truth, tgt[:, 1]),
                          np.interp(e.t_update_s, t_truth, tgt[:, 2])], dtype=np.float64)
        center = chief + np.asarray(e.relative_state.r_rel_m, dtype=np.float64)
        out.append({"t_s": float(e.t_update_s),
                    "center_eci_m": [float(x) for x in center],
                    "semi_axes_m": [float(x) for x in axes],
                    "orientationBasis": [[float(vecs[r][k]) for k in range(3)]
                                         for r in range(3)]})
    return out


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------


def _dropout_events(log: GncRunLog) -> dict:
    """Frame-gap accounting per sensor: expected ticks (from the declared
    schedule on the run clock) minus the observations that actually arrived.
    A drop is the ABSENCE of a record -- the honest accounting is by schedule,
    never by hope."""
    t_end = float(log.config.get("t_end_s", 0.0))
    arrived: dict[str, list[float]] = {}
    for ob in log.records.get("observation", ()):
        arrived.setdefault(str(ob.sensor_id), []).append(float(ob.t_sample_s))
    per: dict[str, dict] = {}
    drops: list[dict] = []
    for spec in log.clock.sensors:
        sid = str(spec.sensor_id)
        period = float(spec.period_s)
        phase = float(spec.phase_s)
        got = np.array(sorted(arrived.get(sid, [])), dtype=np.float64)
        n_exp = int(math.floor((t_end - phase) / period)) + 1 if t_end >= phase else 0
        expected = phase + period * np.arange(max(n_exp, 0))
        missing = [float(t) for t in expected
                   if len(got) == 0 or np.min(np.abs(got - t)) > period * 0.25]
        declared = [int(t) for t in (spec.dropped_ticks or ())]
        per[sid] = {"period_s": period, "phase_s": phase, "expected_ticks": int(max(n_exp, 0)),
                    "received": int(len(got)), "dropped": len(missing),
                    "declared_dropped_ticks": declared}
        drops.extend({"t_s": t, "sensor_id": sid} for t in missing)
    return {"sensors": per, "drops": drops}


def _sidecar_events(log: GncRunLog, supervision: Mapping | None, drops: dict) -> list[dict]:
    events: list[dict] = []
    for d in drops["drops"]:
        events.append({"t_s": d["t_s"], "kind": "gnc_dropout", "body_id": "chaser",
                       "title": f"Sensor frame dropped ({d['sensor_id']})"})
    for ev in log.records.get("control_event", ()):
        cmd = float(np.linalg.norm(ev.commanded_delta_v_m_s))
        real = float(np.linalg.norm(ev.realized_delta_v_m_s))
        if cmd == 0.0 and real == 0.0:
            continue
        state = "executed" if ev.executed else "MISFIRED"
        events.append({"t_s": float(ev.t_exec_s), "kind": "gnc_burn", "body_id": "chaser",
                       "title": f"Burn {state} ({real:.3f} m/s of {cmd:.3f} commanded)"})
    if supervision is not None:
        for ev in supervision.get("events", ()):
            events.append({"t_s": float(ev["t_s"]), "kind": "gnc_supervision",
                           "body_id": "chaser",
                           "title": f"Supervisor {ev['state_from']} -> {ev['state_to']} "
                                    f"[{ev['trigger']}] {ev['detail']}".strip()})
    return sorted(events, key=lambda e: e["t_s"])


# ---------------------------------------------------------------------------
# evidence blocks
# ---------------------------------------------------------------------------


def _local_evidence_block(log: GncRunLog, log_path: str | Path | None) -> dict:
    block = {
        "source": "recorded run log replayed locally (NOT independent of itself)",
        "run_id": log.run_id,
        "run_digest": log.digest(),
        "record_counts": {k: len(v) for k, v in log.records.items()},
    }
    if log_path is not None:
        block["log_artifact_sha256"] = hashlib.sha256(
            Path(str(log_path)).read_bytes()).hexdigest()
    return block


def _independent_evidence_block(report: Mapping | None) -> dict | None:
    """Mirror of the consumed verification report -- its kind, digest, and the
    four error-class lines verbatim from ``inputs.error_classes``. A line that
    did not run is reported with its reason, never as a pass."""
    if report is None:
        return None
    classes = {cls: {"result": line.get("result"), "reason": line.get("reason"),
                     "checks": dict(line.get("checks") or {})}
               for cls, line in (report.get("inputs", {}).get("error_classes") or {}).items()}
    return {
        "source": "E13-07 verification-report-1.0, produced by orbitopt.verify."
                  "gnc_closed_loop against this run log",
        "report_kind": report.get("kind"),
        "report_digest": str(report.get("digest", "")),
        "error_classes": classes,
        "navigation": report.get("inputs", {}).get("sections", {}).get("navigation"),
    }


def _report_sections(report: Mapping) -> dict:
    return report.get("inputs", {}).get("sections", {}) or {}


def _guard_against_report(sidecar: dict, report: Mapping) -> None:
    """The sidecar's summary numbers must BE the report's, not merely close:
    the same evaluator produced both, and any drift means the artefacts shown
    in the studio are not the artefacts that were graded."""
    nav_rep = _report_sections(report).get("navigation") or {}
    for stream in ("observations", "estimates"):
        rep, got = nav_rep.get(stream) or {}, sidecar["nav_summary"][stream]
        for key in ("scored", "gate_violations", "unscoreable", "max_nis", "mean_nis_per_dof"):
            a, b = rep.get(key), got.get(key)
            if a is None or b is None:
                if a != b:
                    raise ValueError(f"gnc studio: {stream}.{key} disagrees with the "
                                     f"verification report ({a!r} vs {b!r})")
                continue
            if abs(float(a) - float(b)) > 1.0e-12:
                raise ValueError(f"gnc studio: {stream}.{key} disagrees with the "
                                 f"verification report ({a!r} vs {b!r})")
    ctl_rep = _report_sections(report).get("control") or {}
    for key in ("execution_error_max_m_s", "events"):
        a, b = ctl_rep.get(key), sidecar["control_summary"].get(key)
        if a is not None and b is not None and abs(float(a) - float(b)) > 1.0e-12:
            raise ValueError(f"gnc studio: control.{key} disagrees with the "
                             f"verification report ({a!r} vs {b!r})")


# ---------------------------------------------------------------------------
# the bridge
# ---------------------------------------------------------------------------


def build_gnc_studio_run(log: GncRunLog | str | Path, *,
                         assessment: Mapping | None = None,
                         verification_report: Mapping | str | Path | None = None,
                         title: str | None = None) -> GncStudioRun:
    """Replay a sealed GNC run as a Flight Lab run plus the ``gnc.json`` sidecar.

    ``assessment`` is the campaign's own assessment dict when the log came from
    :func:`orbitopt.rendezvous.gnc_campaign.fly_closed_loop_sample` -- its
    ``supervision`` block supplies the state-transition stream.
    ``verification_report`` (document or path) is the E13-07 report consumed;
    supplying it binds it to this log (digest check) and makes the independent
    evidence block real; without it the block is ``None`` and nothing is
    claimed about independent verification.
    """
    from orbitopt.verify.gnc_closed_loop import (maneuver_execution, navigation_consistency,
                                                 verify_evidence_binding)

    log_path: str | Path | None = log if isinstance(log, (str, Path)) else None
    if log_path is not None:
        from orbitopt.gnc.recording import read_run_log
        log = read_run_log(log_path)

    report_doc: dict | None = None
    report_sha: str | None = None
    if verification_report is not None:
        if isinstance(verification_report, (str, Path)):
            report_sha = hashlib.sha256(
                Path(str(verification_report)).read_bytes()).hexdigest()
            report_doc = json.loads(Path(str(verification_report)).read_text(encoding="utf-8"))
        else:
            report_doc = dict(verification_report)
        verify_evidence_binding(report_doc, log)

    frame = str(log.config["frame"])
    epoch_utc = _epoch_utc(log)
    t_truth, tgt, chs = _truth_streams(log)

    nav = navigation_consistency(log)
    ctl = maneuver_execution(log)
    track_err = tracking_error(log)
    series = _series_from(log, nav)
    ellipsoids = _ellipsoid_samples(log, t_truth, tgt)
    drops = _dropout_events(log)
    supervision = None if assessment is None else assessment.get("supervision")

    scored_estimates = [e for e in sorted(log.records.get("estimate", ()),
                                          key=lambda e: e.t_update_s)
                        if float(e.t_update_s) in set(series["est_times_s"])]
    est_anchor = _est_abs_track(scored_estimates, t_truth, tgt)

    target_track = _track("target", frame, epoch_utc, t_truth, tgt[:, :3], tgt[:, 3:])
    chaser_track = _track("chaser", frame, epoch_utc, t_truth, chs[:, :3], chs[:, 3:])
    est_track = _track("chaser_est", frame, epoch_utc, est_anchor[0], est_anchor[1],
                       est_anchor[2],
                       extras={
                           "gnc_est_pos_err_m": tuple(series["est_pos_err_m"]),
                           "gnc_est_vel_err_m_s": tuple(series["est_vel_err_m_s"]),
                           "gnc_cov_pos_trace_m2": tuple(series["cov_pos_trace_m2"]),
                           "gnc_cov_1sigma_rho_m": tuple(series["cov_1sigma_rho_m"]),
                           **({"gnc_nees_per_dof": tuple(series["est_nees_per_dof"])}
                              if all(v is not None for v in series["est_nees_per_dof"])
                              else {})})

    classes = (report_doc or {}).get("inputs", {}).get("error_classes") or {}

    def class_result(cls: str) -> str:
        return str((classes.get(cls) or {}).get("result", "not_checked"))

    physics = {"passed": True, "failed": False}.get(class_result("model_error"))
    mission_ok = (class_result("model_error") != "failed"
                  and class_result("numerical_error") == "passed"
                  and class_result("navigation_inconsistency") == "passed") \
        if classes else None
    converged = ctl["checks"].get(
        "optimizer_control_nonconvergence.execution_within_declared_bounds")

    checks = tuple(
        Check(id=f"gnc-{cls.replace('_', '-')}",
              passed=class_result(cls) == "passed",
              value=None, limit=None, unit="",
              reason_code="" if class_result(cls) in ("passed", "failed")
              else "not_checked")
        for cls in ("numerical_error", "model_error", "navigation_inconsistency",
                    "optimizer_control_nonconvergence")) if classes else ()

    run = FlightRun(
        run_id=f"{log.run_id}-studio",
        target_sha256=log.digest(),
        vehicle_sha256=NO_VEHICLE_SHA256,
        git_sha=MARKER_GIT_SHA,
        created_utc=MARKER_CREATED_UTC,
        assumptions=(
            (f"GNC studio replay of {log.run_id}: truth tracks are the record's Truth stream "
             "verbatim; the chaser_est track anchors the record's relative state on the truth "
             "chief position (linear between truth samples), exactly the anchor the estimator's "
             "make_estimate used, sampled at the scored-estimate instants -- display-derived, "
             "the auditable estimate is the record. Its gnc_* columns are the "
             "navigation_consistency rows verbatim."),
            (f"Every track carries a constant {TRACK_MASS_KG:,.0f} kg: the GNC contract has no "
             "mass dynamics (the mass row is illustrative, not flown data)."),
            ("Estimation error / NEES / covariance series in gnc.json are the "
             "verify.gnc_closed_loop navigation_consistency rows, sampled on the "
             "scored-estimate schedule -- the same rows and instants the "
             "verification report summarizes."),
            ("Covariance ellipsoids in the sidecar are sampled display metadata: every "
             f"{ELLIPSOID_EVERY}-th update, at most {ELLIPSOID_MAX} samples, "
             f"{ELLIPSOID_SIGMA:.0f}-sigma position-block axes in the run frame."),
            ("git_sha=0*40 and created_utc=2026-01-01T00:00:00Z are fixed markers so "
             "regenerating this run directory reproduces it."),
            ("backend omitted: the studio bridge is not a Flight Lab physics backend; "
             "GNC measurement/control/supervision events live in the gnc.json sidecar "
             "because the flight-result-1.0 event vocabulary is ascent-only."),
        ),
        tracks=(target_track, chaser_track, est_track),
        events=(),
        verification=Verification(
            optimizer_converged=None if converged is None else bool(converged),
            physics_verified=physics,
            mission_constraints_passed=None if mission_ok is None else bool(mission_ok),
            backend_self_check=None,
            checks=checks,
        ),
    )

    independent = _independent_evidence_block(report_doc)
    if report_sha is not None and independent is not None:
        independent["report_artifact_sha256"] = report_sha

    sidecar = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "run_id": log.run_id,
        "frame": frame,
        "epoch_utc": epoch_utc,
        "timeline_end_s": float(log.config.get("t_end_s", t_truth[-1])),
        "title": title or f"GNC closed-loop replay of {log.run_id}",
        "local_replay": _local_evidence_block(log, log_path),
        "independent_verification": independent,
        "nav_summary": {
            "gate_prob": nav["gate_prob"],
            "observations": _summary(nav["observations"]),
            "estimates": _summary(nav["estimates"]),
        },
        "control_summary": {
            "events": int(ctl["events"]),
            "execution_error_max_m_s": float(ctl["execution_error_max_m_s"]),
            "settled": bool(ctl["settled"]),
        },
        "tracking": {k: v for k, v in track_err.items() if isinstance(v, (int, float))},
        "measurement": drops,
        "supervision": None if supervision is None else {
            "final_state": supervision.get("final_state"),
            "events": [dict(e) for e in supervision.get("events", ())],
            "config": supervision.get("config"),
        },
        "series": series,
        "ellipsoids": ellipsoids,
        "ellipsoid_note": (f"sampled display metadata: every {ELLIPSOID_EVERY}-th estimate "
                           f"(max {ELLIPSOID_MAX}), {ELLIPSOID_SIGMA:.0f}-sigma position-block "
                           "eigen-axes, inertial orientationBasis columns; the full covariance "
                           "history stays in the sealed run log."),
        "events": _sidecar_events(log, supervision, drops),
    }
    if report_doc is not None:
        _guard_against_report(sidecar, report_doc)
    return GncStudioRun(run=run, sidecar=sidecar, verification_report=report_doc)


def _summary(stream: dict) -> dict:
    return {k: stream[k] for k in ("scored", "gate_violations", "unscoreable",
                                   "max_nis", "mean_nis_per_dof")}


# ---------------------------------------------------------------------------
# tracks / series helpers (defined after the bridge; order kept flat & explicit)
# ---------------------------------------------------------------------------


def _track(body_id: str, frame: str, epoch_utc: str, t_s: np.ndarray, r: np.ndarray,
           v: np.ndarray, extras: Mapping[str, np.ndarray] | None = None) -> BodyTrack:
    return BodyTrack(
        body_id=body_id, parent_body_id=None, frame=frame, epoch_utc=epoch_utc,
        provenance="official",
        t_s=tuple(float(x) for x in t_s),
        mass_kg=tuple([TRACK_MASS_KG] * len(t_s)),
        position_m=tuple(tuple(float(x) for x in row) for row in r),
        velocity_m_s=tuple(tuple(float(x) for x in row) for row in v),
        extra_columns={name: tuple(float(x) for x in col)
                       for name, col in (extras or {}).items()},
    )


def _est_abs_track(estimates, t_truth: np.ndarray, tgt: np.ndarray):
    """(t_s, position, velocity) of the estimator's absolute view: truth chief
    (linear between truth samples) + the record's relative state. Pass the
    SCORED estimate records so the display track's grid is exactly the grid
    the report's error/NEES rows are reported on."""
    estimates = sorted(estimates, key=lambda e: e.t_update_s)
    times = np.array([float(e.t_update_s) for e in estimates], dtype=np.float64)
    pos = np.column_stack([
        np.interp(times, t_truth, tgt[:, k])
        + np.array([e.relative_state.r_rel_m[k] for e in estimates])
        for k in range(3)])
    vel = np.column_stack([
        np.interp(times, t_truth, tgt[:, 3 + k])
        + np.array([e.relative_state.v_rel_m_s[k] for e in estimates])
        for k in range(3)])
    return times, pos, vel


def _series_from(log: GncRunLog, nav: dict) -> dict:
    """Curves straight off the verification evaluator's rows (see the module
    docstring), sampled on the scored-estimate schedule -- the panel plots
    them against ``est_times_s``, the same instants the report scored."""
    est_rows = nav["estimates"]["rows"]
    scored = [r for r in est_rows if r.get("gate_passed") is not None]
    est_t = np.array([r["t_s"] for r in scored], dtype=np.float64)
    estimates = log.records.get("estimate", ())
    est_all_t = np.array([e.t_update_s for e in estimates], dtype=np.float64)
    cov_trace, cov_sig = _cov_blocks(estimates)
    trace_at = np.array([cov_trace[i] for i in
                         (int(np.searchsorted(est_all_t, t + 1e-9) - 1) for t in est_t)],
                        dtype=np.float64)
    sig_at = np.array([cov_sig[i] for i in
                       (int(np.searchsorted(est_all_t, t + 1e-9) - 1) for t in est_t)],
                      dtype=np.float64)
    return {
        "est_times_s": [float(t) for t in est_t],
        "est_nis": [float(r["nis"]) for r in scored],
        "est_dof": [int(r["dof"]) for r in scored],
        "est_nees_per_dof": [float(r["nis"] / r["dof"]) if r["dof"] else None
                             for r in scored],
        "est_pos_err_m": [float(r["position_error_m"]) for r in scored],
        "est_vel_err_m_s": [float(r["velocity_error_m_s"]) for r in scored],
        "cov_pos_trace_m2": [float(v) for v in trace_at],
        "cov_1sigma_rho_m": [float(v) for v in sig_at],
    }


# ---------------------------------------------------------------------------
# LEO-RDV-001 reference replay + writer
# ---------------------------------------------------------------------------


def fly_leo_rdv_001_studio(*, tof_s: float = 3416.666, seed_base: int = 20000000,
                           cluster: int = 1, index: int = 5, supervised: bool = True,
                           verify: bool = True,
                           campaign_overrides: Mapping | None = None):
    """Fly and build the reference studio replay: the LEO-RDV-001 campaign
    sample the E13-06 tests use, supervised, verified with the E13-07 tool.

    Returns ``(GncStudioRun, outcome, assessment)``; ``verify=False`` skips the
    independent report (the studio then shows the local replay with NO
    independent block -- an honest gap, never a fake green)."""
    from orbitopt.rendezvous import load_reference_config
    from orbitopt.rendezvous import robustness as rb
    from orbitopt.rendezvous.gnc_campaign import ClosedLoopSampleConfig, fly_closed_loop_sample
    from orbitopt.verify.gnc_closed_loop import verify_gnc_closed_loop

    cfg = load_reference_config()
    campaign = rb.load_reference_campaign()
    if campaign_overrides:
        import dataclasses as dc
        campaign = dc.replace(campaign, **dict(campaign_overrides))
    supervisor = None
    if supervised:
        from orbitopt.gnc.supervisor import SupervisorConfig
        supervisor = SupervisorConfig()
    # The CW predictor omits differential J2 and nonlinear relative acceleration.
    # Allow roughly 14 mm/s of unmodelled velocity change over the 20 s sensor cadence.
    replay_filter = ClosedLoopSampleConfig(q_accel_m2_s3=1.0e-5)
    outcome, assessment, log = fly_closed_loop_sample(
        cfg, campaign, tof_s, seed_base, cluster, index, clc=replay_filter, supervisor=supervisor)
    if log is None:
        raise ValueError(f"fly_leo_rdv_001_studio: the sample failed to plan ({outcome.reason})")
    report = verify_gnc_closed_loop(log) if verify else None
    studio = build_gnc_studio_run(log, assessment=assessment, verification_report=report,
                                  title=f"LEO-RDV-001 closed-loop studio replay (sample "
                                        f"{seed_base}/{cluster}/{index})")
    return studio, outcome, assessment


def write_gnc_studio_run(dir: Path | str, studio: GncStudioRun) -> Path:
    """Materialise the studio run: the FlightResult v1 directory plus
    ``gnc.json`` and (when one was consumed) ``gnc_verification.json``."""
    root = Path(dir)
    write_run(root, studio.run)
    (root / "gnc.json").write_text(json.dumps(studio.sidecar) + "\n", encoding="utf-8")
    if studio.verification_report is not None:
        (root / "gnc_verification.json").write_text(
            json.dumps(studio.verification_report) + "\n", encoding="utf-8")
    return root
