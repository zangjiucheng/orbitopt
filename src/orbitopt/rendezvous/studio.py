"""Mission Studio bridge: replay a planned rendezvous inside Flight Lab (E12-06).

One entry point, :func:`build_studio_run`, turns a rendezvous mission config
(plan + independent verification included) into the artefacts Flight Lab
replays and the E12-01 multi-spacecraft contract audits:

* a **FlightResult v1 run directory** (``orbitopt.flightlab.contract``): two
  ``BodyTrack``s -- ``target`` and ``chaser`` -- on one shared epoch and one
  shared GCRS timeline, with the LVLH relative series (range, speed, x/y/z)
  carried as lossless chaser CSV extra columns;
* a ``multi-spacecraft-mission-1.0`` document (roster, keep-out sphere and
  hold-point volume, maneuver/hold events on the shared epoch) and its
  ``multispacecraft-replay-1.0`` unified replay, both validated before write;
* a ``rendezvous.json`` **sidecar** with the plan, the verification report,
  the maneuver events and the scene marker samples. The sidecar exists
  because the ``flight-result-1.0`` event vocabulary is ascent-only and the
  v1 schemas are frozen: rendezvous events never enter ``events.json``, the
  UI reads them here instead (E12-06 decision).

Honesty rules this bridge keeps:

* the timeline ENDS at the hold-point insertion burn -- no fabricated coast
  after arrival; the final chaser sample carries the POST-burn velocity, so
  the last relative-speed sample is exactly the verification report's
  ``relative_speed_m_s`` (same ``eci_to_lvlh`` call, same operands);
* propagation is the acceptance model itself
  (:func:`orbitopt.relative.propagate_pair`, two-body + J2 RK4), legs
  re-seeded at each impulsive burn exactly like ``verify_plan``;
* masses are a constant illustrative 5000 kg (the point-mass model has no
  mass dynamics) and ``git_sha``/``created_utc`` are fixed markers, so
  regenerating the run directory reproduces it; both go on record in the
  manifest's ``assumptions``;
* ``backend`` stays ``None``: the studio bridge is not a Flight Lab physics
  backend (the E1-01 vocabulary has no rendezvous engine).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from orbitopt.flightlab.contract import (
    BodyTrack,
    Check,
    FlightRun,
    Verification,
    write_run,
)
from orbitopt.mission.contract import Epoch
from orbitopt.mission.multispacecraft import (
    MissionEvent,
    MissionTolerances,
    MultiSpacecraftMission,
    SpacecraftSpec,
    Zone,
    build_unified_replay,
    mission_to_dict,
    validate_mission,
)
from orbitopt.relative.frames import eci_to_lvlh, lvlh_to_eci
from orbitopt.relative.propagation import EARTH_J2, propagate_pair
from orbitopt.rendezvous.config import RendezvousMissionConfig
from orbitopt.rendezvous.planner import initial_states, plan_rendezvous
from orbitopt.rendezvous.verify import verify_plan
from orbitopt.timescales import utc_datetime_to_mjd2000, utc_mjd2000_to_tdb_seconds

#: Sidecar schema tag (this module's own document, not a frozen v1 schema).
SIDECAR_SCHEMA_VERSION = "rendezvous-studio-1.0"

#: Shared mission epoch: t=0 of the config's timeline, a fixed calendar
#: instant so regenerated artefacts are identical (config times are elapsed
#: seconds from t0; the wall-clock anchor is a studio convention).
MISSION_EPOCH_UTC = "2026-01-01T00:00:00Z"

#: Constant mass on both tracks -- the point-mass rendezvous model has no
#: mass dynamics; recorded as a manifest assumption, never as flown data.
TRACK_MASS_KG = 5000.0

#: Keep-out sphere radius [m] centred on the target at arrival. The check is
#: the LIVE chaser-to-centre distance, not an observed zone crossing.
KEEPOUT_RADIUS_M = 500.0

#: Fixed marker digests (precedent: ``sixdof/two_stage.py``): this run has no
#: FlightTarget and no vehicle file, so its identity fields are the sha256 of
#: what they stand for -- the mission config, and a no-vehicle marker string.
NO_VEHICLE_SHA256 = hashlib.sha256(b"orbitopt:e12-06:rendezvous-studio:no-vehicle").hexdigest()
MARKER_GIT_SHA = "0" * 40
MARKER_CREATED_UTC = "2026-01-01T00:00:00Z"

#: Nominal sample step [s] of the replay grid (each leg is subdivided to it;
#: the integrator still sub-steps within it at ~500 steps/orbit).
DEFAULT_SAMPLE_DT_S = 5.0

_HOLD_POINT_ZONE = "hold-point"
_KEEPOUT_ZONE = "keepout"


def rendezvous_config_sha256(cfg: RendezvousMissionConfig) -> str:
    """Digest of the mission configuration the run replays (stands in for
    the FlightTarget digest of an ascent run): canonical JSON of every
    config field, sha256."""
    doc = {
        "mission_id": cfg.mission_id,
        "title": cfg.title,
        "dynamics": {"frame": cfg.dynamics.frame, "mu": cfg.dynamics.mu, "j2": cfg.dynamics.j2},
        "target": {"altitude_m": cfg.target.altitude_m,
                   "inclination_deg": cfg.target.inclination_deg,
                   "raan_deg": cfg.target.raan_deg,
                   "argument_of_latitude_deg": cfg.target.argument_of_latitude_deg},
        "chaser": {"altitude_m": cfg.chaser.altitude_m,
                   "initial_separation_m": cfg.chaser.initial_separation_m},
        "hold_point_rel_lvlh": list(cfg.hold_point_rel_lvlh),
        "delta_v_budget_m_s": cfg.delta_v_budget_m_s,
        "max_position_error_m": cfg.max_position_error_m,
        "max_relative_speed_m_s": cfg.max_relative_speed_m_s,
        "windows": {"drift_max_s": cfg.windows.drift_max_s,
                    "transfer_tof_min_s": cfg.windows.transfer_tof_min_s,
                    "transfer_tof_max_s": cfg.windows.transfer_tof_max_s},
        "phasing": {"departure_gate_y_m": cfg.phasing.departure_gate_y_m,
                    "gate_scan_step_s": cfg.phasing.gate_scan_step_s},
        "optimizer": {"newton_tol_m": cfg.optimizer.newton_tol_m,
                      "newton_max_iter": cfg.optimizer.newton_max_iter,
                      "jacobian_eps_m_s": cfg.optimizer.jacobian_eps_m_s,
                      "golden_max_iter": cfg.optimizer.golden_max_iter},
    }
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StudioRun:
    """Everything the bridge produced: the Flight Lab run, the E12-01 mission
    and replay documents, and the ``rendezvous.json`` sidecar document."""

    run: FlightRun
    mission_doc: dict
    replay_doc: dict
    sidecar: dict
    keepout_min_range_m: float


def _leg_grid(t_start: float, t_end: float, dt: float) -> np.ndarray:
    n = max(2, round((t_end - t_start) / dt) + 1)
    return np.linspace(t_start, t_end, n)


def _norms(rows: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(rows, dtype=float).reshape(-1, 3), axis=1)


def build_studio_run(cfg: RendezvousMissionConfig, *, sample_dt_s: float = DEFAULT_SAMPLE_DT_S,
                     tolerances: MissionTolerances | None = None) -> StudioRun:
    """Plan, independently verify, and replay ``cfg`` as a Flight Lab run.

    Raises what ``plan_rendezvous``/``verify_plan``/``validate_mission`` raise
    -- a mission that cannot be planned or whose tracks disagree with the
    shared reference produces no artefacts.
    """
    if not (math.isfinite(sample_dt_s) and 0.5 <= sample_dt_s <= 60.0):
        raise ValueError(f"sample_dt_s must be in [0.5, 60] s, got {sample_dt_s!r}")
    plan = plan_rendezvous(cfg)
    ver = verify_plan(cfg, plan)
    frame, mu = cfg.dynamics.frame, cfg.dynamics.mu
    j2 = EARTH_J2 if cfg.dynamics.j2 else 0.0
    m1, m2 = plan.maneuvers
    t1 = float(plan.drift_end_s)
    t2 = float(plan.arrival_epoch_s)

    r0t, v0t, r0c, v0c = initial_states(cfg)
    dv1 = np.asarray(m1.delta_v_m_s, dtype=float)
    dv2 = np.asarray(m2.delta_v_m_s, dtype=float)
    hold = np.asarray(cfg.hold_point_rel_lvlh, dtype=float)
    leg_a = propagate_pair(r0t, v0t, r0c, v0c, _leg_grid(0.0, t1, sample_dt_s),
                           frame=frame, mu=mu, j2=j2, where="studio: drift leg")
    rt1, vt1 = leg_a.chief_r[0, -1], leg_a.chief_v[0, -1]
    rc1, vc1 = leg_a.deputy_r[0, -1], leg_a.deputy_v[0, -1]
    leg_b = propagate_pair(rt1, vt1, rc1, vc1 + dv1, _leg_grid(0.0, t2 - t1, sample_dt_s),
                           frame=frame, mu=mu, j2=j2, where="studio: transfer leg")
    # Arrival re-derived on verify_plan's own two-point legs from the t0
    # states (not the dense grid's end, whose sub-stepping differs slightly),
    # so the last CSV row grades to the report's numbers to the bit.
    leg_a_exact = propagate_pair(r0t, v0t, r0c, v0c, np.array([0.0, t1]),
                                 frame=frame, mu=mu, j2=j2, where="studio: drift leg (exact)")
    rt1x, vt1x = leg_a_exact.chief_r[0, -1], leg_a_exact.chief_v[0, -1]
    rc1x, vc1x = leg_a_exact.deputy_r[0, -1], leg_a_exact.deputy_v[0, -1]
    arrival = propagate_pair(rt1x, vt1x, rc1x, vc1x + dv1, np.array([0.0, plan.tof_s]),
                             frame=frame, mu=mu, j2=j2, where="studio: arrival")
    leg_b.chief_r[0, -1], leg_b.chief_v[0, -1] = arrival.chief_r[0, -1], arrival.chief_v[0, -1]
    leg_b.deputy_r[0, -1], leg_b.deputy_v[0, -1] = arrival.deputy_r[0, -1], arrival.deputy_v[0, -1]

    t_s = np.concatenate((leg_a.times, t1 + leg_b.times[1:]))
    t_s[-1] = t2  # exact arrival on the timeline (grid addition can sit an ulp off)
    target_r = np.concatenate((leg_a.chief_r[0], leg_b.chief_r[0][1:]))
    target_v = np.concatenate((leg_a.chief_v[0], leg_b.chief_v[0][1:]))
    chaser_r = np.concatenate((leg_a.deputy_r[0], leg_b.deputy_r[0][1:]))
    chaser_v = np.concatenate((leg_a.deputy_v[0], leg_b.deputy_v[0][1:]))
    rel = np.concatenate((leg_a.rel_lvlh[0], leg_b.rel_lvlh[0][1:]))

    rt2, vt2 = leg_b.chief_r[0, -1], leg_b.chief_v[0, -1]
    rc2 = leg_b.deputy_r[0, -1]
    vc2_post = leg_b.deputy_v[0, -1] + dv2          # the burn is part of the arrival
    rho_arr, rho_dot_arr = eci_to_lvlh(rt2, vt2, rc2 - rt2, vc2_post - vt2, frame,
                                       where="studio: arrival")
    rel[-1, :3], rel[-1, 3:] = rho_arr, rho_dot_arr
    chaser_v[-1] = vc2_post
    _check_against_report(rho_arr, rho_dot_arr, hold, ver)

    rel_range = _norms(rel[:, :3])
    rel_speed = _norms(rel[:, 3:])
    keepout_center = np.asarray(rt2, dtype=float)
    keepout_min = float(np.min(_norms(chaser_r - keepout_center)))

    mjd0 = utc_datetime_to_mjd2000(
        datetime.strptime(MISSION_EPOCH_UTC, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc))
    tdb0 = utc_mjd2000_to_tdb_seconds(mjd0)

    def epoch_at(t: float) -> Epoch:
        # TDB label: to_tdb_seconds() is then exact, so an event at track end
        # is inside coverage (UTC labels would drift ms against the track).
        return Epoch(mjd2000=(tdb0 + t) / 86400.0 + 0.5, timescale="TDB")

    hold_center_eci = np.asarray(
        lvlh_to_eci(rt2, vt2, hold, np.zeros(3), frame, where="studio: hold zone")[0], dtype=float)

    target_track = _track("target", frame, t_s, target_r, target_v)
    chaser_track = _track("chaser", frame, t_s, chaser_r, chaser_v, extras={
        "rel_range_m": rel_range,
        "rel_speed_m_s": rel_speed,
        "rel_x_lvlh_m": rel[:, 0],
        "rel_y_lvlh_m": rel[:, 1],
        "rel_z_lvlh_m": rel[:, 2],
    })

    run = FlightRun(
        run_id=f"{cfg.mission_id.lower()}-studio",
        target_sha256=rendezvous_config_sha256(cfg),
        vehicle_sha256=NO_VEHICLE_SHA256,
        git_sha=MARKER_GIT_SHA,
        created_utc=MARKER_CREATED_UTC,
        assumptions=(
            (f"Mission studio replay of {cfg.mission_id}: two-body + J2 (j2={cfg.dynamics.j2}) "
             "RK4 pair propagation on the config's frame/mu; burns impulsive at the planned epochs."),
            (f"Both tracks carry a constant {TRACK_MASS_KG:,.0f} kg: the point-mass rendezvous "
             "model has no mass dynamics (the mass row is illustrative, not flown data)."),
            ("The timeline ends at the hold-point insertion burn (arrival); no coast segment "
             "is appended after arrival."),
            (f"git_sha={MARKER_GIT_SHA} and created_utc={MARKER_CREATED_UTC} are fixed markers "
             "so regenerating this run directory reproduces it."),
            ("backend omitted: the studio bridge is not a Flight Lab physics backend; "
             "maneuver/hold/zone events live in the rendezvous.json sidecar because the "
             "flight-result-1.0 event vocabulary is ascent-only."),
            (f"keep-out sphere radius {KEEPOUT_RADIUS_M:.0f} m centred on the target at "
             "arrival; the check grades the live chaser-to-centre distance, not an observed "
             "zone crossing."),
        ),
        tracks=(target_track, chaser_track),
        events=(),
        verification=Verification(
            optimizer_converged=True,
            physics_verified=True,
            mission_constraints_passed=bool(ver.passed),
            backend_self_check=None,
            checks=_run_checks(cfg, plan, ver, keepout_min),
        ),
    )

    mission = MultiSpacecraftMission(
        mission_id=cfg.mission_id,
        title=f"{cfg.title} — Mission Studio replay",
        frame=frame,
        epoch=Epoch(mjd2000=mjd0, timescale="UTC"),
        spacecraft=(
            SpacecraftSpec("target", "TARGET", {"altitude_m": str(cfg.target.altitude_m)}),
            SpacecraftSpec("chaser", "CHASER", {"altitude_m": str(cfg.chaser.altitude_m)}),
        ),
        zones=(
            Zone(_KEEPOUT_ZONE, center_m=tuple(float(x) for x in keepout_center),
                 radius_m=KEEPOUT_RADIUS_M),
            Zone(_HOLD_POINT_ZONE, center_m=tuple(float(x) for x in hold_center_eci),
                 radius_m=cfg.max_position_error_m),
        ),
        events=(
            MissionEvent("chaser", "maneuver", epoch_at(t1), counterpart_id="target",
                         data={"maneuver": "transfer_injection",
                               "delta_v_m_s": f"{float(np.linalg.norm(dv1)):.6f}"}),
            MissionEvent("chaser", "hold_point_arrival", epoch_at(t2), counterpart_id="target",
                         hold_point_id=_HOLD_POINT_ZONE,
                         data={"position_error_m": f"{ver.position_error_m:.6e}",
                               "relative_speed_m_s": f"{ver.relative_speed_m_s:.6e}"}),
        ),
    )
    report = validate_mission(mission, (target_track, chaser_track), tolerances)
    if not report.passed:
        raise ValueError("studio mission validation failed: "
                         + "; ".join(f"{c.id}({c.reason_code})" for c in report.failures))
    replay_doc = build_unified_replay(mission, (target_track, chaser_track), tolerances)

    sidecar = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "mission_id": cfg.mission_id,
        "config_sha256": run.target_sha256,
        "epoch_utc": MISSION_EPOCH_UTC,
        "timeline_end_s": t2,
        "plan": plan.to_dict(),
        "verification": ver.to_dict(),
        "events": [
            {"t_s": t1, "kind": "maneuver", "body_id": "chaser", "counterpart_id": "target",
             "title": f"Maneuver · transfer injection (Δv {np.linalg.norm(dv1):.3f} m/s)"},
            {"t_s": t2, "kind": "hold_point_arrival", "body_id": "chaser",
             "counterpart_id": "target", "hold_point_id": _HOLD_POINT_ZONE,
             "title": f"Hold-point insertion (Δv {np.linalg.norm(dv2):.3f} m/s)"},
        ],
        "keepout": {
            "zone": _KEEPOUT_ZONE,
            "radius_m": KEEPOUT_RADIUS_M,
            "center_eci_at_arrival_m": [float(x) for x in keepout_center],
            "min_range_m": keepout_min,
            "respected": keepout_min >= KEEPOUT_RADIUS_M,
        },
        "hold_point": {
            "zone": _HOLD_POINT_ZONE,
            "rel_lvlh_m": [float(x) for x in hold],
            "tolerance_m": float(cfg.max_position_error_m),
            "center_eci_at_arrival_m": [float(x) for x in hold_center_eci],
            "position_error_m": ver.position_error_m,
            "relative_speed_m_s": ver.relative_speed_m_s,
        },
        "markers": [
            {"id": "keepout_center", "name": "Keep-out centre (target @ arrival)",
             "color": "#e05d5d", "times_s": [float(t) for t in t_s],
             "positions_m": [[float(x) for x in row] for row in target_r]},
            {"id": "hold_point", "name": f"Hold point (LVLH {hold[0]:.0f}, {hold[1]:.0f}, {hold[2]:.0f} m)",
             "color": "#f2c66d", "times_s": [float(t) for t in t_s],
             "positions_m": [[float(x) for x in row] for row in _hold_track(target_r, target_v, hold, frame)]},
        ],
    }
    return StudioRun(run=run, mission_doc=mission_to_dict(mission), replay_doc=replay_doc,
                     sidecar=sidecar, keepout_min_range_m=keepout_min)


def _plan(cfg: RendezvousMissionConfig):
    from orbitopt.rendezvous.planner import plan_rendezvous
    return plan_rendezvous(cfg)


def _check_against_report(rho: np.ndarray, rho_dot: np.ndarray, hold: np.ndarray,
                          ver) -> None:
    """The last replay sample must grade to the verification report's numbers
    (the same frames call on the same states -- a drift here means the replay
    diverged from the verified plan, and every curve plotted against it would
    silently disagree with the report)."""
    err = max(
        abs(float(np.linalg.norm(rho - hold)) - ver.position_error_m),
        abs(float(np.linalg.norm(rho_dot)) - ver.relative_speed_m_s),
    )
    if not err <= 1e-9:
        raise ValueError(f"studio replay disagrees with the verification report by {err:g} (m / m/s)")


def _hold_track(target_r: np.ndarray, target_v: np.ndarray, hold: np.ndarray, frame: str):
    """Hold-point ECI position at every target sample: the fixed LVLH offset
    carried by the target's moving frame."""
    return np.array([lvlh_to_eci(r, v, hold, np.zeros(3), frame, where="studio: hold marker")[0]
                     for r, v in zip(target_r, target_v)])


def _track(body_id: str, frame: str, t_s: np.ndarray, r: np.ndarray, v: np.ndarray,
           extras: dict[str, np.ndarray] | None = None) -> BodyTrack:
    return BodyTrack(
        body_id=body_id, parent_body_id=None, frame=frame, epoch_utc=MISSION_EPOCH_UTC,
        provenance="official",
        t_s=tuple(float(x) for x in t_s),
        mass_kg=tuple([TRACK_MASS_KG] * len(t_s)),
        position_m=tuple(tuple(float(x) for x in row) for row in r),
        velocity_m_s=tuple(tuple(float(x) for x in row) for row in v),
        extra_columns={name: tuple(float(x) for x in col) for name, col in (extras or {}).items()},
    )


def _run_checks(cfg: RendezvousMissionConfig, plan, ver, keepout_min: float) -> tuple[Check, ...]:
    named = {
        "gate-within-drift-window": (ver.checks["gate_within_drift_window"],
                                     plan.drift_end_s, cfg.windows.drift_max_s, "s"),
        "tof-within-transfer-window": (ver.checks["tof_within_transfer_window"],
                                       plan.tof_s, cfg.windows.transfer_tof_max_s, "s"),
        "position-error-within-tolerance": (ver.checks["position_error_within_tolerance"],
                                            ver.position_error_m, cfg.max_position_error_m, "m"),
        "relative-speed-within-tolerance": (ver.checks["relative_speed_within_tolerance"],
                                            ver.relative_speed_m_s, cfg.max_relative_speed_m_s, "m/s"),
        "delta-v-within-budget": (ver.checks["delta_v_within_budget"],
                                  ver.total_delta_v_m_s, cfg.delta_v_budget_m_s, "m/s"),
        "keepout-respected": (keepout_min >= KEEPOUT_RADIUS_M,
                              keepout_min, KEEPOUT_RADIUS_M, "m"),
    }
    return tuple(Check(id=cid, passed=passed, value=float(value), limit=float(limit), unit=unit)
                 for cid, (passed, value, limit, unit) in named.items())


def write_studio_run(dir: Path | str, studio: StudioRun) -> Path:
    """Materialise the studio run: the FlightResult v1 directory plus
    ``rendezvous.json``, ``mission.json`` and ``replay.json``."""
    root = Path(dir)
    write_run(root, studio.run)
    (root / "mission.json").write_text(
        json.dumps(studio.mission_doc, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    (root / "replay.json").write_text(json.dumps(studio.replay_doc) + "\n", encoding="utf-8")
    (root / "rendezvous.json").write_text(json.dumps(studio.sidecar) + "\n", encoding="utf-8")
    return root
