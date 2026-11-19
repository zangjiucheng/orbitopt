"""E8-01 full-chain reference validation of the E5-05 reference mission.

This module grades the *recorded* E5-05 full mission -- ascent, staging,
fairing jettison, SECO/insertion, coast, booster boostback/entry/landing --
against independent references committed alongside it, and writes one
:class:`orbitopt.verify.report`-shaped ``VerificationReport`` (kind
``full-chain-reference``). It runs no simulation: every number compared is a
committed artifact under the E5-05 reference root, whose digests are bound by
``PROVENANCE.json`` (the same tamper check
``scripts/e5-05-reference.py``'s ``verify`` command applies, replicated by
:func:`verify_reference_digests` so this report cannot be graded off edited
data).

The reference sources (per the E4-03 evidence taxonomy of task #263)
---------------------------------------------------------------------
* ``independent_model`` -- the shared ballistic window only: the raw
  OpenTsiolkovsky debris track against the independently formulated recovery
  model's booster track (gfold-planned, repo-propagated) between separation
  and boostback ignition. **E8-05 relabel:** the ascent event and insertion
  comparisons against OpenTsiolkovsky's *own raw output* used to be labelled
  here; they compare the chain against the program that computed the ascent
  and are extraction/serialisation checks, so they are now
  ``numerical_consistency`` / ``analytical``. Independent-model evidence for
  an OpenTsiolkovsky-computed ascent is provided separately, by the native
  point-mass twin of :mod:`orbitopt.verify.independent_dynamics` (E8-05) on
  the E5-07 reference.
* ``analytical`` -- closed forms over the recorded states: Keplerian
  elements recomputed from the recorded SECO state (two-body identities of
  vis-viva and the period), the insertion elements against the chain's
  achieved orbit, the Tsiolkovsky ideal-Δv bound on the stage-2 burn, and
  the two-body propagation of the booster's boostback-coast arc up to the
  entry interface.
* ``numerical_consistency`` -- agreement inside the sealed chain record:
  the contract track against the frame-bridged raw backend state, event
  times extracted independently (powered/throttle transitions, mass
  bookkeeping) from the raw state history, chain event records against those
  extractions.
* ``design_input`` -- the landing gates at touchdown, graded against the
  flight target's own limits. Honestly labelled: this is *not* independent
  validation, and no telemetry of this synthetic vehicle exists. There is
  no ``historical_telemetry`` evidence anywhere in this report.

Honesty notes (the gate must not over-claim, E8-03 discipline)
--------------------------------------------------------------
* The ascent was computed *by* OpenTsiolkovsky; since E8-05 the event and
  insertion comparisons against its raw output are labelled as what they
  are -- extraction and closed-form mechanics over the sealed record, not
  independent-model evidence for the ascent physics. The shared ballistic
  window keeps ``independent_model``: there the comparison is OpenTsiolkovsky
  against the repo's own recovery propagator over the same unpowered arc.
* The recovery leg was driven from the recorded separation state; the
  shared ballistic window before the boostback ignition (both records
  unpowered, same body, same environment) is what carries the recovery
  leg's independent-model evidence.
* Landing itself is graded only by closed-form coast propagation
  (analytical), record consistency (numerical) and target gates
  (design_input). The report's ``inputs.evidence_map`` says this per check.

Tolerances are declared in :data:`FULL_CHAIN_TOLERANCES` with an auditable
``basis`` for each band; :func:`build_full_chain_report` embeds the table and
the reference digests in the report so a reviewer never needs this source.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
REFERENCE_ROOT = REPO / "tests" / "data" / "mission" / "reference" / "e5-05-reference-mission"
E505_SCRIPT = REPO / "scripts" / "e505_reference.py"

#: Report kind of the document :func:`build_full_chain_report` returns.
FULL_CHAIN_REPORT_KIND = "full-chain-reference"

G0 = 9.80665  # m/s^2, standard gravity (Tsiolkovsky conversion)

#: One band per graded check: unit, limit and the auditable *basis* of the
#: number (recorder grid resolution, J2 magnitude, model difference budget).
#: Observed residuals at commit time are recorded in ``docs/verification.md``
#: (E8-01) and in ``inputs.check_details`` of every report.
FULL_CHAIN_TOLERANCES: dict[str, dict] = {
    "ascent.event.meco": {
        "unit": "s", "limit": 2.0,
        "basis": "OpenTsiolkovsky output grid is 2 s; the event must fall "
                 "within one output interval of the thrust-off transition "
                 "extracted from the raw state history.",
    },
    "ascent.event.separation": {
        "unit": "s", "limit": 2.0,
        "basis": "same 2 s raw grid: the staging mass step is bracketed by "
                 "the two enclosing raw samples, so +-2 s is the extraction "
                 "resolution, not a physical slack.",
    },
    "ascent.event.fairing_jettison": {
        "unit": "s", "limit": 0.5,
        "basis": "the raw fairing body track begins at the jettison sample; "
                 "event vs track start is exact on the 2 s grid, 0.5 s bands "
                 "sample rounding.",
    },
    "ascent.event.seco": {
        "unit": "s", "limit": 2.0,
        "basis": "same 2 s raw grid around the stage-2 thrust-off "
                 "transition (burn-end sample appended by the recorder).",
    },
    "ascent.separation_window_position": {
        "unit": "m", "limit": 25.0,
        "basis": "shared ballistic window (separation -> booster ignition, "
                 "throttle 0 in both records): raw OT debris track "
                 "(interpolated across its 2 s grid) vs the gfold recovery "
                 "booster track. Bands the atmosphere-model difference "
                 "between OpenTsiolkovsky and the gfold standard atmosphere "
                 "plus linear-interpolation error; observed < 5 m.",
    },
    "ascent.separation_window_velocity": {
        "unit": "m/s", "limit": 0.05,
        "basis": "same shared ballistic window; observed ~ 2.4 mm/s.",
    },
    "insertion.semi_major_axis": {
        "unit": "m", "limit": 250.0,
        "basis": "closed-form osculating elements of the frame-bridged raw "
                 "SECO state vs the chain's achieved parking orbit: "
                 "osculating-vs-J2-mean split at 278 km/51 deg is O(100 m) "
                 "and 12 s of coast drag at ~0.5 kg/m^3-class density decays "
                 "a by tens of metres; observed -67 m.",
    },
    "insertion.eccentricity": {
        "unit": "-", "limit": 5e-6,
        "basis": "same comparison; observed 1.2e-6 (mean-vs-osculating e "
                 "oscillation at 51 deg).",
    },
    "insertion.inclination": {
        "unit": "deg", "limit": 0.01,
        "basis": "inclination is J2-invariant and drag-insensitive; any "
                 "gap would be an extraction/frame error; observed 2.3e-4.",
    },
    "insertion.raan": {
        "unit": "deg", "limit": 0.01,
        "basis": "both sides computed in GCRS at the same epoch, so the "
                 "difference is frame-bridge + element-machinery only; "
                 "observed 7.7e-5.",
    },
    "ascent.seco_state_position": {
        "unit": "m", "limit": 0.1,
        "basis": "contract stage-2 track end vs the LAUNCH_ECI->GCRS bridge "
                 "of the raw backend final state: serialisation + frame "
                 "round-trip; observed 6e-11 m.",
    },
    "ascent.seco_state_velocity": {
        "unit": "m/s", "limit": 0.001,
        "basis": "same serialisation/bridge round-trip; observed 9e-13 m/s.",
    },
    "analytic.seco_orbit_energy": {
        "unit": "m", "limit": 1.0,
        "basis": "vis-viva semi-major axis recomputed from the raw backend "
                 "state vector must equal the backend's own reported orbit: "
                 "a pure internal-consistency identity of the external "
                 "recording; observed 5 mm.",
    },
    "analytic.seco_orbit_eccentricity": {
        "unit": "-", "limit": 1e-7,
        "basis": "same identity for eccentricity; observed 6.6e-10.",
    },
    "analytic.seco_orbit_period": {
        "unit": "s", "limit": 0.05,
        "basis": "closed-form period 2*pi*sqrt(a^3/mu) vs the backend's "
                 "reported period; observed 6 microseconds.",
    },
    "analytic.seco_orbit_inclination": {
        "unit": "deg", "limit": 0.1,
        "basis": "closed-form inclination from the state vector vs the "
                 "backend's reported inclination (its value carries a "
                 "sub-arcminute trajectory-averaging convention); observed "
                 "0.018 deg.",
    },
    "ascent.stage2_dv_ledger": {
        "unit": "m/s", "limit": 0.20,
        "basis": "Tsiolkovsky ideal dv = Isp*g0*ln(m_ignition/m_seco) is an "
                 "upper bound; the graded quantity is the relative loss "
                 "(ideal - achieved inertial gain)/ideal over the 248 s "
                 "stage-2 burn: gravity + drag losses at 24-70 deg "
                 "elevation. 0.20 is the design allowance of ADR-0008 F9; "
                 "observed 0.122.",
    },
    "recovery.coast_arc_position": {
        "unit": "m", "limit": 1200.0,
        "basis": "exact two-body propagation of the track state at "
                 "boostback MECO to the first track sample below the 100 km "
                 "entry interface: the residual is J2 secular drift over "
                 "207 s plus the onset of drag below the interface; ~2x the "
                 "recorded 575 m and consistent with O(100 m)-class J2 "
                 "along-track growth.",
    },
    "recovery.coast_arc_velocity": {
        "unit": "m/s", "limit": 6.0,
        "basis": "same arc, velocity; observed 3.8 m/s (J2 + interface "
                 "deceleration).",
    },
    "recovery.event.landing_burn_ignition": {
        "unit": "s", "limit": 2.0,
        "basis": "landing-burn ignition event vs the first powered track "
                 "sample after the entry-burn MECO: 2 s track grid, so the "
                 "edge is bracketed within one interval; observed 0.5 s.",
    },
    "recovery.event.touchdown": {
        "unit": "s", "limit": 0.05,
        "basis": "touchdown event time vs the impact-sampled final track "
                 "row (exact in the sealed record) and vs the impact event; "
                 "0.05 s = half the 0.1 s landing integrator step.",
    },
    "recovery.touchdown_gates": {
        "unit": "-", "limit": 1.0,
        "basis": "composite of the four touchdown gates against the flight "
                 "target's own limits (design input, graded as such), each "
                 "as a fraction of its gate: position error <= 25 m "
                 "(launch-site radius gate), |vertical speed| <= target "
                 "max_touchdown_speed_m_s, lateral <= "
                 "max_touchdown_lateral_speed_m_s, reserve >= "
                 "min_propellant_reserve_kg (graded inverted: limit/value). "
                 "limit 1.0 = all gates inside.",
    },
}

REQUIRED_REFERENCE_FILES = (
    "PROVENANCE.json",
    "mission-optimized/launch/ascent/events.json",
    "mission-optimized/launch/ascent/manifest.json",
    "mission-optimized/launch/ascent/opentsiolkovsky/summary.json",
    "mission-optimized/launch/ascent/opentsiolkovsky/trajectory.csv",
    "mission-optimized/launch/ascent/opentsiolkovsky/bodies/stage1.csv",
    "mission-optimized/launch/ascent/opentsiolkovsky/bodies/stage2_fairing.csv",
    "mission-optimized/launch/ascent/trajectories/stage2.csv",
    "mission-optimized/mission.verification.json",
    "mission-optimized/launch/target.yaml",
    "mission-optimized/recovery/run/verification.json",
    "mission/recovery/run/events.json",
    "mission/recovery/run/trajectories/booster.csv",
)


class ReferenceNotFound(FileNotFoundError):
    """The committed E5-05 reference tree is not present (skip-and-arm)."""


def _load_e505():
    spec = importlib.util.spec_from_file_location("e505_reference", E505_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {E505_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_reference_digests(reference_root: Path = REFERENCE_ROOT) -> dict:
    """Re-hash every file under *reference_root* against ``PROVENANCE.json``.

    Same tamper check the E5-05 gate applies (missing / extra / changed
    files), exposed here so the full-chain report always records it."""
    prov_path = Path(reference_root) / "PROVENANCE.json"
    if not prov_path.is_file():
        raise ReferenceNotFound(f"{prov_path}: run scripts/e505_reference.py update")
    recorded = json.loads(prov_path.read_text(encoding="utf-8"))["files"]
    present = {}
    for path in sorted(Path(reference_root).rglob("*")):
        if path.is_file() and path.name != "PROVENANCE.json":
            present[str(path.relative_to(reference_root))] = hashlib.sha256(
                path.read_bytes()).hexdigest()
    fails = [f"missing: {n}" for n in sorted(set(recorded) - set(present))]
    fails += [f"unexpected extra file: {n}" for n in sorted(set(present) - set(recorded))]
    fails += [f"content changed without the update script: {n}"
              for n in sorted(set(present) & set(recorded))
              if present[n] != recorded[n]]
    return {"ok": not fails, "files_checked": len(present), "failures": fails}


def _read_csv_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _col(rows: list[dict], name: str) -> list[float]:
    return [float(r[name]) for r in rows]


def _interp(t: float, ts, values) -> "object":
    import numpy as np

    return np.interp(t, np.asarray(ts, dtype=float), np.asarray(values, dtype=float))


def _state_at(t: float, rows: list[dict],
              pos=("x_m", "y_m", "z_m"), vel=("vx_m_s", "vy_m_s", "vz_m_s")):
    """Track state (GCRS 6-vector) linearly interpolated to epoch *t*."""
    import numpy as np

    ts = _col(rows, "t_s")
    return np.concatenate([
        [_interp(t, ts, _col(rows, c)) for c in pos],
        [_interp(t, ts, _col(rows, c)) for c in vel],
    ])


def _events(path: Path) -> list[dict]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return doc if isinstance(doc, list) else doc["events"]


def build_full_chain_report(reference_root: Path | None = None, *,
                            check_digests: bool = True) -> dict:
    """Grade the recorded E5-05 full chain against its independent references.

    Returns a ``verification-report-1.0`` document of kind
    ``full-chain-reference`` whose ``status`` carries the per-check
    categories and E4-03 evidence types, whose ``inputs.check_details``
    carries value / limit / reference / basis for every check, and whose
    ``inputs.reference`` records the provenance digests, toolchain pins and
    per-check evidence map. Raises :class:`ReferenceNotFound` when the
    committed reference tree is absent."""
    import numpy as np
    import yaml
    from orbitopt.flightlab.checks import orbital_elements
    from orbitopt.flightlab.frames import to_gcrs
    from orbitopt.verify.benchmarks import MU_EARTH_M3_S2, propagate_kepler_analytic
    from orbitopt.verify.report import (
        CHECK_MODEL, CHECK_NUMERICAL, EVIDENCE_ANALYTICAL, EVIDENCE_DESIGN_INPUT,
        EVIDENCE_INDEPENDENT_MODEL, EVIDENCE_NUMERICAL, provenance, report_digest,
        status_from_checks, to_jsonable)

    root = Path(reference_root) if reference_root is not None else REFERENCE_ROOT
    if not (root / "PROVENANCE.json").is_file():
        raise ReferenceNotFound(f"{root}: run scripts/e505_reference.py update")
    missing = [f for f in REQUIRED_REFERENCE_FILES if not (root / f).is_file()]
    if missing:
        raise ReferenceNotFound(f"reference files missing under {root}: {missing}")
    digests = verify_reference_digests(root) if check_digests else None
    if digests is not None and not digests["ok"]:
        raise RuntimeError(f"E5-05 reference tamper check failed: {digests['failures']}")

    e505 = _load_e505()
    prov = json.loads((root / "PROVENANCE.json").read_text(encoding="utf-8"))
    ascent_dir = root / "mission-optimized" / "launch" / "ascent"
    ot_dir = ascent_dir / "opentsiolkovsky"
    manifest = json.loads((ascent_dir / "manifest.json").read_text(encoding="utf-8"))
    epoch = datetime.fromisoformat(manifest["created_utc"].removesuffix("Z")).replace(
        tzinfo=timezone.utc)

    details: list[dict] = []

    def grade(check_id: str, category: str, evidence: str, value, reference: str):
        band = FULL_CHAIN_TOLERANCES[check_id]
        limit = band["limit"]
        passed = abs(float(value)) <= float(limit)
        details.append({
            "id": check_id, "category": category, "evidence": evidence,
            "unit": band["unit"], "limit": limit, "value": float(value),
            "passed": passed, "reference": reference, "basis": band["basis"],
        })
        return check_id, passed

    checks: dict[str, bool] = {}
    categories: dict[str, str] = {}
    evidence: dict[str, str] = {}

    def add(check_id: str, category: str, evidence_type: str, value, reference: str) -> None:
        name, ok = grade(check_id, category, evidence_type, value, reference)
        checks[name] = ok
        categories[name] = category
        evidence[name] = evidence_type

    chain_events = _events(ascent_dir / "events.json")
    ot = json.loads((ot_dir / "summary.json").read_text(encoding="utf-8"))
    ot_events = ot["events"]
    target = yaml.safe_load((root / "mission-optimized" / "launch" / "target.yaml").read_text(
        encoding="utf-8"))
    rec_dir = root / "mission" / "recovery" / "run"
    booster = _read_csv_rows(rec_dir / "trajectories" / "booster.csv")
    rec_events = _events(rec_dir / "events.json")
    mission_report = json.loads((root / "mission-optimized" / "mission.verification.json").read_text(
        encoding="utf-8"))

    def ev(kind: str, t_min: float | None = None, ot_name: str | None = None) -> dict:
        for e in chain_events:
            if e["kind"] != kind:
                continue
            if t_min is not None and e["t_s"] < t_min:
                continue
            if ot_name is not None and (e.get("data") or {}).get("ot_event") != ot_name:
                continue
            return e
        raise KeyError(f"chain event {kind} not found")

    # --- A. ascent events vs the raw OpenTsiolkovsky state history ---------
    raw = _read_csv_rows(ot_dir / "trajectory.csv")
    raw_t = _col(raw, "time_s")
    raw_thrust = _col(raw, "thrust_n")
    raw_mass = _col(raw, "mass_kg")
    raw_stage = [int(r["stage"]) for r in raw]

    meco_t = ev("meco")["t_s"]
    add("ascent.event.meco", CHECK_MODEL, EVIDENCE_NUMERICAL,
        next(t for t, th, st in zip(raw_t, raw_thrust, raw_stage)
             if st == 1 and th == 0.0 and t > 100.0) - meco_t,
        "raw trajectory.csv first stage-1 thrust-off sample after t > 100 s")

    sep_t = ev("separation", ot_name="separation")["t_s"]
    add("ascent.event.separation", CHECK_MODEL, EVIDENCE_NUMERICAL,
        next(t for t, m in zip(raw_t, raw_mass) if m < 200000.0) - sep_t,
        "raw trajectory.csv staging mass step (311 t -> 128 t)")

    jet_t = ev("separation", ot_name="jettison")["t_s"]
    fairing_raw = _read_csv_rows(ot_dir / "bodies" / "stage2_fairing.csv")
    add("ascent.event.fairing_jettison", CHECK_MODEL, EVIDENCE_NUMERICAL,
        float(fairing_raw[0]["time_s"]) - jet_t,
        "raw bodies/stage2_fairing.csv track-start sample")

    seco_t = ev("seco")["t_s"]
    last_burn2 = max(t for t, th, st in zip(raw_t, raw_thrust, raw_stage)
                     if st == 2 and th > 0.0)
    seco_raw = next(t for t, th, st in zip(raw_t, raw_thrust, raw_stage)
                    if st == 2 and th == 0.0 and t > last_burn2)
    add("ascent.event.seco", CHECK_MODEL, EVIDENCE_NUMERICAL,
        seco_raw - seco_t,
        "raw trajectory.csv stage-2 thrust-off transition after the final "
        "powered sample")

    # --- B. shared ballistic window: recovery leg vs raw OT debris ---------
    ot_event = {f"{e['kind']}#{e.get('stage')}": e for e in ot_events}
    ign_rec = next(e["t_s"] for e in rec_events if e["kind"] == "ignition" and e["t_s"] > sep_t)
    deb = _read_csv_rows(ot_dir / "bodies" / "stage1.csv")
    deb_t = _col(deb, "time_s")
    deb_r = np.column_stack([_col(deb, c) for c in
                             ("pos_eci_x_m", "pos_eci_y_m", "pos_eci_z_m")])
    deb_v = np.column_stack([_col(deb, c) for c in
                             ("vel_eci_x_mps", "vel_eci_y_mps", "vel_eci_z_mps")])
    booster_t = _col(booster, "t_s")
    window = [t for t in booster_t if sep_t + 0.5 < t < ign_rec]
    with e505.frozen_earth_orientation():
        rg, vg = to_gcrs(deb_r, deb_v, "LAUNCH_ECI", epoch, np.asarray(deb_t))
        dp = max(float(np.linalg.norm(
            np.interp(t, deb_t, rg[:, k])
            - np.interp(t, booster_t, _col(booster, ("x_m", "y_m", "z_m")[k]))))
            for t in window for k in range(3))
        dv = max(float(np.linalg.norm(
            np.interp(t, deb_t, vg[:, k])
            - np.interp(t, booster_t, _col(booster, ("vx_m_s", "vy_m_s", "vz_m_s")[k]))))
            for t in window for k in range(3))
        add("ascent.separation_window_position", CHECK_MODEL, EVIDENCE_INDEPENDENT_MODEL,
            dp, "raw OT stage-1 debris track vs booster track over the "
                "unpowered window between separation and boostback ignition")
        add("ascent.separation_window_velocity", CHECK_MODEL, EVIDENCE_INDEPENDENT_MODEL,
            dv, "same window, velocity")

        # --- C. terminal insertion state vs external recording + chain ------
        fs = ot["final_state"]
        seco_r = np.asarray(fs["position_eci_m"], dtype=float)
        seco_v = np.asarray(fs["velocity_eci_mps"], dtype=float)
        b_r, b_v = to_gcrs(seco_r[None, :], seco_v[None, :], "LAUNCH_ECI", epoch,
                           np.asarray([fs["time_s"]]))
        el = orbital_elements(b_r[0], b_v[0])
        ach = mission_report["targets"]["achieved_parking_orbit"]
        add("insertion.semi_major_axis", CHECK_MODEL, EVIDENCE_ANALYTICAL,
            el["semi_major_axis_m"] - ach["semi_major_axis_m"],
            "closed-form elements of the frame-bridged raw SECO state vs the "
            "chain's achieved parking orbit")
        add("insertion.eccentricity", CHECK_MODEL, EVIDENCE_ANALYTICAL,
            el["eccentricity"] - ach["eccentricity"], "same comparison")
        add("insertion.inclination", CHECK_MODEL, EVIDENCE_ANALYTICAL,
            el["inclination_deg"] - ach["inclination_deg"], "same comparison")
        add("insertion.raan", CHECK_MODEL, EVIDENCE_ANALYTICAL,
            el["raan_deg"] - ach["raan_deg"],
            "same comparison (both computed in GCRS at the mission epoch)")

        # --- D. serialisation round-trip of the SECO state -------------------
        track2 = _read_csv_rows(ascent_dir / "trajectories" / "stage2.csv")
        last = track2[-1]
        dp_ser = float(np.linalg.norm(
            b_r[0] - np.array([float(last[c]) for c in ("x_m", "y_m", "z_m")])))
        dv_ser = float(np.linalg.norm(
            b_v[0] - np.array([float(last[c]) for c in ("vx_m_s", "vy_m_s", "vz_m_s")])))
        add("ascent.seco_state_position", CHECK_NUMERICAL, EVIDENCE_NUMERICAL,
            dp_ser, "contract stage-2 track final sample vs frame-bridged raw "
                    "backend final state (same epoch)")
        add("ascent.seco_state_velocity", CHECK_NUMERICAL, EVIDENCE_NUMERICAL,
            dv_ser, "same comparison, velocity")

        # --- E. analytic identities inside the external recording ------------
        ot_orbit = fs["orbit"]
        add("analytic.seco_orbit_energy", CHECK_NUMERICAL, EVIDENCE_ANALYTICAL,
            el["semi_major_axis_m"] - ot_orbit["semi_major_axis_m"],
            "vis-viva a from the raw backend state vs the backend's own orbit")
        add("analytic.seco_orbit_eccentricity", CHECK_NUMERICAL, EVIDENCE_ANALYTICAL,
            el["eccentricity"] - ot_orbit["eccentricity"], "same identity")
        add("analytic.seco_orbit_period", CHECK_NUMERICAL, EVIDENCE_ANALYTICAL,
            el["period_s"] - ot_orbit["period_s"],
            "closed-form period from the state vs the backend's reported period")
        add("analytic.seco_orbit_inclination", CHECK_NUMERICAL, EVIDENCE_ANALYTICAL,
            el["inclination_deg"] - ot_orbit["inclination_deg"], "same identity")

    # --- F. Tsiolkovsky ideal-dv bound on the stage-2 burn -------------------
    vehicle = yaml.safe_load((REPO / "src" / "orbitopt" / "data" / "flightlab" /
                              "vehicles" / "synthetic_two_stage_rtls.yaml").read_text(
                                 encoding="utf-8"))
    isp2 = float(vehicle["stages"][1]["engine"]["isp_vac_s"])
    ig2 = ot_event["ignition#2"]
    bo2 = ot_event["burnout#2"]
    ideal = isp2 * G0 * math.log(float(ig2["mass_kg"]) / float(bo2["mass_kg"]))
    gained = float(bo2["speed_inertial_mps"]) - float(ig2["speed_inertial_mps"])
    loss_fraction = (ideal - gained) / ideal
    add("ascent.stage2_dv_ledger", CHECK_NUMERICAL, EVIDENCE_ANALYTICAL,
        loss_fraction if 0.0 <= loss_fraction else 1.0,
        f"Isp*g0*ln({ig2['mass_kg']:.0f}/{bo2['mass_kg']:.0f}) = {ideal:.1f} m/s "
        f"ideal vs {gained:.1f} m/s inertial gain "
        f"(OT events ignition#2/burnout#2)")

    # --- G. two-body propagation of the boostback coast arc ------------------
    meco_bb = min(e["t_s"] for e in rec_events if e["kind"] == "meco" and e["t_s"] > sep_t)
    s0 = _state_at(meco_bb, booster)
    iface_idx = next(i for i, r in enumerate(booster)
                     if float(r["t_s"]) > meco_bb + 60.0 and float(r["altitude_m"]) < 100000.0)
    s_face = np.array([float(booster[iface_idx][c]) for c in
                       ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s")])
    s1 = propagate_kepler_analytic(s0, float(booster[iface_idx]["t_s"]) - meco_bb)
    add("recovery.coast_arc_position", CHECK_NUMERICAL, EVIDENCE_ANALYTICAL,
        float(np.linalg.norm(s1[:3] - s_face[:3])),
        "exact two-body coast from boostback MECO to the first sample below "
        "the 100 km entry interface vs the flown track")
    add("recovery.coast_arc_velocity", CHECK_NUMERICAL, EVIDENCE_ANALYTICAL,
        float(np.linalg.norm(s1[3:] - s_face[3:])), "same arc, velocity")

    # --- H. landing events vs the sealed booster state history ---------------
    ign_lb = next(e for e in rec_events if e["kind"] == "landing_start")
    first_powered = min(float(r["t_s"]) for r in booster
                        if float(r["throttle"]) > 0.0 and float(r["t_s"]) >= ign_lb["t_s"])
    add("recovery.event.landing_burn_ignition", CHECK_MODEL, EVIDENCE_NUMERICAL,
        first_powered - ign_lb["t_s"],
        "landing_start event vs first powered booster sample after the "
        "entry-burn MECO")
    touchdown = next(e for e in rec_events if e["kind"] == "touchdown")
    impact = next(e for e in rec_events if e["kind"] == "impact" and e["t_s"] > 500.0)
    add("recovery.event.touchdown", CHECK_MODEL, EVIDENCE_NUMERICAL,
        max(abs(float(booster[-1]["t_s"]) - touchdown["t_s"]),
            abs(impact["t_s"] - touchdown["t_s"])),
        "touchdown vs impact event vs the impact-sampled final booster row")

    # --- I. touchdown gates (design input, graded honestly) ------------------
    rec_ver = json.loads((root / "mission-optimized" / "recovery" / "run" /
                          "verification.json").read_text(encoding="utf-8"))
    gate_vals = {c["id"]: c for c in rec_ver["checks"]}
    recov = target["recovery"]
    gates = {
        "position_error_m": (gate_vals["landing-position"]["value"],
                            gate_vals["landing-position"]["limit"]),
        "vertical_speed_m_s": (abs(gate_vals["landing-vertical-speed"]["value"]),
                               recov["max_touchdown_speed_m_s"]),
        "lateral_speed_m_s": (gate_vals["landing-lateral-speed"]["value"],
                              recov["max_touchdown_lateral_speed_m_s"]),
        "propellant_reserve_kg": (gate_vals["recovery-propellant-reserve"]["value"],
                                  recov["min_propellant_reserve_kg"]),
    }
    reserve_val, reserve_lim = gates.pop("propellant_reserve_kg")
    utilisation = max(max(val / lim for val, lim in gates.values()),
                      reserve_lim / reserve_val if reserve_val > 0.0 else math.inf)
    add("recovery.touchdown_gates", CHECK_MODEL, EVIDENCE_DESIGN_INPUT,
        utilisation,
        "touchdown gates vs the flight target's own limits "
        "(landing-position 25 m; target recovery block; reserve = final mass "
        "minus stage-1 dry mass): "
        + ", ".join(f"{k} {v:.4g} / {l:g}" for k, (v, l) in gates.items()))

    status = status_from_checks(checks, categories, evidence)
    report = {
        "schemaVersion": "1.0",
        "kind": FULL_CHAIN_REPORT_KIND,
        "label": ("E5-05 reference mission full chain (MECO, separation, fairing "
                  "jettison, SECO/insertion, RTLS coast/entry, landing) against the "
                  "raw OpenTsiolkovsky recording, closed-form mechanics and the "
                  "flight target"),
        "status": status,
        "inputs": {
            "reference": {
                "root": str(root.relative_to(REPO)) if root.is_relative_to(REPO) else str(root),
                "description": prov["reference"],
                "provenance_files_checked": (digests or {}).get("files_checked"),
                "provenance_ok": None if digests is None else digests["ok"],
                "requires_opentsiolkovsky": prov.get("requires_opentsiolkovsky"),
                "requires_gfold": prov.get("requires_gfold"),
                "toolchain": prov.get("toolchain"),
                "git_sha_recording": prov.get("git_sha_recording"),
                "epoch_utc": manifest["created_utc"],
            },
            "tolerances": {k: dict(v) for k, v in FULL_CHAIN_TOLERANCES.items()},
            "check_details": details,
            "evidence_map": dict(sorted(evidence.items())),
            "ascent_event_source": "chain events (run dir) graded against raw "
                                   "OpenTsiolkovsky state-history extractions",
            "honesty_notes": [
                "The ascent was computed by OpenTsiolkovsky; its event checks "
                "validate the chain's extraction/framing of the external "
                "recording, while the recovery leg compares two independently "
                "formulated models (gfold vs OpenTsiolkovsky) over the shared "
                "ballistic window.",
                "Landing has no external or telemetry reference: touchdown is "
                "graded by record consistency (numerical_consistency), the "
                "coast arc by a closed form (analytical) and the gates by the "
                "target itself (design_input). No check here claims "
                "historical_telemetry evidence.",
            ],
        },
        "propagations": [],
        "residuals": [],
        "targets": {
            "achieved_parking_orbit": dict(ach),
            "reference_seco_orbit": dict(ot_orbit),
            "converged": bool(mission_report["targets"].get("converged", True)),
        },
        "provenance": provenance(),
    }
    report = to_jsonable(report)
    report["digest"] = report_digest(report)
    return report


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m orbitopt.verify.full_chain",
        description="Grade the recorded E5-05 reference mission full chain "
                    "against its independent references (E8-01).")
    ap.add_argument("--reference", type=Path, default=REFERENCE_ROOT,
                    help="E5-05 reference root (default: the committed tree)")
    ap.add_argument("--out", type=Path, help="write the report JSON here")
    ap.add_argument("--json", action="store_true", help="print the full report JSON")
    args = ap.parse_args(argv)
    try:
        report = build_full_chain_report(args.reference)
    except ReferenceNotFound as exc:
        print(f"reference not recorded: {exc}", file=sys.stderr)
        return 2
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n",
                            encoding="utf-8")
        print(f"wrote {args.out}")
    if args.json:
        print(json.dumps(report, indent=1, sort_keys=True))
    status = report["status"]
    failed = [n for n, ok in status["checks"].items() if not ok]
    for d in report["inputs"]["check_details"]:
        mark = "ok  " if d["passed"] else "FAIL"
        print(f"{mark} {d['id']:<44} {d['value']:+.6g} [{d['evidence']}]")
    print(f"full-chain reference: {status['result']}"
          f" ({sum(status['checks'].values())}/{len(status['checks'])} checks;"
          f" independent validation: {status['independent_validation']['result']})")
    for line in failed:
        print(f"  failed: {line}", file=sys.stderr)
    return 0 if status["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
