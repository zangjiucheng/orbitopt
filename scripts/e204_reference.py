#!/usr/bin/env python3
"""E2-04 frozen dual-branch reference mission: run, compare, verify, update.

The committed reference under ``tests/data/mission/reference/e2-04-frozen-mission/``
is ONE complete dual-branch mission, frozen end to end (task #257):

* ``ascent/``   -- the frozen FLIGHT INPUT: a two-stage ascent run directory
  (liftoff -> MECO -> separation -> SECO) recorded in ECEF, whose separation
  hands the booster branch 35 t of propellant for the ride home. Inputs are
  pinned as the literal constants in this file (pad, epoch, separation state,
  insertion state, masses); the run directory committed here is the byte
  output of ``build``, digests over the real target/vehicle documents.
* ``mission/``  -- the frozen RESULT: ``run_two_stage_leo`` over the ascent
  with the recovery stage declared -- the upper branch parked at 400 km and
  the booster branch flown back and landed (FL-302 plan, FL-301 propagation,
  FL-303/FL-305 landing burn on the pinned gfold) -- with its unified
  ``mission.verification.json`` and merged ``recovery/run``.
* ``tolerances.json`` / ``PROVENANCE.json`` -- the declared numeric band a
  regression run must match, and the file digests + toolchain fingerprint
  that froze it (the backend-version record the acceptance criterion asks
  for: gfold pin, numpy/scipy/astropy, IERS pin, plus the digests of every
  frozen input).

The mission itself is declared in the packaged scenario
``src/orbitopt/data/mission_two_stage_leo_recovery.yaml`` -- frozen inputs,
pinned recovery plan recipe, and ``propagation_result: not_checked`` for the
tudatpy leg the gfold-pinned landing stack cannot run (M-07 owns that leg).

Subcommands
    run       --out DIR   replay the frozen mission into DIR (needs the
                          landing stack: gfold 0.3.2, per envs/flightlab-landing-ci.txt)
    compare DIR           compare a fresh run against the reference
    verify                provenance check: committed files unmodified
    update                regenerate ascent fixture + reference (maintainers)
    tag-check --base SHA  enforce the [e204-reference-update] commit tag

G5 discipline (as in FL-601): the reference is updated ONLY by ``update``,
and the commit carrying the change must say so with the
``[e204-reference-update]`` tag -- CI runs ``verify`` (catches hand-edited
reference files) and ``tag-check`` (catches untagged updates).

    python scripts/e204_reference.py run --out /tmp/e204
    python scripts/e204_reference.py compare /tmp/e204
    python scripts/e204_reference.py verify
    python scripts/e204_reference.py update   # maintainers, with the tag
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REF_ROOT = REPO / "tests" / "data" / "mission" / "reference" / "e2-04-frozen-mission"
ASCENT_DIR = REF_ROOT / "ascent"
MISSION_DIR = REF_ROOT / "mission"
TOLERANCES = REF_ROOT / "tolerances.json"
PROVENANCE = REF_ROOT / "PROVENANCE.json"
SCENARIO_YAML = REPO / "src" / "orbitopt" / "data" / "mission_two_stage_leo_recovery.yaml"
TARGET = REPO / "src" / "orbitopt" / "data" / "flightlab" / "scenarios" / "leo_two_stage.yaml"
VEHICLE = REPO / "src" / "orbitopt" / "data" / "flightlab" / "vehicles" / "synthetic_two_stage.yaml"
COMMIT_TAG = "[e204-reference-update]"
GOLDEN_GFOLD = "0.3.2"

# ----------------------------------------------------------- frozen inputs

#: The frozen ascent, as literal numbers (regenerated only by ``build`` /
#: ``update``). Every state is ECEF -- no frame bridge stands between the
#: frozen separation state and the recovery numerics.
ASCENT = {
    "run_id": "e2-04-frozen-ascent",
    "epoch_utc": "2026-11-10T12:00:00Z",   # the target's epoch; past -> finalized IERS
    "pad": (28.5619, -80.5772, 3.0),       # the scenario target's launch_site
    "t_liftoff_s": 0.0,
    "t_max_q_s": 57.339,
    "t_meco_s": 193.770,
    "t_separation_s": 207.0,
    "t_ignition2_s": 215.0,
    "t_seco_s": 622.0,
    # separation: 8 km downrange, 40 km up, 250 m/s on a +2 deg flight-path
    # angle heading east -- the FL-302 textbook RTLS separation, and 35 t of
    # propellant still in the stage-1 tanks (dry 30 t, wet 65 t).
    "sep_downrange_east_m": 8000.0,
    "sep_altitude_m": 40000.0,
    "sep_speed_m_s": 250.0,
    "sep_fpa_deg": 2.0,
    "sep_heading_deg": 90.0,
    "booster_wet_sep_kg": 65000.0,
    "booster_dry_kg": 30000.0,
    "stage2_wet_sep_kg": 55000.0,
    "stack_liftoff_kg": 435000.0,
    # SECO insertion: 400 km circular, 51.6 deg (defined in GCRS, stored ECEF)
    "insertion_altitude_m": 400000.0,
    "insertion_inclination_deg": 51.6,
    "stage2_seco_kg": 13282.392560150383,
    # ascent loads recorded on the flight (Gate-B checks of the flown arc)
    "max_q_pa": 36690.0,
    "max_accel_g": 4.1,
}

#: Default declared tolerances (written by ``update`` on first use, kept
#: verbatim afterwards -- changing them is itself a reference update).
#: A numeric comparison passes when |candidate - reference| <= abs + rel*|ref|.
#: The reference was recorded on Python 3.13 / numpy 2.5 / scipy 1.18 while
#: the gate runs on the landing CI stack (py 3.12 / numpy 1.26 / scipy 1.13):
#: the recovery chain is deterministic given its pinned inputs, so the band
#: only has to absorb cross-build floating-point drift -- 1 cm / 0.1 mm/s
#: leaves the pinned-solver results far more room than gfold's own solver
#: tolerances need while still failing any physics change that matters.
DEFAULT_TOLERANCES = {
    "t_s_abs": 1e-6,
    "mass_kg_abs": 1e-6,
    "position_m_abs": 1e-2,
    "velocity_m_s_abs": 1e-4,
    "extra_abs": {
        "altitude_m": 1e-2,
        "latitude_deg": 1e-8,
        "longitude_deg": 1e-8,
        "downrange_m": 1e-2,
        "dynamic_pressure_pa": 1e-1,
        "mach": 1e-6,
        "accel_g": 1e-6,
        "q_bar_pa": 1e-1,
        "density_kg_m3": 1e-9,
    },
    "extra_abs_default": 1e-3,
    "check_value_abs": 1e-4,
    "check_value_rel": 1e-9,
    "event_data_abs": 1e-6,
    "event_data_rel": 1e-9,
    "report_value_abs": 1e-4,
    "report_value_rel": 1e-9,
}


def _load_repo_package() -> None:
    sys.path.insert(0, str(REPO / "src"))


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pin_earth_orientation() -> str:
    """Freeze the frame bridge's Earth orientation to the IERS data the env
    ships (FL-601's pin, same reasoning: no daily IERS-A download)."""
    from astropy.utils import iers

    iers.conf.auto_download = False
    iers.conf.iers_degraded_accuracy = "ignore"
    try:
        import astropy_iers_data

        return str(astropy_iers_data.__version__)
    except ImportError:
        return "astropy bundled"


def _toolchain() -> dict:
    """The backend-version fingerprint of the stack that recorded/gated the
    reference: every package whose numbers land in the reference files."""
    from importlib.metadata import version as pkg_version

    def _v(name: str) -> str:
        try:
            return pkg_version(name)
        except Exception:
            return "absent"

    import platform
    import sys as _sys

    return {
        "python": _sys.version.split()[0],
        "platform": platform.platform(),
        "orbitopt": _v("orbitopt"),
        "gfold": _v("gfold"),
        "cvxpy": _v("cvxpy"),
        "numpy": _v("numpy"),
        "scipy": _v("scipy"),
        "astropy": _v("astropy"),
        "astropy_iers_data": _pin_earth_orientation(),
        "opentsiolkovsky": "not used: the ascent input is a frozen run directory",
        "tudatpy": "not used: propagation leg declared not_checked (see scenario header)",
    }


# ---------------------------------------------------------------- build

def _build_ascent_run():
    """The frozen ascent FlightRun, built from the ASCENT literals (module
    docstring): a contract-valid directory that ``run_two_stage_leo`` grades
    green and that carries a RECOVERABLE separation state (unlike the Gate B
    reference ascent, which burns every kilo it carries)."""
    import numpy as np

    _load_repo_package()
    from orbitopt.flightlab import frames
    from orbitopt.flightlab.contract import (
        BodyTrack, Check, Event, FlightRun, Verification,
    )
    from orbitopt.flightlab.recovery.boostback import _enu_basis

    a = ASCENT
    lat, lon, alt = a["pad"]
    basis = _enu_basis(lat, lon)
    E, N, U = (np.array(c, dtype=float) for c in basis)   # noqa: N806 (ENU triad)
    hdg = math.radians(a["sep_heading_deg"])
    horiz = math.sin(hdg) * E + math.cos(hdg) * N          # heading 90 deg = due east
    pad = frames.geodetic_to_ecef(lat, lon, alt)
    fpa = math.radians(a["sep_fpa_deg"])
    speed = a["sep_speed_m_s"]
    sep = np.array(pad, dtype=float) + a["sep_downrange_east_m"] * E \
        + a["sep_altitude_m"] * U
    sep_v = speed * (math.cos(fpa) * horiz + math.sin(fpa) * U)
    sep_r = tuple(float(c) for c in sep)
    sep_v = tuple(float(c) for c in sep_v)

    # SECO insertion: 400 km circular at 51.6 deg, defined in GCRS (the
    # contract's frame for insertion states -- checks.insertion_state reads
    # the raw track numbers). The stage-2 track therefore flies GCRS, the
    # booster track ECEF; the fork/continuity gates are frame-aware.
    from orbitopt.flightlab import checks as gate_checks

    r_orb = gate_checks.EARTH_RADIUS_M + a["insertion_altitude_m"]
    v_orb = math.sqrt(gate_checks.EARTH_MU / r_orb)
    inc = math.radians(a["insertion_inclination_deg"])
    epoch = datetime.fromisoformat(a["epoch_utc"].replace("Z", "+00:00")).replace(tzinfo=None)
    seco_r = tuple(float(c) for c in np.array([r_orb, 0.0, 0.0]))
    seco_v = tuple(float(c) for c in np.array(
        [0.0, v_orb * math.cos(inc), v_orb * math.sin(inc)]))
    sep_g, sep_gv = frames.to_gcrs(np.array([sep_r]), np.array([sep_v]),
                                   "ECEF", epoch, np.array([float(a["t_separation_s"])]))
    sep_r_g = tuple(float(c) for c in sep_g[0])
    sep_v_g = tuple(float(c) for c in sep_gv[0])

    zero_v = (0.0, 0.0, 0.0)
    epoch_s = a["epoch_utc"]
    run = FlightRun(
        run_id=a["run_id"],
        target_sha256=_digest(TARGET), vehicle_sha256=_digest(VEHICLE),
        git_sha=subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                               capture_output=True, text=True).stdout.strip() or "unknown",
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        engines={"orbitopt-e204-fixture": "1.0"},
        assumptions=(
            "frozen E2-04 ascent input (scripts/e204_reference.py build): pinned "
            "separation + insertion states (provenance: illustrative), not a re-fly",
            "stage-1/upper-stage tracks are two-sample handoff arcs (liftoff->"
            "separation, separation->SECO); the mission gates every boundary "
            "against the recorded states themselves",
            "booster/stage-1 tracks ECEF: the separation state the recovery "
            "chain consumes is exactly the number recorded here; stage-2 track "
            "GCRS: the contract's frame for the insertion state",
        ),
        tracks=(
            BodyTrack(body_id="stage1", parent_body_id=None, frame="ECEF",
                      epoch_utc=epoch_s, provenance="illustrative",
                      t_s=(a["t_liftoff_s"], a["t_separation_s"]),
                      mass_kg=(a["stack_liftoff_kg"],
                               a["booster_wet_sep_kg"] + a["stage2_wet_sep_kg"]),
                      position_m=(tuple(pad), sep_r),
                      velocity_m_s=(zero_v, sep_v)),
            BodyTrack(body_id="stage1_debris", parent_body_id="stage1", frame="ECEF",
                      epoch_utc=epoch_s, provenance="illustrative",
                      t_s=(a["t_separation_s"], a["t_separation_s"] + 1.0),
                      mass_kg=(a["booster_wet_sep_kg"], a["booster_wet_sep_kg"]),
                      position_m=(sep_r, sep_r),
                      velocity_m_s=(sep_v, sep_v)),
            BodyTrack(body_id="stage2", parent_body_id="stage1", frame="GCRS",
                      epoch_utc=epoch_s, provenance="illustrative",
                      t_s=(a["t_separation_s"], a["t_seco_s"]),
                      mass_kg=(a["stage2_wet_sep_kg"], a["stage2_seco_kg"]),
                      position_m=(sep_r_g, seco_r),
                      velocity_m_s=(sep_v_g, seco_v)),
        ),
        events=(
            Event(t_s=a["t_liftoff_s"], kind="liftoff", body_id="stage1", data={}),
            Event(t_s=a["t_max_q_s"], kind="max_q", body_id="stage1", data={}),
            Event(t_s=a["t_meco_s"], kind="meco", body_id="stage1", data={}),
            Event(t_s=a["t_separation_s"], kind="separation", body_id="stage2",
                  data={"separated_from": "stage1"}),
            Event(t_s=a["t_ignition2_s"], kind="ignition", body_id="stage2", data={}),
            Event(t_s=a["t_seco_s"], kind="seco", body_id="stage2", data={}),
        ),
        verification=Verification(
            optimizer_converged=True,
            physics_verified=None,
            mission_constraints_passed=True,
            checks=(
                Check(id="max-q-within-limit", passed=True, value=a["max_q_pa"],
                      limit=60000.0, unit="Pa"),
                Check(id="max-accel-within-limit", passed=True, value=a["max_accel_g"],
                      limit=5.0, unit="g"),
            ),
        ),
        backend=None,
    )
    return run


def cmd_build(out_dir: Path) -> int:
    """Materialise the frozen ascent run directory (ASCENT -> FlightResult)."""
    _load_repo_package()
    from orbitopt.flightlab.contract import write_run

    _pin_earth_orientation()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_run(out_dir, _build_ascent_run())
    print(f"frozen ascent fixture built from pinned constants -> {out_dir}")
    return 0


# ---------------------------------------------------------------- scenario

def load_scenario(ascent_dir: Path | None = None):
    """The packaged E2-04 scenario with its relative paths resolved against
    the repo root (``ascent_dir`` overrides the frozen ascent input -- used
    only by ``update`` while regenerating into a temporary directory)."""
    _load_repo_package()
    from orbitopt.mission.two_stage_leo import load_scenario as _load

    scenario = _load(SCENARIO_YAML)
    doc = scenario.to_dict()
    doc["ascent"]["run_dir"] = str(ascent_dir or ASCENT_DIR)
    doc["ascent"]["target_file"] = str(TARGET)
    doc["recovery"]["target_file"] = str(TARGET)
    from orbitopt.mission.two_stage_leo import TwoStageLeoScenario

    return TwoStageLeoScenario.from_dict(doc)


def _run_mission(out_dir: Path):
    from orbitopt.mission.two_stage_leo import run_two_stage_leo

    out_dir.mkdir(parents=True, exist_ok=True)
    mission = run_two_stage_leo(load_scenario(), out_dir, propagate=False)
    return mission


def cmd_run(out_dir: Path) -> int:
    _load_repo_package()
    iers_data = _pin_earth_orientation()
    print(f"Earth orientation: bundled IERS data only (astropy-iers-data {iers_data}), no download")
    mission = _run_mission(out_dir)
    print(f"mission status: {mission.status}")
    print(f"report: {mission.report_path}")
    if mission.status != "OK":
        failed = [c.id for c in mission.checks if not c.passed]
        print(f"failed checks: {failed}", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------- compare


def _num_diff(label: str, got, ref, atol: float, rtol: float, fails: list[str]) -> None:
    if got is None or ref is None:
        if got != ref:
            fails.append(f"{label}: {got!r} != reference {ref!r}")
        return
    if abs(got - ref) > atol + rtol * abs(ref):
        fails.append(f"{label}: {got!r} differs from reference {ref!r} by "
                     f"{abs(got - ref):.3e} > {atol:.1e} + {rtol:.1e}*|ref|")


def _compare_dicts(label: str, got: dict, ref: dict, atol: float, rtol: float,
                   fails: list[str]) -> None:
    if set(got) != set(ref):
        fails.append(f"{label}: keys {sorted(set(got) ^ set(ref))} differ")
    for k in sorted(set(got) & set(ref)):
        g, r = got[k], ref[k]
        if isinstance(g, (int, float)) and isinstance(r, (int, float)):
            _num_diff(f"{label}[{k}]", float(g), float(r), atol, rtol, fails)
        elif isinstance(g, dict) and isinstance(r, dict):
            _compare_dicts(f"{label}[{k}]", g, r, atol, rtol, fails)
        elif g != r:
            fails.append(f"{label}[{k}]: {g!r} != reference {r!r}")


def compare_runs(candidate_dir: Path, tol: dict) -> list[str]:
    """Field-by-field comparison of the merged recovery run against the
    reference run (FL-601's discipline; manifest fields that legitimately
    differ between machines/runs -- created_utc, git_sha -- are excluded)."""
    _load_repo_package()
    from orbitopt.flightlab.contract import read_run

    cand = read_run(candidate_dir)
    ref = read_run(MISSION_DIR / "recovery" / "run")
    fails: list[str] = []

    if cand.run_id != ref.run_id:
        fails.append(f"run_id: {cand.run_id!r} != reference {ref.run_id!r}")
    for field in ("target_sha256", "vehicle_sha256"):
        if getattr(cand, field) != getattr(ref, field):
            fails.append(f"{field}: differs from the reference run's input digest")
    if dict(cand.engines) != dict(ref.engines):
        fails.append(f"engines: {dict(cand.engines)} != reference {dict(ref.engines)} "
                     "-- the gate must run on the pinned solver stack")

    cand_tracks = {t.body_id: t for t in cand.tracks}
    ref_tracks = {t.body_id: t for t in ref.tracks}
    if set(cand_tracks) != set(ref_tracks):
        fails.append(f"bodies: {sorted(set(cand_tracks) ^ set(ref_tracks))} differ")
    for body in sorted(set(cand_tracks) & set(ref_tracks)):
        ct, rt = cand_tracks[body], ref_tracks[body]
        if ct.frame != rt.frame or ct.provenance != rt.provenance:
            fails.append(f"{body}: frame/provenance differ")
        if len(ct.t_s) != len(rt.t_s):
            fails.append(f"{body}: {len(ct.t_s)} samples != reference {len(rt.t_s)}")
            continue
        for i, (t, r) in enumerate(zip(ct.t_s, rt.t_s)):
            _num_diff(f"{body}[{i}].t_s", t, r, tol["t_s_abs"], 0.0, fails)
            _num_diff(f"{body}[{i}].mass_kg", ct.mass_kg[i], rt.mass_kg[i],
                      tol["mass_kg_abs"], 0.0, fails)
            for axis, a_, b_ in zip("xyz", ct.position_m[i], rt.position_m[i]):
                _num_diff(f"{body}[{i}].{axis}_m", a_, b_, tol["position_m_abs"], 0.0, fails)
            for axis, a_, b_ in zip(("vx", "vy", "vz"), ct.velocity_m_s[i], rt.velocity_m_s[i]):
                _num_diff(f"{body}[{i}].{axis}_m_s", a_, b_, tol["velocity_m_s_abs"], 0.0, fails)
            if set(ct.extra_columns) != set(rt.extra_columns):
                fails.append(f"{body}[{i}]: extra columns differ")
                continue
            for name in rt.extra_columns:
                _num_diff(f"{body}[{i}].{name}", ct.extra_columns[name][i],
                          rt.extra_columns[name][i],
                          tol["extra_abs"].get(name, tol["extra_abs_default"]), 0.0, fails)

    if len(cand.events) != len(ref.events):
        fails.append(f"events: {len(cand.events)} != reference {len(ref.events)}")
    for i, (ce, re_) in enumerate(zip(cand.events, ref.events)):
        _num_diff(f"events[{i}].t_s", ce.t_s, re_.t_s, tol["t_s_abs"], 0.0, fails)
        if ce.kind != re_.kind or ce.body_id != re_.body_id:
            fails.append(f"events[{i}]: {ce.kind}@{ce.body_id} != reference {re_.kind}@{re_.body_id}")
        _compare_dicts(f"events[{i}].data", dict(ce.data), dict(re_.data),
                       tol["event_data_abs"], tol["event_data_rel"], fails)

    for verdict in ("optimizer_converged", "physics_verified", "mission_constraints_passed"):
        cv, rv = getattr(cand.verification, verdict), getattr(ref.verification, verdict)
        if cv != rv:
            fails.append(f"verification.{verdict}: {cv!r} != reference {rv!r}")
    cand_checks = {c.id: c for c in cand.verification.checks}
    ref_checks = {c.id: c for c in ref.verification.checks}
    if set(cand_checks) != set(ref_checks):
        fails.append(f"checks: ids {sorted(set(cand_checks) ^ set(ref_checks))} differ")
    for cid in sorted(set(cand_checks) & set(ref_checks)):
        cc, rc = cand_checks[cid], ref_checks[cid]
        if cc.passed != rc.passed:
            fails.append(f"check {cid}: passed={cc.passed} != reference {rc.passed}")
        if cc.unit != rc.unit or cc.reason_code != rc.reason_code:
            fails.append(f"check {cid}: unit/reason differ from reference")
        _num_diff(f"check {cid}.value", cc.value, rc.value,
                  tol["check_value_abs"], tol["check_value_rel"], fails)
        _num_diff(f"check {cid}.limit", cc.limit, rc.limit,
                  tol["check_value_abs"], tol["check_value_rel"], fails)
    return fails


def _check_report_field(label: str, got, ref, tol, fails: list[str]) -> None:
    if isinstance(got, (int, float)) and isinstance(ref, (int, float)):
        _num_diff(label, float(got), float(ref), tol["report_value_abs"],
                  tol["report_value_rel"], fails)
    elif got != ref:
        fails.append(f"{label}: {got!r} != reference {ref!r}")


def compare_report(candidate_report: dict, tol: dict) -> list[str]:
    """The mission-report half of the comparison: the verdict conjunction
    (status.result), the full check-bool surface (status.checks), the phase
    verdicts (inputs.full_mission.phases) and the achieved-orbit residuals
    (targets) of a fresh run against the committed reference report. Path
    fields (inputs.ascent.run_dir) are not compared -- the candidate was run
    into its own directory; the run directory CONTENTS are compared by
    ``compare_runs``."""
    ref = json.loads((MISSION_DIR / "mission.verification.json").read_text(encoding="utf-8"))
    fails: list[str] = []

    _check_report_field("status.result", candidate_report["status"]["result"],
                        ref["status"]["result"], tol, fails)
    got_checks = candidate_report["status"].get("checks", {})
    ref_checks = ref["status"].get("checks", {})
    if set(got_checks) != set(ref_checks):
        fails.append(f"status.checks: ids {sorted(set(got_checks) ^ set(ref_checks))} differ")
    for cid in sorted(set(got_checks) & set(ref_checks)):
        if got_checks[cid] != ref_checks[cid]:
            fails.append(f"status.checks.{cid}: {got_checks[cid]!r} != reference {ref_checks[cid]!r}")

    def _phases(rep: dict) -> dict:
        return {p["phase"]: p for p in rep.get("inputs", {}).get("full_mission", {}).get("phases", [])}

    got_ph, ref_ph = _phases(candidate_report), _phases(ref)
    if set(got_ph) != set(ref_ph):
        fails.append(f"phases: ids {sorted(set(got_ph) ^ set(ref_ph))} differ")
    for name in sorted(set(got_ph) & set(ref_ph)):
        gp, rp = got_ph[name], ref_ph[name]
        for key in ("verdict", "applicable", "reason", "reason_codes"):
            if gp.get(key) != rp.get(key):
                fails.append(f"phase {name}.{key}: {gp.get(key)!r} != reference {rp.get(key)!r}")
        _compare_dicts(f"phase {name}.checks", dict(gp.get("checks") or {}),
                       dict(rp.get("checks") or {}), 0.0, 0.0, fails)

    _compare_dicts("targets.achieved", candidate_report["targets"]["achieved_parking_orbit"],
                   ref["targets"]["achieved_parking_orbit"],
                   tol["report_value_abs"], tol["report_value_rel"], fails)
    _compare_dicts("targets.deviations", candidate_report["targets"]["deviations"],
                   ref["targets"]["deviations"],
                   tol["report_value_abs"], tol["report_value_rel"], fails)
    return fails


def cmd_compare(candidate_dir: Path) -> int:
    _load_repo_package()
    if not TOLERANCES.is_file():
        print(f"{TOLERANCES}: missing", file=sys.stderr)
        return 2
    tol = json.loads(TOLERANCES.read_text(encoding="utf-8"))
    from orbitopt.verify.report import read_report

    fails = compare_report(read_report(candidate_dir / "mission.verification.json"), tol)
    fails += compare_runs(candidate_dir / "recovery" / "run", tol)
    if fails:
        print(f"E2-04 REGRESSION: {len(fails)} difference(s) vs {REF_ROOT} "
              f"(tolerances: {TOLERANCES})", file=sys.stderr)
        for line in fails[:40]:
            print(f"  {line}", file=sys.stderr)
        if len(fails) > 40:
            print(f"  ... {len(fails) - 40} more", file=sys.stderr)
        print("If this is an INTENDED change (new gfold pin, intended physics "
              "shift), regenerate with:  python scripts/e204_reference.py update "
              f"-- and commit with {COMMIT_TAG} in the message.", file=sys.stderr)
        return 1
    print(f"E2-04 frozen-mission regression OK: {candidate_dir} matches the committed "
          "reference within the declared tolerances")
    return 0


# ---------------------------------------------------------------- verify / update


def _reference_files() -> list[Path]:
    return sorted(p for p in REF_ROOT.rglob("*")
                  if p.is_file() and p.name != PROVENANCE.name)


def cmd_verify() -> int:
    if not PROVENANCE.is_file():
        print(f"{PROVENANCE}: missing -- run scripts/e204_reference.py update", file=sys.stderr)
        return 1
    recorded = json.loads(PROVENANCE.read_text(encoding="utf-8"))["files"]
    present = {str(p.relative_to(REF_ROOT)): _digest(p) for p in _reference_files()}
    fails = []
    for name in sorted(set(recorded) - set(present)):
        fails.append(f"missing: {name}")
    for name in sorted(set(present) - set(recorded)):
        fails.append(f"unexpected extra file (added by hand?): {name}")
    for name in sorted(set(present) & set(recorded)):
        if present[name] != recorded[name]:
            fails.append(f"content changed without the update script: {name}")
    if fails:
        print("REFERENCE TAMPER CHECK FAILED -- committed reference files were "
              "changed without scripts/e204_reference.py update:", file=sys.stderr)
        for line in fails:
            print(f"  {line}", file=sys.stderr)
        return 1
    print(f"reference OK: {len(present)} files match {PROVENANCE.name} digests")
    return 0


def cmd_update() -> int:
    """Regenerate the frozen ascent AND the mission reference from the pinned
    inputs, and rewrite the provenance digests + toolchain fingerprint. The
    ONLY sanctioned way to change the reference; the commit carrying the
    change must additionally carry COMMIT_TAG (CI's tag-check enforces that,
    and CI's verify fails any change this script did not write)."""
    import shutil
    import tempfile

    _pin_earth_orientation()
    with tempfile.TemporaryDirectory(prefix="e204-ref-") as tmp:
        tmp_root = Path(tmp)
        if cmd_build(tmp_root / "ascent") != 0:
            return 1
        scenario_dir = tmp_root / "mission"
        # run against the freshly built fixture (not the committed one)
        _load_repo_package()
        from orbitopt.mission.two_stage_leo import run_two_stage_leo

        scenario_dir.mkdir(parents=True, exist_ok=True)
        mission = run_two_stage_leo(load_scenario(ascent_dir=tmp_root / "ascent"),
                                    scenario_dir, propagate=False)
        if mission.status != "OK":
            print(f"refusing to record a red reference mission (status {mission.status!r}); "
                  "fix the mission before updating the reference", file=sys.stderr)
            return 1
        if REF_ROOT.exists():
            shutil.rmtree(REF_ROOT)
        REF_ROOT.mkdir(parents=True)
        shutil.copytree(tmp_root / "ascent", ASCENT_DIR)
        shutil.copytree(scenario_dir, MISSION_DIR)
    if not TOLERANCES.is_file():
        TOLERANCES.write_text(json.dumps(DEFAULT_TOLERANCES, indent=2) + "\n",
                              encoding="utf-8")
        print(f"wrote default {TOLERANCES}")
    prov = {
        "generated_by": "scripts/e204_reference.py update",
        "reference": "E2-04 frozen dual-branch reference mission "
                     "(ascent + parking + booster RTLS, E2-03 full-mission gate)",
        "scenario": str(SCENARIO_YAML.relative_to(REPO)),
        "target": str(TARGET.relative_to(REPO)),
        "vehicle": str(VEHICLE.relative_to(REPO)),
        "frozen_inputs": {
            "ascent_constants": {k: list(v) if isinstance(v, tuple) else v
                                 for k, v in ASCENT.items()},
            "target_sha256": _digest(TARGET),
            "vehicle_sha256": _digest(VEHICLE),
            "recovery_plan_options": {"initial_guess": [18.4, 17.75, 1.0, 4.36, 0.0,
                                                        17456.0, 23.0], "max_fevals": 18},
        },
        "requires_gfold": GOLDEN_GFOLD,
        "earth_orientation": f"bundled IERS only, astropy-iers-data {_pin_earth_orientation()}",
        "toolchain": _toolchain(),
        "requires_commit_tag": COMMIT_TAG,
        "git_sha_recording": subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip(),
        "files": {str(p.relative_to(REF_ROOT)): _digest(p) for p in _reference_files()},
    }
    PROVENANCE.write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"reference updated: {len(prov['files'])} files under {REF_ROOT}")
    return 0


# ---------------------------------------------------------------- tag-check


def cmd_tag_check(base: str | None) -> int:
    """CI guard (G5): if the diff touches the reference directory, some commit
    message in the pushed range must carry the update tag."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(REPO), *args],
                              capture_output=True, text=True, check=True).stdout

    if base is None or set(base) == {"0"}:
        print("tag-check skipped: no usable diff base (workflow_dispatch or "
              "force-push); the provenance verify still runs")
        return 0
    try:
        changed = git("diff", "--name-only", f"{base}..HEAD").splitlines()
    except subprocess.CalledProcessError:
        print(f"tag-check skipped: base {base} not present in this clone")
        return 0
    touched = [p for p in changed if p.startswith("tests/data/mission/reference/")]
    if not touched:
        print("reference files untouched by this change")
        return 0
    messages = git("log", "--format=%B", f"{base}..HEAD")
    if COMMIT_TAG not in messages:
        print(f"{len(touched)} reference file(s) changed but no commit message "
              f"carries {COMMIT_TAG}:", file=sys.stderr)
        for p in touched[:10]:
            print(f"  {p}", file=sys.stderr)
        return 1
    print(f"reference change is tagged {COMMIT_TAG}")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run", help="replay the frozen mission into --out")
    run_p.add_argument("--out", required=True, type=Path)
    cmp_p = sub.add_parser("compare", help="compare a fresh run dir against the reference")
    cmp_p.add_argument("dir", type=Path)
    sub.add_parser("verify", help="check committed reference files against PROVENANCE.json")
    build_p = sub.add_parser("build", help="materialise the frozen ascent fixture only")
    build_p.add_argument("--out", required=True, type=Path)
    sub.add_parser("update", help=f"regenerate the reference (commit the result with {COMMIT_TAG})")
    tag_p = sub.add_parser("tag-check", help="CI: reference diffs need the commit tag")
    tag_p.add_argument("--base", default=None)
    args = parser.parse_args(argv)
    if args.cmd == "run":
        return cmd_run(args.out)
    if args.cmd == "compare":
        return cmd_compare(args.dir)
    if args.cmd == "verify":
        return cmd_verify()
    if args.cmd == "build":
        return cmd_build(args.out)
    if args.cmd == "update":
        return cmd_update()
    if args.cmd == "tag-check":
        return cmd_tag_check(args.base)
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
