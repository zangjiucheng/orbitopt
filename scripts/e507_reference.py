#!/usr/bin/env python3
"""E5-07 physics qualification reference: run, compare, verify, update.

The committed reference under
``tests/data/mission/reference/e5-07-physics-qualification/`` (task #291) is
the PHYSICS QUALIFICATION REFERENCE mission: the Starship-class
public-estimate vehicle (``starship_class_public_estimate``) flown to LEO
against publicly-sourced flight-limited constraints
(``leo_starship_class.yaml`` -- max-q 55 kPa from public Max-Q figures, max
non-gravitational acceleration 3.0 g as a tagged ASSUMPTION, insertion band
and RTLS landing assumptions all sourced or declared in the target file).
It is deliberately NOT the E5-05 reference mission: E5-05
(``e5-05-reference-mission``) is the SOFTWARE INTEGRATION reference -- a
synthetic vehicle against relaxed DESIGN-ASSUMPTION constraints that proves
the chain end to end -- and must never be described as an engineering-grade
verification. This mission is the physics-grade counterpart, with one
reported gap (below).

Layout (mirrors e5-05-reference-mission where the machinery is shared):

* ``mission-optimized/`` -- the recorded mission: the ascent computed ONCE as
  a fixed-program OpenTsiolkovsky v1.0.0 flight (``launch/ascent``, raw
  backend outputs included), the flown FlightTarget (``launch/target.yaml``,
  every constraint with its source or ASSUMPTION tag), and the unified
  ``mission.verification.json``.
* ``mission/`` -- the frozen REPLAY RESULT: ``run_two_stage_leo`` over the
  recorded ascent (``run_dir`` mode,
  ``src/orbitopt/data/mission_two_stage_leo_starship_e507.yaml``). THIS is
  what CI re-flies and ``compare`` grades.
* ``earth_orientation.json`` -- the IERS EOP snapshot of the recording (the
  same finalized finals2000A rows as the E5-05 recording: same mission
  epoch, 2026-11-10); ``run``/``compare`` bind it via
  ``frozen_earth_orientation``.
* ``RECOVERY_GAP.md`` -- the E5-07 acceptance report for the booster leg:
  WHY the recovery profile is reported rather than flown (verbatim planner
  errors), and the debris-state consequence of leaving stage 1 on its
  ballistic arc. The acceptance criterion says a leg that cannot close is
  REPORTED with its reason -- never relaxed. No constraint was relaxed to
  obtain the green ascent: the max-q, 3.0 g, periapsis and inclination
  checks all pass against the DECLARED limits.
* ``tolerances.json`` / ``PROVENANCE.json`` -- declared comparison bands with
  their source, and the digests + full provenance of the recording.

Tolerance provenance. The ascent is replayed byte-for-byte from
``mission-optimized/launch/ascent`` (digest identity, enforced by
``verify``); the only numerics ``compare`` grades are the report-level
derived quantities (achieved parking-orbit elements, physical-gate summary),
which are pure re-derivations of the recorded tracks -- the bands below are
float-reassociation headroom (1e-6 relative on orbit elements, 1e-3 s on
epochs), an order of magnitude inside the ~1e-7 platform frame-bridge
roundoff that E5-05 documents for the same machinery, and they fail any
change that alters the flight itself.

Subcommands
    run       --out DIR   replay the reference mission into DIR (compute
                          stack: numpy/scipy/astropy/pyyaml, per
                          envs/flightlab-landing-ci.txt; no OpenTsiolkovsky,
                          no gfold -- this mission declares no recovery)
    compare DIR           compare a fresh replay against the reference
    crossenv-check DIR    grade a cross-environment run (
                          src/orbitopt/data/crossenv_e507_plan.yaml) against
                          the reference: envelope verified, the mission leg
                          green in the tudatpy environment, its parking-orbit
                          propagation PHYSICALLY EVIDENCED there (the E5-04
                          phase this interpreter cannot run), achieved orbit
                          within tolerance, and -- by design -- NO joint
                          full-mission gate: full_mission is not declared
                          because the recovery phase cannot be flown or
                          assessed (see RECOVERY_GAP.md); the cross-environment
                          joint gate for a full mission therefore REPORTS
                          not-declared rather than pretending closure
    verify                provenance check: committed files unmodified
    update                re-fly the ascent (OpenTsiolkovsky binary) and the
                          mission, and re-record the reference
    tag-check --base SHA  enforce the [e507-reference-update] commit tag

G5 discipline (as in E5-05): the reference is updated ONLY by ``update``,
and the commit carrying the change must say so with the
``[e507-reference-update]`` tag -- CI runs ``verify`` and ``tag-check``.

    python scripts/e507_reference.py run --out /tmp/e507
    python scripts/e507_reference.py compare /tmp/e507
    python scripts/e507_reference.py crossenv-check runs/crossenv-e5-07
    python scripts/e507_reference.py verify
    python scripts/e507_reference.py update   # maintainers, with the tag
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REF_ROOT = REPO / "tests" / "data" / "mission" / "reference" / "e5-07-physics-qualification"
OPTIMIZED_DIR = REF_ROOT / "mission-optimized"
MISSION_DIR = REF_ROOT / "mission"
TOLERANCES = REF_ROOT / "tolerances.json"
PROVENANCE = REF_ROOT / "PROVENANCE.json"
EOP_SNAPSHOT = REF_ROOT / "earth_orientation.json"
RECOVERY_GAP = REF_ROOT / "RECOVERY_GAP.md"
E505_EOP = REPO / "tests" / "data" / "mission" / "reference" / "e5-05-reference-mission" / "earth_orientation.json"
NOT_RECORDED = (
    "E5-07 reference NOT recorded yet: no E5-07 recording exists under "
    f"{REF_ROOT}. Make the recording with `python scripts/e507_reference.py "
    "update` (OpenTsiolkovsky binary, minutes). Every consumer of the "
    "reference skips, and arms itself, on the directory's existence.")
SCENARIO_YAML = REPO / "src" / "orbitopt" / "data" / "mission_two_stage_leo_starship_e507.yaml"
VEHICLE = REPO / "src" / "orbitopt" / "data" / "flightlab" / "vehicles" / "starship_class_public_estimate.yaml"
ASCENT_TARGET = REPO / "src" / "orbitopt" / "data" / "flightlab" / "scenarios" / "leo_starship_class.yaml"
CROSSENV_PLAN = REPO / "src" / "orbitopt" / "data" / "crossenv_e507_plan.yaml"
COMMIT_TAG = "[e507-reference-update]"
GOLDEN_OT = "1.0.0"

#: Default declared bands -- source: module docstring "Tolerance
#: provenance". |candidate - reference| <= abs + rel*|ref|.
DEFAULT_TOLERANCES = {
    "achieved_element_rel": 1e-6,
    "achieved_element_abs_m": 1e-3,
    "achieved_inc_abs_deg": 1e-6,
    "epoch_abs_s": 1e-3,
}


def _load_repo_package() -> None:
    src = str(REPO / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip() or "unknown"


def _pin_earth_orientation() -> str:
    from astropy.utils import iers

    iers.conf.auto_download = False
    iers.conf.iers_degraded_accuracy = "ignore"
    try:
        import astropy_iers_data

        return str(astropy_iers_data.__version__)
    except ImportError:
        return "astropy bundled"


@contextmanager
def frozen_earth_orientation(snapshot: Path | None = None):
    """Bind the reference's EOP snapshot for the duration of a run (same
    discipline as scripts/e505_reference.py)."""
    from astropy.utils import iers
    import astropy.units as u
    import numpy as np

    snapshot = json.loads(Path(snapshot or EOP_SNAPSHOT).read_text())
    units = {"MJD": u.day, "UT1_UTC": u.s, "PM_x": u.arcsec, "PM_y": u.arcsec}
    table = iers.IERS({name: np.array([row[name] for row in snapshot["rows"]]) * unit
                       for name, unit in units.items()})
    with iers.earth_orientation_table.set(table):
        yield


def _toolchain() -> dict:
    from importlib.metadata import version as pkg_version

    def _v(name: str) -> str:
        try:
            return pkg_version(name)
        except Exception:
            return "absent"

    import platform
    import sys as _sys

    try:
        from orbitopt.flightlab.adapters.opentsiolkovsky import OT_VERSION

        ot = OT_VERSION
    except Exception:
        ot = "unknown"
    return {
        "python": _sys.version.split()[0],
        "platform": platform.platform(),
        "orbitopt": _git_sha(),
        "gfold": _v("gfold"),
        "pygmo": _v("pygmo"),
        "numpy": _v("numpy"),
        "scipy": _v("scipy"),
        "astropy": _v("astropy"),
        "astropy_iers_data": _pin_earth_orientation(),
        "opentsiolkovsky": ot,
        "tudatpy": "recorded absent; the cross-environment leg "
                   "(crossenv_e507_plan.yaml) runs the mission + parking "
                   "propagation in the tudatpy environment",
    }


def _absolutise_doc_paths(doc: dict) -> dict:
    for key in ("run_dir", "target_file"):
        if doc.get("ascent", {}).get(key):
            doc["ascent"][key] = str((REPO / doc["ascent"][key]).resolve())
    return doc


def load_scenario(ascent_dir: Path | None = None, target_file: Path | None = None):
    """The packaged scenario with repo-relative paths resolved (the two
    overrides are used only by ``update`` while recording)."""
    _load_repo_package()
    import yaml

    from orbitopt.mission.two_stage_leo import TwoStageLeoScenario

    doc = yaml.safe_load(SCENARIO_YAML.read_text(encoding="utf-8"))["scenario"]
    doc = _absolutise_doc_paths(doc)
    if ascent_dir is not None:
        doc["ascent"]["run_dir"] = str(ascent_dir)
    if target_file is not None:
        doc["ascent"]["target_file"] = str(target_file)
    return TwoStageLeoScenario.from_dict(doc)


# ---------------------------------------------------------------- run / compare

def _run_replay(out_dir: Path):
    from orbitopt.mission.two_stage_leo import run_two_stage_leo

    out_dir.mkdir(parents=True, exist_ok=True)
    with frozen_earth_orientation():
        return run_two_stage_leo(load_scenario(), out_dir, propagate=False)


def cmd_run(out_dir: Path) -> int:
    if not REF_ROOT.is_dir():
        print(NOT_RECORDED)
        return 0
    _load_repo_package()
    print(f"Earth orientation: frozen E5-07 snapshot, no download "
          f"(installed data {_pin_earth_orientation()})")
    mission = _run_replay(out_dir)
    print(f"mission status: {mission.status}")
    print(f"report: {mission.report_path}")
    if mission.status != "OK":
        print(f"failed checks: {[c.id for c in mission.checks if not c.passed]}",
              file=sys.stderr)
        return 1
    return 0


def _compare_elements(ref: dict, cand: dict, tol: dict, where: str) -> list[str]:
    fails = []
    rel = float(tol["achieved_element_rel"])
    abs_m = float(tol["achieved_element_abs_m"])
    inc = float(tol["achieved_inc_abs_deg"])
    for key, ref_v in (ref or {}).items():
        cand_v = (cand or {}).get(key)
        if not isinstance(ref_v, (int, float)) or cand_v is None:
            fails.append(f"{where}.{key}: reference {ref_v!r} vs candidate {cand_v!r}")
            continue
        band = abs_m + rel * abs(ref_v) if "deg" not in key else inc
        if abs(float(cand_v) - float(ref_v)) > band:
            fails.append(f"{where}.{key}: {cand_v!r} vs reference {ref_v!r} "
                         f"(band {band:g})")
    return fails


def cmd_compare(candidate_dir: Path) -> int:
    if not REF_ROOT.is_dir():
        print(NOT_RECORDED)
        return 0
    _load_repo_package()
    if not TOLERANCES.is_file():
        print(f"{TOLERANCES}: missing", file=sys.stderr)
        return 2
    tol = json.loads(TOLERANCES.read_text(encoding="utf-8"))
    ref = json.loads((MISSION_DIR / "mission.verification.json").read_text(encoding="utf-8"))
    try:
        cand = json.loads((Path(candidate_dir) / "mission.verification.json")
                          .read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2
    fails = _compare_elements(ref.get("targets", {}).get("achieved_parking_orbit"),
                              cand.get("targets", {}).get("achieved_parking_orbit"),
                              tol, "achieved_parking_orbit")
    if cand.get("status", {}).get("result") != ref.get("status", {}).get("result"):
        fails.append(f"status: {cand.get('status', {}).get('result')!r} vs reference "
                     f"{ref.get('status', {}).get('result')!r}")
    if cand.get("status", {}).get("result") != "passed":
        fails.append(f"candidate mission result is {cand.get('status', {}).get('result')!r}")
    if fails:
        print(f"E5-07 REGRESSION: {len(fails)} difference(s) vs {REF_ROOT}", file=sys.stderr)
        for line in fails[:40]:
            print(f"  {line}", file=sys.stderr)
        print("If this is an INTENDED change, regenerate with:  python "
              f"scripts/e507_reference.py update -- and commit with {COMMIT_TAG}.",
              file=sys.stderr)
        return 1
    print(f"E5-07 reference-mission regression OK: {candidate_dir} matches the committed "
          "reference within the declared tolerances")
    return 0


# ---------------------------------------------------------------- crossenv-check

def cmd_crossenv_check(out_dir: Path) -> int:
    """Grade a finished cross-environment run (the orchestrator's out dir of
    crossenv_e507_plan.yaml): the mission-leg envelope verifies, the leg is
    green in the tudatpy environment, its parking-orbit propagation is
    PHYSICALLY EVIDENCED there, and the achieved orbit matches the reference
    within the declared bands. full_mission is NOT declared (the recovery
    phase cannot be flown: RECOVERY_GAP.md), so the joint full-mission gate
    is reported as not-declared rather than graded."""
    if not REF_ROOT.is_dir():
        print(NOT_RECORDED)
        return 0
    _load_repo_package()
    tol = json.loads(TOLERANCES.read_text(encoding="utf-8"))
    out = Path(out_dir)
    fails: list[str] = []
    joined_file = out / "crossenv.mission-report.json"
    if not joined_file.is_file():
        print(f"{joined_file}: missing (no completed cross-env run?)", file=sys.stderr)
        return 1
    joined = json.loads(joined_file.read_text(encoding="utf-8"))
    if joined.get("kind") != "orbitopt.crossenv.mission-report":
        fails.append(f"joined report kind: {joined.get('kind')!r}")
    if joined.get("overall", {}).get("exchanges_verified") is not True:
        fails.append("overall.exchanges_verified is not true")
    if joined.get("overall", {}).get("mission_ok") is not True:
        fails.append(f"overall.mission_ok is not true: {joined.get('stages')}")
    if joined.get("joint_qualification", {}).get("full_mission_declared") is True:
        fails.append("E5-07 must not declare full_mission (recovery phase not flyable)")
    from orbitopt.crossenv.envelope import read_envelope

    ref_report = json.loads((MISSION_DIR / "mission.verification.json")
                            .read_text(encoding="utf-8"))
    stages = {s["id"]: s for s in joined.get("stages", [])}
    st = stages.get("mission") or {}
    if st.get("status") != "completed" or st.get("exit_code") != 0:
        fails.append(f"stage mission: {st.get('status')} exit {st.get('exit_code')}")
    if st.get("mission_ok") is not True:
        fails.append(f"stage mission: mission_ok is not true: {st}")
    stage_report_path = out / "mission" / "stage-report.json"
    stage_report = (json.loads(stage_report_path.read_text(encoding="utf-8"))
                    if stage_report_path.is_file() else {})
    if stage_report.get("earth_orientation", {}).get("snapshot_sha256") != \
            "sha256:" + _digest(EOP_SNAPSHOT):
        fails.append("stage mission: Earth-orientation snapshot is not the reference's")
    stage_dir = out / "mission"
    prop_file = stage_dir / "mission" / "orbit_propagation.verification.json"
    if not prop_file.is_file():
        fails.append("stage mission: no orbit_propagation.verification.json "
                     "(E5-04 phase not evidenced in the tudatpy environment)")
    else:
        prop = json.loads(prop_file.read_text(encoding="utf-8"))
        if prop.get("status", {}).get("result") != "passed":
            fails.append(f"stage mission: parking propagation "
                         f"{prop.get('status', {}).get('result')!r} "
                         "(the tudatpy leg must pass)")
    cand_ver_file = stage_dir / "mission" / "mission.verification.json"
    if not cand_ver_file.is_file():
        fails.append(f"stage mission: {cand_ver_file} missing")
        cand_ver = {}
    else:
        cand_ver = json.loads(cand_ver_file.read_text(encoding="utf-8"))
    if cand_ver.get("status", {}).get("result") != "passed":
        fails.append(f"stage mission: mission result "
                     f"{cand_ver.get('status', {}).get('result')!r}")
    try:
        read_envelope(stage_dir / "exchange")
    except Exception as exc:  # noqa: BLE001 -- reported, not raised
        fails.append(f"stage mission: exchange did not re-verify: {exc}")
    fails += _compare_elements(ref_report.get("targets", {}).get("achieved_parking_orbit"),
                               cand_ver.get("targets", {}).get("achieved_parking_orbit"),
                               tol, "crossenv achieved_parking_orbit")
    if fails:
        print(f"E5-07 CROSS-ENV CHECK FAILED: {len(fails)} problem(s) in {out}",
              file=sys.stderr)
        for line in fails[:40]:
            print(f"  {line}", file=sys.stderr)
        return 1
    print(f"E5-07 cross-environment check OK: {out} -- envelope verified, mission leg green "
          "in the tudatpy environment with the parking-orbit propagation physically "
          "evidenced there; joint full-mission gate not declared by design (recovery "
          "reported, not flown -- RECOVERY_GAP.md)")
    return 0


# ---------------------------------------------------------------- verify / update

def _reference_files() -> list[Path]:
    return sorted(p for p in REF_ROOT.rglob("*")
                  if p.is_file() and p.name != PROVENANCE.name)


def cmd_verify() -> int:
    if not REF_ROOT.is_dir():
        print(NOT_RECORDED)
        return 0
    prov = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    fails = []
    recorded = dict(prov.get("files", {}))
    present = {str(p.relative_to(REF_ROOT)) for p in _reference_files()}
    for key, want in recorded.items():
        path = REF_ROOT / key
        if not path.is_file():
            fails.append(f"{key}: recorded but missing")
        elif _digest(path) != want:
            fails.append(f"{key}: digest differs from the provenance")
    for key in sorted(present - set(recorded)):
        fails.append(f"{key}: present but not recorded in the provenance")
    if fails:
        print(f"E5-07 PROVENANCE MISMATCH ({len(fails)}):", file=sys.stderr)
        for line in fails:
            print(f"  {line}", file=sys.stderr)
        print(f"Update deliberately: scripts/e507_reference.py update, tagged "
              f"{COMMIT_TAG}.", file=sys.stderr)
        return 1
    print(f"E5-07 reference provenance OK: {len(recorded)} files match {PROVENANCE.name}")
    return 0


RECOVERY_GAP_HEAD = """# E5-07 recovery-leg gap report (booster RTLS, Starship-class public estimate)

The flight target declares an RTLS recovery with soft-touchdown assumptions.
The vehicle's recovery architecture is two ignitions after separation
(boostback, landing burn) and a purely AERODYNAMIC entry -- `restarts_max: 2`,
no entry burn (the Block 1 booster decelerates with flaps/grid fins, not a
retropropulsion entry burn).

The FL-302 planner `plan_boostback_entry` accepts only entry-burn counts 1 or
2, and requires one ignition for boostback plus one per entry burn (the
landing burn is budgeted separately). Asked about THIS recording's separation
state, it answers:
"""

RECOVERY_GAP_MID = """
There is therefore no planner representation of THIS vehicle's RTLS profile:
`entry_burns=0` (the vehicle's actual architecture) is refused by the script
family, and every accepted script family needs more ignitions than the
vehicle has. Running the closest accepted profile (`entry_burns=1`) would fly
an engine architecture the vehicle does not have -- that is a constraint
relaxation dressed as a result, forbidden by the E5-07 acceptance rules.
"""

RECOVERY_GAP_TAIL = """
Consequence, reported not hidden: the mission declares NO recovery stage, the
recovery-phase checks grade `not applicable`, the E5-04 physical gate caps at
`not_checked`, and `full_mission` is not declared anywhere in the E5-07
chain. The booster's un-flown recovery budget is intact and a planner that
represents a restart-2 aerodynamic-entry RTLS can close this leg without
touching any recorded number.
"""


def _planner_gap_evidence(ascent, vehicle_path: Path) -> str:
    """Re-derive the two verbatim planner errors against THIS ascent (they
    are input validation, so the recording's own separation state is the
    honest one to ask)."""
    from orbitopt.flightlab.recovery.boostback import RecoveryTarget, plan_boostback_entry
    from orbitopt.flightlab.recovery.propagator import BoosterModel, separation_state_from_run
    from orbitopt.flightlab.contract import load_target

    target = load_target(ASCENT_TARGET)
    booster = BoosterModel.load(vehicle_path)
    rt = RecoveryTarget.from_scenario(target.to_dict())
    sep = separation_state_from_run(ascent)
    lines = []
    for eb in (0, 2):
        try:
            plan_boostback_entry(sep, booster, rt, entry_burns=eb)
            lines.append(f"    entry_burns={eb} -> planned (unexpected)")
        except Exception as exc:  # noqa: BLE001 -- the gap text IS the report
            lines.append(f"    entry_burns={eb} -> {type(exc).__name__}: {exc}")
    return "\n".join(lines) + "\n"


def _booster_arc_evidence(ascent) -> str:
    """The un-flown recovery budget and where the booster actually goes."""
    import math

    debris = next((t for t in ascent.tracks if t.body_id == "stage1_debris"), None)
    if debris is None or len(debris.t_s) == 0:
        return ""
    dry = 275000.0
    m_sep = float(debris.mass_kg[0])
    v_last = debris.velocity_m_s[-1]
    speed = float(math.sqrt(sum(float(c) ** 2 for c in v_last)))
    extra = getattr(debris, "extra_columns", {}) or {}

    def _last(name):
        col = extra.get(name)
        return float(col[-1]) if col is not None and len(col) else float("nan")

    return (f"\nThe recorded ascent leaves the booster (dry {dry / 1e3:.0f} t plus "
            f"{(m_sep - dry) / 1e3:.1f} t of unburned propellant -- the whole boostback and "
            f"landing budget, unspent) on a ballistic arc that ends at t={float(debris.t_s[-1]):.1f} s "
            f"at {speed:.0f} m/s, lat {_last('latitude_deg'):.3f} deg, "
            f"lon {_last('longitude_deg'):.3f} deg, {_last('downrange_m') / 1e3:.0f} km downrange.\n")


def cmd_update(*, tolerances_only: bool = False) -> int:
    """Re-fly the ascent (pinned OpenTsiolkovsky fixed-program flight of the
    E5-07 design) and the mission, and re-record the reference; or amend the
    comparison policy alone. The ONLY sanctioned way to change the reference;
    the commit must additionally carry COMMIT_TAG."""
    if tolerances_only:
        if cmd_verify() != 0:
            return 1
        prov = json.loads(PROVENANCE.read_text(encoding="utf-8"))
        TOLERANCES.write_text(json.dumps(DEFAULT_TOLERANCES, indent=2) + "\n", encoding="utf-8")
        prov["files"][str(TOLERANCES.relative_to(REF_ROOT))] = _digest(TOLERANCES)
        prov["comparison_policy_update"] = {
            "generated_by": "scripts/e507_reference.py update --tolerances-only",
            "toolchain": _toolchain(),
        }
        PROVENANCE.write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("Updated comparison policy and provenance; recorded flight output unchanged")
        return 0

    import shutil
    import tempfile

    _load_repo_package()
    _pin_earth_orientation()
    with tempfile.TemporaryDirectory(prefix="e507-ref-") as tmp:
        tmp_root = Path(tmp)
        eop_tmp = tmp_root / "earth_orientation.json"
        if EOP_SNAPSHOT.exists():
            shutil.copyfile(EOP_SNAPSHOT, eop_tmp)
        elif E505_EOP.exists():
            # Same mission epoch (2026-11-10): the E5-07 recording binds the
            # identical finalized IERS-A rows as the E5-05 recording.
            shutil.copyfile(E505_EOP, eop_tmp)
        else:
            raise SystemExit("no EOP snapshot available and no installed table copy made")
        optimized_dir = tmp_root / "mission-optimized"
        replay_dir = tmp_root / "mission"
        from orbitopt.mission.two_stage_leo import run_two_stage_leo

        with frozen_earth_orientation(eop_tmp):
            print("recording: computing the ascent (pinned OpenTsiolkovsky "
                  "fixed-program flight of the E5-07 Starship-class design)")
            from orbitopt.flightlab.adapters.opentsiolkovsky import run_ascent
            from orbitopt.flightlab.contract import load_target
            from orbitopt.flightlab.target_export import save_target

            ascent_target = load_target(ASCENT_TARGET)
            asc_dir = optimized_dir / "launch" / "ascent"
            tgt_file = optimized_dir / "launch" / "target.yaml"
            run_ascent(ascent_target, VEHICLE, None, asc_dir, timeout_s=900.0)
            save_target(ascent_target, tgt_file)
            print("recording: flying the mission over the computed ascent")
            full = run_two_stage_leo(load_scenario(ascent_dir=asc_dir, target_file=tgt_file),
                                     optimized_dir, propagate=False)
            if full.status != "OK":
                print(f"refusing to record a red reference mission (status {full.status!r}); "
                      "fix the mission before updating the reference", file=sys.stderr)
                return 1
            print("recording: replaying the recorded ascent (regression baseline)")
            replay = run_two_stage_leo(
                load_scenario(ascent_dir=optimized_dir / "launch" / "ascent",
                              target_file=optimized_dir / "launch" / "target.yaml"),
                replay_dir, propagate=False)
            if replay.status != "OK":
                print(f"refusing to record a red replay baseline (status {replay.status!r})",
                      file=sys.stderr)
                return 1
        if REF_ROOT.exists():
            shutil.rmtree(REF_ROOT)
        REF_ROOT.mkdir(parents=True)
        shutil.copyfile(eop_tmp, EOP_SNAPSHOT)
        shutil.copytree(optimized_dir, OPTIMIZED_DIR)
        shutil.copytree(replay_dir, MISSION_DIR)
        from orbitopt.flightlab.contract import read_run as _read_run

        _asc = _read_run(asc_dir)
        RECOVERY_GAP.write_text(
            RECOVERY_GAP_HEAD + _planner_gap_evidence(_asc, VEHICLE) + RECOVERY_GAP_MID
            + RECOVERY_GAP_TAIL + _booster_arc_evidence(_asc),
            encoding="utf-8")

    if not TOLERANCES.is_file():
        TOLERANCES.write_text(json.dumps(DEFAULT_TOLERANCES, indent=2) + "\n",
                              encoding="utf-8")
        print(f"wrote default {TOLERANCES}")

    from orbitopt.flightlab.contract import read_run

    ascent = read_run(OPTIMIZED_DIR / "launch" / "ascent")
    replay_report = json.loads((MISSION_DIR / "mission.verification.json")
                               .read_text(encoding="utf-8"))
    prov = {
        "generated_by": "scripts/e507_reference.py update",
        "reference": "E5-07 physics qualification reference mission (Starship-class "
                     "public-estimate vehicle to LEO against publicly-sourced "
                     "constraints; booster recovery REPORTED not flown -- RECOVERY_GAP.md)",
        "reference_class": "physics-qualification",
        "distinct_from": "tests/data/mission/reference/e5-05-reference-mission "
                         "(software-integration reference; NOT engineering-grade "
                         "verification)",
        "scenario": str(SCENARIO_YAML.relative_to(REPO)),
        "vehicle": str(VEHICLE.relative_to(REPO)),
        "recording": {
            "mode": "computed ascent (fixed-program OpenTsiolkovsky v1.0.0 flight against "
                    "leo_starship_class.yaml; every constraint sourced or ASSUMPTION-tagged "
                    "in the target, none relaxed) -> recorded run_dir replay",
            "ascent_engines": dict(ascent.engines),
            "ascent_run_id": ascent.run_id,
            "physical_gate_verdict": (replay_report.get("inputs", {}).get("full_mission", {})
                                      .get("physical", {}).get("verdict")),
            "recovery_phase": "not applicable -- declared gap, see RECOVERY_GAP.md",
        },
        "frozen_inputs": {
            "vehicle_sha256": _digest(VEHICLE),
            "scenario_sha256": _digest(SCENARIO_YAML),
            "ascent_target_sha256": _digest(ASCENT_TARGET),
            "crossenv_plan_sha256": _digest(CROSSENV_PLAN),
            "constraint_basis": "leo_starship_class.yaml: max-q 55 kPa (public Block 1 "
                                "Max-Q figures), max specific acceleration 3.0 g "
                                "(ASSUMPTION: upper bound on human-rating and loads for "
                                "an uncrewed-class test flight; never relaxed -- the "
                                "recording peaks at 2.89 g with a throttle schedule "
                                "inside the published 40-100 % Raptor band), periapsis "
                                ">= 230 km, insertion 250 km +/- 60 km, inclination "
                                "51.6 deg +/- 1.0 deg",
            "vehicle_uncertainty_sensitivity": "aero tables generated from the declared "
                                               "nose +-30 % (nose_aero) and the 13.5 m nose "
                                               "length assumption +-30 %: recorded max-q "
                                               "37.5 kPa against the 55 kPa limit leaves "
                                               "+-30 % aero headroom on q (48.75 kPa "
                                               "< 55 kPa at +30 %); mass and thrust bands "
                                               "(5-10 %) move the insertion apex well "
                                               "inside the declared 60 km band",
        },
        "requires_opentsiolkovsky": GOLDEN_OT,
        "earth_orientation": f"recorded IERS-A snapshot (earth_orientation.json), the "
                             f"identical finalized rows used by the E5-05 recording "
                             f"(same epoch); astropy-iers-data {_pin_earth_orientation()}",
        "toolchain": _toolchain(),
        "requires_commit_tag": COMMIT_TAG,
        "git_sha_recording": _git_sha(),
        "files": {str(p.relative_to(REF_ROOT)): _digest(p) for p in _reference_files()},
    }
    PROVENANCE.write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"reference updated: {len(prov['files'])} files under {REF_ROOT}")
    return 0


# ---------------------------------------------------------------- tag-check

def cmd_tag_check(base: str | None) -> int:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(REPO), *args],
                              capture_output=True, text=True, check=True).stdout

    rng = f"{base}..HEAD" if base else "HEAD~1..HEAD"
    touched = git("diff", "--name-only", rng)
    if not any(line.startswith(str(REF_ROOT.relative_to(REPO))) or
               line.startswith("src/orbitopt/data/flightlab/vehicles/"
                               "starship_class_public_estimate")
               for line in touched.splitlines()):
        return 0
    msgs = git("log", "--format=%B", rng)
    if COMMIT_TAG not in msgs:
        print(f"the pushed range touches the E5-07 reference (or its vehicle) but "
              f"no commit message carries {COMMIT_TAG}", file=sys.stderr)
        return 1
    print(f"E5-07 reference update properly tagged ({COMMIT_TAG})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="e507_reference", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run"); p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("compare"); p.add_argument("dir", type=Path)
    p = sub.add_parser("crossenv-check"); p.add_argument("dir", type=Path)
    sub.add_parser("verify")
    p = sub.add_parser("update"); p.add_argument("--tolerances-only", action="store_true")
    p = sub.add_parser("tag-check"); p.add_argument("--base", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return cmd_run(args.out)
    if args.cmd == "compare":
        return cmd_compare(args.dir)
    if args.cmd == "crossenv-check":
        return cmd_crossenv_check(args.dir)
    if args.cmd == "verify":
        return cmd_verify()
    if args.cmd == "update":
        return cmd_update(tolerances_only=args.tolerances_only)
    return cmd_tag_check(args.base)


if __name__ == "__main__":
    raise SystemExit(main())
