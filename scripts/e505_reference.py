#!/usr/bin/env python3
"""E5-05 reproducible full-physical reference mission: run, compare, verify, update.

The committed reference under ``tests/data/mission/reference/e5-05-reference-mission/``
(task #272) is the epic-P0 reference scenario in its complete form -- BOTH
branches flown from a COMPUTED ascent, nothing preset:

* ``mission-optimized/`` -- the RECORDED mission of record
  (``src/orbitopt/data/mission_two_stage_leo_rtls_reference.yaml``): the
  ascent computed by the pinned OpenTsiolkovsky v1.0.0 binary
  (``launch/ascent``, raw backend outputs included), the M-04 launch-pipeline
  report and the FlightTarget it built (``launch/target.yaml``, recovery
  block included), the booster flown RTLS driven from the canonical
  separation handoff, and the unified ``mission.verification.json``. This
  directory is the provenance anchor: every regression artifact descends
  from it.
* ``mission/`` -- the frozen REPLAY RESULT: ``run_two_stage_leo`` over the
  recorded ascent in ``run_dir`` mode (the packaged
  ``src/orbitopt/data/mission_two_stage_leo_rtls_replay.yaml``), recovery
  re-flown on the pinned recipe from the same handoff. THIS is what CI
  re-flies and ``compare`` grades -- the optimize half is recorded once and
  owned by the Gate-B M-07 ascent regression, not re-optimised per push.
* ``earth_orientation.json`` -- the IERS EOP snapshot the recording used
  (finalized IERS-A rows bracketing the mission epoch); ``run``/``compare``
  bind it via ``frozen_earth_orientation`` so the frame bridge is the
  recording's, whatever table the host ships.
* ``tolerances.json`` / ``PROVENANCE.json`` -- the declared numeric bands
  with their source (below), and the digests + full provenance of the
  recording: code version (git sha), dependency matrix, OpenTsiolkovsky and
  gfold pins, IERS snapshot source, environment. The acceptance criterion's
  "provenance" and "tolerance with a source" live in these two files.

Tolerance provenance. The bands below (written by ``update``, kept verbatim
afterwards -- changing them is itself a tagged reference update) exist
because the replay re-flies the RECOVERY chain on a possibly different
build of the same pinned stack: the ascent itself is replayed byte-for-byte
from ``mission-optimized/launch/ascent`` (no tolerance applies -- digest
identity, enforced by ``verify``), and the recovery plan restarts from the
identical separation state on FL-305's pinned warm start
(``initial_guess`` + ``max_fevals: 18``, the FL-305 acceptance recipe), so
the only legitimate source of difference is cross-build floating-point
drift. 1 cm / 0.1 mm/s flight-state bands (as E2-04 established on the same
chain) leave the pinned-solver results far more room than gfold's own solver
tolerances need while still failing any physics change that matters; the
1e-6 transform/constants bands carry the platform-dependent frame-bridge
roundoff (~1.4e-7 on the reviewed Linux/macOS builds); the landing-plan
diagnostic band (1 g fuel / 10 ms TOF) is 50x inside plan_scp's own 0.05 kg
stopping budget and 1/10 of the closed-loop landing integrator's 0.1 s step.

Subcommands
    run       --out DIR   replay the reference mission into DIR (needs the
                          landing stack: gfold 0.3.2, per envs/
                          flightlab-landing-ci.txt; no OpenTsiolkovsky, no pygmo)
    compare DIR           compare a fresh replay against the reference
    crossenv-check DIR    grade a cross-environment run (
                          src/orbitopt/data/crossenv_e505_plan.yaml) against
                          the reference: exchanges verified, mission green,
                          the recovery leg driven from the reference's
                          separation handoff, run numerics within tolerance
    verify                provenance check: committed files unmodified
    update                re-fly the WHOLE mission (OpenTsiolkovsky binary +
                          pygmo + gfold; ~30 min) and re-record the reference
      --tolerances-only   update policy/EOP metadata; preserve flight outputs
    tag-check --base SHA  enforce the [e505-reference-update] commit tag

G5 discipline (as in E2-04/FL-601): the reference is updated ONLY by
``update``, and the commit carrying the change must say so with the
``[e505-reference-update]`` tag -- CI runs ``verify`` and ``tag-check``.

    python scripts/e505_reference.py run --out /tmp/e505
    python scripts/e505_reference.py compare /tmp/e505
    python scripts/e505_reference.py crossenv-check runs/crossenv-e5-05
    python scripts/e505_reference.py verify
    python scripts/e505_reference.py update   # maintainers, with the tag
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REF_ROOT = REPO / "tests" / "data" / "mission" / "reference" / "e5-05-reference-mission"
OPTIMIZED_DIR = REF_ROOT / "mission-optimized"
MISSION_DIR = REF_ROOT / "mission"
TOLERANCES = REF_ROOT / "tolerances.json"
PROVENANCE = REF_ROOT / "PROVENANCE.json"
EOP_SNAPSHOT = REF_ROOT / "earth_orientation.json"
NOT_RECORDED = (
    "E5-05 reference NOT recorded yet: no E5-05 recording exists under "
    f"{REF_ROOT}. RTLS from a computed OpenTsiolkovsky ascent is physically "
    "infeasible for the current catalog vehicle (stage-1 separates dry, and "
    "no stage-2 slack can carry the recovery propellant RTLS needs -- see the "
    "ContextDB record e5-05-rtls-infeasibility-evidence). The recording "
    "awaits an epic-owner decision (vehicle/ascent redesign or acceptance "
    "revision); every consumer of the reference skips, and arms itself, on "
    "the directory's existence.")
SCENARIO_YAML = REPO / "src" / "orbitopt" / "data" / "mission_two_stage_leo_rtls_reference.yaml"
REPLAY_YAML = REPO / "src" / "orbitopt" / "data" / "mission_two_stage_leo_rtls_replay.yaml"
VEHICLE = REPO / "src" / "orbitopt" / "data" / "flightlab" / "vehicles" / "synthetic_two_stage.yaml"
COMMIT_TAG = "[e505-reference-update]"
GOLDEN_GFOLD = "0.3.2"
GOLDEN_OT = "1.0.0"
MISSION_EPOCH = "2026-11-10T00:00:00"  # the launch declaration's date_utc (finalized IERS)

#: Default declared tolerances -- every band's source is the module
#: docstring's "Tolerance provenance"; written by ``update`` on first use,
#: kept verbatim afterwards (changing them is itself a tagged update). A
#: numeric comparison passes when |candidate - reference| <= abs + rel*|ref|.
DEFAULT_TOLERANCES = {
    "t_s_abs": 1e-6,
    "mass_kg_abs": 1e-6,
    # Re-flown recovery numerics: cross-build drift only (docstring).
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
    # Advisory SOCP insertion-plan diagnostics, not flown touchdown values.
    # 1 g is 50x tighter than plan_scp's 0.05 kg stopping budget; 10 ms is
    # one tenth of the closed-loop landing integrator's default 0.1 s step.
    "landing_plan_diagnostic_abs": {"fuel_planned_kg": 1e-3, "tof_flying_s": 1e-2},
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


def _git_sha() -> str:
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip() or "unknown"


def _pin_earth_orientation() -> str:
    """Disable daily IERS downloads and identify the installed data package.
    The frame bridge itself uses :func:`frozen_earth_orientation`."""
    from astropy.utils import iers

    iers.conf.auto_download = False
    iers.conf.iers_degraded_accuracy = "ignore"
    try:
        import astropy_iers_data

        return str(astropy_iers_data.__version__)
    except ImportError:
        return "astropy bundled"


@contextmanager
def frozen_earth_orientation():
    """Bind the reference's EOP snapshot for the duration of a run, restoring
    the caller's table afterwards (same discipline as scripts/
    e204_reference.py: a host shipping a newer table must not move the
    frame bridge under a regression run)."""
    from astropy.utils import iers
    import astropy.units as u
    import numpy as np

    snapshot = json.loads(EOP_SNAPSHOT.read_text())
    units = {"MJD": u.day, "UT1_UTC": u.s, "PM_x": u.arcsec, "PM_y": u.arcsec}
    table = iers.IERS({name: np.array([row[name] for row in snapshot["rows"]]) * unit
                       for name, unit in units.items()})
    with iers.earth_orientation_table.set(table):
        yield


def _write_earth_orientation_snapshot() -> None:
    """Record the finalized EOP rows bracketing the mission epoch from the
    installed table. Kept verbatim afterwards: an existing snapshot is never
    overwritten outside a full ``update``."""
    if EOP_SNAPSHOT.exists():
        return
    from astropy.time import Time
    from astropy.utils import iers
    import astropy_iers_data

    table = iers.IERS_A.open(iers.IERS_A_FILE)
    day = math.floor(Time(MISSION_EPOCH).mjd)
    rows = [{name: float(row[name].value) for name in ("MJD", "UT1_UTC", "PM_x", "PM_y")}
            for row in table if day - 1 <= float(row["MJD"].value) <= day + 2]
    if len(rows) != 4:
        raise RuntimeError("installed table does not bracket the mission epoch "
                           f"({MISSION_EPOCH}): {len(rows)} rows")
    EOP_SNAPSHOT.write_text(json.dumps({
        "source": "IERS-A finals2000A.all",
        "source_version": str(astropy_iers_data.__version__),
        "epoch_utc": MISSION_EPOCH,
        "units": {"MJD": "day", "UT1_UTC": "s", "PM_x": "arcsec", "PM_y": "arcsec"},
        "rows": rows}, indent=2, sort_keys=True) + "\n")


def _toolchain() -> dict:
    """The backend-version fingerprint of the stack that records/gates the
    reference: every package whose numbers land in the reference files."""
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
        "orbitopt": _v("orbitopt") if _v("orbitopt") != "absent" else _git_sha(),
        "gfold": _v("gfold"),
        "cvxpy": _v("cvxpy"),
        "pygmo": _v("pygmo"),
        "numpy": _v("numpy"),
        "scipy": _v("scipy"),
        "astropy": _v("astropy"),
        "astropy_iers_data": _pin_earth_orientation(),
        "opentsiolkovsky": ot,
        "tudatpy": "recorded absent; cross-environment replay evidences the "
                   "propagation leg in the tudatpy environment (crossenv_e505_plan.yaml)",
    }


# ---------------------------------------------------------------- scenario

def _absolutise_doc_paths(doc: dict) -> dict:
    for key in ("run_dir", "target_file"):
        if doc.get("ascent", {}).get(key):
            doc["ascent"][key] = str((REPO / doc["ascent"][key]).resolve())
    if doc.get("recovery", {}).get("target_file"):
        doc["recovery"]["target_file"] = str((REPO / doc["recovery"]["target_file"]).resolve())
    return doc


def load_replay_scenario(ascent_dir: Path | None = None, target_file: Path | None = None):
    """The packaged replay scenario with its repo-relative reference paths
    resolved (``ascent_dir``/``target_file`` override where the recorded
    ascent is read from -- used only by ``update`` while recording)."""
    _load_repo_package()
    import yaml

    from orbitopt.mission.two_stage_leo import TwoStageLeoScenario

    doc = yaml.safe_load(REPLAY_YAML.read_text(encoding="utf-8"))["scenario"]
    doc = _absolutise_doc_paths(doc)
    if ascent_dir is not None:
        doc["ascent"]["run_dir"] = str(ascent_dir)
    if target_file is not None:
        doc["ascent"]["target_file"] = str(target_file)
        doc["recovery"]["target_file"] = str(target_file)
    return TwoStageLeoScenario.from_dict(doc)


def load_full_scenario():
    """The mission of record (optimize mode -- needs the compute stack)."""
    _load_repo_package()
    from orbitopt.mission.two_stage_leo import load_scenario as _load

    return _load(SCENARIO_YAML)


# ---------------------------------------------------------------- run

def _run_replay(out_dir: Path):
    from orbitopt.mission.two_stage_leo import run_two_stage_leo

    out_dir.mkdir(parents=True, exist_ok=True)
    with frozen_earth_orientation():
        return run_two_stage_leo(load_replay_scenario(), out_dir, propagate=False)


def cmd_run(out_dir: Path) -> int:
    if not REF_ROOT.is_dir():
        print(NOT_RECORDED)
        return 0
    _load_repo_package()
    _pin_earth_orientation()
    print(f"Earth orientation: frozen E5-05 snapshot (installed IERS data "
          f"{_pin_earth_orientation()}), no download")
    mission = _run_replay(out_dir)
    print(f"mission status: {mission.status}")
    print(f"physical gate:    "
          f"{mission.report['inputs']['full_mission'].get('physical', {}).get('verdict')}")
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
    if not math.isfinite(got) or not math.isfinite(ref):
        fails.append(f"{label}: non-finite numeric value {got!r} or reference {ref!r}")
        return
    if abs(got - ref) > atol + rtol * abs(ref):
        fails.append(f"{label}: {got!r} differs from reference {ref!r} by "
                     f"{abs(got - ref):.3e} > {atol:.1e} + {rtol:.1e}*|ref|")


def _compare_values(label: str, got, ref, atol: float, rtol: float,
                    fails: list[str]) -> None:
    if isinstance(got, bool) or isinstance(ref, bool):
        if type(got) is not type(ref) or got != ref:
            fails.append(f"{label}: {got!r} != reference {ref!r}")
    elif isinstance(got, (int, float)) and isinstance(ref, (int, float)):
        _num_diff(label, float(got), float(ref), atol, rtol, fails)
    elif isinstance(got, dict) and isinstance(ref, dict):
        _compare_dicts(label, got, ref, atol, rtol, fails)
    elif isinstance(got, (list, tuple)) and isinstance(ref, (list, tuple)):
        if len(got) != len(ref):
            fails.append(f"{label}: sequence lengths {len(got)} != reference {len(ref)}")
        for i, (g, r) in enumerate(zip(got, ref)):
            _compare_values(f"{label}[{i}]", g, r, atol, rtol, fails)
    elif got != ref:
        fails.append(f"{label}: {got!r} != reference {ref!r}")


def _compare_dicts(label: str, got: dict, ref: dict, atol: float, rtol: float,
                   fails: list[str], absolute_overrides: dict | None = None) -> None:
    if set(got) != set(ref):
        fails.append(f"{label}: keys {sorted(set(got) ^ set(ref))} differ")
    for k in sorted(set(got) & set(ref)):
        _compare_values(f"{label}[{k}]", got[k], ref[k],
                        (absolute_overrides or {}).get(k, atol), rtol, fails)


def compare_runs(candidate_dir: Path, tol: dict) -> list[str]:
    """Field-by-field comparison of the merged recovery run against the
    reference run (FL-601/E2-04 discipline; manifest fields that legitimately
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
                       tol["event_data_abs"], tol["event_data_rel"], fails,
                       absolute_overrides=(tol.get("landing_plan_diagnostic_abs")
                                           if ce.kind == re_.kind == "landing_start" else None))

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


def _compare_physical(label: str, got: dict, ref: dict, fails: list[str]) -> None:
    """The E5-04 physical gate: verdict, reason codes and the five required
    checks' verdicts must match the reference exactly (tri-state, no band)."""
    for key in ("verdict", "reason_codes"):
        if got.get(key) != ref.get(key):
            fails.append(f"{label}.{key}: {got.get(key)!r} != reference {ref.get(key)!r}")
    got_checks = {d.get("check"): d for d in got.get("checks", [])}
    ref_checks = {d.get("check"): d for d in ref.get("checks", [])}
    if set(got_checks) != set(ref_checks):
        fails.append(f"{label}.checks: ids {sorted(set(got_checks) ^ set(ref_checks))} differ")
    for name in sorted(set(got_checks) & set(ref_checks)):
        for key in ("verdict", "required", "checks_passed", "checks_total",
                    "reason_codes", "checks"):
            if got_checks[name].get(key) != ref_checks[name].get(key):
                fails.append(f"{label}.{name}.{key}: {got_checks[name].get(key)!r} "
                             f"!= reference {ref_checks[name].get(key)!r}")


def _compare_coupling(label: str, got: dict | None, ref: dict | None,
                      fails: list[str]) -> None:
    """E5-02 coupling across a regression run: both runs must branch from the
    one digest-identified separation state, branch-for-branch."""
    if bool(got) != bool(ref):
        fails.append(f"{label}: presence {bool(got)} != reference {bool(ref)}")
        return
    if not got:
        return
    if got.get("handoff_id") != ref.get("handoff_id"):
        fails.append(f"{label}.handoff_id: {got.get('handoff_id')!r} != reference "
                     f"{ref.get('handoff_id')!r} -- the two runs did not branch from "
                     "the same separation state")
    got_b = {b["kind"]: b for b in got.get("branches", [])}
    ref_b = {b["kind"]: b for b in ref.get("branches", [])}
    if set(got_b) != set(ref_b):
        fails.append(f"{label}.branches: kinds {sorted(set(got_b) ^ set(ref_b))} differ")
    for kind in sorted(set(got_b) & set(ref_b)):
        if got_b[kind].get("handoff_id") != ref_b[kind].get("handoff_id"):
            fails.append(f"{label}.branches[{kind}].handoff_id differs from the reference")
        if got_b[kind].get("passed") != ref_b[kind].get("passed"):
            fails.append(f"{label}.branches[{kind}].passed: {got_b[kind].get('passed')} "
                         f"!= reference {ref_b[kind].get('passed')}")


def _check_report_field(label: str, got, ref, tol, fails: list[str]) -> None:
    _compare_values(label, got, ref, tol["report_value_abs"], tol["report_value_rel"], fails)


def compare_report(candidate_report: dict, tol: dict) -> list[str]:
    """The mission-report half of the comparison: the verdict conjunction, the
    full check-bool surface, the phase verdicts, the E5-04 physical gate, the
    E5-02 coupled-branch identity and the achieved-orbit residuals of a fresh
    replay against the committed reference report. Path fields are not
    compared -- the candidate was run into its own directory; the run
    directory CONTENTS are compared by ``compare_runs``."""
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

    got_full = candidate_report.get("inputs", {}).get("full_mission", {})
    ref_full = ref.get("inputs", {}).get("full_mission", {})
    if "physical" in ref_full or "physical" in got_full:
        _compare_physical("physical", got_full.get("physical") or {},
                          ref_full.get("physical") or {}, fails)
    _compare_coupling("coupled_branches",
                      candidate_report.get("inputs", {}).get("handoffs", {}).get("coupled_branches"),
                      ref.get("inputs", {}).get("handoffs", {}).get("coupled_branches"), fails)
    got_rec, ref_rec = (candidate_report.get("inputs", {}).get("recovery") or {},
                        ref.get("inputs", {}).get("recovery") or {})
    if bool(got_rec.get("driven_from_separation")) != bool(ref_rec.get("driven_from_separation")):
        fails.append("recovery.driven_from_separation differs from the reference")
    if got_rec.get("separation_handoff_id") != ref_rec.get("separation_handoff_id"):
        fails.append(f"recovery.separation_handoff_id: {got_rec.get('separation_handoff_id')!r} "
                     f"!= reference {ref_rec.get('separation_handoff_id')!r}")

    _compare_dicts("targets.achieved", candidate_report["targets"]["achieved_parking_orbit"],
                   ref["targets"]["achieved_parking_orbit"],
                   tol["report_value_abs"], tol["report_value_rel"], fails)
    _compare_dicts("targets.deviations", candidate_report["targets"]["deviations"],
                   ref["targets"]["deviations"],
                   tol["report_value_abs"], tol["report_value_rel"], fails)
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
    from orbitopt.verify.report import read_report

    fails = compare_report(read_report(candidate_dir / "mission.verification.json"), tol)
    fails += compare_runs(candidate_dir / "recovery" / "run", tol)
    if fails:
        print(f"E5-05 REGRESSION: {len(fails)} difference(s) vs {REF_ROOT} "
              f"(tolerances: {TOLERANCES})", file=sys.stderr)
        for line in fails[:40]:
            print(f"  {line}", file=sys.stderr)
        if len(fails) > 40:
            print(f"  ... {len(fails) - 40} more", file=sys.stderr)
        print("If this is an INTENDED change (new gfold pin, intended physics "
              "shift), regenerate with:  python scripts/e505_reference.py update "
              f"-- and commit with {COMMIT_TAG} in the message.", file=sys.stderr)
        return 1
    print(f"E5-05 reference-mission regression OK: {candidate_dir} matches the committed "
          "reference within the declared tolerances")
    return 0


# ---------------------------------------------------------------- crossenv-check

def cmd_crossenv_check(out_dir: Path) -> int:
    """Grade a finished cross-environment run (the orchestrator's out dir of
    crossenv_e505_plan.yaml) against the reference: the two envelopes verify
    independently, both legs are green, the recovery leg was driven from the
    reference's separation handoff, and the sealed recovery run matches the
    reference within the declared tolerances."""
    if not REF_ROOT.is_dir():
        print(NOT_RECORDED)
        return 0
    _load_repo_package()
    from orbitopt.crossenv.envelope import read_envelope

    out = Path(out_dir)
    tol = json.loads(TOLERANCES.read_text(encoding="utf-8"))
    joined_file = out / "crossenv.mission-report.json"
    fails: list[str] = []
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

    ref_report = json.loads((MISSION_DIR / "mission.verification.json")
                            .read_text(encoding="utf-8"))
    ref_handoff = (ref_report.get("inputs", {}).get("handoffs", {})
                   .get("coupled_branches", {}) or {}).get("handoff_id")
    stages = {s["id"]: s for s in joined.get("stages", [])}
    for name in ("ascent", "recovery"):
        st = stages.get(name) or {}
        if st.get("status") != "completed" or st.get("exit_code") != 0:
            fails.append(f"stage {name}: {st.get('status')} exit {st.get('exit_code')}")
        if st.get("mission_ok") is not True:
            fails.append(f"stage {name}: mission_ok is not true")
        exchange = out / name / "exchange"
        try:
            read_envelope(exchange)  # validates the manifest schema + every payload digest
        except Exception as exc:  # noqa: BLE001 -- every failure reported, not raised
            fails.append(f"stage {name}: exchange did not re-verify: {exc}")
    asc = stages.get("ascent") or {}
    if "coupled-separation-handoff" not in (asc.get("payloads") or {}):
        fails.append("ascent stage sealed no coupled-separation-handoff payload")
    rec = stages.get("recovery") or {}
    rec_report_file = out / "recovery" / "stage-report.json"
    rec_report = (json.loads(rec_report_file.read_text(encoding="utf-8"))
                  if rec_report_file.is_file() else {})
    if rec_report.get("driven_from_separation") is not True:
        fails.append("recovery stage was not driven from the separation handoff")
    if rec_report.get("separation_handoff_id") != ref_handoff:
        fails.append(f"recovery stage separation_handoff_id "
                     f"{rec_report.get('separation_handoff_id')!r} != the reference's "
                     f"{ref_handoff!r} -- the two environments did not branch from "
                     "the one recorded state")
    fails += compare_runs(out / "recovery" / "exchange" / "recovery" / "run", tol)

    if fails:
        print(f"E5-05 CROSS-ENV CHECK FAILED: {len(fails)} problem(s) in {out}", file=sys.stderr)
        for line in fails[:40]:
            print(f"  {line}", file=sys.stderr)
        return 1
    print(f"E5-05 cross-environment check OK: {out} -- two envelopes verified, both legs "
          "green, booster driven from the reference separation handoff across the "
          "environment boundary, recovery numerics within tolerance")
    return 0


# ---------------------------------------------------------------- verify / update

def _reference_files() -> list[Path]:
    return sorted(p for p in REF_ROOT.rglob("*")
                  if p.is_file() and p.name != PROVENANCE.name)


def cmd_verify() -> int:
    if not REF_ROOT.is_dir():
        print(NOT_RECORDED)
        return 0
    if not PROVENANCE.is_file():
        print(f"{PROVENANCE}: missing -- run scripts/e505_reference.py update", file=sys.stderr)
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
              "changed without scripts/e505_reference.py update:", file=sys.stderr)
        for line in fails:
            print(f"  {line}", file=sys.stderr)
        return 1
    print(f"reference OK: {len(present)} files match {PROVENANCE.name} digests")
    return 0


def cmd_update(*, tolerances_only: bool = False) -> int:
    """Re-fly the WHOLE mission (optimize ascent on the pinned OpenTsiolkovsky
    + pygmo, then the run_dir replay) and re-record the reference, or amend
    the comparison policy alone. The ONLY sanctioned way to change the
    reference; the commit must additionally carry COMMIT_TAG."""
    if tolerances_only:
        if cmd_verify() != 0:
            return 1
        prov = json.loads(PROVENANCE.read_text(encoding="utf-8"))
        previous = prov.get("comparison_policy_update", {}).get(
            "previous_tolerances_sha256", _digest(TOLERANCES))
        _write_earth_orientation_snapshot()
        TOLERANCES.write_text(json.dumps(DEFAULT_TOLERANCES, indent=2) + "\n", encoding="utf-8")
        key = str(TOLERANCES.relative_to(REF_ROOT))
        prov["files"][key] = _digest(TOLERANCES)
        prov["files"]["earth_orientation.json"] = _digest(EOP_SNAPSHOT)
        prov["comparison_policy_update"] = {
            "generated_by": "scripts/e505_reference.py update --tolerances-only",
            "previous_tolerances_sha256": previous,
            "toolchain": _toolchain(),
        }
        PROVENANCE.write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("Updated comparison policy, EOP snapshot and provenance; recorded flight output unchanged")
        return 0

    import shutil
    import tempfile

    _load_repo_package()
    _pin_earth_orientation()
    _write_earth_orientation_snapshot()
    snapshot_bytes = EOP_SNAPSHOT.read_bytes()
    with tempfile.TemporaryDirectory(prefix="e505-ref-") as tmp:
        tmp_root = Path(tmp)
        optimized_dir = tmp_root / "mission-optimized"
        replay_dir = tmp_root / "mission"
        from orbitopt.mission.two_stage_leo import run_two_stage_leo

        with frozen_earth_orientation():
            print("recording: flying the mission of record (computed ascent, "
                  "optimize mode -- this takes tens of minutes)")
            full = run_two_stage_leo(load_full_scenario(), optimized_dir)
            if full.status != "OK":
                print(f"refusing to record a red reference mission (status {full.status!r}); "
                      "fix the mission before updating the reference", file=sys.stderr)
                return 1
            print("recording: replaying the recorded ascent (regression baseline)")
            replay = run_two_stage_leo(
                load_replay_scenario(ascent_dir=optimized_dir / "launch" / "ascent",
                                     target_file=optimized_dir / "launch" / "target.yaml"),
                replay_dir, propagate=False)
            if replay.status != "OK":
                print(f"refusing to record a red replay baseline (status {replay.status!r})",
                      file=sys.stderr)
                return 1
        if REF_ROOT.exists():
            shutil.rmtree(REF_ROOT)
        REF_ROOT.mkdir(parents=True)
        EOP_SNAPSHOT.write_bytes(snapshot_bytes)
        shutil.copytree(optimized_dir, OPTIMIZED_DIR)
        shutil.copytree(replay_dir, MISSION_DIR)

    if not TOLERANCES.is_file():
        TOLERANCES.write_text(json.dumps(DEFAULT_TOLERANCES, indent=2) + "\n",
                              encoding="utf-8")
        print(f"wrote default {TOLERANCES}")
    from orbitopt.flightlab.contract import read_run

    ascent = read_run(OPTIMIZED_DIR / "launch" / "ascent")
    replay_report = json.loads((MISSION_DIR / "mission.verification.json")
                               .read_text(encoding="utf-8"))
    prov = {
        "generated_by": "scripts/e505_reference.py update",
        "reference": "E5-05 reproducible full-physical reference mission "
                     "(computed ascent + parking + booster RTLS, coupled branches, "
                     "E5-04 physical gate)",
        "scenario": str(SCENARIO_YAML.relative_to(REPO)),
        "replay_scenario": str(REPLAY_YAML.relative_to(REPO)),
        "vehicle": str(VEHICLE.relative_to(REPO)),
        "recording": {
            "mode": "computed ascent (ascent.mode: optimize, M-07 compute recipe: cmaes "
                    "40x40, seed 11, 4 workers, refly) -> recorded run_dir replay",
            "ascent_engines": dict(ascent.engines),
            "ascent_run_id": ascent.run_id,
            "physical_gate_verdict": (replay_report.get("inputs", {}).get("full_mission", {})
                                      .get("physical", {}).get("verdict")),
            "separation_handoff_id": (replay_report.get("inputs", {}).get("handoffs", {})
                                      .get("coupled_branches", {}) or {}).get("handoff_id"),
        },
        "frozen_inputs": {
            "recovery_plan_options": {"initial_guess": [18.4, 17.75, 1.0, 4.36, 0.0,
                                                        17456.0, 23.0], "max_fevals": 18},
            "plan_options_source": "FL-305 acceptance recipe (NOMINAL_PLAN_GUESS warm "
                                   "start + FL-304 local-refinement budget), pinned in "
                                   "both scenario documents",
            "vehicle_sha256": _digest(VEHICLE),
            "scenario_sha256": _digest(SCENARIO_YAML),
            "replay_scenario_sha256": _digest(REPLAY_YAML),
        },
        "requires_gfold": GOLDEN_GFOLD,
        "requires_opentsiolkovsky": GOLDEN_OT,
        "earth_orientation": f"recorded IERS-A snapshot (earth_orientation.json), "
                            f"astropy-iers-data {_pin_earth_orientation()}",
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
    run_p = sub.add_parser("run", help="replay the reference mission into --out")
    run_p.add_argument("--out", required=True, type=Path)
    cmp_p = sub.add_parser("compare", help="compare a fresh replay dir against the reference")
    cmp_p.add_argument("dir", type=Path)
    cx_p = sub.add_parser("crossenv-check",
                          help="grade a crossenv_e505_plan.yaml run against the reference")
    cx_p.add_argument("dir", type=Path)
    sub.add_parser("verify", help="check committed reference files against PROVENANCE.json")
    update_p = sub.add_parser("update", help=f"re-fly and re-record the reference (commit the result with {COMMIT_TAG})")
    update_p.add_argument("--tolerances-only", action="store_true",
                          help="update comparison policy/EOP metadata without re-recording flight output")
    tag_p = sub.add_parser("tag-check", help="CI: reference diffs need the commit tag")
    tag_p.add_argument("--base", default=None)
    args = parser.parse_args(argv)
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
    if args.cmd == "tag-check":
        return cmd_tag_check(args.base)
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
