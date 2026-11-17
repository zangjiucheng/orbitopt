#!/usr/bin/env python3
"""FL-601 Gate B regression reference: record, verify, compare, update.

The committed reference under ``tests/data/flightlab/reference/`` is one real
Gate B flight (``orbitopt.flightlab.simulate.simulate`` on the packaged
``leo_two_stage`` target + ``synthetic_two_stage`` vehicle, ascent-only, no
tudatpy re-fly) flown with the pinned OpenTsiolkovsky v1.0.0 binary. The CI
slow job fetches that binary, flies the same mission and runs
``compare`` against the reference with the declared tolerances in
``tolerances.json``.

Subcommands
    run     --out DIR       fly the reference mission into DIR
    compare DIR             compare a fresh run against the reference
    verify                  provenance check: reference files unmodified
    update                  regenerate the reference from a fresh OT flight
    tag-check --base SHA    enforce the [fl601-reference-update] commit tag

G5 discipline: the reference is updated ONLY by ``update`` (which rewrites
the provenance digests), and the commit carrying the change must say so with
the ``[fl601-reference-update]`` tag -- CI runs ``verify`` (catches hand
edited reference files) and ``tag-check`` (catches updates smuggled in
without the tag).

    python scripts/fl601_reference.py verify
    python scripts/fl601_reference.py run --out /tmp/gateb
    python scripts/fl601_reference.py compare /tmp/gateb
    python scripts/fl601_reference.py update   # maintainers, with the tag
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REF_ROOT = REPO / "tests" / "data" / "flightlab" / "reference"
REF_RUN_DIR = REF_ROOT / "gateb-leo-two-stage"
TOLERANCES = REF_ROOT / "tolerances.json"
PROVENANCE = REF_ROOT / "PROVENANCE.json"
COMMIT_TAG = "[fl601-reference-update]"

TARGET = REPO / "src" / "orbitopt" / "data" / "flightlab" / "scenarios" / "leo_two_stage.yaml"
VEHICLE = REPO / "src" / "orbitopt" / "data" / "flightlab" / "vehicles" / "synthetic_two_stage.yaml"

#: Default declared tolerances, written by ``update`` on first use and kept
#: verbatim afterwards (changing them is itself a reference update). Each
#: numeric comparison passes when |candidate - reference| <= abs + rel*|reference|.
# Positions and velocities are GCRS states after the frame bridge; the
# reference is recorded in a container and CI flies on a native runner, and
# the two agree to ~2e-6 m / ~4e-7 m/s (the fairing's last ballistic samples
# grow the last-bit differences most). The tolerances sit ~50x above that
# floor -- 0.1 mm, 1 um/s -- and still far below any real change. Times
# likewise: an interpolated ground impact (the fairing's) moved by 3 ns, so
# 1 us, ~0.3 mm of the fairing's fall.
DEFAULT_TOLERANCES = {
    "t_s_abs": 1e-6,
    "mass_kg_abs": 1e-6,
    "position_m_abs": 1e-4,
    "velocity_m_s_abs": 1e-6,
    "extra_abs": {
        "altitude_m": 1e-4,
        "latitude_deg": 1e-9,
        "longitude_deg": 1e-9,
        "downrange_m": 1e-4,
        "dynamic_pressure_pa": 1e-2,
        "mach": 1e-7,
    },
    "extra_abs_default": 1e-6,
    "check_value_abs": 1e-9,
    "check_value_rel": 1e-9,
    "event_data_abs": 1e-9,
    "event_data_rel": 1e-9,
    # E8-04: band for re-deriving the committed GCRS tracks from the raw
    # recorded OT outputs under the pinned Earth orientation, on any lock-
    # stack platform (linux-x86_64, darwin-x86_64, darwin-arm64). It equals
    # the flight tolerances above because it exercises the same frame bridge;
    # keeping it separate lets cross-platform consistency be tightened (or
    # audited) without touching the flight comparison. Measured margin on
    # the recording stack: ~1e-9 m.
    "cross_platform": {"position_m_abs": 1e-4, "velocity_m_s_abs": 1e-6},
}


def _load_repo_package() -> None:
    sys.path.insert(0, str(REPO / "src"))


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------- run


def _pin_earth_orientation() -> str:
    """Disable IERS downloads and identify the installed data package.

    Thin delegation to the shared E8-04 mechanism (orbitopt.reference_eop) --
    the frame bridge itself is frozen per the reference snapshot, see
    :func:`frozen_reference`."""
    _load_repo_package()
    from orbitopt import reference_eop

    return reference_eop.pin_no_download()


#: The stack the committed Gate B frames were recorded on (git history,
#: commit ef59934); the EOP snapshot next to the reference pins the Earth
#: orientation those frames imply.
RECORDED_IERS_VERSION = "0.2026.8.31.0.57.9"
RECORDED_ASTROPY_SERIES = "6.1"
GATEB_EPOCH_UTC = "2026-11-10T12:00:00"
EOP_SNAPSHOT = REF_ROOT / "earth_orientation.json"
EOP_SNAPSHOT_NOTE = (
    "Gate B GCRS reference frames are defined by the DEFAULT table path of "
    "astropy 6.1.x with downloads off: the active table is IERS_B from "
    "astropy-iers-data 0.2026.8.31 (effective UT1-UTC +0.0097405 s at the "
    "epoch). The snapshot pins that table by version + effective values, NOT "
    "by an override built from the finals2000A (IERS-A) rows -- those rows "
    "carry a different UT1-UTC (-0.0613499 s; delta 0.071 s = 1.07 arcsec ~ "
    "28 m at orbital radius), so an override table is a different Earth "
    "orientation, not the recorded one. See docs/reference-tolerance-audit.md.")


def _ensure_earth_orientation_snapshot(force: bool = False) -> None:
    """Create the EOP snapshot on the recording stack (idempotent; ``force``
    re-records it from the currently installed stack, for a full re-record)."""
    _load_repo_package()
    from orbitopt import reference_eop

    if force and EOP_SNAPSHOT.exists():
        EOP_SNAPSHOT.unlink()
    reference_eop.write_pinned_bundled_snapshot(
        EOP_SNAPSHOT, GATEB_EPOCH_UTC, reference_eop.pin_no_download(),
        _installed_astropy_series(), EOP_SNAPSHOT_NOTE)


def _installed_astropy_series() -> str:
    import astropy

    return ".".join(str(astropy.__version__).split(".")[:2])


def frozen_reference():
    """Context manager freezing the frame bridge to the recorded reference
    Earth orientation, per the snapshot's mode (E8-04 shared mechanism)."""
    _load_repo_package()
    from orbitopt import reference_eop

    return reference_eop.freeze_reference(reference_eop.load_snapshot(EOP_SNAPSHOT))


def cmd_run(out_dir: Path) -> int:
    _load_repo_package()
    from orbitopt.flightlab.contract import load_target
    from orbitopt.flightlab.simulate import simulate

    iers_data = _pin_earth_orientation()
    print(f"Earth orientation: bundled IERS data only (astropy-iers-data {iers_data}), no download")

    out_dir.mkdir(parents=True, exist_ok=True)
    with frozen_reference():
        run = simulate(load_target(TARGET), VEHICLE, out_dir, refly=False,
                       run_recovery=False, timeout_s=300.0)
    print(f"flew {run.run_id} from {TARGET.name} + {VEHICLE.name} -> {out_dir}")
    print("  (ascent-only reference: refly=False, run_recovery=False -- the "
          "regression covers the OT adapter + checks pipeline, not the "
          "tudatpy re-fly or the FL-304 recovery chain, which are exercised "
          "by the fixture tests in the main CI job)")
    return 0


# --------------------------------------------------------------- compare


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
    """Field-by-field comparison of a fresh Gate B run against the committed
    reference, within the declared tolerances. Returns failure lines (empty
    list = match). Manifest fields that legitimately differ between machines
    and runs (created_utc, git_sha, toolchain, assumptions) are not
    compared -- the environment fingerprint they carry is checked separately
    (tests/test_flightlab_contract.py asserts it is present; CI's fetch step
    asserts the binary digest)."""
    _load_repo_package()
    from orbitopt.flightlab.contract import read_run

    cand = read_run(candidate_dir)
    ref = read_run(REF_RUN_DIR)
    fails: list[str] = []

    if cand.run_id != ref.run_id:
        fails.append(f"run_id: {cand.run_id!r} != reference {ref.run_id!r}")
    for field in ("target_sha256", "vehicle_sha256"):
        if getattr(cand, field) != getattr(ref, field):
            fails.append(f"{field}: differs from the reference run's input digest")
    if dict(cand.engines) != dict(ref.engines):
        fails.append(f"engines: {dict(cand.engines)} != reference {dict(ref.engines)}")

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
            for axis, a, b in zip("xyz", ct.position_m[i], rt.position_m[i]):
                _num_diff(f"{body}[{i}].{axis}_m", a, b, tol["position_m_abs"], 0.0, fails)
            for axis, a, b in zip(("vx", "vy", "vz"), ct.velocity_m_s[i], rt.velocity_m_s[i]):
                _num_diff(f"{body}[{i}].{axis}_m_s", a, b, tol["velocity_m_s_abs"], 0.0, fails)
            if set(ct.extra_columns) != set(rt.extra_columns):
                fails.append(f"{body}[{i}]: extra columns differ")
                continue
            for name in rt.extra_columns:
                _num_diff(f"{body}[{i}].{name}", ct.extra_columns[name][i], rt.extra_columns[name][i],
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


def cmd_compare(candidate_dir: Path) -> int:
    if not TOLERANCES.is_file():
        print(f"{TOLERANCES}: missing", file=sys.stderr)
        return 2
    with frozen_reference():
        fails = compare_runs(candidate_dir, json.loads(TOLERANCES.read_text(encoding="utf-8")))
    if fails:
        print(f"GATE B REGRESSION: {len(fails)} difference(s) vs "
              f"{REF_RUN_DIR} (tolerances: {TOLERANCES})", file=sys.stderr)
        for line in fails[:40]:
            print(f"  {line}", file=sys.stderr)
        if len(fails) > 40:
            print(f"  ... {len(fails) - 40} more", file=sys.stderr)
        print("If this is an INTENDED change (new OT pin, intended physics "
              "shift), regenerate with:  python scripts/fl601_reference.py update "
              f"-- and commit with {COMMIT_TAG} in the message.", file=sys.stderr)
        return 1
    print(f"Gate B regression OK: {candidate_dir} matches the committed reference "
          "within the declared tolerances")
    return 0


# --------------------------------------------------------------- verify / update


def _reference_files() -> list[Path]:
    return sorted(p for p in REF_ROOT.rglob("*")
                  if p.is_file() and p.name != PROVENANCE.name)


def cmd_verify() -> int:
    if not PROVENANCE.is_file():
        print(f"{PROVENANCE}: missing -- run scripts/fl601_reference.py update", file=sys.stderr)
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
              "changed without scripts/fl601_reference.py update:", file=sys.stderr)
        for line in fails:
            print(f"  {line}", file=sys.stderr)
        return 1
    print(f"reference OK: {len(present)} files match {PROVENANCE.name} digests")
    return 0


def cmd_update(*, tolerances_only: bool = False, reason: str = "") -> int:
    """Regenerate the reference from a real OT flight and rewrite the
    provenance digests. This is the ONLY sanctioned way to change the
    reference; the commit carrying the change must additionally carry
    COMMIT_TAG in its message (CI's tag-check enforces that, and CI's
    verify fails any change this script did not write).

    ``--tolerances-only`` amends the comparison policy (DEFAULT_TOLERANCES ->
    tolerances.json) plus the EOP snapshot and provenance digests WITHOUT
    re-flying; the frozen flight output and recording toolchain stay
    untouched. It requires ``--reason`` and appends an audit entry to
    PROVENANCE.json (E8-04, orbitopt.reference_audit)."""
    if tolerances_only:
        return cmd_update_tolerances_only(reason)
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory(prefix="fl601-ref-") as tmp:
        run_dir = Path(tmp) / "run"
        if cmd_run(run_dir) != 0:
            return 1
        if REF_RUN_DIR.exists():
            shutil.rmtree(REF_RUN_DIR)
        shutil.copytree(run_dir, REF_RUN_DIR)
    _ensure_earth_orientation_snapshot(force=True)
    if not TOLERANCES.is_file():
        TOLERANCES.parent.mkdir(parents=True, exist_ok=True)
        TOLERANCES.write_text(json.dumps(DEFAULT_TOLERANCES, indent=2) + "\n",
                              encoding="utf-8")
        print(f"wrote default {TOLERANCES}")
    files = {str(p.relative_to(REF_ROOT)): _digest(p) for p in _reference_files()}
    prov = {
        "generated_by": "scripts/fl601_reference.py update",
        "reference": "Gate B (FL-006) ascent-only regression flight",
        "target": str(TARGET.relative_to(REPO)),
        "vehicle": str(VEHICLE.relative_to(REPO)),
        "simulate_options": {"refly": False, "run_recovery": False},
        "earth_orientation": _earth_orientation_provenance(),
        "requires_commit_tag": COMMIT_TAG,
        "tolerance_audit": json.loads(PROVENANCE.read_text(encoding="utf-8")).get(
            "tolerance_audit", []),
        "git_sha_recording": subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip(),
        "files": files,
    }
    PROVENANCE.write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"reference updated: {len(files)} files under {REF_ROOT}")
    return 0


def _earth_orientation_provenance() -> str:
    _load_repo_package()
    from orbitopt import reference_eop

    return ("bundled IERS only, pinned snapshot earth_orientation.json "
            "(mode pinned-bundled, no download), astropy-iers-data "
            f"{reference_eop.pin_no_download()}")


def cmd_update_tolerances_only(reason: str) -> int:
    """E8-04: amend the comparison policy without re-flying. The committed
    hashes are verified first so this cannot bless an unrelated edited
    reference file, and the change is audited (reason, old -> new, hashes)."""
    if not reason.strip():
        print("update --tolerances-only requires --reason (audited change)",
              file=sys.stderr)
        return 2
    if cmd_verify() != 0:
        return 1
    _load_repo_package()
    from orbitopt import reference_audit

    try:
        old_tol, old_sha = reference_audit.committed_file(
            REPO, str(TOLERANCES.relative_to(REPO)))
    except ValueError:
        old_tol = json.loads(TOLERANCES.read_text(encoding="utf-8"))
        old_sha = _digest(TOLERANCES)
    _ensure_earth_orientation_snapshot()
    TOLERANCES.write_text(json.dumps(DEFAULT_TOLERANCES, indent=2) + "\n", encoding="utf-8")
    prov = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    prov["earth_orientation"] = _earth_orientation_provenance()
    for name in ("tolerances.json", "earth_orientation.json"):
        if (REF_ROOT / name).is_file():
            prov["files"][name] = _digest(REF_ROOT / name)
    entry = reference_audit.make_entry(
        reason=reason,
        changes=reference_audit.diff_tolerances(old_tol, DEFAULT_TOLERANCES),
        previous_tolerances_sha256=old_sha,
        tolerances_sha256=_digest(TOLERANCES))
    reference_audit.append_audit(prov, entry)
    PROVENANCE.write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"comparison policy updated ({len(entry['changes'])} tolerance change(s), "
          "audited); frozen flight output unchanged")
    return 0


# ------------------------------------------------- cross-platform consistency


def cross_platform_observation() -> dict:
    """Re-derive the committed GCRS tracks from the raw recorded OT outputs
    under the pinned Earth orientation and measure the difference against
    the recorded reference (E8-04 cross-platform consistency).

    Runs the *production* adapter conversion (opentsiolkovsky adapter on
    ``opentsiolkovsky/``) on the current platform, so a platform difference
    in the frame bridge shows up as a track difference. No OT binary
    needed -- the raw outputs are committed. Returns the observation dict
    for ``tests/data/e804/cross-platform.json``."""
    _load_repo_package()
    import platform as _platform
    import re

    from orbitopt.flightlab.adapters.opentsiolkovsky import Vehicle, parse_ot_outputs
    from orbitopt.flightlab.contract import load_target, read_run

    manifest = json.loads((REF_RUN_DIR / "manifest.json").read_text(encoding="utf-8"))
    (config_sha,) = re.findall(r"run-config sha256 ([0-9a-f]{64})",
                               "\n".join(manifest["assumptions"]))
    with frozen_reference():
        cand = parse_ot_outputs(
            REF_RUN_DIR / "opentsiolkovsky", load_target(TARGET), Vehicle.load(VEHICLE),
            manifest["run_id"], manifest["git_sha"], config_sha,
            manifest["vehicle_sha256"])
    ref = read_run(REF_RUN_DIR)
    cand_tracks = {t.body_id: t for t in cand.tracks}
    ref_tracks = {t.body_id: t for t in ref.tracks}
    pos = vel = 0.0
    for body in sorted(set(cand_tracks) & set(ref_tracks)):
        ct, rt = cand_tracks[body], ref_tracks[body]
        for i in range(min(len(ct.position_m), len(rt.position_m))):
            pos = max(pos, max(abs(a - b) for a, b in zip(ct.position_m[i], rt.position_m[i])))
            vel = max(vel, max(abs(a - b) for a, b in
                               zip(ct.velocity_m_s[i], rt.velocity_m_s[i])))
    return {"id": f"{_platform.system().lower()}-{_platform.machine()}",
            "astropy": _installed_astropy_series_full(),
            "astropy_iers_data": _pin_earth_orientation(),
            "max_position_m": pos, "max_velocity_m_s": vel}


def _installed_astropy_series_full() -> str:
    import astropy

    return str(astropy.__version__)


def cmd_cross_platform_record() -> int:
    """Print the JSON ledger line for this platform (paste into
    tests/data/e804/cross-platform.json when reviewing the audit)."""
    import sys as _sys

    tol = json.loads(TOLERANCES.read_text(encoding="utf-8")).get("cross_platform")
    obs = cross_platform_observation()
    ok = (obs["max_position_m"] <= tol["position_m_abs"]
          and obs["max_velocity_m_s"] <= tol["velocity_m_s_abs"])
    print(json.dumps(obs, sort_keys=True))
    if not ok:
        print(f"OUTSIDE the cross-platform band {tol}", file=_sys.stderr)
        return 1
    return 0


# --------------------------------------------------------------- tag-check


def cmd_tag_check(base: str | None) -> int:
    """CI guard (G5): if the diff touches tests/data/flightlab/reference/,
    some commit message in the pushed range must carry the update tag."""
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
    touched = [p for p in changed if p.startswith("tests/data/flightlab/reference/")]
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
    run_p = sub.add_parser("run", help="fly the reference mission into --out")
    run_p.add_argument("--out", required=True, type=Path)
    cmp_p = sub.add_parser("compare", help="compare a fresh run dir against the reference")
    cmp_p.add_argument("dir", type=Path)
    sub.add_parser("verify", help="check committed reference files against PROVENANCE.json")
    upd_p = sub.add_parser("update", help=f"regenerate the reference (commit the result with {COMMIT_TAG})")
    upd_p.add_argument("--tolerances-only", action="store_true",
                       help="rewrite tolerances.json/EOP snapshot + audit, without re-flying")
    upd_p.add_argument("--reason", default="",
                       help="required with --tolerances-only; the audit entry's rationale")
    sub.add_parser("cross-platform-record",
                   help="measure this platform's re-derivation diff vs the reference "
                        "(E8-04; prints a JSON ledger line)")
    tag_p = sub.add_parser("tag-check", help="CI: reference diffs need the commit tag")
    tag_p.add_argument("--base", default=None)
    args = parser.parse_args(argv)
    if args.cmd == "run":
        return cmd_run(args.out)
    if args.cmd == "compare":
        return cmd_compare(args.dir)
    if args.cmd == "verify":
        return cmd_verify()
    if args.cmd == "update":
        return cmd_update(tolerances_only=args.tolerances_only, reason=args.reason)
    if args.cmd == "cross-platform-record":
        return cmd_cross_platform_record()
    if args.cmd == "tag-check":
        return cmd_tag_check(args.base)
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
