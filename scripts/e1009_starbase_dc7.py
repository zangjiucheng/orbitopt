"""E10-09 (task #306): cross-environment CI qualification of the final schemes.

The E10 campaign's final designs are the two sealed E5-08 variants -- DC7
(Kourou, ``starship_class_rtls_dc7``) and Starbase (``starship_class_rtls_starbase``,
E10-08's broader guidance envelope applies to the ascent guidance, the design
records stand). This script is the DEDICATED cross-environment regression and
evidence check for them, plus the tolerance audit record that freezes their
reference outputs:

    python scripts/e1009_starbase_dc7.py verify
    python scripts/e1009_starbase_dc7.py crossenv-check runs/crossenv-e5-08-dc7 --variant dc7
    python scripts/e1009_starbase_dc7.py tag-check --base <sha>

``verify`` re-hashes every sealed tree against its ``PROVENANCE.json`` (the
tolerances file included -- tolerances are frozen reference outputs), checks
the frozen crossenv plans against the digests recorded in
``tests/data/mission/reference/e10-09-ci/tolerance-audit.json``, and keeps the
ORIGINAL E5-08 record classified as the expected-fail negative regression.

``crossenv-check`` grades a finished ``orbitopt crossenv run`` of one variant,
E5-05 discipline: the verdict is re-computed in THIS process from the sealed
exchanges (the report is graded, never trusted), the joint full-mission gate
document must show all five physical checks passed with every evidence source
pinned to an envelope digest that still re-hashes, the qualification block
(with its ``independently_verified`` verdict and reason) must match a re-grade
of the sealed evidence, and the recovery-leg touchdown numerics must sit
inside the frozen tolerances of the sealed tree.

``tag-check`` is the audit gate (G5 convention): a change touching a scheme's
tolerances file or its frozen crossenv plan must touch the tolerance-audit
record in the same diff -- a tolerance change without an audit entry fails CI.

The epic's aspirational two-ignition hot-staging closure gate was classified
out of vehicle capability (E10-01, E10-03, E10-07): these three-ignition
cold-staging records ARE the final schemes, graded exactly as recorded -- the
joint gate's ``qualification.independently_verified`` is False for both (the
mission chain carries no validation-grade independent check; the E8-05 twin
evidence lives in the sealed reports), and that verdict is asserted with its
reason, not hidden.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

REFS = REPO / "tests/data/mission/reference"
AUDIT_FILE = REFS / "e10-09-ci/tolerance-audit.json"
AUDIT_KIND = "orbitopt.e10-09.tolerance-audit"

PHYSICAL_FIVE = ("orbit_insertion", "parking_propagation", "landing",
                 "mass_conservation", "epoch_frame_continuity")

SCHEMES = {
    "dc7": {
        "reference": REFS / "e5-08-new-vehicle-dc7",
        "plan": REPO / "src/orbitopt/data/crossenv_e508_dc7_plan.yaml",
    },
    "starbase": {
        "reference": REFS / "e5-08-new-vehicle-starbase",
        "plan": REPO / "src/orbitopt/data/crossenv_e508_starbase_plan.yaml",
    },
}
# The original E5-08 design (sealed honest gap record) stays in CI as the
# expected-fail negative regression: its joint gate must keep grading a CLEAN
# fail (only the landing check red, the other four physical checks green) --
# scripts/e508_new_vehicle.py crossenv-check enforces that shape.
NEGATIVE_REFERENCE = REFS / "e5-08-new-vehicle"


def _sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _rel(p) -> str:
    return Path(p).resolve().relative_to(REPO).as_posix()


# ---------------------------------------------------------------------- audit

def load_audit() -> dict:
    doc = json.loads(AUDIT_FILE.read_text(encoding="utf-8"))
    if doc.get("kind") != AUDIT_KIND:
        raise SystemExit(f"{_rel(AUDIT_FILE)}: unexpected kind {doc.get('kind')!r}")
    return doc


def _scheme_expectation(audit_scheme: dict, scheme: dict) -> list[str]:
    """One scheme's audit entry vs the tree/plan on disk: digests and the
    recorded tolerance VALUES must match byte-for-byte."""
    fails: list[str] = []
    ref, plan = scheme["reference"], scheme["plan"]
    tol_file = ref / "tolerances.json"
    if not tol_file.is_file():
        return [f"{_rel(tol_file)}: missing"]
    if not plan.is_file():
        return [f"{_rel(plan)}: missing"]
    if audit_scheme.get("reference") != _rel(ref):
        fails.append(f"audit reference {_rel(ref)}: {_rel(audit_scheme.get('reference'))!r}")
    if audit_scheme.get("crossenv_plan") != _rel(plan):
        fails.append(f"audit crossenv_plan {_rel(plan)}: "
                     f"{_rel(audit_scheme.get('crossenv_plan'))!r}")
    if audit_scheme.get("tolerances_sha256") != _sha256(tol_file):
        fails.append(f"{_rel(tol_file)}: live file does not match the audit-recorded "
                     f"digest -- a tolerance change needs a new audit entry")
    if audit_scheme.get("crossenv_plan_sha256") != _sha256(plan):
        fails.append(f"{_rel(plan)}: live plan does not match the audit-recorded digest "
                     "-- the frozen reference outputs moved without an audit entry")
    live = json.loads(tol_file.read_text(encoding="utf-8"))
    if audit_scheme.get("values") != live:
        fails.append(f"{_rel(tol_file)}: audit-recorded values differ from the file")
    trail = audit_scheme.get("audit_trail") or []
    if not trail or not all({"as_of", "change"} <= set(e) for e in trail):
        fails.append("audit_trail: every entry needs as_of + change")
    return fails


def _sealed_tree(ref: Path, *, expect_class: str) -> list[str]:
    """Re-hash one sealed reference tree against its PROVENANCE.json."""
    from orbitopt.crossenv.envelope import file_digest

    fails: list[str] = []
    prov_file = ref / "PROVENANCE.json"
    if not prov_file.is_file():
        return [f"{_rel(prov_file)}: missing"]
    prov = json.loads(prov_file.read_text(encoding="utf-8"))
    if prov.get("reference_class") != expect_class:
        fails.append(f"{_rel(prov_file)}: reference_class "
                     f"{prov.get('reference_class')!r} != {expect_class!r}")
    files = prov.get("files") or {}
    if "tolerances.json" not in files:
        fails.append("PROVENANCE does not seal tolerances.json: the tolerance file is "
                     "not a frozen reference output")
    for rel, digest in files.items():
        f = ref / rel
        if not f.is_file():
            fails.append(f"{rel}: sealed file missing from the tree")
        elif file_digest(f) != "sha256:" + digest:
            fails.append(f"{rel}: sealed digest does not re-hash (tamper or intended "
                         "change without re-seal?)")
    frozen = prov.get("frozen_inputs") or {}
    for key, fname in (("vehicle_sha256", "vehicle"), ("scenario_sha256", "scenario")):
        src = prov.get(fname)
        if key in frozen and src and frozen[key] != _sha256(REPO / src):
            fails.append(f"{_rel(ref)}: frozen {key} != the on-disk {src}")
    return fails


def cmd_verify(variants=None) -> int:
    fails: list[str] = []
    audit = load_audit()
    for name in (variants or sorted(SCHEMES)):
        scheme = SCHEMES[name]
        fails += _sealed_tree(scheme["reference"], expect_class="new-vehicle-qualification")
        fails += _scheme_expectation((audit.get("schemes") or {}).get(name) or {}, scheme)
    # the negative regression stays classified as the gap it is
    fails += _sealed_tree(NEGATIVE_REFERENCE, expect_class="new-vehicle-qualification-gap")
    rec = (audit.get("negative_regression") or {})
    script = NEGATIVE_REFERENCE / "recovery-script.json"
    if json.loads(script.read_text(encoding="utf-8")).get("status") != "not_closed":
        fails.append(f"{_rel(script)}: the negative regression no longer reads not_closed "
                     "-- it is no longer an expected-fail and must not be re-badged")
    if rec.get("reference") != _rel(NEGATIVE_REFERENCE):
        fails.append("audit negative_regression.reference does not point at the sealed gap")
    if fails:
        print(f"E10-09 verify FAILED ({len(fails)} problem(s)):", file=sys.stderr)
        for line in fails[:40]:
            print(f"  {line}", file=sys.stderr)
        return 1
    print("E10-09 verify OK: sealed trees re-hash, tolerances and frozen crossenv plans "
          "match the audit record, the original E5-08 gap stands as the negative regression")
    return 0


# --------------------------------------------------------------- crossenv-check

def grade_joint_evidence(out: Path, joined: dict) -> list[str]:
    """Re-grade the joint gate from the sealed exchanges and compare -- the
    E5-05 discipline, plus the per-check evidence-source pinning (E5-06)."""
    from orbitopt.crossenv.envelope import CrossEnvError, envelope_digest, read_envelope
    from orbitopt.crossenv.qualification import JOINT_KIND, joint_qualification

    fails: list[str] = []
    joint = joined.get("joint_qualification")
    if not isinstance(joint, dict) or joint.get("kind") != JOINT_KIND:
        return ["joined report carries no joint qualification document (E5-06)"]
    for entry in joined.get("stages", []):
        ex = out / str(entry["id"]) / "exchange"
        try:
            read_envelope(ex)
            now = envelope_digest(ex)
        except CrossEnvError as exc:
            fails.append(f"stage {entry['id']}: exchange did not re-verify: {exc}")
            continue
        if entry.get("exchange_digest") and entry["exchange_digest"] != now:
            fails.append(f"stage {entry['id']}: report pins envelope {entry['exchange_digest']}, "
                         f"the envelope on disk hashes {now}")
    recomputed = joint_qualification(scenario_path=joined.get("scenario"),
                                     stages=joined.get("stages", []),
                                     plan_full_mission=bool(joint.get("plan_full_mission")))
    if joint.get("full_mission_declared") is not True:
        fails.append("the run does not declare full_mission")
    if recomputed["verdict"] != joint.get("verdict"):
        fails.append(f"joint verdict re-graded: the report says {joint.get('verdict')!r}, "
                     f"the sealed evidence grades {recomputed['verdict']!r}")
    if joint.get("verdict") != "passed":
        fails.append(f"joint full-mission verdict: {joint.get('verdict')} "
                     f"(problems: {joint.get('problems')})")
    if joint.get("problems"):
        fails.append(f"joint gate problems: {joint['problems']}")
    if joint.get("reason_codes"):
        fails.append(f"joint gate reason codes: {joint['reason_codes']}")
    phys = joint.get("physical") or {}
    if recomputed["physical"]["verdict"] != joint.get("physical_mission_verdict"):
        fails.append("physical_mission_verdict does not match a re-grade of the sealed evidence")
    if joint.get("physical_mission_verdict") != "passed" or phys.get("verdict") != "passed":
        fails.append(f"joint physical gate: {joint.get('physical_mission_verdict')} "
                     f"{phys.get('reason_codes')}")
    by_name = {c.get("check"): c for c in phys.get("checks") or []}
    for name in PHYSICAL_FIVE:
        doc = by_name.get(name) or {}
        if doc.get("verdict") != "passed":
            fails.append(f"physical check {name}: "
                         f"{doc.get('verdict') or 'absent from the joint gate document'}")
        srcs = [s for s in doc.get("sources") or [] if s.get("exchange_digest")]
        if not srcs:
            fails.append(f"physical check {name}: no evidence source pinned to an "
                         "envelope digest")
        for src in srcs:
            ex = out / str(src.get("stage")) / "exchange"
            try:
                now = envelope_digest(ex)
            except CrossEnvError as exc:
                fails.append(f"physical check {name}: source envelope {ex} unreadable: {exc}")
                continue
            if now != src["exchange_digest"]:
                fails.append(f"physical check {name}: evidence pinned to "
                             f"{src['exchange_digest']} but stage {src['stage']!r} envelope "
                             f"now hashes {now}")
    if (joint.get("coverage") or {}).get("complete") is not True:
        fails.append("joint required-phase coverage incomplete: "
                     f"{(joint.get('coverage') or {}).get('summary')}")
    if recomputed["qualification"] != joint.get("qualification"):
        fails.append("joint qualification block does not match a re-grade of the sealed "
                     "evidence (report tampered or evidence moved?)")
    qual = joint.get("qualification") or {}
    if not isinstance(qual.get("independently_verified"), bool):
        fails.append("joint qualification.independently_verified is not a verdict")
    elif not qual["independently_verified"]:
        # Recorded honestly: with a passed joint gate the ONLY admissible blocker is the
        # absent validation-grade independent check in the mission chain itself (the
        # E8-05 twin evidence lives in the sealed reports, graded by test_e508_*.py).
        expect = ("no passing validation-grade independent evidence "
                  f"(status.independent_validation: {qual.get('independent_validation')})")
        if qual.get("reason") != expect:
            fails.append(f"independently_verified is False for an unexpected reason: "
                         f"{qual.get('reason')!r} (expected exactly {expect!r} -- any "
                         "other blocker means the joint gate did not really pass)")
    return fails


def grade_sealed_continuity(out: Path, scheme: dict) -> list[str]:
    """The run branched from the SEALED reference: the same Earth-orientation
    snapshot both legs ran under, the recovery leg driven from the canonical
    separation handoff the sealed mission records, and the flown touchdown
    inside the frozen tolerances."""
    from orbitopt.flightlab.contract import read_run

    ref = scheme["reference"]
    fails: list[str] = []
    tol = json.loads((ref / "tolerances.json").read_text(encoding="utf-8"))
    eop = "sha256:" + _sha256(ref / "earth_orientation.json")
    for name in ("ascent", "recovery"):
        report_file = out / name / "stage-report.json"
        if not report_file.is_file():
            fails.append(f"stage {name}: no stage-report.json")
            continue
        st = json.loads(report_file.read_text(encoding="utf-8"))
        if st.get("earth_orientation", {}).get("snapshot_sha256") != eop:
            fails.append(f"stage {name}: Earth-orientation snapshot is not the sealed one")
    rec_file = out / "recovery" / "stage-report.json"
    if rec_file.is_file():
        rec = json.loads(rec_file.read_text(encoding="utf-8"))
        sealed = json.loads((ref / "mission" / "mission.verification.json")
                            .read_text(encoding="utf-8"))
        want = (sealed.get("inputs", {}).get("handoffs", {})
                .get("coupled_branches", {}) or {}).get("handoff_id")
        if rec.get("driven_from_separation") is not True:
            fails.append("recovery stage was not driven from the separation handoff")
        if rec.get("separation_handoff_id") != want:
            fails.append(f"recovery stage separation_handoff_id "
                         f"{rec.get('separation_handoff_id')!r} != the sealed "
                         f"{want!r} -- the two environments did not branch from the "
                         "one recorded state")

    cand_dir = out / "recovery" / "exchange" / "recovery" / "run"
    if not cand_dir.is_dir():
        return fails + [f"{cand_dir}: the recovery leg sealed no flight run"]
    cand, want = read_run(cand_dir), read_run(ref / "mission" / "recovery" / "run")
    if cand.run_id != want.run_id:
        fails.append(f"recovery run_id {cand.run_id!r} != the sealed {want.run_id!r}")
    for field in ("target_sha256", "vehicle_sha256"):
        if getattr(cand, field) != getattr(want, field):
            fails.append(f"recovery run {field}: differs from the sealed run's input pin")
    if dict(cand.engines) != dict(want.engines):
        fails.append("recovery run engines differ -- the gate must run on the pinned "
                     "solver stack")
    gev = {e.kind: dict(e.data) for e in want.events}
    cev = {e.kind: dict(e.data) for e in cand.events}
    td, ct = gev.get("touchdown") or {}, cev.get("touchdown") or {}
    if not td or not ct:
        fails.append("no touchdown event in the sealed and/or candidate recovery run")
    else:
        for key, band in (("position_error_m", tol["touchdown_pos_m"]),
                          ("vertical_speed_m_s", tol["touchdown_vert_m_s"]),
                          ("lateral_speed_m_s", tol["touchdown_lateral_m_s"])):
            if abs(float(ct[key]) - float(td[key])) > band:
                fails.append(f"touchdown {key}: {ct[key]} vs sealed {td[key]} "
                             f"(band {band})")
    imw, imc = gev.get("impact") or {}, cev.get("impact") or {}
    if not imw or not imc:
        fails.append("no impact event in the sealed and/or candidate recovery run")
    elif abs(float(imc["mass_kg"]) - float(imw["mass_kg"])) > tol["touchdown_propellant_kg"]:
        fails.append(f"impact mass: {imc['mass_kg']} vs sealed {imw['mass_kg']} "
                     f"(band {tol['touchdown_propellant_kg']})")
    return fails


def cmd_crossenv_check(out: Path, variant: str) -> int:
    scheme = SCHEMES[variant]
    joined_file = out / "crossenv.mission-report.json"
    if not joined_file.is_file():
        print(f"{joined_file}: missing (no completed cross-env run?)", file=sys.stderr)
        return 1
    joined = json.loads(joined_file.read_text(encoding="utf-8"))
    fails: list[str] = []
    if joined.get("kind") != "orbitopt.crossenv.mission-report":
        fails.append(f"joined report kind: {joined.get('kind')!r}")
    overall = joined.get("overall") or {}
    if overall.get("exchanges_verified") is not True:
        fails.append("overall.exchanges_verified is not true")
    if overall.get("mission_ok") is not True:
        fails.append(f"overall.mission_ok is not true: {joined.get('stages')}")
    for entry in joined.get("stages", []):
        if entry.get("status") != "completed" or entry.get("mission_ok") is not True:
            fails.append(f"stage {entry.get('id')}: {entry.get('status')} "
                         f"mission_ok={entry.get('mission_ok')}")
    fails += grade_joint_evidence(out, joined)
    fails += grade_sealed_continuity(out, scheme)
    joint = joined.get("joint_qualification") or {}
    if fails:
        print(f"E10-09 CROSS-ENV CHECK FAILED ({variant}): {len(fails)} problem(s) in {out}",
              file=sys.stderr)
        for line in fails[:40]:
            print(f"  {line}", file=sys.stderr)
        return 1
    print(f"E10-09 cross-environment check OK ({variant}): {out} -- joint gate "
          f"{joint.get('verdict')} with physical_mission_verdict "
          f"{joint.get('physical_mission_verdict')} over five evidenced checks "
          f"(independently_verified={joint.get('qualification', {}).get('independently_verified')}), "
          "sources re-hashed, driven from the sealed separation handoff, touchdown "
          "inside the frozen tolerances")
    return 0


# -------------------------------------------------------------------- G5 audit

def audit_required(changed: list[str], audit_rel: str) -> list[str]:
    """The files of a frozen scheme that may only change together with the
    audit record; returns the offending files (empty = no audit violation
    possible, no watch hit at all)."""
    watched: list[str] = []
    for scheme in SCHEMES.values():
        watched.append(_rel(scheme["reference"] / "tolerances.json"))
        watched.append(_rel(scheme["plan"]))
    touched = {p for p in changed}
    hit = sorted(set(watched) & touched)
    if hit and audit_rel not in touched:
        return hit
    return []


def cmd_tag_check(base: str | None) -> int:
    """CI guard (G5): a tolerance/plan change of a frozen scheme requires an
    audit entry in the SAME change."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(REPO), *args],
                              capture_output=True, text=True, check=True).stdout

    if base is None or set(base) == {"0"}:
        print("tag-check skipped: no usable diff base (workflow_dispatch or force-push); "
              "the verify step still runs")
        return 0
    try:
        changed = git("diff", "--name-only", f"{base}..HEAD").splitlines()
    except subprocess.CalledProcessError:
        print(f"tag-check skipped: base {base} not present in this clone")
        return 0
    hit = audit_required(changed, _rel(AUDIT_FILE))
    if not hit:
        print("frozen tolerance/plan files untouched by this change")
        return 0
    print(f"{len(hit)} frozen scheme file(s) changed without touching the tolerance audit "
          f"record ({_rel(AUDIT_FILE)}):", file=sys.stderr)
    for p in hit[:10]:
        print(f"  {p}", file=sys.stderr)
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="sealed trees + audit record + negative regression")
    v.add_argument("--variant", choices=sorted(SCHEMES))
    c = sub.add_parser("crossenv-check", help="grade a finished cross-env run of one variant")
    c.add_argument("out_dir", type=Path)
    c.add_argument("--variant", required=True, choices=sorted(SCHEMES))
    t = sub.add_parser("tag-check", help="tolerance/plan changes require an audit entry")
    t.add_argument("--base", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "verify":
        return cmd_verify([args.variant] if args.variant else None)
    if args.cmd == "crossenv-check":
        return cmd_crossenv_check(args.out_dir, args.variant)
    return cmd_tag_check(args.base)


if __name__ == "__main__":
    raise SystemExit(main())
