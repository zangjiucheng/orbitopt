"""Flight Lab ascent -> verified parking orbit -> TMI plan (FL-203 handoff).

End-to-end: a RAAN-constrained flight target gets its in-plane liftoff epoch
from the launch-window model, the pygmo ascent optimizer (FL-201/202, over
OpenTsiolkovsky) flies to it, the achieved RAAN is graded in GCRS against
``checks.RAAN_TOLERANCE_DEG`` with a liftoff-epoch sensitivity report, the
verified insertion state is exported as the parking orbit the patched-conic
injection flows consume, and a trans-Mars injection burn is planned from it.

Fidelity split (the two are deliberately not mixed):
  * ascent (liftoff -> SECO, steps 1-4) is 3-DOF point-mass flight with drag
    and the US76 atmosphere, integrated by OpenTsiolkovsky and graded by the
    Flight Lab Gate A/B checks -- NOT patched conic;
  * everything after insertion (step 5: parking-orbit J2 precession and the
    TMI burn) is patched conic -- the regular ``orbitopt.injection``
    machinery, exactly as fed by a launch window in
    ``examples/13_launch_to_injection.py``.

Run: python examples/16_flightlab_ascent_to_parking_orbit.py
        [--raan-deg 125.7425] [--window-days 14] [--sensitivity-s 60]
        [--generations 20 --population 32 --workers 4] [--objective payload]
Needs the OpenTsiolkovsky binary (python scripts/fetch_opentsiolkovsky.py);
pygmo + scipy for the optimizer (without them the vehicle's built-in program
is graded instead and the handoff gate reports honestly). The packaged
scenario's RTLS floor is a fixture the demonstrator booster cannot meet, so
the demo flies the contract's expendable configuration (recovery.kind=none)
-- see step 4; no tolerance is relaxed for it.
"""
from __future__ import annotations

import argparse
import dataclasses
import shutil
import sys
import tempfile
from pathlib import Path

from orbitopt.flightlab import checks
from orbitopt.flightlab.contract import load_target, read_run
from orbitopt.flightlab.parking_export import parking_orbit_from_insertion
from orbitopt.flightlab.vehicle_source import data_root
from orbitopt.flightlab.window_targeting import (DEFAULT_WINDOW_DAYS,
                                                 calibrate_epoch,
                                                 derive_in_plane_liftoff,
                                                 raan_report,
                                                 window_sensitivity)

SCENARIO = data_root() / "scenarios" / "leo_two_stage.yaml"
VEHICLE = data_root() / "vehicles" / "synthetic_two_stage.yaml"


def _read_program(csv_path: Path) -> list:
    from orbitopt.flightlab.adapters.opentsiolkovsky import AttitudePoint

    rows = csv_path.read_text(encoding="utf-8").splitlines()[1:]
    return [AttitudePoint(*[float(t) for t in line.split(",")]) for line in rows]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raan-deg", type=float, default=125.7425,
                    help="target RAAN at the scenario epoch (default: the "
                         "plane the packaged synthetic scenario flies)")
    ap.add_argument("--window-days", type=float, default=DEFAULT_WINDOW_DAYS)
    ap.add_argument("--sensitivity-s", type=float, default=60.0,
                    help="report RAAN at liftoff +/- this many seconds (0 = skip)")
    ap.add_argument("--generations", type=float, default=20)
    ap.add_argument("--population", type=float, default=32)
    ap.add_argument("--workers", type=float, default=4)
    ap.add_argument("--objective", default="payload",
                    choices=("payload", "reserve", "booster_reserve"))
    args = ap.parse_args()

    from orbitopt.flightlab.adapters.opentsiolkovsky import (OTBinaryMissingError,
                                                             find_binary)
    try:
        find_binary()
    except OTBinaryMissingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    base = load_target(SCENARIO)
    # The packaged scenario's 1000 kg RTLS floor is a design-assumption
    # fixture the demonstrator booster cannot meet (FL-102 measures ~0 kg
    # of first-stage propellant left at separation -- its burn profile
    # depletes it), and stage recovery is outside the ascent->parking
    # handoff: this demo flies the expendable configuration
    # (recovery.kind = none, exactly what target_export emits for a
    # recovery-less mission). Every constraint the handoff gate depends on
    # -- orbit band, loads, upper-stage reserve, and the RAAN band at
    # checks.RAAN_TOLERANCE_DEG, untouched -- stays declared and graded.
    target = dataclasses.replace(
        base, target_orbit=dataclasses.replace(base.target_orbit,
                                               raan_deg=args.raan_deg),
        recovery=dataclasses.replace(base.recovery, kind="none"),
        assumptions=tuple(base.assumptions) + (
            "FL-203 handoff demo: flown expendable (recovery.kind=none); the "
            "scenario's RTLS propellant floor is a fixture assumption outside "
            "the ascent->parking handoff gate",
        ))

    print("== 1. launch window (epoch derivation; feeds the 3-DOF gates) ====")
    der = derive_in_plane_liftoff(target, window_days=args.window_days)
    w = der.window
    print(f"anchor epoch     {der.anchor_utc}, requested RAAN "
          f"{der.requested_raan_deg:.4f} deg")
    target = der.target
    print(f"in-plane pass    {target.mission.epoch_utc} at azimuth "
          f"{w.azimuth_optimal_deg:.2f} deg ({w.direction} node crossing) -- "
          "J2-regressed plane at the target's mean altitude")
    print(f"graded RAAN      {der.epoch_raan_deg:.4f} deg at that epoch "
          "(plane has regressed since the anchor)")
    der, cal = calibrate_epoch(der, VEHICLE)
    target = der.target
    print(f"calibrated epoch {target.mission.epoch_utc} ({cal['correction_s']:+.1f} s "
          "shift): two probe flights measure the achieved-RAAN-vs-epoch slope "
          f"({cal['measured_slope_deg_s']:.5f} deg/s ~ sidereal) and move the "
          "liftoff to where THIS ascent's plane meets the target plane -- "
          "the geometric pass alone leaves ~2 deg of ascent tilt")

    print("\n== 2. ascent optimisation (3-DOF point-mass, OpenTsiolkovsky) ==")
    work = Path(tempfile.mkdtemp(prefix="fl203-example-"))
    out = work / "run"
    have_pygmo = True
    try:
        import pygmo  # noqa: F401
        import scipy  # noqa: F401
    except ImportError:
        have_pygmo = False
    run = None
    result = None
    prog = None
    if have_pygmo:
        from orbitopt.flightlab.optimize.run import optimize_ascent

        offsets = ((-args.sensitivity_s, args.sensitivity_s)
                   if args.sensitivity_s > 0 else None)
        result = optimize_ascent(
            target, VEHICLE, out, objective=args.objective,
            generations=int(args.generations), population=int(args.population),
            workers=int(args.workers) or None, seed=16, refly=False,
            derive_window=False,  # epoch derived and calibrated in step 1
            sensitivity_offsets_s=offsets, target_path=str(SCENARIO))
        run = read_run(out)
        prog = _read_program(out / "attitude_program.csv")
        print(f"status {result['status']}  ({result['algorithm']['global']}, "
              f"{result['evaluations']} flights evaluated)")
        for code in result["reason_codes"]:
            print(f"  reason: {code}")
    else:
        print("pygmo/scipy unavailable -- grading the vehicle's built-in "
              "program (no optimisation, the RAAN equality cannot be enforced)")
        from orbitopt.flightlab.adapters.opentsiolkovsky import run_ascent

        run = run_ascent(target, VEHICLE, None, out / "flight")
        verdict = checks.evaluate(run, target)
        print("mission constraints passed:", verdict.mission_constraints_passed)

    print("\n== 3. RAAN verification (GCRS elements of the insertion state) ==")
    report = (result["raan"] if result and result.get("raan")
              else raan_report(run, target))
    print(f"achieved RAAN    {report['achieved_raan_deg']:.4f} deg "
          f"(target {report['target_raan_deg']:.4f} deg, "
          f"error {report['error_deg']:+.4f} deg, tolerance "
          f"{report['tolerance_deg']:g} deg): "
          f"{'PASS' if report['passed'] else 'FAIL'}")
    print(f"OT frame RAAN    {report['ot_launch_aligned_raan_deg']:.4f} deg "
          "-- OpenTsiolkovsky's launch-aligned pseudo-inertial frame; a "
          "diagnostic only, the grade is always the GCRS number above")
    rows = (report.get("window_sensitivity")
            or (window_sensitivity(target, VEHICLE, prog,
                                   (-args.sensitivity_s, args.sensitivity_s))
                if args.sensitivity_s > 0 else []))
    for row in rows:
        print(f"liftoff {row['liftoff_offset_s']:+6.0f} s "
              f"({row['liftoff_epoch_utc']}): RAAN "
              f"{row['achieved_raan_deg']:.4f} deg, error "
              f"{row['error_deg']:+.4f} deg "
              f"{'PASS' if row['passed_within_tolerance'] else 'FAIL'}")
    print("expected slope: one sidereal rotation -- RAAN moves "
          "~0.25 deg per minute of liftoff shift")

    print("\n== 4. handoff: GCRS insertion state -> ParkingOrbit ============")
    try:
        handoff = parking_orbit_from_insertion(run)
    except ValueError as exc:
        print(f"GATE CLOSED: {exc}")
        print("the ascent did not pass Gate B; a failed ascent is never "
              "laundered into a parking orbit -- fix the flight, not the gate")
        shutil.rmtree(work, ignore_errors=True)
        return 1
    o = handoff.orbit
    path = handoff.save(work / "ascent_handoff.json")
    t_ins = checks.insertion_state(run)["t_s"]
    print(f"parking orbit    {o.altitude_km:.1f} km circular (mean of the "
          f"achieved rp/ra; true ecc {handoff.elements['eccentricity']:.4f} "
          f"kept in ascent_handoff.json), i={o.inclination_deg:.3f} deg, "
          f"RAAN {o.raan_deg:.3f} deg")
    print(f"epoch            {handoff.epoch_utc} (SECO at T+{t_ins:.0f} s)  "
          f"-> {path}")

    print("\n== 5. patched conic from here on: TMI from the exported orbit ==")
    from orbitopt.injection import InjectionTarget, plan_injection

    mt = InjectionTarget.from_c3_rla_dla(9.0, 340.0, 20.0, name="TMI")
    burn = plan_injection(o, mt, min_coast_s=600.0)
    print(f"TMI burn         {burn.epoch_mjd2000:.6f} MJD2000, coast "
          f"{burn.coast_s / 60:.1f} min, arg. latitude "
          f"{burn.arg_latitude_deg:.2f} deg, dv {burn.delta_v_mag_m_s:.1f} m/s, "
          f"C3 {burn.c3_km2_s2:.3f} km^2/s^2")
    print("from this ParkingOrbit the full numeric verification chain of")
    print("examples/13 (verify_injection / launch_injection) applies unchanged.")

    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
