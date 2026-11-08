"""Launch site -> LEO parking orbit -> TMI / TLI injection, verified end to end.

Plans the first launch window from the site for the requested departure,
the parking orbit, the injection burn (epoch, coast, delta-v) and then
propagates the whole chain numerically with Earth J2, differentially
correcting the burn and writing a verification report.

Run: python examples/13_launch_to_injection.py [--site cape-canaveral]
        [--start 2026-11-08] [--c3 9.0 --rla 340 --dla 20 | --tli --apogee-km 384400 --ra 120 --dec 10]
        [--altitude-km 185] [--vehicle falcon-heavy --payload 4000]
        [--backend numpy|tudat] [--report injection.verification.json]
Needs numpy + scipy (tudatpy only for --backend tudat).
"""
from __future__ import annotations

import argparse

import numpy as np

from orbitopt.injection import InjectionTarget, plan_departure, verify_injection
from orbitopt.launch_constraints import LaunchConstraints
from orbitopt.launch_windows import mjd2000_to_datetime


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--site", default="cape-canaveral")
    ap.add_argument("--start", default="2026-11-08T00:00:00Z")
    ap.add_argument("--c3", type=float, default=9.0)
    ap.add_argument("--rla", type=float, default=340.0)
    ap.add_argument("--dla", type=float, default=20.0)
    ap.add_argument("--tli", action="store_true", help="elliptic (TLI) target instead of C3/RLA/DLA")
    ap.add_argument("--apogee-km", type=float, default=384_400.0)
    ap.add_argument("--ra", type=float, default=120.0)
    ap.add_argument("--dec", type=float, default=10.0)
    ap.add_argument("--altitude-km", type=float, default=185.0)
    ap.add_argument("--min-coast-s", type=float, default=600.0)
    ap.add_argument("--vehicle", default=None)
    ap.add_argument("--payload", type=float, default=4000.0)
    ap.add_argument("--backend", default="numpy", choices=("numpy", "tudat"))
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    if args.tli:
        target = InjectionTarget.from_apogee(args.apogee_km, args.ra, args.dec, name="TLI")
    else:
        target = InjectionTarget.from_c3_rla_dla(args.c3, args.rla, args.dla, name="TMI")
    plan = plan_departure(args.site, target, start=args.start, altitude_km=args.altitude_km,
                          min_coast_s=args.min_coast_s)
    lv, o, b = plan.launch, plan.orbit, plan.burn
    print(f"lift-off   {mjd2000_to_datetime(lv.liftoff_mjd2000):%Y-%m-%d %H:%M:%S} UTC  "
          f"azimuth {lv.azimuth_deg:.2f} deg ({lv.site_id})")
    print(f"parking    {o.altitude_km:.0f} km, i={o.inclination_deg:.3f} deg, RAAN={o.raan_deg:.3f} deg, "
          f"period {o.period_s / 60:.1f} min")
    print(f"{target.name:<10} {mjd2000_to_datetime(b.epoch_mjd2000):%Y-%m-%d %H:%M:%S} UTC  "
          f"coast {b.coast_s / 60:.1f} min, arg. latitude {b.arg_latitude_deg:.2f} deg, "
          f"dv {b.delta_v_mag_m_s:.1f} m/s, C3 {b.c3_km2_s2:.3f} km^2/s^2")

    lc = None
    if args.vehicle:
        lc = LaunchConstraints.from_names(args.vehicle, args.payload, args.site, frame="equatorial")
    v = verify_injection(plan, backend=args.backend, launch_constraints=lc, report_path=args.report)
    print(f"verify     planned error {v.planned_error:.3g} {v.error_units} -> {v.final_error:.3g} "
          f"{v.error_units} after {v.iterations} correction(s); |correction| "
          f"{v.correction_dv_m_s:.2f} m/s; corrected dv {np.linalg.norm(v.delta_v_m_s):.1f} m/s")
    for name, ok in sorted(v.report["status"]["checks"].items()):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("PASSED" if v.passed else "FAILED")


if __name__ == "__main__":
    main()
