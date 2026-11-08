"""Mars 2020 (Perseverance) Earth-departure replay -- every layer of the
framework, checked against a real flown mission.

Real reference: lift-off 2020-07-30 11:50 UTC from SLC-41 (Cape Canaveral),
Atlas V 541, Mars landing 2021-02-18 (~203 d time of flight). The vehicle
database has no Atlas V 541; Atlas V 551 stands in for it.

  1. porkchop of the 2020 opportunity (batched Lambert grid)
  2. multi-basin multistart search for the minimum-dv transfer
  3. the same search under launch constraints (vehicle C3 + site DLA), and
     which vehicles could carry the payload at all
  4. that day's launch windows from KSC for the real departure asymptote
  5. LEO parking orbit -> TMI burn, propagated with J2, Newton-corrected,
     verification report written

The full tudatpy cruise (TCM + Mars orbit insertion) is the config-driven
`orbitopt run examples/mars2020_replay.yaml`.

Run: python examples/14_mars2020_replay.py [--out out/mars2020]
Needs the compute env (pykep, pygmo, scipy, matplotlib).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pykep as pk

from orbitopt.bodies import mjd2000_from_date, planet
from orbitopt.injection import InjectionTarget, plan_departure, verify_injection
from orbitopt.lambert.gpu_batch import solve_lambert_batch
from orbitopt.launch_constraints import LaunchConstraints
from orbitopt.launch_vehicles import OutOfPerformanceRange, get_launch_vehicle
from orbitopt.launch_windows import (asymptote_from_v_inf, asymptote_launch_windows,
                                     format_windows, mjd2000_to_datetime)
from orbitopt.optimize.multistart import find_diverse_basins, multistart_optimize
from orbitopt.problems.transfer_2body import Transfer2BodyProblem
from orbitopt.viz.porkchop import compute_porkchop

SITE, VEHICLE, PAYLOAD_KG = "ksc", "atlas-v-551", 4037.0
REAL_LIFTOFF = datetime(2020, 7, 30, 11, 50, tzinfo=timezone.utc)
REAL_TOF_DAYS = 203.4
T0_RANGE = (mjd2000_from_date(2020, 6, 15), mjd2000_from_date(2020, 9, 15))
TOF_RANGE = (150.0, 350.0)
FLEET = ("falcon-9", "atlas-v-551", "long-march-5", "vulcan-vc6", "falcon-heavy")

earth, mars = planet("earth"), planet("mars")


def transfer(t0, tof, lc=None):
    """Lambert transfer summary for departure t0 [MJD2000] and tof [days]."""
    r1, v1p = (np.asarray(a) for a in earth.eph(pk.epoch(t0)))
    r2, v2p = (np.asarray(a) for a in mars.eph(pk.epoch(t0 + tof)))
    res = solve_lambert_batch(r1[None], r2[None], np.array([tof * pk.DAY2SEC]), pk.MU_SUN, use_gpu=False)
    v_inf = res.v1[0] - v1p
    rla, dla, _ = asymptote_from_v_inf(v_inf / 1000.0, frame="ecliptic")
    out = {"departure": f"{mjd2000_to_datetime(t0):%Y-%m-%d}", "tof_days": round(tof, 1),
           "c3_km2_s2": round(float(v_inf @ v_inf) / 1e6, 3), "rla_deg": round(rla, 2), "dla_deg": round(dla, 2)}
    if lc is not None:
        out["feasible"] = bool(lc.evaluate(v_inf[None]).feasible[0])
    return out


def search(basins, lc=None, seed=7):
    factory = lambda a, b: Transfer2BodyProblem(earth, mars, a, b, launch_constraints=lc)  # noqa: E731
    return multistart_optimize(factory, basins, full_bounds=(T0_RANGE, TOF_RANGE),
                               pop_size=64, generations=80, seed=seed)


def max_payload_for_c3(vehicle_id, c3, hi):
    """Largest payload [kg] (bisection, up to ``hi``) the vehicle lifts to ``c3``."""
    veh, lo = get_launch_vehicle(vehicle_id), 100.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        try:
            ok = veh.max_c3(mid) >= c3
        except OutOfPerformanceRange:
            ok = False
        lo, hi = (mid, hi) if ok else (lo, mid)
    return round(lo)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="out/mars2020")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}

    print("1. porkchop")
    grid = compute_porkchop(earth, mars, T0_RANGE, TOF_RANGE, n_t0=180, n_tof=180, use_gpu=False)
    c3 = np.where(grid.converged, grid.c3_departure, np.inf)
    i, j = np.unravel_index(np.argmin(c3), c3.shape)
    summary["porkchop_min_c3"] = transfer(grid.t0_grid[j], grid.tof_grid[i])
    print("   window-minimum C3:", summary["porkchop_min_c3"])

    print("2. multistart search")
    dv = np.where(grid.converged, grid.total_dv, np.inf)
    basins = find_diverse_basins(dv, grid.t0_grid, grid.tof_grid, max_basins=6, min_separation=(20.0, 40.0))
    report = search(basins)
    best = report.best
    summary["optimum"] = dict(transfer(best.x, best.y), total_dv_m_s=round(best.value, 1))
    print(report.summary(unit="m/s"))

    print(f"3. launch constraints ({VEHICLE}, {PAYLOAD_KG:.0f} kg, {SITE})")
    lc = LaunchConstraints.from_names(VEHICLE, PAYLOAD_KG, SITE)
    constrained = search(basins, lc).best
    summary["constrained_optimum"] = transfer(constrained.x, constrained.y, lc)
    need_c3 = summary["optimum"]["c3_km2_s2"]
    fleet = []
    for vid in FLEET:
        max_c3 = LaunchConstraints.from_names(vid, PAYLOAD_KG, SITE).describe()["max_c3_km2_s2"]
        row = {"vehicle": vid, "max_c3_at_payload": round(max_c3, 2), "can_fly": max_c3 >= need_c3}
        if not row["can_fly"]:
            row["max_payload_kg_at_optimum_c3"] = max_payload_for_c3(vid, need_c3, PAYLOAD_KG)
        fleet.append(row)
        print("  ", row)
    summary["fleet"] = fleet

    print("4. launch windows on the real departure day")
    real = transfer(mjd2000_from_date(2020, 7, 30), REAL_TOF_DAYS, lc)
    windows = asymptote_launch_windows(SITE, real["rla_deg"], real["dla_deg"],
                                       mjd2000_from_date(2020, 7, 30), 1.0, None, 30.0)
    print(format_windows(windows))
    nearest = min(windows, key=lambda w: abs((w.optimal_utc - REAL_LIFTOFF).total_seconds()))
    summary["launch_window"] = {"asymptote": real, "predicted_optimum_utc": f"{nearest.optimal_utc:%H:%M:%S}",
                                "minus_real_min": round((nearest.optimal_utc - REAL_LIFTOFF).total_seconds() / 60, 1)}
    print("   predicted optimum vs real lift-off:", summary["launch_window"])

    print("5. parking orbit -> TMI, verified")
    target = InjectionTarget.from_c3_rla_dla(real["c3_km2_s2"], real["rla_deg"], real["dla_deg"], name="TMI")
    plan = plan_departure(SITE, target, liftoff=nearest.optimal_mjd2000, altitude_km=185.0, min_coast_s=600.0)
    v = verify_injection(plan, launch_constraints=LaunchConstraints.from_names(VEHICLE, PAYLOAD_KG, SITE, frame="equatorial"),
                         report_path=str(out / "injection.verification.json"))
    summary["injection"] = {
        "parking_inclination_deg": round(plan.orbit.inclination_deg, 2),
        "coast_min": round(plan.burn.coast_s / 60, 1), "tmi_dv_m_s": round(plan.burn.delta_v_mag_m_s, 1),
        "residual_before": round(v.planned_error, 2), "residual_after": round(v.final_error, 3),
        "residual_units": v.error_units, "checks_passed": f"{sum(v.report['status']['checks'].values())}"
                                                           f"/{len(v.report['status']['checks'])}",
        "passed": v.passed,
    }
    print("  ", summary["injection"])

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out / 'summary.json'} and {out / 'injection.verification.json'}")


if __name__ == "__main__":
    main()
