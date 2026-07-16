"""Optimize a GOES-style GTO -> GEO orbit-raising campaign: from a
geostationary transfer orbit the launch vehicle delivers, find the multi-burn
liquid-apogee-engine schedule that circularizes at GEO and removes the residual
inclination for minimum total delta-v, subject to a finite-burn (per-apogee-
pass) cap. See docs/goes_gto_geo_mission_plan.md for the mission context.

The finite-burn cap is what makes this a *multi*-burn problem: under ideal
impulse a single combined apogee burn is optimal, so the number of burns is set
by how much delta-v the engine can deliver per apogee pass, not by any delta-v
saving. Sweeping n_burns shows the minimum feasible count -- ~5 for a GOES-like
injection, matching the flown campaigns.

Run: python examples/10_gto_geo_orbit_raising.py
"""
from __future__ import annotations

import numpy as np

from orbitopt.optimize.runner import run_optimization
from orbitopt.problems.geo_raising import (
    GeoRaisingProblem,
    single_impulse_geo_insertion_ms,
)

# GOES-16-like injection delivered by Atlas V 541 (rounded published elements).
# gto_apogee_km only feeds single_impulse_geo_insertion_ms's closed form below --
# GeoRaisingProblem always models the shared burn apogee at r_geo (see the
# "known floor" note in problems/geo_raising.py), so it isn't a constructor
# parameter of the problem itself.
GOES_GTO = dict(gto_perigee_km=8108.0, gto_apogee_km=35286.0, gto_inclination_deg=10.6)
GEO_RAISING_KWARGS = {k: v for k, v in GOES_GTO.items() if k != "gto_apogee_km"}
APOGEE_DWELL_S = 41 * 60  # GOES-16 held each LAE burn to < 41 min


def main():
    r_geo = GeoRaisingProblem(n_burns=2, **GEO_RAISING_KWARGS).r_geo
    cap = GeoRaisingProblem(n_burns=2, **GEO_RAISING_KWARGS).max_dv_per_pass_ms(APOGEE_DWELL_S)
    capacity = GeoRaisingProblem(n_burns=2, **GEO_RAISING_KWARGS).tsiolkovsky_capacity_ms()
    single = single_impulse_geo_insertion_ms(**GOES_GTO)

    print("GOES GTO -> GEO orbit raising")
    print(f"  injection      : {GOES_GTO['gto_perigee_km']:.0f} x {GOES_GTO['gto_apogee_km']:.0f} km, "
          f"i = {GOES_GTO['gto_inclination_deg']:.1f} deg")
    print(f"  GEO radius     : {r_geo:.1f} km (v_geo = {np.sqrt(398600.4418 / r_geo):.4f} km/s)")
    print(f"  1-impulse dv   : {single:.1f} m/s (ideal, ignores finite burn)")
    print(f"  per-pass cap   : {cap:.1f} m/s (458 N / 5192 kg x {APOGEE_DWELL_S / 60:.0f} min dwell)")
    print(f"  propellant cap : {capacity:.1f} m/s (Tsiolkovsky, 5192/2857 kg, Isp 324 s)")
    print(f"  -> expect min feasible n_burns ~ ceil({single:.0f}/{cap:.0f}) = {int(np.ceil(single / cap))}\n")

    print(f"{'n_burns':>7} | {'total dv':>9} | {'max burn':>9} | feasible")
    print("-" * 44)
    champion = None
    for n_burns in range(2, 7):
        problem = GeoRaisingProblem(n_burns=n_burns, max_dv_per_burn_ms=cap, **GEO_RAISING_KWARGS)
        x, _, _ = run_optimization(problem, pop_size=160, generations=150, seed=1, verbose=False)
        d = problem.decode(x)
        flag = "yes" if d["feasible"] else "no"
        print(f"{n_burns:>7} | {d['dv_total_ms']:>7.1f} m/s | {d['max_burn_ms']:>7.1f} m/s | {flag}")
        if d["feasible"] and champion is None:
            champion = (problem, x, d)

    if champion is None:
        print("\nNo feasible schedule within n_burns <= 6.")
        return

    problem, x, d = champion
    print(f"\nMinimum feasible campaign: {d['n_burns']} apogee burns, "
          f"total dv = {d['dv_total_ms']:.1f} m/s "
          f"({100 * d['dv_total_ms'] / capacity:.0f}% of propellant budget)\n")
    print(f"{'burn':>4} | {'dv (m/s)':>8} | {'plane chg':>9} | {'perigee after':>14} | {'incl after':>10}")
    print("-" * 60)
    for k in range(d["n_burns"]):
        print(f"{k + 1:>4} | {d['dv_per_burn_ms'][k]:>8.1f} | "
              f"{d['plane_change_deg'][k]:>7.2f} deg | "
              f"{d['perigee_after_km'][k]:>11.0f} km | "
              f"{d['inclination_after_deg'][k]:>7.2f} deg")
    print(f"\nterminal orbit: a = {d['r_geo_km']:.1f} km circular, e ~ 0, i ~ 0 deg "
          f"(GEO, station {d['target_longitude_deg']:.1f} deg lon)")
    print("Screening model only (impulsive, apogee at GEO radius -- a slight dv floor).")

    _verify_in_tudatpy(problem, x)


def _verify_in_tudatpy(problem, x):
    """Stage two: re-fly the screening champion under J2+J22+Sun+Moon and refine
    the final apogee burn to true GEO -- the framework's high-fidelity check."""
    from orbitopt.verify.geo_insertion import verify_geo_raising

    print("\n--- tudatpy verification (J2 + J22 + Sun + Moon) ---")
    r = verify_geo_raising(problem, x, step_size=120.0, stationkeeping_days=3.0)
    before, after = r.achieved_uncorrected, r.achieved_corrected
    print(f"screening burns re-flown, before final-burn correction:")
    print(f"   a = {before['a_km']:.1f} km   e = {before['e']:.5f}   i = {before['i_deg']:.3f} deg")
    print(f"after target_geo_insertion (final burn nudged {np.linalg.norm(r.corrected_final_burn_ms - r.ideal_final_burn_ms):.2f} m/s):")
    print(f"   a = {after['a_km']:.3f} km ({after['a_km'] - r.r_geo_km:+.3f} km vs GEO)   "
          f"e = {after['e']:.6f}   i = {after['i_deg']:.3f} deg")
    sk = r.stationkeeping
    print(f"circular-GEO (a, e) targets {'reached' if r.converged else 'NOT reached'}; "
          f"residual {after['i_deg']:.3f} deg plane error needs a node-targeted trim.")
    print(f"station-keeping drift over {sk['days']:.0f} d (unmanaged): "
          f"di = {sk['di_deg']:+.4f} deg (N-S, luni-solar), da = {sk['da_km']:+.2f} km, de = {sk['de']:+.6f}")


if __name__ == "__main__":
    main()
