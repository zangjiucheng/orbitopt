"""Verify a pykep Lambert-arc prediction against a numerical, N-body-aware
propagation in tudatpy, for a single Earth->Mars leg.

pykep's Lambert solution assumes pure two-body (Sun-only) motion between
r1 and r2; tudatpy propagates the same initial state under point-mass
gravity from Sun + Earth + Mars + Jupiter and reports how far the resulting
arrival state drifts from the patched-conic prediction.

Run: python examples/03_verify_with_tudat.py
"""
from __future__ import annotations

import numpy as np
import pykep as pk

from orbitopt.bodies import mjd2000_from_date, planet
from orbitopt.lambert.cpu import solve_lambert_single
from orbitopt.verify.tudat_propagate import (
    compare_to_lambert_prediction,
    propagate_two_body_leg,
)


def main():
    earth = planet("earth")
    mars = planet("mars")

    t0 = mjd2000_from_date(2026, 10, 1)
    tof_days = 220.0
    tof_s = tof_days * pk.DAY2SEC

    r1, v1_earth = earth.eph(pk.epoch(t0))
    r2, v2_mars = mars.eph(pk.epoch(t0 + tof_days))

    (v1, v2), = solve_lambert_single(r1, r2, tof_s, mu=pk.MU_SUN, max_revs=0)[:1]

    print(f"pykep Lambert solution: departure dv = "
          f"{np.linalg.norm(np.asarray(v1) - np.asarray(v1_earth)):.1f} m/s, "
          f"arrival v_inf = {np.linalg.norm(np.asarray(v2) - np.asarray(v2_mars)):.1f} m/s")

    print("Propagating with tudatpy (Sun + Earth + Mars + Jupiter point masses)...")
    result = propagate_two_body_leg(
        r1, v1, tof_s,
        perturbing_bodies=("Sun", "Earth", "Mars", "Jupiter"),
        central_body="Sun",
        step_size=3600.0,
    )

    diff = compare_to_lambert_prediction(result, r2, v2)
    print(f"\nDrift vs. pykep patched-conic prediction after {tof_days:.0f} days:")
    print(f"  position error: {diff['position_error_km']:,.1f} km")
    print(f"  velocity error: {diff['velocity_error_m_s']:.2f} m/s")
    print("(non-zero by construction -- Earth/Mars/Jupiter perturb the transfer "
          "that pykep modelled as an unperturbed two-body arc)")


if __name__ == "__main__":
    main()
