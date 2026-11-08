"""Compare legacy closest-approach detection (fixed-step RK4 + nearest saved
sample) against the task-#136 verification layer (adaptive RKDP 8(7) +
range-rate root finding with local re-propagation), across output steps.

Part A (always runs; numpy/scipy only): a Keplerian lunar flyby with an
analytic 100 km-altitude perilune -- legacy = nearest sample on the grid,
new = ``orbitopt.verify.events.find_closest_approach_event`` with a DOP853
re-propagation hook.

Part B (needs tudatpy + SPICE kernels): the same perilune state placed
around the real Moon on 2026-04-06, back-propagated 6 h analytically and
then propagated forward 12 h through tudat's Earth+Moon+Sun point-mass model,
once with ``integrator="rk4"`` + ``method="sample"`` (legacy) and once with
the defaults (``"dop853"`` + ``method="event"``).

    PYTHONPATH=src python examples/11_event_location_comparison.py
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from orbitopt.verify import events as ev

MU_MOON = 4.9048695e12
R_MOON_KM = 1737.4
RP = (R_MOON_KM + 100.0) * 1e3
V_INF = 1000.0
STEPS = (15.0, 60.0, 300.0, 600.0, 1800.0)


def _two_body(_t, y, mu=MU_MOON):
    r = y[:3]
    return np.concatenate([y[3:], -mu * r / np.linalg.norm(r) ** 3])


def _perilune_state():
    vp = np.sqrt(V_INF**2 + 2 * MU_MOON / RP)
    return np.array([RP, 0.0, 0.0, 0.0, vp, 0.0])


def part_a():
    print("Part A: analytic Keplerian flyby, true perilune distance = %.3f km" % (RP / 1e3))
    print(f"{'step [s]':>9} {'legacy [km]':>14} {'legacy err':>12} {'event [km]':>14} {'event err [m]':>14}")
    yp, t_peri, span = _perilune_state(), 12_345.678, 6 * 3600.0
    back = solve_ivp(_two_body, (t_peri, t_peri - span), yp, method="DOP853", rtol=1e-13, atol=1e-8, dense_output=True)
    fwd = solve_ivp(_two_body, (t_peri, t_peri + span), yp, method="DOP853", rtol=1e-13, atol=1e-8, dense_output=True)

    def refine(y0, t0, duration, max_step):
        sol = solve_ivp(_two_body, (t0, t0 + duration), y0, method="DOP853", rtol=1e-12, atol=1e-6, max_step=max_step)
        return sol.t, sol.y.T

    for step in STEPS:
        grid = np.arange(t_peri - span + 0.37 * step, t_peri + span, step)
        states = np.array([(back if t < t_peri else fwd).sol(t) for t in grid])
        legacy = np.min(np.linalg.norm(states[:, :3], axis=1)) / 1e3
        new = ev.find_closest_approach_event(grid, states, refine=refine).value
        print(f"{step:9.0f} {legacy:14.3f} {legacy - RP / 1e3:12.3f} {new:14.6f} {(new - RP / 1e3) * 1e3:14.3f}")


def part_b():
    try:
        from orbitopt.bodies import mjd2000_from_date, mjd2000_to_ephemeris_seconds
        from orbitopt.verify import tudat_propagate as tp
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"\nPart B skipped (tudatpy stack unavailable: {exc!r})")
        return
    t_peri = mjd2000_to_ephemeris_seconds(mjd2000_from_date(2026, 4, 6))
    lead = 6 * 3600.0
    back = solve_ivp(_two_body, (0.0, -lead), _perilune_state(), method="DOP853", rtol=1e-13, atol=1e-8)
    rel0 = back.y[:, -1]
    t0 = t_peri - lead
    moon0 = tp.spice_body_state_fn("Moon", "Earth")(t0)
    y0 = moon0 + rel0
    print("\nPart B: tudat Earth+Moon+Sun, 12 h window around a ~100 km perilune")
    print(f"{'step [s]':>9} {'rk4+sample [km]':>16} {'dop853+event [km]':>18} {'diff [km]':>10}")
    for step in STEPS:
        old = tp.propagate_multi_arc(y0[:3], y0[3:], t0, [{"type": "coast", "duration": 2 * lead}],
                                     step_size=step, integrator="rk4")
        new = tp.propagate_multi_arc(y0[:3], y0[3:], t0, [{"type": "coast", "duration": 2 * lead}],
                                     step_size=step)
        d_old = tp.find_closest_approach(old, t0, "Moon", method="sample").distance_km
        d_new = tp.find_closest_approach(new, t0, "Moon").distance_km
        print(f"{step:9.0f} {d_old:16.3f} {d_new:18.3f} {d_old - d_new:10.3f}")


if __name__ == "__main__":
    part_a()
    part_b()
