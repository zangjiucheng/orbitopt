"""End-to-end cislunar free-return pipeline, modelled on Artemis II: a
patched-conic Lambert arc seeds a GPU-batched coarse global search over the
translunar-injection burn, whose best candidate is refined by a tudatpy
differential corrector to hit the real Artemis II perilune altitude
(6,545 km above the lunar surface).

This does NOT reproduce Artemis II's actual flown trajectory -- NASA hasn't
published navigation-grade state vectors or SPICE kernels for it (see the
project's research notes). It independently re-solves the same free-return
boundary-value problem and lands on a trajectory of the same class and
comparable magnitudes.

Known limitation, stated plainly: this pipeline targets lunar flyby
DISTANCE only, in whatever direction the Lambert-seeded initial guess
happens to miss by. A genuine unpowered free return additionally requires
aiming the B-plane crossing so the post-flyby trajectory naturally
re-intersects Earth -- that's a 2-parameter aim-point targeting problem this
example does not attempt (see orbitopt/verify/differential_correction.py
for the extension point). Expect this run's post-flyby trajectory to NOT
re-enter Earth's atmosphere on its own.

Run: python examples/05_artemis2_free_return.py
"""
from __future__ import annotations

import time

import numpy as np
import pygmo as pg
import pykep as pk

from orbitopt.bodies import mjd2000_from_date, mjd2000_to_ephemeris_seconds, moon_state
from orbitopt.lambert.cpu import solve_lambert_single
from orbitopt.problems.free_return import FreeReturnScreeningProblem
from orbitopt.verify.differential_correction import target_lunar_flyby
from orbitopt.verify.tudat_propagate import (
    find_altitude_crossing,
    find_closest_approach,
    propagate_multi_arc,
)

ARTEMIS_II_PERILUNE_ALTITUDE_KM = 6545.0
MOON_RADIUS_KM = 1737.4
ARTEMIS_II_MAX_EARTH_DISTANCE_KM = 406771.0
EARTH_RADIUS_KM = 6378.0


def lambert_seeded_initial_guess(r0, v0_circular, r_moon_arrival, coast_seconds, mu_earth):
    """Patched-conic first guess: the Earth-only 2-body Lambert arc from the
    parking-orbit position to the Moon's real position at arrival time. This
    ignores the Moon's own gravity entirely (that's what the differential
    corrector is for) but gives a plane- and energy-appropriate starting
    burn instead of an arbitrary posigrade kick.
    """
    (v1, v2), = solve_lambert_single(r0, r_moon_arrival, coast_seconds, mu=mu_earth, max_revs=0)[:1]
    return v1 - v0_circular


def plane_aligned_parking_orbit(r_moon_arrival, v_moon_arrival, altitude_km, mu_earth):
    """Choose a parking-orbit orientation whose plane contains the Moon's
    arrival direction, at a 150-degree transfer angle behind it -- without
    this, an arbitrarily-oriented parking orbit forces the Lambert arc
    through a huge, unrealistic plane change (see the project's research
    notes: an unaligned first attempt needed an 11+ km/s "TLI" burn).

    Oriented so the plane also contains the Moon's own velocity at arrival
    (i.e. close to the Moon's own orbital plane, up to its ~5 degree tilt
    from the ecliptic) rather than an ecliptic-pole reference: for any
    reference vector R, the plane spanned by {u, R-(R.u)u} always has normal
    u x R, which is *exactly* perpendicular to R by construction -- using
    R=ecliptic-Z (an earlier version of this function did) therefore always
    forced an exactly-90-degree-inclined, near-polar transfer, regardless of
    where the Moon actually is. Using the Moon's own velocity avoids that
    trap and gives a genuinely moderate, physically-motivated inclination.
    """
    u = r_moon_arrival / np.linalg.norm(r_moon_arrival)
    reference = v_moon_arrival / np.linalg.norm(v_moon_arrival)
    if abs(np.dot(u, reference)) > 0.9:
        reference = np.array([0.0, 0.0, 1.0])
    e2 = reference - np.dot(reference, u) * u
    e2 /= np.linalg.norm(e2)
    e1 = u

    theta = np.radians(150.0)
    r0_hat = np.cos(theta) * e1 - np.sin(theta) * e2
    tangent_hat = np.sin(theta) * e1 + np.cos(theta) * e2

    r0 = (6378.0 + altitude_km) * 1000.0 * r0_hat
    v_circ = np.sqrt(mu_earth / np.linalg.norm(r0))
    v0 = v_circ * tangent_hat
    return r0, v0


def main():
    departure_mjd2000 = mjd2000_from_date(2026, 8, 1)
    # 4.5 days: with plane_aligned_parking_orbit aligned to the Moon's own
    # orbital plane (see that function's docstring for why this replaced an
    # earlier, buggy ecliptic-pole-referenced version that always forced a
    # near-polar transfer), this is the nearby coast time that converges
    # cleanly -- confirmed by sweeping nearby coast times at this departure
    # date for one where target_lunar_flyby's fixed miss-vector-direction
    # approach (see that function's docstring) actually converges.
    coast_days_guess = 4.5
    reference_et = mjd2000_to_ephemeris_seconds(departure_mjd2000)
    mu_earth = pk.MU_EARTH

    r_moon_arrival, v_moon_arrival = moon_state(departure_mjd2000 + coast_days_guess)
    r_moon_arrival = np.asarray(r_moon_arrival)
    v_moon_arrival = np.asarray(v_moon_arrival)

    r0, v0 = plane_aligned_parking_orbit(r_moon_arrival, v_moon_arrival, altitude_km=185.0, mu_earth=mu_earth)
    dv_lambert_guess = lambert_seeded_initial_guess(
        r0, v0, r_moon_arrival, coast_days_guess * 86400.0, mu_earth,
    )
    print(f"Lambert-seeded TLI guess: {np.linalg.norm(dv_lambert_guess):.1f} m/s "
          f"(real Artemis II TLI, from a much higher phasing orbit, was ~387 m/s -- "
          f"this example departs directly from a low 185 km circular orbit instead, "
          f"so a much larger burn is expected and correct here)")

    target_distance_km = ARTEMIS_II_PERILUNE_ALTITUDE_KM + MOON_RADIUS_KM

    print("\nGPU coarse screening around the Lambert guess...")
    dv_guess_km_s = dv_lambert_guess / 1000.0
    search_margin_km_s = 0.5  # box around the Lambert guess, not a blind wide search
    screen = FreeReturnScreeningProblem(
        r0_km=r0 / 1000.0, v0_pre_burn_km_s=v0 / 1000.0, epoch0_seconds=reference_et,
        target_distance_km=target_distance_km, coast_days=coast_days_guess,
        dv_bounds_km_s=(
            (dv_guess_km_s - search_margin_km_s).tolist(),
            (dv_guess_km_s + search_margin_km_s).tolist(),
        ),
        n_steps=300, use_gpu=False,
    )
    pop_size, generations = 100, 25
    t0 = time.perf_counter()
    pop = pg.population(pg.problem(screen), size=pop_size, seed=1)
    algo = pg.pso_gen(gen=generations)
    algo.set_bfe(pg.bfe())
    pop = pg.algorithm(algo).evolve(pop)
    x, f = pop.champion_x, pop.champion_f
    print(f"  best coarse candidate: {f[0]:.1f} km from target, dv={x} km/s "
          f"({time.perf_counter() - t0:.1f}s, {pop_size * generations} candidate-evals)")

    dv_guess = x * 1000.0  # km/s -> m/s
    coast_guess_s = screen.time_to_closest_approach(x)
    print(f"  coarse time-to-closest-approach: {coast_guess_s/86400.0:.2f} days")

    print("\nRefining with tudatpy differential correction (Earth+Moon+Sun point mass)...")
    result = target_lunar_flyby(
        r0=r0, v0_pre_burn=v0, initial_epoch=0.0,
        dv_guess=dv_guess, coast_duration_guess=max(coast_guess_s, 86400.0),
        target_distance_km=target_distance_km,
        reference_epoch_ephemeris_seconds=reference_et,
    )
    print(f"  converged: {result.converged} in {result.iterations} iterations")
    print(f"  refined delta-v: {result.delta_v} m/s, |dv|={np.linalg.norm(result.delta_v):.1f} m/s")
    print(f"  targeted perilune distance: {result.final_distance_km:.1f} km "
          f"(target {target_distance_km:.1f} km)")

    if not result.converged:
        print("\nDifferential correction did not converge -- refusing to treat the last "
              "(unconverged) delta-v as a real trajectory. Residual history (km): "
              f"{result.residual_history_km}. Try a tighter GPU-screening search box "
              "or a different departure date.")
        return

    print("\nIndependent fine-resolution verification + free-return check (9.5-day coast)...")
    full = propagate_multi_arc(
        r0, v0, 0.0,
        arcs=[
            {"type": "impulsive_burn", "delta_v": result.delta_v},
            {"type": "coast", "duration": 9.5 * 86400.0},
        ],
        perturbing_bodies=("Earth", "Moon", "Sun"), step_size=30.0,
    )
    closest = find_closest_approach(full, reference_et, body_name="Moon")
    radii_km = np.linalg.norm(full.states[:, :3], axis=1) / 1000.0
    print(f"  true closest approach to Moon: {closest.distance_km:.1f} km "
          f"at t={closest.epoch/86400.0:.2f} days "
          f"(Artemis II: {target_distance_km:.1f} km at ~4 days)")
    print(f"  max Earth-centered distance reached: {radii_km.max():.0f} km "
          f"(Artemis II: {ARTEMIS_II_MAX_EARTH_DISTANCE_KM:.0f} km)")

    entry = find_altitude_crossing(full, central_body_radius_km=EARTH_RADIUS_KM, threshold_altitude_km=121.92)
    if entry is not None:
        print(f"  Earth entry interface reached at t={entry.epoch/86400.0:.2f} days -- "
              f"genuine free return (no correction burns needed)")
    else:
        print("  no Earth entry interface within 9.5 days: this trajectory does NOT "
              "free-return on its own -- expected, see this file's module docstring "
              "(distance-only targeting, no B-plane aim-point control)")


if __name__ == "__main__":
    main()
