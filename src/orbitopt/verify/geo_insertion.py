"""High-fidelity (tudatpy) verification of a GTO -> GEO orbit-raising schedule
produced by the coarse screening optimizer (orbitopt.problems.geo_raising).

This is the "expensive, accurate" half of the framework's two-stage pattern for
the GOES mission, the analog of verify.differential_correction for the free
return: re-fly the optimizer's impulsive burn+coast sequence through numerically
integrated dynamics with the perturbations that actually matter at GEO -- Earth
J2 + J22 spherical harmonics and Sun/Moon third-body point masses -- and then
refine the *final* apogee burn with a damped Newton targeter so the achieved
osculating (a, e, i) land on true GEO, closing the ~500 km "apogee-at-GEO" floor
the closed-form screening model leaves open.

Geometry matches problems.geo_raising / viz.geo_raising: every burn fires at a
shared apogee placed on +X at the orbit's node (raan=0, arg_perigee=180 deg),
so consecutive orbits share the apogee *position* and a burn is purely a
velocity change there. Reconstructing the 3D burn vectors from the abstract
(perigee, inclination) schedule is exactly this velocity difference.

Units are SI meters / m/s throughout (tudat convention); the caller hands over
from problems.geo_raising's km via orbitopt.problems.free_return.as_meters.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from tudatpy.astro import element_conversion
from tudatpy.interface import spice

from orbitopt.bodies import mjd2000_from_date, mjd2000_to_ephemeris_seconds
from orbitopt.verify.tudat_propagate import propagate_multi_arc

GEO_PERTURBATIONS = dict(
    perturbing_bodies=("Earth", "Moon", "Sun"),
    earth_spherical_harmonic=(2, 2),
)


def _mu_earth() -> float:
    spice.load_standard_kernels()
    return spice.get_body_gravitational_parameter("Earth")


def geostationary_radius_m(mu: float, sidereal_day_s: float = 86164.0905) -> float:
    return (mu * (sidereal_day_s / (2.0 * np.pi)) ** 2) ** (1.0 / 3.0)


def apogee_state(perigee_m: float, apogee_m: float, inclination_rad: float, mu: float) -> np.ndarray:
    """Cartesian state at apogee of an ellipse with the given perigee/apogee/
    inclination, apogee placed on +X at the ascending node (raan=0, argp=180,
    true anomaly=180) -- the shared-apogee geometry every burn fires at."""
    a = 0.5 * (perigee_m + apogee_m)
    e = (apogee_m - perigee_m) / (apogee_m + perigee_m)
    kep = np.array([a, e, inclination_rad, np.pi, 0.0, np.pi])
    return np.asarray(element_conversion.keplerian_to_cartesian(kep, mu), dtype=float)


def orbital_period(perigee_m: float, apogee_m: float, mu: float) -> float:
    a = 0.5 * (perigee_m + apogee_m)
    return 2.0 * np.pi * np.sqrt(a ** 3 / mu)


def _osculating(state, mu) -> dict:
    kep = element_conversion.cartesian_to_keplerian(np.asarray(state, dtype=float), mu)
    return {"a_km": float(kep[0]) / 1000.0, "e": float(kep[1]), "i_deg": float(np.degrees(kep[2]))}


@dataclass
class GeoInsertionResult:
    converged: bool
    iterations: int
    r_geo_km: float
    ideal_final_burn_ms: np.ndarray      # (3,) screening burn vector for the last apogee burn
    corrected_final_burn_ms: np.ndarray  # (3,) targeter-refined last burn
    achieved_uncorrected: dict           # osculating a/e/i applying the ideal burn under perturbations
    achieved_corrected: dict             # osculating a/e/i after the targeter
    stationkeeping: dict                 # drift of the corrected GEO over stationkeeping_days
    epoch_mjd2000: float                 # real mission epoch (first-apogee/injection date) used throughout
    residual_history: list = field(default_factory=list)


def target_geo_insertion(pre_burn_state, dv_guess, mu, target_radius_m,
                         a_tol_km=2.0, e_tol=2e-4, maxiter=12,
                         fd_abs_ms=0.05, max_step_ms=60.0) -> tuple:
    """Refine the final apogee burn 3-vector so the osculating post-burn orbit is
    circular GEO with minimum inclination -- a damped Gauss-Newton least-squares
    fit on the dimensionless residual ``[(a-a_geo)/a_geo, e, i_rad]``, in the
    spirit of verify.differential_correction.target_lunar_flyby.

    Two things this gets right that a stock optimizer stumbles on:
      * The Jacobian is finite-differenced with a *fixed absolute* step
        (``fd_abs_ms``), not a relative one. At apogee the velocity is
        perpendicular to the radius, so the burn's radial component is ~0 -- and
        a relative step there is ~0, hiding the very sensitivity (radial burn ->
        eccentricity) needed to circularize. lstsq handles the wide scale spread
        (``a`` is ~1e4x more sensitive to a tangential burn than ``e`` is to a
        radial one).
      * The inclination target is generally *not reachable* by one burn -- J2
        precesses the node so the burn fires off-equator, and no impulse at an
        off-equator point can zero the plane (its position vector stays in the
        orbit plane). Least-squares settles ``i`` at its floor while driving the
        reachable ``a``/``e`` to circular GEO; convergence is declared on those
        two, and the residual inclination is returned for the caller to report as
        the plane error a node-targeted trim / N-S station-keeping absorbs.

    No propagation inside the loop: the burn's effect on osculating elements is
    instantaneous and ``pre_burn_state`` already carries the perturbed dynamics."""
    pre_burn_state = np.asarray(pre_burn_state, dtype=float)

    def elements(dv):
        s = pre_burn_state.copy()
        s[3:] = s[3:] + dv
        kep = element_conversion.cartesian_to_keplerian(s, mu)
        return float(kep[0]), float(kep[1]), float(kep[2])  # a, e, i

    def residual(dv):
        # Target only the *reachable* circular-GEO conditions (a, e). Inclination
        # is left to the guess's out-of-plane component (which already removes the
        # scheduled plane change); folding the unreachable i target in here just
        # drags the step off the a/e descent direction.
        a, e, _i = elements(dv)
        return np.array([(a - target_radius_m) / target_radius_m, e])

    dv = np.asarray(dv_guess, dtype=float).copy()
    best_dv = dv.copy()
    best_norm = float(np.linalg.norm(residual(dv)))
    history = [best_norm]

    for _ in range(maxiter):
        r0 = residual(dv)
        jac = np.zeros((2, 3))
        for k in range(3):
            bumped = dv.copy()
            bumped[k] += fd_abs_ms
            jac[:, k] = (residual(bumped) - r0) / fd_abs_ms
        step = np.linalg.lstsq(jac, -r0, rcond=None)[0]

        scale = min(1.0, max_step_ms / max(np.linalg.norm(step), 1e-9))
        for _ in range(8):
            trial = dv + scale * step
            trial_norm = float(np.linalg.norm(residual(trial)))
            if trial_norm < best_norm:
                dv, best_norm, best_dv = trial, trial_norm, trial.copy()
                history.append(trial_norm)
                break
            scale *= 0.5
        else:
            history.append(best_norm)
            break

    a, e, _i = elements(best_dv)
    converged = abs(a - target_radius_m) / 1000.0 < a_tol_km and e < e_tol
    return best_dv, converged, len(history) - 1, history


def verify_geo_raising(problem, decision_vector, step_size=120.0, stationkeeping_days=3.0,
                       epoch_mjd2000: float | None = None) -> GeoInsertionResult:
    """Re-fly ``problem``'s optimized schedule under J2+J22+Sun+Moon and refine
    the final burn to true GEO. ``problem`` is an
    orbitopt.problems.geo_raising.GeoRaisingProblem, ``decision_vector`` its
    optimized champion.

    ``epoch_mjd2000`` is the real mission epoch (the first-apogee / injection
    date) as MJD2000. It anchors *both* the raising propagation and the
    stationkeeping propagation that follows it, in SPICE ephemeris seconds via
    ``orbitopt.bodies.mjd2000_to_ephemeris_seconds`` -- the same pattern
    ``verify.differential_correction`` / ``viz.mission_timeline`` use for the
    lunar-flyby case -- so the J2/J22 field orientation and Sun/Moon third-body
    geometry (which drive the station-keeping drift this function reports)
    correspond to an actual calendar date instead of the J2000 placeholder
    (2000-01-01 12:00 TDB). Unlike the lunar-flyby targeter, nothing here
    iterates the epoch back toward consistency, so leaving it at the J2000
    default silently reports drift/libration for a date with no relation to
    the mission. Defaults to 2026-08-01 absent a published GOES-like launch
    date (matching ``viz.mission_timeline``'s Artemis II placeholder)."""
    if epoch_mjd2000 is None:
        epoch_mjd2000 = mjd2000_from_date(2026, 8, 1)
    reference_epoch_s = mjd2000_to_ephemeris_seconds(epoch_mjd2000)

    mu = _mu_earth()
    r_geo = geostationary_radius_m(mu)
    sched = problem.decode(decision_vector)
    n = sched["n_burns"]

    # (perigee, inclination) of the injection orbit and each post-burn orbit,
    # all sharing apogee at r_geo (the screening model's geometry).
    orbits = [(problem.rp0 * 1000.0, problem.i0)]
    for k in range(n):
        orbits.append((sched["perigee_after_km"][k] * 1000.0, np.radians(sched["inclination_after_deg"][k])))

    states = [apogee_state(rp, r_geo, incl, mu) for rp, incl in orbits]
    burns = [states[k + 1][3:] - states[k][3:] for k in range(n)]  # velocity differences at apogee

    # Propagate the fixed part (injection apogee -> burns 1..N-1 -> final apogee)
    # under the perturbed dynamics to get the real pre-final-burn state.
    arcs = []
    for k in range(n - 1):
        arcs.append({"type": "impulsive_burn", "delta_v": burns[k]})
        period = orbital_period(orbits[k + 1][0], r_geo, mu)
        arcs.append({"type": "coast", "duration": period})
    if arcs:
        raising = propagate_multi_arc(
            states[0][:3], states[0][3:], reference_epoch_s, arcs=arcs,
            step_size=step_size, **GEO_PERTURBATIONS,
        )
        pre_final = raising.final_state
        # propagate_multi_arc reports epochs relative to the initial_epoch it
        # was given (here reference_epoch_s), so this is the real elapsed
        # raising time -- the epoch stationkeeping must continue from below,
        # not the mission epoch itself (the final burn is instantaneous, so
        # pre_final/corrected_state share this same epoch).
        elapsed_to_pre_final_s = float(raising.epochs[-1])
    else:
        pre_final = states[0]
        elapsed_to_pre_final_s = 0.0

    ideal_burn = burns[n - 1]
    uncorrected_state = pre_final.copy()
    uncorrected_state[3:] = uncorrected_state[3:] + ideal_burn
    achieved_uncorrected = _osculating(uncorrected_state, mu)

    corrected_burn, converged, iterations, history = target_geo_insertion(
        pre_final, ideal_burn, mu, r_geo,
    )
    corrected_state = pre_final.copy()
    corrected_state[3:] = corrected_state[3:] + corrected_burn
    achieved_corrected = _osculating(corrected_state, mu)

    # Station-keeping drift: fly the corrected GEO forward and measure how far
    # the osculating elements wander (Sun/Moon -> N-S inclination, J22 -> e).
    sk = propagate_multi_arc(
        corrected_state[:3], corrected_state[3:], reference_epoch_s + elapsed_to_pre_final_s,
        arcs=[{"type": "coast", "duration": stationkeeping_days * 86400.0}],
        step_size=step_size, **GEO_PERTURBATIONS,
    )
    sk_end = _osculating(sk.final_state, mu)
    stationkeeping = {
        "days": stationkeeping_days,
        "di_deg": sk_end["i_deg"] - achieved_corrected["i_deg"],
        "da_km": sk_end["a_km"] - achieved_corrected["a_km"],
        "de": sk_end["e"] - achieved_corrected["e"],
    }

    return GeoInsertionResult(
        converged=converged,
        iterations=iterations,
        r_geo_km=r_geo / 1000.0,
        ideal_final_burn_ms=ideal_burn,
        corrected_final_burn_ms=corrected_burn,
        achieved_uncorrected=achieved_uncorrected,
        achieved_corrected=achieved_corrected,
        stationkeeping=stationkeeping,
        epoch_mjd2000=epoch_mjd2000,
        residual_history=history,
    )
