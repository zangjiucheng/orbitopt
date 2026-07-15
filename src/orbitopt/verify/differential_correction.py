"""Fixed-time differential-correction (Newton-Raphson shooting) targeter for
a translunar-injection burn.

This is the piece that turns a coarse GPU-screened guess (see
orbitopt.problems.free_return / orbitopt.dynamics.nbody_gpu) into a precise
trajectory: hold the coast duration fixed at (a refined estimate of) the
guess's own closest-approach time, then adjust the TLI delta-v vector via
Newton-Raphson so the high-fidelity tudatpy-propagated miss vector at that
fixed time scales to the desired lunar flyby distance, in the same direction
the initial guess was already missing by. Holding time fixed rather than
re-solving for the optimal flyby time each iteration is what makes this a
square (3 unknowns, 3 equations), well-posed root-finding problem instead of
an underdetermined optimization -- the standard simplification for a first
correct targeter; letting time float too is a natural extension (see the
module docstring's limitations note at the bottom of this file).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from orbitopt.verify.tudat_propagate import (
    body_position_at_absolute_epoch,
    find_closest_approach,
    propagate_multi_arc,
)


@dataclass
class TargetingResult:
    delta_v: np.ndarray  # (3,) m/s, converged TLI burn vector
    converged: bool
    iterations: int
    coast_duration: float  # seconds, the fixed time-of-flight targeted
    final_miss_km: np.ndarray  # (3,) achieved (spacecraft - Moon) vector at coast_duration
    final_distance_km: float
    residual_history_km: list = field(default_factory=list)


def _miss_vector_m(
    r0, v0_pre_burn, initial_epoch, delta_v, coast_duration,
    reference_epoch_ephemeris_seconds, perturbing_bodies, step_size,
):
    result = propagate_multi_arc(
        r0, v0_pre_burn, initial_epoch,
        arcs=[
            {"type": "impulsive_burn", "delta_v": delta_v},
            {"type": "coast", "duration": coast_duration},
        ],
        perturbing_bodies=perturbing_bodies, step_size=step_size,
    )
    spacecraft_pos_m = result.final_state[:3]
    absolute_epoch = reference_epoch_ephemeris_seconds + result.epochs[-1]
    moon_pos_m = body_position_at_absolute_epoch("Moon", absolute_epoch)
    return spacecraft_pos_m - moon_pos_m, result


def _refine_coast_duration(
    r0, v0_pre_burn, initial_epoch, dv_guess, coast_duration_guess,
    reference_epoch_ephemeris_seconds, perturbing_bodies, step_size,
):
    """Locate the nominal-trajectory's own closest-approach time near
    ``coast_duration_guess``, to use as the fixed target time for Newton
    iteration. Falls back to ``coast_duration_guess`` itself (with the miss
    vector evaluated there instead) if the coarse-model-supplied guess
    turns out to be nowhere near an actual lunar encounter -- e.g. because
    the GPU screening stage's cheap two-body Moon model mispredicted the
    encounter geometry badly enough that the high-fidelity trajectory
    doesn't pass near the Moon at all. Without this guard, locking onto
    whatever the global distance minimum happens to be over a wide window
    (which could be an incidental near-Earth point at a wildly different
    time, not a lunar encounter) sends the subsequent Newton iteration
    chasing a target that has no nearby feasible solution -- this is what
    caused an observed divergence to a residual of tens of millions of km
    in practice, not a rare corner case.
    """
    window = max(1.5 * coast_duration_guess, coast_duration_guess + 2.0 * 86400.0)
    result = propagate_multi_arc(
        r0, v0_pre_burn, initial_epoch,
        arcs=[
            {"type": "impulsive_burn", "delta_v": dv_guess},
            {"type": "coast", "duration": window},
        ],
        perturbing_bodies=perturbing_bodies, step_size=step_size,
    )
    closest = find_closest_approach(result, reference_epoch_ephemeris_seconds, body_name="Moon")

    plausible_time = abs(closest.epoch - coast_duration_guess) < 0.5 * coast_duration_guess
    plausible_distance = closest.distance_km < 200_000.0
    if plausible_time and plausible_distance:
        return closest.epoch
    return coast_duration_guess


def target_lunar_flyby(
    r0,
    v0_pre_burn,
    initial_epoch,
    dv_guess,
    coast_duration_guess,
    target_distance_km,
    reference_epoch_ephemeris_seconds,
    perturbing_bodies=("Earth", "Moon", "Sun"),
    step_size=60.0,
    maxiter=15,
    tol_km=50.0,
    fd_step_m_s=1.0,
    max_step_m_s=500.0,
) -> TargetingResult:
    """Adjust a TLI delta_v so the spacecraft's high-fidelity (tudatpy,
    Earth+Moon+Sun point-mass) trajectory passes ``target_distance_km`` from
    the Moon's center, in the same flyby geometry (direction of miss) that
    ``dv_guess`` already produces -- i.e. this dials in the right altitude
    for an already-roughly-aimed trajectory, it does not search for a flyby
    direction from scratch (that's the GPU coarse screening stage's job).

    ``step_size`` defaults to 60s, not tudat_propagate's usual few-hundred-
    second default, because a fixed-step RK4's local truncation error is set
    by the trajectory's local curvature, and curvature near a close lunar
    flyby (thousands of km altitude, several km/s relative speed) is far
    sharper than during an interplanetary coast: an empirical convergence
    check at this module's default target distance found step_size=300s
    reporting an 8281 km flyby for a trajectory whose true (step_size<=15s
    converged) closest approach was 4379 km -- essentially a wrong answer
    from a step size that would be entirely reasonable during a plain
    multi-day coast. 60s undershoots the converged value by ~4 km, well
    inside any physically meaningful targeting tolerance.

    r0/v0_pre_burn: (3,) m, m/s -- state immediately before the TLI burn.
    initial_epoch: arbitrary reference epoch (seconds) matching
        propagate_multi_arc's convention; reference_epoch_ephemeris_seconds
        is the REAL SPICE ephemeris time that initial_epoch corresponds to.
    dv_guess: (3,) m/s initial TLI burn guess (e.g. from GPU coarse search).
    coast_duration_guess: seconds, rough estimate of time-to-closest-approach
        for dv_guess -- refined internally via one exploratory propagation
        + find_closest_approach before the fixed-time Newton loop starts.
    """
    dv_guess = np.asarray(dv_guess, dtype=float)

    coast_duration = _refine_coast_duration(
        r0, v0_pre_burn, initial_epoch, dv_guess, coast_duration_guess,
        reference_epoch_ephemeris_seconds, perturbing_bodies, step_size,
    )

    def miss(dv_trial):
        m, _ = _miss_vector_m(
            r0, v0_pre_burn, initial_epoch, dv_trial, coast_duration,
            reference_epoch_ephemeris_seconds, perturbing_bodies, step_size,
        )
        return m

    m0 = miss(dv_guess)
    d0_km = np.linalg.norm(m0) / 1000.0
    if d0_km < 1e-6:
        target_vector_m = np.array([target_distance_km * 1000.0, 0.0, 0.0])
    else:
        target_vector_m = m0 * (target_distance_km / d0_km)

    dv = dv_guess.copy()
    residual_history_km = []
    converged = False
    iterations = 0

    for iterations in range(1, maxiter + 1):
        current_miss = miss(dv)
        residual_m = current_miss - target_vector_m
        residual_km = np.linalg.norm(residual_m) / 1000.0
        residual_history_km.append(float(residual_km))

        if residual_km < tol_km:
            converged = True
            break

        jacobian = np.zeros((3, 3))
        for k in range(3):
            perturbed = dv.copy()
            perturbed[k] += fd_step_m_s
            jacobian[:, k] = (miss(perturbed) - current_miss) / fd_step_m_s

        try:
            delta_dv = np.linalg.solve(jacobian, -residual_m)
        except np.linalg.LinAlgError:
            delta_dv = np.linalg.lstsq(jacobian, -residual_m, rcond=None)[0]

        # Undamped Newton on this system can and does diverge in practice
        # (observed: residual ballooning to tens of millions of km within a
        # few iterations) when the starting guess's linearization doesn't
        # hold all the way to the full step -- e.g. a GPU-coarse-screened
        # guess whose true high-fidelity Jacobian differs enough from the
        # two-body-approximate one implicit in the guess. A capped,
        # backtracking step (classic damped Newton / Armijo-style
        # sufficient-decrease check) trades a few extra iterations for
        # actually converging instead of running away.
        step_scale = min(1.0, max_step_m_s / max(np.linalg.norm(delta_dv), 1e-9))
        for _ in range(6):
            trial_dv = dv + step_scale * delta_dv
            trial_residual_km = np.linalg.norm(miss(trial_dv) - target_vector_m) / 1000.0
            if trial_residual_km < residual_km:
                break
            step_scale *= 0.5
        else:
            trial_dv = dv + step_scale * delta_dv

        dv = trial_dv

    final_miss_m = miss(dv)
    return TargetingResult(
        delta_v=dv,
        converged=converged,
        iterations=iterations,
        coast_duration=coast_duration,
        final_miss_km=final_miss_m / 1000.0,
        final_distance_km=float(np.linalg.norm(final_miss_m) / 1000.0),
        residual_history_km=residual_history_km,
    )
