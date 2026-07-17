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

from orbitopt.verify.b_plane import (
    b_plane_angle_deg,
    b_plane_magnitude_for_periapsis,
    b_plane_state,
    perpendicular_basis,
    periapsis_distance_m,
)
from orbitopt.verify.tudat_propagate import (
    body_gravitational_parameter,
    body_position_at_absolute_epoch,
    body_velocity_at_absolute_epoch,
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


@dataclass
class BPlaneTargetingResult:
    delta_v: np.ndarray  # (3,) m/s, converged TLI burn vector
    converged: bool
    iterations: int
    coast_duration: float  # seconds, the fixed time-of-flight targeted
    final_b_vec_m: np.ndarray  # (3,) achieved B-plane vector at coast_duration
    final_periapsis_km: float
    final_b_plane_angle_deg: float
    final_v_inf_m_s: float
    residual_history_km: list = field(default_factory=list)


def _miss_vector_m(
    r0, v0_pre_burn, initial_epoch, delta_v, coast_duration,
    reference_epoch_ephemeris_seconds, perturbing_bodies, step_size,
):
    # propagate_multi_arc feeds its initial_epoch straight into tudatpy,
    # which uses it as the absolute SPICE ephemeris epoch for every
    # Earth/Moon/Sun point-mass gravity lookup during integration -- so it
    # must be the real mission epoch, not the caller's arbitrary local
    # clock (initial_epoch, e.g. 0.0), or the dynamics get integrated
    # through a J2000 gravity field while the miss vector below is judged
    # against the Moon's real position at reference_epoch_ephemeris_seconds.
    absolute_initial_epoch = reference_epoch_ephemeris_seconds + initial_epoch
    result = propagate_multi_arc(
        r0, v0_pre_burn, absolute_initial_epoch,
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


def _relative_state_m(
    r0, v0_pre_burn, initial_epoch, delta_v, coast_duration,
    reference_epoch_ephemeris_seconds, perturbing_bodies, step_size,
):
    """Like _miss_vector_m, but returns the full (position, velocity)
    relative to the Moon, not just the position miss vector -- B-plane
    geometry (b_plane.b_plane_state) needs both to derive B and
    v_infinity from energy/angular-momentum conservation. Same absolute-
    epoch requirement as _miss_vector_m: propagate_multi_arc's dynamics
    are driven by real SPICE epochs, so initial_epoch must be offset by
    reference_epoch_ephemeris_seconds before use, not passed as the
    caller's arbitrary local clock.
    """
    absolute_initial_epoch = reference_epoch_ephemeris_seconds + initial_epoch
    result = propagate_multi_arc(
        r0, v0_pre_burn, absolute_initial_epoch,
        arcs=[
            {"type": "impulsive_burn", "delta_v": delta_v},
            {"type": "coast", "duration": coast_duration},
        ],
        perturbing_bodies=perturbing_bodies, step_size=step_size,
    )
    spacecraft_state = result.final_state
    absolute_epoch = reference_epoch_ephemeris_seconds + result.epochs[-1]
    moon_pos_m = body_position_at_absolute_epoch("Moon", absolute_epoch)
    moon_vel_m_s = body_velocity_at_absolute_epoch("Moon", absolute_epoch)
    r_rel_m = spacecraft_state[:3] - moon_pos_m
    v_rel_m_s = spacecraft_state[3:] - moon_vel_m_s
    return r_rel_m, v_rel_m_s, result


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
    # Same real-epoch requirement as _miss_vector_m: propagate_multi_arc's
    # initial_epoch drives tudatpy's SPICE-backed gravity environment
    # directly, so it must be the absolute mission epoch, not the caller's
    # arbitrary local clock.
    absolute_initial_epoch = reference_epoch_ephemeris_seconds + initial_epoch
    result = propagate_multi_arc(
        r0, v0_pre_burn, absolute_initial_epoch,
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

    Note: this targets a *fixed Cartesian miss-vector direction* (the initial
    guess's own miss direction, rescaled to the target distance) -- a good
    approximation as long as the initial guess's own miss is already large
    compared to the Moon's local curvature scale (a few hundred km or more).
    For an initial guess whose *uncorrected* trajectory already passes very
    close to (or through) the Moon, that direction is poorly conditioned and
    this targeter may not converge; callers with that failure mode should
    adjust their coarse guess (e.g. departure epoch / coast time) to produce
    a less degenerate starting miss, rather than rely on this function to
    recover from it.
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
    stall_count = 0

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
        #
        # If every backtracking halving still fails to improve on the
        # current residual (observed near a close lunar encounter, where
        # the linearization breaks down well before the capped step size),
        # the step must NOT be taken anyway -- silently committing to the
        # smallest-but-still-worse trial (the previous behavior here) walks
        # the solution slightly further from the target on *every* such
        # iteration, which reads as slow divergence rather than a stall.
        # Hold dv unchanged for this iteration instead; a run of consecutive
        # stalls means the local Jacobian genuinely has no improving
        # direction at this point (recomputing it at the same dv would just
        # reproduce the same result), so give up rather than burn the rest
        # of maxiter doing nothing.
        step_scale = min(1.0, max_step_m_s / max(np.linalg.norm(delta_dv), 1e-9))
        improved = False
        for _ in range(12):
            trial_dv = dv + step_scale * delta_dv
            trial_residual_km = np.linalg.norm(miss(trial_dv) - target_vector_m) / 1000.0
            if trial_residual_km < residual_km:
                dv = trial_dv
                improved = True
                break
            step_scale *= 0.5

        if improved:
            stall_count = 0
        else:
            stall_count += 1
            if stall_count >= 3:
                break

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


def _target_lunar_b_plane_fixed_time(
    r0, v0_pre_burn, initial_epoch, dv_guess, coast_duration,
    target_periapsis_km, target_b_plane_angle_deg,
    reference_epoch_ephemeris_seconds, mu_moon,
    perturbing_bodies, step_size, maxiter, tol_km, fd_step_m_s, max_step_m_s,
):
    """One fixed-time B-plane Newton-Raphson attempt -- see
    target_lunar_b_plane for why this is wrapped in an outer re-refinement
    loop rather than used directly. Returns (dv, converged, iterations,
    residual_history_km, final_b_vec_m, final_v_inf_vec_m_s), or raises
    ValueError if dv_guess itself isn't locally hyperbolic at coast_duration.
    """
    def relative_state(dv_trial):
        r_rel, v_rel, _ = _relative_state_m(
            r0, v0_pre_burn, initial_epoch, dv_trial, coast_duration,
            reference_epoch_ephemeris_seconds, perturbing_bodies, step_size,
        )
        return r_rel, v_rel

    def b_plane(dv_trial):
        r_rel, v_rel = relative_state(dv_trial)
        b_vec, v_inf_vec = b_plane_state(r_rel, v_rel, mu_moon)
        if b_vec is None:
            # Not locally hyperbolic at this trial -- push the solver away
            # from this region rather than crashing: a residual that grows
            # with distance from a nominal reference behaves like a wall a
            # damped-Newton backtracking search naturally backs off from.
            return None, None
        return b_vec, v_inf_vec

    r_rel0, v_rel0 = relative_state(dv_guess)
    b_vec0, v_inf_vec0 = b_plane_state(r_rel0, v_rel0, mu_moon)
    if b_vec0 is None:
        raise ValueError(
            "dv_guess's coasted trajectory is not locally hyperbolic near the Moon at "
            "coast_duration -- target_lunar_b_plane needs a real hyperbolic flyby arc to "
            "measure a B-plane angle within; adjust dv_guess/coast_duration_guess so the "
            "initial guess actually passes near the Moon."
        )
    v_inf0 = float(np.linalg.norm(v_inf_vec0))
    s_hat = v_inf_vec0 / v_inf0
    t_hat, r_hat = perpendicular_basis(s_hat)

    b_target_mag_m = b_plane_magnitude_for_periapsis(target_periapsis_km * 1000.0, v_inf0, mu_moon)
    angle_rad = np.radians(target_b_plane_angle_deg)
    target_b_vec_m = b_target_mag_m * (np.cos(angle_rad) * t_hat + np.sin(angle_rad) * r_hat)

    def residual(dv_trial):
        b_vec, v_inf_vec = b_plane(dv_trial)
        if b_vec is None:
            return None
        return b_vec - target_b_vec_m

    dv = dv_guess.copy()
    residual_history_km = []
    converged = False
    iterations = 0
    stall_count = 0

    for iterations in range(1, maxiter + 1):
        current_residual = residual(dv)
        if current_residual is None:
            # The starting point itself is degenerate for this trial dv --
            # nudge back toward dv_guess (known-good) and try again next
            # iteration rather than attempting a Jacobian here.
            dv = 0.5 * (dv + dv_guess)
            residual_history_km.append(float("nan"))
            continue
        residual_km = np.linalg.norm(current_residual) / 1000.0
        residual_history_km.append(float(residual_km))

        if residual_km < tol_km:
            converged = True
            break

        jacobian = np.zeros((3, 3))
        jacobian_ok = True
        for k in range(3):
            perturbed = dv.copy()
            perturbed[k] += fd_step_m_s
            perturbed_residual = residual(perturbed)
            if perturbed_residual is None:
                jacobian_ok = False
                break
            jacobian[:, k] = (perturbed_residual - current_residual) / fd_step_m_s

        if not jacobian_ok:
            stall_count += 1
            if stall_count >= 3:
                break
            continue

        try:
            delta_dv = np.linalg.solve(jacobian, -current_residual)
        except np.linalg.LinAlgError:
            delta_dv = np.linalg.lstsq(jacobian, -current_residual, rcond=None)[0]

        # Same damped-Newton backtracking as target_lunar_flyby -- see that
        # function's own comment for why undamped Newton isn't safe here.
        step_scale = min(1.0, max_step_m_s / max(np.linalg.norm(delta_dv), 1e-9))
        improved = False
        for _ in range(12):
            trial_dv = dv + step_scale * delta_dv
            trial_residual = residual(trial_dv)
            if trial_residual is not None:
                trial_residual_km = np.linalg.norm(trial_residual) / 1000.0
                if trial_residual_km < residual_km:
                    dv = trial_dv
                    improved = True
                    break
            step_scale *= 0.5

        if improved:
            stall_count = 0
        else:
            stall_count += 1
            if stall_count >= 3:
                break

    final_b_vec, final_v_inf_vec = b_plane(dv)
    if final_b_vec is None:
        final_b_vec, final_v_inf_vec = b_vec0, v_inf_vec0  # last-resort fallback for reporting only
    return dv, converged, iterations, residual_history_km, final_b_vec, final_v_inf_vec


def target_lunar_b_plane(
    r0,
    v0_pre_burn,
    initial_epoch,
    dv_guess,
    coast_duration_guess,
    target_periapsis_km,
    target_b_plane_angle_deg,
    reference_epoch_ephemeris_seconds,
    mu_moon=None,
    perturbing_bodies=("Earth", "Moon", "Sun"),
    step_size=60.0,
    maxiter=15,
    tol_km=50.0,
    fd_step_m_s=1.0,
    max_step_m_s=500.0,
    outer_rounds=4,
) -> BPlaneTargetingResult:
    """Genuine 2-DOF B-plane targeter: adjust a TLI delta_v so the
    spacecraft's high-fidelity trajectory hits a *chosen* B-plane aim
    point at the Moon -- both the periapsis distance (as target_lunar_flyby
    already does) AND the B-plane angle (which side of the Moon the flyby
    passes on, and so which way the post-flyby return leg bends -- see
    b_plane.b_plane_angle_deg). target_lunar_flyby cannot do this: it
    rescales whatever miss *direction* the initial guess happens to
    produce, so the resulting B-plane angle -- and so the return leg's
    entry altitude and entry angle -- is incidental, not controlled. This
    function is the real fix free_return_search.py's module docstring and
    DEFAULT_TRANSFER_THETA_DEG's docstring (orbitopt.viz.mission_timeline)
    both flagged as still-needed follow-up work.

    Each round is a square (3 equations, 3 unknowns) Newton-Raphson, same
    well-posed structure as target_lunar_flyby, holding time fixed at a
    refined closest-approach estimate for that round -- the *target*
    differs from target_lunar_flyby's: instead of "the initial guess's own
    miss direction, rescaled", it's a specific (T_hat, R_hat) B-plane basis
    point chosen by the caller. The 3D residual is the B-plane vector
    difference (achieved B_vec minus target B_vec) rather than a raw
    Cartesian position-miss difference: B_vec is an invariant of the
    *unperturbed* two-body arc (energy + angular-momentum conservation, see
    b_plane.b_plane_state), so it stays a well-conditioned target even
    evaluated slightly off the true periapsis time, the same reason
    free_return_search.py already uses it for *reporting*.

    Why ``outer_rounds`` at all, when target_lunar_flyby gets away with a
    single fixed-time Newton solve: empirically confirmed (not assumed) that
    a single round's fixed evaluation time is NOT reliably the trajectory's
    own true closest-approach time once a *large* B-plane correction moves
    the periapsis substantially (e.g. targeting 8,282 km periapsis starting
    from a 4,499 km one) -- correcting delta_v shifts *when* the true
    closest approach happens, so a time fixed before that correction can
    land the corrector on a stale, no-longer-representative sample and
    stall (observed: residual plateauing across iterations, Newton unable
    to find any further improving direction). This wasn't a target_lunar_
    flyby problem in practice because free_return_search.py's outer theta
    search only ever asked it for *small* nearby corrections from an
    already-close Lambert guess. A direct B-plane search naturally wants to
    make bigger, more deliberate periapsis/angle moves, so this needs to be
    robust to them: each outer round re-runs _refine_coast_duration on the
    previous round's own delta_v (even if that round didn't fully converge)
    before attempting another fixed-time Newton solve -- an outer fixed-
    point iteration on "what time is periapsis" wrapped around the inner
    "what delta_v hits the target at that time", continuing until the
    B-plane residual is within tol_km or outer_rounds is exhausted.

    The (T_hat, R_hat) basis and the v_infinity magnitude used to convert
    target_periapsis_km into a B-plane magnitude (b_plane.
    b_plane_magnitude_for_periapsis) are both re-derived at the START of
    each outer round from that round's own current delta_v -- consistent
    with target_lunar_flyby's established pattern of linearizing around a
    known-local geometry, just re-anchored each round instead of once,
    since a large correction can shift v_infinity enough that a basis
    computed once at the start would no longer be near-orthogonal to the
    later rounds' actual v_infinity direction.

    mu_moon: pass explicitly to skip a SPICE lookup (body_gravitational_
        parameter) on every call; None resolves it once here.

    Raises ValueError if dv_guess's own coasted trajectory near the Moon
    isn't locally hyperbolic (b_plane_state returns None) -- unlike
    target_lunar_flyby, this can't fall back to "just rescale whatever
    direction happened to come out", since a target B-plane angle has no
    meaning without a real hyperbolic arc to measure angles within.
    """
    dv_guess = np.asarray(dv_guess, dtype=float)
    if mu_moon is None:
        mu_moon = body_gravitational_parameter("Moon")

    dv = dv_guess
    coast_duration_seed = coast_duration_guess
    all_residual_history_km = []
    converged = False
    total_iterations = 0
    coast_duration = None
    final_b_vec = final_v_inf_vec = None

    for _round in range(1, outer_rounds + 1):
        coast_duration = _refine_coast_duration(
            r0, v0_pre_burn, initial_epoch, dv, coast_duration_seed,
            reference_epoch_ephemeris_seconds, perturbing_bodies, step_size,
        )
        dv, converged, iterations, residual_history_km, final_b_vec, final_v_inf_vec = _target_lunar_b_plane_fixed_time(
            r0, v0_pre_burn, initial_epoch, dv, coast_duration,
            target_periapsis_km, target_b_plane_angle_deg,
            reference_epoch_ephemeris_seconds, mu_moon,
            perturbing_bodies, step_size, maxiter, tol_km, fd_step_m_s, max_step_m_s,
        )
        all_residual_history_km.extend(residual_history_km)
        total_iterations += iterations
        coast_duration_seed = coast_duration
        if converged:
            break

    final_v_inf = float(np.linalg.norm(final_v_inf_vec))
    return BPlaneTargetingResult(
        delta_v=dv,
        converged=converged,
        iterations=total_iterations,
        coast_duration=coast_duration,
        final_b_vec_m=final_b_vec,
        final_periapsis_km=float(periapsis_distance_m(np.linalg.norm(final_b_vec), final_v_inf, mu_moon) / 1000.0),
        final_b_plane_angle_deg=b_plane_angle_deg(final_b_vec, final_v_inf_vec),
        final_v_inf_m_s=final_v_inf,
        residual_history_km=all_residual_history_km,
    )
