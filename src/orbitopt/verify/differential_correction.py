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
from orbitopt.verify import report as _report
from orbitopt.verify.tudat_propagate import (
    body_gravitational_parameter,
    body_position_at_absolute_epoch,
    body_velocity_at_absolute_epoch,
    find_altitude_crossing,
    find_closest_approach,
    find_periapsides,
    propagate_multi_arc,
    PropagationResult,
    propagation_report_section,
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
    report: dict | None = None  # machine-readable verification report (orbitopt.verify.report)


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
    report: dict | None = None  # machine-readable verification report (orbitopt.verify.report)


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
    report_path=None,
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

    final_miss_m, final_result = _miss_vector_m(
        r0, v0_pre_burn, initial_epoch, dv, coast_duration,
        reference_epoch_ephemeris_seconds, perturbing_bodies, step_size,
    )
    report = _report.build_report(
        kind="lunar-flyby-tli", label=f"ref-et-{float(reference_epoch_ephemeris_seconds):.3f}",
        propagations=[propagation_report_section(final_result, "tli-coast")],
        residuals=[_report.state_residual(
            "flyby_miss_vs_aim_point", "Moon-relative targeted miss vector at the fixed flyby time",
            position=final_miss_m, reference_position=target_vector_m,
            tolerance={"position_m": tol_km * 1000.0},
        )],
        inputs={"pre_burn_state": np.concatenate([np.asarray(r0, dtype=float), np.asarray(v0_pre_burn, dtype=float)]),
                "initial_epoch": float(initial_epoch), "dv_guess_m_s": dv_guess,
                "coast_duration_guess_s": float(coast_duration_guess), "target_distance_km": float(target_distance_km),
                "reference_epoch_ephemeris_s": float(reference_epoch_ephemeris_seconds), "step_size_s": float(step_size)},
        targets={"converged": converged, "iterations": iterations, "delta_v_m_s": dv,
                 "coast_duration_s": coast_duration, "final_distance_km": float(np.linalg.norm(final_miss_m) / 1000.0),
                 "residual_history_km": residual_history_km},
    )
    _report.finalize_report(report, report_path)
    return TargetingResult(
        delta_v=dv,
        converged=converged,
        iterations=iterations,
        coast_duration=coast_duration,
        final_miss_km=final_miss_m / 1000.0,
        final_distance_km=float(np.linalg.norm(final_miss_m) / 1000.0),
        residual_history_km=residual_history_km,
        report=report,
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
    report_path=None,
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
    final_periapsis_km = float(periapsis_distance_m(np.linalg.norm(final_b_vec), final_v_inf, mu_moon) / 1000.0)
    final_angle_deg = b_plane_angle_deg(final_b_vec, final_v_inf_vec)
    _r, _v, final_result = _relative_state_m(
        r0, v0_pre_burn, initial_epoch, dv, coast_duration,
        reference_epoch_ephemeris_seconds, perturbing_bodies, step_size,
    )
    report = _report.build_report(
        kind="lunar-b-plane-tli", label=f"ref-et-{float(reference_epoch_ephemeris_seconds):.3f}",
        propagations=[propagation_report_section(final_result, "tli-coast")],
        inputs={"pre_burn_state": np.concatenate([np.asarray(r0, dtype=float), np.asarray(v0_pre_burn, dtype=float)]),
                "initial_epoch": float(initial_epoch), "dv_guess_m_s": dv_guess,
                "coast_duration_guess_s": float(coast_duration_guess),
                "target_periapsis_km": float(target_periapsis_km),
                "target_b_plane_angle_deg": float(target_b_plane_angle_deg),
                "reference_epoch_ephemeris_s": float(reference_epoch_ephemeris_seconds), "step_size_s": float(step_size)},
        targets={"converged": converged, "iterations": total_iterations, "delta_v_m_s": dv,
                 "coast_duration_s": coast_duration, "final_periapsis_km": final_periapsis_km,
                 "periapsis_error_km": final_periapsis_km - float(target_periapsis_km),
                 "final_b_plane_angle_deg": float(final_angle_deg),
                 "b_plane_angle_error_deg": float((final_angle_deg - float(target_b_plane_angle_deg) + 180.0) % 360.0 - 180.0),
                 "final_v_inf_m_s": final_v_inf, "residual_history_km": all_residual_history_km},
    )
    _report.finalize_report(report, report_path)
    return BPlaneTargetingResult(
        delta_v=dv,
        converged=converged,
        iterations=total_iterations,
        coast_duration=coast_duration,
        final_b_vec_m=final_b_vec,
        final_periapsis_km=final_periapsis_km,
        final_b_plane_angle_deg=final_angle_deg,
        final_v_inf_m_s=final_v_inf,
        residual_history_km=all_residual_history_km,
        report=report,
    )


# --------------------------------------------------------------------------
# Joint lunar B-plane + Earth-return entry-corridor targeting (task #144)
# --------------------------------------------------------------------------

EARTH_RADIUS_KM = 6378.0  # same spherical-Earth radius viz.mission_timeline reports altitudes against
EARTH_ENTRY_INTERFACE_ALTITUDE_KM = 121.92  # 400,000 ft, the conventional atmospheric entry interface
LUNAR_RETURN_ENTRY_FPA_DEG = -6.0
"""Default target inertial flight-path angle at the entry interface: the
middle of an Orion-class lunar-return skip-entry corridor (~-5.5 to -6.5
degrees flown; anything much steeper than ~-7 degrees means unsurvivable
deceleration loads, much shallower than ~-5 degrees risks skipping out)."""
LUNAR_RETURN_ENTRY_CORRIDOR_DEG = (-7.0, -5.0)
"""(steepest, shallowest) inertial entry flight-path angle accepted as
inside the safe lunar-return corridor."""


@dataclass
class FreeReturnTargetingResult:
    delta_v: np.ndarray  # (3,) m/s, converged TLI burn vector
    converged: bool  # joint Newton met both the perilune and the return-perigee tolerances
    in_corridor: bool  # TudatPy-verified entry FPA inside entry_corridor_deg
    iterations: int  # joint-Newton iterations (the B-plane seeding stage is reported separately)
    coast_duration: float  # seconds, TLI -> lunar B-plane sample epoch (the refined closest approach)
    perilune_epoch: float  # seconds after TLI, located by root finding on the verified propagation
    perilune_distance_km: float  # verified closest approach to the Moon's centre
    b_plane_angle_deg: float  # achieved lunar B-plane angle (b_plane.b_plane_angle_deg convention)
    return_perigee_km: float  # verified vacuum (no-atmosphere) return perigee radius from Earth's centre
    target_return_perigee_km: float  # vacuum perigee radius that yields target_entry_fpa_deg
    entry_epoch: float | None  # seconds after TLI of the downward entry-interface crossing
    entry_state: np.ndarray | None  # (6,) Earth-centred state at the entry interface
    entry_speed_m_s: float | None
    entry_flight_path_angle_deg: float | None  # inertial, TudatPy-propagated (negative = descending)
    b_plane_seed: BPlaneTargetingResult | None = None
    residual_history_km: list = field(default_factory=list)  # per joint iteration: [perilune err, perigee err]
    report: dict | None = None


def flight_path_angle_deg(state):
    """Inertial flight-path angle (degrees, negative while descending) of a
    central-body-relative Cartesian state."""
    r = np.asarray(state[:3], dtype=float)
    v = np.asarray(state[3:6], dtype=float)
    return float(np.degrees(np.arcsin(np.clip(np.dot(r, v) / (np.linalg.norm(r) * np.linalg.norm(v)), -1.0, 1.0))))


def perigee_radius_for_entry_angle_m(specific_energy, entry_radius_m, fpa_deg, mu):
    """Vacuum (two-body) perigee radius of the Earth-return conic with
    ``specific_energy`` that crosses ``entry_radius_m`` at inertial flight-
    path angle ``fpa_deg`` -- the entry angle expressed as a smooth, always-
    defined return-perigee target (an entry angle only exists once the
    trajectory actually dips below the interface; a perigee radius exists
    for every bound or hyperbolic return, so a Newton iteration can be
    driven with it from anywhere). From h = r_EI v_EI cos(gamma) and
    eps r_p^2 + mu r_p - h^2/2 = 0, written in the cancellation-free form.
    """
    v_entry = np.sqrt(2.0 * (specific_energy + mu / entry_radius_m))
    h = entry_radius_m * v_entry * np.cos(np.radians(fpa_deg))
    return float(h * h / (mu + np.sqrt(mu * mu + 2.0 * specific_energy * h * h)))


def _patched_conic_b_plane_seed(
    r_rel_m, v_rel_m_s, sample_epoch_abs, target_periapsis_km, target_return_perigee_m,
    mu_moon, mu_earth, entry_radius_m, target_fpa_deg, n_angles=720,
):
    """Pick the lunar B-plane angle whose *patched-conic* return conic comes
    closest to ``target_return_perigee_m`` -- a cheap analytic seed for the
    B-plane targeter, not a solution: holding the incoming asymptote of the
    current trajectory fixed, sweep the aim point around the B-plane at the
    target periapsis distance, deflect the asymptote by the corresponding
    two-body turn angle (sin(delta/2) = 1/e), add the Moon's velocity and
    read the Earth-return perigee off the resulting geocentric conic placed
    at the Moon's position. Returns (b_plane_angle_deg, predicted_perigee_m)
    in b_plane.b_plane_angle_deg's convention (the angle of the *outgoing*
    B vector in the (T, R) basis of its own outgoing v_infinity -- the
    convention target_lunar_b_plane converges to), or (None, None) if the
    current trajectory isn't locally hyperbolic at the Moon.
    """
    _b_rev, v_inf_rev = b_plane_state(r_rel_m, -np.asarray(v_rel_m_s, dtype=float), mu_moon)
    if v_inf_rev is None:
        return None, None
    v_inf_in = -v_inf_rev
    v_inf = float(np.linalg.norm(v_inf_in))
    s_in = v_inf_in / v_inf
    t_in, r_in = perpendicular_basis(s_in)
    rp_m = target_periapsis_km * 1000.0
    ecc = 1.0 + rp_m * v_inf * v_inf / mu_moon
    turn = 2.0 * np.arcsin(1.0 / ecc)
    b_mag = b_plane_magnitude_for_periapsis(rp_m, v_inf, mu_moon)
    moon_r = body_position_at_absolute_epoch("Moon", sample_epoch_abs)
    moon_v = body_velocity_at_absolute_epoch("Moon", sample_epoch_abs)

    best = (np.inf, None, None)
    for ang in np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False):
        b_hat = np.cos(ang) * t_in + np.sin(ang) * r_in
        # Gravity bends the asymptote toward the body, i.e. away from +B.
        v_out = v_inf * (np.cos(turn) * s_in - np.sin(turn) * b_hat)
        r_geo, v_geo = moon_r, moon_v + v_out
        eps = 0.5 * float(v_geo @ v_geo) - mu_earth / float(np.linalg.norm(r_geo))
        h_vec = np.cross(r_geo, v_geo)
        e_vec = np.cross(v_geo, h_vec) / mu_earth - r_geo / np.linalg.norm(r_geo)
        perigee_m = float(h_vec @ h_vec) / mu_earth / (1.0 + np.linalg.norm(e_vec))
        target_m = perigee_radius_for_entry_angle_m(eps, entry_radius_m, target_fpa_deg, mu_earth) \
            if eps + mu_earth / entry_radius_m > 0.0 else target_return_perigee_m
        err = abs(perigee_m - target_m)
        if err < best[0]:
            h_flyby = np.cross(b_mag * b_hat, v_inf_in)
            b_out = np.cross(h_flyby, v_out) / (v_inf * v_inf)
            best = (err, b_plane_angle_deg(b_out, v_out), perigee_m)
    return best[1], best[2]


def _flyby_and_return(
    r0, v0_pre_burn, initial_epoch, delta_v, coast_duration, return_window_s,
    reference_epoch_ephemeris_seconds, perturbing_bodies, step_size, mu_moon,
):
    """One TLI -> lunar B-plane sample (exactly at ``coast_duration``) ->
    Earth-return propagation. Returns a dict with the lunar B-plane state at
    the sample epoch and the first Earth periapsis passage of the return
    leg, or None if the trajectory isn't hyperbolic at the Moon or never
    comes back down to a perigee inside ``return_window_s`` (escape / a
    trial that hit the point-mass Earth and terminated early)."""
    absolute_initial_epoch = reference_epoch_ephemeris_seconds + initial_epoch
    result = propagate_multi_arc(
        r0, v0_pre_burn, absolute_initial_epoch,
        arcs=[
            {"type": "impulsive_burn", "delta_v": delta_v},
            {"type": "coast", "duration": coast_duration},
            {"type": "coast", "duration": return_window_s},
        ],
        perturbing_bodies=perturbing_bodies, step_size=step_size,
    )
    (_s0, e0), (s1, e1) = result.segments
    sample_epoch_abs = reference_epoch_ephemeris_seconds + result.epochs[e0 - 1]
    sc_state = result.states[e0 - 1]
    r_rel = sc_state[:3] - body_position_at_absolute_epoch("Moon", sample_epoch_abs)
    v_rel = sc_state[3:] - body_velocity_at_absolute_epoch("Moon", sample_epoch_abs)
    b_vec, v_inf_vec = b_plane_state(r_rel, v_rel, mu_moon)
    if b_vec is None:
        return None
    if result.epochs[-1] < coast_duration + 0.999 * return_window_s:
        return None  # propagation aborted early (e.g. a trial passing through the point-mass Earth)
    return_leg = PropagationResult(
        epochs=result.epochs[s1:e1], states=result.states[s1:e1], final_state=result.final_state,
        epoch_offset=result.epoch_offset, propagator=result.propagator,
    )
    # Skip periapsides still inside the Moon's neighbourhood of the sample
    # epoch: only the return leg's own perigee is the entry condition.
    perigees = [ev for ev in find_periapsides(return_leg, refine=True)
                if ev.epoch > coast_duration + 0.5 * 86400.0]
    if not perigees:
        return None
    perigee = perigees[0]
    v_inf = float(np.linalg.norm(v_inf_vec))
    return {
        "b_vec": b_vec, "v_inf_vec": v_inf_vec,
        "periapsis_m": float(periapsis_distance_m(np.linalg.norm(b_vec), v_inf, mu_moon)),
        "perigee_epoch": float(perigee.epoch),
        "perigee_m": float(perigee.value) * 1000.0,
        "perigee_state": np.asarray(perigee.state, dtype=float),
        "propagation": result,
    }


def target_free_return_entry_corridor(
    r0,
    v0_pre_burn,
    initial_epoch,
    dv_guess,
    coast_duration_guess,
    target_periapsis_km,
    reference_epoch_ephemeris_seconds,
    target_entry_fpa_deg=LUNAR_RETURN_ENTRY_FPA_DEG,
    entry_corridor_deg=LUNAR_RETURN_ENTRY_CORRIDOR_DEG,
    earth_radius_km=EARTH_RADIUS_KM,
    entry_interface_altitude_km=EARTH_ENTRY_INTERFACE_ALTITUDE_KM,
    b_plane_angle_guess_deg=None,
    return_window_s=6.0 * 86400.0,
    perturbing_bodies=("Earth", "Moon", "Sun"),
    step_size=600.0,
    maxiter=25,
    periapsis_tol_km=50.0,
    perigee_tol_km=2.0,
    fd_step_m_s=0.01,
    max_step_m_s=25.0,
    b_plane_maxiter=6,
    b_plane_outer_rounds=2,
    b_plane_tol_km=200.0,
    report_path=None,
) -> FreeReturnTargetingResult:
    """Free-return TLI targeting that jointly controls the lunar B-plane aim
    point AND the Earth return -- the return perigee and so the inertial
    entry flight-path angle at the entry interface -- instead of leaving the
    entry angle incidental (target_lunar_flyby only targets a flyby
    distance; whatever the return leg does afterwards is luck).

    Three stages:

    1. **Seed** (analytic, ~free): a patched-conic sweep of the B-plane
       angle at ``target_periapsis_km`` (_patched_conic_b_plane_seed) picks
       the aim point whose two-body return conic best matches the corridor,
       unless ``b_plane_angle_guess_deg`` is given.
    2. **B-plane targeting** (existing target_lunar_b_plane, unmodified):
       steer the high-fidelity trajectory onto that (periapsis, angle) aim
       point. This moves the flyby from "whatever direction the guess
       happened to miss in" to the right side of the Moon, close enough to a
       free return for stage 3's linearization to hold.
    3. **Joint Newton** (high fidelity, TudatPy): 3 unknowns (TLI delta-v),
       2 equations -- lunar periapsis (from the B-plane state at the fixed
       sample epoch, a two-body invariant) = ``target_periapsis_km``, and
       the root-found vacuum return perigee = the radius that crosses the
       entry interface at ``target_entry_fpa_deg``
       (perigee_radius_for_entry_angle_m, re-evaluated at each trial's own
       return energy). The minimum-norm Gauss-Newton step (pseudo-inverse)
       uses the spare degree of freedom to stay as close as possible to the
       current burn; same capped, backtracking damping as the other
       targeters here. A vacuum perigee rather than the entry angle itself
       is the residual because it is defined (and smooth) for every trial,
       including ones that miss the atmosphere entirely.

    The result is then **verified** by a separate TudatPy propagation from
    TLI through the entry interface: the reported entry flight-path angle
    is the one measured on that propagation at the root-found downward
    crossing of ``entry_interface_altitude_km`` (find_altitude_crossing),
    not the conic prediction, and ``in_corridor`` checks it against
    ``entry_corridor_deg`` = (steepest, shallowest).

    ``step_size`` is the adaptive DOP853 integrator's *maximum* step (the
    error control at VERIFY_RTOL is what sets accuracy), so the coarser
    600 s default here only cuts the saved-history size and run time.
    ``fd_step_m_s`` is small because the return perigee is extremely
    sensitive to the TLI burn (order 10^3-10^4 km per m/s through the lunar
    flyby): a 1 m/s difference step would be far outside the linear range.

    ``b_plane_maxiter`` / ``b_plane_outer_rounds`` / ``b_plane_tol_km`` cap
    stage 2: it only has to put the flyby on the right side of the Moon
    (an aim point within a few hundred to ~1,000 km is plenty for stage 3,
    which re-solves the periapsis exactly anyway), and its fixed-time Newton
    tends to creep rather than converge once the remaining correction is
    mostly in the B-plane angle -- letting it run to its own 50 km default
    tolerance roughly doubles the wall time for no gain in the final answer.
    """
    dv_guess = np.asarray(dv_guess, dtype=float)
    mu_moon = body_gravitational_parameter("Moon")
    mu_earth = body_gravitational_parameter("Earth")
    entry_radius_m = (earth_radius_km + entry_interface_altitude_km) * 1000.0
    nominal_target_perigee_m = perigee_radius_for_entry_angle_m(
        0.0, entry_radius_m, target_entry_fpa_deg, mu_earth)  # parabolic-return estimate, seed fallback only

    # ---- Stage 1: analytic B-plane angle seed --------------------------------
    seed_angle_deg = b_plane_angle_guess_deg
    seed_perigee_m = None
    if seed_angle_deg is None:
        coast_seed = _refine_coast_duration(
            r0, v0_pre_burn, initial_epoch, dv_guess, coast_duration_guess,
            reference_epoch_ephemeris_seconds, perturbing_bodies, step_size,
        )
        r_rel, v_rel, _ = _relative_state_m(
            r0, v0_pre_burn, initial_epoch, dv_guess, coast_seed,
            reference_epoch_ephemeris_seconds, perturbing_bodies, step_size,
        )
        seed_angle_deg, seed_perigee_m = _patched_conic_b_plane_seed(
            r_rel, v_rel, reference_epoch_ephemeris_seconds + initial_epoch + coast_seed,
            target_periapsis_km, nominal_target_perigee_m, mu_moon, mu_earth,
            entry_radius_m, target_entry_fpa_deg,
        )
        coast_duration_guess = coast_seed
        if seed_angle_deg is None:
            raise ValueError(
                "dv_guess's coasted trajectory is not hyperbolic near the Moon -- cannot seed a "
                "B-plane aim point; adjust dv_guess/coast_duration_guess so the guess passes the Moon."
            )

    # ---- Stage 2: existing B-plane targeter onto the seeded aim point --------
    b_plane_seed = target_lunar_b_plane(
        r0, v0_pre_burn, initial_epoch, dv_guess, coast_duration_guess,
        target_periapsis_km, seed_angle_deg, reference_epoch_ephemeris_seconds,
        mu_moon=mu_moon, perturbing_bodies=perturbing_bodies, step_size=step_size,
        maxiter=b_plane_maxiter, outer_rounds=b_plane_outer_rounds, tol_km=b_plane_tol_km,
    )
    coast_duration = b_plane_seed.coast_duration

    # ---- Stage 3: joint lunar-periapsis + return-perigee Newton --------------
    def evaluate(dv_trial):
        out = _flyby_and_return(
            r0, v0_pre_burn, initial_epoch, dv_trial, coast_duration, return_window_s,
            reference_epoch_ephemeris_seconds, perturbing_bodies, step_size, mu_moon,
        )
        if out is None:
            return None, None
        ps = out["perigee_state"]
        eps = 0.5 * float(ps[3:] @ ps[3:]) - mu_earth / float(np.linalg.norm(ps[:3]))
        if eps + mu_earth / entry_radius_m <= 0.0:
            return None, None
        target_perigee_m = perigee_radius_for_entry_angle_m(eps, entry_radius_m, target_entry_fpa_deg, mu_earth)
        out["target_perigee_m"] = target_perigee_m
        res_km = np.array([
            out["periapsis_m"] - target_periapsis_km * 1000.0,
            out["perigee_m"] - target_perigee_m,
        ]) / 1000.0
        return res_km, out

    def merit(res_km):
        # Scale each residual by its own tolerance so neither dominates.
        return float(np.linalg.norm(res_km / np.array([periapsis_tol_km, perigee_tol_km])))

    dv = np.asarray(b_plane_seed.delta_v, dtype=float).copy()
    residual_history_km = []
    converged = False
    iterations = 0
    stall_count = 0
    res, cur = evaluate(dv)
    if res is None:
        raise RuntimeError(
            "B-plane-targeted trajectory does not return to an Earth perigee within return_window_s "
            f"(seed angle {seed_angle_deg:.1f} deg) -- try another b_plane_angle_guess_deg or coast time."
        )

    for iterations in range(1, maxiter + 1):
        residual_history_km.append([float(res[0]), float(res[1])])
        if abs(res[0]) < periapsis_tol_km and abs(res[1]) < perigee_tol_km:
            converged = True
            break

        jacobian = np.zeros((2, 3))
        jacobian_ok = True
        for k in range(3):
            perturbed = dv.copy()
            perturbed[k] += fd_step_m_s
            res_k, _ = evaluate(perturbed)
            if res_k is None:
                jacobian_ok = False
                break
            jacobian[:, k] = (res_k - res) / fd_step_m_s
        if not jacobian_ok:
            break

        delta_dv = -np.linalg.pinv(jacobian) @ res  # minimum-norm Gauss-Newton step
        step_scale = min(1.0, max_step_m_s / max(np.linalg.norm(delta_dv), 1e-12))
        current_merit = merit(res)
        improved = False
        for _ in range(12):
            trial_dv = dv + step_scale * delta_dv
            trial_res, trial_out = evaluate(trial_dv)
            if trial_res is not None and merit(trial_res) < current_merit:
                dv, res, cur = trial_dv, trial_res, trial_out
                improved = True
                break
            step_scale *= 0.5
        if improved:
            stall_count = 0
        else:
            stall_count += 1
            if stall_count >= 2:
                break

    # ---- Verification: independent TudatPy propagation through entry --------
    verify_duration = cur["perigee_epoch"] + 0.25 * 86400.0
    verified = propagate_multi_arc(
        r0, v0_pre_burn, reference_epoch_ephemeris_seconds + initial_epoch,
        arcs=[{"type": "impulsive_burn", "delta_v": dv}, {"type": "coast", "duration": verify_duration}],
        perturbing_bodies=perturbing_bodies, step_size=step_size,
    )
    perilune = find_closest_approach(verified, reference_epoch_ephemeris_seconds + initial_epoch, body_name="Moon")
    after_flyby = PropagationResult(
        epochs=verified.epochs[verified.epochs > perilune.epoch + 0.5 * 86400.0],
        states=verified.states[verified.epochs > perilune.epoch + 0.5 * 86400.0],
        final_state=verified.final_state, epoch_offset=verified.epoch_offset, propagator=verified.propagator,
    )
    perigees = find_periapsides(after_flyby, refine=True)
    return_perigee_km = float(perigees[0].value) if perigees else float(np.linalg.norm(cur["perigee_state"][:3]) / 1000.0)
    entry = find_altitude_crossing(after_flyby, earth_radius_km, entry_interface_altitude_km)
    entry_fpa = entry_speed = None
    if entry is not None:
        entry_fpa = flight_path_angle_deg(entry.spacecraft_state)
        entry_speed = float(np.linalg.norm(entry.spacecraft_state[3:]))
    steepest, shallowest = entry_corridor_deg
    in_corridor = entry_fpa is not None and steepest <= entry_fpa <= shallowest

    b_angle = b_plane_angle_deg(cur["b_vec"], cur["v_inf_vec"])
    report = _report.build_report(
        kind="lunar-free-return-entry-corridor", label=f"ref-et-{float(reference_epoch_ephemeris_seconds):.3f}",
        propagations=[propagation_report_section(verified, "tli-flyby-entry")],
        residuals=[{
            "name": "entry_flight_path_angle_in_corridor",
            "reference": "TudatPy-propagated inertial flight-path angle at the entry-interface crossing "
                         "vs the safe lunar-return entry corridor",
            "position_residual_m": None, "velocity_residual_m_s": None,
            "value_deg": entry_fpa,
            "tolerance": {"corridor_deg": [float(steepest), float(shallowest)]},
            "within_tolerance": bool(in_corridor),
        }],
        inputs={"pre_burn_state": np.concatenate([np.asarray(r0, dtype=float), np.asarray(v0_pre_burn, dtype=float)]),
                "initial_epoch": float(initial_epoch), "dv_guess_m_s": dv_guess,
                "coast_duration_guess_s": float(coast_duration_guess),
                "target_periapsis_km": float(target_periapsis_km),
                "target_entry_fpa_deg": float(target_entry_fpa_deg),
                "entry_interface_altitude_km": float(entry_interface_altitude_km),
                "earth_radius_km": float(earth_radius_km),
                "reference_epoch_ephemeris_s": float(reference_epoch_ephemeris_seconds), "step_size_s": float(step_size)},
        targets={"converged": converged, "in_corridor": bool(in_corridor), "iterations": iterations,
                 "delta_v_m_s": dv, "coast_duration_s": coast_duration,
                 "seed_b_plane_angle_deg": float(seed_angle_deg),
                 "seed_patched_conic_perigee_km": None if seed_perigee_m is None else seed_perigee_m / 1000.0,
                 "b_plane_stage_converged": bool(b_plane_seed.converged),
                 "b_plane_angle_deg": float(b_angle),
                 "perilune_epoch_s": float(perilune.epoch), "perilune_distance_km": float(perilune.distance_km),
                 "return_perigee_km": return_perigee_km,
                 "target_return_perigee_km": cur["target_perigee_m"] / 1000.0,
                 "entry_epoch_s": None if entry is None else float(entry.epoch),
                 "entry_speed_m_s": entry_speed, "entry_flight_path_angle_deg": entry_fpa,
                 "residual_history_km": residual_history_km},
    )
    _report.finalize_report(report, report_path)
    return FreeReturnTargetingResult(
        delta_v=dv, converged=converged, in_corridor=bool(in_corridor), iterations=iterations,
        coast_duration=coast_duration, perilune_epoch=float(perilune.epoch),
        perilune_distance_km=float(perilune.distance_km), b_plane_angle_deg=float(b_angle),
        return_perigee_km=return_perigee_km, target_return_perigee_km=cur["target_perigee_m"] / 1000.0,
        entry_epoch=None if entry is None else float(entry.epoch),
        entry_state=None if entry is None else np.asarray(entry.spacecraft_state, dtype=float),
        entry_speed_m_s=entry_speed, entry_flight_path_angle_deg=entry_fpa,
        b_plane_seed=b_plane_seed, residual_history_km=residual_history_km, report=report,
    )
