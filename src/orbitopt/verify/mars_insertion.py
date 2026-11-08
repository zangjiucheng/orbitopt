"""High-fidelity (tudatpy) targeting for the Earth->Mars transfer's two real
burns beyond the idealized Lambert departure: a trajectory-correction
maneuver (TCM) that closes the gap between the idealized two-body departure
and reality, and the Mars-orbit-insertion (MOI) burn that captures into a
target orbit once there.

Why a TCM is needed at all, not just an idealization footnote: a Lambert arc
solved from the *idealized* departure state (Earth's position at t0, plus the
optimizer's v-infinity) does NOT coincide with the *real* departure state --
the spacecraft actually leaves Earth's sphere of influence a few days later,
by which time Earth's own heliocentric velocity has rotated a couple of
degrees, and the real departure trajectory starts several thousand km/s off
the idealized one in velocity space. Over a ~300-day transfer this is not a
small effect: propagating the real post-departure state under Earth's own
third-body perturbation (measured, not assumed) misses Mars by more than two
million km, an order of magnitude past Mars's own sphere of influence
(~577,000 km). This is exactly the situation every real interplanetary
mission flies a TCM for -- so this module targets one, the same way
verify.differential_correction targets a TLI burn for the lunar case, rather
than hand-waving the idealized Lambert arc as if it were flight-ready.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from tudatpy.astro import element_conversion

from orbitopt.verify import report as _report
from orbitopt.verify.b_plane import (
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
    propagation_report_section,
)

CRUISE_PERTURBATIONS = dict(perturbing_bodies=("Sun", "Earth", "Mars", "Jupiter"))


@dataclass
class TcmTargetingResult:
    delta_v: np.ndarray  # (3,) m/s, converged TCM burn vector
    converged: bool
    iterations: int
    coast_duration: float  # seconds, the fixed time-of-flight targeted
    final_state: np.ndarray  # (6,) heliocentric m, m/s at coast_duration
    final_miss_km: float  # distance from Mars center at the fixed coast_duration (not the periapsis)
    residual_history_km: list = field(default_factory=list)
    # Periapsis of the osculating Mars-relative hyperbola at coast_duration --
    # what the B-plane stage actually targets (target_periapsis_km).
    predicted_periapsis_km: float = float("nan")
    b_plane_residual_history_km: list = field(default_factory=list)
    report: dict | None = None  # machine-readable verification report (orbitopt.verify.report)


def _miss_km(sc_r_helio, sc_v_helio, initial_epoch_et, dv_trial, coast_duration, step_size):
    result = propagate_multi_arc(
        sc_r_helio, sc_v_helio, initial_epoch_et,
        arcs=[{"type": "impulsive_burn", "delta_v": dv_trial}, {"type": "coast", "duration": coast_duration}],
        central_body="Sun", step_size=step_size, **CRUISE_PERTURBATIONS,
    )
    arrival_epoch_et = initial_epoch_et + float(result.epochs[-1])
    mars_r = body_position_at_absolute_epoch("Mars", arrival_epoch_et, central_body="Sun")
    return result.final_state[:3] - mars_r, result, arrival_epoch_et


def _relative_state(sc_r_helio, sc_v_helio, initial_epoch_et, dv_trial, coast_duration, step_size):
    """Mars-relative (r, v) at the end of the burn + coast."""
    _, result, arrival_epoch_et = _miss_km(sc_r_helio, sc_v_helio, initial_epoch_et, dv_trial, coast_duration, step_size)
    mars_r = body_position_at_absolute_epoch("Mars", arrival_epoch_et, central_body="Sun")
    mars_v = body_velocity_at_absolute_epoch("Mars", arrival_epoch_et, central_body="Sun")
    return result.final_state[:3] - mars_r, result.final_state[3:] - mars_v, result


def _refine_b_plane(sc_r_helio, sc_v_helio, departure_epoch_et, dv, coast_duration, step_size,
                    target_periapsis_km, tol_km, maxiter, fd_step_m_s, max_step_m_s):
    """Second targeting stage: Newton on the osculating B-plane vector so the
    *periapsis* of the Mars-relative hyperbola -- not the distance at a fixed
    epoch -- equals ``target_periapsis_km``.

    Why this stage exists: the first stage aims the spacecraft at a point
    ``target_periapsis_km`` from Mars's center at a fixed time. Gravity
    focusing bends the approach hyperbola inward, so the true periapsis ends
    up well below that aim distance (measured: 3,889.5 km aimed -> 3,642 km
    flown for the 2026 window, and 3,341 km -- under Mars's 3,389.5 km
    surface -- for the 2020 window). Targeting the impact parameter
    |B| = sqrt(rp (rp + 2 mu / v_inf^2)) instead fixes the periapsis itself.
    Square 3x3 Newton system: (B.T, B.R) plus the along-asymptote position
    S.r at the fixed epoch (a time-of-arrival constraint). The B direction
    of the first-stage solution is kept (same "rescale the existing miss
    direction" convention as the first stage).

    Returns (dv, converged, predicted_periapsis_km, residual_history_km).
    """
    mu = body_gravitational_parameter("Mars")
    target_rp_m = target_periapsis_km * 1000.0

    def evaluate(dv_trial):
        """(b_vec, v_inf_vec, r_rel, rp_m) at the fixed coast end, or Nones."""
        r_rel, v_rel, _ = _relative_state(sc_r_helio, sc_v_helio, departure_epoch_et, dv_trial, coast_duration, step_size)
        b, v_inf = b_plane_state(r_rel, v_rel, mu)
        if b is None:
            return None, None, r_rel, float("nan")
        rp = float(periapsis_distance_m(float(np.linalg.norm(b)), float(np.linalg.norm(v_inf)), mu))
        return b, v_inf, r_rel, rp

    b0, v_inf0, r0, rp0 = evaluate(dv)
    if b0 is None:
        return dv, False, float("nan"), []
    # Frozen targeting frame from the first-stage solution: S along v_inf,
    # (T, R) spanning the B-plane. The aim point keeps the current B
    # direction; the third equation pins S . r_rel (the along-asymptote
    # position at the fixed epoch) to its starting value, i.e. holds the
    # arrival time. Without it, B (2 components) leaves one dv direction
    # free, and the min-norm step drifted the arrival time by days -- the
    # fixed-epoch state then sat ~600,000 km out and the osculating B
    # stopped describing the real encounter (measured, 2020 window).
    s_hat = v_inf0 / float(np.linalg.norm(v_inf0))
    t_hat, r_hat = perpendicular_basis(s_hat)
    b_dir = np.array([b0 @ t_hat, b0 @ r_hat])
    b_dir /= float(np.linalg.norm(b_dir))
    s0 = float(r0 @ s_hat)

    def residual(b, v_inf, r_rel):
        b_mag_target = b_plane_magnitude_for_periapsis(target_rp_m, float(np.linalg.norm(v_inf)), mu)
        bt, br = b_mag_target * b_dir
        return np.array([b @ t_hat - bt, b @ r_hat - br, r_rel @ s_hat - s0])

    history, converged, rp = [], False, rp0
    for _ in range(maxiter):
        b, v_inf, r_rel, rp = evaluate(dv)
        err_km = abs(rp - target_rp_m) / 1000.0
        history.append(err_km)
        if err_km < tol_km:
            converged = True
            break
        res = residual(b, v_inf, r_rel)
        jac = np.zeros((3, 3))
        for k in range(3):
            bumped = dv.copy()
            bumped[k] += fd_step_m_s
            bb, bv, br_rel, _ = evaluate(bumped)
            if bb is None:
                return dv, False, rp / 1000.0, history
            jac[:, k] = (residual(bb, bv, br_rel) - res) / fd_step_m_s
        try:
            step = np.linalg.solve(jac, -res)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(jac, -res, rcond=None)[0]
        scale = min(1.0, max_step_m_s / max(float(np.linalg.norm(step)), 1e-12))
        res_norm = float(np.linalg.norm(res))
        for _ in range(10):
            trial = dv + scale * step
            tb, tv, tr, _ = evaluate(trial)
            if tb is not None and float(np.linalg.norm(residual(tb, tv, tr))) < res_norm:
                dv = trial
                break
            scale *= 0.5
        else:
            break  # line search failed: stop with the best dv so far
    if not converged:
        _, _, _, rp = evaluate(dv)
        converged = abs(rp - target_rp_m) / 1000.0 < tol_km
    return dv, converged, rp / 1000.0, history


def _refine_coast_duration(sc_r_helio, sc_v_helio, departure_epoch_et, dv, coast_duration_guess, step_size):
    window = coast_duration_guess + 20.0 * 86400.0
    probe = propagate_multi_arc(
        sc_r_helio, sc_v_helio, departure_epoch_et,
        arcs=[{"type": "impulsive_burn", "delta_v": dv}, {"type": "coast", "duration": window}],
        central_body="Sun", step_size=step_size, **CRUISE_PERTURBATIONS,
    )
    closest = find_closest_approach(probe, departure_epoch_et, body_name="Mars", central_body="Sun")
    if abs(closest.epoch - coast_duration_guess) < 0.3 * coast_duration_guess:
        return closest.epoch
    return coast_duration_guess


def target_mars_approach(
    sc_r_helio,
    sc_v_helio,
    departure_epoch_et,
    coast_duration_guess,
    target_periapsis_km,
    step_size=1800.0,
    maxiter=15,
    tol_km=500.0,
    fd_step_m_s=0.5,
    max_step_m_s=300.0,
    outer_rounds=3,
    report_path=None,
    periapsis_tol_km=10.0,
    b_plane_fd_step_m_s=0.01,
    b_plane_max_step_m_s=5.0,
) -> TcmTargetingResult:
    """Fixed-time Newton-Raphson shooting on a trajectory-correction burn
    (fired at ``departure_epoch_et``, i.e. right after Earth-SOI exit) so
    the real, perturbed heliocentric trajectory passes ``target_periapsis_km``
    from Mars's center, in the same miss *direction* the uncorrected
    (dv=0) coast already produces -- same "rescale the existing miss
    direction" simplification as verify.differential_correction.
    target_lunar_flyby, and for the same reason: a good approximation as
    long as the uncorrected miss is already large next to Mars's local
    curvature scale, which it is here (the uncorrected miss is over a
    million km; see the module docstring).

    Wrapped in ``outer_rounds`` of re-refinement (mirrors
    verify.differential_correction.target_lunar_b_plane, for the identical
    reason its own docstring gives): a correction burn shifts *when* the
    trajectory is nearest Mars, so a coast_duration fixed before that
    correction can land Newton on a stale target time -- confirmed in
    practice here, not assumed, by the first cut of this function landing
    within 1000 km of Mars's center at a time 1.6 days after the fixed
    target it was aiming for.

    That fixed-time position stage only gets the trajectory into the right
    neighbourhood; a second B-plane stage (``_refine_b_plane``) then targets
    the periapsis of the approach hyperbola to ``periapsis_tol_km``. The
    result is ``converged`` only if both stages converge.
    """
    dv = np.zeros(3)
    coast_duration = coast_duration_guess

    for _round in range(outer_rounds):
        coast_duration = _refine_coast_duration(sc_r_helio, sc_v_helio, departure_epoch_et, dv, coast_duration, step_size)

        def miss(dv_trial):
            m, _, _ = _miss_km(sc_r_helio, sc_v_helio, departure_epoch_et, dv_trial, coast_duration, step_size)
            return m

        m0 = miss(dv)
        d0_km = float(np.linalg.norm(m0)) / 1000.0
        target_vector_m = m0 * (target_periapsis_km / d0_km) if d0_km > 1e-6 else np.array([target_periapsis_km * 1000.0, 0.0, 0.0])

        residual_history_km = []
        converged = False
        iterations = 0
        stall_count = 0

        for iterations in range(1, maxiter + 1):
            current_miss = miss(dv)
            residual = current_miss - target_vector_m
            residual_km = float(np.linalg.norm(residual)) / 1000.0
            residual_history_km.append(residual_km)

            if residual_km < tol_km:
                converged = True
                break

            jac = np.zeros((3, 3))
            for k in range(3):
                bumped = dv.copy()
                bumped[k] += fd_step_m_s
                jac[:, k] = (miss(bumped) - current_miss) / fd_step_m_s

            try:
                step = np.linalg.solve(jac, -residual)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(jac, -residual, rcond=None)[0]

            scale = min(1.0, max_step_m_s / max(float(np.linalg.norm(step)), 1e-9))
            improved = False
            for _ in range(10):
                trial = dv + scale * step
                trial_km = float(np.linalg.norm(miss(trial) - target_vector_m)) / 1000.0
                if trial_km < residual_km:
                    dv, improved = trial, True
                    break
                scale *= 0.5
            if improved:
                stall_count = 0
            else:
                stall_count += 1
                if stall_count >= 3:
                    break

        if converged:
            break

    position_stage_converged = converged
    predicted_periapsis_km, b_history = float("nan"), []
    if position_stage_converged:
        dv, converged, predicted_periapsis_km, b_history = _refine_b_plane(
            sc_r_helio, sc_v_helio, departure_epoch_et, dv, coast_duration, step_size,
            target_periapsis_km, periapsis_tol_km, maxiter, b_plane_fd_step_m_s, b_plane_max_step_m_s,
        )

    final_m, final_result, arrival_et = _miss_km(sc_r_helio, sc_v_helio, departure_epoch_et, dv, coast_duration, step_size)
    periapsis_ok = bool(np.isfinite(predicted_periapsis_km)
                        and abs(predicted_periapsis_km - target_periapsis_km) <= periapsis_tol_km)
    report = _report.build_report(
        kind="mars-tcm", label=f"departure-et-{float(departure_epoch_et):.3f}",
        propagations=[propagation_report_section(final_result, "cruise")],
        residuals=[{
            "name": "periapsis_vs_target",
            "reference": "periapsis radius of the osculating Mars-relative hyperbola at the fixed "
                         "arrival time vs target_periapsis_km (B-plane stage)",
            "position_residual_m": (abs(predicted_periapsis_km - target_periapsis_km) * 1000.0
                                    if np.isfinite(predicted_periapsis_km) else None),
            "velocity_residual_m_s": None,
            "tolerance": {"position_m": periapsis_tol_km * 1000.0},
            "within_tolerance": periapsis_ok,
        }],
        inputs={"sc_state_helio": np.concatenate([np.asarray(sc_r_helio, dtype=float), np.asarray(sc_v_helio, dtype=float)]),
                "departure_epoch_et": float(departure_epoch_et), "coast_duration_guess_s": float(coast_duration_guess),
                "target_periapsis_km": float(target_periapsis_km), "step_size_s": float(step_size)},
        targets={"converged": converged, "iterations": iterations, "delta_v_m_s": dv,
                 "coast_duration_s": coast_duration, "final_miss_km": float(np.linalg.norm(final_m)) / 1000.0,
                 "residual_history_km": residual_history_km,
                 "position_stage_converged": position_stage_converged,
                 "target_periapsis_km": float(target_periapsis_km),
                 "predicted_periapsis_km": predicted_periapsis_km,
                 "b_plane_residual_history_km": b_history},
    )
    _report.finalize_report(report, report_path)
    return TcmTargetingResult(
        delta_v=dv,
        converged=converged,
        iterations=iterations,
        coast_duration=coast_duration,
        final_state=final_result.final_state,
        final_miss_km=float(np.linalg.norm(final_m)) / 1000.0,
        residual_history_km=residual_history_km,
        predicted_periapsis_km=predicted_periapsis_km,
        b_plane_residual_history_km=b_history,
        report=report,
    )


@dataclass
class PeriapsisResult:
    r_rel: np.ndarray  # (3,) m, Mars-relative, at periapsis
    v_rel: np.ndarray  # (3,) m/s, Mars-relative, at periapsis
    arrival_epoch_et: float
    trajectory_epochs_et: np.ndarray  # (n,) absolute ephemeris seconds, departure_epoch_et..periapsis
    trajectory_positions_m: np.ndarray  # (n, 3) heliocentric m -- central_body="Sun" throughout,
    # so this is already heliocentric with no per-sample body-position addition needed


def locate_mars_periapsis(sc_r_helio, sc_v_helio, departure_epoch_et, dv, coast_duration_guess, bulk_step_size=1800.0, fine_step_size=20.0, fine_half_window=2.0 * 86400.0) -> PeriapsisResult:
    """Precisely locate the true periapsis of a Mars encounter near
    ``coast_duration_guess``. find_closest_approach is only as good as the
    step size feeding it, and a step fine enough for a multi-hundred-day
    cruise is far too coarse right at a close, fast planetary encounter --
    confirmed empirically here, not assumed (a 300s-step search landed on a
    sample still 1000+ km further from Mars than the true minimum, with
    r.v/|r||v| = 0.7 instead of ~0 -- the same class of RK4-step-size-near-
    close-encounter gotcha the lunar-flyby targeter's own docstring
    documents, see README.md). Two propagate_multi_arc calls, not one at a
    uniformly fine step: a bulk coarse-step coast covers the ~300-day
    approach cheaply, then a short fine-step coast over just the final
    ``2 * fine_half_window`` around the encounter resolves the true minimum.

    Also returns the bulk+fine trajectory actually propagated (heliocentric,
    since both stages use central_body="Sun"), truncated to periapsis -- the
    caller should display *this* trajectory for the cruise leg rather than
    separately re-propagating one, so the exported trail is one continuous
    numerical solution with no seam. (An earlier version of this pipeline
    displayed an independently-computed cruise leg at a coarser 3600s step
    glued to this function's own result at the approach; even though both
    used identical physics, the differing step sizes' RK4 truncation error
    left a measured ~90,000 km, ~70 km/s-implied-speed residual at the seam
    -- a visible "teleport" at Mars arrival. Reusing this function's own
    trajectory eliminates that residual by construction, not by loosening a
    tolerance.)
    """
    bulk_duration = max(coast_duration_guess - fine_half_window, 0.0)
    bulk = propagate_multi_arc(
        sc_r_helio, sc_v_helio, departure_epoch_et,
        arcs=[{"type": "impulsive_burn", "delta_v": dv}, {"type": "coast", "duration": bulk_duration}],
        central_body="Sun", step_size=bulk_step_size, **CRUISE_PERTURBATIONS,
    )
    bulk_end_epoch = departure_epoch_et + float(bulk.epochs[-1])

    fine = propagate_multi_arc(
        bulk.final_state[:3], bulk.final_state[3:], bulk_end_epoch,
        arcs=[{"type": "coast", "duration": 2.0 * fine_half_window}],
        central_body="Sun", step_size=fine_step_size, **CRUISE_PERTURBATIONS,
    )
    closest = find_closest_approach(fine, bulk_end_epoch, body_name="Mars", central_body="Sun")
    arrival_epoch_et = bulk_end_epoch + closest.epoch
    mars_r = body_position_at_absolute_epoch("Mars", arrival_epoch_et, central_body="Sun")
    mars_v = body_velocity_at_absolute_epoch("Mars", arrival_epoch_et, central_body="Sun")
    r_rel = closest.spacecraft_state[:3] - mars_r
    v_rel = closest.spacecraft_state[3:] - mars_v

    fine_keep = fine.epochs <= closest.epoch
    trajectory_epochs_et = np.concatenate([
        departure_epoch_et + bulk.epochs,
        bulk_end_epoch + fine.epochs[fine_keep],
    ])
    trajectory_positions_m = np.concatenate([bulk.states[:, :3], fine.states[fine_keep, :3]])

    return PeriapsisResult(
        r_rel=r_rel, v_rel=v_rel, arrival_epoch_et=arrival_epoch_et,
        trajectory_epochs_et=trajectory_epochs_et, trajectory_positions_m=trajectory_positions_m,
    )


@dataclass
class MoiResult:
    delta_v_ms: np.ndarray  # (3,) m/s, Mars-relative
    pre_burn_state: np.ndarray  # (6,) Mars-relative m, m/s
    post_burn_state: np.ndarray  # (6,) Mars-relative m, m/s
    achieved: dict  # osculating a_km/e/i_deg immediately after the burn
    stationkeeping: dict  # drift of a/e/i over a short coast, Sun-perturbed
    report: dict | None = None  # machine-readable verification report (orbitopt.verify.report)


def insert_mars_orbit(
    r_rel_m,
    v_rel_m_s,
    periapsis_km,
    apoapsis_km,
    mu_mars=None,
) -> tuple:
    """Closed-form apsis-preserving capture burn: fired exactly at the
    incoming hyperbola's periapsis (``r_rel_m``/``v_rel_m_s``, Mars-relative,
    assumed already there -- r . v ~ 0), it changes only the *other* apsis,
    from the hyperbola's implicit one at infinity to ``apoapsis_km`` --
    same apsis-preserving shape as viz.geo_raising's GEO-raising burns, just
    a capture (hyperbola -> ellipse) instead of a raise (ellipse -> bigger
    ellipse), and with no inclination change to solve for (nothing here
    constrains Mars orbit inclination the way a ground-station longitude
    constrains GEO). Purely a tangential speed change: vis-viva at the fixed
    periapsis radius, current (hyperbolic) semi-major axis in, target
    (elliptical) semi-major axis out.

    Returns (delta_v_ms, mu_mars) -- the caller propagates/verifies.
    """
    if mu_mars is None:
        mu_mars = body_gravitational_parameter("Mars")
    r = np.asarray(r_rel_m, dtype=float)
    v = np.asarray(v_rel_m_s, dtype=float)
    rp_m = periapsis_km * 1000.0
    ra_m = apoapsis_km * 1000.0
    a_captured = 0.5 * (rp_m + ra_m)

    r_mag = float(np.linalg.norm(r))
    v_hat = v / float(np.linalg.norm(v))
    speed_before = float(np.linalg.norm(v))
    a_before = 1.0 / (2.0 / r_mag - speed_before ** 2 / mu_mars)  # negative (hyperbolic)

    speed_after = np.sqrt(mu_mars * (2.0 / r_mag - 1.0 / a_captured))
    v_after = speed_after * v_hat
    delta_v = v_after - v

    return delta_v, mu_mars, a_before


def verify_mars_orbit_insertion(
    r_rel_m,
    v_rel_m_s,
    periapsis_km,
    apoapsis_km,
    arrival_epoch_et,
    stationkeeping_days=2.0,
    step_size=120.0,
    report_path=None,
) -> MoiResult:
    """Compute the MOI burn (insert_mars_orbit) and confirm the resulting
    orbit actually stays captured (doesn't immediately escape or crash) by
    coasting it forward a couple of days under real Sun-third-body
    perturbation -- the Mars analog of verify.geo_insertion's
    stationkeeping check."""
    delta_v, mu_mars, _a_before = insert_mars_orbit(r_rel_m, v_rel_m_s, periapsis_km, apoapsis_km)
    post_state = np.concatenate([np.asarray(r_rel_m, dtype=float), np.asarray(v_rel_m_s, dtype=float) + delta_v])

    def osculating(state):
        kep = element_conversion.cartesian_to_keplerian(state, mu_mars)
        return {"a_km": float(kep[0]) / 1000.0, "e": float(kep[1]), "i_deg": float(np.degrees(kep[2]))}

    achieved = osculating(post_state)

    sk = propagate_multi_arc(
        post_state[:3], post_state[3:], arrival_epoch_et,
        arcs=[{"type": "coast", "duration": stationkeeping_days * 86400.0}],
        central_body="Mars", perturbing_bodies=("Mars", "Sun"), step_size=step_size,
    )
    sk_end = osculating(sk.final_state)
    stationkeeping = {
        "days": stationkeeping_days,
        "da_km": sk_end["a_km"] - achieved["a_km"],
        "de": sk_end["e"] - achieved["e"],
        "di_deg": sk_end["i_deg"] - achieved["i_deg"],
    }

    report = _report.build_report(
        kind="mars-orbit-insertion", label=f"arrival-et-{float(arrival_epoch_et):.3f}",
        # No mu= override: the conservation check must use the GM tudat
        # actually integrated with. SPICE's Mars GM differs from tudat's by
        # ~1.9e-8 (relative), which on an e~0.75 capture orbit shows up as a
        # ~1.2e-7 apparent energy "drift" (the 1/r term swings by 2a/rp - 2a/ra)
        # and failed the 1e-8 tolerance even though the integration was fine
        # (angular momentum, which doesn't involve mu, drifted ~1e-13).
        propagations=[propagation_report_section(sk, "capture-stationkeeping")],
        inputs={"pre_burn_state_mars": np.concatenate([np.asarray(r_rel_m, dtype=float), np.asarray(v_rel_m_s, dtype=float)]),
                "periapsis_km": float(periapsis_km), "apoapsis_km": float(apoapsis_km),
                "arrival_epoch_et": float(arrival_epoch_et), "stationkeeping_days": float(stationkeeping_days),
                "step_size_s": float(step_size)},
        targets={"delta_v_m_s": delta_v, "achieved": achieved, "stationkeeping": stationkeeping,
                 "target_a_km": 0.5 * (float(periapsis_km) + float(apoapsis_km)),
                 "a_error_km": achieved["a_km"] - 0.5 * (float(periapsis_km) + float(apoapsis_km))},
    )
    _report.finalize_report(report, report_path)

    return MoiResult(
        delta_v_ms=delta_v,
        pre_burn_state=np.concatenate([np.asarray(r_rel_m, dtype=float), np.asarray(v_rel_m_s, dtype=float)]),
        post_burn_state=post_state,
        achieved=achieved,
        stationkeeping=stationkeeping,
        report=report,
    )
