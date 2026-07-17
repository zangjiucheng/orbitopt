"""Search for a genuine (zero-correction-burn) lunar free-return geometry:
a TLI burn whose post-flyby return leg naturally reaches Earth's
atmospheric entry interface, not just a lunar flyby at the right distance.

target_lunar_flyby (differential_correction.py) only targets the lunar
flyby distance, "in whatever direction the initial guess was already
missing by" -- it has no notion of the *return* leg at all, so whether the
resulting trajectory happens to swing back close to Earth is incidental,
not something that function controls (see its own docstring). The
departure geometry's B-plane angle at the Moon -- which side of the Moon
the flyby passes on, holding the flyby *distance* (and so the Moon-safety
constraint) fixed -- is the degree of freedom that actually steers the
return leg. This module treats target_lunar_flyby as a proven, unmodified
inner solve (redesigning its own Newton iteration around B-plane
coordinates directly was tried and reverted earlier in this project's
history -- real convergence regressions, not a style preference) and
layers a B-plane-informed outer search on top of it.

The outer search's free variable is the caller's own parking-orbit phase
angle (theta), not B-plane coordinates directly -- varying theta changes
the Lambert-seeded initial guess's direction, which target_lunar_flyby
then preserves through its own correction, which in turn determines the
B-plane angle of the *converged* trajectory. Reporting that angle (via
orbitopt.verify.b_plane) is what makes this a B-plane-*informed* search,
even though theta -- not B-plane angle -- is what's actually being swept:
the Lambert/parking-orbit machinery only knows how to construct a guess
from theta, so driving B-plane angle directly would need re-deriving that
construction in reverse, a materially bigger (and, given the above,
higher-risk) undertaking than reusing what already exists reliably.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from orbitopt.lambert.cpu import solve_lambert_single
from orbitopt.verify.b_plane import b_plane_angle_deg, b_plane_state
from orbitopt.verify.differential_correction import target_lunar_flyby
from orbitopt.verify.tudat_propagate import (
    PropagationResult,
    body_gravitational_parameter,
    body_position_at_absolute_epoch,
    body_velocity_at_absolute_epoch,
    find_altitude_crossing,
    propagate_multi_arc,
)


@dataclass
class FreeReturnCandidate:
    theta_deg: float
    converged: bool
    dv_m_s: float | None = None
    delta_v: np.ndarray | None = None
    coast_duration_s: float | None = None
    periapsis_km: float | None = None
    b_plane_angle_deg: float | None = None
    return_perigee_altitude_km: float | None = None  # None if no return leg found in the search window
    entry_day: float | None = None  # set only if the return leg actually reaches the entry interface
    rejection_reason: str | None = None
    propagation: PropagationResult | None = field(default=None, repr=False)

    @property
    def is_genuine_free_return(self) -> bool:
        return self.converged and self.entry_day is not None

    @property
    def residual_km(self) -> float | None:
        """Signed distance (km) from the entry interface -- positive means
        the return leg fell short (didn't reach entry), 0 or negative means
        it reached (or would have overshot past) the entry interface.
        Secant search below drives this toward 0."""
        if self.return_perigee_altitude_km is None:
            return None
        return self.return_perigee_altitude_km


def evaluate_theta(
    theta_deg, parking_orbit_fn, r_moon_arrival, coast_days_guess,
    target_periapsis_km, reference_et, mu_earth, mu_moon,
    entry_altitude_km, earth_radius_km, search_window_days=14.0,
    step_size_s=120.0, max_dv_m_s=5000.0,
) -> FreeReturnCandidate:
    """Evaluate one candidate parking-orbit phase angle: build its
    Lambert-seeded TLI guess, refine with target_lunar_flyby (unmodified),
    propagate the full post-flyby arc, and characterize both the lunar
    B-plane geometry and the Earth return leg.

    step_size_s defaults coarser than mission_timeline's own 60s export
    step -- fine enough to rank candidates during a search that evaluates
    many of them, not meant to be the final exported trajectory's fidelity
    (re-run the winning theta at the finer step for that, same as any other
    propagate_multi_arc caller with this tradeoff).
    """
    r0, v0 = parking_orbit_fn(theta_deg)
    try:
        (v1, _v2), = solve_lambert_single(r0, r_moon_arrival, coast_days_guess * 86400.0, mu=mu_earth, max_revs=0)[:1]
    except Exception as exc:  # noqa: BLE001 -- pykep's Lambert solver can hit real geometric degeneracies (see gpu_batch.py/cpu.py callers elsewhere in this project); a failure here is a legitimate "this theta doesn't work", not a bug
        return FreeReturnCandidate(theta_deg=theta_deg, converged=False, rejection_reason=f"lambert failed: {exc}")
    dv_guess = v1 - v0

    targeting = target_lunar_flyby(
        r0=r0, v0_pre_burn=v0, initial_epoch=0.0,
        dv_guess=dv_guess, coast_duration_guess=coast_days_guess * 86400.0,
        target_distance_km=target_periapsis_km,
        reference_epoch_ephemeris_seconds=reference_et,
    )
    if not targeting.converged:
        return FreeReturnCandidate(theta_deg=theta_deg, converged=False, rejection_reason="differential correction did not converge")

    dv_final = float(np.linalg.norm(targeting.delta_v))
    if dv_final > max_dv_m_s:
        # A real, repeatedly-observed failure mode of this differential
        # corrector for certain geometries: it converges, but to a
        # multi-km/s "wrong branch" solution with no resemblance to a
        # sensible direct-injection TLI burn (confirmed by direct
        # inspection during this search's own development -- 6600-8700 m/s
        # candidates, vs ~3800-4800 m/s for the physically sane family).
        # Reject rather than let a numerically-converged-but-unrealistic
        # result pollute the search.
        return FreeReturnCandidate(theta_deg=theta_deg, converged=False, dv_m_s=dv_final,
                                    rejection_reason=f"converged to implausible dv={dv_final:.0f}m/s (wrong branch)")

    full = propagate_multi_arc(
        r0, v0, 0.0,
        arcs=[
            {"type": "impulsive_burn", "delta_v": targeting.delta_v},
            {"type": "coast", "duration": search_window_days * 86400.0},
        ],
        perturbing_bodies=("Earth", "Moon", "Sun"), step_size=step_size_s,
    )

    # B-plane geometry at the lunar encounter, for reporting/diagnosis (see
    # this module's docstring for why this is informative rather than the
    # search's own control variable). full.states is Earth-centered, so
    # both the spacecraft and the Moon need converting to Moon-relative.
    encounter_epoch = reference_et + targeting.coast_duration
    idx_encounter = int(np.argmin(np.abs(full.epochs - targeting.coast_duration)))
    moon_r_at_encounter = body_position_at_absolute_epoch("Moon", encounter_epoch)
    moon_v_at_encounter = body_velocity_at_absolute_epoch("Moon", encounter_epoch)
    mu_moon_val = mu_moon if mu_moon is not None else body_gravitational_parameter("Moon")
    r_rel_moon = full.states[idx_encounter, :3] - moon_r_at_encounter
    v_rel_moon = full.states[idx_encounter, 3:] - moon_v_at_encounter
    b_vec, v_inf_vec = b_plane_state(r_rel_moon, v_rel_moon, mu_moon_val)
    angle = b_plane_angle_deg(b_vec, v_inf_vec) if b_vec is not None else None

    entry = find_altitude_crossing(full, central_body_radius_km=earth_radius_km, threshold_altitude_km=entry_altitude_km)
    radii_km = np.linalg.norm(full.states[:, :3], axis=1) / 1000.0
    after_perilune = full.epochs > targeting.coast_duration
    return_perigee_alt_km = None
    if after_perilune.any():
        idx = int(np.argmin(np.where(after_perilune, radii_km, np.inf)))
        return_perigee_alt_km = float(radii_km[idx] - earth_radius_km)

    entry_day = None
    if entry is not None and entry.epoch > targeting.coast_duration:
        entry_day = float(entry.epoch / 86400.0)
        return_perigee_alt_km = min(return_perigee_alt_km, entry_altitude_km) if return_perigee_alt_km is not None else entry_altitude_km

    return FreeReturnCandidate(
        theta_deg=theta_deg, converged=True, dv_m_s=dv_final, delta_v=targeting.delta_v,
        coast_duration_s=targeting.coast_duration, periapsis_km=targeting.final_distance_km,
        b_plane_angle_deg=angle, return_perigee_altitude_km=return_perigee_alt_km,
        entry_day=entry_day, propagation=full,
    )


def search_free_return_theta(
    parking_orbit_fn, r_moon_arrival, coast_days_guess, target_periapsis_km,
    reference_et, mu_earth, entry_altitude_km, earth_radius_km,
    theta_seeds_deg=(150.0, 134.0), max_evaluations=10, tol_km=50.0,
    mu_moon=None, log=None,
):
    """Secant search on theta, driving the return leg's closest-Earth-
    approach altitude toward entry_altitude_km, informed (see this
    module's docstring) but not driven by B-plane analysis. Infeasible
    trial points (non-convergence or an implausible "wrong branch" delta-v
    -- both real, observed failure modes of the inner solve for certain
    geometries) are skipped with a small perturbation rather than treated
    as a hard search failure, since this objective's feasible region isn't
    contiguous.

    Returns the best FreeReturnCandidate found (lowest surviving residual;
    stops early once one reaches genuine entry) -- check
    .is_genuine_free_return before assuming the search fully closed the
    gap, since a truncated budget may not find one.
    """
    def _log(msg):
        if log is not None:
            log(msg)

    evaluated: dict[float, FreeReturnCandidate] = {}

    def eval_theta(theta_deg):
        theta_deg = round(float(theta_deg), 3)
        if theta_deg in evaluated:
            return evaluated[theta_deg]
        cand = evaluate_theta(
            theta_deg, parking_orbit_fn, r_moon_arrival, coast_days_guess,
            target_periapsis_km, reference_et, mu_earth, mu_moon,
            entry_altitude_km, earth_radius_km,
        )
        evaluated[theta_deg] = cand
        status = (
            f"entry at t={cand.entry_day:.2f}d" if cand.entry_day is not None else
            f"return_alt={cand.return_perigee_altitude_km:.0f}km" if cand.return_perigee_altitude_km is not None else
            (cand.rejection_reason or "no return leg")
        )
        _log(f"theta={theta_deg:7.2f}deg  converged={cand.converged}  {status}"
             + (f"  dv={cand.dv_m_s:.0f}m/s  B-angle={cand.b_plane_angle_deg:.1f}deg" if cand.converged else ""))
        return cand

    best = None

    def consider(cand):
        nonlocal best
        if not cand.converged or cand.return_perigee_altitude_km is None:
            return
        if best is None or cand.return_perigee_altitude_km < best.return_perigee_altitude_km:
            best = cand

    seeds = [eval_theta(t) for t in theta_seeds_deg]
    for s in seeds:
        consider(s)

    feasible = [s for s in seeds if s.converged and s.return_perigee_altitude_km is not None]
    if len(feasible) < 2:
        # Can't form a secant without two feasible points -- fall back to a
        # local grid around whichever seed(s) did work.
        base = feasible[0].theta_deg if feasible else theta_seeds_deg[0]
        for offset in (-4.0, -2.0, 2.0, 4.0, -6.0, 6.0):
            if len(evaluated) >= max_evaluations:
                break
            cand = eval_theta(base + offset)
            consider(cand)
            if cand.is_genuine_free_return:
                return cand
        return best

    x0, x1 = feasible[-2].theta_deg, feasible[-1].theta_deg
    f0, f1 = feasible[-2].return_perigee_altitude_km, feasible[-1].return_perigee_altitude_km

    while len(evaluated) < max_evaluations:
        if f1 == f0:
            break  # no slope information left to extrapolate from
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        x2 = max(min(x2, x1 + 20.0), x1 - 20.0)  # cap the secant step -- this objective is not smooth/global

        cand = None
        for perturb in (0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0):
            if len(evaluated) >= max_evaluations:
                break
            trial = eval_theta(x2 + perturb)
            if trial.converged and trial.return_perigee_altitude_km is not None:
                cand = trial
                break
        if cand is None:
            break  # the whole neighborhood around the secant estimate is infeasible -- stop rather than spin

        consider(cand)
        if cand.is_genuine_free_return or (best is not None and best.residual_km is not None and best.residual_km <= tol_km):
            break

        x0, f0 = x1, f1
        x1, f1 = cand.theta_deg, cand.return_perigee_altitude_km

    return best
