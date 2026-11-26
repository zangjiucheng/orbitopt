"""E12-05 (task #395): independent verification of LEO-RDV-001 by a TudatPy
two-spacecraft re-flight.

The plan (E12-03, task #393) is graded on orbitopt's own nonlinear pair model
(:func:`orbitopt.relative.propagation.propagate_pair`: fixed-step RK4, analytic
J2). That verdict could still hide a mis-implemented propagator, so this module
re-flies Target (chief) and Chaser (deputy) in **TudatPy** -- an independently
formulated propagator (C++ core, SPICE astrodynamics stack, adaptive integration,
spherical-harmonic gravity) -- with the plan applied as a pulse sequence
(coast / impulsive delta-v / coast, the arcs
:class:`orbitopt.verify.tudat_propagate.ReusableCoastPropagator` speaks). The
pair is re-flown in **ECLIPJ2000** -- the SPICE kernel pool tudatpy loads
defines no GCRS/ICRS frame at all, and a rejected frame makes tudat abort the
propagation at t=0 *without raising*; guarded). Because a zonal field is
axisymmetric only about the spin axis, the config's GCRS states and the
inertial burns are rotated into ECLIPJ2000 by the J2000.0 obliquity before
flying, and the config's exact GM and J2 are forced onto the tudat
environment through ``central_gravity``. Every graded comparison is a
chief-LVLH quantity or a norm, invariant under that one constant rotation,
so every difference measured is integrator + gravity formulation, never unit
or frame bookkeeping.

Three error sources are quantified **separately** -- never blended into one
``passed`` flag (task #395 acceptance):

``inputs["error_sources"]["numerical_propagation"]``
    TudatPy convergence sweeps over the tolerance axis (adaptive DOP853, four
    decades around the production defaults) and the step axis (legacy fixed
    RK4, step-halving grid), plus the closed-form two-body (Kepler) replay of
    the point-mass re-flight legs. Answers: how much of any discrepancy is
    just integration error.

``inputs["error_sources"]["approximate_model_cw"]``
    The whole mission replayed on the linear CW model (the planner's fast
    model, E12-02) with the same pulse sequence, compared against the
    nonlinear main model at the acceptance quantities. This is the price of
    the approximation, measured.

``inputs["error_sources"]["optimizer_nonconvergence"]``
    The plan's own landing residual against the config's Newton tolerance,
    and delta-v stationarity when the golden-section window is closed to
    +-0.5 s (half the planner's stop criterion).

On top of these three diagnostic lines, the headline acceptance comparison --
hold-point position error, relative speed, minimum approach distance and total
delta-v between the **TudatPy re-flight and the main model** -- is graded as
``model_agreement`` on evidence ``independent_model``. ``historical_telemetry``
appears nowhere: nothing here is compared against flown data.

Cross-environment execution (approach: reuse crossenv). The re-flight needs the
tudatpy interpreter, which the main orbitopt environment does not carry. When
``ORBITOPT_TUDAT_PYTHON`` names one, the re-flight runs as a crossenv **stage**
(runner ``rendezvous_refly``): the tudat-environment subprocess recomputes the
plan and its verdict with this same package (env-parity recheck), re-flies the
pair, and seals its evidence into a digest-pinned exchange envelope
(:mod:`orbitopt.crossenv.envelope`) that the orchestrator re-validates on read.
Running inside a tudat environment executes the same worker in-process.
**No backend, no fake pass**: the independent-model, Kepler and convergence
lines are then absent and recorded ``not_checked`` with reason
``REFLY_BACKEND_MISSING`` (the task-#261 rule); the CW and optimizer lines are
environment-free and still graded.

    from orbitopt.verify.rendezvous_refly import verify_rendezvous_refly
    result = verify_rendezvous_refly(out_dir="runs/e12-05")
    result["report"]["status"]["categories"]            # per-category, tri-state
    result["report"]["inputs"]["error_sources"]         # the separate lines
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

REFLY_REPORT_KIND = "orbitopt.verify.rendezvous-refly"
REFLY_SCHEMA_VERSION = "1.0"
#: verification-report-1.0 ``kind`` of the report this module assembles.
REPORT_KIND = "rendezvous-refly"
REFLY_STAGE_RUNNER = "rendezvous_refly"
REFLY_PAYLOAD_NAME = "refly-evidence"
#: missing-backend marker (same tri-state policy as task #261: absent evidence
#: is never a pass; it is recorded, not scored).
REFLY_BACKEND_MISSING = "REFLY_BACKEND_MISSING"

#: Fixed scan step [s] for the minimum-approach sample scan on both models:
#: the separation minimum of the transfer arc is smooth at this scale (both
#: models save their samples on the identical grid; no interpolation of either
#: trajectory enters the number).
MIN_APPROACH_SCAN_STEP_S = 2.0

#: Tolerance-axis sweep: factors on (VERIFY_RTOL, VERIFY_ATOL); the production
#: default (factor 1.0) is included and the reference run is one decade
#: tighter, mirroring docs' integrator-convergence study (task #242).
SWEEP_TOL_FACTORS = (1000.0, 100.0, 10.0, 1.0)
SWEEP_TOL_REF_FACTOR = 0.1
#: Step-axis sweep (legacy fixed-step RK4, step-halving grid) and reference.
SWEEP_RK4_STEPS_S = (180.0, 60.0, 30.0, 15.0, 7.5)
SWEEP_RK4_REF_STEP_S = 3.75

#: Declared cross-model tolerances between the TudatPy re-flight and the main
#: model, on the four acceptance quantities (task #395). Both models claim the
#: SAME physics (same GM, same J2, same frame), so the band bounds the
#: implementation difference only: adaptive DOP853 against a fixed-step RK4,
#: plus a spherical-harmonic(2,0) field against the analytic J2 acceleration
#: (identical truncated field, independently formulated). Bands carry >= 2x
#: margin over the reference-mission measurements recorded in
#: docs/benchmarks/e12_05_independent_refly.json.
REFLY_VS_MAIN_TOLERANCES = {
    "hold_point_position_error_m": 1.0,   # |pos_err_refly - pos_err_main|
    "relative_speed_m_s": 0.01,           # |rel_speed_refly - rel_speed_main|
    "min_approach_distance_m": 1.0,       # |minapp_refly - minapp_main|
    "total_delta_v_m_s": 1.0e-9,          # the pulse sequence is the SAME plan
    "maneuver_sequence_m_s": 1.0e-9,      # max |element| diff burn epochs/dv
}
#: Kepler closed-form reference of the point-mass re-flight at the leg ends --
#: the pure numerical-propagation error of the tudat run (analytic vs
#: numerical, same two-body model), declared at the production tolerances.
KEPLER_VS_REFLY_TOLERANCES = {
    "position_m": 1.0,
    "velocity_m_s": 0.02,
}
#: Convergence-sweep bounds at the production settings (final-state error of
#: the transfer leg against the tightest run of the same axis): the numerical
#: error bound the verification must demonstrate before the headline
#: comparison is worth anything.
SWEEP_BOUNDS = {
    "tolerance_axis_position_m": 0.5,
    "step_axis_60s_position_m": 50.0,
}
#: CW is an approximation by design -- these are the *declared validity*
#: bounds for using it on this mission's quantities: the planning model's
#: error budget (linearization + the J2 it ignores), declared with >= 2.3x
#: margin over the reference-mission measurements recorded in
#: docs/benchmarks/e12_05_independent_refly.json (253 m / 0.198 m/s / 240 m).
#: Pointedly, the CW arrival-speed miss alone (0.198 m/s) consumes the whole
#: 0.2 m/s gate tolerance -- the planner is right to grade on the nonlinear
#: model (E12-03), and this line documents why.
CW_VS_NONLINEAR_TOLERANCES = {
    "hold_point_position_error_m": 600.0,
    "relative_speed_m_s": 0.50,
    "min_approach_distance_m": 600.0,
}
#: Optimizer lines: the |d(total dv)| allowed when the golden window is
#: closed to +-0.5 s (the landing-residual half uses the config's own
#: ``newton_tol_m``).
OPTIMIZER_STATIONARITY_TOL_M_S = 0.01


class ReflyBackendError(RuntimeError):
    """The tudatpy backend the re-flight needs is not reachable, or the
    worker environment could not fly the pair."""


# ---------------------------------------------------------------------------
# worker: the TudatPy re-flight (runs in the tudat environment)
# ---------------------------------------------------------------------------

def maneuvers_digest(plan) -> str:
    """Canonical digest of a plan's maneuver sequence (epochs + inertial
    delta-v) -- the identity every cross-environment comparison keys on."""
    doc = [m.to_dict() for m in plan.maneuvers]
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode("utf-8")).hexdigest()


def sweep_table(runs: list, leg_end_epoch: float, epoch_anchor: float) -> tuple[list, float]:
    """Convergence-sweep table: every run compared against the last (the
    reference run) at the latest common sample epoch <= ``leg_end_epoch``.

    Each run is ``(settings, epochs, history)``, and every history row is the
    concatenated pair state ``[r_c, v_c, r_d, v_d]`` (what
    ``np.concatenate([chief, deputy], axis=1)`` of two ``[r, v]`` state
    histories produces). The graded metrics are therefore the norm over BOTH
    crafts' position columns ``d[0:3] + d[6:9]`` and over both crafts'
    velocity columns ``d[3:6] + d[9:12]`` -- taking ``d[:6]`` / ``d[6:]``
    would mix position with velocity in one norm and report a deputy position
    difference as a velocity in m/s (task #411). Returns the table and the
    comparison epoch in mission time.
    """
    common = min(float(epochs[epochs <= leg_end_epoch + 1e-6][-1])
                 for _, epochs, _ in runs)
    ref = runs[-1][2][int(np.argmin(np.abs(runs[-1][1] - common)))]
    table = []
    for settings, epochs, hist in runs[:-1]:
        d = hist[int(np.argmin(np.abs(epochs - common)))] - ref
        table.append({**settings,
                      "position_error_m": float(np.linalg.norm(
                          np.concatenate([d[0:3], d[6:9]]))),
                      "velocity_error_m_s": float(np.linalg.norm(
                          np.concatenate([d[3:6], d[9:12]])))})
    return table, common - epoch_anchor


def physics_verdict_from_refly_report(report, plan) -> bool | None:
    """``physics_verified`` derived from evidence (E12-09, the task-#261 rule):
    True/False only when an independent re-flight result exists **for this
    very plan** -- the report is a rendezvous-refly report of the same
    mission, its main-model maneuver digest equals ``plan``'s, and the
    independent-model line reached a verdict (``passed`` -> True, ``failed``
    -> False). No report, a foreign/mismatched one, or a ``not_checked``
    independent line all answer ``None``: not assessed, never a fake pass.
    ``report`` is the loaded report document or the path to one."""
    if report is None:
        return None
    try:
        doc = report if isinstance(report, Mapping) else json.loads(
            Path(str(report)).read_text(encoding="utf-8"))
        if doc.get("kind") != REPORT_KIND:
            return None
        inputs = doc["inputs"]
        if inputs.get("mission_id") != plan.mission_id:
            return None
        if inputs["main_model"]["maneuvers_sha256"] != maneuvers_digest(plan):
            return None
        result = inputs["error_sources"]["independent_model_agreement"]["result"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return {"passed": True, "failed": False}.get(result)


def _maneuver_diff_magnitude(plan_a, plan_b) -> float:
    """Max abs element difference over both burn epochs and both inertial
    delta-v vectors -- the plan-identity measure across environments (float
    reproducibility, not trust)."""
    worst = 0.0
    for ma, mb in zip(plan_a.maneuvers, plan_b.maneuvers):
        worst = max(worst, abs(ma.epoch_s - mb.epoch_s))
        worst = max(worst, float(np.max(np.abs(np.asarray(ma.delta_v_m_s)
                                               - np.asarray(mb.delta_v_m_s)))))
    return worst


def run_refly_in_current_environment(cfg, config_path=None) -> dict:
    """Fly the full mission of ``cfg`` independently in tudatpy and return the
    sealed-ready evidence document (pure data, JSON-safe). Call only in an
    environment that has tudatpy; raises :class:`ReflyBackendError` if not."""
    try:
        import tudatpy  # noqa: F401
    except ImportError as exc:
        raise ReflyBackendError(
            f"{REFLY_BACKEND_MISSING}: tudatpy is not importable in {sys.executable!r}") from exc

    from orbitopt.crossenv.envelope import environment_record
    from orbitopt.relative.frames import eci_to_lvlh
    from orbitopt.relative.propagation import EARTH_J2, EARTH_RE
    from orbitopt.rendezvous.planner import initial_states, plan_rendezvous
    from orbitopt.rendezvous.verify import verify_plan
    from orbitopt.verify.tudat_propagate import (
        DEFAULT_INTEGRATOR,
        VERIFY_ATOL,
        VERIFY_RTOL,
        ReusableCoastPropagator,
    )

    frame = cfg.dynamics.frame
    if frame != "GCRS":
        raise ReflyBackendError(
            f"the re-flight is defined for the config frame GCRS; frame {frame!r} is "
            "not wired -- refusing a silent frame switch")
    # The SPICE kernel pool tudatpy loads standard defines no GCRS/ICRS frame
    # (measured: SPICE(UNKNOWNFRAME)) -- and a rejected frame makes tudat's
    # propagation *abort at t=0 and hand back a one-sample history*, so the
    # pair is flown in ECLIPJ2000. That frame is NOT GCRS: they differ by the
    # obliquity of the ecliptic (23.4 deg about x) even at a common J2000.0
    # anchor, and a zonal field is only axisymmetric about the SPIN axis --
    # feeding GCRS components in as ecliptic components misplaces every J2
    # torque (measured: 133 m of false cross-model hold-point disagreement).
    # The config states and the inertial burns are therefore rotated into
    # ECLIPJ2000 by the IAU2006 precession angle at J2000.0; every graded
    # quantity is chief-LVLH or a norm, invariant under this one constant
    # rotation applied to both crafts, so no comparison needs it undone.
    propagation_frame = "ECLIPJ2000"
    epoch_anchor = 0.0
    mu = cfg.dynamics.mu
    plan = plan_rendezvous(cfg)
    vmain = verify_plan(cfg, plan)
    m1, m2 = plan.maneuvers
    t1, tof = float(m1.epoch_s), float(plan.tof_s)
    eps = math.radians(84381.448 / 3600.0)  # obliquity at J2000.0, IAU76: ECLIPJ2000
                                            # is exactly this rotation of GCRS about x
                                            # (verified against SPICE to 1e-7)
    rot = np.array([[1.0, 0.0, 0.0],
                    [0.0, math.cos(eps), math.sin(eps)],
                    [0.0, -math.sin(eps), math.cos(eps)]])
    block = np.block([[rot, np.zeros((3, 3))], [np.zeros((3, 3)), rot]])
    dv1 = rot @ np.asarray(m1.delta_v_m_s, dtype=float)
    dv2 = rot @ np.asarray(m2.delta_v_m_s, dtype=float)
    r0c, v0c, r0d, v0d = initial_states(cfg)
    s_chief0 = block @ np.concatenate([r0c, v0c])
    s_deputy0 = block @ np.concatenate([r0d, v0d])

    def propagator(*, j2: bool, integrator=DEFAULT_INTEGRATOR, step_size=60.0,
                   rtol=VERIFY_RTOL, atol=VERIFY_ATOL) -> ReusableCoastPropagator:
        # The config's exact (mu, R, J2) forced onto the tudat environment
        # (same declared physics, independently formulated and integrated);
        # the (2,0) spherical-harmonic field reproduces the analytic J2
        # acceleration term for term -- degree/order 2,0 carries no J22.
        gravity = {"gravitational_parameter": mu, "reference_radius": EARTH_RE,
                   "j2": EARTH_J2 if j2 else 0.0}
        return ReusableCoastPropagator(
            central_body="Earth", perturbing_bodies=("Earth",), step_size=step_size,
            earth_spherical_harmonic=(2, 0) if j2 else None,
            integrator=integrator, rtol=rtol, atol=atol,
            central_gravity=gravity, frame=propagation_frame)

    def coast(prop: ReusableCoastPropagator, state, t0: float, dur: float,
              where: str) -> tuple[np.ndarray, np.ndarray]:
        """prop.coast at mission time t0 -> absolute epoch, as (epochs, states)
        -- with the abort guard: a rejected frame, a missing kernel or a bad
        initial state makes tudat terminate at (or near) the start and return
        what it has, *without raising* -- a single-sample history must be an
        error here, not a silent fake coast."""
        epochs, states = prop.coast(state, epoch_anchor + t0, dur)
        epochs = np.asarray(epochs, dtype=float)
        states = np.asarray(states, dtype=float)
        end_gap = (epoch_anchor + t0 + dur) - float(np.max(epochs))
        if len(states) < 2 or end_gap > (1.5 * prop._step_size + 0.5):
            raise ReflyBackendError(
                f"tudat coast for {where} advanced only to "
                f"{float(np.max(epochs)) - epoch_anchor:.3f} s of the requested "
                f"{t0 + dur:.3f} s ({len(states)} samples) -- propagation aborted "
                f"(frame/kernel issue? frame={propagation_frame!r})")
        return epochs, states

    def fly_mission(prop: ReusableCoastPropagator) -> dict:
        """Drift leg, departure burn, transfer leg, arrival burn -- the plan
        as a pulse sequence. Returns the states the quantities are graded on."""
        chief_t1 = coast(prop, s_chief0, 0.0, t1, "chief drift leg")[1][-1]
        chief_t2 = coast(prop, chief_t1, t1, tof, "chief transfer leg")[1][-1]
        deputy_t1_burnt = coast(prop, s_deputy0, 0.0, t1, "deputy drift leg")[1][-1].copy()
        deputy_t1_burnt[3:] += dv1
        deputy_t2_pre = coast(prop, deputy_t1_burnt, t1, tof, "deputy transfer leg")[1][-1]
        deputy_t2 = deputy_t2_pre.copy()
        deputy_t2[3:] += dv2
        rho, rho_dot = eci_to_lvlh(chief_t2[:3], chief_t2[3:], deputy_t2[:3] - chief_t2[:3],
                                   deputy_t2[3:] - chief_t2[3:], frame,
                                   where="rendezvous_refly: arrival")
        hold = np.asarray(cfg.hold_point_rel_lvlh, dtype=float)
        return {
            "chief_t1": chief_t1, "chief_t2": chief_t2,
            "deputy_t1_burnt": deputy_t1_burnt, "deputy_t2_pre": deputy_t2_pre,
            "deputy_t2": deputy_t2,
            "arrival_rel_lvlh_m": rho, "arrival_rel_lvlh_rate_m_s": rho_dot,
            "position_error_m": float(np.linalg.norm(rho - hold)),
            "relative_speed_m_s": float(np.linalg.norm(rho_dot)),
        }

    def fly_min_approach(prop: ReusableCoastPropagator, chief_t1, deputy_t1_burnt) -> dict:
        """Minimum separation over the transfer leg [t1, t2] on the fixed-step
        scan grid (both crafts saved on the identical grid -- the RK4 saves
        its state every step, so this is the raw integration history)."""
        _, sc = coast(prop, chief_t1, t1, tof, "chief transfer scan")
        _, sd = coast(prop, deputy_t1_burnt, t1, tof, "deputy transfer scan")
        sep = np.linalg.norm(sd[:len(sc), :3] - sc[:len(sd), :3], axis=1)
        idx = int(np.argmin(sep))
        return {"min_approach_distance_m": float(sep[idx]),
                "min_approach_epoch_s": t1 + MIN_APPROACH_SCAN_STEP_S * idx}

    # --- matched-physics key re-flight: production integrator, J2 on -------
    j2_prop = propagator(j2=True)
    j2_run = fly_mission(j2_prop)
    j2_scan = fly_min_approach(
        propagator(j2=True, integrator="rk4", step_size=MIN_APPROACH_SCAN_STEP_S),
        j2_run["chief_t1"], j2_run["deputy_t1_burnt"])

    # --- point-mass re-flight (the case the Kepler closed form speaks to) ---
    pm_prop = propagator(j2=False)
    pm_run = fly_mission(pm_prop)
    pm_scan = fly_min_approach(
        propagator(j2=False, integrator="rk4", step_size=MIN_APPROACH_SCAN_STEP_S),
        pm_run["chief_t1"], pm_run["deputy_t1_burnt"])

    # --- convergence sweep over the transfer leg (point-mass case) ---------
    # Fixed-step RK4 saves its samples on its own grid, which straddles the
    # requested end epoch, so comparing each run's LAST sample would grade a
    # pure time-offset artefact (a 60 s grid lands 31 s late: 2.3e5 m of fake
    # "integration error"). Every run is instead compared at the latest
    # sample epoch no coarser grid overshoots past: all RK4 steps here are
    # binary multiples of the reference step, so that epoch is a sample of
    # every run; for the adaptive tolerance axis every run terminates
    # exactly on the leg end and the common epoch is the leg end itself.
    leg_end_epoch = epoch_anchor + t1 + tof

    def pair_history(prop: ReusableCoastPropagator, where: str):
        ec, sc = coast(prop, pm_run["chief_t1"], t1, tof, where + " (chief)")
        ed, sd = coast(prop, pm_run["deputy_t1_burnt"], t1, tof, where + " (deputy)")
        return ec, np.concatenate([sc, sd], axis=1)

    tol_runs = []
    for f in SWEEP_TOL_FACTORS + (SWEEP_TOL_REF_FACTOR,):
        p = propagator(j2=False, integrator="dop853", step_size=60.0,
                       rtol=VERIFY_RTOL * f, atol=VERIFY_ATOL * f)
        epochs, hist = pair_history(p, "sweep tolerance axis")
        tol_runs.append((dict(integrator="dop853", rtol_factor=f, rtol=VERIFY_RTOL * f,
                              atol=VERIFY_ATOL * f), epochs, hist))
    tol_table, tol_common = sweep_table(tol_runs, leg_end_epoch, epoch_anchor)

    step_runs = []
    for s in list(SWEEP_RK4_STEPS_S) + [SWEEP_RK4_REF_STEP_S]:
        epochs, hist = pair_history(propagator(j2=False, integrator="rk4", step_size=s),
                                    f"sweep step axis ({s} s)")
        step_runs.append((dict(integrator="rk4", step_size=s), epochs, hist))
    step_table, step_common = sweep_table(step_runs, leg_end_epoch, epoch_anchor)

    def observed_order(table: list, ratio: float) -> float:
        slopes = [math.log(a["position_error_m"] / b["position_error_m"]) / math.log(ratio)
                  for a, b in zip(table[:-1], table[1:])
                  if a["position_error_m"] > 0 and b["position_error_m"] > 0]
        return float(np.median(slopes)) if slopes else float("nan")

    tol_at_default = tol_table[SWEEP_TOL_FACTORS.index(1.0)]
    step_at_60 = step_table[SWEEP_RK4_STEPS_S.index(60.0)]

    failed = []
    if not (j2_run["position_error_m"] <= cfg.max_position_error_m):
        failed.append("refly_position_error_within_tolerance")
    if not (j2_run["relative_speed_m_s"] <= cfg.max_relative_speed_m_s):
        failed.append("refly_relative_speed_within_tolerance")
    if not (plan.total_delta_v_m_s <= plan.delta_v_budget_m_s):
        failed.append("refly_delta_v_within_budget")
    if not vmain.passed:
        failed.append("main_model_recheck_in_refly_env")

    config_sha = (hashlib.sha256(Path(str(config_path)).read_bytes()).hexdigest()
                  if config_path else None)
    return {
        "kind": REFLY_REPORT_KIND,
        "schema_version": REFLY_SCHEMA_VERSION,
        "mission_id": cfg.mission_id,
        "frame": frame,
        # Absolute states in refly[]/convergence[] are ECLIPJ2000 components at
        # epoch_anchor + mission time: SPICE exposes no GCRS frame, so the pair
        # is flown in ECLIPJ2000 with states/burns rotated out of GCRS by the
        # J2000.0 obliquity anchor (see worker) -- every graded comparison is
        # made in chief-LVLH or as a norm, invariant under that one rotation.
        "propagation_frame": propagation_frame,
        "epoch_anchor_s": epoch_anchor,
        "config_sha256": config_sha,
        "environment": environment_record("tudat"),
        "plan": plan.to_dict(),
        "maneuvers_sha256": maneuvers_digest(plan),
        "mission_in_refly_env": {
            "position_error_m": vmain.position_error_m,
            "relative_speed_m_s": vmain.relative_speed_m_s,
            "total_delta_v_m_s": vmain.total_delta_v_m_s,
            "checks": dict(vmain.checks),
            "passed": bool(vmain.passed),
        },
        "refly": {
            "j2": {
                "model": j2_prop.model_description(),
                "arrival_rel_lvlh_m": [float(x) for x in j2_run["arrival_rel_lvlh_m"]],
                "arrival_rel_lvlh_rate_m_s": [float(x) for x in j2_run["arrival_rel_lvlh_rate_m_s"]],
                "position_error_m": j2_run["position_error_m"],
                "relative_speed_m_s": j2_run["relative_speed_m_s"],
                **j2_scan,
            },
            "point_mass": {
                "model": pm_prop.model_description(),
                "arrival_rel_lvlh_m": [float(x) for x in pm_run["arrival_rel_lvlh_m"]],
                "arrival_rel_lvlh_rate_m_s": [float(x) for x in pm_run["arrival_rel_lvlh_rate_m_s"]],
                "position_error_m": pm_run["position_error_m"],
                "relative_speed_m_s": pm_run["relative_speed_m_s"],
                **pm_scan,
                "chief_state_at_arrival": [float(x) for x in pm_run["chief_t2"]],
                "deputy_state_after_gate": [float(x) for x in pm_run["deputy_t1_burnt"]],
                "deputy_transfer_end_state": [float(x) for x in pm_run["deputy_t2_pre"]],
                "deputy_state_at_arrival": [float(x) for x in pm_run["deputy_t2"]],
            },
        },
        "convergence": {
            "scenario": "transfer leg [t1, t2], point mass, both crafts, error "
                        "against the tightest run of the same axis at the latest "
                        "common sample epoch (see compare_epoch_s per axis)",
            "tolerance_axis": {
                "runs": tol_table,
                "compare_epoch_s": tol_common,
                "reference_rtol": VERIFY_RTOL * SWEEP_TOL_REF_FACTOR,
                "reference_atol": VERIFY_ATOL * SWEEP_TOL_REF_FACTOR,
                "bound_at_default_m": tol_at_default["position_error_m"],
                "bound_at_default_m_s": tol_at_default["velocity_error_m_s"],
                "observed_order": observed_order(tol_table, 10.0),
            },
            "step_axis": {
                "runs": step_table,
                "compare_epoch_s": step_common,
                "reference_step_s": SWEEP_RK4_REF_STEP_S,
                "bound_at_60s_m": step_at_60["position_error_m"],
                "bound_at_60s_m_s": step_at_60["velocity_error_m_s"],
                "observed_order": observed_order(step_table, 2.0),
            },
        },
        "ok": not failed,
        "failed_checks": failed,
    }


# ---------------------------------------------------------------------------
# main-environment references: CW mission replay, Kepler legs, main quantities
# ---------------------------------------------------------------------------

def cw_mission_replay(cfg, plan) -> dict:
    """The planned mission replayed on the linear CW model: the pair's LVLH
    state drifts by the CW STM to the planned gate epoch, the plan's inertial
    pulses are rotated into the *nominal* circular chief's LVLH frame at the
    burn epochs and applied, and the transfer is propagated by the same STM
    on a 1 s grid (enough to locate the separation minimum: the relative
    motion over 1 s is metres). The CW model shares none of the nonlinear
    machinery -- only the pulses -- so the difference at the acceptance
    quantities is the price of the linearization (plus the J2 it ignores).
    """
    from orbitopt.relative.cw import mean_motion_circular, propagate_cw
    from orbitopt.relative.frames import eci_to_lvlh
    from orbitopt.rendezvous.planner import _circular_state, initial_states

    frame, mu = cfg.dynamics.frame, cfg.dynamics.mu
    r0c, v0c, r0d, v0d = initial_states(cfg)
    a_t = float(np.linalg.norm(r0c))
    n = mean_motion_circular(mu, a_t)
    inc = math.radians(cfg.target.inclination_deg)
    raan = math.radians(cfg.target.raan_deg)
    u0 = math.radians(cfg.target.argument_of_latitude_deg)

    def dv_to_lvlh(dv_inertial, t: float) -> np.ndarray:
        rc, vc = _circular_state(a_t, inc, raan, u0 + n * t, mu)
        _, dv_lvlh = eci_to_lvlh(rc, vc, np.zeros(3), np.asarray(dv_inertial, dtype=float),
                                 frame, where="cw_mission_replay: dv rotation")
        return dv_lvlh

    rho0, rd0 = eci_to_lvlh(r0c, v0c, r0d - r0c, v0d - v0c, frame,
                            where="cw_mission_replay: t0")
    m1, m2 = plan.maneuvers
    t1, tof = float(m1.epoch_s), float(plan.tof_s)
    rho1, rd1 = propagate_cw(rho0, rd0, n, t1, where="cw_mission_replay: drift")
    rd1_b = rd1 + dv_to_lvlh(m1.delta_v_m_s, t1)

    tau = np.arange(0.0, tof, 1.0)
    tau = np.append(tau, tof)
    rho_t, _ = propagate_cw(np.repeat(rho1[None, :], tau.size, axis=0),
                            np.repeat(rd1_b[None, :], tau.size, axis=0), n, tau,
                            where="cw_mission_replay: transfer")
    sep = np.linalg.norm(rho_t, axis=1)
    idx = int(np.argmin(sep))
    rho2, rd2 = propagate_cw(rho1, rd1_b, n, tof, where="cw_mission_replay: arrival")
    rd2 += dv_to_lvlh(m2.delta_v_m_s, t1 + tof)
    hold = np.asarray(cfg.hold_point_rel_lvlh, dtype=float)
    return {
        "arrival_rel_lvlh_m": [float(x) for x in rho2],
        "arrival_rel_lvlh_rate_m_s": [float(x) for x in rd2],
        "position_error_m": float(np.linalg.norm(rho2 - hold)),
        "relative_speed_m_s": float(np.linalg.norm(rd2)),
        "min_approach_distance_m": float(sep[idx]),
        "min_approach_epoch_s": float(t1 + tau[idx]),
        "total_delta_v_m_s": float(np.linalg.norm(m1.delta_v_m_s)
                                   + np.linalg.norm(m2.delta_v_m_s)),
        "chief_mean_motion_rad_s": float(n),
    }


def main_model_key_quantities(cfg, plan) -> dict:
    """The acceptance quantities on the main model: the re-propagated plan's
    verdict (:func:`orbitopt.rendezvous.verify.verify_plan`, the only place
    the LEO-RDV-001 verdict lives) plus the minimum approach distance over
    the transfer leg, sampled on the same 2 s grid the re-flight scan uses."""
    from orbitopt.relative.propagation import EARTH_J2, propagate_pair
    from orbitopt.rendezvous.planner import initial_states
    from orbitopt.rendezvous.verify import verify_plan

    vmain = verify_plan(cfg, plan)
    j2 = EARTH_J2 if cfg.dynamics.j2 else 0.0
    frame, mu = cfg.dynamics.frame, cfg.dynamics.mu
    m1, _gate_burn = plan.maneuvers  # the gate burn acts only after the scan window
    t1, tof = float(m1.epoch_s), float(plan.tof_s)
    r0c, v0c, r0d, v0d = initial_states(cfg)
    leg_a = propagate_pair(r0c, v0c, r0d, v0d, np.array([0.0, t1]), frame=frame, mu=mu, j2=j2,
                           where="main_model_key_quantities: drift leg")
    dv1 = np.asarray(m1.delta_v_m_s, dtype=float)
    grid = np.arange(0.0, tof, MIN_APPROACH_SCAN_STEP_S)
    grid = np.append(grid, tof)
    leg_b = propagate_pair(leg_a.chief_r[0, 1], leg_a.chief_v[0, 1],
                           leg_a.deputy_r[0, 1], leg_a.deputy_v[0, 1] + dv1,
                           grid, frame=frame, mu=mu, j2=j2,
                           where="main_model_key_quantities: transfer scan")
    sep = np.linalg.norm(leg_b.deputy_r[0] - leg_b.chief_r[0], axis=1)
    idx = int(np.argmin(sep))
    return {
        "position_error_m": vmain.position_error_m,
        "relative_speed_m_s": vmain.relative_speed_m_s,
        "total_delta_v_m_s": vmain.total_delta_v_m_s,
        "min_approach_distance_m": float(sep[idx]),
        "min_approach_epoch_s": float(t1 + grid[idx]),
        "arrival_rel_lvlh_m": [float(x) for x in vmain.arrival_rel_lvlh_m],
        "arrival_rel_lvlh_rate_m_s": [float(x) for x in vmain.arrival_rel_lvlh_rate_m_s],
        "maneuvers_sha256": maneuvers_digest(plan),
        "verification_checks": dict(vmain.checks),
        "verification_passed": bool(vmain.passed),
    }


def kepler_point_mass_reference(cfg, plan) -> dict:
    """Closed-form two-body replay of the point-mass re-flight (analytic
    reference for the numerical-error line): each coast leg propagated exactly
    by :func:`orbitopt.verify.benchmarks.propagate_kepler_analytic`, the pulse
    a velocity addition at the leg boundary -- the same pulse sequence, zero
    numerical error."""
    from orbitopt.rendezvous.planner import initial_states
    from orbitopt.verify.benchmarks import propagate_kepler_analytic

    mu = cfg.dynamics.mu
    m1 = plan.maneuvers[0]
    t1, tof = float(m1.epoch_s), float(plan.tof_s)
    r0c, v0c, r0d, v0d = initial_states(cfg)
    chief = propagate_kepler_analytic(np.concatenate([r0c, v0c]), t1 + tof, mu)
    leg1 = propagate_kepler_analytic(np.concatenate([r0d, v0d]), t1, mu)
    leg1[3:] += np.asarray(m1.delta_v_m_s, dtype=float)
    deputy_transfer_end = propagate_kepler_analytic(leg1, tof, mu)
    return {"chief_at_arrival": chief, "deputy_after_gate": leg1,
            "deputy_transfer_end": deputy_transfer_end}


# ---------------------------------------------------------------------------
# report assembly
# ---------------------------------------------------------------------------

def _scalar_residual(name: str, reference: str, measured: float, tol: float, *,
                     unit: str, category: str, evidence: str, note: str = "") -> dict:
    """A hand-written scalar residual (the report schema accepts bare-string
    ``evidence`` next to the ``category``; ``state_residual`` mints the full
    qualification block for the state compares)."""
    r = {"name": name, "reference": reference, "category": category, "evidence": evidence,
         "tolerance": ({"position_m": tol} if unit == "m" else {"velocity_m_s": tol}),
         "within_tolerance": bool(measured <= tol), "unit": unit}
    if unit == "m":
        r["position_residual_m"] = float(measured)
        r["velocity_residual_m_s"] = None
    else:
        r["position_residual_m"] = None
        r["velocity_residual_m_s"] = float(measured)
    if note:
        r["note"] = note
    return r


def refly_residuals(cfg, main_q: Mapping, cw_q: Mapping | None, refly_doc: Mapping | None,
                    kepler: Mapping | None) -> list[dict]:
    """The residual lines, grouped by prefix: ``optimizer.*`` (non-convergence),
    ``numerical.*`` (integration error: sweeps + Kepler), ``refly.*``
    (independent-model headline), ``cw.*`` (approximate model). Absent groups
    simply produce no residuals -- their error-source line then reports
    ``not_checked``, never a silent pass."""
    residuals: list[dict] = []
    opt_tol = cfg.optimizer.newton_tol_m

    residuals.append(_scalar_residual(
        "optimizer.landing_residual", "planner Newton targeting (burn-1 aim at the Hold Point)",
        main_q["position_error_m"], opt_tol, unit="m",
        category="numerical_consistency", evidence="design_input",
        note="the plan's own arrival residual against the config's newton_tol_m -- how much of "
             "any discrepancy the optimizer's leftover non-convergence could explain"))
    val = main_q.get("dv_stationarity_m_s")
    if val is not None:
        residuals.append(_scalar_residual(
            "optimizer.delta_v_stationarity",
            "plan total delta-v, golden window closed to +-0.5 s", abs(float(val)),
            OPTIMIZER_STATIONARITY_TOL_M_S, unit="m_s",
            category="numerical_consistency", evidence="numerical_consistency",
            note="re-optimizing inside a window half the planner's golden-section stop "
                 "criterion must not move the objective: the search converged, not stalled"))

    if refly_doc is not None:
        conv = refly_doc["convergence"]
        tb, sb = conv["tolerance_axis"], conv["step_axis"]

        def _order(v):  # NaN (identical runs, or none pairable) None-ises through JSON
            return "n/a" if v is None else f"{v:.2f}"

        residuals.append(_scalar_residual(
            "numerical.tudat_tolerance_axis",
            "DOP853 reference run, one decade tighter than production",
            tb["bound_at_default_m"], SWEEP_BOUNDS["tolerance_axis_position_m"], unit="m",
            category="numerical_consistency", evidence="numerical_consistency",
            note=f"transfer-leg final-state position error of the production tolerances "
                 f"(rtol/atol {tb['reference_rtol'] * 10.0:.0e}/{tb['reference_atol'] * 10.0:.0e}) "
                 f"vs the reference at epoch {tb['compare_epoch_s']:.1f} s (adaptive runs "
                 f"terminate on the leg end); observed convergence order "
                 f"{_order(tb['observed_order'])}; "
                 f"velocity bound {tb['bound_at_default_m_s']:.3e} m/s"))
        residuals.append(_scalar_residual(
            "numerical.tudat_step_axis_60s",
            f"RK4 reference at step {SWEEP_RK4_REF_STEP_S} s",
            sb["bound_at_60s_m"], SWEEP_BOUNDS["step_axis_60s_position_m"], unit="m",
            category="numerical_consistency", evidence="numerical_consistency",
            note=f"legacy fixed-step RK4 at 60 s vs the {SWEEP_RK4_REF_STEP_S} s reference "
                 f"at their latest common sample epoch {sb['compare_epoch_s']:.1f} s "
                 f"(fixed-step grids straddle the leg end; comparing last samples would "
                 f"measure a time offset, not integration error); observed order "
                 f"{_order(sb['observed_order'])}, expected ~4)"))
        if kepler is not None:
            from orbitopt.relative.frames import eci_to_lvlh
            pm = refly_doc["refly"]["point_mass"]
            chief = np.asarray(pm["chief_state_at_arrival"], dtype=float)
            dep = np.asarray(pm["deputy_transfer_end_state"], dtype=float)
            kc = np.asarray(kepler["chief_at_arrival"], dtype=float)
            kd = np.asarray(kepler["deputy_transfer_end"], dtype=float)
            rho_n, rd_n = eci_to_lvlh(chief[:3], chief[3:], dep[:3] - chief[:3],
                                      dep[3:] - chief[3:], "GCRS",
                                      where="refly_residuals: kepler numerical pair")
            rho_a, rd_a = eci_to_lvlh(kc[:3], kc[3:], kd[:3] - kc[:3],
                                      kd[3:] - kc[3:], "GCRS",
                                      where="refly_residuals: kepler analytic pair")
            tol = KEPLER_VS_REFLY_TOLERANCES
            residuals.append(_scalar_residual(
                "numerical.point_mass_kepler_position",
                "Kepler closed form at the transfer end, chief LVLH",
                float(np.linalg.norm(rho_n - rho_a)), tol["position_m"], unit="m",
                category="numerical_consistency", evidence="analytical",
                note="point-mass pair re-flight vs the closed-form two-body replay at the "
                     "transfer end. Compared in the chief's LVLH frame: the ECLIPJ2000/GCRS "
                     "anchor rotation is common-mode on the pair and cancels, so the number "
                     "is pure numerical-integration error, not frame bookkeeping"))
            residuals.append(_scalar_residual(
                "numerical.point_mass_kepler_velocity",
                "Kepler closed form at the transfer end, chief LVLH",
                float(np.linalg.norm(rd_n - rd_a)), tol["velocity_m_s"], unit="m_s",
                category="numerical_consistency", evidence="analytical",
                note="same closed-form pair compare on the relative rate"))

        tol = REFLY_VS_MAIN_TOLERANCES
        rj = refly_doc["refly"]["j2"]
        residuals.append(_scalar_residual(
            "refly.hold_point_position_error", "main model (verify_plan, RK4 + analytic J2)",
            abs(rj["position_error_m"] - main_q["position_error_m"]),
            tol["hold_point_position_error_m"], unit="m", category="model_agreement",
            evidence="independent_model",
            note="TudatPy (spherical-harmonic(2,0) gravity, adaptive integration) re-flight vs "
                 "orbitopt's own pair propagator, on the config's exact GM/J2/frame"))
        residuals.append(_scalar_residual(
            "refly.relative_speed", "main model (verify_plan)",
            abs(rj["relative_speed_m_s"] - main_q["relative_speed_m_s"]),
            tol["relative_speed_m_s"], unit="m_s", category="model_agreement",
            evidence="independent_model"))
        residuals.append(_scalar_residual(
            "refly.min_approach_distance", "main model (propagate_pair, 2 s scan)",
            abs(rj["min_approach_distance_m"] - main_q["min_approach_distance_m"]),
            tol["min_approach_distance_m"], unit="m", category="model_agreement",
            evidence="independent_model"))
        residuals.append(_scalar_residual(
            "refly.total_delta_v", "plan pulse sequence (same plan re-flown)",
            abs(refly_doc["plan"]["total_delta_v_m_s"] - main_q["total_delta_v_m_s"]),
            tol["total_delta_v_m_s"], unit="m_s", category="numerical_consistency",
            evidence="numerical_consistency",
            note="the re-flight applies the plan's own pulses; the worker recomputed the plan "
                 "in its own environment -- equality bounds the float reproducibility across "
                 f"environments (digests: worker {refly_doc['maneuvers_sha256'][:12]}..., main "
                 f"{main_q['maneuvers_sha256'][:12]}...)"))
        ident = main_q.get("maneuver_identity_m_s")
        if ident is not None:
            residuals.append(_scalar_residual(
                "refly.maneuver_sequence_identity", "main-environment plan (epochs + inertial dv)",
                float(ident), tol["maneuver_sequence_m_s"], unit="m_s",
                category="numerical_consistency", evidence="numerical_consistency",
                note="max |element| difference of the two burn epochs and delta-v vectors "
                     "between the environments -- the headline comparison is only "
                     "meaningful if both flew the same plan"))

    if cw_q is not None:
        tol = CW_VS_NONLINEAR_TOLERANCES
        residuals.append(_scalar_residual(
            "cw.hold_point_position_error", "nonlinear main model (verify_plan)",
            abs(cw_q["position_error_m"] - main_q["position_error_m"]),
            tol["hold_point_position_error_m"], unit="m", category="model_agreement",
            evidence="analytical",
            note="CW linearization + J2 omission measured at the mission's own acceptance "
                 "quantity -- the planning model is expected to differ; this bounds how much"))
        residuals.append(_scalar_residual(
            "cw.relative_speed", "nonlinear main model (verify_plan)",
            abs(cw_q["relative_speed_m_s"] - main_q["relative_speed_m_s"]),
            tol["relative_speed_m_s"], unit="m_s", category="model_agreement",
            evidence="analytical"))
        residuals.append(_scalar_residual(
            "cw.min_approach_distance", "nonlinear main model (2 s scan)",
            abs(cw_q["min_approach_distance_m"] - main_q["min_approach_distance_m"]),
            tol["min_approach_distance_m"], unit="m", category="model_agreement",
            evidence="analytical"))
    return residuals


def build_refly_report(cfg, main_q: Mapping, cw_q: Mapping | None, refly_doc: Mapping | None,
                       kepler: Mapping | None, *,
                       not_checked_reasons: Sequence[str] = (),
                       crossenv: Mapping | None = None) -> dict:
    """Assemble the digest-pinned verification-report-1.0 document. The error
    sources are separate residual lines *and* separate entries of
    ``inputs["error_sources"]``; the report's own ``status.categories`` and
    ``status.evidence`` rollups keep the three classes separate too -- nothing
    here is a single blended ``passed``."""
    from orbitopt.verify.report import build_report, validate_report

    residuals = refly_residuals(cfg, main_q, cw_q, refly_doc, kepler)
    by_name = {r["name"]: r for r in residuals if "within_tolerance" in r}

    def line(prefix: str) -> dict:
        names = sorted(n for n in by_name if n.startswith(prefix))
        ok = [bool(by_name[n]["within_tolerance"]) for n in names]
        measured = {}
        for n in names:
            r = by_name[n]
            pos, vel = r.get("position_residual_m"), r.get("velocity_residual_m_s")
            measured[n] = pos if pos is not None else vel
        return {"checks": names,
                "result": "not_checked" if not names else
                          ("passed" if all(ok) else "failed"),
                "measured": measured}

    opt_tol = cfg.optimizer.newton_tol_m
    error_sources = {
        "numerical_propagation": {
            "question": "how much of any difference is integration error",
            "tolerances": dict(SWEEP_BOUNDS), **line("numerical."),
            "note": "TudatPy tolerance/step sweeps + Kepler closed-form compare of the same "
                    "point-mass model"},
        "approximate_model_cw": {
            "question": "what does the CW planning approximation cost on this mission",
            "tolerances": dict(CW_VS_NONLINEAR_TOLERANCES), **line("cw.")},
        "optimizer_nonconvergence": {
            "question": "how much could leftover optimizer error explain",
            "tolerances": {"landing_residual_m": opt_tol,
                           "delta_v_stationarity_m_s": OPTIMIZER_STATIONARITY_TOL_M_S},
            **line("optimizer.")},
        "independent_model_agreement": {
            "question": "do the two independently formulated pair models agree on the four "
                        "acceptance quantities",
            "tolerances": dict(REFLY_VS_MAIN_TOLERANCES), **line("refly.")},
    }
    headline = error_sources["independent_model_agreement"]["result"]
    inputs = {
        "mission_id": cfg.mission_id,
        "frame": cfg.dynamics.frame,
        "headline_acceptance": headline,
        "declared_tolerances": {
            "refly_vs_main": dict(REFLY_VS_MAIN_TOLERANCES),
            "kepler_vs_refly": dict(KEPLER_VS_REFLY_TOLERANCES),
            "sweep_bounds": dict(SWEEP_BOUNDS),
            "cw_vs_nonlinear": dict(CW_VS_NONLINEAR_TOLERANCES),
            "optimizer_landing_m": opt_tol,
            "optimizer_stationarity_m_s": OPTIMIZER_STATIONARITY_TOL_M_S,
        },
        "error_sources": error_sources,
        "main_model": dict(main_q),
        "cw_replay": dict(cw_q) if cw_q else None,
        "refly_environment": None if refly_doc is None else {
            "env": refly_doc["environment"],
            "mission_in_refly_env": refly_doc["mission_in_refly_env"],
            "convergence": refly_doc["convergence"],
        },
        "not_checked_reasons": list(not_checked_reasons),
        "crossenv": dict(crossenv) if crossenv else None,
        "refly_mission_failed_checks": (
            list(refly_doc.get("failed_checks") or []) if refly_doc is not None else None),
        "historical_telemetry": "absent by construction -- this report grades independent-model, "
                                "analytical, numerical-consistency and design-input evidence only",
    }
    targets = {"converged": bool(main_q.get("verification_passed", False)),
               "note": "plan targeting convergence (verify_plan); the optimizer line of "
                       "error_sources carries the residual detail"}
    report = build_report(
        REPORT_KIND,
        f"E12-05 independent re-flight of {cfg.mission_id} (TudatPy two-spacecraft re-fly)",
        [], residuals, inputs=inputs, targets=targets)
    validate_report(report)
    return report


# ---------------------------------------------------------------------------
# backend resolution + orchestration
# ---------------------------------------------------------------------------

def resolve_refly_backend() -> tuple[str | None, str | None]:
    """``("crossenv", interpreter)`` when ``ORBITOPT_TUDAT_PYTHON`` names one,
    ``("in_process", None)`` when tudatpy imports here, else ``(None, None)``."""
    exe = os.environ.get("ORBITOPT_TUDAT_PYTHON")
    if exe:
        return "crossenv", exe
    import importlib.util
    if importlib.util.find_spec("tudatpy") is not None:
        return "in_process", None
    return None, None


def refly_via_crossenv(cfg, config_path: Path, out_dir, interpreter: str) -> dict:
    """Run the re-flight as a crossenv stage in ``interpreter`` and return
    ``{"document", "envelope_digest", "stage_report"}`` after re-validating
    the sealed envelope in this process (schema + every digest, re-computed
    over here before anything is read)."""
    from orbitopt.crossenv.envelope import envelope_digest, read_payloads
    from orbitopt.crossenv.plan import (
        CrossEnvPlan,
        CrossEnvStage,
        run_stage_subprocess,
        write_resolved_plan,
    )

    out = Path(out_dir)
    stage = CrossEnvStage(id="refly", runner=REFLY_STAGE_RUNNER, env="tudat")
    plan = CrossEnvPlan(mission_id=f"e12-05-refly-{cfg.mission_id}", scenario=Path(config_path),
                        environments={"tudat": str(interpreter)}, stages=(stage,), out=out)
    resolved = write_resolved_plan(plan, out / "plan.resolved.json")
    proc = run_stage_subprocess(plan, stage, out, resolved)
    report_path = out / stage.id / "stage-report.json"
    stage_report = (json.loads(report_path.read_text(encoding="utf-8"))
                    if report_path.is_file() else None)
    if proc.returncode != 0 or not stage_report or stage_report.get("status") != "completed":
        tail = (proc.stderr or proc.stdout or "").strip()[-1500:]
        raise ReflyBackendError(
            f"crossenv re-flight stage failed (rc={proc.returncode}): "
            f"{tail or (stage_report or {}).get('summary') or 'no stage report'}")
    exchange = out / stage.id / "exchange"
    doc = read_payloads(exchange)[REFLY_PAYLOAD_NAME]  # validates envelope + digests first
    if doc.get("kind") != REFLY_REPORT_KIND or doc.get("mission_id") != cfg.mission_id:
        raise ReflyBackendError(
            f"sealed evidence is {doc.get('kind')!r}/{doc.get('mission_id')!r}, not "
            f"{REFLY_REPORT_KIND!r}/{cfg.mission_id!r}")
    return {"document": doc, "envelope_digest": envelope_digest(exchange),
            "stage_report": stage_report}


def verify_rendezvous_refly(*, config_path=None, out_dir: str | Path = "runs/e12-05-refly",
                            stationarity: bool = True) -> dict:
    """The task's one command: plan + verify on the main model, replay on CW,
    re-fly the pair in TudatPy (crossenv when ``ORBITOPT_TUDAT_PYTHON`` is
    set, in-process when this environment has tudatpy, honest ``not_checked``
    when neither), compare against the Kepler closed form, and write
    ``rendezvous-refly-report.json`` under ``out_dir``.
    """
    import dataclasses

    from orbitopt.rendezvous.config import load_rendezvous_config
    from orbitopt.rendezvous.planner import plan_rendezvous
    from orbitopt.verify.report import write_report

    if config_path is None:
        config_path = Path(str(resources.files("orbitopt").joinpath("data", "leo_rdv_001.yaml")))
    config_path = Path(config_path)
    cfg = load_rendezvous_config(config_path)
    plan = plan_rendezvous(cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    main_q = main_model_key_quantities(cfg, plan)
    if stationarity:
        half = 0.5  # half the planner's 1 s golden-section stop criterion
        narrowed = dataclasses.replace(
            cfg.windows,
            transfer_tof_min_s=max(cfg.windows.transfer_tof_min_s, plan.tof_s - half),
            transfer_tof_max_s=min(cfg.windows.transfer_tof_max_s, plan.tof_s + half))
        side = plan_rendezvous(dataclasses.replace(cfg, windows=narrowed))
        main_q["dv_stationarity_m_s"] = side.total_delta_v_m_s - plan.total_delta_v_m_s
    cw_q = cw_mission_replay(cfg, plan)

    not_checked: list[str] = []
    refly_doc = None
    crossenv_meta = None
    backend, interpreter = resolve_refly_backend()
    if backend == "crossenv":
        run = refly_via_crossenv(cfg, config_path, out, interpreter)
        refly_doc = run["document"]
        crossenv_meta = {k: v for k, v in run.items() if k != "document"}
    elif backend == "in_process":
        refly_doc = run_refly_in_current_environment(cfg, config_path)
    else:
        not_checked.append(
            f"{REFLY_BACKEND_MISSING}: no ORBITOPT_TUDAT_PYTHON and no importable tudatpy in "
            f"{sys.executable!r} -- the independent-model, Kepler and convergence lines are "
            "NOT_CHECKED (recorded, never passed)")

    kepler = (kepler_point_mass_reference(cfg, plan) if refly_doc is not None else None)
    if refly_doc is not None:
        main_q["maneuver_identity_m_s"] = _maneuver_diff_magnitude(
            plan, _plan_from_doc(refly_doc["plan"]))
    report = build_refly_report(cfg, main_q, cw_q, refly_doc, kepler,
                                not_checked_reasons=not_checked, crossenv=crossenv_meta)
    path = write_report(report, out / "rendezvous-refly-report.json")
    return {"report": report, "refly": refly_doc, "plan": plan, "main_quantities": main_q,
            "cw": cw_q, "kepler": kepler, "report_path": str(path)}


def _plan_from_doc(doc: Mapping) -> Any:
    """Rebuild a :class:`orbitopt.rendezvous.planner.RendezvousPlan` from a
    sealed ``plan.to_dict()`` document -- just enough for identity comparison
    (maneuver epochs + delta-v)."""
    from orbitopt.rendezvous.planner import Maneuver

    m = doc["maneuvers"]
    return type("P", (), {"maneuvers": tuple(
        Maneuver(str(x["kind"]), float(x["epoch_s"]), tuple(float(v) for v in x["delta_v_eci_m_s"]))
        for x in m)})()


# ---------------------------------------------------------------------------
# crossenv stage entry (registered in orbitopt.crossenv.stages)
# ---------------------------------------------------------------------------

def refly_stage_impl(cfg, config_path, exchange_dir: Path) -> tuple[list, dict]:
    """Worker half of the ``rendezvous_refly`` stage: fly the re-flight here,
    write the evidence artifact, return ``(payload specs, evidence doc)`` --
    the stage runner reads ok/failed_checks off the doc. Sealing itself stays
    in the crossenv stage runner (crossenv owns the envelope)."""
    from orbitopt.verify.report import to_jsonable

    doc = run_refly_in_current_environment(cfg, config_path)
    reports = Path(exchange_dir) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "rendezvous-refly-evidence.json").write_text(
        json.dumps(to_jsonable(doc), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    payloads = [{"name": REFLY_PAYLOAD_NAME, "type": "report",
                 "schema_version": REFLY_SCHEMA_VERSION, "report_kind": REFLY_REPORT_KIND,
                 "path": "reports/rendezvous-refly-evidence.json", "digest_kind": "file"}]
    return payloads, doc


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="E12-05 independent TudatPy re-flight of a "
                                             "rendezvous mission (default: LEO-RDV-001)")
    ap.add_argument("--config", default=None, help="rendezvous mission config (default: "
                                                   "packaged leo_rdv_001.yaml)")
    ap.add_argument("-o", "--out", default="runs/e12-05-refly")
    ap.add_argument("--no-stationarity", action="store_true",
                    help="skip the +-0.5 s re-optimizations (faster, weaker optimizer line)")
    args = ap.parse_args(argv)
    res = verify_rendezvous_refly(config_path=args.config, out_dir=args.out,
                                  stationarity=not args.no_stationarity)
    from orbitopt.verify.report import format_status
    report = res["report"]
    st = report["status"]
    print(f"{res['report_path']}: {format_status(st)}")
    for name, src in report["inputs"]["error_sources"].items():
        measured = ", ".join(f"{k.split('.')[-1]}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                             for k, v in src["measured"].items())
        print(f"  {name}: {src['result']} [{measured}]")
    iv = st["independent_validation"]
    print(f"  independent_validation: {iv['result']} "
          f"({iv['checks_passed']}/{iv['checks_total']} checks)")
    for r in report["inputs"]["not_checked_reasons"]:
        print(f"  not_checked: {r}")
    headline = report["inputs"]["headline_acceptance"]
    if headline == "not_checked":
        return 2
    return 0 if headline == "passed" and st["result"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
