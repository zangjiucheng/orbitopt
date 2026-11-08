"""High-fidelity verification of pykep-optimized trajectories using tudatpy.

pykep's Lambert/MGA solutions are two-body-per-leg (patched conic):
Keplerian arcs glued together at flyby/manoeuvre points, ignoring all
perturbations. Before trusting an optimized trajectory, propagate the same
initial state through tudatpy's numerical, N-body-capable dynamics (point
masses of all major bodies by default, extensible to SRP/harmonics/drag) and
compare the resulting arrival state against what pykep predicted -- if they
diverge by more than a mission's nav budget, the patched-conic solution needs
mid-course correction modelling, not just a better optimizer.

Integrators (task #136): the *screening* layer (``dynamics.nbody_gpu``) keeps
fixed-step RK4 because it batches thousands of candidates on the GPU. This
*verification* layer defaults to an adaptive, error-controlled 8th-order
Dormand-Prince integrator (tudat's ``rkdp_87``, the DOP853 family) with
``step_size`` reused as the *maximum* step -- so the saved history is never
coarser than the old RK4 one, while the local truncation error is held to
``VERIFY_RTOL``/``VERIFY_ATOL``. Pass ``integrator="rk4"`` to reproduce the
legacy fixed-step results. Events (periapsis / closest approach, SOI and
entry-interface crossings) are located by root finding on event functions,
see :mod:`orbitopt.verify.events`, instead of by picking saved samples.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from tudatpy import constants, numerical_simulation
from tudatpy.interface import spice
from tudatpy.numerical_simulation import environment_setup, propagation_setup

from orbitopt.verify import events as _events
from orbitopt.verify import report as _report

# Verification-layer integrator defaults (see module docstring).
DEFAULT_INTEGRATOR = "dop853"
VERIFY_RTOL = 1e-11
VERIFY_ATOL = 1e-6  # m and m/s (element-wise on the Cartesian state)
VERIFY_MIN_STEP = 1e-3  # s

_SPICE_LOADED = False


def _ensure_spice_loaded():
    global _SPICE_LOADED
    if not _SPICE_LOADED:
        spice.load_standard_kernels()
        _SPICE_LOADED = True


@dataclass
class PropagationResult:
    epochs: np.ndarray  # (n,) seconds since J2000
    states: np.ndarray  # (n, 6) cartesian [x,y,z,vx,vy,vz], SI units, central-body frame
    final_state: np.ndarray  # (6,)
    # Absolute tudat epoch corresponding to epochs == 0 (propagate_multi_arc
    # reports epochs relative to its initial_epoch) and the propagator that
    # produced the history; together they let event location re-propagate a
    # bracketing interval at fine resolution. Both optional / back-compatible.
    epoch_offset: float = 0.0
    propagator: Optional["ReusableCoastPropagator"] = None
    # Verification-report metadata (task #137): the dynamics/integrator
    # description, the (start, stop) index range of every coast arc in
    # ``states`` (burns split arcs), and a hook that re-runs the identical
    # propagation with keyword overrides (tighter tolerances, two-body only).
    model: Optional[dict] = None
    segments: Optional[list] = None
    rerun: Optional[Callable[..., "PropagationResult"]] = field(default=None, repr=False)


def _integrator_settings(integrator, step_size, rtol=VERIFY_RTOL, atol=VERIFY_ATOL, min_step=VERIFY_MIN_STEP):
    """Build tudat integrator settings.

    ``"rk4"``: legacy fixed-step RK4 with ``step_size``.
    ``"dop853"`` (alias ``"rkdp87"``): adaptive Runge-Kutta-Dormand-Prince
    8(7) with ``step_size`` as the maximum step. Supports both the tudatpy
    >= 0.8 API (``runge_kutta_variable_step``) and the older
    ``runge_kutta_variable_step_size``.
    """
    integ = propagation_setup.integrator
    name = str(integrator).lower()
    if name == "rk4":
        return integ.runge_kutta_fixed_step(float(step_size), integ.CoefficientSets.rk_4)
    if name not in ("dop853", "rkdp87", "rkdp_87"):
        raise ValueError(f"Unknown integrator {integrator!r}; expected 'dop853' or 'rk4'")
    max_step = float(step_size)
    initial_step = min(max_step, 60.0)
    coeffs = integ.CoefficientSets.rkdp_87
    if hasattr(integ, "runge_kutta_variable_step"):
        control = integ.step_size_control_elementwise_scalar_tolerance(float(rtol), float(atol))
        validation = integ.step_size_validation(float(min_step), max_step)
        return integ.runge_kutta_variable_step(initial_step, coeffs, control, validation)
    return integ.runge_kutta_variable_step_size(
        initial_step, coeffs, float(min_step), max_step, float(rtol), float(atol),
    )


def integrator_description(integrator, step_size, rtol=VERIFY_RTOL, atol=VERIFY_ATOL, min_step=VERIFY_MIN_STEP) -> dict:
    """JSON-ready description of what :func:`_integrator_settings` builds."""
    name = str(integrator).lower()
    if name == "rk4":
        return {"name": "rk4", "method": "Runge-Kutta 4 (fixed step)", "adaptive": False,
                "step_s": float(step_size), "rtol": None, "atol": None,
                "min_step_s": None, "max_step_s": float(step_size)}
    return {"name": "dop853", "method": "Runge-Kutta-Dormand-Prince 8(7) (tudat rkdp_87), adaptive",
            "adaptive": True, "step_s": None, "rtol": float(rtol), "atol": float(atol),
            "atol_units": "m and m/s, element-wise", "initial_step_s": min(float(step_size), 60.0),
            "min_step_s": float(min_step), "max_step_s": float(step_size)}


def _time_termination(final_epoch, integrator):
    # Adaptive steps don't land on final_epoch by themselves; callers use the
    # final state *at* the requested time (TCM targeting), so stop exactly.
    exact = str(integrator).lower() != "rk4"
    return propagation_setup.propagator.time_termination(
        float(final_epoch), terminate_exactly_on_final_condition=exact,
    )


def propagate_two_body_leg(
    r1,
    v1,
    tof_seconds,
    perturbing_bodies=("Sun", "Earth", "Mars", "Jupiter"),
    central_body="Sun",
    step_size=3600.0,
    save_history=True,
    departure_epoch_ephemeris_seconds=0.0,
    integrator=DEFAULT_INTEGRATOR,
    rtol=VERIFY_RTOL,
    atol=VERIFY_ATOL,
) -> PropagationResult:
    """Numerically propagate an initial Cartesian state [r1, v1] for
    ``tof_seconds`` under point-mass gravity from every body in
    ``perturbing_bodies`` (central_body's own point mass is always included),
    using the adaptive DOP853-class integrator (``step_size`` = max step;
    ``integrator="rk4"`` for the legacy fixed-step RK4). Meant to verify a single pykep Lambert
    leg: pass pykep's departure r1 and its solved v1 as the initial state.

    ``departure_epoch_ephemeris_seconds`` is the mission's real departure
    epoch, in SPICE ephemeris seconds since J2000 (2000-01-01 12:00:00 TDB) --
    e.g. via ``orbitopt.bodies.mjd2000_to_ephemeris_seconds``. It anchors the
    propagation's absolute time so perturbing-body SPICE positions (Earth,
    Mars, Jupiter, ...) are sampled at the actual mission date throughout the
    propagation window, not at the J2000 placeholder; leaving it at the
    default of 0.0 places the whole leg at J2000 and can put a perturbing
    body hundreds of millions to over a billion km from its real position.
    """
    _ensure_spice_loaded()

    body_settings = environment_setup.get_default_body_settings(
        list(perturbing_bodies), central_body, "ECLIPJ2000",
    )
    bodies = environment_setup.create_system_of_bodies(body_settings)

    bodies.create_empty_body("Spacecraft")

    acceleration_settings_on_spacecraft = {
        body: [propagation_setup.acceleration.point_mass_gravity()]
        for body in perturbing_bodies
    }
    acceleration_settings = {"Spacecraft": acceleration_settings_on_spacecraft}
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, ["Spacecraft"], [central_body],
    )

    initial_state = np.concatenate([np.asarray(r1, dtype=float), np.asarray(v1, dtype=float)])

    initial_epoch = float(departure_epoch_ephemeris_seconds)
    final_epoch = initial_epoch + float(tof_seconds)

    integrator_settings = _integrator_settings(integrator, step_size, rtol=rtol, atol=atol)
    termination_condition = _time_termination(final_epoch, integrator)

    output_variables = None
    propagator_settings = propagation_setup.propagator.translational(
        [central_body],
        acceleration_models,
        ["Spacecraft"],
        initial_state,
        initial_epoch,
        integrator_settings,
        termination_condition,
    )

    dynamics_simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)
    state_history = dynamics_simulator.propagation_results.state_history

    epochs = np.array(sorted(state_history.keys()))
    states = np.array([state_history[t] for t in epochs])

    model = {
        "function": "propagate_two_body_leg",
        "central_body": central_body,
        "perturbing_bodies": list(perturbing_bodies),
        "earth_spherical_harmonic": None,
        "integrator": integrator_description(integrator, step_size, rtol, atol),
        "initial_state": initial_state,
        "initial_epoch_ephemeris_s": initial_epoch,
        "duration_s": float(tof_seconds),
        "arcs": [{"type": "coast", "duration": float(tof_seconds)}],
    }

    def rerun(**overrides):
        kw = dict(perturbing_bodies=perturbing_bodies, central_body=central_body, step_size=step_size,
                  departure_epoch_ephemeris_seconds=departure_epoch_ephemeris_seconds,
                  integrator=integrator, rtol=rtol, atol=atol)
        overrides.pop("earth_spherical_harmonic", None)
        kw.update(overrides)
        return propagate_two_body_leg(r1, v1, tof_seconds, **kw)

    return PropagationResult(
        epochs=epochs, states=states, final_state=states[-1], epoch_offset=0.0,
        model=model, segments=[(0, len(epochs))], rerun=rerun,
    )


def compare_to_lambert_prediction(prop_result: PropagationResult, r2_lambert, v2_lambert,
                                  report=True, report_path=None):
    """Compare a numerically propagated final state to pykep's patched-conic
    Lambert-arc prediction (r2, v2). Returns a dict of position/velocity
    discrepancies -- large values indicate the two-body assumption breaks
    down for this leg (e.g. a close planetary flyby not modelled as an
    instantaneous v-infinity match).

    With ``report`` (default) and a result from :func:`propagate_two_body_leg`
    / :func:`propagate_multi_arc`, the dict also carries ``"report"``: the
    machine-readable verification report (:mod:`orbitopt.verify.report`),
    written to ``report_path`` if given.
    """
    r2_num, v2_num = prop_result.final_state[:3], prop_result.final_state[3:]
    r2_lambert = np.asarray(r2_lambert, dtype=float)
    v2_lambert = np.asarray(v2_lambert, dtype=float)

    pos_error = np.linalg.norm(r2_num - r2_lambert)
    vel_error = np.linalg.norm(v2_num - v2_lambert)
    out = {
        "position_error_km": pos_error / 1000.0,
        "velocity_error_m_s": vel_error,
    }
    if report and prop_result.model is not None:
        out["report"] = _report.finalize_report(_report.build_report(
            kind="lambert-leg",
            label=f"departure-et-{prop_result.model['initial_epoch_ephemeris_s']:.3f}",
            propagations=[propagation_report_section(prop_result, "leg")],
            residuals=[_report.state_residual(
                "arrival_vs_lambert", "pykep patched-conic Lambert arrival state (r2, v2)",
                state=prop_result.final_state, reference_state=np.concatenate([r2_lambert, v2_lambert]),
            )],
            inputs={"r2_lambert_m": r2_lambert, "v2_lambert_m_s": v2_lambert},
        ), report_path)
    return out


def _propagate_single_coast(bodies, acceleration_models, central_body, spacecraft_state, epoch_start, duration, step_size, integrator=DEFAULT_INTEGRATOR, rtol=VERIFY_RTOL, atol=VERIFY_ATOL):
    integrator_settings = _integrator_settings(integrator, step_size, rtol=rtol, atol=atol)
    termination_condition = _time_termination(epoch_start + float(duration), integrator)
    propagator_settings = propagation_setup.propagator.translational(
        [central_body],
        acceleration_models,
        ["Spacecraft"],
        spacecraft_state,
        epoch_start,
        integrator_settings,
        termination_condition,
    )
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)
    state_history = dynamics_simulator.propagation_results.state_history
    epochs = np.array(sorted(state_history.keys()))
    states = np.array([state_history[t] for t in epochs])
    return epochs, states


class ReusableCoastPropagator:
    """The build-once core of :func:`propagate_multi_arc`, exposed so a caller
    that must drive a mission arc-by-arc in its own Python loop can still pay
    the system-of-bodies + acceleration-model + SPICE setup a single time.

    ``propagate_multi_arc`` builds that dynamical model once and reuses it for
    every arc it's handed -- but only when the whole arc sequence is known
    upfront. A caller whose *next* arc depends on the numerically-propagated
    end state of the current one (e.g. viz.geo_raising applies an analytic
    apsis burn computed from each coast's actual final state, then decimates
    that arc on its own smoothness budget before the next coast) can't hand
    over the full sequence, and so used to call propagate_multi_arc once per
    arc -- rebuilding the identical environment ~9 times per mission. Building
    one of these and calling :meth:`coast` per arc gives byte-identical
    dynamics (same integrator, same bodies, same accelerations) with the setup
    done once. See propagate_multi_arc's docstring for the parameters, which
    mean exactly the same thing here.

    It is also the re-propagation hook used by event location: :meth:`coast`
    accepts a per-call ``max_step`` so a bracketing interval can be
    re-integrated at a fine output spacing with identical dynamics.
    """

    def __init__(
        self,
        central_body="Earth",
        perturbing_bodies=("Earth", "Moon", "Sun"),
        step_size=60.0,
        earth_spherical_harmonic=None,
        integrator=DEFAULT_INTEGRATOR,
        rtol=VERIFY_RTOL,
        atol=VERIFY_ATOL,
    ):
        _ensure_spice_loaded()
        body_settings = environment_setup.get_default_body_settings(
            list(perturbing_bodies), central_body, "ECLIPJ2000",
        )
        bodies = environment_setup.create_system_of_bodies(body_settings)
        bodies.create_empty_body("Spacecraft")

        def _gravity_for(body):
            if body == central_body and earth_spherical_harmonic is not None:
                degree, order = earth_spherical_harmonic
                return propagation_setup.acceleration.spherical_harmonic_gravity(int(degree), int(order))
            return propagation_setup.acceleration.point_mass_gravity()

        acceleration_settings = {"Spacecraft": {body: [_gravity_for(body)] for body in perturbing_bodies}}
        self._bodies = bodies
        self._acceleration_models = propagation_setup.create_acceleration_models(
            bodies, acceleration_settings, ["Spacecraft"], [central_body],
        )
        self._central_body = central_body
        self._step_size = float(step_size)
        self._integrator = integrator
        self._rtol = float(rtol)
        self._atol = float(atol)
        self._perturbing_bodies = tuple(perturbing_bodies)
        self._earth_spherical_harmonic = (
            None if earth_spherical_harmonic is None else tuple(int(x) for x in earth_spherical_harmonic)
        )

    def model_description(self) -> dict:
        """Dynamics + integrator description for verification reports."""
        return {
            "central_body": self._central_body,
            "perturbing_bodies": list(self._perturbing_bodies),
            "earth_spherical_harmonic": None if self._earth_spherical_harmonic is None
            else list(self._earth_spherical_harmonic),
            "integrator": integrator_description(self._integrator, self._step_size, self._rtol, self._atol),
        }

    def coast(self, spacecraft_state, epoch_start, duration, max_step=None):
        """Propagate one coast arc from ``spacecraft_state`` (SI, 6-vector) for
        ``duration`` seconds starting at ``epoch_start`` (seconds), returning
        ``(epochs, states)`` -- the same raw arrays :func:`propagate_multi_arc`
        collects per arc, in the same absolute-epoch convention (i.e. epochs
        start at ``epoch_start``). ``max_step`` overrides the step size (the
        max step for the adaptive integrator) for this call only."""
        step = self._step_size if max_step is None else float(max_step)
        return _propagate_single_coast(
            self._bodies, self._acceleration_models, self._central_body,
            np.asarray(spacecraft_state, dtype=float), float(epoch_start), duration, step,
            self._integrator, self._rtol, self._atol,
        )

    def refine_fn(self, epoch_offset=0.0):
        """Adapter to :data:`orbitopt.verify.events.RefineFn` for histories
        whose epochs are relative to ``epoch_offset`` (absolute tudat time)."""
        def refine(state, t_rel, duration, max_step):
            epochs, states = self.coast(state, epoch_offset + t_rel, duration, max_step=max_step)
            return np.asarray(epochs) - epoch_offset, states
        return refine


def propagate_multi_arc(
    r0,
    v0,
    initial_epoch,
    arcs,
    central_body="Earth",
    perturbing_bodies=("Earth", "Moon", "Sun"),
    step_size=60.0,
    earth_spherical_harmonic=None,
    integrator=DEFAULT_INTEGRATOR,
    rtol=VERIFY_RTOL,
    atol=VERIFY_ATOL,
) -> PropagationResult:
    """Propagate a spacecraft through a sequence of ``arcs`` -- coasts and
    instantaneous impulsive burns -- meant for a multi-phase mission like
    parking orbit -> perigee-raise -> phasing orbit -> TLI -> lunar coast ->
    TCMs -> Earth entry. The system-of-bodies and acceleration models are
    built once (see :class:`ReusableCoastPropagator`) and reused for every
    coast arc, since rebuilding them per arc is wasteful and unnecessary (the
    dynamical model doesn't change, only the initial state/epoch of each
    segment). Burn arcs consume no propagated time; they just add ``delta_v``
    (m/s) to the velocity of the last saved state before the next coast
    resumes from there. The returned PropagationResult concatenates every
    coast arc's history into one continuous timeline; epochs remain relative
    to ``initial_epoch`` (i.e. epoch 0.0 corresponds to ``initial_epoch``, not
    J2000).

    ``earth_spherical_harmonic``: pass ``(degree, order)`` (e.g. ``(2, 2)`` for
    J2 + the J22 tesseral triaxiality) to model the *central* body's gravity
    with a spherical-harmonic field instead of a point mass -- the README's
    named extension point, and the perturbation that drives GEO station-keeping
    (J2 -> orbit-plane precession; J22 -> east-west libration toward the stable
    longitudes). Third bodies in ``perturbing_bodies`` stay point masses (the
    dominant one for GEO is the Sun+Moon N-S inclination drift). ``None`` keeps
    the point-mass-only default (back-compatible with the free-return callers).

    ``integrator``: ``"dop853"`` (default; adaptive RKDP 8(7), ``step_size``
    is the maximum step so the saved history is never coarser than before) or
    ``"rk4"`` (legacy fixed step).
    """
    propagator = ReusableCoastPropagator(
        central_body=central_body, perturbing_bodies=perturbing_bodies,
        step_size=step_size, earth_spherical_harmonic=earth_spherical_harmonic,
        integrator=integrator, rtol=rtol, atol=atol,
    )

    current_state = np.concatenate([np.asarray(r0, dtype=float), np.asarray(v0, dtype=float)])
    current_epoch = float(initial_epoch)

    all_epochs = []
    all_states = []
    segments = []
    n_saved = 0

    for arc in arcs:
        arc_type = arc["type"]
        if arc_type == "coast":
            epochs, states = propagator.coast(current_state, current_epoch, arc["duration"])
            all_epochs.append(epochs)
            all_states.append(states)
            segments.append((n_saved, n_saved + len(epochs)))
            n_saved += len(epochs)
            current_state = states[-1]
            current_epoch = epochs[-1]
        elif arc_type == "impulsive_burn":
            delta_v = np.asarray(arc["delta_v"], dtype=float)
            current_state = current_state.copy()
            current_state[3:] = current_state[3:] + delta_v
        else:
            raise ValueError(f"Unknown arc type: {arc_type!r}")

    if not all_epochs:
        raise ValueError("propagate_multi_arc requires at least one 'coast' arc to produce a history.")

    epochs = np.concatenate(all_epochs) - float(initial_epoch)
    states = np.concatenate(all_states, axis=0)

    model = propagator.model_description()
    model.update(
        function="propagate_multi_arc",
        initial_state=np.concatenate([np.asarray(r0, dtype=float), np.asarray(v0, dtype=float)]),
        initial_epoch_ephemeris_s=float(initial_epoch),
        duration_s=float(epochs[-1]),
        arcs=[
            {"type": "coast", "duration": float(a["duration"])} if a["type"] == "coast"
            else {"type": "impulsive_burn", "delta_v": np.asarray(a["delta_v"], dtype=float)}
            for a in arcs
        ],
    )

    def rerun(**overrides):
        kw = dict(central_body=central_body, perturbing_bodies=perturbing_bodies, step_size=step_size,
                  earth_spherical_harmonic=earth_spherical_harmonic, integrator=integrator,
                  rtol=rtol, atol=atol)
        kw.update(overrides)
        return propagate_multi_arc(r0, v0, initial_epoch, arcs, **kw)

    return PropagationResult(
        epochs=epochs, states=states, final_state=current_state,
        epoch_offset=float(initial_epoch), propagator=propagator,
        model=model, segments=segments, rerun=rerun,
    )


def propagation_report_section(
    prop_result: PropagationResult,
    label: str,
    convergence_check: bool = True,
    conservation_check: bool = True,
    mu=None,
) -> dict:
    """One ``propagations[]`` entry of a verification report for a result of
    :func:`propagate_multi_arc` / :func:`propagate_two_body_leg` -- see
    :func:`orbitopt.verify.report.propagation_section`. ``mu`` defaults to
    the SPICE GM of the propagation's central body."""
    if prop_result.model is None:
        raise ValueError("propagation_report_section needs a PropagationResult from "
                         "propagate_multi_arc / propagate_two_body_leg (model metadata missing)")
    if mu is None:
        mu = body_gravitational_parameter(prop_result.model["central_body"])
    return _report.propagation_section(
        prop_result, label, mu=mu,
        convergence_check=convergence_check, conservation_check=conservation_check,
    )


@dataclass
class ClosestApproachResult:
    epoch: float  # seconds since initial_epoch of the propagation, matching prop_result.epochs
    distance_km: float
    spacecraft_state: np.ndarray  # (6,)


def body_position_at_absolute_epoch(body_name, absolute_epoch, central_body="Earth", frame="ECLIPJ2000"):
    _ensure_spice_loaded()
    position = spice.get_body_cartesian_position_at_epoch(
        body_name, central_body, frame, "NONE", float(absolute_epoch),
    )
    return np.asarray(position, dtype=float).reshape(3)


def body_velocity_at_absolute_epoch(body_name, absolute_epoch, central_body="Earth", frame="ECLIPJ2000"):
    """Same convention as ``body_position_at_absolute_epoch`` but returns the (3,) m/s
    velocity instead -- needed to build a relative (spacecraft - body) velocity, e.g. for
    B-plane flyby targeting (``verify.differential_correction``), where the encounter's
    incoming-asymptote direction depends on the Moon's own motion, not just its position.
    """
    _ensure_spice_loaded()
    state = spice.get_body_cartesian_state_at_epoch(
        target_body_name=body_name, observer_body_name=central_body,
        reference_frame_name=frame, aberration_corrections="NONE",
        ephemeris_time=float(absolute_epoch),
    )
    return np.asarray(state[3:], dtype=float).reshape(3)


def body_gravitational_parameter(body_name):
    """SPICE-reported GM (m^3/s^2) for ``body_name`` -- e.g. for computing the osculating
    two-body orbital elements of a spacecraft's motion relative to that body near a close
    encounter (B-plane targeting).
    """
    _ensure_spice_loaded()
    return float(spice.get_body_gravitational_parameter(body_name))


def spice_body_state_fn(body_name, central_body="Earth", frame="ECLIPJ2000"):
    """``absolute_epoch -> (6,)`` SPICE state of ``body_name`` w.r.t.
    ``central_body`` -- the ephemeris callable :mod:`orbitopt.verify.events`
    expects."""
    _ensure_spice_loaded()

    def state(absolute_epoch):
        return np.asarray(spice.get_body_cartesian_state_at_epoch(
            target_body_name=body_name, observer_body_name=central_body,
            reference_frame_name=frame, aberration_corrections="NONE",
            ephemeris_time=float(absolute_epoch),
        ), dtype=float).reshape(6)
    return state


def _refine_for(prop_result, refine):
    """Resolve the ``refine`` argument of the event finders: True -> use the
    propagator recorded on ``prop_result`` (if any), False/None -> dense
    output only, or an explicit RefineFn."""
    if refine is True:
        if prop_result.propagator is None:
            return None
        return prop_result.propagator.refine_fn(prop_result.epoch_offset)
    if refine is False or refine is None:
        return None
    return refine


def find_closest_approach(
    prop_result: PropagationResult,
    reference_epoch_ephemeris_seconds,
    body_name="Moon",
    central_body="Earth",
    method="event",
    refine=True,
) -> ClosestApproachResult:
    """Closest approach of ``prop_result`` to ``body_name`` (e.g. the Moon).

    ``prop_result``'s epochs are relative to whatever ``initial_epoch`` was
    passed into propagate_multi_arc, so the absolute SPICE ephemeris time for
    history epoch t is ``reference_epoch_ephemeris_seconds + t``.

    ``method="event"`` (default): the minimum is located as a root of the
    relative range rate (r - r_b).(v - v_b) with Brent's method on a cubic
    Hermite dense output of the history, then -- when ``refine`` is True and
    the history came from propagate_multi_arc -- the bracketing interval is
    re-propagated at fine resolution with the same dynamics and re-solved.
    The reported distance therefore no longer depends on the output step.
    ``method="sample"``: legacy nearest-saved-sample scan (kept for
    comparison; its resolution is limited by the step size).
    """
    if method == "sample":
        absolute_epochs = reference_epoch_ephemeris_seconds + prop_result.epochs
        body_positions = np.array([
            body_position_at_absolute_epoch(body_name, t, central_body=central_body)
            for t in absolute_epochs
        ])
        spacecraft_positions = prop_result.states[:, :3]
        distances = np.linalg.norm(spacecraft_positions - body_positions, axis=1)
        idx = int(np.argmin(distances))
        return ClosestApproachResult(
            epoch=float(prop_result.epochs[idx]),
            distance_km=float(distances[idx]) / 1000.0,
            spacecraft_state=prop_result.states[idx].copy(),
        )
    if method != "event":
        raise ValueError(f"Unknown method {method!r}; expected 'event' or 'sample'")

    ev = _events.find_closest_approach_event(
        prop_result.epochs, prop_result.states,
        body_state_fn=spice_body_state_fn(body_name, central_body),
        epoch_offset=float(reference_epoch_ephemeris_seconds),
        refine=_refine_for(prop_result, refine),
    )
    return ClosestApproachResult(epoch=ev.epoch, distance_km=ev.value, spacecraft_state=np.asarray(ev.state).copy())


def find_periapsides(
    prop_result: PropagationResult,
    reference_epoch_ephemeris_seconds=0.0,
    body_name=None,
    central_body="Earth",
    refine=True,
):
    """Every periapsis passage of ``prop_result`` w.r.t. ``body_name`` (None:
    the propagation's own central body, i.e. the frame origin), each located
    by root finding on the range rate. Returns a list of
    :class:`orbitopt.verify.events.Event` with ``value`` = periapsis distance
    in km and ``epoch`` in the history's convention."""
    body_fn = _events._zero_body_state if body_name is None else spice_body_state_fn(body_name, central_body)
    return _events.find_periapsides(
        prop_result.epochs, prop_result.states, body_state_fn=body_fn,
        epoch_offset=float(reference_epoch_ephemeris_seconds),
        refine=_refine_for(prop_result, refine),
    )


def find_soi_crossings(
    prop_result: PropagationResult,
    reference_epoch_ephemeris_seconds,
    body_name,
    soi_radius_km,
    central_body="Sun",
    direction=0,
    refine=True,
):
    """SOI entry (``direction=-1``) / exit (``+1``) / both (0) crossings of a
    sphere of ``soi_radius_km`` around ``body_name`` (see
    :func:`orbitopt.verify.events.soi_radius_km`), located by root finding on
    |r - r_b| - R_soi."""
    return _events.find_sphere_crossings(
        prop_result.epochs, prop_result.states, soi_radius_km,
        body_state_fn=spice_body_state_fn(body_name, central_body),
        epoch_offset=float(reference_epoch_ephemeris_seconds),
        direction=direction, refine=_refine_for(prop_result, refine),
    )


@dataclass
class AltitudeCrossingResult:
    epoch: float  # seconds since initial_epoch of the propagation
    altitude_km: float
    spacecraft_state: np.ndarray  # (6,)


def find_altitude_crossing(
    prop_result: PropagationResult,
    central_body_radius_km,
    threshold_altitude_km,
    method="event",
    refine=True,
):
    """First downward crossing of ``threshold_altitude_km`` above a spherical
    central body of radius ``central_body_radius_km`` (e.g. 121.92 km, the
    400,000 ft Earth entry interface). Returns None if the propagated span
    never goes below the threshold; if the very first sample is already below
    it, that sample is returned (as before).

    ``method="event"`` (default) root-finds |r| - (R + h) on the Hermite dense
    output (plus re-propagation of the bracket when ``refine``);
    ``method="linear"`` is the legacy linear interpolation between the two
    bracketing samples.
    """
    radii_km = np.linalg.norm(prop_result.states[:, :3], axis=1) / 1000.0
    altitudes_km = radii_km - float(central_body_radius_km)
    below = altitudes_km < float(threshold_altitude_km)

    if not np.any(below):
        return None

    crossing_indices = np.nonzero(below)[0]
    idx_after = int(crossing_indices[0])
    if idx_after == 0:
        return AltitudeCrossingResult(
            epoch=float(prop_result.epochs[0]), altitude_km=float(altitudes_km[0]),
            spacecraft_state=prop_result.states[0].copy(),
        )

    if method == "event":
        threshold_radius_km = float(central_body_radius_km) + float(threshold_altitude_km)
        found = _events.find_sphere_crossings(
            prop_result.epochs, prop_result.states, threshold_radius_km,
            direction=-1, refine=_refine_for(prop_result, refine), first_only=True,
        )
        if found:
            ev = found[0]
            return AltitudeCrossingResult(
                epoch=ev.epoch, altitude_km=ev.value - float(central_body_radius_km),
                spacecraft_state=np.asarray(ev.state).copy(),
            )
        # fall through to linear if bracketing failed (shouldn't happen)
    elif method != "linear":
        raise ValueError(f"Unknown method {method!r}; expected 'event' or 'linear'")

    idx_before = idx_after - 1
    alt_before, alt_after = altitudes_km[idx_before], altitudes_km[idx_after]
    if alt_after == alt_before:
        frac = 0.0
    else:
        frac = (float(threshold_altitude_km) - alt_before) / (alt_after - alt_before)
    frac = float(np.clip(frac, 0.0, 1.0))

    epoch_before, epoch_after = prop_result.epochs[idx_before], prop_result.epochs[idx_after]
    state_before, state_after = prop_result.states[idx_before], prop_result.states[idx_after]

    epoch = epoch_before + frac * (epoch_after - epoch_before)
    state = state_before + frac * (state_after - state_before)
    altitude = alt_before + frac * (alt_after - alt_before)

    return AltitudeCrossingResult(epoch=float(epoch), altitude_km=float(altitude), spacecraft_state=state)
