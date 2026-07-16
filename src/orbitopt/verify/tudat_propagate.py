"""High-fidelity verification of pykep-optimized trajectories using tudatpy.

pykep's Lambert/MGA solutions are two-body-per-leg (patched conic):
Keplerian arcs glued together at flyby/manoeuvre points, ignoring all
perturbations. Before trusting an optimized trajectory, propagate the same
initial state through tudatpy's numerical, N-body-capable dynamics (point
masses of all major bodies by default, extensible to SRP/harmonics/drag) and
compare the resulting arrival state against what pykep predicted -- if they
diverge by more than a mission's nav budget, the patched-conic solution needs
mid-course correction modelling, not just a better optimizer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tudatpy import constants, numerical_simulation
from tudatpy.interface import spice
from tudatpy.numerical_simulation import environment_setup, propagation_setup

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


def propagate_two_body_leg(
    r1,
    v1,
    tof_seconds,
    perturbing_bodies=("Sun", "Earth", "Mars", "Jupiter"),
    central_body="Sun",
    step_size=3600.0,
    save_history=True,
    departure_epoch_ephemeris_seconds=0.0,
) -> PropagationResult:
    """Numerically propagate an initial Cartesian state [r1, v1] for
    ``tof_seconds`` under point-mass gravity from every body in
    ``perturbing_bodies`` (central_body's own point mass is always included),
    using a fixed-step RK4 integrator. Meant to verify a single pykep Lambert
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

    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
        step_size, propagation_setup.integrator.CoefficientSets.rk_4,
    )
    termination_condition = propagation_setup.propagator.time_termination(final_epoch)

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

    return PropagationResult(epochs=epochs, states=states, final_state=states[-1])


def compare_to_lambert_prediction(prop_result: PropagationResult, r2_lambert, v2_lambert):
    """Compare a numerically propagated final state to pykep's patched-conic
    Lambert-arc prediction (r2, v2). Returns a dict of position/velocity
    discrepancies -- large values indicate the two-body assumption breaks
    down for this leg (e.g. a close planetary flyby not modelled as an
    instantaneous v-infinity match).
    """
    r2_num, v2_num = prop_result.final_state[:3], prop_result.final_state[3:]
    r2_lambert = np.asarray(r2_lambert, dtype=float)
    v2_lambert = np.asarray(v2_lambert, dtype=float)

    pos_error = np.linalg.norm(r2_num - r2_lambert)
    vel_error = np.linalg.norm(v2_num - v2_lambert)
    return {
        "position_error_km": pos_error / 1000.0,
        "velocity_error_m_s": vel_error,
    }


def _propagate_single_coast(bodies, acceleration_models, central_body, spacecraft_state, epoch_start, duration, step_size):
    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
        step_size, propagation_setup.integrator.CoefficientSets.rk_4,
    )
    termination_condition = propagation_setup.propagator.time_termination(epoch_start + float(duration))
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


def propagate_multi_arc(
    r0,
    v0,
    initial_epoch,
    arcs,
    central_body="Earth",
    perturbing_bodies=("Earth", "Moon", "Sun"),
    step_size=60.0,
    earth_spherical_harmonic=None,
) -> PropagationResult:
    """Propagate a spacecraft through a sequence of ``arcs`` -- coasts and
    instantaneous impulsive burns -- meant for a multi-phase mission like
    parking orbit -> perigee-raise -> phasing orbit -> TLI -> lunar coast ->
    TCMs -> Earth entry. The system-of-bodies and acceleration models are
    built once and reused for every coast arc, since rebuilding them per arc
    is wasteful and unnecessary (the dynamical model doesn't change, only the
    initial state/epoch of each segment). Burn arcs consume no propagated
    time; they just add ``delta_v`` (m/s) to the velocity of the last saved
    state before the next coast resumes from there. The returned
    PropagationResult concatenates every coast arc's history into one
    continuous timeline; epochs remain relative to ``initial_epoch`` (i.e.
    epoch 0.0 corresponds to ``initial_epoch``, not J2000).

    ``earth_spherical_harmonic``: pass ``(degree, order)`` (e.g. ``(2, 2)`` for
    J2 + the J22 tesseral triaxiality) to model the *central* body's gravity
    with a spherical-harmonic field instead of a point mass -- the README's
    named extension point, and the perturbation that drives GEO station-keeping
    (J2 -> orbit-plane precession; J22 -> east-west libration toward the stable
    longitudes). Third bodies in ``perturbing_bodies`` stay point masses (the
    dominant one for GEO is the Sun+Moon N-S inclination drift). ``None`` keeps
    the point-mass-only default (back-compatible with the free-return callers).
    """
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

    acceleration_settings_on_spacecraft = {body: [_gravity_for(body)] for body in perturbing_bodies}
    acceleration_settings = {"Spacecraft": acceleration_settings_on_spacecraft}
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, ["Spacecraft"], [central_body],
    )

    current_state = np.concatenate([np.asarray(r0, dtype=float), np.asarray(v0, dtype=float)])
    current_epoch = float(initial_epoch)

    all_epochs = []
    all_states = []

    for arc in arcs:
        arc_type = arc["type"]
        if arc_type == "coast":
            epochs, states = _propagate_single_coast(
                bodies, acceleration_models, central_body, current_state,
                current_epoch, arc["duration"], step_size,
            )
            all_epochs.append(epochs)
            all_states.append(states)
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

    return PropagationResult(epochs=epochs, states=states, final_state=current_state)


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


def find_closest_approach(
    prop_result: PropagationResult,
    reference_epoch_ephemeris_seconds,
    body_name="Moon",
    central_body="Earth",
) -> ClosestApproachResult:
    """Post-hoc scan of an already-saved ``prop_result`` state history to find
    the saved sample closest to ``body_name`` (e.g. the Moon). ``prop_result``'s
    epochs are relative to whatever ``initial_epoch`` was passed into
    propagate_multi_arc, which is an arbitrary reference, not J2000 -- so the
    absolute SPICE ephemeris time for sample i is
    ``reference_epoch_ephemeris_seconds + prop_result.epochs[i]``. This is
    deliberately simple nearest-saved-sample analysis rather than a proper
    tudatpy termination-event search: it's robust and good enough for
    research use as long as ``step_size`` in propagate_multi_arc is fine
    enough (e.g. <=300 s) during the arc expected to contain the closest
    approach; coarser sampling just means coarser resolution on the reported
    minimum, not an incorrect one.
    """
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


@dataclass
class AltitudeCrossingResult:
    epoch: float  # seconds since initial_epoch of the propagation, linearly interpolated
    altitude_km: float
    spacecraft_state: np.ndarray  # (6,), linearly interpolated


def find_altitude_crossing(prop_result: PropagationResult, central_body_radius_km, threshold_altitude_km):
    """Post-hoc scan of ``prop_result`` for the first sample-to-sample interval
    in which altitude above a spherical central body of radius
    ``central_body_radius_km`` crosses below ``threshold_altitude_km`` (e.g.
    121.92 km for the 400,000 ft Earth entry interface), linearly
    interpolating epoch/state between the two bracketing saved samples for a
    more precise crossing. Returns None if the propagated span never crosses
    the threshold.
    """
    radii_km = np.linalg.norm(prop_result.states[:, :3], axis=1) / 1000.0
    altitudes_km = radii_km - float(central_body_radius_km)
    below = altitudes_km < float(threshold_altitude_km)

    if not np.any(below):
        return None

    crossing_indices = np.nonzero(below)[0]
    idx_after = int(crossing_indices[0])
    if idx_after == 0:
        idx_before = 0
    else:
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
