"""M-05 (task #223): independent TudatPy propagation of the injected orbit.

The end-to-end chain (#215) hands off the parking-orbit state at SECO; M-02
(:mod:`orbitopt.mission.ascent`) proved that state *is* the ascent's orbit.
This module propagates it forward ``N`` revolutions with tudatpy -- a
different physics stack than the 3-DOF ascent that produced it -- with a
configurable force model (J2, exponential-atmosphere drag, third-body point
masses), and quantifies how the high-fidelity orbit diverges from the
analytic two-body reference along the way. The acceptance evidence is a
verification report, not a hidden pass: for a pure point-mass run the
numerical-minus-analytic difference is integration error and gets a
tolerance gate; for perturbed runs the difference is physics (J2 drift, drag
decay) and is recorded without a gate.

Frame and time handling follow the M-01 contract and the FL-006 re-fly:

* the handoff is consumed in its contract frame; anything not GCRS is
  bridged through :func:`orbitopt.flightlab.frames.to_gcrs` *here*, inside
  the segment (never downstream);
* the tudatpy propagation runs in ECLIPJ2000 (the frame
  ``environment_setup.get_default_body_settings`` is built around), reached
  by the SPICE rotation GCRS -> ECLIPJ2000 at the handoff epoch, exactly like
  ``orbitopt.flightlab.simulate.orbit_refly``; every saved sample is rotated
  back ECLIPJ2000 -> GCRS at its own epoch, so the final handoff is a
  same-frame state the contract gate can measure;
* the epoch is the state's :class:`~orbitopt.mission.contract.Epoch` in TDB
  seconds (:meth:`Epoch.to_tdb_seconds`), tudatpy's own ephemeris scale.

Central gravity is overridden to the package's (GM, radius, J2) triple --
same constants as :mod:`orbitopt.flightlab.coast` and
:mod:`orbitopt.injection` -- so the analytic reference and the numerical
propagation integrate the *same* two-body problem and the report's
conservation re-run uses the same mu (see
``orbitopt.verify.tudat_propagate._override_central_gravity``).

    from orbitopt.mission.ascent import ascent_handoff_state
    from orbitopt.mission.propagation import (
        DragConfig, PropagationForceModel, OrbitPropagationSegment)

    segment = OrbitPropagationSegment(
        force_model=PropagationForceModel(j2=True, third_bodies=("Sun", "Moon")),
        revolutions=2.0, report_path="reports/m05.json")
    result = segment.run(ascent_handoff_state(run))

Unlike the contract package this module needs the compute stack (numpy, and
tudatpy lazily, only when a propagation actually runs); it is deliberately
not re-exported from ``orbitopt.mission.__init__``, so importing the
contract never drags tudatpy in.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from orbitopt.flightlab.coast import coast_arc
from orbitopt.flightlab.checks import orbital_elements
from orbitopt.mission.contract import Epoch, HandoffState, SegmentResult, validate_state
from orbitopt.timescales import SECONDS_PER_DAY, mjd2000_to_utc_datetime
from orbitopt.verify import report as _report

__all__ = [
    "DragConfig",
    "EARTH_GRAVITY",
    "J2_EARTH",
    "MU_EARTH",
    "R_EARTH",
    "InjectedOrbitPropagation",
    "OrbitPropagationSegment",
    "PropagationForceModel",
    "analytic_two_body_comparison",
    "element_history",
    "handoff_to_tudat_initial_state",
    "orbit_propagation_report",
    "propagate_injected_orbit",
    "propagation_from_arrays",
]

#: Earth (GM, radius, J2) used by BOTH sides of the comparison: the tudat
#: central-gravity override and the analytic Kepler reference. Same triple
#: as orbitopt.flightlab.coast / orbitopt.injection.
MU_EARTH = 3.986004418e14  # m^3/s^2
R_EARTH = 6378137.0  # m
J2_EARTH = 1.08262668e-3  # -
EARTH_GRAVITY = {"gravitational_parameter": MU_EARTH, "reference_radius": R_EARTH,
                 "j2": J2_EARTH}

#: Central body of every M-05 propagation (a parking orbit goes round Earth).
CENTRAL_BODY = "Earth"

#: Maximum step (s) of the adaptive integrator -- also the floor on how
#: coarsely the saved history can sample the arc.
DEFAULT_MAX_STEP_S = 300.0
#: Revolutions is a mission parameter, not a physical one; the default just
#: has to cover at least one full orbit.
DEFAULT_REVOLUTIONS = 1.0
#: Samples of the osculating-element time series written into the report.
DEFAULT_HISTORY_SAMPLES = 240
#: Samples of the analytic (two-body Kepler) comparison along the arc.
DEFAULT_COMPARISON_SAMPLES = 96

#: Gate limits applied to the analytic comparison ONLY for a pure point-mass
#: run, where numerical-minus-analytic is integration error. A DOP853 run at
#: the verify tolerances over a handful of LEO revolutions is centimetre
#: level, so 100 m / 0.01 m/s flags a broken handoff, not physics.
POINT_MASS_POS_TOL_M = 100.0
POINT_MASS_VEL_TOL_M_S = 0.01

#: tudatpy frame the propagation runs in (the frame the default body settings
#: are built around), and the SPICE frame aliases probed at the boundaries
#: (mirrors ``orbitopt.flightlab.simulate._FRAME_ALIASES`` without importing
#: that module's CLI plumbing).
_TUDAT_FRAME = "ECLIPJ2000"
_FRAME_ALIASES = {
    "GCRS": ("J2000", "ICRS", "ICRF"),
    "ICRS": ("GCRS", "J2000", "ICRF"),
    "J2000": ("GCRS", "ICRS", "ICRF"),
    "ICRF": ("GCRS", "ICRS", "J2000"),
    _TUDAT_FRAME: ("B1950", "ECLIP"),
}


# --------------------------------------------------------------------------
# Force model configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DragConfig:
    """Constant-drag-coefficient ball through an exponential atmosphere --
    the classic low-Earth parking-orbit decay model (Cd 2.2, rho0 4.947e-13
    kg/m^3 with a 51.65 km scale height, the pair the tudatpy perturbed-orbit
    examples use)."""

    drag_coefficient: float = 2.2
    reference_area_m2: float = 20.0
    reference_density_kg_m3: float = 4.947e-13
    scale_height_m: float = 51650.0

    def __post_init__(self):
        for name in ("drag_coefficient", "reference_area_m2", "reference_density_kg_m3",
                     "scale_height_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"DragConfig.{name} must be finite and > 0, got {value}")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict:
        return {
            "model": "exponential_atmosphere_constant_cd",
            "drag_coefficient": self.drag_coefficient,
            "reference_area_m2": self.reference_area_m2,
            "reference_density_kg_m3": self.reference_density_kg_m3,
            "scale_height_m": self.scale_height_m,
        }


@dataclass(frozen=True)
class PropagationForceModel:
    """The configurable force model of an M-05 propagation: Earth gravity
    (point mass, or degree-2 field with ``j2=True``), optional atmospheric
    drag, and any number of third-body point masses (``("Sun", "Moon")`` is
    the pair that drives the LEO inclination drift)."""

    j2: bool = True
    drag: DragConfig | None = None
    third_bodies: tuple[str, ...] = ()

    def __post_init__(self):
        bodies = tuple(str(b) for b in self.third_bodies)
        for body in bodies:
            if not body:
                raise ValueError("third_bodies entries must be non-empty body names")
        if CENTRAL_BODY in bodies:
            raise ValueError(
                f"{CENTRAL_BODY} is the central body, not a third body "
                "(its gravity is the j2 / point-mass term)"
            )
        object.__setattr__(self, "third_bodies", tuple(dict.fromkeys(bodies)))

    @property
    def bodies(self) -> tuple[str, ...]:
        """Bodies tudat must create: central body first, then third bodies."""
        return (CENTRAL_BODY,) + self.third_bodies

    def to_dict(self) -> dict:
        return {
            "central_body": CENTRAL_BODY,
            "central_gravity": dict(EARTH_GRAVITY),
            "j2": bool(self.j2),
            "drag": None if self.drag is None else self.drag.to_dict(),
            "third_bodies": list(self.third_bodies),
        }


def _force_inventory(force_model: PropagationForceModel) -> dict:
    """Force inventory for the report's dynamics block: gravity entries via
    the shared verify reporter, plus the drag line when configured (the
    reporter only knows about gravity)."""
    inventory = _report.perturbation_inventory(
        CENTRAL_BODY, list(force_model.bodies),
        (2, 0) if force_model.j2 else None, frame=_TUDAT_FRAME)
    if force_model.drag is not None:
        inventory["forces"].append(
            {"body": CENTRAL_BODY, "role": "central", "type": "aerodynamic_drag",
             **force_model.drag.to_dict()})
        inventory["not_modelled"] = [line for line in inventory["not_modelled"]
                                     if "atmospheric_drag" not in line]
        inventory["summary"]["atmospheric_drag"] = True
        inventory["summary"]["central_point_mass_only"] = False
    return inventory


# --------------------------------------------------------------------------
# Handoff state <-> tudat initial condition / saved history
# --------------------------------------------------------------------------


def _to_gcrs(state: HandoffState) -> HandoffState:
    """The state with r/v in GCRS: a GCRS state passes through, an
    ``ECEF``/``LAUNCH_ECI`` one goes through the Flight Lab frame bridge
    *inside this segment*, as the M-01 handoff rules demand."""
    if state.frame == "GCRS":
        return state
    from orbitopt.flightlab.frames import to_gcrs  # lazy: numpy + astropy

    epoch_utc = mjd2000_to_utc_datetime(state.epoch.to_utc_mjd2000())
    r_g, v_g = to_gcrs([tuple(state.r_m)], [tuple(state.v_m_s)], state.frame, epoch_utc, [0.0])
    return HandoffState(
        epoch=state.epoch, frame="GCRS",
        r_m=tuple(float(x) for x in np.asarray(r_g)[0]),
        v_m_s=tuple(float(x) for x in np.asarray(v_g)[0]),
        mass_kg=state.mass_kg, stage_config=state.stage_config,
        metadata={**state.metadata, "m05_frame_bridged_from": state.frame},
    )


def _spice_rotation(spice, from_frame: str, to_frame: str, et: float) -> np.ndarray:
    """3x3 SPICE rotation ``from_frame`` -> ``to_frame`` at ephemeris second
    ``et``, probed through frame aliases (GCRS/J2000/ICRS) like the FL-006
    re-fly. The LAST alias's error is raised -- if the canonical pair fails,
    that is the error worth seeing."""
    last_error: Exception | None = None
    for src in (from_frame,) + _FRAME_ALIASES.get(from_frame, ()):
        for dst in (to_frame,) + _FRAME_ALIASES.get(to_frame, ()):
            try:
                return np.asarray(spice.compute_rotation_matrix_between_frames(
                    src, dst, float(et)), dtype=float)
            except Exception as exc:  # noqa: BLE001 -- alias probing
                last_error = exc
    raise RuntimeError(f"SPICE cannot rotate {from_frame!r} -> {to_frame!r} at et={et}") from last_error


def handoff_to_tudat_initial_state(state: HandoffState) -> tuple[np.ndarray, float]:
    """The M-05 handoff -> the tudat propagation initial condition: the
    6-vector [x, y, z, vx, vy, vz] (m, m/s) in the propagation frame
    (ECLIPJ2000) and the ephemeris-seconds-since-J2000 TDB epoch. The state
    is bridged to GCRS if its contract frame is not inertial GCRS, the epoch
    goes through the contract's own timescale conversion, and the vector is
    rotated GCRS -> ECLIPJ2000 with SPICE at that exact epoch -- the same
    boundary treatment as ``orbitopt.flightlab.simulate.orbit_refly``.
    Imports tudatpy (needs the compute stack installed)."""
    from tudatpy.interface import spice

    from orbitopt.verify.tudat_propagate import _ensure_spice_loaded

    validate_state(state)
    gcrs = _to_gcrs(state)
    et = gcrs.epoch.to_tdb_seconds()
    _ensure_spice_loaded()
    rot = _spice_rotation(spice, "GCRS", _TUDAT_FRAME, et)
    state6 = np.concatenate((np.asarray(gcrs.r_m, dtype=float),
                             np.asarray(gcrs.v_m_s, dtype=float)))
    return _rotate6(rot, state6), float(et)


def _rotate6(rot: np.ndarray, state6: np.ndarray) -> np.ndarray:
    """Apply a 3x3 frame rotation to both blocks of a 6-vector state."""
    out = np.empty(6)
    out[:3] = rot @ state6[:3]
    out[3:] = rot @ state6[3:]
    return out


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------


@dataclass
class InjectedOrbitPropagation:
    """One M-05 propagation: the saved history in both frames, the osculating
    element history, the analytic two-body comparison and the ``rerun`` hook
    the verification report's convergence/conservation checks drive."""

    initial_state: HandoffState
    force_model: PropagationForceModel
    epochs_s: np.ndarray  # (n,) seconds after the handoff epoch
    states_ecliptic: np.ndarray  # (n, 6) propagation frame (ECLIPJ2000), SI
    states_gcrs: np.ndarray  # (n, 6) contract frame, SI
    et0: float  # ephemeris seconds of the first sample
    period_s: float
    duration_s: float
    revolutions: float
    elements_history: tuple[Mapping[str, Any], ...] = ()
    analytic: Mapping[str, Any] = field(default_factory=dict)
    integrator: str = "dop853"
    max_step_s: float = DEFAULT_MAX_STEP_S
    rtol: float = 1e-11
    atol: float = 1e-6
    rerun: Callable[..., Any] | None = field(default=None, repr=False)

    @property
    def final_state_gcrs(self) -> np.ndarray:
        return self.states_gcrs[-1]

    @property
    def central_mu(self) -> float:
        return MU_EARTH

    @property
    def point_mass_only(self) -> bool:
        """True when the only acceleration is Earth's point mass -- then
        numerical-minus-analytic is pure integration error and the report
        gates it."""
        fm = self.force_model
        return not fm.j2 and fm.drag is None and not fm.third_bodies

    def elements_at(self, index: int) -> dict:
        state = self.states_gcrs[int(index)]
        return orbital_elements(state[:3], state[3:], MU_EARTH)

    def initial_elements(self) -> dict:
        return self.elements_at(0)

    def final_elements(self) -> dict:
        return self.elements_at(-1)

    def model_dict(self) -> dict:
        """Report ``propagations[0]`` model metadata -- the keys the
        verify-report propagation section reads off ``PropagationResult.model``
        (see ``tudat_propagate.propagation_report_section``)."""
        from orbitopt.verify.tudat_propagate import integrator_description

        return {
            "function": "propagate_injected_orbit",
            "central_body": CENTRAL_BODY,
            "perturbing_bodies": list(self.force_model.bodies),
            "earth_spherical_harmonic": [2, 0] if self.force_model.j2 else None,
            "integrator": integrator_description(self.integrator, self.max_step_s,
                                                 self.rtol, self.atol),
            "central_gravitational_parameter": MU_EARTH,
            "central_gravity": dict(EARTH_GRAVITY),
            "initial_state": self.states_ecliptic[0],
            "initial_epoch_ephemeris_s": self.et0,
            "duration_s": self.duration_s,
            "arcs": [{"type": "coast", "duration": self.duration_s}],
        }


def _to_propagation_result(prop: InjectedOrbitPropagation):
    """A ``PropagationResult`` view of an M-05 propagation, so the shared
    verify-report machinery (step statistics, convergence re-run, two-body
    conservation) processes it unchanged. ``epochs`` stay history-relative;
    the absolute start is recorded as ``epoch_offset``."""
    from orbitopt.verify.tudat_propagate import PropagationResult

    return PropagationResult(
        epochs=prop.epochs_s, states=prop.states_ecliptic,
        final_state=prop.states_ecliptic[-1], epoch_offset=prop.et0,
        model=prop.model_dict(), segments=[(0, int(len(prop.epochs_s)))],
        rerun=prop.rerun,
    )


# --------------------------------------------------------------------------
# Analytic reference + element history
# --------------------------------------------------------------------------


def _hermite(epochs: np.ndarray, states: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Cubic Hermite interpolation of a saved (position, velocity) history at
    query times ``t`` -- exact-shape interpolation between saved samples (a
    linear interp of a 300 s LEO step is kilometres of sagitta). Positions
    interpolate with the sample velocities as derivatives, velocities with
    the derivative of the same cubic."""
    epochs = np.asarray(epochs, dtype=float)
    states = np.asarray(states, dtype=float)
    t = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.empty((len(t), 6))
    idx = np.clip(np.searchsorted(epochs, t) - 1, 0, len(epochs) - 2)
    for k, q in enumerate(t):
        i = int(idx[k])
        h = float(epochs[i + 1] - epochs[i])
        s = 0.0 if h <= 0.0 else float((q - epochs[i]) / h)
        h00 = 2 * s ** 3 - 3 * s ** 2 + 1
        h10 = s ** 3 - 2 * s ** 2 + s
        h01 = -2 * s ** 3 + 3 * s ** 2
        h11 = s ** 3 - s ** 2
        out[k, :3] = (h00 * states[i, :3] + h10 * h * states[i, 3:]
                      + h01 * states[i + 1, :3] + h11 * h * states[i + 1, 3:])
        if h > 0.0:
            dh00 = (6 * s ** 2 - 6 * s) / h
            dh10 = (3 * s ** 2 - 4 * s + 1) / h
            dh01 = (-6 * s ** 2 + 6 * s) / h
            dh11 = (3 * s ** 2 - 2 * s) / h
            out[k, 3:] = (dh00 * states[i, :3] + dh10 * h * states[i, 3:]
                          + dh01 * states[i + 1, :3] + dh11 * h * states[i + 1, 3:])
        else:
            out[k, 3:] = states[i, 3:]
    return out


def analytic_two_body_comparison(propagation: InjectedOrbitPropagation,
                                 samples: int = DEFAULT_COMPARISON_SAMPLES) -> dict:
    """Numerical-minus-analytic discrepancy along the arc, in the propagation
    frame: the two-body Kepler solution of the same initial state and GM
    (:func:`orbitopt.flightlab.coast.coast_arc`) on a uniform time grid,
    against the saved history Hermite-interpolated to that grid. Sample 0 is
    the shared initial condition and is excluded from the maxima. Recorded
    (not gated) unless the run is pure point mass, where it is pure
    integration error."""
    if samples < 2:
        raise ValueError(f"comparison samples must be >= 2, got {samples}")
    times, positions, velocities = coast_arc(
        propagation.states_ecliptic[0, :3], propagation.states_ecliptic[0, 3:],
        propagation.duration_s, samples=int(samples), mu=MU_EARTH)
    numeric = _hermite(propagation.epochs_s, propagation.states_ecliptic,
                       np.asarray(times, dtype=float))
    dr = np.linalg.norm(numeric[:, :3] - np.asarray(positions, dtype=float), axis=1)
    dv = np.linalg.norm(numeric[:, 3:] - np.asarray(velocities, dtype=float), axis=1)
    return {
        "reference": "orbitopt.flightlab.coast.coast_arc (two-body Kepler, same GM)",
        "frame": _TUDAT_FRAME,
        "samples": int(samples),
        "excluded_initial_sample": True,
        "max_position_m": float(dr[1:].max()),
        "max_velocity_m_s": float(dv[1:].max()),
        "final_position_m": float(dr[-1]),
        "final_velocity_m_s": float(dv[-1]),
    }


def element_history(propagation: InjectedOrbitPropagation,
                    samples: int = DEFAULT_HISTORY_SAMPLES) -> tuple[Mapping[str, Any], ...]:
    """Osculating Kepler elements (GCRS, same convention as M-02's
    ``orbitopt.mission.ascent.parking_orbit_elements``) along the saved
    history: rows at uniformly spread saved samples, first and last always
    included, each carrying its time since the handoff epoch."""
    if samples < 2:
        raise ValueError(f"history samples must be >= 2, got {samples}")
    n = int(len(propagation.epochs_s))
    rows = []
    for i in _sample_indices(n, samples):
        state = propagation.states_gcrs[i]
        rows.append({"t_s": float(propagation.epochs_s[i]),
                     **orbital_elements(state[:3], state[3:], MU_EARTH)})
    return tuple(rows)


def _sample_indices(n_saved: int, n_rows: int) -> list[int]:
    """Indices of the saved history the element/state history rows are built
    from: uniformly spread, first and last always included."""
    return [int(i) for i in np.unique(
        np.linspace(0, int(n_saved) - 1, min(int(n_rows), int(n_saved))).round().astype(int))]


# --------------------------------------------------------------------------
# Propagation
# --------------------------------------------------------------------------


def _aerodynamic_acceleration(environment_setup, propagation_setup, body_settings,
                              drag: DragConfig):
    """Constant-Cd aerodynamic acceleration through an exponential Earth
    atmosphere, across the tudatpy API generations: modern (>= 1.0) declares
    the coefficient interface on the body settings and the acceleration takes
    no arguments; the legacy form took (Cd, reference area) positionally. The
    atmosphere module name went singular/plural once, so both are probed."""
    atmosphere_mod = getattr(environment_setup, "atmosphere", None) \
        or getattr(environment_setup, "atmospheres", None)
    if atmosphere_mod is None:
        raise RuntimeError("this tudatpy has no environment_setup.atmosphere[s] module")
    body_settings.get(CENTRAL_BODY).atmosphere_settings = \
        atmosphere_mod.exponential_atmosphere(drag.reference_density_kg_m3, drag.scale_height_m)
    coefficients = getattr(environment_setup, "aerodynamic_coefficients", None)
    if coefficients is not None and hasattr(coefficients, "constant"):
        body_settings.get("Spacecraft").aerodynamic_coefficient_settings = coefficients.constant(
            drag.reference_area_m2, [drag.drag_coefficient, 0.0, 0.0])
        return propagation_setup.acceleration.aerodynamic()
    return propagation_setup.acceleration.aerodynamic(
        drag.drag_coefficient, drag.reference_area_m2)


def propagate_injected_orbit(
    state: HandoffState,
    force_model: PropagationForceModel | None = None,
    revolutions: float = DEFAULT_REVOLUTIONS,
    *,
    max_step_s: float = DEFAULT_MAX_STEP_S,
    integrator: str = "dop853",
    rtol: float = 1e-11,
    atol: float = 1e-6,
    history_samples: int = DEFAULT_HISTORY_SAMPLES,
    comparison_samples: int = DEFAULT_COMPARISON_SAMPLES,
) -> InjectedOrbitPropagation:
    """Propagate the injected state ``revolutions`` parking-orbit revolutions
    with tudatpy under ``force_model`` (default: J2), returning the saved
    history in the propagation frame and the contract frame, the osculating
    element history and the two-body analytic comparison.

    Duration is ``revolutions`` x the initial osculating period (from the
    state's own GCRS elements under :data:`MU_EARTH`); the arc must be a
    closed orbit. The whole history is saved (the adaptive integrator saves
    every accepted step) and rotated back to GCRS sample by sample, so
    ``states_gcrs[-1]`` is a contract-frame state the next handoff can use.
    The result carries a ``rerun`` hook honouring the verify-report overrides
    (``rtol``/``atol``/``step_size``/``perturbing_bodies``/
    ``earth_spherical_harmonic``) so the step-convergence and two-body
    conservation checks run on the identical model.
    """
    from orbitopt.verify.tudat_propagate import _ensure_spice_loaded, _override_central_gravity
    from tudatpy.numerical_simulation import environment_setup, propagation_setup

    validate_state(state)
    fm = force_model if force_model is not None else PropagationForceModel()
    revs = float(revolutions)
    if not math.isfinite(revs) or revs <= 0.0:
        raise ValueError(f"revolutions must be finite and > 0, got {revolutions!r}")
    if not (float(max_step_s) > 0.0) or int(history_samples) < 2 or int(comparison_samples) < 2:
        raise ValueError("max_step_s must be > 0 and sample counts >= 2")

    gcrs = _to_gcrs(state)
    el0 = orbital_elements(gcrs.r_m, gcrs.v_m_s, MU_EARTH)
    if not math.isfinite(el0["period_s"]) or el0["period_s"] <= 0.0:
        raise ValueError(f"injected state is not on a closed (elliptical) orbit: {el0}")
    duration = revs * el0["period_s"]

    _ensure_spice_loaded()
    state6_ecl, et0 = handoff_to_tudat_initial_state(gcrs)

    body_settings = environment_setup.get_default_body_settings(
        list(fm.bodies), CENTRAL_BODY, _TUDAT_FRAME)
    central_gravity = _override_central_gravity(body_settings, CENTRAL_BODY, EARTH_GRAVITY)
    if fm.drag is not None:
        drag_acceleration = _aerodynamic_acceleration(
            environment_setup, propagation_setup, body_settings, fm.drag)
    bodies = environment_setup.create_system_of_bodies(body_settings)
    bodies.create_empty_body("Spacecraft")
    if fm.drag is not None:
        try:
            bodies.get("Spacecraft").mass = gcrs.mass_kg
        except AttributeError as exc:
            raise RuntimeError("this tudatpy cannot set the spacecraft mass; drag "
                               "propagation would integrate an undefined-mass ball") from exc

    earth_accelerations = [
        propagation_setup.acceleration.spherical_harmonic_gravity(2, 0) if fm.j2
        else propagation_setup.acceleration.point_mass_gravity()]
    if fm.drag is not None:
        earth_accelerations.append(drag_acceleration)
    spacecraft_accelerations = {CENTRAL_BODY: earth_accelerations}
    for body in fm.third_bodies:
        spacecraft_accelerations[body] = [propagation_setup.acceleration.point_mass_gravity()]
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, {"Spacecraft": spacecraft_accelerations}, ["Spacecraft"], [CENTRAL_BODY])

    from orbitopt.verify.tudat_propagate import _propagate_single_coast

    epochs, states = _propagate_single_coast(
        bodies, acceleration_models, CENTRAL_BODY, state6_ecl, et0, duration,
        max_step_s, integrator, rtol, atol)

    prop = propagation_from_arrays(
        gcrs, fm, revs, epochs, states, et0, el0["period_s"],
        integrator=integrator, max_step_s=max_step_s, rtol=rtol, atol=atol)
    prop.model_dict = _memo_model_dict(prop, central_gravity)  # type: ignore[method-assign]
    return _finish(prop, history_samples, comparison_samples)


def _memo_model_dict(prop: InjectedOrbitPropagation, central_gravity: dict):
    """Model metadata with the *actual* override the propagation applied
    (``_override_central_gravity``'s normalised C20 and frame) folded in."""
    base = prop.model_dict()

    def model_dict() -> dict:
        doc = dict(base)
        doc["central_gravity"] = dict(central_gravity)
        return doc

    return model_dict


def propagation_from_arrays(
    gcrs_state: HandoffState, force_model: PropagationForceModel, revolutions: float,
    epochs: np.ndarray, states_ecliptic: np.ndarray, et0: float, period_s: float,
    *, integrator: str = "dop853", max_step_s: float = DEFAULT_MAX_STEP_S,
    rtol: float = 1e-11, atol: float = 1e-6, rotate_to_gcrs: bool = True,
) -> InjectedOrbitPropagation:
    """Assemble an :class:`InjectedOrbitPropagation` from a raw saved history
    (tudat absolute epochs + propagation-frame states), rotating the samples
    back to GCRS at their own epochs. Split out of
    :func:`propagate_injected_orbit` so the post-processing is testable
    against a recorded history without running a propagation. Set
    ``rotate_to_gcrs=False`` when the history is already in GCRS (the fake-SPICE
    test path)."""
    epochs = np.asarray(epochs, dtype=float)
    states_ecliptic = np.asarray(states_ecliptic, dtype=float)
    if epochs.ndim != 1 or len(epochs) < 2:
        raise ValueError("a propagation history needs at least two saved samples")
    if states_ecliptic.shape != (len(epochs), 6):
        raise ValueError(f"states {states_ecliptic.shape} do not line up with epochs {len(epochs)}")
    if rotate_to_gcrs:
        from tudatpy.interface import spice

        from orbitopt.verify.tudat_propagate import _ensure_spice_loaded

        _ensure_spice_loaded()
        gcrs = np.empty_like(states_ecliptic)
        for i, t in enumerate(epochs):
            rot = _spice_rotation(spice, _TUDAT_FRAME, "GCRS", et0 + float(t))
            gcrs[i, :3] = rot @ states_ecliptic[i, :3]
            gcrs[i, 3:] = rot @ states_ecliptic[i, 3:]
    else:
        gcrs = states_ecliptic.copy()
    return InjectedOrbitPropagation(
        initial_state=gcrs_state, force_model=force_model,
        epochs_s=epochs - float(et0), states_ecliptic=states_ecliptic, states_gcrs=gcrs,
        et0=float(et0), period_s=float(period_s),
        duration_s=float(epochs[-1] - et0), revolutions=float(revolutions),
        integrator=str(integrator), max_step_s=float(max_step_s),
        rtol=float(rtol), atol=float(atol),
    )


def _finish(prop: InjectedOrbitPropagation, history_samples: int,
            comparison_samples: int) -> InjectedOrbitPropagation:
    """Attach the element history, the analytic comparison and the rerun hook
    the verify-report checks drive."""
    prop.elements_history = element_history(prop, history_samples)
    prop.analytic = analytic_two_body_comparison(prop, comparison_samples)

    def rerun(**overrides):
        overrides = dict(overrides)
        overrides.setdefault("max_step_s", prop.max_step_s)
        step = float(overrides.pop("step_size", overrides["max_step_s"]))
        perturbing = overrides.pop("perturbing_bodies", None)
        harmonics = overrides.pop("earth_spherical_harmonic", "keep")
        fm = prop.force_model
        if perturbing is not None:
            third = tuple(str(b) for b in perturbing if str(b) != CENTRAL_BODY)
            two_body_only = harmonics is None and not third
            fm = _replace_force_model(fm, j2=harmonics is not None, third_bodies=third,
                                      drag=None if two_body_only else fm.drag)
        elif harmonics != "keep":
            fm = _replace_force_model(fm, j2=harmonics is not None)
        kw = dict(max_step_s=step, integrator=prop.integrator, rtol=prop.rtol, atol=prop.atol,
                  history_samples=history_samples, comparison_samples=comparison_samples)
        kw.update(overrides)
        return _to_propagation_result(propagate_injected_orbit(
            prop.initial_state, fm, prop.revolutions, **kw))

    prop.rerun = rerun
    return prop


def _replace_force_model(fm: PropagationForceModel, **changes) -> PropagationForceModel:
    return PropagationForceModel(
        j2=changes.get("j2", fm.j2),
        drag=changes.get("drag", fm.drag),
        third_bodies=changes.get("third_bodies", fm.third_bodies))


# --------------------------------------------------------------------------
# Verification report
# --------------------------------------------------------------------------


def orbit_propagation_report(
    prop: InjectedOrbitPropagation,
    label: str = "injected-orbit",
    *,
    write_path: str | Path | None = None,
    convergence_check: bool = True,
    conservation_check: bool = True,
) -> dict:
    """The M-05 verification report (kind ``orbit-propagation``, schema
    ``verification-report-1.0``): the propagation section -- dynamics
    inventory of the *configured* force model, step statistics, the
    convergence re-run and the two-body conservation re-run -- plus the
    analytic two-body comparison as a residual (tolerance-gated only for a
    pure point-mass run), and under ``inputs`` the osculating-element time
    series, the GCRS state history and the analytic discrepancy summary.
    Written to ``write_path`` (or ``$ORBITOPT_VERIFICATION_REPORT_DIR``) via
    the shared reporter."""
    from orbitopt.verify.tudat_propagate import propagation_report_section

    fm = prop.force_model
    section = propagation_report_section(
        _to_propagation_result(prop), label,
        convergence_check=convergence_check, conservation_check=conservation_check)
    # The section's inventory follows the propagated *bodies* and cannot see
    # drag; replace it with the configured model's, and never let a drag run
    # claim two-body conservation of the propagated model.
    section["dynamics"] = _force_inventory(fm)
    if fm.drag is not None:
        propagated = section.get("conservation", {}).get("propagated_model", {})
        if propagated.get("applicable"):
            propagated["applicable"] = False
            propagated["reason"] = ("atmospheric drag is dissipative: two-body energy drift of "
                                    "the propagated model is physical decay, not numerical error")

    tolerance = ({"position_m": POINT_MASS_POS_TOL_M, "velocity_m_s": POINT_MASS_VEL_TOL_M_S}
                 if prop.point_mass_only else None)
    residual = _report.state_residual(
        "analytic_two_body_coast",
        "two-body Kepler coast of the same initial state and GM "
        "(orbitopt.flightlab.coast.coast_arc)",
        state=prop.states_ecliptic[-1], reference_state=_final_analytic_state(prop),
        tolerance=tolerance)

    state_history = [
        {"t_s": float(prop.epochs_s[i]), "position_m": prop.states_gcrs[i, :3],
         "velocity_m_s": prop.states_gcrs[i, 3:]}
        for i in _sample_indices(len(prop.epochs_s), len(prop.elements_history))
    ]
    report = _report.build_report(
        kind="orbit-propagation", label=str(label),
        propagations=[section], residuals=[residual],
        inputs={
            "force_model": fm.to_dict(),
            "handoff": {"frame": prop.initial_state.frame,
                        "epoch_mjd2000": prop.initial_state.epoch.mjd2000,
                        "epoch_timescale": prop.initial_state.epoch.timescale,
                        "mass_kg": prop.initial_state.mass_kg,
                        "initial_elements": prop.initial_elements()},
            "revolutions": prop.revolutions,
            "period_s": prop.period_s,
            "duration_s": prop.duration_s,
            "orbit_propagation": {
                "elements_history": list(prop.elements_history),
                "state_history_gcrs": state_history,
            },
            "analytic_comparison": {**dict(prop.analytic), "gated": prop.point_mass_only},
        },
    )
    return _report.finalize_report(report, write_path)


def _final_analytic_state(prop: InjectedOrbitPropagation) -> np.ndarray:
    """Analytic two-body state at the history's actual end epoch (adaptive
    runs end exactly on the requested duration; fixed-step runs may not)."""
    times, positions, velocities = coast_arc(
        prop.states_ecliptic[0, :3], prop.states_ecliptic[0, 3:],
        max(float(prop.duration_s), 1e-9), samples=2, mu=MU_EARTH)
    return np.concatenate([np.asarray(positions[-1], dtype=float),
                           np.asarray(velocities[-1], dtype=float)])


# --------------------------------------------------------------------------
# The segment
# --------------------------------------------------------------------------


@dataclass
class OrbitPropagationSegment:
    """M-05 as a :class:`~orbitopt.mission.contract.MissionSegment`: consume
    the injected-orbit handoff state, propagate it ``revolutions`` with
    tudatpy, and hand off the propagated final state (GCRS, same epoch
    timescale, same mass and stage config, propagation provenance in the
    metadata). The verification report lands at ``report_path`` when given,
    so the task report always carries the element history and the analytic
    comparison.

    Gating note: ``run_checked``'s position/velocity continuity checks are
    for *transform* handoffs (ascent -> parking orbit). A propagation moves
    the state along the orbit on purpose, so gate this segment on its
    verification report (analytic comparison, convergence, conservation), not
    on position continuity."""

    force_model: PropagationForceModel = field(default_factory=PropagationForceModel)
    revolutions: float = DEFAULT_REVOLUTIONS
    max_step_s: float = DEFAULT_MAX_STEP_S
    integrator: str = "dop853"
    report_path: str | Path | None = None
    label: str | None = None

    @property
    def name(self) -> str:
        return "tudatpy-injected-orbit-propagation"

    def run(self, initial_state: HandoffState) -> SegmentResult:
        validate_state(initial_state)
        prop = propagate_injected_orbit(
            initial_state, self.force_model, self.revolutions,
            max_step_s=self.max_step_s, integrator=self.integrator)
        report = orbit_propagation_report(
            prop, label=self.label or "injected-orbit", write_path=self.report_path)
        final = prop.final_state_gcrs
        final_epoch = Epoch(
            mjd2000=initial_state.epoch.mjd2000 + prop.duration_s / SECONDS_PER_DAY,
            timescale=initial_state.epoch.timescale)
        metadata = {**initial_state.metadata, "m05_propagation": {
            "revolutions": prop.revolutions,
            "duration_s": prop.duration_s,
            "force_model": prop.force_model.to_dict(),
            "final_elements": prop.final_elements(),
            "analytic_comparison": dict(prop.analytic),
            "report_digest": report["digest"],
            "report_result": report["status"]["result"],
        }}
        events = ({"kind": "orbit_propagation",
                   "duration_s": prop.duration_s,
                   "revolutions": prop.revolutions,
                   "final_epoch_tdb_s": final_epoch.to_tdb_seconds()},)
        diagnostics = {
            "report_status": report["status"],
            "report_digest": report["digest"],
            "analytic_comparison": dict(prop.analytic),
            "n_saved_samples": int(len(prop.epochs_s)),
            "propagation": prop,
        }
        return SegmentResult(
            segment=self.name,
            final_state=HandoffState(
                epoch=final_epoch, frame="GCRS",
                r_m=tuple(float(x) for x in final[:3]),
                v_m_s=tuple(float(x) for x in final[3:]),
                mass_kg=initial_state.mass_kg,
                stage_config=initial_state.stage_config,
                metadata=metadata),
            events=events, diagnostics=diagnostics)
