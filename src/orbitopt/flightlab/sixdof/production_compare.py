"""E1-04 (task #253): the production ascent, flown by both 6-DOF backends.

The acceptance question for the production aero path is not "does JSBSim
agree with the native rigid-body plant" in the abstract -- the two plants
keep documented differences that no parity source removes:

  * both engines debit ambient back pressure over the same physical 2 m^2
    nozzle exit (E6-01, #273: ``T = thrust_vac - A*p`` with the choked flow
    ``thrust_vac/(Isp_vac*g0)``), but JSBSim reads the ambient from its own
    atmosphere model while the native plant reads US76 -- an engine *model*
    the two now share, implemented twice,
  * JSBSim recomputes mass, CG and inertia from the tank contents every
    frame and its TVC is an actuator (lag, slew rate, travel limit); the
    native plant integrates a constant inertia tensor about the CG with an
    instantaneous gimbal,
  * JSBSim applies its aero forces at a fixed ``AERORP``.  The generated
    tables carry a CG-coupling cancellation (``jsbsim_aero._cg_cancel``) so
    that the *net* moment is the aero document's own CG-referenced couple at
    every propellant load -- but the force is still applied at one fixed
    point, where the native model applies it at the live CG,
  * JSBSim evaluates its rotated tables with linear interpolation in
    ``(Mach, alpha)``; the native model interpolates the body tables and
    rotates afterwards (quantified by ``jsbsim_aero.promotion_fidelity``).

This module therefore flies the SAME scenario -- the packaged E1-04 ascent,
open loop, neutral gimbal, full throttle to burnout -- through both plants
from the same initial state, and reports the divergence trace-by-trace with
checks whose bands the ADR (docs/adr/flightlab-0007) explains: what is
engine, what is actuator, what is aero reference point, what is gravity and
frame integration.  The angle-of-attack checks therefore cover the *powered*
window only (through and shortly after max-Q): after the engine stops the
uncontrolled attitude of a ballistic stage diverges on any two plants whose
gravity/frame integration differs, and that is disclosed as
``alpha_coast_deg`` rather than checked.  The native plant runs its comparison
with ``j2=False`` so the comparison isolates thrust and aero from the two
plants' gravity models.  It is the quantification the task's third acceptance
bullet asks for; the choice of JSBSim as the aero production path, and why,
is recorded in the ADR too.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from orbitopt.flightlab.contract import Check
from orbitopt.flightlab.parity import load_parity_scenario, native_booster
from orbitopt.flightlab.sixdof.jsbsim_aero import PRODUCTION_AIRCRAFT
from orbitopt.flightlab.sixdof.jsbsim_driver import InitialConditions, SixDofSim
from orbitopt.flightlab.sixdof.rigidbody import (RigidBodyPhysics, RigidBodyState,
                                                 constant_control, propagate)

__all__ = ["ProductionComparison", "compare_production_paths", "jsbsim_ascent",
           "native_ascent"]

#: PSF -> Pa (the JSBSim side reports dynamic pressure in lb/ft^2).
PSF_TO_PA = 47.88025

#: The comparison runs (s): the packaged E1-04 scenario's own numbers.
SCENARIO_DURATION_S = 104.0
SCENARIO_DT_S = 0.02
SCENARIO_SAMPLE_EVERY = 20


def native_ascent(*, dt_s: float = SCENARIO_DT_S, duration_s: float = SCENARIO_DURATION_S,
                  throttle: float = 1.0, sample_every: int = SCENARIO_SAMPLE_EVERY,
                  vehicle=None,
                  physics: RigidBodyPhysics | None = None) -> dict[str, np.ndarray]:
    """The E1-04 ascent on the native rigid-body plant: the parity booster
    (masses, engine, the packaged aero document) launched from the pad, held
    open-loop at the commanded attitude with a neutral gimbal. Returned as
    the trace columns :func:`compare_production_paths` matches on."""
    vehicle = vehicle or native_booster()
    state0 = RigidBodyState.from_launch(lat_geod_deg=0.0, lon_deg=0.0, alt_m=0.0,
                                        pitch_deg=90.0, heading_deg=90.0,
                                        mass_kg=vehicle.mass_wet_kg)
    run = propagate(state0, vehicle, constant_control(throttle), duration_s,
                    dt_s=dt_s, sample_every=sample_every, physics=physics)
    trace = {column: np.asarray(run[column], dtype=float)
             for column in ('t_s', 'mass_kg', 'altitude_m', 'mach', 'alpha_deg',
                            'q_bar_pa', 'q_rad_s', 'thrust_n')}
    trace['speed_m_s'] = np.linalg.norm(
        np.stack([run['vx_m_s'], run['vy_m_s'], run['vz_m_s']]), axis=0)
    return trace


def jsbsim_ascent(*, dt_s: float = SCENARIO_DT_S, duration_s: float = SCENARIO_DURATION_S,
                  throttle: float = 1.0, sample_every: int = SCENARIO_SAMPLE_EVERY,
                  model: str = PRODUCTION_AIRCRAFT) -> dict[str, np.ndarray]:
    """The same ascent on the JSBSim production airframe, recorded through the
    driver's per-frame control hook (the driver's own sample columns carry
    neither dynamic pressure nor angle of attack)."""
    sim = SixDofSim(dt_s, model=model)
    sim.initialize(InitialConditions(lat_geod_deg=0.0, lon_deg=0.0, alt_m=0.0,
                                     pitch_deg=90.0, heading_deg=90.0))
    sim.ignite(throttle)
    rows: list[dict[str, float]] = []
    frame = 0

    def record(t_s: float, s: SixDofSim) -> None:
        nonlocal frame
        if frame % int(sample_every) == 0:
            smp = s.sample()
            rows.append({'t_s': smp['t_s'],
                         'mass_kg': smp['mass_kg'],
                         'altitude_m': smp['altitude_m'],
                         'mach': float(s['velocities/mach']),
                         'alpha_deg': float(s['aero/alpha-deg']),
                         'q_bar_pa': float(s['aero/qbar-psf']) * PSF_TO_PA,
                         'q_rad_s': smp['q_rad_s'],
                         'thrust_n': smp['thrust_n'],
                         'speed_m_s': math.sqrt(smp['vx_m_s'] ** 2 + smp['vy_m_s'] ** 2
                                                + smp['vz_m_s'] ** 2)})
        frame += 1

    sim.run_for(duration_s, record)
    return {key: np.array([row[key] for row in rows]) for key in rows[0]}


#: Declared bands for the backend-to-backend comparison (E1-04). They are
#: NOT tolerances on a shared truth -- the two plants disagree by design (see
#: the module docstring) -- they are the bands inside which the disagreement
#: observed in docs/adr/flightlab-0007 stays the *disclosed* one, and a
#: regression (a swapped engine, an aero term dropped in the XML) is caught.
#: The native and JSBSim engines now book the same choked flow over the same
#: exit area (E6-01, #273), so the mass traces may only disagree by JSBSim's
#: one-frame-late drain (measured 12 kg of a 60 t load, against 164 kg before
#: the engines were aligned) and the two atmosphere models.
MASS_PROP_REL_BAND = 0.002
ALT_MECO_REL_BAND = 0.15
SPEED_MECO_REL_BAND = 0.10
#: Measured 0.14 % (ADR flightlab-0009) -- the two plants now find max-Q at
#: the same instant; what is left is US76 vs JSBSim's atmosphere.
MAXQ_REL_BAND = 0.05
PITCH_RATE_ABS_BAND_RAD_S = 0.02
#: The window of the E1-04 ascent where the aero has authority: pad to just
#: past max-Q.  Alpha is checked on this window only -- past it the
#: uncontrolled attitude of the two plants diverges for reasons the ADR
#: discloses (frame/gravity integration of an unguided ballistic-ish stage),
#: reported as ``alpha_coast_deg``.
ALPHA_WINDOW_UNTIL_S = 60.0
ALPHA_WINDOW_BAND_DEG = 3.0
#: The production vehicle flies a held-vertical ascent on its own aerodynamics:
#: JSBSim's own alpha stays under a degree for the whole flight.
JSBSIM_ALPHA_MAX_BAND_DEG = 1.0
#: Both engines debit ambient back pressure over the same 2 m^2 exit
#: (E6-01), so the pad thrust ratio must sit at 1.0. Measured 0.999995 -- the
#: band is for the two atmosphere models disagreeing at the pad and JSBSim's
#: lb/ft^2 rounding, not for the propulsion law any more.
PAD_THRUST_RATIO_REL_BAND = 0.005


@dataclass(frozen=True)
class ProductionComparison:
    """Both traces, the headline metrics and the declared-band checks."""

    jsbsim: Mapping[str, np.ndarray]
    native: Mapping[str, np.ndarray]
    metrics: Mapping[str, float]
    checks: tuple[Check, ...] = field(default=())

    def summary(self) -> str:
        lines = [f"{'metric':38s} value"]
        for key, value in self.metrics.items():
            lines.append(f'{key:38s} {value:.6g}')
        for c in self.checks:
            lines.append(f'{"[PASS] " if c.passed else "[FAIL] " + c.id} '
                         f'{c.value if c.value is not None else ""}')
        return '\n'.join(lines)


def _aligned(native: Mapping[str, np.ndarray],
             jsbsim: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Both traces live on the same mission clock (every ``sample_every``
    frames from t = 0); the native run additionally appends its terminal
    state. Return the boolean view of the native samples that exist in the
    JSBSim trace, and the matching JSBSim indices."""
    step = native['t_s'][1] - native['t_s'][0]
    j_idx = np.minimum(np.searchsorted(jsbsim['t_s'], native['t_s']),
                       len(jsbsim['t_s']) - 1)
    ok = np.abs(jsbsim['t_s'][j_idx] - native['t_s']) <= step / 2.0
    return ok, j_idx


def compare_production_paths(*, dt_s: float = SCENARIO_DT_S,
                             duration_s: float = SCENARIO_DURATION_S,
                             throttle: float = 1.0,
                             sample_every: int = SCENARIO_SAMPLE_EVERY,
                             model: str = PRODUCTION_AIRCRAFT) -> ProductionComparison:
    """Fly the E1-04 ascent on both plants and settle the differences with
    checks (``Check`` from the Flight Lab contract, so a caller can publish
    them like any other verdict). Never raises on a failing check."""
    vehicle = native_booster(load_parity_scenario())
    # j2 off on the native side: JSBSim's standard gravity is central, and
    # leaving the native J2 on would fold a gravity-model difference into an
    # aero comparison (disclosed in ADR flightlab-0007).
    nat = native_ascent(dt_s=dt_s, duration_s=duration_s, throttle=throttle,
                        sample_every=sample_every, vehicle=vehicle,
                        physics=RigidBodyPhysics(j2=False))
    jbs = jsbsim_ascent(dt_s=dt_s, duration_s=duration_s, throttle=throttle,
                        sample_every=sample_every, model=model)
    ok, j_idx = _aligned(nat, jbs)
    n = {k: v[ok] for k, v in nat.items()}
    j = {k: v[j_idx[ok]] for k, v in jbs.items()}
    t_end = float(n['t_s'][-1])

    m = {
        'overlap_s': t_end,
        'mass_abs_kg': float(np.max(np.abs(n['mass_kg'] - j['mass_kg']))),
        'alt_meco_rel': float(abs(n['altitude_m'][-1] - j['altitude_m'][-1])
                              / max(n['altitude_m'][-1], j['altitude_m'][-1], 1.0)),
        'speed_meco_rel': float(abs(n['speed_m_s'][-1] - j['speed_m_s'][-1])
                                / max(n['speed_m_s'][-1], j['speed_m_s'][-1], 1.0)),
        'max_q_native_pa': float(np.max(n['q_bar_pa'])),
        'max_q_jsbsim_pa': float(np.max(j['q_bar_pa'])),
        'max_q_rel': float(abs(np.max(n['q_bar_pa']) - np.max(j['q_bar_pa']))
                           / max(np.max(n['q_bar_pa']), np.max(j['q_bar_pa']), 1.0)),
        'alpha_window_deg': float(np.max(np.abs((n['alpha_deg'] - j['alpha_deg'])
                                                [n['t_s'] <= ALPHA_WINDOW_UNTIL_S]))),
        'alpha_jsbsim_max_deg': float(np.max(np.abs(j['alpha_deg']))),
        'alpha_coast_deg': float(abs(n['alpha_deg'][-1] - j['alpha_deg'][-1])),
        'mass_propellant_kg': float(vehicle.propellant_mass_kg),
        'pitch_rate_max_abs_rad_s': float(np.max(np.abs(n['q_rad_s'] - j['q_rad_s']))),
        'pad_thrust_ratio': float(j['thrust_n'][1] / max(n['thrust_n'][1], 1e-9)),
        # E6-01 (#273): both engines debit ambient pressure over the same
        # 2 m^2 exit, so the pad thrust ratio is 1.0 by construction.  Before
        # E6-01 the token nozzle left JSBSim vacuum-like and the ratio sat at
        # isp_vacuum/isp_sea_level (~1.135); that debit is now shared, so any
        # drift off 1.0 is an engine parameter, not a disclosed model form.
        'pad_thrust_ratio_expected': 1.0,
    }
    m['pad_thrust_ratio_rel'] = float(abs(m['pad_thrust_ratio']
                                         - m['pad_thrust_ratio_expected'])
                                      / m['pad_thrust_ratio_expected'])
    checks = [
        Check(id='e104-mass-agreement',
              passed=m['mass_abs_kg'] <= MASS_PROP_REL_BAND * m['mass_propellant_kg'],
              value=m['mass_abs_kg'],
              limit=MASS_PROP_REL_BAND * m['mass_propellant_kg'], unit='kg',
              reason_code='' if m['mass_abs_kg'] <= MASS_PROP_REL_BAND * m['mass_propellant_kg']
              else 'E104_PROPELLANT_DRAIN_DIVERGED'),
        Check(id='e104-max-q-agreement', passed=m['max_q_rel'] <= MAXQ_REL_BAND,
              value=m['max_q_rel'], limit=MAXQ_REL_BAND,
              reason_code='' if m['max_q_rel'] <= MAXQ_REL_BAND else
              'E104_MAX_Q_DIVERGED'),
        Check(id='e104-meco-altitude-agreement',
              passed=m['alt_meco_rel'] <= ALT_MECO_REL_BAND, value=m['alt_meco_rel'],
              limit=ALT_MECO_REL_BAND,
              reason_code='' if m['alt_meco_rel'] <= ALT_MECO_REL_BAND else
              'E104_MECO_ALTITUDE_DIVERGED'),
        Check(id='e104-meco-speed-agreement',
              passed=m['speed_meco_rel'] <= SPEED_MECO_REL_BAND,
              value=m['speed_meco_rel'], limit=SPEED_MECO_REL_BAND,
              reason_code='' if m['speed_meco_rel'] <= SPEED_MECO_REL_BAND else
              'E104_MECO_SPEED_DIVERGED'),
        Check(id='e104-pitch-rate-agreement',
              passed=m['pitch_rate_max_abs_rad_s'] <= PITCH_RATE_ABS_BAND_RAD_S,
              value=m['pitch_rate_max_abs_rad_s'], limit=PITCH_RATE_ABS_BAND_RAD_S,
              unit='rad/s',
              reason_code='' if m['pitch_rate_max_abs_rad_s'] <= PITCH_RATE_ABS_BAND_RAD_S
              else 'E104_ATTITUDE_RATE_DIVERGED'),
        Check(id='e104-alpha-agreement-through-max-q',
              passed=m['alpha_window_deg'] <= ALPHA_WINDOW_BAND_DEG,
              value=m['alpha_window_deg'], limit=ALPHA_WINDOW_BAND_DEG, unit='deg',
              reason_code='' if m['alpha_window_deg'] <= ALPHA_WINDOW_BAND_DEG else
              'E104_ALPHA_DIVERGED'),
        Check(id='e104-jsbsim-alpha-held',
              passed=m['alpha_jsbsim_max_deg'] <= JSBSIM_ALPHA_MAX_BAND_DEG,
              value=m['alpha_jsbsim_max_deg'], limit=JSBSIM_ALPHA_MAX_BAND_DEG, unit='deg',
              reason_code='' if m['alpha_jsbsim_max_deg'] <= JSBSIM_ALPHA_MAX_BAND_DEG else
              'E104_AERO_ATTITUDE_NOT_HELD'),
        Check(id='e104-pad-thrust-debit-shared',
              passed=m['pad_thrust_ratio_rel'] <= PAD_THRUST_RATIO_REL_BAND,
              value=m['pad_thrust_ratio_rel'], limit=PAD_THRUST_RATIO_REL_BAND,
              reason_code='' if m['pad_thrust_ratio_rel'] <= PAD_THRUST_RATIO_REL_BAND
              else 'E104_PAD_THRUST_RATIO_UNEXPLAINED'),
    ]
    return ProductionComparison(jsbsim=jbs, native=nat, metrics=m,
                                checks=tuple(checks))
