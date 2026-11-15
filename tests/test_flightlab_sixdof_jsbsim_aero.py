"""E1-04 (#253): the production JSBSim fidelity path.

The production 6-DOF path is the generated airframe
``data/flightlab/jsbsim/aircraft/spike_stage_aero`` -- the E4 spike airframe
with the packaged aero document compiled into its aerodynamics section.  The
tests below hold that pipeline up:

* the artefact is what its generator produces (no hand-edited XML),
* the compiled tables stay a rounding error next to the model they were
  compiled from (``promotion_fidelity``),
* the parity gate knows the production airframe and its reference geometry,
* the mass properties, the CG/AERORP coupling and the TVC actuator behave on
  the *production* airframe, not just the zero-aero spike,
* the packaged ``e104_aero_ascent`` scenario flies it end to end, and the
  backend-to-backend comparison (``production_compare``) stays inside the
  bands its ADR (docs/adr/flightlab-0007) discloses.

Skipped without ``jsbsim`` for the runtime tests (opt-in pip dep:
``pip install -r envs/sixdof-requirements.txt``); the XML/parity tests run
everywhere.
"""
from __future__ import annotations

import math

import pytest

from orbitopt.flightlab.sixdof.jsbsim_aero import (  # noqa: E402
    PRODUCTION_AIRCRAFT,
    aircraft_aero_note,
    production_model,
    production_aircraft_xml,
    promotion_fidelity,
    regenerate,
)

# Declared thresholds.
#: The largest coefficient difference the compiled tables may show against
#: the body-axis model (absolute; JSBSim linearly interpolates the rotated
#: tables, the native model interpolates the body tables and rotates after).
#: The worst case sits at the 15 deg grid edge, well outside anything the
#: E1-04 ascent flies.
PROMOTION_BAND = 0.005
#: Sea-level held-vertical ascent: the production airframe must hold alpha to
#: a tenth of a degree while thrusting through max-Q.
HELD_ALPHA_DEG = 0.1
#: Backend-to-backend comparison: alpha inside the aero-authority window.
ALPHA_WINDOW_UNTIL_S = 60.0
ALPHA_WINDOW_BAND_DEG = 3.0


def test_packaged_artifact_is_what_the_generator_produces():
    # The checked-in aircraft must byte-match a fresh generation of its own
    # aero document; hand edits or a stale artefact fail here.
    path, fresh = regenerate(check=True)
    assert fresh, f'{path} has drifted from data/flightlab/aero/spike_stage.yaml'


def test_promotion_fidelity_stays_a_rounding_error():
    fid = promotion_fidelity(production_model())
    assert fid.worst() <= PROMOTION_BAND
    # The pitch/yaw couple terms are exact: the document's own coefficients
    # go into the table verbatim, and the CG-coupling correction cancels the
    # AERORP couple analytically, not by fitting.
    assert fid.cm == pytest.approx(0.0, abs=1e-9)
    assert fid.cn_yaw == pytest.approx(0.0, abs=1e-9)


def test_parity_gate_knows_the_production_airframe():
    from orbitopt.flightlab.parity import check_parity
    verdicts = {c.id: c.passed for c in check_parity()}
    assert verdicts['parity-jsbsim-production-inherited'] is True
    assert verdicts['parity-jsbsim-production-aero'] is True
    assert verdicts['parity-jsbsim-production-aero-geometry'] is True


def test_zero_aero_spike_stays_zero_aero():
    # The E4 airframe is untouched: only the generated copy carries aero.
    assert aircraft_aero_note('spike_stage') == 'zero-aero'
    assert aircraft_aero_note(PRODUCTION_AIRCRAFT).startswith(
        'aerodynamic tables compiled')
    prod = production_aircraft_xml(production_model())
    assert 'aero/coefficient/CMcg' in prod.xml  # CG-coupling cancellation
    assert 'aero/coefficient/CMa' in prod.xml   # the document's own couple


pytest.importorskip('jsbsim')

from orbitopt.flightlab.sixdof.jsbsim_driver import (  # noqa: E402
    SPIKE_STAGE,
    InitialConditions,
    SixDofSim,
)


def _production(sim_dt: float = 0.01, alt_m: float = 300e3,
                pitch_deg: float = 90.0) -> SixDofSim:
    sim = SixDofSim(sim_dt, model=PRODUCTION_AIRCRAFT)
    sim.initialize(InitialConditions(lat_geod_deg=0.0, lon_deg=0.0, alt_m=alt_m,
                                     pitch_deg=pitch_deg, heading_deg=90.0))
    return sim


def test_mass_properties_follow_fuel_on_the_production_airframe():
    # Same tanks/inertias as the E4 spike (the parity gate checks the XML),
    # recomputed live by JSBSim's mass balance on the production airframe.
    sim = _production()
    sim.ignite(1.0)
    sim.run_for(101.0)
    s = sim.samples
    assert s[0]['mass_kg'] == pytest.approx(SPIKE_STAGE['dry_mass_kg']
                                            + SPIKE_STAGE['propellant_mass_kg'], abs=1.5)
    assert s[-1]['thrust_n'] == 0.0
    assert s[-1]['mass_kg'] == pytest.approx(SPIKE_STAGE['dry_mass_kg'], abs=1.5)
    assert s[-1]['iyy_kg_m2'] < s[0]['iyy_kg_m2']
    # Forward LOX drains faster: the CG travels aft, towards (and onto) the
    # AERORP, where the couple-cancellation term goes to zero.
    assert s[-1]['cg_x_m'] > s[0]['cg_x_m'] + 1.0
    assert s[-1]['cg_x_m'] == pytest.approx(20.0, abs=0.05)


def test_actuator_dynamics_survive_the_aero_upgrade():
    # The TVC is inherited verbatim from the E4 airframe; the FCS sees the
    # same actuator on the production model.  Vacuum IC so this tests the
    # actuator, not the aero.
    cmd, tau = 0.01, SPIKE_STAGE['gimbal_lag_tau_s']
    sim = _production()
    sim.ignite(1.0)
    sim.set_gimbal(cmd, -cmd)
    sim.run_for(1.0)
    for r in sim.samples[1:]:
        expected = cmd * (1.0 - math.exp(-(r['t_s'] - sim.dt_s / 2) / tau))
        assert abs(r['gimbal_pitch_actual_rad'] - expected) <= 1e-2 * cmd
        assert abs(r['gimbal_yaw_actual_rad'] + expected) <= 1e-2 * cmd


def test_actuator_rate_and_travel_limits_on_production():
    sim = _production()
    sim.ignite(1.0)
    sim.set_gimbal(0.5, 0.0)
    sim.run_for(1.0)
    act = [r['gimbal_pitch_actual_rad'] for r in sim.samples]
    slew = [(b - a) / sim.dt_s for a, b in zip(act, act[1:])]
    assert max(slew) <= SPIKE_STAGE['gimbal_rate_limit_rad_s'] + 1e-6
    assert max(slew) >= SPIKE_STAGE['gimbal_rate_limit_rad_s'] - 1e-6
    assert act[-1] == pytest.approx(SPIKE_STAGE['gimbal_limit_rad'], abs=1e-6)


def test_sea_level_ascent_thrusts_through_real_dynamic_pressure():
    sim = _production(alt_m=0.0)
    sim.ignite(1.0)
    q_max = 0.0

    def watch(t_s: float, s: SixDofSim) -> None:
        nonlocal q_max
        q_max = max(q_max, float(s['aero/qbar-psf']))

    sim.run_for(35.0, watch)
    # Through max-Q (PSF -> ~70 kPa): the aero is real, not a stub.
    assert q_max * 47.88025 > 30e3
    assert float(sim['metrics/Sw-sqft']) == pytest.approx(10.7521 * 10.76391, rel=1e-3)
    assert abs(float(sim['aero/alpha-rad'])) < math.radians(HELD_ALPHA_DEG)
    # The CG-coupling correction is live: the CG sits aft of its wet value.
    assert float(sim['inertia/cg-x-in']) > 720.0


@pytest.mark.slow
def test_e104_scenario_flys_on_the_production_path(tmp_path):
    from orbitopt.flightlab.adapters.sixdof import packaged_config, run_ascent
    run = run_ascent(packaged_config('e104_aero_ascent'), tmp_path / 'e104')
    # ADR flightlab-0004 (independent verification): the adapter vouches for
    # its own output only through backend_self_check; physics_verified is set
    # by the independent re-fly in simulate(), never by the backend itself.
    assert run.verification.backend_self_check is True
    assert run.verification.physics_verified is None
    ids = {c.id for c in run.verification.checks if c.id.startswith('sixdof-mass')}
    assert ids, 'the F-03 mass-model checks must run on the production path too'
    assert all(c.passed for c in run.verification.checks)
    assert any('aerodynamic tables compiled' in a for a in run.assumptions)


@pytest.mark.slow
def test_backend_comparison_stays_inside_its_disclosed_bands():
    from orbitopt.flightlab.sixdof.production_compare import compare_production_paths
    comp = compare_production_paths()
    assert all(c.passed for c in comp.checks), comp.summary()
    assert comp.metrics['alpha_window_deg'] <= ALPHA_WINDOW_BAND_DEG
