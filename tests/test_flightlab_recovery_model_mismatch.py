"""R-03: truth dynamics vs guidance dynamics with a configurable model
mismatch (orbitopt.flightlab.recovery.landing: GuidanceModelMismatch,
fly_landing_burn_closed_loop(mismatch=...), assess_landing_model_mismatch)
and its pipeline switch (``plan_options["landing"]["mismatch"]``).

The law and the referee are FL-305's, untouched; the mismatch biases only
what the guidance believes while the FL-301 propagator flies the truth. The
acceptance pair:

* zero mismatch is *bit-for-bit* the plain FL-305 flight (regression
  criterion -- compared against the default call, not against constants);
* a non-zero mismatch moves landing accuracy and fuel measurably and
  in isolation: a gravity bias costs fuel, a pad-position (navigation)
  bias costs accuracy, and the sweep reports the two per entry.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from orbitopt.flightlab.frames import geodetic_to_ecef
from orbitopt.flightlab.recovery import (
    NO_MODEL_MISMATCH,
    BoosterModel,
    BoosterState,
    GuidanceModelMismatch,
    LandingBurnEnu,
    RecoveryTarget,
    assess_landing_model_mismatch,
    fly_landing_burn_closed_loop,
)
from orbitopt.flightlab.recovery.boostback import _enu_basis
from orbitopt.flightlab.recovery.pipeline import _landing_flight

REPO_ROOT = Path(__file__).resolve().parents[1]
VEHICLE_YAML = REPO_ROOT / "src/orbitopt/data/flightlab/vehicles/synthetic_two_stage.yaml"
VEHICLE = BoosterModel.load(VEHICLE_YAML)

EPOCH = datetime(2026, 11, 10, 12, 0, 0)
PAD = (28.5619, -80.5772, 3.0)
PAD_ECEF = np.asarray(geodetic_to_ecef(*PAD), dtype=np.float64)
_BAS = np.vstack([np.asarray(b, dtype=np.float64) for b in _enu_basis(PAD[0], PAD[1])])
SITE = RecoveryTarget(kind="rtls", name="launch_site", latitude_deg=PAD[0],
                      longitude_deg=PAD[1], altitude_m=PAD[2])

# FL-302's corridor handoff for the textbook RTLS separation (the same state
# tests/test_flightlab_recovery_terminal_guidance.py flies closed-loop).
CASE_HANDOFF = ((-351.15, 21.11, 3015.87), (-1.16, 0.15, -306.01), 41660.9)


def _states(case=CASE_HANDOFF, t0=300.0):
    pos, vel, m0 = case
    enu = LandingBurnEnu(t_s=t0, position_enu_m=pos, velocity_enu_m=vel, mass_kg=m0)
    r = PAD_ECEF + _BAS.T @ np.asarray(pos, dtype=np.float64)
    v = _BAS.T @ np.asarray(vel, dtype=np.float64)
    ecef = BoosterState(t_s=t0, epoch_utc=EPOCH, frame="ECEF",
                        position_m=tuple(float(c) for c in r),
                        velocity_m_s=tuple(float(c) for c in v), mass_kg=m0)
    return enu, ecef


def _fly(mismatch=None, **kwargs):
    pytest.importorskip("gfold")
    enu, ecef = _states()
    return fly_landing_burn_closed_loop(enu, ecef, SITE, VEHICLE,
                                        mismatch=mismatch, **kwargs)


# ---------------------------------------------------------------------------
# the knob itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"thrust_scale": 0.0}, {"thrust_scale": -1.0},
    {"gravity_bias_m_s2": float("nan")},
    {"pad_offset_east_m": float("inf")},
])
def test_mismatch_rejects_nonsense(kwargs):
    with pytest.raises(ValueError):
        GuidanceModelMismatch(**kwargs)


def test_mismatch_identity_and_roundtrip():
    assert GuidanceModelMismatch().is_identity
    assert NO_MODEL_MISMATCH.is_identity
    mm = GuidanceModelMismatch(gravity_bias_m_s2=0.2, thrust_scale=0.9,
                               mass_bias_kg=250.0, pad_offset_east_m=-15.0,
                               pad_offset_north_m=30.0)
    assert not mm.is_identity
    assert GuidanceModelMismatch(**mm.to_dict()) == mm
    assert GuidanceModelMismatch.from_option(None) is NO_MODEL_MISMATCH
    assert GuidanceModelMismatch.from_option(mm) is mm
    assert GuidanceModelMismatch.from_option({"thrust_scale": 0.9}) \
        == GuidanceModelMismatch(thrust_scale=0.9)
    with pytest.raises(TypeError):
        GuidanceModelMismatch.from_option("0.9")


def test_perceived_gravity_must_stay_positive():
    from orbitopt.flightlab.recovery import TERMINAL_GUIDANCE_DEFAULT
    from orbitopt.flightlab.recovery.landing import _ClosedLoopLanding
    with pytest.raises(ValueError, match="not positive"):
        _ClosedLoopLanding(VEHICLE, TERMINAL_GUIDANCE_DEFAULT,
                           pad_lat_deg=PAD[0], pad_lon_deg=PAD[1],
                           pad_altitude_m=PAD[2], pad_ecef_m=PAD_ECEF,
                           gravity_m_s2=9.8,
                           mismatch=GuidanceModelMismatch(gravity_bias_m_s2=-9.8))


# ---------------------------------------------------------------------------
# pipeline switch
# ---------------------------------------------------------------------------

def test_pipeline_landing_option_takes_the_mismatch():
    assert callable(_landing_flight({"mismatch": {"gravity_bias_m_s2": 0.2}}))
    assert callable(_landing_flight({"mismatch": GuidanceModelMismatch(thrust_scale=0.9),
                                     "design_throttle": 0.85}))
    assert callable(_landing_flight({"mismatch": None}))
    with pytest.raises(ValueError, match="single"):
        _landing_flight({"mode": "single", "mismatch": {"thrust_scale": 0.9}})
    with pytest.raises(TypeError):
        _landing_flight({"mismatch": {"gravity_bias": 0.2}})  # unknown field


# ---------------------------------------------------------------------------
# zero mismatch == the plain FL-305 flight, bit for bit
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_zero_mismatch_is_bitwise_identical_to_fl305():
    plain = _fly()
    for zero in (NO_MODEL_MISMATCH,
                 GuidanceModelMismatch(),
                 GuidanceModelMismatch.from_option({"gravity_bias_m_s2": 0.0,
                                                   "thrust_scale": 1.0,
                                                   "mass_bias_kg": 0.0,
                                                   "pad_offset_east_m": 0.0,
                                                   "pad_offset_north_m": 0.0})):
        res = _fly(zero)
        assert res.landed and res.reason_codes == ()  # the FL-305 verdict stands
        assert res.assumptions == plain.assumptions    # no mismatch note recorded
        np.testing.assert_array_equal(res.track.times_s, plain.track.times_s)
        np.testing.assert_array_equal(res.track.position_ecef_m,
                                      plain.track.position_ecef_m)
        np.testing.assert_array_equal(res.track.mass_kg, plain.track.mass_kg)
        assert res.position_error_m == plain.position_error_m


@pytest.mark.slow
def test_mismatched_flight_is_deterministic():
    mm = {"gravity_bias_m_s2": 0.3, "thrust_scale": 0.95}
    a, b = _fly(mm), _fly(mm)
    np.testing.assert_array_equal(a.track.mass_kg, b.track.mass_kg)
    assert a.position_error_m == b.position_error_m


# ---------------------------------------------------------------------------
# non-zero mismatch moves accuracy and fuel, each in isolation
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_gravity_bias_costs_fuel_without_sinking_the_pad():
    base = _fly()
    res = _fly({"gravity_bias_m_s2": 0.3})
    assert res.landed, res.reason_codes  # a modest bias stays inside limits
    assert any("R-03 guidance model mismatch" in a for a in res.assumptions)
    used_base = float(base.track.mass_kg[0] - base.track.mass_kg[-1])
    used = float(res.track.mass_kg[0] - res.track.mass_kg[-1])
    # fighting heavier perceived gravity decelerates harder from ignition
    # on: the truth burns more, and the touchdown stays inside every limit.
    assert used > used_base
    assert res.position_error_m <= 25.0


@pytest.mark.slow
def test_pad_navigation_bias_costs_accuracy_not_fuel():
    base = _fly()
    res = _fly({"pad_offset_east_m": 40.0})
    # the guidance delivers the vehicle to the pad it believes in -- 40 m
    # east of the truth pad, where the referee grades: outside the 25 m
    # acceptance radius, and the position check is the one that fails.
    assert not res.landed
    assert res.reason_codes == ("LANDING_POSITION_OUT_OF_TOLERANCE",)
    assert 30.0 < res.position_error_m < 55.0
    # a pure navigation bias costs aim, not propellant: the burn timeline is
    # the same guidance-vs-truth dialogue as nominal, off by well under 1 %.
    used_base = float(base.track.mass_kg[0] - base.track.mass_kg[-1])
    used = float(res.track.mass_kg[0] - res.track.mass_kg[-1])
    assert abs(used - used_base) < 1e-2 * used_base


@pytest.mark.slow
def test_thrust_scale_mismatch_changes_the_flown_throttle():
    base = _fly()
    res = _fly({"thrust_scale": 0.97})
    assert res.landed, res.reason_codes  # guidance under-believes its engine
    used_base = float(base.track.mass_kg[0] - base.track.mass_kg[-1])
    used = float(res.track.mass_kg[0] - res.track.mass_kg[-1])
    assert used > used_base  # a bigger throttle for the same request
    # a mismatch big enough to break the limits reads honest red, not green:
    hard = _fly({"thrust_scale": 0.93})
    assert not hard.landed and hard.reason_codes


@pytest.mark.slow
def test_sweep_reports_accuracy_and_fuel_per_mismatch():
    pytest.importorskip("gfold")
    enu, ecef = _states()
    out = assess_landing_model_mismatch(
        enu, ecef, SITE, VEHICLE,
        [None, GuidanceModelMismatch(gravity_bias_m_s2=0.3),
         GuidanceModelMismatch(pad_offset_east_m=40.0)])
    assert len(out) == 3
    assert [a.mismatch.is_identity for a in out] == [True, False, False]
    zero, gravity, nav = out
    assert zero.result.landed and zero.propellant_used_kg is not None
    assert gravity.result.landed
    assert gravity.propellant_used_kg > zero.propellant_used_kg
    assert not nav.result.landed
    assert nav.result.position_error_m > zero.result.position_error_m
    d = zero.to_dict()
    assert d["landed"] is True
    assert d["mismatch"] == NO_MODEL_MISMATCH.to_dict()
    assert isinstance(d["propellant_used_kg"], float)
