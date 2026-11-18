"""Nose-shape aerodynamics and the Starship-class vehicle file built on it.

The model is a preliminary-design estimate, so these tests pin its physics
(limits, orderings, continuity) and -- the point of generating the tables --
that the aero numbers committed in the vehicle YAML equal the ones generated
from the geometry declared beside them.
"""
from __future__ import annotations

import math

import pytest

from orbitopt.flightlab import vehicle_catalog as vc
from orbitopt.flightlab.nose_aero import (
    DEFAULT_MACH,
    NOSE_SHAPES,
    NoseSpec,
    base_drag_ca,
    ca_table,
    cone_pressure_coefficient,
    normal_force_slope_per_rad,
    skin_friction_ca,
    stage_nose_aero,
    wave_drag_ca,
    wetted_area_m2,
)
from orbitopt.flightlab.vehicle_view import load_vehicle_view

STARSHIP = "starship_class_public_estimate"


def test_cone_pressure_limits():
    m = 5.0
    assert cone_pressure_coefficient(m, 1e-4) == pytest.approx(0.0, abs=1e-3)
    angles = [math.radians(d) for d in (2, 5, 10, 20, 30)]
    cps = [cone_pressure_coefficient(m, a) for a in angles]
    assert all(c is not None for c in cps) and cps == sorted(cps)
    # hypersonic limit: close to the Newtonian 2 sin^2(theta) at large angles
    hyper = cone_pressure_coefficient(20.0, math.radians(30))
    assert hyper == pytest.approx(2.0 * math.sin(math.radians(30)) ** 2, rel=0.1)
    assert cone_pressure_coefficient(0.9, math.radians(5)) is None          # no shock below Mach 1
    assert cone_pressure_coefficient(1.5, math.radians(40)) is None         # the shock detaches


@pytest.mark.parametrize("shape", NOSE_SHAPES)
def test_profiles_run_from_the_tip_to_the_base(shape):
    nose = NoseSpec(shape, 13.5, 9.0)
    x, r = nose.profile()
    assert x[0] == 0.0 and r[0] == pytest.approx(0.0, abs=1e-9)
    assert x[-1] == pytest.approx(13.5) and r[-1] == pytest.approx(4.5, rel=1e-6)
    assert all(b >= a - 1e-12 for a, b in zip(r, r[1:]))                    # grows to the base


def test_nose_spec_rejects_nonsense():
    with pytest.raises(ValueError, match="shape"):
        NoseSpec("flat", 5.0, 3.0)
    with pytest.raises(ValueError, match="positive"):
        NoseSpec("cone", 0.0, 3.0)


def test_wave_drag_behaves_like_a_nose():
    ogive = NoseSpec("tangent_ogive", 13.5, 9.0)
    assert wave_drag_ca(ogive, 0.5) == 0.0 and wave_drag_ca(ogive, 0.9) == 0.0
    rise = [wave_drag_ca(ogive, m) for m in (0.95, 1.1, 1.3)]
    assert 0.0 < rise[0] < rise[1] < rise[2]                                # smooth transonic rise
    assert wave_drag_ca(ogive, 1.5) > wave_drag_ca(ogive, 4.0) > wave_drag_ca(ogive, 12.0) > 0.0
    long_nose = NoseSpec("tangent_ogive", 27.0, 9.0)
    assert wave_drag_ca(long_nose, 3.0) < 0.6 * wave_drag_ca(ogive, 3.0)    # slenderness pays
    blunt = NoseSpec("cone", 4.5, 9.0)
    assert wave_drag_ca(blunt, 3.0) > 2.0 * wave_drag_ca(ogive, 3.0)
    haack, cone = NoseSpec("haack", 13.5, 9.0), NoseSpec("cone", 13.5, 9.0)
    assert wave_drag_ca(haack, 3.0) < wave_drag_ca(cone, 3.0)               # minimum-drag body


def test_base_and_skin_friction_components():
    assert base_drag_ca(1.0 - 1e-9) == pytest.approx(base_drag_ca(1.0 + 1e-9), rel=1e-6)
    nose = NoseSpec("tangent_ogive", 13.5, 9.0)
    ref = math.pi / 4 * 81.0
    wet = wetted_area_m2(nose, 100.0)
    assert wet > math.pi * 9.0 * 100.0                                      # body plus nose
    assert skin_friction_ca(0.0, wet, ref) > skin_friction_ca(5.0, wet, ref) > 0.0
    on = ca_table(nose, 100.0, power_on=True)
    off = ca_table(nose, 100.0, power_on=False)
    for (m, a), (m2, b) in zip(on, off):
        assert m == m2 and b - a == pytest.approx(base_drag_ca(m), abs=2e-4)
    assert [row[0] for row in on] == list(DEFAULT_MACH)
    assert normal_force_slope_per_rad() == 2.0


# ------------------------------------------------------------ the Starship-class file

@pytest.fixture(scope="module")
def starship_doc():
    return vc.load_validated(vc.catalog_paths()[STARSHIP])


def test_committed_aero_equals_the_aero_generated_from_the_declared_nose(starship_doc):
    generated = stage_nose_aero(starship_doc)
    assert len(generated) == len(starship_doc["stages"]) == 2
    for stage, gen in zip(starship_doc["stages"], generated):
        aero = stage["aero"]
        for key in ("ca", "ca_power_off"):
            assert [r[0] for r in aero[key]] == [r[0] for r in gen[key]]
            assert [r[1] for r in aero[key]] == pytest.approx([r[1] for r in gen[key]], abs=1e-9)
        assert aero["cn_alpha_per_rad"] == gen["cn_alpha_per_rad"]
    # the stage carrying the whole stack has the longer body, hence more skin friction
    assert generated[0]["body_length_m"] > generated[1]["body_length_m"]
    assert generated[0]["ca"][0][1] > generated[1]["ca"][0][1]


def test_starship_class_ignition_sequence(starship_doc):
    """The user-set design rule: the booster relights twice (boostback and
    landing, no entry burn), the ship flies one continuous burn."""
    booster = starship_doc["booster"]["engines"]
    assert booster["restarts_max"] == 2 and booster["usable_for"] == ["boostback", "landing"]
    assert "restart" not in starship_doc                                      # no coast, no relight
    view = load_vehicle_view(STARSHIP)
    assert [len(s.segments) for s in view.stages] == [1, 1]
    assert view.restart is None and view.geometry.fairing_length is None      # the nose is the ship's


def test_starship_class_numbers_are_a_coherent_stack():
    view = load_vehicle_view(STARSHIP)
    assert view.stages[0].diameter_m == 9.0 and view.stages[1].diameter_m == 9.0
    assert 1.3 < view.liftoff_twr_sl < 1.7                                    # Starship lifts off at ~1.5 g
    assert view.stages[1].thrust_vac_n == pytest.approx(14.7e6, rel=0.02)     # published ship engine set
    assert 9_000.0 < view.total_delta_v_m_s < 11_500.0
    assert view.geometry.declared and view.geometry.nose_shape == "tangent_ogive"
    assert view.provenance == "public_estimate" and view.disclaimer
    nose = view.provenance_for("geometry.nose.length_m")
    assert nose is not None and "no public value" in nose.source
