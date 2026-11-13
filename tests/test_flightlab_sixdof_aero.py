"""F-02 (#227): the aerodynamic coefficient model -- interpolation, boundary
and extrapolation behaviour of the coefficient tables, the axisymmetric
promotion and the packaged models, the angle/force/moment conventions, and the
max-Q diagnostic.

Deliberately compute-free (numpy + the aero/atmosphere modules only), the same
convention as ``test_flightlab_contract.py``: everything here runs in well
under a second on a base install. The 6-DOF side of the same acceptance
(aero feeding the integrator, max-Q of a flown run) lives in
``test_flightlab_sixdof_rigidbody.py``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from orbitopt.flightlab.atmosphere import ExponentialAtmosphere, dynamic_pressure_pa
from orbitopt.flightlab.sixdof.aero import (
    AERO_AXES,
    AeroError,
    AeroModel,
    CoefficientTable,
    MaxQ,
    angle_of_attack_deg,
    load_packaged_aero,
    max_q_of,
    sideslip_deg,
    validate_aero_document,
)

# --------------------------------------------------------------------------
# tables: interpolation and boundaries
# --------------------------------------------------------------------------

MACH = (0.2, 0.8, 1.2, 2.0, 4.0)
CA = (0.32, 0.38, 0.55, 0.42, 0.30)


def ca_1d(policy: str = "clamp") -> CoefficientTable:
    return CoefficientTable(name="ca", axes=("mach",), axis_values={"mach": MACH},
                            values=CA, extrapolation=policy)


def test_nodes_are_exact_and_midpoints_are_linear():
    t = ca_1d()
    for mach, want in zip(MACH, CA):
        assert t.evaluate(mach=mach) == pytest.approx(want)
    # linear between (0.8, 0.38) and (1.2, 0.55): a half-step in reads the half-sum
    assert t.evaluate(mach=1.0) == pytest.approx(0.5 * (0.38 + 0.55))
    # and a quarter-step
    assert t.evaluate(mach=0.9) == pytest.approx(0.38 + 0.25 * (0.55 - 0.38))


def test_boundaries_are_inclusive_and_clamped_beyond():
    t = ca_1d("clamp")
    assert t.evaluate(mach=MACH[0]) == pytest.approx(CA[0])
    assert t.evaluate(mach=MACH[-1]) == pytest.approx(CA[-1])
    # below the first node and above the last: the edge value, numpy.interp style
    assert t.evaluate(mach=-10.0) == pytest.approx(CA[0])
    assert t.evaluate(mach=30.0) == pytest.approx(CA[-1])


def test_linear_extrapolation_continues_the_end_segment_slope():
    t = ca_1d("linear")
    slope_low = (CA[1] - CA[0]) / (MACH[1] - MACH[0])
    assert t.evaluate(mach=0.0) == pytest.approx(CA[0] - slope_low * MACH[0])
    slope_high = (CA[-1] - CA[-2]) / (MACH[-1] - MACH[-2])
    assert t.evaluate(mach=5.0) == pytest.approx(CA[-1] + slope_high * (5.0 - MACH[-1]))
    # inside the table the policy is irrelevant: the same linear blend
    assert t.evaluate(mach=1.0) == pytest.approx(0.5 * (0.55 + 0.38))


def test_error_policy_refuses_outside_and_accepts_the_edges():
    t = ca_1d("error")
    assert t.evaluate(mach=0.2) == pytest.approx(CA[0])  # the edge itself is inside
    assert t.evaluate(mach=4.0) == pytest.approx(CA[-1])
    assert t.evaluate(mach=1.5) == pytest.approx(0.50125)  # 1.2->2.0 segment, 3/8 of it
    with pytest.raises(AeroError, match="outside the table range"):
        t.evaluate(mach=4.5)
    with pytest.raises(AeroError, match="mach"):
        t.evaluate(mach=0.1)


def test_2d_table_is_bilinear_and_ignores_axes_it_does_not_tabulate():
    # cn on (mach, alpha): the product of a Mach ramp and an AoA ramp is the
    # one field whose bilinear read is exact everywhere -- good check
    mach, alpha = (0.5, 1.5), (-5.0, 5.0)
    values = [[m * a for a in alpha] for m in mach]
    t = CoefficientTable(name="cn", axes=("mach", "alpha_deg"),
                         axis_values={"mach": mach, "alpha_deg": alpha}, values=values)
    assert t.evaluate(mach=1.0, alpha_deg=0.0) == pytest.approx(0.0)
    assert t.evaluate(mach=1.0, alpha_deg=5.0) == pytest.approx(5.0)  # linear in each axis
    assert t.evaluate(mach=0.5, alpha_deg=0.0) == pytest.approx(0.0)
    # an axis the table does not carry is ignored, not demanded
    assert t.evaluate(mach=1.5, alpha_deg=-5.0, beta_deg=3.0) == pytest.approx(-7.5)


def test_missing_and_unknown_coordinates_raise():
    t = CoefficientTable(name="ca", axes=("mach",), axis_values={"mach": MACH}, values=CA)
    assert t.evaluate(mach=1.0, alpha_deg=5.0) == pytest.approx(0.465)  # alpha ignored
    with pytest.raises(AeroError, match="needs a value"):
        t.evaluate(alpha_deg=5.0)
    with pytest.raises(AeroError, match="unknown coordinate"):
        t.evaluate(mach=1.0, altitude_m=10.0)


def test_vectorised_queries_broadcast_and_stay_scalar_for_scalars():
    t = ca_1d()
    out = t.evaluate(mach=np.array([0.8, 1.0, 3.0]))
    assert out.shape == (3,)
    assert out[0] == pytest.approx(0.38)
    assert out[2] == pytest.approx(0.36)  # 0.75 of the way down (2.0,0.42)-(4.0,0.30)
    grid = t.evaluate(mach=np.array([[0.8], [1.0]]), alpha_deg=np.array([0.0]))
    assert grid.shape == (2, 1)
    assert isinstance(t.evaluate(mach=1.0), float)


def test_table_construction_validation():
    with pytest.raises(AeroError, match="at least one axis"):
        CoefficientTable(name="x", axes=(), axis_values={}, values=[1.0])
    with pytest.raises(AeroError, match="unknown axis"):
        CoefficientTable(name="x", axes=("aoa_deg",), axis_values={"aoa_deg": (0.0, 1.0)},
                         values=[0.0, 1.0])
    with pytest.raises(AeroError, match="in .* order"):
        CoefficientTable(name="x", axes=("alpha_deg", "mach"),
                         axis_values={"mach": (0.2, 2.0), "alpha_deg": (-5.0, 5.0)},
                         values=[[0.0, 1.0], [2.0, 3.0]])
    with pytest.raises(AeroError, match="shaped"):
        CoefficientTable(name="x", axes=("mach",), axis_values={"mach": (0.2, 2.0)},
                         values=[1.0, 2.0, 3.0])
    with pytest.raises(AeroError, match="increase strictly"):
        CoefficientTable(name="x", axes=("mach",), axis_values={"mach": (2.0, 0.2)},
                         values=[1.0, 2.0])
    with pytest.raises(AeroError, match="extrapolation"):
        ca_1d("nearest")


def test_from_document_with_shared_and_inline_axis_nodes():
    doc = {"axes": ["mach"], "values": list(CA), "extrapolation": "linear",
           "label": "test ca", "provenance": {"kind": "illustrative"}}
    t = CoefficientTable.from_document("ca", doc, axes={"mach": list(MACH)})
    assert t.evaluate(mach=1.0) == pytest.approx(0.465)
    assert t.extrapolation == "linear"
    inline = {"axes": ["mach"], "mach": list(MACH), "values": list(CA)}
    t2 = CoefficientTable.from_document("ca", inline)
    assert t2.evaluate(mach=2.0) == pytest.approx(0.42)
    with pytest.raises(AeroError, match="no nodes"):
        CoefficientTable.from_document("ca", {"axes": ["mach"], "values": list(CA)})


# --------------------------------------------------------------------------
# angles, dynamic pressure, max-Q
# --------------------------------------------------------------------------

def test_alpha_and_beta_sign_conventions():
    assert angle_of_attack_deg([100.0, 0.0, 0.0]) == pytest.approx(0.0)
    # z is down: air moving toward +z in body axes (a body flying nose-up to its
    # flight path) reads a positive alpha, nose-down a negative one
    assert angle_of_attack_deg([100.0, 0.0, 100.0]) == pytest.approx(45.0)
    assert angle_of_attack_deg([100.0, 0.0, -100.0]) == pytest.approx(-45.0)
    assert sideslip_deg([100.0, 100.0, 0.0]) == pytest.approx(45.0)  # sliding right -> +beta
    assert sideslip_deg([100.0, -100.0, 0.0]) == pytest.approx(-45.0)
    # a zero-speed state is a valid query (a vehicle on the pad)
    assert angle_of_attack_deg([0.0, 0.0, 0.0]) == 0.0
    assert sideslip_deg([0.0, 0.0, 0.0]) == 0.0


def test_dynamic_pressure_is_half_rho_v_squared():
    assert dynamic_pressure_pa(1.225, 100.0) == pytest.approx(0.5 * 1.225 * 100.0 ** 2)


def test_max_q_of_picks_and_refines_the_interior_peak():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    q = np.array([100.0, 800.0, 900.0, 400.0])
    peak = max_q_of(t, q, altitude_m=[0.0, 100.0, 200.0, 300.0], mach=[0.3, 0.7, 0.8, 0.75])
    assert peak.q_pa == pytest.approx(900.0)  # the recorded sample, never the parabola
    assert peak.t_s == pytest.approx(2.0 - 1.0 / 3.0)  # vertex of the parabola 800/900/400
    assert peak.interpolated_t_s
    assert peak.altitude_m == pytest.approx(200.0)  # the peak sample's altitude
    assert peak.mach == pytest.approx(0.8)


def test_max_q_of_edge_peaks_and_degenerate_shapes():
    monotonic = max_q_of([0.0, 1.0, 2.0], [100.0, 200.0, 300.0])
    assert monotonic.t_s == pytest.approx(2.0)  # first/last sample: no parabola
    assert not monotonic.interpolated_t_s
    flat = max_q_of([0.0, 1.0], [500.0, 500.0])
    assert flat.q_pa == pytest.approx(500.0)
    assert not flat.interpolated_t_s
    with pytest.raises(AeroError, match="no samples"):
        max_q_of([], [])
    with pytest.raises(AeroError, match="must match"):
        max_q_of([0.0, 1.0], [1.0])
    assert MaxQ(q_pa=25000.0, t_s=60.0).q_kpa == pytest.approx(25.0)


# --------------------------------------------------------------------------
# the model: axisymmetric promotion, forces, documents
# --------------------------------------------------------------------------

def spike_model() -> AeroModel:
    return AeroModel.axisymmetric(name="test-boost",
                                  ca_vs_mach=([0.0, 1.1, 2.5], [0.35, 0.55, 0.40]),
                                  cn_alpha_per_rad=0.2, static_margin=0.05,
                                  reference_area_m2=10.0, reference_length_m=15.0,
                                  reference_span_m=4.0, diameter_m=3.7,
                                  pitch_damping_cm_q=-2.0, yaw_damping_cn_yaw_r=-2.0,
                                  provenance={"kind": "illustrative"})


def test_axisymmetric_promotion_follows_the_documented_formulas():
    m = spike_model()
    slope = 0.2
    a = math.radians(5.0)
    c = m.coefficients(2.5, 5.0, 0.0)  # grid node: the formulas hold exactly
    assert c["ca"] == pytest.approx(0.40 + slope * a * a, rel=1e-9)  # induced drag
    assert c["cn"] == pytest.approx(slope * a, rel=1e-9)
    assert c["cm"] == pytest.approx(-0.05 * slope * a, rel=1e-9)     # stable
    assert c["cy"] == pytest.approx(0.0, abs=1e-12)
    b = math.radians(5.0)
    c2 = m.coefficients(0.5, 0.0, 5.0)
    assert c2["cy"] == pytest.approx(-slope * b, rel=1e-9)           # restoring
    assert c2["ca"] == pytest.approx(0.35 + (0.55 - 0.35) * (0.5 / 1.1) + slope * b * b,
                                    rel=1e-9)
    assert c2["cm"] == pytest.approx(0.0, abs=1e-12)
    # between nodes the curve is linear: cn is exactly linear in alpha everywhere
    assert m.coefficients(1.1, 2.5, 0.0)["cn"] == pytest.approx(slope * math.radians(2.5),
                                                                rel=1e-9)
    assert m.provenance_tag == "illustrative"


def test_force_and_moment_sign_conventions():
    m = spike_model()
    # zero angle: pure drag, no moment
    force, moment = m.force_and_moment(30000.0, 1.5, 0.0, 0.0)
    assert force[0] < 0.0 < force[1] + 1.0  # drag opposes +x... force[1] is ~0
    assert force[1] == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(moment, 0.0)
    ca_1_5 = 0.55 + (0.40 - 0.55) * ((1.5 - 1.1) / (2.5 - 1.1))
    assert force[0] == pytest.approx(-30000.0 * 10.0 * ca_1_5)
    # positive AoA: normal force pushes toward -z (up), pitch moment is restoring
    force, moment = m.force_and_moment(30000.0, 1.5, 5.0, 0.0)
    assert force[2] < 0.0
    assert moment[1] < 0.0
    # positive sideslip: side force toward -y (restoring), yaw moment restoring
    force, moment = m.force_and_moment(30000.0, 1.5, 0.0, 5.0)
    assert force[1] < 0.0
    assert moment[2] < 0.0
    # a pitch rate is damped (needs the airspeed to build the dimensionless rate)
    force, moment = m.force_and_moment(30000.0, 1.5, 0.0, 0.0,
                                       omega_rad_s=[0.0, 0.5, 0.0], speed_of_sound_m_s=330.0)
    assert moment[1] < 0.0
    with pytest.raises(AeroError, match="speed_of_sound"):
        m.force_and_moment(30000.0, 1.5, 0.0, 0.0, omega_rad_s=[0.0, 0.5, 0.0])
    with pytest.raises(AeroError, match="dynamic pressure"):
        m.force_and_moment(-1.0, 1.5, 0.0, 0.0)


def test_document_round_trip_through_to_document_and_files(tmp_path):
    m = spike_model()
    doc = m.to_document()
    validate_aero_document(doc)
    m2 = AeroModel.from_document(doc)
    assert m2.name == m.name
    for state in [(0.5, 3.0, 2.0), (1.5, 0.0, 0.0), (2.5, -8.0, 6.0)]:
        f1 = m.force_and_moment(25000.0, *state)
        f2 = m2.force_and_moment(25000.0, *state)
        assert np.allclose(f1[0], f2[0]) and np.allclose(f1[1], f2[1])
    path = m.write(tmp_path / "roundtrip.yaml")
    from orbitopt.flightlab.sixdof.aero import load_aero_model
    m3 = load_aero_model(path)
    assert np.allclose(m3.force_and_moment(25000.0, 2.0, 4.0, 0.0)[0],
                       m.force_and_moment(25000.0, 2.0, 4.0, 0.0)[0])


def test_axisymmetric_document_round_trip_via_from_document():
    m = spike_model()
    ax = AeroModel.from_document({"schema_version": "flight-aero-1.0",
                                  "kind": "axisymmetric", "name": "test-boost",
                                  "reference": {"diameter_m": 3.7, "area_m2": 10.0,
                                                "length_m": 15.0, "span_m": 4.0},
                                  "ca": [[0.0, 0.35], [1.1, 0.55], [2.5, 0.40]],
                                  "cn_alpha_per_rad": 0.2, "static_margin": 0.05,
                                  "damping": {"cm_q": -2.0, "cn_yaw_r": -2.0},
                                  "provenance": {"kind": "illustrative"}})
    assert ax.tables["ca"].axes == AERO_AXES
    assert np.allclose(ax.force_and_moment(25000.0, 2.0, 4.0, 0.0)[0],
                       m.force_and_moment(25000.0, 2.0, 4.0, 0.0)[0])


def test_schema_rejects_broken_documents():
    good = spike_model().to_document()
    validate_aero_document(good)
    for bad, where in [
        ({k: v for k, v in good.items() if k != "coefficients"}, "coefficients"),
        ({**good, "bogus_key": 1}, "bogus_key"),
        ({**good, "schema_version": "flight-aero-9.9"}, "schema_version"),
        ({**good, "kind": "splines"}, "kind"),
        ({**good, "extrapolation": "magic"}, "extrapolation"),
        ({**good, "coefficients": {**good["coefficients"], "ca": {"axes": ["mach"],
                                                                  "values": 3.0}}}, "values"),
    ]:
        with pytest.raises(AeroError, match=where):
            AeroModel.from_document(bad)


def test_packaged_spike_stage_model_matches_the_vehicle_catalog_table():
    m = load_packaged_aero("spike_stage")
    assert m.provenance_tag == "illustrative"
    assert m.tables.keys() == {"ca", "cn", "cy", "cl", "cm", "cn_yaw"}
    # ca on the alpha=beta= line must reproduce the catalog's cd_mach nodes
    for mach, cd in [(0.0, 0.35), (0.8, 0.38), (1.1, 0.55), (1.5, 0.50), (2.5, 0.40),
                     (4.0, 0.32), (8.0, 0.26)]:
        assert m.coefficients(mach, 0.0, 0.0)["ca"] == pytest.approx(cd, abs=1e-12)
    # and the wave-drag peak reads between the nodes exactly as the table says
    assert m.coefficients(0.95, 0.0, 0.0)["ca"] == pytest.approx(0.38 + 0.5 * (0.55 - 0.38))
    assert m.reference_area_m2 == pytest.approx(math.pi * 3.7 ** 2 / 4.0)


def test_exponential_atmosphere_agrees_with_us76_near_sea_level():
    exp = ExponentialAtmosphere()
    from orbitopt.flightlab.atmosphere import density_us76_kg_m3
    assert exp.density_kg_m3(0.0) == pytest.approx(1.225)
    assert exp.density_kg_m3(1000.0) == pytest.approx(density_us76_kg_m3(1000.0), rel=0.03)
    # pressure consistent with the ideal gas at the model's constant temperature
    assert exp.pressure_pa(5000.0) == pytest.approx(
        exp.density_kg_m3(5000.0) * 287.05287 * 288.150)
