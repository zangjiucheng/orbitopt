"""FL-201: parametric ascent guidance (decision vector -> program + vehicle).

Fast, no binary: the attitude program, the derived vehicle document and the
bounds are pure functions of the packaged scenario/vehicle. The derived file
is checked against the adapter's real loader (Vehicle.load, i.e. OT's config
schema at build_config time) and the *base* file's bytes for the purity the
optimizer's parallel safety depends on.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from orbitopt.flightlab.adapters.opentsiolkovsky import AttitudePoint, Vehicle
from orbitopt.flightlab.contract import load_target
from orbitopt.flightlab.guidance import ascent_param as ap

REPO = Path(__file__).resolve().parents[1]
VEHICLE = REPO / "src" / "orbitopt" / "data" / "flightlab" / "vehicles" / "synthetic_two_stage.yaml"
TARGET = REPO / "src" / "orbitopt" / "data" / "flightlab" / "scenarios" / "leo_two_stage.yaml"


@pytest.fixture
def target():
    return load_target(TARGET)


@pytest.fixture
def vehicle_doc():
    return yaml.safe_load(VEHICLE.read_text(encoding="utf-8"))


def _mid(lo, hi):
    return [(a + b) / 2.0 for a, b in zip(lo, hi)]


# --- vector layout ----------------------------------------------------------

def test_decision_names_by_objective():
    assert ap.decision_names("reserve")[-1] == "stage2_burn_s"
    assert ap.decision_names("reserve") == ap.decision_names("booster_reserve")
    assert ap.decision_names("payload")[-1] == "payload_kg"
    assert ap.vector_size("payload") == 8 and ap.vector_size("reserve") == 7
    with pytest.raises(ap.GuidanceError):
        ap.decision_names("cost")


def test_unpack_validates():
    lo_hi_ok = [3.0, 5.0, 20.0, 10.0, 45.0, 5.0, 300.0, 1200.0]
    p = ap.unpack(lo_hi_ok, "payload")
    assert p.payload_kg == 1200.0 and p.coast_s == 5.0
    with pytest.raises(ap.GuidanceError):
        ap.unpack(lo_hi_ok[:7], "payload")          # payload objective wants 8
    with pytest.raises(ap.GuidanceError):
        ap.unpack(lo_hi_ok, "reserve")              # reserve objective wants 7
    with pytest.raises(ap.GuidanceError):
        ap.unpack([float("nan")] * 8, "payload")


def test_as_vector_roundtrip():
    vec = [3.0, 5.0, 20.0, 10.0, 45.0, 5.0, 300.0, 1200.0]
    assert ap.as_vector(ap.unpack(vec, "payload")) == vec
    p7 = ap.unpack(vec[:7], "reserve")
    assert ap.as_vector(p7) == vec[:7]              # zero payload -> no 8th slot


# --- bounds ------------------------------------------------------------------

@pytest.mark.parametrize("objective", ["payload", "reserve"])
def test_bounds_shape(vehicle_doc, target, objective):
    lo, hi = ap.bounds(vehicle_doc, target, objective)
    n = ap.vector_size(objective)
    assert len(lo) == len(hi) == n
    assert all(a < b and math.isfinite(a) and math.isfinite(b) for a, b in zip(lo, hi))
    assert all(0.0 <= t <= 60.0 for t in [lo[5], hi[5]])     # coast >= 0, bounded


def test_bounds_azimuth_centred_on_inclination_nominal(vehicle_doc, target):
    lo, hi = ap.bounds(vehicle_doc, target, "reserve")
    az_lo, az_hi = lo[4], hi[4]
    nominal = ap.nominal_azimuth_deg(target.target_orbit.inclination_deg,
                                     target.launch_site.latitude_deg)
    assert az_lo < nominal < az_hi
    assert az_hi - az_lo == pytest.approx(2.0 * ap.AZIMUTH_WINDOW_DEG)


def test_nominal_azimuth_known_values():
    # KSC (28.5 N) to 51.6 deg: the classic 45 deg launch azimuth.
    assert ap.nominal_azimuth_deg(51.6, 28.5) == pytest.approx(45.0, abs=0.2)
    # equator to 0 deg is a due-east launch; 28.5 deg asks for 90 - 28.5.
    assert ap.nominal_azimuth_deg(0.0, 0.0) == pytest.approx(90.0)
    assert ap.nominal_azimuth_deg(28.5, 0.0) == pytest.approx(61.5, abs=0.1)
    with pytest.raises(ap.GuidanceError):
        ap.nominal_azimuth_deg(28.0, 51.6)          # site above the target plane


def test_bounds_cutoff_respects_propellant(vehicle_doc, target):
    last = vehicle_doc["stages"][-1]
    burnout = ap.burnout_time_s(last)
    lo, hi = ap.bounds(vehicle_doc, target, "payload")
    assert hi[6] <= burnout                        # cutoff inside implied burnout
    assert lo[6] > 0.0 and hi[7] > 0.0              # payload bound from upper stage
    assert hi[7] == pytest.approx(ap.PAYLOAD_MAX_PROP_FRACTION
                                  * ap.propellant_mass_kg(last))


def test_bounds_need_a_usable_vehicle(target):
    with pytest.raises(ap.GuidanceError):
        ap.bounds({"stages": []}, target, "payload")
    bare = {"stages": [{"mass_initial_kg": 10.0, "mass_dry_kg": 10.0}],
            "simulation": {"end_time_s": 600.0}}
    with pytest.raises(ap.GuidanceError):            # no propellant -> no burnout time
        ap.bounds(bare, target, "payload")


# --- derived vehicle ----------------------------------------------------------

def test_derive_vehicle_purity_and_payload(vehicle_doc, target):
    base = vehicle_doc
    before = yaml.safe_dump(base)
    lo, hi = ap.bounds(base, target, "payload")
    x = _mid(lo, hi)
    doc = ap.derive_vehicle(base, x)
    assert yaml.safe_dump(base) == before           # base document untouched
    s0, sN = doc["stages"][0], doc["stages"][-1]
    assert s0 is not base["stages"][0]               # deep copy, not live nesting
    assert s0 == base["stages"][0]                   # stage 1 byte-for-byte identical
    assert sN["mass_initial_kg"] - base["stages"][-1]["mass_initial_kg"] == pytest.approx(x[7])
    assert sN["mass_dry_kg"] - base["stages"][-1]["mass_dry_kg"] == pytest.approx(x[7])
    assert sN["engine"]["burn_start_s"] == x[5]      # coast after separation
    # cutoff, stage-local: the burn lasts x[6] whatever the coast
    assert sN["engine"]["burn_end_s"] == pytest.approx(x[5] + x[6])
    if len(base["stages"]) > 2:
        for a, b in zip(doc["stages"][1:-1], base["stages"][1:-1]):
            assert a == b                            # payload only in the LAST stage


def test_derive_vehicle_keyword_payload_overrides_vector(vehicle_doc, target):
    lo, hi = ap.bounds(vehicle_doc, target, "reserve")
    x = _mid(lo, hi)
    doc = ap.derive_vehicle(vehicle_doc, x, payload_kg=2500.0)
    last = doc["stages"][-1]
    assert last["mass_dry_kg"] == pytest.approx(
        yaml.safe_load(VEHICLE.read_text())["stages"][-1]["mass_dry_kg"] + 2500.0)


def test_derived_vehicle_passes_ot_schema(vehicle_doc, target, tmp_path):
    lo, hi = ap.bounds(vehicle_doc, target, "payload")
    doc = ap.derive_vehicle(vehicle_doc, _mid(lo, hi))
    path = tmp_path / "vehicle.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    vehicle = Vehicle.load(path)                    # the adapter's real loader
    assert vehicle.stages[-1]["engine"]["burn_start_s"] == pytest.approx(
        doc["stages"][-1]["engine"]["burn_start_s"])
    assert vehicle.stages[-1]["mass_dry_kg"] == pytest.approx(
        doc["stages"][-1]["mass_dry_kg"])


def test_derive_vehicle_rejects_wrong_length():
    with pytest.raises(ap.GuidanceError):
        ap.derive_vehicle({"stages": [{}]}, [1.0, 2.0])


# --- attitude program ----------------------------------------------------------

def test_program_shape(vehicle_doc, target):
    lo, hi = ap.bounds(vehicle_doc, target, "payload")
    p = ap.unpack(_mid(lo, hi), "payload")
    prog = ap.attitude_program(p, vehicle_doc)
    assert all(isinstance(pt, AttitudePoint) for pt in prog)
    assert len(prog) >= 5
    ts = [pt.t_s for pt in prog]
    assert all(b > a for a, b in zip(ts, ts[1:]))   # strictly increasing
    end_time = vehicle_doc["simulation"]["end_time_s"]
    assert ts[0] == 0.0 and ts[-1] <= end_time
    assert all(0.0 <= pt.elevation_deg <= 90.0 for pt in prog)
    assert len({pt.azimuth_deg for pt in prog}) == 1  # constant azimuth
    assert prog[0].elevation_deg == 90.0             # starts vertical
    # pitch-down is monotone after the kick
    assert prog[-1].elevation_deg <= prog[3].elevation_deg <= prog[2].elevation_deg


@pytest.mark.parametrize("coast_s", [0.0, 30.0])
def test_stage2_ramp_spans_the_burn_whatever_the_coast(vehicle_doc, coast_s):
    """The stage-2 pitch ramp runs el1 -> el2 between ignition and
    PITCH2_LEAD_S before cutoff, with or without a coast. (It used to reach
    el2 during the coast and hold it for the whole burn, which left a
    low-T/W upper stage no way to flatten its climb.)"""
    p = ap.AscentParams(10.0, 8.0, 40.0, 5.0, 45.0, coast_s, 300.0)
    prog = ap.attitude_program(p, vehicle_doc)
    sep = ap.separation_time_s(vehicle_doc)
    ign2, cutoff = sep + coast_s, sep + coast_s + 300.0

    def elev(t):
        pts = [(pt.t_s, pt.elevation_deg) for pt in prog]
        for (t0, e0), (t1, e1) in zip(pts, pts[1:]):
            if t0 <= t <= t1:
                return e0 + (e1 - e0) * (t - t0) / (t1 - t0)
        return pts[-1][1]

    assert elev(ign2) == pytest.approx(40.0)            # held through the coast
    mid = 0.5 * (ign2 + cutoff - ap.PITCH2_LEAD_S)
    assert elev(mid) == pytest.approx(22.5, abs=0.1)    # halfway down the ramp
    assert elev(cutoff - ap.PITCH2_LEAD_S) == pytest.approx(5.0)


def test_program_degrades_gracefully_out_of_bounds(vehicle_doc, target):
    p = ap.AscentParams(-10.0, 400.0, -5.0, 120.0, 45.0, 1e6, 1e7)
    prog = ap.attitude_program(p, vehicle_doc)       # clamped, not rejected
    assert len(prog) >= 3 and all(math.isfinite(pt.t_s) for pt in prog)
    assert all(0.0 <= pt.elevation_deg <= 90.0 for pt in prog)


def test_decode_returns_program_and_vehicle(vehicle_doc, target):
    lo, hi = ap.bounds(vehicle_doc, target, "payload")
    x = _mid(lo, hi)
    prog, doc = ap.decode(x, target, vehicle_doc, "payload")
    assert isinstance(prog, list) and isinstance(prog[0], AttitudePoint)
    assert doc["stages"][-1]["mass_dry_kg"] > vehicle_doc["stages"][-1]["mass_dry_kg"]
    prog7, doc7 = ap.decode(x[:7], target, vehicle_doc, "reserve", payload_kg=0.0)
    assert doc7["stages"][-1]["mass_dry_kg"] == vehicle_doc["stages"][-1]["mass_dry_kg"]
    assert len(prog7) >= 3
