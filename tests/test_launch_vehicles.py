"""Checks for orbitopt.launch_vehicles (launch-vehicle performance model)."""
from __future__ import annotations

import copy
import re

import jsonschema
import pytest
import yaml

from orbitopt.launch_vehicles import (
    PROVENANCES,
    OutOfPerformanceRange,
    get_launch_vehicle,
    known_vehicles,
    load_launch_vehicles,
    parse_launch_vehicles,
)

_MINIMAL = {
    "schema_version": "1.0",
    "vehicles": {
        "toy": {
            "name": "Toy",
            "orbit_capability": {"LEO": {"payload_kg": 1000, "provenance": "official", "source": "toy guide"}},
            "c3_curve": {"provenance": "public_estimate", "source": "toy estimate",
                         "points": [[0, 1000], [10, 600], [20, 400]]},
            "upper_stage": {"max_restarts": 1, "max_coast_s": 3600, "provenance": "official",
                            "source": "toy guide"},
        }
    },
}


def test_default_database_has_the_initial_vehicles():
    keys = set(known_vehicles())
    assert {"falcon-9", "falcon-heavy", "sls-block-1", "atlas-v-551", "vulcan-vc6",
            "long-march-5", "long-march-3b"} <= keys


@pytest.mark.parametrize("key", known_vehicles())
def test_every_value_is_tagged_with_provenance(key):
    v = get_launch_vehicle(key)
    assert v.c3_provenance in PROVENANCES
    assert v.upper_stage.provenance in PROVENANCES
    assert v.upper_stage.max_restarts >= 0 and v.upper_stage.max_coast_s >= 0
    for cap in v.orbit_capability.values():
        assert cap.provenance in PROVENANCES


@pytest.mark.parametrize("key", known_vehicles())
def test_max_payload_and_max_c3_are_inverse(key):
    v = get_launch_vehicle(key)
    lo, hi = v.c3_range
    for c3 in (lo, (lo + hi) / 2, hi):
        m = v.max_payload(c3)
        assert v.max_c3(m) == pytest.approx(c3, abs=1e-9)


def test_interpolation_between_points():
    v = parse_launch_vehicles(_MINIMAL)["toy"]
    assert v.max_payload(0) == 1000
    assert v.max_payload(5) == pytest.approx(800)
    assert v.max_payload(15) == pytest.approx(500)
    assert v.max_c3(800) == pytest.approx(5)
    assert v.max_c3(400) == pytest.approx(20)


@pytest.mark.parametrize("c3", [-0.1, 20.1, 1e6])
def test_c3_out_of_range_raises_and_is_flagged(c3):
    v = parse_launch_vehicles(_MINIMAL)["toy"]
    with pytest.raises(OutOfPerformanceRange, match="extrapolate"):
        v.max_payload(c3)
    r = v.query_max_payload(c3)
    assert not r.in_range and r.value is None and "outside" in r.message


@pytest.mark.parametrize("m", [399.0, 1000.1])
def test_payload_out_of_range_raises_and_is_flagged(m):
    v = parse_launch_vehicles(_MINIMAL)["toy"]
    with pytest.raises(OutOfPerformanceRange):
        v.max_c3(m)
    r = v.query_max_c3(m)
    assert not r.in_range and r.value is None


def test_in_range_query_reports_provenance():
    r = get_launch_vehicle("falcon-9").query_max_payload(10.0)
    assert r.in_range and r.value > 0 and r.provenance == "public_estimate"


def test_orbit_capability_lookup_and_missing_orbit():
    v = get_launch_vehicle("atlas-v-551")
    assert v.payload_to("leo").provenance == "official"
    with pytest.raises(OutOfPerformanceRange, match="TLI"):
        v.payload_to("TLI")


def test_unknown_vehicle_raises_keyerror():
    with pytest.raises(KeyError, match="known"):
        get_launch_vehicle("saturn-v")


def test_non_monotonic_curve_rejected():
    bad = copy.deepcopy(_MINIMAL)
    bad["vehicles"]["toy"]["c3_curve"]["points"] = [[0, 1000], [10, 1100]]
    with pytest.raises(ValueError, match="decreasing"):
        parse_launch_vehicles(bad)


def test_bad_provenance_rejected_by_schema():
    bad = copy.deepcopy(_MINIMAL)
    bad["vehicles"]["toy"]["c3_curve"]["provenance"] = "guess"
    with pytest.raises(jsonschema.ValidationError):
        parse_launch_vehicles(bad)


def test_custom_data_file(tmp_path):
    p = tmp_path / "lv.yaml"
    p.write_text(yaml.safe_dump(_MINIMAL), encoding="utf-8")
    vehicles = load_launch_vehicles(p)
    assert list(vehicles) == ["toy"]
    assert vehicles["toy"].upper_stage.max_burns == 2


# ---- data sources (task #155) ----------------------------------------------

@pytest.mark.parametrize("key", known_vehicles())
def test_every_value_cites_source_accuracy_and_date(key):
    v = get_launch_vehicle(key)
    recs = v.data_sources()
    assert len(recs) == 2 + len(v.orbit_capability)
    for r in recs:
        assert r.source.strip(), f"{r.item}: empty source"
        assert r.provenance in PROVENANCES
        assert r.accuracy, f"{r.item}: no accuracy"
        assert re.fullmatch(r"\d{4}(-\d{2}(-\d{2})?)?", r.as_of), f"{r.item}: bad as_of {r.as_of!r}"


@pytest.mark.parametrize("section", ["orbit", "c3_curve", "upper_stage"])
def test_missing_source_rejected_by_schema(section):
    bad = copy.deepcopy(_MINIMAL)
    toy = bad["vehicles"]["toy"]
    target = toy["orbit_capability"]["LEO"] if section == "orbit" else toy[section]
    del target["source"]
    with pytest.raises(jsonschema.ValidationError):
        parse_launch_vehicles(bad)


def test_data_sources_selection():
    v = get_launch_vehicle("falcon-9")
    (c3,) = v.data_sources(["c3_curve"])
    assert c3.provenance == "public_estimate" and "NASA" in c3.source
    assert v.data_sources(["leo"])[0].provenance == "official"
    with pytest.raises(KeyError):
        v.data_sources(["MEO"])
