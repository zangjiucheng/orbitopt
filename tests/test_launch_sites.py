"""Checks for orbitopt.launch_sites (launch-site database). Compute-free:
needs only pyyaml + jsonschema."""
from __future__ import annotations

import copy
import math

import jsonschema
import pytest
import yaml

from orbitopt.launch_sites import (
    get_launch_site,
    inclination_from_azimuth,
    load_launch_sites,
    reachable_inclination_range,
    rotation_speed_gain,
    validate_launch_site_data,
)

REQUIRED_SITES = ["cape-canaveral", "vandenberg", "kourou", "jiuquan", "wenchang", "baikonur"]
FIELDS = ["location", "altitude", "azimuth_corridors", "rotation_speed_gain"]


def _raw():
    from importlib import resources
    return yaml.safe_load(resources.files("orbitopt").joinpath("data", "launch_sites.yaml").read_text("utf-8"))


def test_packaged_database_has_initial_sites():
    db = load_launch_sites()
    for sid in REQUIRED_SITES:
        assert sid in db


@pytest.mark.parametrize("sid", REQUIRED_SITES)
def test_every_field_is_sourced(sid):
    site = get_launch_site(sid)
    for field in FIELDS:
        assert site.sources_for(field), f"{sid}: no source for {field}"


@pytest.mark.parametrize("name,expected", [
    ("KSC", "cape-canaveral"), ("cape canaveral", "cape-canaveral"),
    ("VAFB", "vandenberg"), ("CSG", "kourou"), ("Kourou", "kourou"),
    ("酒泉", "jiuquan"), ("JSLC", "jiuquan"), ("文昌", "wenchang"),
    ("Tyuratam", "baikonur"), ("BAIKONUR", "baikonur"),
])
def test_lookup_by_name_or_alias(name, expected):
    assert get_launch_site(name).id == expected


def test_unknown_site_raises_keyerror():
    with pytest.raises(KeyError, match="unknown launch site"):
        get_launch_site("Atlantis")


def test_rotation_gain_physics():
    assert rotation_speed_gain(0.0) == pytest.approx(465.1, abs=0.1)
    assert rotation_speed_gain(90.0) == pytest.approx(0.0, abs=1e-9)
    assert get_launch_site("kourou").rotation_speed_gain_m_s > get_launch_site("baikonur").rotation_speed_gain_m_s


def test_inclination_from_azimuth():
    assert inclination_from_azimuth(28.6, 90.0) == pytest.approx(28.6)
    assert inclination_from_azimuth(28.6, 0.0) == pytest.approx(90.0)
    assert inclination_from_azimuth(28.6, 270.0) == pytest.approx(180.0 - 28.6)
    # cos i = cos(phi) sin(Az)
    i = inclination_from_azimuth(45.0, 60.0)
    assert math.cos(math.radians(i)) == pytest.approx(math.cos(math.radians(45)) * math.sin(math.radians(60)))


def test_reachable_inclinations_known_sites():
    lo, hi = reachable_inclination_range("KSC")
    assert lo == pytest.approx(28.61, abs=0.01)  # due-east launch reaches |lat|
    assert 55 < hi < 62
    lo, hi = reachable_inclination_range("Kourou")  # wraps through north: reaches polar/SSO
    assert lo == pytest.approx(5.24, abs=0.01)
    assert hi > 98
    lo, hi = reachable_inclination_range("Vandenberg")  # no eastward launches
    assert lo > 60 and hi > 98
    ksc = get_launch_site("KSC")
    assert ksc.can_reach_inclination(51.6)
    assert not ksc.can_reach_inclination(98.0)
    assert get_launch_site("baikonur").can_reach_inclination(51.6)
    assert not get_launch_site("baikonur").can_reach_inclination(46.0)


def test_inclination_never_below_latitude():
    for site in load_launch_sites():
        lo, hi = site.reachable_inclination_range()
        assert lo >= abs(site.latitude_deg) - 1e-9
        assert hi <= 180.0 - abs(site.latitude_deg) + 1e-9


def test_disjoint_corridors_give_intervals(tmp_path):
    data = _raw()
    data["sites"] = [copy.deepcopy(data["sites"][0])]
    data["sites"][0]["azimuth_corridors_deg"] = [[80.0, 100.0], [170.0, 190.0]]
    p = tmp_path / "sites.yaml"
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    site = load_launch_sites(p).get("KSC")
    ivs = site.reachable_inclination_intervals()
    assert len(ivs) == 2
    assert not site.can_reach_inclination(60.0)


def test_custom_file_and_validation(tmp_path):
    data = _raw()
    bad = copy.deepcopy(data)
    bad["sites"][0]["rotation_speed_gain_m_s"] = 300.0
    with pytest.raises(ValueError, match="inconsistent"):
        validate_launch_site_data(bad)
    bad = copy.deepcopy(data)
    del bad["sites"][0]["sources"]
    with pytest.raises(jsonschema.ValidationError):
        validate_launch_site_data(bad)
    bad = copy.deepcopy(data)
    bad["sites"][1]["aliases"].append("KSC")
    with pytest.raises(ValueError, match="used by both"):
        validate_launch_site_data(bad)
    p = tmp_path / "sites.json"
    import json
    p.write_text(json.dumps(data), encoding="utf-8")
    assert len(load_launch_sites(p)) == len(data["sites"])
