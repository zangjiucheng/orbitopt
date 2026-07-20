"""Checks for orbitopt.mission_config (the YAML mission-config read/validate
surface) and the `orbitopt validate`/`orbitopt run` CLI commands built on it.
Schema-validation tests here are deliberately compute-free (no pygmo/tudatpy)
-- that's the whole point of separating config validation from actually
running a mission. End-to-end tests that run the real geo-raising optimizer
are marked slow, matching test_geo_raising.py's convention.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from orbitopt.mission_config import (
    build_scene_from_config,
    is_valid_mission_config,
    known_kinds,
    load_mission_config,
    validate_mission_config,
)
from orbitopt.scene_format import validate_scene

FIXTURE = Path(__file__).parent / "fixtures" / "geo_raising_config.yaml"

MINIMAL_CONFIG = {
    "schemaVersion": "1.0",
    "mission": {"id": "goes-gto-geo", "title": "GOES-16", "kind": "geo-raising"},
    "output": {"scene_file": "scenes/goes-gto-geo.json"},
}


def test_known_kinds_lists_geo_raising():
    assert "geo-raising" in known_kinds()


def test_minimal_config_validates():
    validate_mission_config(MINIMAL_CONFIG)  # must not raise
    assert is_valid_mission_config(MINIMAL_CONFIG)


def test_fixture_config_loads_and_validates():
    config = load_mission_config(FIXTURE)
    assert config["mission"]["kind"] == "geo-raising"


def test_missing_mission_kind_is_invalid():
    broken = copy.deepcopy(MINIMAL_CONFIG)
    del broken["mission"]["kind"]
    assert not is_valid_mission_config(broken)
    # missing/unknown kind can't be schema-checked (there's no schema to
    # check it against yet) -- ValueError, same as scene_format.validate_scene
    # raises plain ValueError for a missing/unknown schemaVersion.
    with pytest.raises(ValueError):
        validate_mission_config(broken)


def test_unknown_top_level_field_is_invalid():
    """additionalProperties: false should catch a typo'd field name --
    unlike scene-1.0.json, a mission config's whole job is catching mistakes
    before any compute runs."""
    broken = copy.deepcopy(MINIMAL_CONFIG)
    broken["mision"] = {}  # typo
    assert not is_valid_mission_config(broken)


def test_n_burns_below_floor_is_invalid():
    """GeoRaisingProblem requires n_burns >= 2 -- the schema should reject a
    config that asks for fewer before any optimizer runs."""
    broken = copy.deepcopy(MINIMAL_CONFIG)
    broken["campaign_search"] = {"n_burns": {"min": 1, "max": 6}}
    assert not is_valid_mission_config(broken)


def test_unknown_mission_kind_is_invalid():
    broken = copy.deepcopy(MINIMAL_CONFIG)
    broken["mission"]["kind"] = "not-a-real-kind"
    with pytest.raises(ValueError):
        validate_mission_config(broken)


def test_load_mission_config_rejects_non_mapping_yaml(tmp_path):
    path = tmp_path / "not_a_mapping.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_mission_config(path)


@pytest.mark.slow
def test_build_scene_from_config_produces_a_valid_scene():
    config = load_mission_config(FIXTURE)
    config = copy.deepcopy(config)
    config["verification"] = {"enabled": False}  # keep this test to just the optimizer, not tudatpy too
    scene = build_scene_from_config(config)
    validate_scene(scene)
    assert scene["id"] == config["mission"]["id"]
    assert scene["bodies"][-1]["id"] == "spacecraft"


@pytest.mark.slow
def test_build_scene_from_config_runs_verification_and_annotates_subtitle():
    config = load_mission_config(FIXTURE)
    config = copy.deepcopy(config)
    config["verification"] = {"stationkeeping_days": 1.0}  # enabled by default; keep it short
    scene = build_scene_from_config(config)
    validate_scene(scene)
    assert "High-fidelity verification" in scene["subtitle"]


def test_yaml_round_trips_through_pyyaml_the_same_as_json_semantics(tmp_path):
    """Sanity check that the fixture is genuinely readable as plain YAML
    (not accidentally JSON-only), since YAML is the whole point of this
    config format."""
    text = FIXTURE.read_text(encoding="utf-8")
    parsed = yaml.safe_load(text)
    assert parsed["mission"]["kind"] == "geo-raising"


# --------------------------------------------------------------------- CLI


def test_cli_validate_prints_ok_for_the_fixture(capsys):
    from orbitopt.cli import main

    main(["validate", str(FIXTURE)])
    out = capsys.readouterr().out
    assert "OK: geo-raising-fixture (kind: geo-raising)" in out


def test_cli_validate_exits_nonzero_and_prints_a_clean_error_for_a_broken_config(tmp_path, capsys):
    from orbitopt.cli import main

    broken = copy.deepcopy(MINIMAL_CONFIG)
    broken["campaign_search"] = {"n_burns": {"min": 1, "max": 6}}
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(broken), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        main(["validate", str(path)])
    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "INVALID" in out
    # a readable schema error, not a bare Python traceback
    assert "Traceback" not in out


@pytest.mark.slow
def test_cli_run_writes_a_scene_file(tmp_path, capsys):
    from orbitopt.cli import main

    out_path = tmp_path / "geo-raising-fixture.json"
    main(["run", str(FIXTURE), "--out", str(out_path), "--skip-verify"])
    out = capsys.readouterr().out
    assert "wrote" in out
    assert out_path.exists()
    validate_scene(json.loads(out_path.read_text(encoding="utf-8")))
