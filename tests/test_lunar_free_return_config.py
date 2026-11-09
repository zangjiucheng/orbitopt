"""Checks for the lunar-free-return (Artemis II) mission-config kind:
compute-free schema/CLI-validation tests, plus a @pytest.mark.slow
end-to-end `orbitopt run` that needs pykep + tudatpy."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from orbitopt.mission_config import (
    config_digest,
    is_valid_mission_config,
    known_kinds,
    load_mission_config,
    validate_mission_config,
)

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "artemis2_free_return.yaml"

MINIMAL_CONFIG = {
    "schemaVersion": "1.0",
    "mission": {"id": "artemis2", "title": "Artemis II", "kind": "lunar-free-return"},
    "output": {"scene_file": "scenes/artemis2.json"},
}


def _with(**sections):
    config = copy.deepcopy(MINIMAL_CONFIG)
    config.update(sections)
    return config


def test_known_kinds_lists_lunar_free_return():
    assert "lunar-free-return" in known_kinds()


def test_minimal_config_validates():
    validate_mission_config(MINIMAL_CONFIG)


def test_example_config_validates_and_matches_built_in_defaults():
    config = load_mission_config(EXAMPLE)
    assert config["mission"]["kind"] == "lunar-free-return"
    assert config["lunar_flyby"]["perilune_altitude_km"] == 6545.0
    assert config["entry"]["target_fpa_deg"] == -6.0


@pytest.mark.parametrize("bad", [
    {"mision": {}},                                              # typo'd section
    {"departure": {"date": [2026, 8]}},                          # not a y/m/d triple
    {"departure": {"date": [2026, 8, 1], "mjd2000": 9709.0}},    # both epochs
    {"entry": {"target_fpa_deg": -9.0}},                         # outside the corridor
    {"lunar_flyby": {"perilune_altitude_km": 0}},
    {"propagation": {"step_size_s": 60, "stepsize": 1}},         # typo'd leaf
    {"verification": {"enabled": "yes"}},
])
def test_invalid_configs_are_rejected(bad):
    assert not is_valid_mission_config(_with(**bad))


def test_kind_does_not_accept_another_kinds_sections():
    assert not is_valid_mission_config(_with(launch_window={"start_date": [2026, 9, 1]}))


def test_config_digest_ignores_key_order():
    a = _with(parking_orbit={"altitude_km": 200.0}, entry={"target_fpa_deg": -6.5})
    b = {k: a[k] for k in reversed(list(a))}
    assert config_digest(a) == config_digest(b)
    assert config_digest(a) != config_digest(MINIMAL_CONFIG)


def test_cli_validate_example(capsys):
    from orbitopt.cli import main

    main(["validate", str(EXAMPLE)])
    assert "OK: artemis2 (kind: lunar-free-return)" in capsys.readouterr().out


def test_cli_view_routes_yaml_through_the_config_cache(tmp_path, monkeypatch):
    """`orbitopt view x.yaml` must validate the config and go through the
    source-fingerprint scene cache rather than trying to read YAML as a
    scene JSON."""
    import orbitopt.mission_config as mc
    from orbitopt import cli

    seen = {}

    def fake_cached(config):
        seen["config"] = config
        return {"id": config["mission"]["id"], "bodies": []}

    monkeypatch.setattr(mc, "cached_scene_from_config", fake_cached)
    path = tmp_path / "a2.yml"
    path.write_text(yaml.safe_dump(MINIMAL_CONFIG), encoding="utf-8")
    assert cli.resolve_scene(str(path))["id"] == "artemis2"
    assert seen["config"]["mission"]["kind"] == "lunar-free-return"

    broken = tmp_path / "broken.yaml"
    broken.write_text(yaml.safe_dump(_with(entry={"target_fpa_deg": 3.0})), encoding="utf-8")
    with pytest.raises(SystemExit):
        cli.resolve_scene(str(broken))


def test_cached_scene_from_config_uses_scene_cache(tmp_path, monkeypatch):
    """The view path reuses orbitopt.scene_cache: a second call with the same
    config is a cache hit (no rebuild), a changed config recomputes."""
    import orbitopt.mission_config as mc
    from orbitopt.scene_format import validate_scene

    monkeypatch.setenv("ORBITOPT_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("ORBITOPT_NO_SCENE_CACHE", raising=False)
    calls = []
    scene = {
        "schemaVersion": "1.0", "id": "x", "title": "x", "distanceUnit": "km",
        "bodies": [{"id": "earth", "name": "Earth", "color": "#3366ff", "kind": "planet",
                    "radiusDisplay": 1.0, "position": [0.0, 0.0, 0.0]}],
    }
    validate_scene(scene)

    def fake_build(config):
        calls.append(config_digest(config))
        out = copy.deepcopy(scene)
        out["_verificationReports"] = [{"kind": "k"}]
        return out

    monkeypatch.setattr(mc, "build_scene_from_config", fake_build)
    first = mc.cached_scene_from_config(MINIMAL_CONFIG)
    assert "_verificationReports" not in first
    mc.cached_scene_from_config(MINIMAL_CONFIG)
    assert len(calls) == 1
    mc.cached_scene_from_config(_with(parking_orbit={"altitude_km": 300.0}))
    assert len(calls) == 2


@pytest.mark.slow
def test_cli_run_writes_scene_and_verification_report(tmp_path, capsys):
    from orbitopt.cli import main
    from orbitopt.scene_format import validate_scene

    config = load_mission_config(EXAMPLE)
    config["output"] = {"scene_file": str(tmp_path / "artemis2.json")}
    path = tmp_path / "artemis2.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    main(["run", str(path)])
    scene = json.loads((tmp_path / "artemis2.json").read_text(encoding="utf-8"))
    validate_scene(scene)
    assert scene["id"] == "artemis2"
    report = json.loads((tmp_path / "artemis2.verification.json").read_text(encoding="utf-8"))
    assert report["kind"] == "lunar-free-return-entry-corridor"
