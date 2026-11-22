"""Resolve replay inputs by their recorded digests, never by a vehicle name."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import yaml
import jsonschema

from orbitopt.flightlab.contract import load_target, target_digest


@lru_cache(maxsize=1)
def _packaged_inputs():
    root = Path(__file__).resolve().parents[1] / "data" / "flightlab"
    vehicles, targets = {}, {}
    for path in (root / "vehicles").glob("*.yaml"):
        vehicles[hashlib.sha256(path.read_bytes()).hexdigest()] = path
    for path in (root / "scenarios").glob("*.yaml"):
        try:
            target = load_target(path)
            targets[target_digest(target.to_dict())] = target
        except (ValueError, KeyError, TypeError, OSError, yaml.YAMLError, jsonschema.ValidationError):
            continue
    return vehicles, targets


def replay_inputs(run, run_dir):
    """Return matching vehicle data and target, or None for missing inputs.

    Local snapshots take precedence over packaged files. A stale or unrelated
    YAML beside a run must never supply its dry mass or recovery destination.
    """
    vehicle, target = None, None
    roots = [Path(run_dir), *list(Path(run_dir).parents)[:3]]
    for root in roots:
        for path in (root / "vehicle.yaml", root / "vehicle" / "vehicle.yaml"):
            if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == run.vehicle_sha256:
                vehicle = yaml.safe_load(path.read_text())
                break
        path = root / "target.yaml"
        if path.is_file() and target is None:
            try:
                candidate = load_target(path)
                if target_digest(candidate.to_dict()) == run.target_sha256:
                    target = candidate
            except (ValueError, KeyError, TypeError, OSError, yaml.YAMLError, jsonschema.ValidationError):
                pass
        if vehicle is not None and target is not None:
            break
    vehicles, targets = _packaged_inputs()
    if vehicle is None and run.vehicle_sha256 in vehicles:
        vehicle = yaml.safe_load(vehicles[run.vehicle_sha256].read_text())
    return vehicle, target or targets.get(run.target_sha256)


@lru_cache(maxsize=1)
def _recorded_mission_configs():
    """Index mission configs whose recorded ascent is available in this checkout."""
    data = Path(__file__).resolve().parents[1] / "data"
    repo = Path(__file__).resolve().parents[3]
    configs = {}
    for path in sorted(data.glob("mission_two_stage_leo*.yaml")):
        try:
            config = yaml.safe_load(path.read_text())
            ascent = config["scenario"]["ascent"]
            if ascent.get("mode") != "run_dir":
                continue
            manifest = json.loads((repo / ascent["run_dir"] / "manifest.json").read_text())
            key = tuple(manifest[field] for field in ("run_id", "target_sha256", "vehicle_sha256"))
            configs.setdefault(key, []).append(config)
        except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError):
            continue
    return configs


def replay_mission_scenario(run):
    """Use the mission's staging policy for its recorded ascent or recovery replay."""
    key = run.run_id, run.target_sha256, run.vehicle_sha256
    configs = _recorded_mission_configs().get(key, ())
    if not configs:
        return None
    from orbitopt.mission.two_stage_leo import scenario_from_config

    scenarios = [scenario_from_config(config) for config in configs]
    first = scenarios[0]
    if all((scenario.handoff, scenario.staging_velocity_m_s) ==
           (first.handoff, first.staging_velocity_m_s) for scenario in scenarios):
        return first
    return None
