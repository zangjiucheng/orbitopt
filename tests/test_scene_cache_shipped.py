"""Shipped precomputed scenes (orbitopt.scene_cache): lookup order, the
stale marker, and the scenes actually committed in the repo."""
from __future__ import annotations

import pytest

from orbitopt import scene_cache as sc


@pytest.fixture(scope="module")
def scene():
    from orbitopt.missions import get

    return get("solar-system").load()


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    shipped, local = tmp_path / "shipped", tmp_path / "local"
    monkeypatch.setattr(sc, "shipped_dir", lambda: shipped)
    monkeypatch.setenv("ORBITOPT_CACHE_DIR", str(local))
    for var in ("ORBITOPT_NO_SCENE_CACHE", sc.ENV_RECOMPUTE, sc.ENV_WRITE_SHIPPED):
        monkeypatch.delenv(var, raising=False)
    return shipped, local


def _counting(scene):
    calls = []

    def compute():
        calls.append(1)
        return dict(scene)
    return compute, calls


def test_write_and_read_shipped_round_trip(scene, tmp_path):
    a = sc.write_shipped("demo", scene, "abc123", tmp_path)
    first = a.read_bytes()
    sc.write_shipped("demo", scene, "abc123", tmp_path)
    assert a.read_bytes() == first  # deterministic bytes: no churn in git
    back, fingerprint = sc.read_shipped("demo", tmp_path)
    assert back == scene and fingerprint == "abc123"
    assert sc.read_shipped("missing", tmp_path) is None


def test_current_shipped_scene_is_used_without_computing(scene, dirs):
    shipped, local = dirs
    sc.write_shipped("demo", scene, sc.source_fingerprint(), shipped)
    compute, calls = _counting(scene)
    got = sc.cached_scene("demo", compute)
    assert calls == [] and sc.pop_shipped_marker(got) is None and got == scene


def test_stale_shipped_scene_loads_marked_and_never_enters_the_local_cache(scene, dirs):
    shipped, local = dirs
    sc.write_shipped("demo", scene, "older-code", shipped)
    compute, calls = _counting(scene)
    got = sc.cached_scene("demo", compute)
    assert calls == [] and got[sc.SHIPPED_STALE_KEY] == "older-code"
    assert sc.pop_shipped_marker(got) == "older-code" and sc.SHIPPED_STALE_KEY not in got
    assert not local.exists() or not list(local.glob("demo.*.json"))


def test_recompute_request_and_local_cache_order(scene, dirs, monkeypatch):
    shipped, local = dirs
    sc.write_shipped("demo", scene, "older-code", shipped)
    monkeypatch.setenv(sc.ENV_RECOMPUTE, "other, demo")
    compute, calls = _counting(scene)
    got = sc.cached_scene("demo", compute)
    assert calls == [1] and sc.SHIPPED_STALE_KEY not in got
    # Now cached locally for this code: that wins over the shipped scene.
    monkeypatch.delenv(sc.ENV_RECOMPUTE)
    got = sc.cached_scene("demo", compute)
    assert calls == [1] and sc.SHIPPED_STALE_KEY not in got


def test_write_mode_computes_and_ships(scene, dirs, monkeypatch):
    shipped, _local = dirs
    monkeypatch.setenv(sc.ENV_WRITE_SHIPPED, "1")
    compute, calls = _counting(scene)
    sc.cached_scene("demo", compute)
    assert calls == [1]
    assert sc.read_shipped("demo", shipped)[1] == sc.source_fingerprint()


def test_every_compute_backed_builtin_ships_a_scene():
    """The repo carries a precomputed scene for each built-in that computes
    (scripts/precompute_scenes.py); each reads back and validates."""
    for mission_id in ("artemis2", "goes", "mars", "starship-flight-15"):
        shipped = sc.read_shipped(mission_id)
        assert shipped is not None, f"{mission_id}: run scripts/precompute_scenes.py"
        scene, fingerprint = shipped
        assert scene["bodies"] and len(fingerprint) == 16
