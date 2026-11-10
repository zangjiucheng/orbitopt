"""Static consistency checks for the conda environment specs and lockfiles.

Cheap (pure YAML parsing, no solver): guards against the spec and its
lockfile disagreeing -- e.g. the old environment.yml declared numpy<2.0
while every real install resolved numpy 2.x -- and against CuPy leaking
into the CPU-only environment. Regenerating the locks is in envs/README.md.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ENVS = Path(__file__).resolve().parent.parent / "envs"
FLAVORS = ("cpu", "gpu")
GPU_PLATFORMS = {"linux-64", "win-64"}
# macOS has no CUDA, so only the CPU env is locked for osx-64 / osx-arm64.
PLATFORMS = {"cpu": GPU_PLATFORMS | {"osx-64", "osx-arm64"}, "gpu": GPU_PLATFORMS}


def _spec(flavor):
    return yaml.safe_load((ENVS / f"{flavor}.yml").read_text(encoding="utf-8"))


def _lock(flavor):
    return yaml.safe_load((ENVS / f"{flavor}.conda-lock.yml").read_text(encoding="utf-8"))


def _spec_names(spec):
    return {re.split(r"[\s=<>!]", d, maxsplit=1)[0] for d in spec["dependencies"] if isinstance(d, str)}


def _locked(lock, platform):
    return {p["name"]: p for p in lock["package"] if p["platform"] == platform}


def _ver(v):
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


@pytest.mark.parametrize("flavor", FLAVORS)
def test_lockfile_covers_every_spec_package_on_every_platform(flavor):
    spec, lock = _spec(flavor), _lock(flavor)
    assert set(lock["metadata"]["platforms"]) == PLATFORMS[flavor]
    assert not [d for d in spec["dependencies"] if not isinstance(d, str)], "no unlocked pip: section"
    for platform in PLATFORMS[flavor]:
        locked = _locked(lock, platform)
        missing = _spec_names(spec) - set(locked)
        assert not missing, f"{flavor}/{platform} lock lacks {missing}"
        assert all(p["manager"] == "conda" for p in locked.values())


@pytest.mark.parametrize("flavor", FLAVORS)
def test_locked_versions_honour_the_declared_constraints(flavor):
    for platform in PLATFORMS[flavor]:
        locked = _locked(_lock(flavor), platform)
        assert locked["python"]["version"].startswith("3.10.")
        # numpy: tudatpy 1.0.0's own conda pin is numpy<2.0; with every
        # package from conda nothing can drag it to 2.x any more.
        assert (1, 26, 4) <= _ver(locked["numpy"]["version"]) < (2, 0, 0)
        # pykep.trajopt still calls scipy.interpolate.interp2d (gone in 1.14).
        assert (1, 11, 0) <= _ver(locked["scipy"]["version"]) < (1, 14, 0)
        # tudatpy 1.0.0 declares boost-cpp 1.84 (0.9.0 linked 1.85 undeclared
        # and was ABI-broken by a later tudat rebuild).
        assert locked["tudatpy"]["version"].startswith("1.0.")
        assert locked["libboost"]["version"].startswith(locked["boost-cpp"]["version"].rsplit(".", 1)[0])


def test_cpu_env_has_no_cuda_and_gpu_env_pins_cuda13_cupy():
    for platform in PLATFORMS["cpu"]:
        cpu = _locked(_lock("cpu"), platform)
        assert "cupy" not in cpu and "cupy-core" not in cpu
        assert "cuda-version" not in cpu
    for platform in PLATFORMS["gpu"]:
        gpu = _locked(_lock("gpu"), platform)
        assert "cupy" in gpu
        assert gpu["cuda-version"]["version"].startswith("13.")


def test_spec_numpy_constraint_is_the_same_in_both_flavors():
    def numpy_spec(flavor):
        return [d for d in _spec(flavor)["dependencies"] if isinstance(d, str) and d.startswith("numpy")]

    assert numpy_spec("cpu") == numpy_spec("gpu") == ["numpy>=1.26.4,<2"]


def test_gpu_spec_is_cpu_spec_plus_cuda():
    cpu, gpu = _spec("cpu"), _spec("gpu")
    assert cpu["channels"] == gpu["channels"]
    assert gpu["dependencies"] == cpu["dependencies"] + ["cupy", "cuda-version=13"]


def test_macos_locks_pin_native_builds_for_each_arch():
    # osx-arm64 (Apple Silicon) and osx-64 (Intel) are separate native
    # solves; each must only pull packages built for its own subdir.
    lock = _lock("cpu")
    for platform in ("osx-64", "osx-arm64"):
        for pkg in _locked(lock, platform).values():
            assert f"/{platform}/" in pkg["url"] or "/noarch/" in pkg["url"], pkg["url"]
