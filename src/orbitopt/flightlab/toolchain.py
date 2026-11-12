"""The toolchain record every Flight Lab run manifest carries (FL-601).

A Flight Lab run is only reproducible if the exact tool versions behind it
are recorded with it: the locked conda env (envs/cpu.yml, CI), the pinned
pip add-ons (envs/flightlab-landing.txt, envs/sixdof-requirements.txt) and
the pinned external OpenTsiolkovsky binary
(scripts/fetch_opentsiolkovsky.py -- version + SHA-256).
:func:`toolchain_record` collects all of that at write time and
:func:`orbitopt.flightlab.contract.write_run` puts it under the manifest's
``toolchain`` key; ``tests/test_flightlab_contract.py`` asserts the record
is present and complete. Values are plain strings; a tool that is not
installed is recorded as ``"absent"``, never omitted -- "absent" is itself
part of the environment fingerprint.

The lookup uses ``importlib.metadata`` only (no imports of the tools
themselves), so recording is cheap and cannot drag a heavy or broken
dependency into the write path.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import platform
import sys
from pathlib import Path

__all__ = ["TOOL_DISTRIBUTIONS", "toolchain_record", "opentsiolkovsky_pinned"]

ABSENT = "absent"

#: Distribution names whose installed version is recorded, in manifest key
#: order. The Flight Lab compute/verify stack (conda-locked) plus the pinned
#: pip add-on layer (gfold, jsbsim, cvxpy) plus the two libraries the write
#: path itself needs (jsonschema, PyYAML).
TOOL_DISTRIBUTIONS: tuple[str, ...] = (
    "numpy",
    "scipy",
    "astropy",
    "tudatpy",
    "gfold",
    "jsbsim",
    "cvxpy",
    "jsonschema",
    "pyyaml",
)


def _dist_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return ABSENT


def _orbitopt_version() -> str:
    try:
        return importlib.metadata.version("orbitopt")
    except importlib.metadata.PackageNotFoundError:
        # running from a source tree (PYTHONPATH=src) without an installed
        # distribution: say so rather than lie with a stale version. The
        # run manifest already carries the git_sha for the exact commit.
        return "unknown (source tree; see git_sha)"


def _pinned_fetch_script() -> Path | None:
    """The fetch script that pins OpenTsiolkovsky, when this checkout ships
    it (source tree / editable install; a plain wheel does not carry
    scripts/)."""
    root = Path(__file__).resolve().parents[3]
    cand = root / "scripts" / "fetch_opentsiolkovsky.py"
    return cand if cand.is_file() else None


def opentsiolkovsky_pinned(fetch_script: Path | None = None) -> tuple[str, str]:
    """(pinned version, pinned SHA-256) for the OpenTsiolkovsky release asset
    of the current platform -- the single source of truth is the fetch
    script's ``ASSETS`` table, which is executed by path (stdlib-only
    module) rather than duplicated here. Returns ("1.0.0", "unknown") when
    the fetch script is not part of this installation."""
    script = fetch_script if fetch_script is not None else _pinned_fetch_script()
    if script is None:
        from orbitopt.flightlab.adapters.opentsiolkovsky import OT_VERSION

        return OT_VERSION, "unknown (fetch script not shipped with this install)"
    spec = importlib.util.spec_from_file_location("_orbitopt_otsk_fetch", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    try:
        tag = module.detect_platform()
    except SystemExit:
        return module.VERSION, "no pinned asset for this platform"
    return module.VERSION, module.ASSETS[tag][1]


def _binary_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def toolchain_record() -> dict[str, str]:
    """The FL-601 manifest record: interpreter, orbitopt itself, every tool
    distribution's installed version, and OpenTsiolkovsky's pinned version
    with a SHA-256 -- the installed binary's own digest when the adapter can
    find it, else the pinned release-asset digest from the fetch script."""
    from orbitopt.flightlab.adapters.opentsiolkovsky import (  # late: avoids a cycle
        OTBinaryMissingError,
        OT_VERSION,
        find_binary,
    )

    record: dict[str, str] = {"python": platform.python_version(), "orbitopt": _orbitopt_version()}
    for dist in TOOL_DISTRIBUTIONS:
        record[dist.replace("-", "_")] = _dist_version(dist)
    record["opentsiolkovsky"] = OT_VERSION
    try:
        record["opentsiolkovsky_sha256"] = _binary_sha256(find_binary())
    except OTBinaryMissingError:
        _, pinned_sha = opentsiolkovsky_pinned()
        record["opentsiolkovsky_sha256"] = f"binary {ABSENT}; pinned asset {pinned_sha}"
    return record


if __name__ == "__main__":  # manual inspection: python -m orbitopt.flightlab.toolchain
    import json

    json.dump(toolchain_record(), sys.stdout, indent=2, sort_keys=True)
    print()
