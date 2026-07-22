"""Cross-session disk cache for computed mission scenes.

A built-in mission's ``load()`` recomputes its whole scene every time it's
selected -- for the compute-backed ones (GOES, Mars, Artemis) that's seconds
of optimization + tudatpy propagation, paid again on every app restart, since
the app's in-memory cache (viz.app._scene_cache) only spans one session.

This module persists each computed scene to disk, keyed by a fingerprint of
the orbitopt source tree, so a repeat load reads the JSON back instead of
recomputing -- but any edit to the code that produces scenes silently
invalidates it. That's the deliberate difference from committing a static
scene JSON: a static file would mask source edits (you'd keep seeing the old
scene until you remembered to regenerate it), which is a real hazard while the
scene-producing code is under active development. Here, changing the code
changes the fingerprint, so the next load recomputes on its own.

The cache is strictly an optimization: every failure path (disabled, no source
tree, unreadable/corrupt file, read-only cache dir) falls back to a plain
recompute rather than raising.
"""
from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Callable

from orbitopt.scene_format import read_scene, write_scene

# Set ORBITOPT_NO_SCENE_CACHE=1 to always recompute (never read/write the
# cache); set ORBITOPT_CACHE_DIR to relocate it off the default under the
# user's cache home.
_ENV_DISABLE = "ORBITOPT_NO_SCENE_CACHE"
_ENV_DIR = "ORBITOPT_CACHE_DIR"


def cache_dir() -> Path:
    """Directory holding cached scene JSON. ORBITOPT_CACHE_DIR overrides;
    otherwise ``$XDG_CACHE_HOME/orbitopt/scenes`` (or ``~/.cache/orbitopt/
    scenes``) -- outside the repo, so cached scenes are never git-tracked."""
    override = os.environ.get(_ENV_DIR)
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "orbitopt" / "scenes"


@lru_cache(maxsize=1)
def source_fingerprint() -> str:
    """Short SHA-256 over the *content* (and relative path) of every ``.py``
    file under the orbitopt package. Any source edit changes it, so every
    cached scene invalidates -- broader than strictly necessary (editing an
    unrelated module also invalidates), but the only cost of a false miss is
    one recompute, whereas a false *hit* would serve a stale scene.

    Computed once per process (lru_cache): the running process's code is fixed
    for its lifetime, so its fingerprint is too. That's also why editing a
    source file on disk mid-session correctly does NOT change the fingerprint
    -- the already-imported code still produces the old scene, and the cache
    must agree with it; the new code (and new fingerprint) take effect on the
    next process start.
    """
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def cached_scene(mission_id: str, compute: Callable[[], dict]) -> dict:
    """Return ``compute()``'s SceneData, served from disk when a cache file
    for this ``mission_id`` at the current source fingerprint exists, and
    (re)computed + written otherwise. Never raises on the cache's behalf: a
    disabled cache, missing source tree, unreadable/corrupt file, or
    unwritable cache dir all fall back to a plain ``compute()``.
    """
    if os.environ.get(_ENV_DISABLE):
        return compute()

    try:
        directory = cache_dir()
        path = directory / f"{mission_id}.{source_fingerprint()}.json"
    except Exception:
        return compute()

    try:
        if path.exists():
            return read_scene(path)
    except Exception:
        pass  # corrupt / schema-incompatible cache file -- recompute below

    scene = compute()

    try:
        directory.mkdir(parents=True, exist_ok=True)
        # Keep one file per mission, not one per code revision ever built:
        # drop this mission's other-fingerprint files before writing the new.
        for stale in directory.glob(f"{mission_id}.*.json"):
            if stale != path:
                stale.unlink()
        write_scene(scene, path)
    except Exception:
        pass  # can't persist (read-only fs, validation, ...) -- scene still returned

    return scene
