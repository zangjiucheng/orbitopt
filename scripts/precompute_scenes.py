"""Regenerate the precomputed built-in mission scenes shipped in
src/orbitopt/data/scenes (see orbitopt.scene_cache).

Run after changing mission code, then commit the files:

    python scripts/precompute_scenes.py            # every built-in mission
    python scripts/precompute_scenes.py starship-flight-15

Each scene is computed afresh (no cache, no shipped scene) and recorded
with the source fingerprint of the code that made it; Mission Control offers
to recompute a shipped scene whose fingerprint differs from the running code.
Starship Flight 15 takes several minutes (return solve + verification +
inclination search).
"""
from __future__ import annotations

import os
import sys
import time


def main(argv: list[str]) -> int:
    from orbitopt.missions import get, list_missions
    from orbitopt.scene_cache import ENV_WRITE_SHIPPED, shipped_dir, source_fingerprint

    os.environ[ENV_WRITE_SHIPPED] = "1"
    ids = argv or [m.id for m in list_missions() if m.id != "solar-system"]  # solar system: instant, uncached
    print(f"source fingerprint {source_fingerprint()} -> {shipped_dir()}")
    for mission_id in ids:
        start = time.perf_counter()
        scene = get(mission_id).load()
        print(f"  {mission_id}: {len(scene['bodies'])} bodies, {time.perf_counter() - start:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
