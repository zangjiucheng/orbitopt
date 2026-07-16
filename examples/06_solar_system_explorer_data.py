"""Cache the "Solar System" SceneData JSON to disk, so
`orbitopt.viz.pv_viewer.load_scene` can open it later without recomputing
(cheap for this scene, but this is also the pattern for any scene that
isn't: see examples/07_mission_timeline_data.py). Re-run to change the
reference epoch.

Run: python examples/06_solar_system_explorer_data.py
Then view it with: orbitopt view scenes/solar-system.json
(or just `orbitopt view solar-system` to compute it live.)
"""
from __future__ import annotations

import json
from pathlib import Path

from orbitopt.viz.solar_system import export_solar_system_data

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "scenes" / "solar-system.json"


def main():
    data = export_solar_system_data(n_samples=180)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
