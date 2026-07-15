"""Cache the "Artemis II Mission" SceneData JSON to disk, so
`orbitopt.viz.pv_viewer.load_scene` can open it later without re-solving
the targeting problem and re-propagating the mission each time (that part
takes a few seconds, mostly the tudatpy propagation at a 60s step). Re-run
this to change the departure date or mission length.

Run: python examples/07_mission_timeline_data.py
Then view it with: python examples/08_pyvista_viewer.py scenes/artemis2-mission.json
"""
from __future__ import annotations

import json
from pathlib import Path

from orbitopt.viz.mission_timeline import compute_and_export_mission

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "scenes" / "artemis2-mission.json"


def main():
    data = compute_and_export_mission()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
