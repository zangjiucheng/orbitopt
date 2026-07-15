"""Regenerate the time-series dataset behind artemis2_mission_timeline.html
(re-solves the free-return targeting problem and propagates the full
mission -- takes a few seconds, mostly the tudatpy propagation at a 60s
step). Re-run this to change the departure date or mission length.

Run: python examples/07_mission_timeline_data.py > mission.json
"""
from __future__ import annotations

import json
import sys

from orbitopt.viz.mission_timeline import compute_and_export_mission


def main():
    data = compute_and_export_mission()
    json.dump(data, sys.stdout, separators=(",", ":"))


if __name__ == "__main__":
    main()
