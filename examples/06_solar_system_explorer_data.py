"""Regenerate the orbital dataset behind the interactive Solar System
Explorer (an HTML/Canvas artifact, not part of this repo -- it embeds this
script's JSON output directly). Re-run this whenever you want the explorer
at a different reference epoch.

Run: python examples/06_solar_system_explorer_data.py > solar_system.json
"""
from __future__ import annotations

import json
import sys

from orbitopt.viz.solar_system import export_solar_system_data


def main():
    data = export_solar_system_data(n_samples=180)
    json.dump(data, sys.stdout, separators=(",", ":"))


if __name__ == "__main__":
    main()
