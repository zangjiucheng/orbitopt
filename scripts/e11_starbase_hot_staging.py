"""E11-06: a Starbase hot-staging ascent -- the sealed Starbase design with the booster's engines kept lit
through separation (E10-02 timeline: continuous stage-1 throttle-down, ship ignition before separation,
booster MECO just after it), so the boostback needs no relight. The DC8 timeline and stage-1 thrust curve
are applied RELATIVE to the Starbase separation (DC8 separates at 125 s, Starbase at 122 s: delta -3 s)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))

SB = REPO / "src/orbitopt/data/flightlab/vehicles/starship_class_rtls_starbase.yaml"
DC8 = REPO / "src/orbitopt/data/flightlab/vehicles/starship_class_rtls_dc8.yaml"
DELTA_S = 122.0 - 125.0


def _shift_curve(curve, d):
    return [[round(float(t) + d, 3), v] for t, v in curve]


def hot_staging_doc(sb: dict | None = None, dc8: dict | None = None, delta_s: float = DELTA_S) -> dict:
    sb = sb or yaml.safe_load(SB.read_text())
    dc8 = dc8 or yaml.safe_load(DC8.read_text())
    doc = copy.deepcopy(sb)
    st = copy.deepcopy(dc8["staging"])
    for k in ("throttle_down_start_s", "ship_ignition_s", "separation_s", "booster_meco_s"):
        st[k] = round(float(st[k]) + delta_s, 3)
    doc["staging"] = st
    s1, d1 = doc["stages"][0], dc8["stages"][0]
    s1["engine"]["thrust_vac_n"] = _shift_curve(d1["engine"]["thrust_vac_n"], delta_s)
    s1["engine"]["burn_end_s"] = round(float(d1["engine"]["burn_end_s"]) + delta_s, 3)
    s1["separation_time_s"] = st["separation_s"]
    doc["name"] = "starship-class-rtls-starbase-hot-staging-v1"
    return doc


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.write_text(yaml.safe_dump(hot_staging_doc(), sort_keys=False, width=120))
    print(out)
