"""Nominal speed-vs-altitude profile of a continuous-burn script (the closed-loop fin reference)."""
import json, sys, tempfile
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "benchmarks"))
import e11_continuous_burn_search as C, e11_unpowered_search as U
from orbitopt.flightlab.contract import load_target
from orbitopt.reference_eop import freeze_reference, load_snapshot
d = json.load(open(sys.argv[1])); x = np.array(d["x"][:13])
with freeze_reference(load_snapshot(U.B.REF / "earth_orientation.json")):
    U._init(d["fins"], 1.0); U._CTX["recipe"] = {**C.RECIPE, "dt_opt": 0.5, "dt_final": 0.25}; U._CTX["target"] = load_target(U.B.SCEN)
    tr = C._plan(x).track
    h = np.asarray(tr.height_m_arr); v = np.linalg.norm(np.asarray(tr.velocity_ecef_m_s), axis=1); t = np.asarray(tr.times_s)
    k0 = int(np.argmax(t > t[0] + 200))             # after the boostback, on the way down
    prof = []
    for hh in (60000, 40000, 25000, 15000, 8000, 5000, 3500):
        idx = [i for i in range(k0, len(h)) if h[i] <= hh]
        if idx: prof.append([float(hh), float(v[idx[0]])])
    Path(sys.argv[2]).write_text(json.dumps(prof)); print(prof)
