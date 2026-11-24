"""E11-06b robustness: FL-304 standard dispersions on a continuous-burn script, fresh seeds.
Pinned = the script flown as it is (no re-planning); refine = FL-304 per-sample local refine."""
from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "benchmarks"))
import e11_continuous_burn_fly as F  # noqa: E402
import e11_unpowered_search as U  # noqa: E402


def main() -> int:
    import orbitopt.flightlab.recovery.montecarlo as mc
    from orbitopt.flightlab.contract import load_target, read_run
    from orbitopt.flightlab.recovery.pipeline import _to_ecef_state
    from orbitopt.flightlab.recovery.reliability import cluster_bootstrap_rate_ci
    from orbitopt.mission.coupled_branches import booster_state_from_run, separation_handoff
    from orbitopt.reference_eop import freeze_reference, load_snapshot

    res = json.loads(Path(sys.argv[1]).read_text())
    mode = sys.argv[2]; out = Path(sys.argv[3]); n = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    seeds = [int(s) for s in (sys.argv[5].split(",") if len(sys.argv) > 5 else ["8", "9", "10"])]
    x = np.array(res["x"][:13])
    target = load_target(U.B.SCEN)
    run = read_run(U.B.ASCENT)

    def sep_ecef(run_):
        sep = booster_state_from_run(run_, separation_handoff(run_))
        if sep.frame == "ECEF":
            return sep
        r, v = _to_ecef_state(sep)
        return replace(sep, frame="ECEF", position_m=tuple(float(c) for c in r),
                       velocity_m_s=tuple(float(c) for c in v))

    mc.separation_state_ecef = sep_ecef          # disperse around the state the mission flies
    opts = F.plan_options(x, F.LANDING)
    if mode == "refine":
        opts.pop("search"); opts["max_fevals"] = 40
    counts, per = [], {}
    with tempfile.TemporaryDirectory() as tmp, freeze_reference(load_snapshot(U.B.REF / "earth_orientation.json")):
        veh = U.B._vehicle_yaml(res["fins"], Path(tmp))
        for seed in seeds:
            r = mc.run_monte_carlo(target, veh, run, seed=seed, n=n, workers=1, plan_options=opts)
            succ = sum(1 for s in r.samples if s.get("outcome") == "landed")
            reasons = {}
            for s in r.samples:
                if s.get("outcome") != "landed":
                    k = str(s.get("reason") or s.get("outcome")); reasons[k] = reasons.get(k, 0) + 1
            counts.append((succ, len(r.samples))); per[str(seed)] = {"successes": succ, "n": len(r.samples), "reasons": reasons}
            print(f"seed {seed}: {succ}/{len(r.samples)} {reasons}", flush=True)
    tot = sum(c for _, c in counts); ok = sum(s for s, _ in counts)
    ci = cluster_bootstrap_rate_ci(counts)
    row = {"mode": mode, "successes": ok, "n": tot, "rate": ok / tot, "cluster_ci95": list(ci) if ci else None,
           "per_seed": per, "seeds": seeds}
    out.write_text(json.dumps(row, indent=1) + "\n"); print(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
