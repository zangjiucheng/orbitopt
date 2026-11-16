"""Bake the E5-05a reference-flight fixture (ADR flightlab-0008).

Flies the shipped ``synthetic_two_stage_rtls`` vehicle live through the
pinned OpenTsiolkovsky adapter and records the run dir that
``mission_two_stage_leo_rtls.yaml`` replays (``ascent.mode: run_dir``), plus
a PROVENANCE.json pinning the separation state the recovery chain plans
from -- read out with the pipeline's own ``separation_state_ecef`` so the
provenance and the graded state cannot drift.

Run from the repository root::

    python scripts/e505a_reference.py
"""
from __future__ import annotations

import datetime
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from orbitopt.flightlab.adapters.opentsiolkovsky import run_ascent  # noqa: E402
from orbitopt.flightlab.contract import load_target, read_run  # noqa: E402
from orbitopt.flightlab.recovery.pipeline import separation_state_ecef  # noqa: E402

FIX = REPO / "tests" / "data" / "mission" / "reference" / "e5-05a-rtls-mission"


def main() -> None:
    vehicle = REPO / "src" / "orbitopt" / "data" / "flightlab" / "vehicles" / \
        "synthetic_two_stage_rtls.yaml"
    target = load_target(REPO / "src" / "orbitopt" / "data" / "flightlab"
                         / "scenarios" / "leo_two_stage_rtls.yaml")
    if FIX.exists():
        shutil.rmtree(FIX)
    ascent = FIX / "ascent"
    ascent.mkdir(parents=True)
    run = run_ascent(target, vehicle, None, ascent, timeout_s=900.0)
    # keep the fixture lean: the contract run dir only (PROVENANCE.json below
    # carries the separation-state provenance the raw outputs would show)
    shutil.rmtree(ascent / "opentsiolkovsky", ignore_errors=True)
    events_s = {e.kind: float(e.t_s) for e in run.events}
    del run  # the fixture is the run dir; the separation state is read back
    sep = separation_state_ecef(read_run(ascent))
    prov = {
        "schema": "e5-05a-rtls-mission-provenance-v1",
        "generated_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(timespec="seconds"),
        "generator": "scripts/e505a_reference.py "
                     "(OpenTsiolkovsky 1.1.0 via the flightlab adapter)",
        "vehicle": "src/orbitopt/data/flightlab/vehicles/synthetic_two_stage_rtls.yaml",
        "ascent_target": "scenarios/leo_two_stage_rtls.yaml",
        "events_s": events_s,
        "separation_state": {
            "t_s": sep.t_s, "frame": sep.frame,
            "position_m": list(sep.position_m),
            "velocity_m_s": list(sep.velocity_m_s),
            "mass_kg": sep.mass_kg,
        },
        "adr": "docs/adr/flightlab-0008",
        "note": "Replayed by mission_two_stage_leo_rtls.yaml (ascent mode run_dir). "
                "The recovery chain plans from THIS computed separation state -- "
                "read out with the pipeline's own separation_state_ecef, not a "
                "textbook or illustrative one. Re-flown after E5-05d (#288) closed "
                "F9: the reference ascent is now the redesigned 128 t restartable "
                "upper stage with a coast gap + re-ignition circularisation, "
                "inserting at rp 281.8 km / ra 296.1 km inside the 250 km band on "
                "the same run that carries the 105.4 t stage-1 RTLS reserve "
                "(ADR-0008 F9 closure).",
    }
    (FIX / "PROVENANCE.json").write_text(json.dumps(prov, indent=2) + "\n",
                                         encoding="utf-8")
    print(f"baked {ascent}: sep mass {sep.mass_kg:.0f} kg")


if __name__ == "__main__":
    main()
