"""Run the independent SECO orbit re-fly (Gate B, tudatpy) on a recorded Flight Lab run and write the verdict back.

The shipped `flight simulate --refly` does exactly this while it builds a run; a run assembled by the mission
chain (E11-06) never went through it, so its Verification tab said "SECO -- parking orbit insertion: NOT CHECKED"
and the Flight summary said "Physics: N/A". Same functions, same verdict logic, applied to the existing run.

    python scripts/e11_run_orbit_refly.py RUN_DIR SCENARIO.yaml
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    from orbitopt.flightlab.checks import evaluate
    from orbitopt.flightlab.contract import Verification, load_target, read_run, write_run
    from orbitopt.flightlab.simulate import orbit_refly

    run_dir, scenario = Path(sys.argv[1]), Path(sys.argv[2])
    run = read_run(run_dir)
    target = load_target(scenario)
    physics, refly_checks, note = orbit_refly(run, orbits=1.05, max_step_s=300.0)
    # a re-fly that was already recorded must not be stacked twice
    keep = tuple(c for c in run.verification.checks if not c.id.startswith("orbit-refly"))
    run = replace(
        run,
        verification=Verification(
            optimizer_converged=run.verification.optimizer_converged,
            physics_verified=physics,
            backend_self_check=run.verification.backend_self_check,
            mission_constraints_passed=run.verification.mission_constraints_passed,
            checks=keep + tuple(refly_checks)),
        assumptions=tuple(a for a in run.assumptions if not a.startswith("orbit re-fly")) + (
            note,
            "E11-06: the independent SECO orbit re-fly (Gate B, tudatpy point-mass, 1.05 orbits) was run on this "
            "recorded run after it was assembled, with the shipped orbitopt.flightlab.simulate.orbit_refly",),
    )
    run = replace(run, verification=evaluate(run, target))
    write_run(run_dir, run)
    print("physics_verified:", run.verification.physics_verified)
    for c in run.verification.checks:
        if c.id.startswith("orbit-refly"):
            print(f"  {'PASS' if c.passed else 'FAIL'} {c.id} value={c.value} limit={c.limit} {c.unit}")
    print(note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
