"""orbitopt.mission_config adapter for ``mission.kind == "flight-lab"``
(M-03, task #221): run a Flight Lab segment through the unified mission
pipeline and land its ``FlightRun`` in the standardized run record.

The config names a FlightTarget file (plus optional vehicle file and the
``simulate`` knobs); ``build_from_config`` wraps the run as a
:class:`~orbitopt.flightlab.segment.FlightLabSegment`, records it on the
pipeline's ``search`` stage (one candidate: the flight actually flown),
records the Gate-B verdict on the ``verify`` stage, and exports the scene
via the FL-005 importer (:func:`orbitopt.flightlab.scene_export.run_to_scene`)
with the flight's verification report attached -- so ``run_pipeline``, the
Verification Gate, ``.pipeline.json`` / ``.verification.json`` outputs and
the viewer all work exactly as for the patched-conic kinds, and an honest
red verdict flows through the pipeline as a degraded (best-effort) export,
never a silent pass.

Target/vehicle paths resolve as given (absolute or CWD-relative), then
against the packaged data root (``orbitopt/data/flightlab`` and its
``scenarios/`` / ``vehicles/`` subdirs), so a config can name
``scenarios/leo_two_stage.yaml`` regardless of where it is run from.
"""
from __future__ import annotations

from pathlib import Path

from orbitopt.pipeline import (
    REFINE,
    SEARCH,
    SELECT,
    VERIFY,
    StageRecorder,
    default_output_path,
)

__all__ = ["build_from_config", "segment_from_config"]


def _resolve_data_file(value: str) -> Path:
    from orbitopt.flightlab.vehicle_source import data_root

    path = Path(value)
    if path.is_file():
        return path
    root = data_root()
    for candidate in (root / value, root / "scenarios" / value, root / "vehicles" / value):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"flight-lab config file not found (tried CWD and data root): {value}")


def segment_from_config(config: dict):
    """Build the :class:`~orbitopt.flightlab.segment.FlightLabSegment` this
    config describes (public so callers/tests can fly the segment without
    the pipeline)."""
    from orbitopt.flightlab.contract import load_target
    from orbitopt.flightlab.segment import FlightLabSegment

    fl = config["flightlab"]
    target_path = _resolve_data_file(fl["target_file"])
    target = load_target(target_path)
    vehicle = fl.get("vehicle_file")
    scene_file = config["output"]["scene_file"]
    run_dir = fl.get("run_dir") or default_output_path(scene_file, ".flightrun")
    verification_enabled = bool((config.get("verification") or {}).get("enabled", True))
    return FlightLabSegment(
        target=target,
        vehicle_path=vehicle,
        run_dir=run_dir,
        refly=bool(fl.get("refly", True)) and verification_enabled,
        refly_orbits=float(fl.get("refly_orbits", 1.05)),
        refly_max_step_s=float(fl.get("refly_max_step_s", 300.0)),
        timeout_s=float(fl.get("timeout_s", 300.0)),
        run_recovery=bool(fl.get("run_recovery", True)),
        recovery_options=dict(fl.get("recovery_options") or {}) or None,
        target_path=target_path,
    )


def build_from_config(config: dict, recorder=None) -> dict:
    """Pipeline builder for kind ``flight-lab`` (see module docstring)."""
    from orbitopt.flightlab.scene_export import run_to_scene
    from orbitopt.flightlab.segment import PHYSICAL_MODEL
    from orbitopt.verify.report import attach_to_scene

    mission = config["mission"]
    fl = config["flightlab"]
    earth_texture = bool(fl.get("earth_texture", True))
    recorder = StageRecorder() if recorder is None else recorder
    segment = segment_from_config(config)

    with recorder.stage(SEARCH) as st:
        result = segment.run()
        st.candidates = [result.to_dict()]
        st.physical_model = dict(PHYSICAL_MODEL)
        st.convergence.update({"converged": result.verification.optimizer_converged,
                               "run_id": result.run.run_id})
        failed = [c for c in result.verification.checks if not c.passed]
        st.constraint_violations.update({f"check.{c.id}": 1.0 for c in failed})
        st.notes.extend(str(a) for a in result.run.assumptions)
        st.notes.append(f"run directory: {result.run_dir}")

    with recorder.stage(VERIFY) as st:
        v = result.verification
        st.physical_model = {"name": "Flight Lab Gate-B verdict (checks.evaluate "
                                      "+ tudatpy orbit re-fly)",
                             "fidelity": "high-fidelity" if v.physics_verified is not None
                                         else "two-body"}
        st.convergence.update({
            "converged": v.mission_constraints_passed,
            "optimizer_converged": v.optimizer_converged,
            "physics_verified": v.physics_verified,
            "mission_constraints_passed": v.mission_constraints_passed,
            "checks_passed": sum(1 for c in v.checks if c.passed),
            "checks_total": len(v.checks),
        })
        if v.physics_verified is None:
            st.notes.append("physics_verified not assessed (re-fly skipped or unavailable) "
                            "-- never treated as passed")

    recorder.skip(SELECT, "single flown candidate -- nothing to select")
    recorder.skip(REFINE, "Flight Lab grades the flight it flew -- no refinement walk")

    scene = run_to_scene(result.run, earth_texture=earth_texture)
    scene["id"] = mission["id"]
    scene["title"] = mission["title"]
    attach_to_scene(scene, result.report)
    return scene
