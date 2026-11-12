"""Gate B (FL-006): `orbitopt flight simulate` -- ascend, check, re-fly.

:func:`simulate` is the Gate-B pipeline: fly a :class:`FlightTarget` with the
OpenTsiolkovsky v1.0.0 adapter (FL-004), assess the vehicle-file checks that
need it (liftoff thrust-to-weight, upper-stage propellant reserve, and
first-stage recovery propellant reserve), independently re-fly the upper
stage's GCRS insertion state for one orbit with tudatpy, and hand the whole
thing to :func:`orbitopt.flightlab.checks.evaluate` for the final
three-boolean verdict. The run directory is the contract's
(orbitopt.flightlab.contract.write_run), so Mission Control / FL-005 see
exactly what flew; ``--scene`` additionally writes ``<run_dir>/scene.json``.

E1 flies a hand-tuned attitude program (optimization is E2), so the verdict
must be honest about the outcome: with the FL-004 synthetic vehicle the
insertion apoapsis overshoots the 400 km target and
``mission_constraints_passed`` comes back ``false`` with reason code
``INSERTION_APOAPSIS_OUT_OF_TOLERANCE`` -- closing that circle is E2's job.
``physics_verified`` is set only by the tudatpy re-fly (never by the backend
that produced the state it re-flies); without tudatpy it stays ``None``
("not assessed"), never a fake ``True``.

FL-304 extends the same pipeline downrange: when the target declares a
recovery (``recovery.kind != none``) and ``run_recovery`` is on (default),
:func:`orbitopt.flightlab.recovery.pipeline.fly_recovery` flies the FL-302
boostback/entry + FL-303 G-FOLD landing burn off the separation state, swaps
the ballistic ``stage1_debris`` track for the flown ``booster`` track, and
merges the gate/landing checks in, so ``mission_constraints_passed`` covers
insertion *and* touchdown on one run. A recovery that cannot even be flown
leaves the run ascent-only; the gate then reads as a failed data-missing
check, never as silence.

Determinism (acceptance criterion): every input to the pipeline -- target,
vehicle file, attitude program, re-fly settings -- is honoured, nothing
wall-clock enters the run, so two calls write byte-identical run directories
(verified by hashing every file of the two outputs in the tests).

    from pathlib import Path
    from orbitopt.flightlab.contract import load_target
    from orbitopt.flightlab.simulate import simulate
    run = simulate(load_target("scenarios/leo_two_stage.yaml"),
                   Path("vehicles/synthetic_two_stage.yaml"), Path("runs/gateb-0001"))
    print(run.verification)   # honest, three independent booleans
"""
from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from orbitopt.flightlab import checks as _checks
from orbitopt.flightlab.adapters.opentsiolkovsky import AttitudePoint, run_ascent
from orbitopt.flightlab.checks import Check, evaluate, insertion_state
from orbitopt.flightlab.contract import (
    FlightRun,
    FlightTarget,
    Verification,
    write_run,
)

__all__ = ["simulate", "evaluate", "orbit_refly"]


def _parse_utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


_FRAME_ALIASES = {
    "GCRS": ("J2000", "ICRS", "ICRF"),
    "ICRS": ("GCRS", "J2000", "ICRF"),
    "J2000": ("GCRS", "ICRS", "ICRF"),
    "ICRF": ("GCRS", "ICRS", "J2000"),
}


def _spice_frame_candidates(frame: str) -> tuple[str, ...]:
    aliases = _FRAME_ALIASES.get(frame, ())
    return (frame,) + tuple(alias for alias in aliases if alias != frame)


def _spice_rotation(spice, from_frame: str, to_frame: str, et: float):
    import numpy as np

    last_error: Exception | None = None
    for src in _spice_frame_candidates(from_frame):
        for dst in _spice_frame_candidates(to_frame):
            try:
                return np.asarray(spice.compute_rotation_matrix_between_frames(src, dst, float(et)),
                                  dtype=float)
            except Exception as exc:
                last_error = exc
    raise RuntimeError(f"SPICE cannot rotate {from_frame!r} -> {to_frame!r}: {last_error}") from last_error


def orbit_refly(run: FlightRun, *, orbits: float = 1.05,
               max_step_s: float = 300.0) -> tuple[bool | None, tuple[Check, ...], str]:
    """Independent re-fly of the run's GCRS insertion state: rotate the SECO
    state GCRS -> ECLIPJ2000 with SPICE (exact, tudatpy's own frame service),
    re-integrate it point-mass-Earth with
    :func:`orbitopt.verify.tudat_propagate.propagate_two_body_leg` for
    ``orbits`` periods (>= 1 full orbit is the task; 1.05 overshoots the
    return-to-start point so a wrapped-around drift cannot hide), rotate the
    final state back with the SPICE rotation *at the final epoch*, and compare
    osculating elements computed with the propagator's own mu. Drift limits
    are the Gate-G5 constants in orbitopt.flightlab.checks.

    Returns ``(physics_verified, checks, note)``: ``None`` physics when the
    backend is missing/failed ("not assessed", reason
    ``REFLY_BACKEND_MISSING``/``REFLY_BACKEND_ERROR``), ``True``/``False``
    from the drift checks otherwise. Never raises -- a re-fly that cannot run
    is an honest None in the verdict, not a crash of the pipeline."""
    try:
        import numpy as np
        from tudatpy.interface import spice
    except ImportError:
        return None, (Check(
            id="orbit-refly-not-assessed", passed=False,
            reason_code="REFLY_BACKEND_MISSING",
        ),), "orbit re-fly: not assessed (tudatpy is not installed)"
    try:
        from orbitopt.timescales import utc_datetime_to_mjd2000, utc_mjd2000_to_tdb_seconds
        from orbitopt.verify.tudat_propagate import propagate_two_body_leg

        st = insertion_state(run)
        el = _checks.orbital_elements(st["position_m"], st["velocity_m_s"])
        if not math.isfinite(el["period_s"]) or el["period_s"] <= 0.0:
            return False, (Check(
                id="orbit-refly", passed=False, reason_code="REFLY_NOT_ORBITAL",
            ),), "orbit re-fly: insertion state is not a closed orbit (physics_verified=false)"

        et0 = utc_mjd2000_to_tdb_seconds(
            utc_datetime_to_mjd2000(_parse_utc(st["epoch_utc"])) + st["t_s"] / 86400.0)
        state0 = np.array([*st["position_m"], *st["velocity_m_s"]], dtype=float)
        rot0 = _spice_rotation(spice, "GCRS", "ECLIPJ2000", et0)
        duration = float(orbits) * el["period_s"]
        result = propagate_two_body_leg(
            rot0 @ state0[:3], rot0 @ state0[3:], duration,
            perturbing_bodies=("Earth",), central_body="Earth",
            step_size=float(max_step_s), departure_epoch_ephemeris_seconds=et0,
        )
        mu = float(result.model["central_gravitational_parameter"])
        et_f = float(result.epochs[-1])
        rot_f = _spice_rotation(spice, "ECLIPJ2000", "GCRS", et_f)
        final = np.asarray(result.final_state, dtype=float)
        state_f = np.concatenate((rot_f @ final[:3], rot_f @ final[3:]))
        el0 = _checks.orbital_elements(state0[:3], state0[3:], mu)
        el1 = _checks.orbital_elements(state_f[:3], state_f[3:], mu)
    except Exception as exc:  # SPICE kernels missing, propagation failure: not-assessed, with the reason
        return None, (Check(
            id="orbit-refly-not-assessed", passed=False, reason_code="REFLY_BACKEND_ERROR",
        ),), f"orbit re-fly: not assessed ({type(exc).__name__}: {exc})"

    da = abs(el1["semi_major_axis_m"] - el0["semi_major_axis_m"])
    de = abs(el1["eccentricity"] - el0["eccentricity"])
    di = abs(el1["inclination_deg"] - el0["inclination_deg"])
    draan = _checks.angle_diff_deg(el1["raan_deg"], el0["raan_deg"])
    made = [
        ("orbit-refly-semimajor-axis", da, _checks.REFLY_MAX_A_M, "m"),
        ("orbit-refly-eccentricity", de, _checks.REFLY_MAX_DE, ""),
        ("orbit-refly-inclination", di, _checks.REFLY_MAX_DI_DEG, "deg"),
        ("orbit-refly-raan", draan, _checks.REFLY_MAX_DRAAN_DEG, "deg"),
    ]
    out = tuple(
        Check(id=cid, passed=value <= limit, value=value, limit=limit, unit=unit,
              reason_code="" if value <= limit else "REFLY_ELEMENT_DRIFT")
        for cid, value, limit, unit in made
    )
    physics = all(c.passed for c in out)
    note = (f"orbit re-fly (FL-006): tudatpy point-mass Earth, {duration:.0f} s "
            f"(={orbits:.2f} insertion periods), GCRS<->ECLIPJ2000 via SPICE at both epochs; "
            f"physics_verified={physics}")
    return physics, out, note


def simulate(target: FlightTarget, vehicle_path: Path, out_dir: Path,
             *, attitude_program: Sequence[AttitudePoint] | None = None,
             refly: bool = True, refly_orbits: float = 1.05,
             refly_max_step_s: float = 300.0, timeout_s: float = 300.0,
             run_recovery: bool = True,
             recovery_options: Mapping[str, Any] | None = None) -> FlightRun:
    """Fly ``target`` with the vehicle file ``vehicle_path`` into the FlightResult
    run directory ``out_dir`` and return the verified :class:`FlightRun`
    (documented in the module docstring; the frozen FL-006 interface).
    ``attitude_program=None`` uses the vehicle file's program; ``refly=False``
    skips the tudatpy re-fly (physics_verified stays None); ``timeout_s``
    bounds the OpenTsiolkovsky subprocess. ``run_recovery`` (default True)
    flies the FL-304 recovery chain -- FL-302 boostback/entry + FL-303 G-FOLD
    landing burn -- whenever the target declares a recovery, merging the
    booster track and its checks into this run before the verdict is settled;
    ``recovery_options`` passes planner knobs (``max_fevals``,
    ``initial_guess``, ...) through to FL-302. Raises what the adapter raises
    (OTBinaryMissingError / jsonschema.ValidationError / RuntimeError) -- the
    verdict is honest, the pipeline is not; a check failing is a red verdict,
    not an exception (a recovery that cannot even be attempted leaves the run
    ascent-only, and the recovery gate lands as a failed data-missing check)."""
    vehicle_path = Path(vehicle_path)
    out_dir = Path(out_dir)
    run = run_ascent(target, vehicle_path, attitude_program, out_dir, timeout_s=timeout_s)
    vehicle_doc: Mapping[str, Any] = yaml.safe_load(vehicle_path.read_text(encoding="utf-8"))

    if run_recovery and target.recovery.kind != "none":
        from orbitopt.flightlab.recovery.pipeline import fly_recovery
        from orbitopt.flightlab.recovery.propagator import RecoveryError
        try:
            run = fly_recovery(run, target, vehicle_path,
                               plan_options=recovery_options).run
        except (ValueError, RecoveryError) as exc:
            run = replace(
                run,
                assumptions=run.assumptions + (
                    f"FL-304 recovery not flown: {type(exc).__name__}: {exc} "
                    "(recovery-landing gate will read as not assessed)",
                ),
            )

    extra = [c for c in (_checks.liftoff_thrust_to_weight_check(run, vehicle_doc),
                         _checks.propellant_reserve_check(run, vehicle_doc, target),
                         _checks.recovery_propellant_reserve_check(run, vehicle_doc, target))
             if c is not None]
    if refly:
        physics, refly_checks, note = orbit_refly(run, orbits=refly_orbits,
                                                 max_step_s=refly_max_step_s)
    else:
        physics, refly_checks, note = None, (), ("orbit re-fly: not requested "
                                                 "(physics_verified stays None = not assessed)")
    run = replace(
        run,
        verification=Verification(
            optimizer_converged=run.verification.optimizer_converged,
            physics_verified=physics,
            mission_constraints_passed=run.verification.mission_constraints_passed,
            checks=run.verification.checks + tuple(extra) + tuple(refly_checks),
        ),
        assumptions=run.assumptions + (
            note,
            "Gate-B verdict (FL-006): orbitopt.flightlab.checks.evaluate -- the three "
            "verification booleans are independent; None means not assessed, never passed",
        ),
    )
    run = replace(run, verification=evaluate(run, target))
    write_run(out_dir, run)
    if run_recovery and target.recovery.kind != "none":
        # The recovery merge retired stage1_debris from the manifest; sweep the
        # trajectory CSV the ascent adapter left behind for it (read_run
        # rejects orphan trajectory files by contract).
        declared = {t.body_id for t in run.tracks}
        for leftover in (out_dir / "trajectories").glob("*.csv"):
            if leftover.stem not in declared:
                leftover.unlink()
    return run
