"""M-04: launch window -> ascent optimization -> injection verification, one call.

``run_launch_pipeline`` chains the three existing stages behind a single
entry point (also exposed as ``orbitopt launch`` on the CLI), taking exactly
the three mission inputs the task names -- launch site, rocket, target orbit
-- plus a not-before launch date:

1. **Launch window selection**: :func:`orbitopt.flightlab.target_export.build_target`
   checks inclination feasibility and (for a RAAN-constrained plane) finds
   the first in-plane node pass within ``window_days``; a target that cannot
   be launched never reaches a simulator.
2. **Ascent optimization**: :func:`orbitopt.flightlab.optimize.run.optimize_ascent`
   (pygmo global search + local refinement) flies the parametric ascent and
   writes the winner as a FlightResult run.
3. **Injection verification**: :func:`injection_verification` derives the
   achieved Keplerian elements of the frame-bridged GCRS insertion state
   (same source as the Gate-B checks) and grades the semi-major axis,
   eccentricity, inclination and RAAN errors against tolerances that come
   from the target's own constraints (never invented here).

Whatever happens, a ``VerificationReport`` (kind ``launch-pipeline``) is
written to ``<out_dir>/launch.verification.json`` carrying the window record,
the optimizer verdict and the per-element injection errors under
``targets``; the pipeline status is mirrored into the report's ``status``.

Failure honesty: the function never raises for a mission-level failure -- it
returns a :class:`LaunchPipelineResult` whose ``status`` is one of
:data:`STATUS_OK`, :data:`STATUS_WINDOW_INFEASIBLE`,
:data:`STATUS_OPTIMIZATION_NOT_CONVERGED` or
:data:`STATUS_INJECTION_OUT_OF_TOLERANCE`, with a human-readable ``message``
naming the reason and machine-readable ``reason_codes``. Only *input* errors
(unknown site, missing vehicle file, malformed orbit spec, missing pygmo)
raise. When both the optimizer and the injection miss, the earlier stage's
status wins -- an unconverged ascent explains an off-target orbit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = ["STATUS_OK", "STATUS_WINDOW_INFEASIBLE", "STATUS_OPTIMIZATION_NOT_CONVERGED",
           "STATUS_INJECTION_OUT_OF_TOLERANCE", "LaunchPipelineResult",
           "run_launch_pipeline", "injection_verification"]

STATUS_OK = "OK"
STATUS_WINDOW_INFEASIBLE = "WINDOW_INFEASIBLE"
STATUS_OPTIMIZATION_NOT_CONVERGED = "OPTIMIZATION_NOT_CONVERGED"
STATUS_INJECTION_OUT_OF_TOLERANCE = "INJECTION_OUT_OF_TOLERANCE"
#: Pipeline-level statuses, in precedence order (earlier stage wins).
STATUSES = (STATUS_WINDOW_INFEASIBLE, STATUS_OPTIMIZATION_NOT_CONVERGED,
            STATUS_INJECTION_OUT_OF_TOLERANCE, STATUS_OK)

_REPORT_FILE = "launch.verification.json"


@dataclass(frozen=True)
class LaunchPipelineResult:
    """JSON-friendly summary of one unified launch-pipeline run. ``to_dict()``
    is what the CLI prints and what a caller records; the written artifacts
    (target.yaml, the ascent run directory, launch.verification.json) carry
    the full detail."""

    status: str
    message: str
    reason_codes: tuple[str, ...] = ()
    window: dict | None = None
    target_path: Path | None = None
    ascent_run_dir: Path | None = None
    run_id: str | None = None
    optimization: dict | None = None
    injection: dict | None = None
    report: dict | None = None
    report_path: Path | None = None
    out_dir: Path = field(default_factory=lambda: Path("."))

    @property
    def ok(self) -> bool:
        """True only for :data:`STATUS_OK` -- every other status failed."""
        return self.status == STATUS_OK

    def to_dict(self) -> dict:
        """JSON-serializable view (report document omitted; ``report_path``
        points at it on disk)."""
        return {
            "status": self.status,
            "message": self.message,
            "reason_codes": list(self.reason_codes),
            "window": self.window,
            "target_file": str(self.target_path) if self.target_path else None,
            "ascent_run_dir": str(self.ascent_run_dir) if self.ascent_run_dir else None,
            "run_id": self.run_id,
            "optimization": _drop_heavy(self.optimization),
            "injection": self.injection,
            "report_file": str(self.report_path) if self.report_path else None,
        }


def _drop_heavy(doc: dict | None) -> dict | None:
    if doc is None:
        return None
    return {k: v for k, v in doc.items() if k != "convergence_history"}


def _resolve_vehicle(vehicle: str | Path) -> Path:
    """Rocket input: an existing YAML path, or a name resolved against the
    packaged vehicle catalog (``data/flightlab/vehicles/<name>.yaml``)."""
    from orbitopt.flightlab.vehicle_source import data_root

    path = Path(vehicle)
    if path.is_file():
        return path
    stem = path.name[:-len(path.suffix)] if path.suffix else path.name
    tried = [path.resolve()]
    for candidate in (data_root() / "vehicles" / f"{stem}.yaml",
                      data_root() / "vehicles" / f"{stem}.yml"):
        if candidate.is_file():
            return candidate
        tried.append(candidate)
    raise FileNotFoundError(
        f"vehicle {vehicle!r} not found; tried:\n"
        + "\n".join(f"  {p}" for p in tried)
        + f"\n(give a path, or a name from {data_root() / 'vehicles'})")


def target_orbit_elements(target: Any) -> dict:
    """The graded target orbit as Keplerian elements: a and e derived from the
    target's periapsis/apoapsis altitudes (circular-orbit-consistent with the
    Gate-B altitude checks), inclination and RAAN taken directly."""
    from orbitopt.flightlab.checks import EARTH_RADIUS_M

    orbit = target.target_orbit
    peri, apo = float(orbit.periapsis_altitude_m), float(orbit.apoapsis_altitude_m)
    a = EARTH_RADIUS_M + 0.5 * (peri + apo)
    return {
        "semi_major_axis_m": a,
        "eccentricity": (apo - peri) / (2.0 * a),
        "inclination_deg": float(orbit.inclination_deg),
        "raan_deg": None if orbit.raan_deg is None else float(orbit.raan_deg),
        "periapsis_altitude_m": peri,
        "apoapsis_altitude_m": apo,
    }


def _injection_tolerances(target: Any) -> dict:
    """Injection tolerances, every one derived from a documented quantity --
    never invented at this layer:

    * semi-major axis: the target's own orbit-altitude tolerance (default
      ``checks.DEFAULT_ORBIT_ALTITUDE_TOLERANCE_M``) -- |da| is exactly the
      mean-altitude error.
    * eccentricity: the same altitude budget expressed in e-units at the
      target's own shape -- with periapsis and apoapsis each inside the
      altitude tolerance the eccentricity error is bounded by
      alt_tol*(1+e_t)/a_t, so an e-check at that level grades the same
      tolerance the Gate-B altitude checks enforce, in the element the
      task grades.
    * inclination: the target's inclination tolerance (default
      ``checks.DEFAULT_INCLINATION_TOLERANCE_DEG``).
    * RAAN: ``checks.RAAN_TOLERANCE_DEG``, the Gate-B insertion-RAAN gate.
    """
    from orbitopt.flightlab import checks

    alt_tol = target.constraints.orbit_altitude_tolerance_m
    alt_tol = checks.DEFAULT_ORBIT_ALTITUDE_TOLERANCE_M if alt_tol is None else float(alt_tol)
    inc_tol = target.constraints.inclination_tolerance_deg
    inc_tol = checks.DEFAULT_INCLINATION_TOLERANCE_DEG if inc_tol is None else float(inc_tol)
    e_t = target_orbit_elements(target)["eccentricity"]
    a_t = target_orbit_elements(target)["semi_major_axis_m"]
    return {
        "semi_major_axis_m": alt_tol,
        "eccentricity": alt_tol * (1.0 + e_t) / a_t,
        "inclination_deg": inc_tol,
        "raan_deg": float(checks.RAAN_TOLERANCE_DEG),
    }


def injection_verification(run: Any, target: Any) -> dict:
    """Grade a flown run's insertion against its target orbit in the four
    elements the task grades (a, e, i, RAAN).

    The achieved state is the frame-bridged GCRS insertion state that
    ``orbitopt.flightlab.checks`` itself grades (``insertion_state`` +
    ``orbital_elements``); the RAAN comparison goes through the Gate-B
    ``insertion-raan-within-tolerance`` check when the run was graded against
    a RAAN-constrained target (``raan_report``), so the pipeline and the run's
    own verdict can never disagree. Returns the block stored under
    ``targets.injection`` of the report: ``achieved`` / ``target`` /
    ``error`` / ``tolerance`` / ``checks`` / ``passed`` / ``reason_codes``.
    """
    from orbitopt.flightlab import checks
    from orbitopt.flightlab.window_targeting import (raan_report,
                                                     signed_angle_diff_deg)

    state = checks.insertion_state(run)
    el = checks.orbital_elements(state["position_m"], state["velocity_m_s"])
    tgt = target_orbit_elements(target)
    tol = _injection_tolerances(target)

    orbital = math.isfinite(el["semi_major_axis_m"]) and el["eccentricity"] < 1.0
    da = el["semi_major_axis_m"] - tgt["semi_major_axis_m"] if orbital else None
    de = el["eccentricity"] - tgt["eccentricity"] if orbital else None
    di = el["inclination_deg"] - tgt["inclination_deg"] if orbital else None

    raan_t = tgt["raan_deg"]
    draan = None
    raan_source = "not graded (target plane unconstrained)"
    if orbital and raan_t is not None:
        try:
            rr = raan_report(run, target)
            draan, raan_pass = float(rr["error_deg"]), bool(rr["passed"])
            raan_source = rr["source"]
        except ValueError:
            draan = signed_angle_diff_deg(float(el["raan_deg"]), raan_t)
            raan_pass = checks.angle_diff_deg(float(el["raan_deg"]), raan_t) \
                <= tol["raan_deg"]
            raan_source = "orbitopt.flightlab.checks elements vs target RAAN " \
                          "(run carries no graded RAAN check)"
    else:
        raan_pass = None

    passed_a = bool(da is not None and abs(da) <= tol["semi_major_axis_m"])
    passed_e = bool(de is not None and abs(de) <= tol["eccentricity"])
    passed_i = bool(di is not None and abs(di) <= tol["inclination_deg"])
    passed_raan = bool(raan_pass) if raan_pass is not None else None

    element_reasons = []
    if not orbital:
        element_reasons.append("INJECTION_NOT_ORBITAL")
    else:
        if not passed_a:
            element_reasons.append("INJECTION_SEMI_MAJOR_AXIS_OUT_OF_TOLERANCE")
        if not passed_e:
            element_reasons.append("INJECTION_ECCENTRICITY_OUT_OF_TOLERANCE")
        if not passed_i:
            element_reasons.append("INJECTION_INCLINATION_OUT_OF_TOLERANCE")
        if passed_raan is False:
            element_reasons.append("INJECTION_RAAN_OUT_OF_TOLERANCE")

    graded = [passed_a, passed_e, passed_i] + ([passed_raan] if passed_raan is not None else [])
    fin = lambda x: float(x) if x is not None and math.isfinite(x) else None  # noqa: E731
    return {
        "source": "orbitopt.flightlab.checks.insertion_state + orbital_elements "
                  "(frame-bridged GCRS insertion state)",
        "insertion_epoch_s": float(state["t_s"]),
        "achieved": {
            "semi_major_axis_m": fin(el["semi_major_axis_m"]),
            "eccentricity": fin(el["eccentricity"]),
            "inclination_deg": fin(el["inclination_deg"]),
            "raan_deg": fin(el["raan_deg"]),
            "periapsis_altitude_m": fin(el["periapsis_altitude_m"]),
            "apoapsis_altitude_m": fin(el["apoapsis_altitude_m"]),
        },
        "target": tgt,
        "error": {"delta_semi_major_axis_m": da, "delta_eccentricity": de,
                  "delta_inclination_deg": di, "delta_raan_deg": draan},
        "tolerance": tol,
        "checks": {"semi_major_axis": passed_a, "eccentricity": passed_e,
                   "inclination": passed_i, "raan": passed_raan},
        "raan_source": raan_source,
        "passed": all(graded),
        "reason_codes": element_reasons,
    }


def _window_block(target: Any, derivation: Any) -> dict:
    if derivation is None:
        return {"mode": "free-plane",
                "note": "target plane unconstrained: requested epoch flown as-is",
                "liftoff_epoch_utc": target.mission.epoch_utc}
    return {"mode": "in-plane",
            "requested_raan_deg": derivation.requested_raan_deg,
            "anchor_utc": derivation.anchor_utc,
            "epoch_raan_deg": derivation.epoch_raan_deg,
            "liftoff_epoch_utc": target.mission.epoch_utc,
            "window": derivation.window.to_dict()}


def _report_inputs(*, site: str, vehicle_path: Path | None, orbit_spec: str,
                   date_utc: Any, raan_deg: float | None, window_days: float,
                   opt_kwargs: Mapping[str, Any], target: Any,
                   run_id: str | None, target_sha256: str | None,
                   out_dir: Path) -> dict:
    return {
        "site": site,
        "vehicle_file": str(vehicle_path) if vehicle_path else None,
        "orbit_spec": orbit_spec,
        "date_utc": str(date_utc),
        "raan_deg": raan_deg,
        "window_days": window_days,
        "liftoff_epoch_utc": None if target is None else target.mission.epoch_utc,
        "run_id": run_id,
        "target_sha256": target_sha256,
        "out_dir": str(out_dir),
        **dict(opt_kwargs),
    }


def _build_report(*, inputs: Mapping, window: dict | None, ascent: dict | None,
                  injection: dict | None, checks_map: Mapping[str, bool],
                  message: str, reason_codes: Sequence[str],
                  status: str, label: str) -> dict:
    from orbitopt.verify.report import (build_report, report_digest,
                                        status_from_checks)

    targets = {
        "pipeline_status": status,
        "reason": message,
        "reason_codes": list(reason_codes),
        "window": window,
        "ascent": ascent,
        "injection": injection,
    }
    report = build_report("launch-pipeline", label, [], inputs=dict(inputs),
                          targets=targets)
    report["status"] = status_from_checks(
        checks_map, {name: _check_category(name) for name in checks_map})
    report["digest"] = report_digest(report)
    return report


def _check_category(name: str) -> str:
    """V-04 (task #243) evidence category of one launch-pipeline check: the
    optimizer converging is numerical consistency (converged targeting);
    the window against the site corridor, the Gate-B verdicts and the
    injection against the target orbit are model agreement."""
    from orbitopt.verify.report import CHECK_MODEL, CHECK_NUMERICAL

    if name == "ascent.optimizer_converged":
        return CHECK_NUMERICAL
    if name in ("window.selected", "ascent.physics_verified",
                "ascent.mission_constraints_passed") or name.startswith("injection."):
        return CHECK_MODEL
    raise ValueError(f"launch-pipeline check {name!r} has no verification category")


def _write(result_kwargs: dict, report: dict, out: Path):
    from orbitopt.verify.report import write_report

    path = write_report(report, out / _REPORT_FILE)
    return LaunchPipelineResult(report_path=path, **result_kwargs)


def run_launch_pipeline(*, site: str, orbit: str | Sequence[float], vehicle: str | Path,
                        date_utc: Any, raan_deg: float | None = None,
                        window_days: float = 14.0, out_dir: str | Path,
                        constraints: dict | None = None,
                        recovery: dict | None = None,
                        objective: str = "payload", payload_kg: float = 0.0,
                        seed: int = 42, algorithm: str = "cmaes",
                        generations: int = 60, population: int = 48,
                        workers: int | None = None, local: bool = True,
                        refly: bool = True, timeout_s: float = 300.0,
                        sensitivity_offsets_s: Sequence[float] | None = None,
                        ) -> LaunchPipelineResult:
    """Run launch window selection -> ascent optimization -> injection
    verification as one pipeline; see the module docstring for the chain and
    the failure semantics.

    ``orbit`` is the CLI spec (``"400x400@51.6"``, km altitudes -- the shape
    :func:`orbitopt.flightlab.target_export.parse_orbit_spec` documents) or a
    ``(periapsis_m, apoapsis_m, inclination_deg)`` triple in metres/degrees.
    ``vehicle`` is a vehicle YAML path or a packaged catalog name.
    ``out_dir`` receives ``target.yaml``, ``ascent/`` (the winner's
    FlightResult run + ``optimization.json``) and ``launch.verification.json``.
    ``sensitivity_offsets_s`` defaults to ``None`` (no extra epoch-offset
    re-flights); the standalone ``flight optimize`` command keeps its own
    default of +/-60 s. Raises KeyError/FileNotFoundError/ValueError/RuntimeError
    on input trouble (unknown site, vehicle file, orbit spec, missing pygmo);
    mission-level failures come back in the result."""
    from orbitopt.flightlab.target_export import (build_target, parse_orbit_spec,
                                                  save_target)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if isinstance(orbit, str):
        peri, apo, inc = parse_orbit_spec(orbit)
        orbit_spec = orbit
    else:
        peri, apo, inc = (float(v) for v in orbit)
        orbit_spec = f"{peri / 1000.0:g}x{apo / 1000.0:g}@{inc:g}"
    vehicle_path = _resolve_vehicle(vehicle)

    # ---- stage 1: launch window selection (fail before anything flies) ----
    try:
        target = build_target(site, peri, apo, inc, raan_deg=raan_deg,
                              date_utc=date_utc, constraints=constraints,
                              recovery=recovery, window_days=window_days)
        from orbitopt.flightlab.window_targeting import derive_in_plane_liftoff
        derivation = (derive_in_plane_liftoff(target, window_days=window_days)
                      if target.target_orbit.raan_deg is not None else None)
    except ValueError as exc:
        inputs = _report_inputs(site=site, vehicle_path=vehicle_path,
                                orbit_spec=orbit_spec, date_utc=date_utc,
                                raan_deg=raan_deg, window_days=window_days,
                                opt_kwargs=_opt_kwargs(locals()), target=None,
                                run_id=None, target_sha256=None, out_dir=out)
        report = _build_report(
            inputs=inputs, window=None, ascent=None, injection=None,
            checks_map={"window.selected": False}, message=str(exc),
            reason_codes=["WINDOW_INFEASIBLE"], status=STATUS_WINDOW_INFEASIBLE,
            label=f"{site} {orbit_spec}")
        return _write(dict(status=STATUS_WINDOW_INFEASIBLE, message=str(exc),
                           reason_codes=("WINDOW_INFEASIBLE",), out_dir=out),
                      report, out)

    target_path = save_target(target, out / "target.yaml")
    window = _window_block(target, derivation)

    # ---- stage 2: ascent optimization (pygmo; winner written as a run) ----
    from orbitopt.flightlab.optimize.run import OptimizeAbort, optimize_ascent

    ascent_dir = out / "ascent"
    opt_kwargs = _opt_kwargs(locals())
    try:
        opt = optimize_ascent(target, vehicle_path, ascent_dir,
                              target_path=target_path, objective=objective,
                              payload_kg=payload_kg, seed=seed,
                              algorithm=algorithm, generations=generations,
                              population=population, workers=workers,
                              local=local, refly=refly, timeout_s=timeout_s,
                              derive_window=raan_deg is not None,
                              window_days=window_days,
                              sensitivity_offsets_s=sensitivity_offsets_s)
    except OptimizeAbort as exc:
        codes = tuple(getattr(exc, "reason_codes", ()) or ())
        ascent = {"status": "NO_VALID_FLIGHT", "optimizer_converged": False,
                  "reason_codes": list(codes), "error": str(exc)}
        report = _build_report(
            inputs=_report_inputs(site=site, vehicle_path=vehicle_path,
                                  orbit_spec=orbit_spec, date_utc=date_utc,
                                  raan_deg=raan_deg, window_days=window_days,
                                  opt_kwargs=opt_kwargs, target=target,
                                  run_id=None, target_sha256=None, out_dir=out),
            window=window, ascent=ascent, injection=None,
            checks_map={"window.selected": True,
                        "ascent.optimizer_converged": False},
            message=f"ascent optimization never produced a valid flight: {exc}",
            reason_codes=("ASCENT_OPTIMIZATION_NOT_CONVERGED", *codes),
            status=STATUS_OPTIMIZATION_NOT_CONVERGED, label=target.mission.id)
        return _write(dict(status=STATUS_OPTIMIZATION_NOT_CONVERGED,
                           message=f"ascent optimization never produced a valid "
                                   f"flight: {exc}",
                           reason_codes=("ASCENT_OPTIMIZATION_NOT_CONVERGED", *codes),
                           window=window, target_path=target_path,
                           optimization=ascent, out_dir=out),
                      report, out)

    run = opt["run"]
    opt_doc = {k: v for k, v in opt.items() if k != "run"}

    # ---- stage 3: injection verification (a, e, i, RAAN) ----
    injection = injection_verification(run, target)
    converged = bool(opt["optimizer_converged"])
    v = run.verification

    if converged and injection["passed"]:
        status, message, codes = STATUS_OK, (
            f"launched {run.run_id} from {site} into {orbit_spec}: ascent "
            f"converged, injection within tolerance"), ()
    elif not converged:
        status = STATUS_OPTIMIZATION_NOT_CONVERGED
        codes = ("ASCENT_OPTIMIZATION_NOT_CONVERGED", *opt["reason_codes"])
        message = (f"ascent optimizer did not converge: status "
                   f"{opt['status']}, reason codes {list(opt['reason_codes'])}")
    else:
        status = STATUS_INJECTION_OUT_OF_TOLERANCE
        codes = ("INJECTION_OUT_OF_TOLERANCE", *injection["reason_codes"])
        message = _injection_message(injection)

    checks_map: dict[str, bool] = {
        "window.selected": True,
        "ascent.optimizer_converged": converged,
    }
    if v.physics_verified is not None:
        checks_map["ascent.physics_verified"] = bool(v.physics_verified)
    if v.mission_constraints_passed is not None:
        checks_map["ascent.mission_constraints_passed"] = bool(v.mission_constraints_passed)
    checks_map["injection.semi_major_axis_within_tolerance"] = injection["checks"]["semi_major_axis"]
    checks_map["injection.eccentricity_within_tolerance"] = injection["checks"]["eccentricity"]
    checks_map["injection.inclination_within_tolerance"] = injection["checks"]["inclination"]
    if injection["checks"]["raan"] is not None:
        checks_map["injection.raan_within_tolerance"] = bool(injection["checks"]["raan"])

    ascent_block = {
        "status": opt["status"],
        "optimizer_converged": converged,
        "reason_codes": list(opt["reason_codes"]),
        "payload_kg": opt["payload_kg"],
        "run_id": opt["run_id"],
        "raan": opt.get("raan"),
    }
    report = _build_report(
        inputs=_report_inputs(site=site, vehicle_path=vehicle_path,
                              orbit_spec=orbit_spec, date_utc=date_utc,
                              raan_deg=raan_deg, window_days=window_days,
                              opt_kwargs=opt_kwargs, target=target,
                              run_id=opt["run_id"],
                              target_sha256=opt["target_sha256"], out_dir=out),
        window=window, ascent=ascent_block, injection=injection,
        checks_map=checks_map, message=message, reason_codes=codes,
        status=status, label=target.mission.id)
    return _write(dict(status=status, message=message, reason_codes=tuple(codes),
                       window=window, target_path=target_path,
                       ascent_run_dir=ascent_dir, run_id=opt["run_id"],
                       optimization=opt_doc, injection=injection, out_dir=out),
                  report, out)


def _opt_kwargs(ns: Mapping[str, Any]) -> dict:
    return {k: ns.get(k) for k in ("objective", "payload_kg", "seed", "algorithm",
                                   "generations", "population", "workers",
                                   "local", "refly", "timeout_s")}


def _injection_message(injection: Mapping) -> str:
    err, tol, chk = injection["error"], injection["tolerance"], injection["checks"]
    parts = []
    if not chk["semi_major_axis"] and err["delta_semi_major_axis_m"] is not None:
        parts.append(f"semi-major axis off by {err['delta_semi_major_axis_m']:+.1f} m "
                     f"(tolerance {tol['semi_major_axis_m']:.1f} m)")
    if not chk["eccentricity"] and err["delta_eccentricity"] is not None:
        parts.append(f"eccentricity off by {err['delta_eccentricity']:+.2e} "
                     f"(tolerance {tol['eccentricity']:.2e})")
    if not chk["inclination"] and err["delta_inclination_deg"] is not None:
        parts.append(f"inclination off by {err['delta_inclination_deg']:+.3f} deg "
                     f"(tolerance {tol['inclination_deg']:.3f} deg)")
    if chk["raan"] is False and err["delta_raan_deg"] is not None:
        parts.append(f"RAAN off by {err['delta_raan_deg']:+.3f} deg "
                     f"(tolerance {tol['raan_deg']:.3f} deg)")
    if not parts:
        parts.append("insertion is not an orbit (hyperbolic or degenerate state)")
    return "injection out of tolerance: " + "; ".join(parts)
