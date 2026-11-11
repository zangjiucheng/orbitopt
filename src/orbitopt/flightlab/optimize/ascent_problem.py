"""pygmo ascent optimization problem for Flight Lab E2 (FL-201).

:class:`AscentProblem` is a pygmo *user-defined problem* (UDP) around the
frozen E1 pipeline: every ``fitness(x)`` decodes the parametric guidance
vector (orbitopt.flightlab.guidance.ascent_param) into an attitude program
plus a derived vehicle file, runs the OpenTsiolkovsky adapter
(:func:`orbitopt.flightlab.adapters.opentsiolkovsky.run_ascent`) in its own
temp directory, and reads the constraint residuals straight out of the Gate-B
verdict (:func:`orbitopt.flightlab.checks.evaluate` -- the single source of
truth: what the optimizer optimizes is exactly what the CLI later grades).

Fitness vector layout (7 numbers, the pygmo constrained-UDP protocol -- see
:meth:`AscentProblem.has_constraints`):

    [ -objective,
      eq_rp,  eq_ra,  eq_inc,                 # actual - target (m, m, deg)
      ineq_maxq,  ineq_maxg,  ineq_reserve ]   # positive = violated

The three eq residuals carry their sign (actual minus target; the target's
tolerance bands come back from :meth:`AscentProblem.get_bounds_eq_ineq`, so a
pygmo-feasible candidate is one ``evaluate()`` would not flag); the three ineq
residuals are signed so ``<= 0`` is satisfied: a max-Q / max-g excess above
the limit, or a propellant-reserve shortfall below it. A constraint the
target does not declare is a flat ``0.0`` (unconstrained); a declared
constraint whose value is missing in the run keeps evaluate's honest
data-missing red and maps to a huge violation.

Objective (maximized, returned negated):

* ``"payload"``        -- payload mass [kg]; the decision vector's 8th
  component, appearing in the derived vehicle as last-stage mass.
* ``"reserve"``        -- upper-stage propellant reserve [kg] at SECO, the
  value of the ``propellant-reserve`` check; payload stays fixed at the
  constructor's ``payload_kg``.
* ``"booster_reserve"`` -- stage-1 reserve [kg] at MECO (the E3 reuse axis;
  implemented here so E3 only adds recovery, not a new problem).

A candidate whose simulation fails (binary error, OT refusing the config,
no insertion state) is *not* an exception -- pygmo must keep searching
through infeasible regions; it returns the penalty vector instead.

Results are cached per problem instance by
``sha256(x, target digest, vehicle-file digest, objective, payload)`` so a
re-evaluation of a sampled point (DE re-judging elites, restarts, a manual
replay) costs nothing; ``__deepcopy__`` deliberately shares the cache between
pygmo's copies of the UDP (same pattern as orbitopt.core.problem.BackendUsage).

pygmo itself is never imported here -- a UDP is duck-typed, so this module
(and its tests) work on a base install; wrap the instance in
``pg.problem(udp)`` from the optimizer script:

    import pygmo as pg
    from orbitopt.flightlab.contract import load_target
    from orbitopt.flightlab.optimize.ascent_problem import AscentProblem
    udp = AscentProblem(load_target(SCENARIO_PATH), VEHICLE_PATH, "payload")
    algo = pg.de122(gens=200, ftol=1e-6)          # de122 honours the eq/ineq bands
    pop = pg.population(pg.problem(udp), size=40, seed=7)
    pop = algo.evolve(pop)
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from orbitopt.flightlab import checks as _checks
from orbitopt.flightlab.adapters.opentsiolkovsky import AttitudePoint, run_ascent
from orbitopt.flightlab.contract import FlightTarget, target_digest
from orbitopt.flightlab.guidance import ascent_param
from orbitopt.flightlab.guidance.ascent_param import GuidanceError

__all__ = ["AscentProblem"]

#: Fitness slots when the simulation produces no verdict at all (objective
#: penalty -- positive, since fitness[0] is the *negated* objective -- and a
#: huge violation per constraint slot).
PENALTY_OBJECTIVE = 1.0e6
PENALTY_VIOLATION = 1.0e4

_EQ_CHECKS = ("insertion-periapsis-within-tolerance",
              "insertion-apoapsis-within-tolerance",
              "insertion-inclination-within-tolerance")
_INEQ_CHECKS = ("max-q-within-limit", "max-accel-within-limit", "propellant-reserve")

_DEFAULT_WEIGHTS = {
    "eq_rp": 1.0e-4, "eq_ra": 1.0e-4, "eq_inc": 10.0,
    "ineq_maxq": 1.0e-4, "ineq_maxg": 10.0, "ineq_reserve": 1.0e-3,
}


class AscentProblem:
    """pygmo UDP: parametric ascent of ``vehicle_path`` toward ``target``.

    ``objective``: ``"payload"`` | ``"reserve"`` | ``"booster_reserve"`` (see
    the module docstring). ``payload_kg`` is a fixed payload for the reserve
    objectives (CLI ``--payload-kg``, default 0) and ignored for
    ``"payload"`` where payload is a decision variable. ``timeout_s`` bounds
    each OpenTsiolkovsky subprocess; ``cache_size`` caps the result cache."""

    def __init__(self, target: FlightTarget, vehicle_path: Path,
                 objective: str, *, payload_kg: float = 0.0,
                 timeout_s: float = 300.0, cache_size: int = 4096):
        if objective not in ascent_param.OBJECTIVES:
            raise ValueError(f"unknown objective {objective!r}; expected one of "
                             f"{ascent_param.OBJECTIVES}")
        self.target = target
        self.vehicle_path = Path(vehicle_path)
        self.objective = objective
        self.payload_kg = float(payload_kg)
        self.timeout_s = float(timeout_s)
        self.cache_size = int(cache_size)
        self._vehicle_doc: dict = yaml.safe_load(
            self.vehicle_path.read_text(encoding="utf-8"))
        if not isinstance(self._vehicle_doc, dict):
            raise ValueError(f"{self.vehicle_path}: vehicle file must be a YAML mapping")
        self._vehicle_sha = hashlib.sha256(self.vehicle_path.read_bytes()).hexdigest()
        self._target_sha = target_digest(target.to_dict())
        self._lo, self._hi = ascent_param.bounds(self._vehicle_doc, target, objective)
        self._cache: dict[str, tuple[float, ...]] = {}

    # --- pygmo UDP protocol -------------------------------------------------
    def get_name(self) -> str:
        return f"orbitopt FL-201 ascent ({self.objective})"

    def get_bounds(self) -> tuple[list[float], list[float]]:
        """(lo, hi) decision bounds, derived from the vehicle document and the
        target (orbitopt.flightlab.guidance.ascent_param.bounds)."""
        return list(self._lo), list(self._hi)

    def bounds(self) -> tuple[list[float], list[float]]:
        """Alias for :meth:`get_bounds` (the FL-201 interface name)."""
        return self.get_bounds()

    def has_constraints(self) -> bool:
        return True

    def get_bounds_eq_ineq(self) -> list[tuple[float, float, float, float]]:
        """Per-constraint (eq_lo, eq_hi, ineq_lo, ineq_hi) bands for the six
        residual slots: the eq bands are the target's own insertion
        tolerances (the Gate-B defaults where the target declares none), the
        ineq bands ``<= 0`` -- so a pygmo-feasible point is exactly a point
        ``evaluate()`` would not flag, and an undeclared constraint is
        unconstrained on both sides of the sign convention used by
        :meth:`fitness`."""
        alt_tol = self.target.constraints.orbit_altitude_tolerance_m
        alt_tol = (_checks.DEFAULT_ORBIT_ALTITUDE_TOLERANCE_M if alt_tol is None
                   else float(alt_tol))
        inc_tol = self.target.constraints.inclination_tolerance_deg
        inc_tol = (_checks.DEFAULT_INCLINATION_TOLERANCE_DEG if inc_tol is None
                   else float(inc_tol))
        inf = float("inf")
        eq = [(-alt_tol, alt_tol, -inf, inf), (-alt_tol, alt_tol, -inf, inf),
              (-inc_tol, inc_tol, -inf, inf)]
        return eq + [(-inf, inf, -inf, 0.0)] * 3

    def fitness(self, x: Sequence[float]) -> list[float]:
        """Fly candidate ``x``, return the 7-slot fitness vector documented at
        module level. Cached; never raises for an infeasible or failed flight
        (the penalty vector is returned instead)."""
        key = self._cache_key(x)
        hit = self._cache.get(key)
        if hit is not None:
            return list(hit)
        vec = self._run_candidate(x)
        if len(self._cache) >= self.cache_size:
            self._cache.clear()
        self._cache[key] = tuple(vec)
        return list(vec)

    # --- FL-201 interface ---------------------------------------------------
    def decode(self, x: Sequence[float]) -> tuple[list[AttitudePoint], dict]:
        """``(attitude program, derived vehicle document)`` for candidate
        ``x`` (the fixed payload is applied for the reserve objectives)."""
        return ascent_param.decode(x, self.target, self._vehicle_doc,
                                   objective=self.objective,
                                   payload_kg=self._fixed_payload())

    def write_vehicle(self, x: Sequence[float], dir: Path) -> Path:
        """Write the derived vehicle YAML for candidate ``x`` into ``dir``
        (created when missing) and return its path. The base vehicle file is
        never modified; the derived document passes the adapter's OT schema
        validation (``Vehicle.load`` / ``build_config``) like any vehicle file."""
        _, doc = self.decode(x)
        out = Path(dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "vehicle.yaml"
        path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        return path

    def penalty_fitness(self, x: Sequence[float],
                        weights: Mapping[str, float] | None = None) -> float:
        """Scalar form of the fitness vector, for single-objective algorithms:
        ``-objective + sum(w_i * violation_i)``, eq residuals entering by
        absolute value and ineq residuals only when violated."""
        f = self.fitness(x)
        w = dict(_DEFAULT_WEIGHTS, **(weights or {}))
        return (f[0] + w["eq_rp"] * abs(f[1]) + w["eq_ra"] * abs(f[2])
                + w["eq_inc"] * abs(f[3]) + w["ineq_maxq"] * max(0.0, f[4])
                + w["ineq_maxg"] * max(0.0, f[5]) + w["ineq_reserve"] * max(0.0, f[6]))

    # --- plumbing ------------------------------------------------------------
    def _fixed_payload(self) -> float | None:
        return None if self.objective == "payload" else self.payload_kg

    def _cache_key(self, x: Sequence[float]) -> str:
        payload = self.payload_kg if self.objective != "payload" else 0.0
        material = json.dumps([self._target_sha, self._vehicle_sha, self.objective,
                               payload, [float(v) for v in x]],
                              sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def __deepcopy__(self, memo: dict) -> "AscentProblem":
        """Share the result cache between pygmo's deep copies of this UDP (a
        problem copy inside an algorithm must still hit the parent's cache);
        everything else is immutable configuration."""
        clone = type(self).__new__(type(self))
        clone.__dict__.update(self.__dict__)  # shares _cache by identity, on purpose
        return clone

    def _grading_target(self) -> FlightTarget:
        """Target used for Gate-B grading inside ``fitness``. The reserve
        residuals -- and the ``reserve`` objective value -- must exist as
        measured checks even when the mission file declares no floor, so a
        missing floor is graded against 0 kg; the check itself, its value and
        the reason machinery stay evaluate()'s, not ours."""
        target = self.target
        reserve = self.target.constraints.min_propellant_reserve_kg
        if reserve is None:
            target = replace(target, constraints=replace(
                target.constraints, min_propellant_reserve_kg=0.0))
        if (self.objective == "booster_reserve"
                and target.recovery.min_propellant_reserve_kg is None):
            target = replace(target, recovery=replace(
                target.recovery, kind="rtls", min_propellant_reserve_kg=0.0))
        return target

    def _run_candidate(self, x: Sequence[float]) -> list[float]:
        try:
            params = ascent_param.unpack(x, self.objective)
        except (GuidanceError, ValueError):
            return self._penalty()
        try:
            program = ascent_param.attitude_program(params, self._vehicle_doc)
            doc = ascent_param.derive_vehicle(
                self._vehicle_doc, ascent_param.as_vector(params),
                payload_kg=self._fixed_payload())
        except (GuidanceError, ValueError):
            return self._penalty()
        tmp = tempfile.mkdtemp(prefix="fl201-")
        try:
            vehicle = Path(tmp) / "vehicle.yaml"
            vehicle.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
            run = run_ascent(self.target, vehicle, program, Path(tmp) / "run",
                             timeout_s=self.timeout_s)
            grading = self._grading_target()
            extra = _checks.propellant_reserve_check(run, doc, grading)
            checks = tuple(run.verification.checks) + ((extra,) if extra else ())
            verdict = _checks.evaluate(
                replace(run, verification=replace(run.verification, checks=checks)),
                grading)
            return self._vector_from(verdict, run, doc, params.payload_kg)
        except Exception:
            return self._penalty()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _penalty(self) -> list[float]:
        return [PENALTY_OBJECTIVE] + [PENALTY_VIOLATION] * 6

    def _vector_from(self, verdict: Any, run: Any, doc: Mapping[str, Any],
                     payload_kg: float) -> list[float]:
        by_id = {c.id: c for c in verdict.checks}
        objective = self._objective_value(run, doc, by_id, payload_kg)
        if objective is None:
            return self._penalty()
        vec = [-objective]
        vec += [self._residual(by_id, cid, signed=True) for cid in _EQ_CHECKS]
        vec += [self._residual(by_id, cid, signed=False) for cid in _INEQ_CHECKS]
        return [PENALTY_OBJECTIVE if i == 0 and not _finite(v)
                else (PENALTY_VIOLATION if not _finite(v) else v)
                for i, v in enumerate(vec)]

    def _objective_value(self, run: Any, doc: Mapping[str, Any],
                         by_id: Mapping[str, Any], payload_kg: float) -> float | None:
        if self.objective == "payload":
            return float(payload_kg)
        if self.objective == "reserve":
            check = by_id.get("propellant-reserve")
            if check is None or check.value is None:
                return None
            return float(check.value)
        dry = (doc.get("stages") or [{}])[0].get("mass_dry_kg")
        track = _checks._released_first_stage_track(run)
        if dry is None or track is None or not track.mass_kg:
            return None
        return float(track.mass_kg[0]) - float(dry)

    def _residual(self, by_id: Mapping[str, Any], check_id: str, *, signed: bool) -> float:
        """``value - limit`` for an eq check; for an ineq check the violation
        in the ``<= 0 satisfied`` convention (reserve flips: shortfall =
        ``limit - value``). An undeclared constraint has no check -> 0.0
        (unconstrained); a declared-but-data-missing check (value None) is a
        huge violation -- evaluate's red is not silently optimized around."""
        check = by_id.get(check_id)
        if check is None:
            return 0.0
        if check.value is None:
            return PENALTY_VIOLATION
        delta = float(check.value) - float(check.limit)
        if signed:
            return delta
        return -delta if check_id == "propellant-reserve" else delta


def _finite(v: float) -> bool:
    return v == v and v not in (float("inf"), float("-inf"))
