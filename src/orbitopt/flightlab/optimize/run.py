"""FL-202: the ascent optimization solve loop (``orbitopt flight optimize``).

Global search with a pygmo single-objective algorithm (default ``cmaes``,
also ``pso_gen``/``sade``) over the FL-201 scalar penalty interface
(:meth:`AscentProblem.penalty_fitness`), evolved through a *process pool*
batch fitness evaluator: every fitness evaluation flies OpenTsiolkovsky in
its own process (and its own temp directory), a generation at a time, so a
laptop's cores are all burning flights instead of one queue. The penalty
weights are rescaled from the FL-201 defaults so that a *full-band*
constraint violation costs exactly one objective span -- see
:func:`_scaled_weights`.

Then a local refinement on the same UDP: pygmo's NLopt SLSQP is tried
first (with a finite-difference gradient, which NLopt demands and pygmo
does not synthesise); if pygmo lacks NLopt, the evolve raises, or the
result is not better, :func:`scipy.optimize.minimize` with ``method =
"SLSQP"`` runs on the same scalar penalty instead. Which one actually ran
is recorded in ``optimization.json`` -- never guessed.

The winner is re-flown once through :func:`orbitopt.flightlab.simulate`
(the full Gate-B pipeline: manifest, checks, optional tudatpy re-fly) and
written as a normal FlightRun next to ``optimization.json``.
``optimizer_converged`` is set from the raw FL-201 residuals -- feasible
inside the target's own tolerance bands -- and never from "the optimizer
terminated"; a run that cannot fly at all ends in ``INFEASIBLE_SIMULATION``
with no run written, never a fake success (FL-202 acceptance).

Determinism: a fixed ``seed`` fixes the CMA-ES/PSO/SADE seeds, the
initial sampling RNG and the process-pool chunking (results are collected
in row order), so the same inputs give a byte-identical
``optimization.json``. No wall-clock value is ever written into it.
"""
from __future__ import annotations

import dataclasses
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from orbitopt.flightlab import checks as _checks
from orbitopt.flightlab.contract import FlightTarget, target_digest, write_run
from orbitopt.flightlab.optimize import ascent_problem as _ap
from orbitopt.flightlab.guidance import ascent_param
from orbitopt.flightlab.optimize.ascent_problem import (
    PENALTY_OBJECTIVE,
    PENALTY_VIOLATION,
    AscentProblem,
)
from orbitopt.flightlab.simulate import simulate

__all__ = ["OptimizeAbort", "optimize_ascent"]

#: INFEASIBLE_* reason code per raw residual slot (1..6), see AscentProblem.
INFEASIBLE_CODES = {
    1: "INFEASIBLE_PERIAPSIS_OUT_OF_TOLERANCE",
    2: "INFEASIBLE_APOAPSIS_OUT_OF_TOLERANCE",
    3: "INFEASIBLE_INCLINATION_OUT_OF_TOLERANCE",
    4: "INFEASIBLE_MAX_Q_EXCEEDED",
    5: "INFEASIBLE_MAX_ACCEL_EXCEEDED",
    6: "INFEASIBLE_PROPELLANT_RESERVE",
}
INFEASIBLE_SIMULATION = "INFEASIBLE_SIMULATION"

ALGORITHMS = ("cmaes", "pso_gen", "sade")

# Process-pool worker state (module level so it survives pickling).
_WORKER: dict[str, Any] = {}


def _worker_init(udp: AscentProblem, weights: dict[str, float]) -> None:
    _WORKER["udp"], _WORKER["weights"] = udp, dict(weights)


def _worker_rows(rows: list[list[float]]) -> list[float]:
    udp, weights = _WORKER["udp"], _WORKER["weights"]
    return [float(udp.penalty_fitness(r, weights)) for r in rows]


class OptimizeAbort(RuntimeError):
    """The solve loop could not produce a flyable winner at all. Carries the
    reason codes so the CLI can report honestly without a run directory."""

    def __init__(self, message: str, reason_codes: Sequence[str] = ()):
        super().__init__(message)
        self.reason_codes = tuple(reason_codes)


def _scaled_weights(udp: AscentProblem) -> dict[str, float]:
    """FL-201 penalty weights rescaled so a full-band violation costs exactly
    one objective span: equality weights per tolerance unit, inequality
    weights per 100 % of the declared limit (FL-201's raw defaults are
    tuned for pygmo's feasibility rules, not for a penalised single-objective
    search, and would let CMA-ES buy payload with a 20 km orbit error)."""
    import yaml

    from orbitopt.flightlab import checks as _checks

    lo, hi = udp.get_bounds()
    c = udp.target.constraints
    alt_tol = float(c.orbit_altitude_tolerance_m
                    or _checks.DEFAULT_ORBIT_ALTITUDE_TOLERANCE_M)
    inc_tol = float(c.inclination_tolerance_deg
                    or _checks.DEFAULT_INCLINATION_TOLERANCE_DEG)
    if udp.objective == "payload":
        span = max(float(hi[-1] - lo[-1]), 1.0)
    else:
        propellant = max(
            (float(s.get("propellant_mass_kg") or 0.0)
             for s in yaml.safe_load(
                 udp.vehicle_path.read_text(encoding="utf-8"))["stages"]),
            default=0.0)
        span = max(propellant, 1000.0)
    floor = c.min_propellant_reserve_kg
    reserve_scale = max(float(floor) if floor is not None else 0.0, 1.0e3)
    return {
        "eq_rp": span / alt_tol,
        "eq_ra": span / alt_tol,
        "eq_inc": span / inc_tol,
        "ineq_maxq": (span / float(c.max_dynamic_pressure_pa)
                      if c.max_dynamic_pressure_pa else 1.0e-4),
        "ineq_maxg": (span / float(c.max_specific_acceleration_g)
                      if c.max_specific_acceleration_g else 10.0),
        "ineq_reserve": span / reserve_scale,
    }


class _PoolPenaltyUdp:
    """Scalar pygmo UDP wrapping an :class:`AscentProblem`: ``fitness`` is
    ``penalty_fitness`` with ``weights``; ``batch_fitness`` evaluates a whole
    generation in a ``ProcessPoolExecutor`` (one pool, lazily created,
    reused across generations; ``workers <= 1`` runs inline).

    Pygmo deep-copies UDPs; the deep copy shares the parent's history log
    and flight counter (same idiom as ``AscentProblem.__deepcopy__``) and
    never owns a pool, so all parallelism stays on the parent thread.
    History: one snapshot per ``batch_fitness`` call -- one call per
    generation for the bfe-wired algorithms -- plus the seeding batch."""

    def __init__(self, udp: AscentProblem, weights: Mapping[str, float],
                 workers: int):
        self._udp = udp
        self._weights = dict(weights)
        self._workers = max(1, int(workers))
        self._pool: ProcessPoolExecutor | None = None
        self._log: list[dict[str, Any]] = []   # shared across deep copies
        self._flights = [0]                    # ditto (cumulative dispatches)

    # --- pygmo UDP protocol (explicit both 2.19- and 2.20-style) ----------
    def get_name(self) -> str:
        return f"orbitopt FL-202 penalty ascent ({self._udp.objective})"

    def get_bounds(self) -> tuple[list[float], list[float]]:
        return self._udp.get_bounds()

    def get_nobj(self) -> int:
        return 1

    def get_nec(self) -> int:
        return 0

    def get_nic(self) -> int:
        return 0

    def has_constraints(self) -> bool:
        return False

    def fitness(self, x: Sequence[float]) -> list[float]:
        self._flights[0] += 1
        return [float(self._udp.penalty_fitness(x, self._weights))]

    def batch_fitness(self, X: Any) -> list[float]:
        flat = np.asarray(X, dtype=float).reshape(-1)
        nx = len(self._udp.get_bounds()[0])
        rows = [list(map(float, r)) for r in flat.reshape(-1, nx)]
        out = self._dispatch(rows)
        self._log.append({
            "generation": len(self._log),
            "batch_size": len(rows),
            "best_penalty": min(out),
            "flights_total": self._flights[0],
        })
        return out

    def penalty_batch(self, rows: Sequence[Sequence[float]]) -> list[float]:
        """Explicit seeding batch (also logged as one generation)."""
        flat = [list(map(float, r)) for r in rows]
        out = self._dispatch(flat)
        self._log.append({
            "generation": len(self._log),
            "batch_size": len(flat),
            "best_penalty": min(out),
            "flights_total": self._flights[0],
        })
        return out

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    # --- internals ----------------------------------------------------------
    def _dispatch(self, rows: list[list[float]]) -> list[float]:
        n = len(rows)
        if self._workers <= 1 or n <= 1:
            self._flights[0] += n
            udp, w = self._udp, self._weights
            return [float(udp.penalty_fitness(r, w)) for r in rows]
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self._workers,
                mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init,
                initargs=(self._udp, self._weights))
        size = max(1, -(-n // (2 * self._workers)))
        out: list[float] = []
        for start in range(0, n, size):
            out.extend(self._pool.submit(
                _worker_rows, rows[start:start + size]).result())
        self._flights[0] += n
        return out

    def __deepcopy__(self, memo: dict) -> "_PoolPenaltyUdp":
        clone = _PoolPenaltyUdp.__new__(_PoolPenaltyUdp)
        clone.__dict__.update(self.__dict__)
        clone._pool = None          # pools belong to the original only
        return clone


class _NloptUdp:
    """Same scalar penalty for pygmo's NLopt UDAs, plus the finite-difference
    objective gradient NLopt requests and pygmo will not synthesise."""

    def __init__(self, udp: AscentProblem, weights: Mapping[str, float]):
        self._udp, self._weights = udp, dict(weights)

    def get_name(self) -> str:
        return "orbitopt FL-202 ascent (nlopt, fd gradient)"

    def get_bounds(self) -> tuple[list[float], list[float]]:
        return self._udp.get_bounds()

    def get_nobj(self) -> int:
        return 1

    def get_nec(self) -> int:
        return 0

    def get_nic(self) -> int:
        return 0

    def has_constraints(self) -> bool:
        return False

    def fitness(self, x: Sequence[float]) -> list[float]:
        return [float(self._udp.penalty_fitness(x, self._weights))]

    def get_gradient(self, x: Sequence[float]) -> list[float]:
        f0 = float(self._udp.penalty_fitness(x, self._weights))
        lo, hi = self._udp.get_bounds()
        grad = []
        for i in range(len(x)):
            h = 1.0e-5 * max(hi[i] - lo[i], 1.0)
            xp = list(map(float, x))
            if xp[i] + h <= hi[i]:
                xp[i] += h
                grad.append((float(self._udp.penalty_fitness(
                    xp, self._weights)) - f0) / h)
            else:                                   # at the upper bound
                xp[i] = hi[i]
                back = float(x[i]) - hi[i]
                grad.append((float(self._udp.penalty_fitness(
                    xp, self._weights)) - f0) / (back or h))
        return grad


def _make_uda(pg: Any, algorithm: str, generations: int, seed: int,
              population: int) -> Any:
    """Raw UDA with the batch fitness evaluator already attached (pygmo
    quirk: ``set_bfe`` exists only on the raw UDA and is silently ignored
    after ``pg.algorithm`` wrapping -- see docs/engineering-notes.md).

    Constructors are positional on purpose: keyword spellings differ across
    pygmo 2.19/2.20 while the full positional overloads (the 2.19-era
    extended ctors) exist in both."""
    if algorithm == "cmaes":
        # cmaes(gen, cc, cs, c1, cmu, sigma0, ftol, xtol, memory,
        #       force_bounds, seed[, population_size]) -- pygmo 2.19 has
        #       the trailing population size, pygmo 2.20 removed it (CMA-ES
        #       then uses its own lambda; our seeding count still applies)
        try:
            uda = pg.cmaes(generations, -1.0, -1.0, -1.0, -1.0, 0.5, 1.0e-6,
                           1.0e-6, False, True, seed, population)
        except TypeError:
            uda = pg.cmaes(generations, -1.0, -1.0, -1.0, -1.0, 0.5, 1.0e-6,
                           1.0e-6, False, True, seed)
    elif algorithm == "pso_gen":
        # pso_gen(gen, seed, swarm_size, omega, ...)
        uda = pg.pso_gen(generations, seed, population)
    elif algorithm == "sade":
        uda = pg.sade(generations, seed)
    else:
        raise ValueError(f"unknown algorithm {algorithm!r}; "
                         f"expected one of {ALGORITHMS}")
    if hasattr(uda, "set_bfe"):
        uda.set_bfe(pg.bfe())
    return uda


def _residual_violations(f: Sequence[float],
                         bands: Sequence[tuple]) -> list[float]:
    """Per-slot violation magnitude (0 when inside the band) for the six
    FL-201 residual slots."""
    out = []
    for i, (eq_lo, eq_hi, ineq_lo, ineq_hi) in enumerate(bands, start=1):
        v = f[i]
        eq = max(0.0, eq_lo - v, v - eq_hi) if eq_hi != float("-inf") else 0.0
        ineq = (max(0.0, v - ineq_hi) if ineq_hi != float("inf") else 0.0) \
            + (max(0.0, ineq_lo - v) if ineq_lo != float("-inf") else 0.0)
        out.append(eq + ineq)
    return out


def _is_penalty_vector(f: Sequence[float]) -> bool:
    return (f[0] == PENALTY_OBJECTIVE
            and all(v == PENALTY_VIOLATION for v in f[1:]))


def _seeded_population(pg: Any, prob: Any, udp: AscentProblem,
                       wrapper: _PoolPenaltyUdp, population: int,
                       seed: int) -> Any:
    """Uniform seeding through the wrapper's own batch (deterministic across
    pygmo versions -- population(seed=...) sampling internals are not) and
    a population built by ``push_back``, so pygmo never re-evaluates it."""
    lo, hi = udp.get_bounds()
    rng = np.random.default_rng([seed, 0x200])
    x = np.asarray(lo) + (np.asarray(hi) - np.asarray(lo)) * rng.random(
        (population, len(lo)))
    f = wrapper.penalty_batch(x)
    pop = pg.population(prob, 0)
    for xi, fi in zip(x, f):
        pop.push_back(list(map(float, xi)), [float(fi)])
    return pop


def _local_refine(pg: Any, udp: AscentProblem, weights: Mapping[str, float],
                  x0: Sequence[float], p0: float) -> tuple[list[float], float, str]:
    """Return ``(x, penalty, engine)``; never raises. NLopt SLSQP first (it
    needs a gradient -- supplied by finite differences), scipy's SLSQP on
    the same scalar penalty otherwise (task: record, never guess, which ran)."""
    lo, hi = udp.get_bounds()
    try:
        prob = pg.problem(_NloptUdp(udp, weights))
        pop = pg.population(prob, 0)
        pop.push_back(list(map(float, x0)))
        nlopt_uda = pg.nlopt("slsqp")
        nlopt_uda.maxeval = 60                  # default 1000 = ~25 min of flights
        algo = pg.algorithm(nlopt_uda)
        pop = algo.evolve(pop)
        x = [float(v) for v in pop.get_x()[0]]
        f = float(udp.penalty_fitness(x, weights))
        if f <= p0:
            return x, f, "pygmo-nlopt-slsqp"
        return list(map(float, x0)), p0, "pygmo-nlopt-slsqp (not improved)"
    except Exception:
        pass
    try:
        from scipy.optimize import minimize
        res = minimize(lambda v: udp.penalty_fitness(v, weights),
                       list(map(float, x0)), method="SLSQP",
                       bounds=list(zip(lo, hi)),
                       options={"maxiter": 25, "ftol": 1.0e-8})
        x = [float(min(max(v, a), b))
             for v, a, b in zip(res.x, lo, hi)]
        f = float(udp.penalty_fitness(x, weights))
        if f <= p0:
            return x, f, "scipy-slsqp"
        return list(map(float, x0)), p0, "scipy-slsqp (not improved)"
    except Exception as exc:
        return list(map(float, x0)), p0, f"skipped ({type(exc).__name__})"


def optimize_ascent(target: FlightTarget, vehicle_path: str | Path,
                   out_dir: str | Path, *, target_path: str | Path | None = None,
                   objective: str = "payload", payload_kg: float = 0.0,
                   seed: int = 42, algorithm: str = "cmaes",
                   generations: int = 60, population: int = 48,
                   workers: int | None = None, local: bool = True,
                   refly: bool = True, timeout_s: float = 300.0) -> dict:
    """Solve, refine, re-fly and write. Returns the parsed ``optimization.json``
    plus the ``run`` object. Raises :class:`OptimizeAbort` (never writes a
    run) when no candidate ever produced a valid flight, and whatever
    ``simulate`` raises for pipeline trouble. ``target_path`` is only used to
    stamp provenance into ``optimization.json``."""
    pg = _import_pygmo()
    out = Path(out_dir)
    udp = AscentProblem(target, Path(vehicle_path), objective,
                        payload_kg=payload_kg, timeout_s=timeout_s)
    weights = _scaled_weights(udp)
    if workers is None:
        workers = max(1, min(8, os.cpu_count() or 1))
    wrapper = _PoolPenaltyUdp(udp, weights, workers)
    try:
        try:
            uda = _make_uda(pg, algorithm, generations, seed, population)
        except TypeError as exc:
            raise OptimizeAbort(
                f"the installed pygmo does not accept the constructor "
                f"arguments for algorithm {algorithm!r} ({exc}); try pygmo "
                f"2.19.7/2.20 (conda-forge)") from exc
        prob = pg.problem(wrapper)
        pop = _seeded_population(pg, prob, udp, wrapper, max(population, 6), seed)
        pop = pg.algorithm(uda).evolve(pop)

        X, F = list(pop.get_x()), list(pop.get_f())
        if not X:
            raise OptimizeAbort("the global search produced no candidates")
        i_best = min(range(len(X)), key=lambda i: float(F[i][0]))
        x_best = [float(v) for v in X[i_best]]
        p_best = float(F[i_best][0])

        engine = "skipped (--no-local)"
        if local:
            x_best, p_best, engine = _local_refine(pg, udp, weights, x_best, p_best)

        f_raw = udp.fitness(x_best)
        if _is_penalty_vector(f_raw):
            raise OptimizeAbort(
                "no candidate ever produced a valid flight -- the best "
                "penalty candidate is a simulated failure",
                [INFEASIBLE_SIMULATION])

        bands = udp.get_bounds_eq_ineq()
        violations = _residual_violations(f_raw, bands)
        residual_ok = all(v == 0.0 for v in violations)

        prog, _doc = udp.decode(x_best)
        derived = udp.write_vehicle(x_best, out / "vehicle")
        run = simulate(target, derived, out, attitude_program=prog,
                       refly=refly, timeout_s=timeout_s)
        verdict = run.verification

        by_id = {c.id: c for c in verdict.checks}
        flagged = any(not by_id[cid].passed
                      for cid in _ap._EQ_CHECKS + _ap._INEQ_CHECKS
                      if cid in by_id)
        converged = bool(residual_ok and not flagged)

        reason_codes = [INFEASIBLE_CODES[i] for i, v in enumerate(violations, 1)
                        if v > 0.0]
        final_failures = list(_checks.failed_reason_codes(verdict))
        assumptions = tuple(run.assumptions) + (
            f"FL-202 ascent optimization: global={algorithm} seed={seed} "
            f"generations={generations} population={population}; "
            f"local={engine}",)
        run = dataclasses.replace(
            run,
            assumptions=assumptions,
            verification=dataclasses.replace(
                verdict, optimizer_converged=converged))
        write_run(out, run)

        write_attitude_program(prog, out / "attitude_program.csv")
        doc = {
            "status": "CONVERGED" if converged else "INFEASIBLE",
            "optimizer_converged": converged,
            "reason_codes": ([] if converged else reason_codes + [
                c for c in final_failures if c not in reason_codes]),
            "objective": objective,
            "payload_kg": float(payload_kg if objective != "payload"
                                else -f_raw[0]),
            "algorithm": {"global": algorithm,
                          "local": engine},
            "seed": int(seed),
            "generations": int(generations),
            "population": int(population),
            "workers": int(workers),
            "evaluations": int(wrapper._flights[0]),
            "penalty_weights": {k: float(v) for k, v in sorted(weights.items())},
            "decision_vector": {k: float(v) for k, v in
                                sorted(dataclasses.asdict(
                                    ascent_param.unpack(x_best, objective)).items())},
            "best_penalty": p_best,
            "best_fitness": {name: float(v) for name, v in zip(
                ("objective", "eq_periapsis_m", "eq_apoapsis_m",
                 "eq_inclination_deg", "ineq_max_q_pa", "ineq_max_accel_g",
                 "ineq_propellant_reserve_kg"),
                [f_raw[0]] + f_raw[1:6] + [f_raw[6]])},
            "residual_violations": {
                name: float(v) for name, v in zip(
                    ("periapsis_m", "apoapsis_m", "inclination_deg",
                     "max_q_pa", "max_accel_g", "propellant_reserve_kg"),
                    violations)},
            "convergence_history": wrapper._log,
            "target_sha256": target_digest(target.to_dict()),
            "target_file": str(target_path) if target_path else None,
            "vehicle_file": str(vehicle_path),
            "run_id": run.run_id,
            "run_dir": Path(out).resolve().name,
            "attitude_program_file": "attitude_program.csv",
        }
        (out / "optimization.json").write_text(
            json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        doc["run"] = run
        return doc
    finally:
        wrapper.shutdown()


def write_attitude_program(prog: Sequence[Any], path: Path) -> None:
    """The derived winner's attitude program, kept next to the run (CSV, so
    the FL-201 round-trip stays inspectable)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["t_s,azimuth_deg,elevation_deg"]
    lines += [f"{p.t_s},{p.azimuth_deg},{p.elevation_deg}" for p in prog]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _import_pygmo() -> Any:
    try:
        import pygmo
    except ImportError as exc:
        raise RuntimeError(
            "pygmo is required for ascent optimization but is not installed. "
            "pygmo ships on conda-forge only (no PyPI wheel): create the "
            "project environment with `mamba env create -f envs/cpu.yml` "
            "(or `mamba install -c conda-forge pygmo`). See envs/README.md."
        ) from exc
    return pygmo
