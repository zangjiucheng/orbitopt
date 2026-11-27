"""Sampling check of whether a compiled IR problem means the same as a reference IR problem.

``semantically_equivalent(compiled, reference)`` probes both problems at the
same seeded points and reports every disagreement it finds:

* variables: the same names (renames only through an explicit ``aliases``
  table, never a guessed permutation), kinds, unit dimensions and finite
  bounds;
* objectives: the same count and senses, and values that agree (in SI) at
  every probe point inside both domains;
* constraints: the same feasibility verdict (domain and every constraint, by
  ``OptimizationProblem.constraints_satisfied``) at every probe point, and a
  residual-sign match per constraint: every reference constraint and finite
  bound that cuts the probe box has a compiled constraint or bound that is
  satisfied at exactly the same points, and the reverse;
* tool calls: the same tool id, the same arguments (numerically, at the probe
  points), sense, rhs and unit. A tool with no implementation in ``tools`` is
  replaced by the same deterministic opaque function on both sides.

Probe points are drawn from the union of both boxes, so a widened or narrowed
finite bound shows up as points one problem accepts and the other rejects. A
side that is open in one problem and finite in the other is compared only up
to the finite side, and reported in ``notes``, not as a difference: this is
the benchmark rule that a compiled problem is judged inside the reference
box. Budgets and verification policies are not compared.

It is a sampling check, not a proof. Two problems that differ only on a set
the probes miss (a thin sliver, a single point) are reported equivalent, and
an equality split into two inequalities is reported different.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from orbitopt.optimization import EvaluationError, Expression, OptimizationProblem, ToolCall, Variable, parse_unit

SPAN = 10.0
"""Width given to a side that is open in both problems."""
MAX_CORNERS = 64
"""Box corners probed: all of them up to this many, a seeded sample beyond."""


@dataclass(frozen=True)
class EquivalenceReport:
    """``equivalent`` is true exactly when ``reasons`` is empty; ``notes`` never make it false."""

    equivalent: bool
    reasons: tuple[str, ...]
    notes: tuple[str, ...]
    points: int


def opaque_tools(*problems: OptimizationProblem) -> dict[str, Callable[..., float]]:
    """A deterministic stand-in in [-1, 1] for every tool id the problems reference."""

    def opaque(tool_id):
        def call(**kwargs):
            key = tool_id + "|" + "|".join(f"{k}={float(v):.12g}" for k, v in sorted(kwargs.items()))
            return int(hashlib.sha256(key.encode()).hexdigest()[:12], 16) / 16 ** 12 * 2.0 - 1.0
        return call

    ids = {item.expression.tool_id for p in problems for item in (*p.objectives, *p.constraints)
           if isinstance(item.expression, ToolCall)}
    return {tool_id: opaque(tool_id) for tool_id in sorted(ids)}


def _domain(v: Variable, scale: float = 1.0) -> tuple[str, float | None, float | None, tuple[float, ...]]:
    """(kind, lower, upper, choices) with binary as integer [0, 1], values multiplied by ``scale``."""
    if v.kind == "binary":
        return "integer", 0.0, 1.0, ()
    lo = None if v.lower is None else v.lower * scale
    hi = None if v.upper is None else v.upper * scale
    return v.kind, lo, hi, tuple(c * scale for c in v.choices)


def _box(domains: Sequence[tuple]) -> tuple[float, float]:
    lows = [d[1] for d in domains if d[1] is not None] + [c for d in domains for c in d[3]]
    highs = [d[2] for d in domains if d[2] is not None] + [c for d in domains for c in d[3]]
    if lows and highs:
        return min(lows), max(highs)
    if lows:
        return min(lows), min(lows) + max(SPAN, abs(min(lows)))
    if highs:
        return max(highs) - max(SPAN, abs(max(highs))), max(highs)
    return -SPAN, SPAN


def _draw(rng: random.Random, kind: str, lo: float, hi: float, choices: Sequence[float]) -> float:
    if choices:
        return rng.choice(list(choices))
    if kind == "integer":
        return float(rng.randint(math.ceil(lo), math.floor(hi)))
    return rng.uniform(lo, hi)


def _points(names: Sequence[str], kinds: Mapping[str, str], boxes: Mapping[str, tuple[float, float]],
            choices: Mapping[str, Sequence[float]], samples: int, seed: int,
            extra: Sequence[Mapping[str, float]] = ()) -> list[dict[str, float]]:
    rng = random.Random(seed)
    points = [{n: _draw(rng, kinds[n], *boxes[n], choices[n]) for n in names} for _ in range(samples)]
    sides = [[float(math.ceil(boxes[n][0])), float(math.floor(boxes[n][1]))] if kinds[n] == "integer"
             else list(boxes[n]) for n in names if not choices[n]]
    corner_names = [n for n in names if not choices[n]]
    if 2 ** len(corner_names) <= MAX_CORNERS:
        corners = list(itertools.product(*sides))
    else:
        corners = [tuple(rng.choice(side) for side in sides) for _ in range(MAX_CORNERS)]
    for corner in corners:
        point = {n: _draw(rng, kinds[n], *boxes[n], choices[n]) for n in names}
        point.update(zip(corner_names, corner))
        points.append(point)
    points.extend(dict(p) for p in extra)
    return points


def sample_points(problem: OptimizationProblem, *, samples: int = 64, seed: int = 0) -> list[dict[str, float]]:
    """Seeded points in the problem's variable box (open sides get ``SPAN``), corners included."""
    domains = {v.name: _domain(v) for v in problem.variables}
    names = [v.name for v in problem.variables]
    return _points(names, {n: d[0] for n, d in domains.items()}, {n: _box([d]) for n, d in domains.items()},
                   {n: d[3] for n, d in domains.items()}, samples, seed)


def _in_domain(domain: tuple, value: float, tol: float) -> bool:
    kind, lo, hi, choices = domain
    if choices:
        return any(abs(value - c) <= tol * max(1.0, abs(c)) for c in choices)
    if lo is not None and value < lo - tol * max(1.0, abs(lo)):
        return False
    if hi is not None and value > hi + tol * max(1.0, abs(hi)):
        return False
    return kind != "integer" or abs(value - round(value)) <= tol


def _si(item) -> float:
    return parse_unit(item.unit).scale


def _agree(x: float, y: float, rtol: float) -> bool:
    return abs(x - y) <= rtol * max(1.0, abs(y))


def conflicting_objectives(problem: OptimizationProblem, *, rtol: float = 1e-9, samples: int = 64,
                           seed: int = 0) -> list[tuple[str, str]]:
    """Pairs of objectives that pull against each other on the same quantity.

    A pair conflicts when its values agree at every sample point and the
    senses differ (maximise and minimise the same thing), or when the values
    are exact negatives and the senses agree.
    """
    tools = opaque_tools(problem)
    values = {o.name: [] for o in problem.objectives}
    for x in sample_points(problem, samples=samples, seed=seed):
        try:
            ev = problem.evaluate(x, tools)
        except EvaluationError:
            continue
        for o in problem.objectives:
            values[o.name].append(ev.objectives[o.name] * _si(o))
    pairs = []
    for a, b in itertools.combinations(problem.objectives, 2):
        va, vb = values[a.name], values[b.name]
        if not va:
            continue
        same = all(_agree(x, y, rtol) for x, y in zip(va, vb))
        negated = all(_agree(x, -y, rtol) for x, y in zip(va, vb))
        if (same and a.sense != b.sense) or (negated and a.sense == b.sense):
            pairs.append((a.name, b.name))
    return pairs


@dataclass(frozen=True)
class _Atom:
    label: str
    equality: bool
    pattern: tuple


def _pattern(residuals: Sequence[float], equality: bool, tol: float) -> tuple:
    if not equality:
        return tuple(r <= tol for r in residuals)
    signs = tuple(0 if abs(r) <= tol else (1 if r > 0 else -1) for r in residuals)
    first = next((s for s in signs if s), 1)
    return tuple(s * first for s in signs)


def _atoms(problem: OptimizationProblem, label: str, evaluations: Sequence, points_b: Sequence[Mapping[str, float]],
           to_b: Mapping[str, str], scale: Mapping[str, float], tol: float) -> list[_Atom]:
    atoms = []
    for c in problem.constraints:
        if isinstance(c.expression, ToolCall):
            continue
        residuals = [ev.residuals[c.name] for ev in evaluations]
        atoms.append(_Atom(f"{label} constraint {c.name!r}", c.sense == "==",
                           _pattern(residuals, c.sense == "==", tol * max(1.0, abs(c.rhs)))))
    for v in problem.variables:
        kind, lo, hi, choices = _domain(v, scale[v.name])
        for side, bound in (("lower", lo), ("upper", hi)):
            if bound is None or choices or v.kind == "binary":
                continue
            sign = -1.0 if side == "lower" else 1.0
            residuals = [sign * (p[to_b[v.name]] - bound) for p in points_b]
            atoms.append(_Atom(f"{label} {side} bound of {v.name!r}", False,
                               _pattern(residuals, False, tol * max(1.0, abs(bound)))))
    return [a for a in atoms if a.equality or not all(a.pattern)]


def _tool_items(problem: OptimizationProblem):
    return [(group, item) for group, items in (("objective", problem.objectives), ("constraint", problem.constraints))
            for item in items if isinstance(item.expression, ToolCall)]


def _si_values(problem: OptimizationProblem, x: Mapping[str, float]) -> dict[str, float]:
    values = {v.name: x[v.name] * _si(v) for v in problem.variables}
    values.update({p.name: p.value * _si(p) for p in problem.parameters})
    return values


def _same_tool_item(ga, ia, gb, ib, si_a, si_b, rtol) -> bool:
    ea, eb = ia.expression, ib.expression
    if ga != gb or ea.tool_id != eb.tool_id or set(ea.args) != set(eb.args):
        return False
    ua, ub = parse_unit(ia.unit), parse_unit(ib.unit)
    if ua.dims != ub.dims or not _agree(ua.scale, ub.scale, 1e-12):
        return False
    if ga == "constraint" and not (ia.sense == ib.sense and _agree(ia.rhs * ua.scale, ib.rhs * ub.scale, rtol)):
        return False
    if ga == "objective" and ia.sense != ib.sense:
        return False
    for arg in ea.args:
        for xa, xb in zip(si_a, si_b):
            try:
                va, vb = Expression(ea.args[arg]).evaluate(xa), Expression(eb.args[arg]).evaluate(xb)
            except EvaluationError:
                continue
            if not _agree(va, vb, rtol):
                return False
    return True


def semantically_equivalent(a: OptimizationProblem, b: OptimizationProblem, *, rtol: float = 1e-6,
                            samples: int = 256, seed: int = 0, aliases: Mapping[str, str] | None = None,
                            reference_x: Mapping[str, float] | Sequence[float] | None = None,
                            tools: Mapping[str, Callable[..., Any]] | None = None) -> EquivalenceReport:
    """Compare compiled problem ``a`` with reference ``b``; see the module docstring for every check.

    ``aliases`` maps a variable name of ``a`` to its name in ``b``.
    ``reference_x`` (by name, or a sequence in ``b``'s variable order, in
    ``b``'s units) is added to the probe points, so the reference optimum is
    always checked.
    """
    reasons: list[str] = []
    notes: list[str] = ["budget and verification policy are not compared"]
    aliases = dict(aliases or {})
    names_a = [v.name for v in a.variables]
    for unknown in sorted(set(aliases) - set(names_a)):
        reasons.append(f"alias for {unknown!r}, which is not a compiled variable")
    to_b = {n: aliases.get(n, n) for n in names_a}
    if len(set(to_b.values())) != len(to_b):
        reasons.append("aliases map two compiled variables to the same reference variable")
    va = {to_b[v.name]: v for v in a.variables}
    vb = {v.name: v for v in b.variables}
    for name in sorted(va.keys() - vb.keys()):
        reasons.append(f"variable {name!r} is only in the compiled problem")
    for name in sorted(vb.keys() - va.keys()):
        reasons.append(f"variable {name!r} is only in the reference problem")
    if reasons:
        return EquivalenceReport(False, tuple(reasons), tuple(notes), 0)

    scale_a: dict[str, float] = {}
    domains: dict[str, tuple[tuple, tuple]] = {}
    for name, v in vb.items():
        ua, ub = parse_unit(va[name].unit), parse_unit(v.unit)
        if ua.dims != ub.dims:
            reasons.append(f"variable {name!r}: unit {va[name].unit!r} and {v.unit!r} differ in dimension")
            continue
        scale_a[va[name].name] = ua.scale / ub.scale
        da, db = _domain(va[name], ua.scale / ub.scale), _domain(v)
        domains[name] = (da, db)
        if da[0] != db[0]:
            reasons.append(f"variable {name!r}: kind {va[name].kind} vs {v.kind}")
        if len(da[3]) != len(db[3]) or not all(_agree(x, y, rtol) for x, y in zip(sorted(da[3]), sorted(db[3]))):
            reasons.append(f"variable {name!r}: choices {list(da[3])} vs {list(db[3])}")
        for i, side in ((1, "lower"), (2, "upper")):
            if da[i] is not None and db[i] is not None and not _agree(da[i], db[i], rtol):
                reasons.append(f"variable {name!r}: {side} bound {da[i]:g} vs {db[i]:g}")
            elif (da[i] is None) != (db[i] is None):
                notes.append(f"variable {name!r}: {side} side is open in the "
                             f"{'compiled' if da[i] is None else 'reference'} problem; compared inside the finite side")
    if len(scale_a) != len(va):
        return EquivalenceReport(False, tuple(reasons), tuple(notes), 0)

    names = [v.name for v in b.variables]
    extra = []
    if reference_x is not None:
        extra.append(dict(reference_x) if isinstance(reference_x, Mapping) else dict(zip(names, reference_x)))
    points_b = _points(names, {n: domains[n][1][0] for n in names},
                       {n: _box(domains[n]) for n in names},
                       {n: tuple(sorted(set(domains[n][0][3]) | set(domains[n][1][3]))) for n in names},
                       samples, seed, extra)
    from_b = {to_b[n]: n for n in names_a}
    points_a = [{from_b[n]: value / scale_a[from_b[n]] for n, value in p.items()} for p in points_b]

    all_tools = {**opaque_tools(a, b), **(tools or {})}
    kept_a, kept_b, evs_a, evs_b = [], [], [], []
    failures = 0
    for pa, pb in zip(points_a, points_b):
        results = []
        for problem, point in ((a, pa), (b, pb)):
            try:
                results.append(problem.evaluate(point, all_tools))
            except EvaluationError:
                results.append(None)
        if (results[0] is None) != (results[1] is None):
            failures += 1
        elif results[0] is not None:
            kept_a.append(pa)
            kept_b.append(pb)
            evs_a.append(results[0])
            evs_b.append(results[1])
    if failures:
        reasons.append(f"evaluation fails in one problem only at {failures} probe points")
    if not kept_b:
        reasons.append("no probe point can be evaluated in both problems")
        return EquivalenceReport(False, tuple(reasons), tuple(notes), len(points_b))

    if len(a.objectives) != len(b.objectives):
        reasons.append(f"{len(a.objectives)} compiled objectives vs {len(b.objectives)} reference objectives")
    else:
        tol = b.verification.constraint_tol
        inside = [i for i, (pa, pb) in enumerate(zip(kept_a, kept_b))
                  if all(_in_domain(_domain(v), pa[v.name], tol) for v in a.variables)
                  and all(_in_domain(_domain(v), pb[v.name], tol) for v in b.variables)]
        if not inside:
            reasons.append("no probe point lies inside both variable domains")
        unmatched = list(a.objectives)
        for ob in b.objectives:
            values_b = [evs_b[i].objectives[ob.name] * _si(ob) for i in inside]
            match = next((oa for oa in unmatched if oa.sense == ob.sense and all(
                _agree(evs_a[i].objectives[oa.name] * _si(oa), vb_, rtol) for i, vb_ in zip(inside, values_b))), None)
            if match is None:
                same_values = next((oa for oa in unmatched if all(
                    _agree(evs_a[i].objectives[oa.name] * _si(oa), vb_, rtol) for i, vb_ in zip(inside, values_b))),
                    None)
                reasons.append(f"reference objective {ob.name!r} ({ob.sense}) has no compiled objective with the "
                               f"same sense and values" + (f"; {same_values.name!r} has the opposite sense"
                                                            if same_values is not None else ""))
            else:
                unmatched.remove(match)

    disagree = [i for i in range(len(kept_b))
                if a.constraints_satisfied(kept_a[i], evs_a[i].residuals)
                != b.constraints_satisfied(kept_b[i], evs_b[i].residuals)]
    if disagree:
        reasons.append(f"feasibility differs at {len(disagree)} of {len(kept_b)} probe points, "
                       f"e.g. {kept_b[disagree[0]]}")

    tol = b.verification.constraint_tol
    atoms_a = _atoms(a, "compiled", evs_a, kept_b, to_b, scale_a, tol)
    atoms_b = _atoms(b, "reference", evs_b, kept_b, {n: n for n in names}, {n: 1.0 for n in names}, tol)
    for mine, theirs, side in ((atoms_b, atoms_a, "compiled"), (atoms_a, atoms_b, "reference")):
        for atom in mine:
            if not any(t.equality == atom.equality and t.pattern == atom.pattern for t in theirs):
                reasons.append(f"{atom.label} has no {side} counterpart satisfied at the same probe points")

    si_a = [_si_values(a, p) for p in kept_a]
    si_b = [_si_values(b, p) for p in kept_b]
    tools_a = _tool_items(a)
    for gb, ib in _tool_items(b):
        match = next(((ga, ia) for ga, ia in tools_a if _same_tool_item(ga, ia, gb, ib, si_a, si_b, rtol)), None)
        if match is None:
            reasons.append(f"reference {gb} {ib.name!r} calls tool {ib.expression.tool_id!r}; no compiled {gb} "
                           "makes the same call with the same sense, rhs and unit")
        else:
            tools_a.remove(match)
    reasons.extend(f"compiled {g} {i.name!r} calls tool {i.expression.tool_id!r} with no reference counterpart"
                   for g, i in tools_a)
    return EquivalenceReport(not reasons, tuple(reasons), tuple(notes), len(kept_b))
