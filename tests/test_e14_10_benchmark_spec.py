"""E14-10: the pre-registered benchmark specification is internally consistent and its reference data is verified.

The spec (``benchmarks/e14_spec``, documented in ``docs/benchmarks/e14_benchmark_spec.md``) is data, not a runner.
These gates check every stored optimum by a means that does not read the stored number (the closed form at the stated
optimiser, brute force written here, scipy cross-checks where scipy is installed, an independent quadrature), that the
expression strings a compiler will see compute the same function as the scoring callables, that every fault injector
changes exactly what it claims, that the statistics refuse the E11R-01 mistake, and that the document and the data
agree, down to the spec digest.
"""
from __future__ import annotations

import copy
import importlib.util
import itertools
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "benchmarks/e14_spec"
DOC = REPO / "docs/benchmarks/e14_benchmark_spec.md"


def _load():
    if "e14_spec" not in sys.modules:
        spec = importlib.util.spec_from_file_location("e14_spec", PKG / "__init__.py",
                                                      submodule_search_locations=[str(PKG)])
        mod = importlib.util.module_from_spec(spec)
        sys.modules["e14_spec"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["e14_spec"]


spec = _load()
P, A, M, F, FC = spec.problems, spec.arms, spec.metrics, spec.faults, spec.formulation
RUNNABLE = P.INSTANCES
SINGLE = [i for i in RUNNABLE if len(i.objectives) == 1]
CONTINUOUS_SINGLE = [i for i in SINGLE if i.variable_type == "continuous"]


def _better(inst, a: float, b: float) -> bool:
    """a strictly better than b by more than the instance tolerance."""
    tol = inst.tolerance.objective_abs + inst.tolerance.objective_rel * abs(b) + 1e-12
    return a < b - tol if inst.sense == "minimize" else a > b + tol


def _feasible(inst, x, tol=None) -> bool:
    return F.max_violation(inst, x) <= (inst.tolerance.constraint_abs if tol is None else tol)


def _sample(inst, rng, k):
    lo = np.array([b[0] for b in inst.bounds])
    hi = np.array([b[1] for b in inst.bounds])
    pts = lo + (hi - lo) * rng.random((k, inst.dimension))
    if inst.variable_type in ("integer", "binary"):
        pts = np.round(pts)
    return pts


# ------------------------------------------------------------------------------------------------- catalogue shape
def test_catalogue_has_at_least_24_runnable_instances_with_unique_ids_and_every_class():
    ids = [i.id for i in RUNNABLE] + [e.id for e in P.EXTERNAL]
    assert len(RUNNABLE) >= 24
    assert len(ids) == len(set(ids))
    assert {i.problem_class for i in RUNNABLE} == set(P.PROBLEM_CLASSES)
    assert {i.family for i in RUNNABLE} >= {"rosenbrock", "rastrigin", "ackley", "constrained_quadratic", "knapsack",
                                             "integration"}


@pytest.mark.parametrize("inst", RUNNABLE, ids=lambda i: i.id)
def test_every_instance_has_all_required_fields(inst):
    assert inst.status == "runnable"
    assert inst.problem_class in P.PROBLEM_CLASSES and inst.reference.method in P.REFERENCE_METHODS
    assert len(inst.bounds) == inst.dimension >= 1 and all(lo < hi for lo, hi in inst.bounds)
    assert len(inst.objective_exprs) == len(inst.objectives) >= 1
    assert (len(inst.objectives) > 1) == (inst.problem_class == "multi_objective")
    assert inst.reference.derivation.strip() and inst.statement.strip()
    assert inst.split in ("tuning", "scoring") and inst.sense in ("minimize", "maximize")
    t = inst.tolerance
    assert t.objective_abs >= 0 and t.objective_rel >= 0 and t.constraint_abs >= 0
    if inst.reference.indicator == "hypervolume":
        assert len(inst.reference.reference_point) == len(inst.objectives)
    if inst.reference.x is not None:
        assert len(inst.reference.x) == inst.dimension
    for c in inst.constraints:
        assert c.kind in ("le", "eq") and c.expr and c.name


def test_external_entries_are_marked_and_not_runnable():
    runnable_ids = {i.id for i in RUNNABLE}
    assert {e.suite for e in P.EXTERNAL} == {"CUTEst", "COCO/BBOB"}
    for e in P.EXTERNAL:
        assert e.status == "requires_optional_dependency" and e.optional_dependency and e.why
        assert e.id not in runnable_ids and e.problem_class in P.PROBLEM_CLASSES


def test_tuning_and_scoring_are_disjoint_and_every_family_is_scored():
    assert not set(M.TUNING_SEEDS) & set(M.SCORING_SEEDS)
    families = {i.family for i in RUNNABLE}
    assert {i.family for i in RUNNABLE if i.split == "scoring"} == families
    assert {i.family for i in RUNNABLE if i.split == "tuning"} == families  # one tuning instance per family
    assert P.by_id("cqp-worked-d2").split == "tuning"  # the worked example is in every prompt draft


@pytest.mark.parametrize("inst", RUNNABLE, ids=lambda i: i.id)
def test_expression_strings_compute_the_same_function_as_the_callables(inst):
    """The compiler sees the strings and scoring uses the callables, so they must agree at points in the box."""
    rng = np.random.default_rng(14)
    pts = _sample(inst, rng, 12)
    if inst.family == "integration":
        pts[:, 1] = rng.integers(1, 65, size=len(pts))  # keep Gauss-Legendre node counts cheap
    if inst.reference.x is not None:
        pts = np.vstack([pts, inst.reference.x])
    for x in pts:
        for expr, fn in zip(inst.objective_exprs, inst.objectives):
            assert P.evaluate_expr(inst, expr, x) == pytest.approx(fn(x), rel=1e-12, abs=1e-12)
        for c in inst.constraints:
            want = c.fn(x)
            got = P.evaluate_expr(inst, c.expr, x)
            assert got == want if math.isinf(want) else got == pytest.approx(want, rel=1e-12, abs=1e-12)


# ---------------------------------------------------------------------------------------- reference optima, per kind
@pytest.mark.parametrize("inst", [i for i in SINGLE if i.reference.x is not None], ids=lambda i: i.id)
def test_stated_optimiser_attains_the_reference_value_and_is_feasible(inst):
    x = inst.reference.x
    assert inst.objectives[0](x) == pytest.approx(inst.reference.value, abs=max(1e-12, inst.tolerance.objective_abs))
    assert _feasible(inst, x, tol=1e-12)


@pytest.mark.parametrize("inst", CONTINUOUS_SINGLE, ids=lambda i: i.id)
def test_no_sampled_feasible_point_beats_the_reference(inst):
    rng = np.random.default_rng(1410)
    for x in _sample(inst, rng, 4000):
        if _feasible(inst, x):
            assert not _better(inst, inst.objectives[0](x), inst.reference.value), x


@pytest.mark.parametrize("inst", [i for i in CONTINUOUS_SINGLE if i.dimension == 2], ids=lambda i: i.id)
def test_two_dimensional_grid_brute_force_finds_nothing_better(inst):
    (l0, h0), (l1, h1) = inst.bounds
    best = None
    for x in itertools.product(np.linspace(l0, h0, 151), np.linspace(l1, h1, 151)):
        if _feasible(inst, x):
            f = inst.objectives[0](x)
            best = f if best is None or _better(inst, f, best) else best
    assert best is not None and not _better(inst, best, inst.reference.value)


def test_rastrigin_one_dimensional_term_brute_force():
    """Rastrigin is separable: f = sum_i (10 + z_i^2 - 10 cos 2 pi z_i), so the n-D minimum is n times the 1-D one."""
    z = np.linspace(-5.12, 5.12, 1_024_001)  # step 1e-5, contains 0
    term = 10.0 + z ** 2 - 10.0 * np.cos(2.0 * np.pi * z)
    assert term.min() == pytest.approx(0.0, abs=1e-12) and abs(z[term.argmin()]) < 1e-9
    for inst in (i for i in RUNNABLE if i.family == "rastrigin"):
        assert inst.reference.value == 0.0


def test_scipy_multistart_never_beats_the_reference_and_reaches_it_on_convex_instances():
    opt = pytest.importorskip("scipy.optimize")
    for inst in CONTINUOUS_SINGLE:
        rng = np.random.default_rng(7)
        cons = [{"type": "ineq" if c.kind == "le" else "eq", "fun": (lambda x, c=c: -c.fn(x) if c.kind == "le"
                                                                     else c.fn(x))} for c in inst.constraints]
        sign = 1.0 if inst.sense == "minimize" else -1.0
        best = None
        for x0 in _sample(inst, rng, 6 if inst.dimension <= 10 else 2):
            r = opt.minimize(lambda x: sign * inst.objectives[0](x), x0, method="SLSQP", bounds=inst.bounds,
                             constraints=cons, options={"maxiter": 500, "ftol": 1e-12})
            if _feasible(inst, r.x):
                f = inst.objectives[0](r.x)
                # SLSQP stops up to ~1e-7 outside a constraint, which can buy that much objective: allow 1e-6 rel
                slack = 1e-6 * max(1.0, abs(inst.reference.value))
                assert (f - inst.reference.value) * (1 if inst.sense == "minimize" else -1) >= -slack, (inst.id, f)
                best = f if best is None or _better(inst, f, best) else best
        if inst.problem_class in ("lp", "convex_qp"):
            assert best == pytest.approx(inst.reference.value, abs=1e-6), inst.id


def test_scipy_differential_evolution_reaches_the_global_reference_in_two_dimensions():
    opt = pytest.importorskip("scipy.optimize")
    for inst in (i for i in CONTINUOUS_SINGLE if i.problem_class == "global" and i.dimension == 2):
        r = opt.differential_evolution(inst.objectives[0], inst.bounds, seed=0, tol=1e-12, polish=True)
        assert r.fun == pytest.approx(inst.reference.value, abs=1e-6), inst.id


@pytest.mark.parametrize("inst", [i for i in RUNNABLE if i.problem_class == "lp" and i.dimension == 2],
                         ids=lambda i: i.id)
def test_two_dimensional_lp_vertex_enumeration(inst):
    """Every vertex is the intersection of two active lines among the constraints and the bounds."""
    lines = []  # a . x = b
    for c in inst.constraints:  # all linear: recover a, b from three evaluations
        b0 = c.fn((0.0, 0.0))
        lines.append((np.array([c.fn((1.0, 0.0)) - b0, c.fn((0.0, 1.0)) - b0]), -b0))
    for k, (lo, hi) in enumerate(inst.bounds):
        e = np.eye(2)[k]
        lines += [(e, lo), (e, hi)]
    vertices = []
    for (a1, b1), (a2, b2) in itertools.combinations(lines, 2):
        mat = np.array([a1, a2])
        if abs(np.linalg.det(mat)) > 1e-12:
            v = np.linalg.solve(mat, [b1, b2])
            if _feasible(inst, v, tol=1e-9):
                vertices.append(inst.objectives[0](v))
    best = min(vertices) if inst.sense == "minimize" else max(vertices)
    assert best == pytest.approx(inst.reference.value, abs=1e-9)


def _knapsack_rows(inst):
    rows = [inst.data[k] for k in sorted(inst.data) if re.fullmatch(r"w\d*", k)]
    caps = [inst.data[k] for k in sorted(inst.data) if re.fullmatch(r"C\d*", k)]
    return np.array(rows), np.array(caps)


@pytest.mark.parametrize("inst", [i for i in RUNNABLE if i.family == "knapsack" and i.variable_type == "binary"],
                         ids=lambda i: i.id)
def test_knapsack_brute_force_over_every_subset(inst):
    n = inst.dimension
    masks = np.arange(2 ** n, dtype=np.uint32)
    bits = ((masks[:, None] >> np.arange(n, dtype=np.uint32)) & 1).astype(np.int64)
    rows, caps = _knapsack_rows(inst)
    ok = np.all(bits @ rows.T <= caps, axis=1)
    values = bits @ np.array(inst.data["v"])
    assert values[ok].max() == inst.reference.value
    if inst.reference.x is not None:  # stored only when the optimum is unique
        assert (values[ok] == inst.reference.value).sum() == 1


def test_lp_relaxation_matches_greedy_and_linprog():
    inst = P.by_id("knapsack-p01-lp-relaxation")
    w, v, cap = np.array(inst.data["w"], float), np.array(inst.data["v"], float), float(inst.data["C"])
    total, room = 0.0, cap
    for i in sorted(range(len(w)), key=lambda i: -v[i] / w[i]):
        take = min(1.0, room / w[i])
        total, room = total + take * v[i], room - take * w[i]
    assert total == pytest.approx(inst.reference.value, abs=1e-9)
    opt = pytest.importorskip("scipy.optimize")
    r = opt.linprog(-v, A_ub=[w], b_ub=[cap], bounds=inst.bounds, method="highs")
    assert -r.fun == pytest.approx(inst.reference.value, abs=1e-9)


# ------------------------------------------------------------------------------------------------------ integration
INTEGRATION = [i for i in RUNNABLE if i.family == "integration"]
_NP = {"sqrt": np.sqrt, "exp": np.exp, "cos": np.cos, "sin": np.sin, "atan": np.arctan, "__builtins__": {}}


def _integrand(inst):
    return lambda t: eval(inst.data["integrand"], dict(_NP, t=t))  # noqa: S307 -- catalogue literal


def _closed_form(inst) -> float:
    """Independent of problems.py: recompute each integral from its antiderivative."""
    eps = 1e-3
    return {
        "exp(t)": math.e - 1.0,
        "1/(1+25*t**2)": 0.4 * math.atan(5.0),
        "1/sqrt(t+0.001)": 2.0 * (math.sqrt(1.0 + eps) - math.sqrt(eps)),
        "1/((t-0.3)**2+0.01)": 10.0 * (math.atan(7.0) + math.atan(3.0)),
        "cos(20*t)": math.sin(20.0) / 20.0,
        "exp(-t**2)": 0.5 * math.sqrt(math.pi) * math.erf(1.0),
    }[inst.data["integrand"]]


def _graded_gauss(f, a, b) -> float:
    """30-node Gauss-Legendre on panels graded geometrically toward a and refined around the interior."""
    t, w = np.polynomial.legendre.leggauss(30)
    edges = sorted({a, b, *(a + (b - a) * 2.0 ** -k for k in range(1, 40)), *np.linspace(a, b, 65)})
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        total += 0.5 * (hi - lo) * float(np.dot(w, f(0.5 * (hi - lo) * t + 0.5 * (hi + lo))))
    return total


@pytest.mark.parametrize("inst", INTEGRATION, ids=lambda i: i.id)
def test_closed_form_integral_agrees_with_high_order_quadrature(inst):
    a, b = inst.data["interval"]
    exact = _closed_form(inst)
    assert _graded_gauss(_integrand(inst), a, b) == pytest.approx(exact, abs=1e-13)
    assert inst.blackboxes["quadrature_error"](*inst.reference.x) <= inst.data["tau"]  # stored exact is the same
    integrate = pytest.importorskip("scipy.integrate")
    q, _ = integrate.quad(lambda t: float(_integrand(inst)(np.array(t))), a, b, epsabs=1e-13, epsrel=1e-12,
                          limit=500, points=[0.3] if "0.3" in inst.data["integrand"] else None)
    assert q == pytest.approx(exact, abs=1e-12)


def test_quadrature_rules_have_their_textbook_exactness():
    """Gauss-Legendre (also after x = a + (b - a) s^2) with N nodes is exact through degree 2N-1 in its variable,
    trapezoid through degree 1, Simpson through degree 3; each fails one degree higher."""
    a, b = 0.2, 1.7

    def mono(k):
        return (lambda x: x ** k), (b ** (k + 1) - a ** (k + 1)) / (k + 1)
    for n in (2, 3, 5):
        f, ex = mono(2 * n - 1)
        assert P.quadrature(f, a, b, 0, n) == pytest.approx(ex, rel=1e-13)
        f, ex = mono(2 * n)
        assert abs(P.quadrature(f, a, b, 0, n) - ex) > 1e-8
        f, ex = mono(n - 1)  # degree n-1 in x is degree 2n-1 in s
        assert P.quadrature(f, a, b, 3, n) == pytest.approx(ex, rel=1e-13)
    for k, rule, n in ((1, 1, 5), (3, 2, 5)):
        f, ex = mono(k)
        assert P.quadrature(f, a, b, rule, n) == pytest.approx(ex, rel=1e-13)
        f, ex = mono(k + 1)
        assert abs(P.quadrature(f, a, b, rule, n) - ex) > 1e-8
    assert math.isnan(P.quadrature(np.exp, a, b, 2, 4)) and math.isnan(P.quadrature(np.exp, a, b, 1, 1))


@pytest.mark.parametrize("inst", INTEGRATION, ids=lambda i: i.id)
def test_minimum_cost_rule_by_exhaustive_search_with_margin(inst):
    """Search N = 1, 2, ... over every rule; the first N meeting the target is optimal because the cost is N."""
    err = inst.blackboxes["quadrature_error"]
    tau = inst.data["tau"]
    found = next((n, r) for n in range(1, P.N_MAX + 1) for r in range(len(P.QUADRATURE_RULES)) if err(r, n) <= tau)
    assert found == (int(inst.reference.value), int(inst.reference.x[0]))
    # not a knife edge: every cheaper decision misses by >= 10 %, and the optimum clears the target by >= 10 %
    n_star = found[0]
    cheaper = min(err(r, n) for n in range(1, n_star) for r in range(len(P.QUADRATURE_RULES)))
    assert cheaper >= 1.1 * tau and err(found[1], n_star) <= tau / 1.1


# -------------------------------------------------------------------------------------------------- multi-objective
MULTI = [i for i in RUNNABLE if i.problem_class == "multi_objective"]


def _hypervolume_2d(points, ref) -> float:
    pts = sorted(p for p in points if p[0] <= ref[0] and p[1] <= ref[1])
    hv, prev = 0.0, ref[1]
    for f1, f2 in pts:
        if f2 < prev:
            hv += (ref[0] - f1) * (prev - f2)
            prev = f2
    return hv


def _pareto_set(inst, t):
    """Parametrise the claimed Pareto set by t in [0, 1]."""
    return {"mo-schaffer-d1": lambda t: [2.0 * t], "mo-two-centres-d2": lambda t: [t, 0.0],
            "mo-zdt1-d5": lambda t: [t, 0, 0, 0, 0], "mo-zdt2-d5": lambda t: [t, 0, 0, 0, 0]}[inst.id](t)


@pytest.mark.parametrize("inst", MULTI, ids=lambda i: i.id)
def test_hypervolume_reference_is_attained_and_never_exceeded(inst):
    ref, hv_star = inst.reference.reference_point, inst.reference.value
    on_front = [tuple(f(_pareto_set(inst, t)) for f in inst.objectives) for t in np.linspace(0, 1, 20001)]
    assert _hypervolume_2d(on_front, ref) == pytest.approx(hv_star, rel=1e-3)
    rng = np.random.default_rng(3)
    sampled = [tuple(f(x) for f in inst.objectives) for x in _sample(inst, rng, 20000)]
    assert _hypervolume_2d(sampled + on_front, ref) <= hv_star + 1e-9
    even50 = [tuple(f(_pareto_set(inst, t)) for f in inst.objectives) for t in np.linspace(0, 1, P.MO_MAX_POINTS)]
    gap = (hv_star - _hypervolume_2d(even50, ref)) / hv_star
    assert gap <= inst.tolerance.objective_rel / 1.5  # the success threshold is reachable with the allowed 50 points


# -------------------------------------------------------------------------------------------------- arms and metrics
def test_five_arms_defined_and_comparable():
    assert A.ARM_IDS == ("A", "B", "C", "D", "E")
    for arm in A.ARMS:
        assert arm.name and arm.human_supplies and arm.solver
        for shared in ("cell seeds", "evaluation budget", "wall-clock cap", "verifier"):
            assert shared in arm.held_fixed, (arm.id, shared)
        uses_llm = arm.id in ("C", "D", "E")
        assert bool(arm.llm_supplies) == uses_llm and (arm.llm_calls_max > 0) == uses_llm
        assert arm.problem_input == ("nl_statement" if uses_llm else "reference_problem")
    by = {a.id: a for a in A.ARMS}
    assert by["D"].solver == by["A"].solver == "fixed"  # D vs A isolates the compiler
    assert set(A.RULE_TABLE) == set(P.PROBLEM_CLASSES)
    assert {x for pair in A.PRIMARY_COMPARISONS for x in pair} <= set(A.ARM_IDS)
    assert A.BUDGET.evaluations(1) == A.BUDGET.evaluations(2) == 4000 and A.BUDGET.evaluations(20) == 40_000


def test_ten_metrics_defined_with_unit_denominator_and_arms():
    assert set(M.METRIC_IDS) == {"ir_schema_validity", "formulation_accuracy", "hard_constraint_preservation",
                                 "false_verification_pass", "budget_enforcement", "experiment_replay",
                                 "optimality_gap", "function_evaluations", "wall_clock", "token_cost"}
    assert len(M.METRIC_IDS) == 10
    for m in M.METRICS:
        assert m.unit and m.denominator and m.definition and m.target and m.arms
        assert set(m.arms) <= set(A.ARM_IDS)
        assert m.threshold_status == "proposed; frozen in ADR optimization-0001 by E14-01"
    assert set(M.PRIMARY_ENDPOINTS) <= set(M.METRIC_IDS)
    assert len(A.PRIMARY_COMPARISONS) * len(M.PRIMARY_ENDPOINTS) == 15  # the Holm family the document states


def test_optimality_gap_orientation():
    assert M.optimality_gap(0.5, 0.6, "minimize") == pytest.approx(0.1)
    assert M.optimality_gap(309.0, 300.0, "maximize") == pytest.approx(9.0 / 309.0)
    assert M.optimality_gap(5 / 6, 0.8, "minimize", "hypervolume") == pytest.approx(5 / 6 - 0.8)


# ------------------------------------------------------------------------------------------------------- statistics
def test_exact_mcnemar_matches_the_e11r01_record():
    """Values the E11R-01 campaign tables report."""
    assert M.exact_mcnemar_p(2, 0) == 0.5 and M.exact_mcnemar_p(3, 0) == 0.25
    assert M.exact_mcnemar_p(11, 0) == pytest.approx(0.0009766, rel=1e-4)
    assert M.exact_mcnemar_p(0, 0) == 1.0


def test_cluster_sign_flip_reduces_to_exact_mcnemar_with_one_pair_per_instance():
    for b, c in itertools.product(range(8), range(8)):
        assert M.cluster_sign_flip_p([1] * b + [-1] * c) == pytest.approx(M.exact_mcnemar_p(b, c), abs=1e-15)


def test_six_informative_instances_are_the_floor_for_any_claim():
    assert M.cluster_sign_flip_p([1] * 5) == pytest.approx(0.0625)
    assert M.cluster_sign_flip_p([1] * 6) == pytest.approx(0.03125)
    assert M.MIN_CLUSTERS_FOR_CLAIM == 6


def test_shared_instances_cannot_manufacture_significance():
    """The E11R-01 trap: 3 of 24 instances flip in all 10 seeds. Pooled over 240 (instance, seed) pairs the exact
    McNemar test calls it overwhelming; clustered by instance it is 3 informative instances, which is not testable,
    and the instance-resampling CI covers 0 while a naive pooled binomial CI does not."""
    per_instance = {f"i{k}": (1.0 if k < 3 else 0.0) for k in range(24)}
    discordance = [10 if k < 3 else 0 for k in range(24)]
    assert M.exact_mcnemar_p(30, 0) < 1e-8
    assert M.cluster_sign_flip_p(discordance) == pytest.approx(0.25)
    lo, hi = M.cluster_bootstrap_ci(per_instance, {k: "f" for k in per_instance}, resamples=4000)
    p_hat = 30 / 240
    naive_lo = p_hat - 1.96 * math.sqrt(p_hat * (1 - p_hat) / 240)
    assert lo <= 0.0 < naive_lo and hi > p_hat
    assert M.verdict(0.25, (lo, hi), informative_clusters=3) == "not testable"


def test_verdict_requires_holm_significance_and_ci_agreement():
    assert M.verdict(0.01, (0.05, 0.3), 8) == "superior"
    assert M.verdict(0.01, (-0.3, -0.05), 8) == "inferior"
    assert M.verdict(0.01, (-0.01, 0.3), 8) == "not shown different"  # test and CI disagree
    assert M.verdict(0.2, (0.05, 0.3), 8) == "not shown different"  # better estimate, no significance
    assert M.verdict(0.001, (0.05, 0.3), 5) == "not testable"


def test_holm_and_wilcoxon():
    assert M.holm({"a": 0.01, "b": 0.04, "c": 0.03}) == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})
    assert M.wilcoxon_signed_rank_p([1, 2, 3, 4, 5]) == pytest.approx(2 / 32)
    assert M.wilcoxon_signed_rank_p([0, 0]) == 1.0
    assert M.wilcoxon_signed_rank_p([math.inf] * 6) == pytest.approx(2 / 64)  # never-solved ranks are the largest
    stats = pytest.importorskip("scipy.stats")
    d = np.random.default_rng(5).normal(0.3, 1.0, 12)
    assert M.wilcoxon_signed_rank_p(list(d)) == pytest.approx(stats.wilcoxon(d, method="exact").pvalue, rel=1e-9)


def test_analysis_groups_cover_every_class_once_and_report_testability():
    classes = [c for group in M.ANALYSIS_GROUPS.values() for c in group]
    assert sorted(classes) == sorted(P.PROBLEM_CLASSES)
    counts = {g: sum(i.split == "scoring" and i.problem_class in cls for i in RUNNABLE)
              for g, cls in M.ANALYSIS_GROUPS.items()}
    assert counts == {"algebraic": 10, "continuous_nonconvex": 10, "black_box": 6, "multi_objective": 3}


# ----------------------------------------------------------------------------------------------------------- faults
def _oracle(result: dict, inst) -> str:
    """Minimal reference verifier: proves each fault is detectable from the artifact plus the reference problem,
    and that the checks are orthogonal (each fault trips its own check first)."""
    ev = result["evidence"]
    if F.artifact_digest(result) != ev["artifact_digest"]:
        return "digest_mismatch"
    if "independent_check" not in ev:
        return "missing_evidence"
    if result["problem_digest"] != F.problem_digest(inst):
        return "problem_mismatch"
    ref, solved = P.neutral_form(inst), ev["solved_problem"]
    if any(s[0] < r[0] or s[1] > r[1] for s, r in zip(solved["bounds"], ref["bounds"])):
        return "unauthorized_relaxation"
    if solved != ref:
        return "formulation_mismatch"
    if ev["cache"]["hit"] and ev["cache"]["problem_digest"] != result["problem_digest"]:
        return "stale_cache"
    check = ev["independent_check"]
    if check["verifier"] not in F.INDEPENDENT_VERIFIERS or check["verifier"] == ev["solver"]:
        return "forged_verification"
    if abs(inst.objectives[0](result["x"]) - result["objective"]) > max(1e-9, inst.tolerance.objective_abs):
        return "objective_mismatch"
    if F.max_violation(inst, result["x"]) > inst.tolerance.constraint_abs:
        return "constraint_violation"
    return "pass"


def _params(fault, inst) -> dict:
    other = next(i for i in RUNNABLE if i.id != inst.id)
    stale = dict(P.neutral_form(inst), objective_exprs=["0"])
    outside = [hi + 1.0 for _, hi in inst.bounds]
    return {"F02": {"x": outside, "objective": inst.objectives[0](outside)},
            "F05": {"other_problem_digest": F.problem_digest(other)},
            "F07": {"expr": "x[0]"},
            "F08": {"previous_problem_digest": F.digest(stale)}}.get(fault.id, {})


FAULT_HOSTS = ["cqp-worked-d2", "cqp-equality-d3", "knapsack-p01-n10", "rosenbrock-d5"]


def test_catalogue_names_nine_faults():
    assert F.FAULT_IDS == tuple(f"F{k:02d}" for k in range(1, 10))
    assert len({f.reason for f in F.FAULTS}) == 9 and all(f.expected_verdict == "rejected" for f in F.FAULTS)


@pytest.mark.parametrize("host", FAULT_HOSTS)
def test_honest_result_passes_the_oracle(host):
    inst = P.by_id(host)
    assert _oracle(F.make_result(inst, inst.reference.x), inst) == "pass"


@pytest.mark.parametrize("host", FAULT_HOSTS)
@pytest.mark.parametrize("fault", F.FAULTS, ids=lambda f: f.id)
def test_fault_changes_exactly_what_it_claims_and_is_rejected_for_its_reason(fault, host):
    inst = P.by_id(host)
    honest = F.make_result(inst, inst.reference.x)
    before = copy.deepcopy(honest)
    faulty = fault.inject(honest, **_params(fault, inst))
    assert honest == before  # pure: the input is untouched
    expected = set(fault.changes) | {"evidence.artifact_digest"}
    assert F.changed_paths(honest, faulty) == expected
    assert (F.artifact_digest(faulty) == faulty["evidence"]["artifact_digest"]) == fault.resealed
    assert _oracle(faulty, inst) == fault.reason


# ---------------------------------------------------------------------------------------------------- formulation
def test_formulation_examples_follow_the_case_format():
    assert len(FC.EXAMPLES) == 6 and len({c.id for c in FC.EXAMPLES}) == 6
    assert {c.expected_outcome for c in FC.EXAMPLES} == set(FC.OUTCOMES)
    ids = {i.id for i in RUNNABLE}
    for case in FC.EXAMPLES:
        assert case.prompt and case.notes
        if case.expected_outcome == "compile":
            assert case.reference_problem in ids
        assert case.reference_problem is None or case.reference_problem in ids


# ------------------------------------------------------------------------------------------------------ isolation
def test_spec_package_imports_nothing_from_orbitopt():
    for path in PKG.glob("*.py"):
        src = path.read_text(encoding="utf-8")
        assert not re.search(r"^\s*(from|import)\s+orbitopt", src, re.M), path.name


# ------------------------------------------------------------------------------------------- document agreement
def _sections() -> dict[str, list[list[str]]]:
    """Map each '## ' heading to the cell lists of its table rows (header and rule rows dropped)."""
    out: dict[str, list[list[str]]] = {}
    current = None
    for line in DOC.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            out[current] = []
        elif current and line.startswith("| `"):
            out[current].append([c.strip() for c in line.strip().strip("|").split("|")])
    return out


def _ids(rows) -> set[str]:
    return {r[0].strip("`") for r in rows}


def _section(prefix: str):
    sections = _sections()
    match = [rows for title, rows in sections.items() if title.startswith(prefix)]
    assert len(match) == 1, f"one section titled '{prefix}...' expected in {DOC.name}"
    return match[0]


def test_document_and_data_agree_on_every_id():
    assert _ids(_section("2. Instance catalogue")) == {i.id for i in RUNNABLE}
    assert _ids(_section("3. External instances")) == {e.id for e in P.EXTERNAL}
    assert _ids(_section("1. Arms")) == set(A.ARM_IDS)
    assert _ids(_section("4. Metrics")) == set(M.METRIC_IDS)
    assert _ids(_section("6. Fault injection")) == set(F.FAULT_IDS)
    assert _ids(_section("7. Formulation cases")) == {c.id for c in FC.EXAMPLES}


def test_document_catalogue_rows_match_the_data():
    """columns: id | family | class | dim | variables | reference | method | tolerance | split"""
    for row in _section("2. Instance catalogue"):
        inst = P.by_id(row[0].strip("`"))
        assert row[1] == inst.family and row[2] == inst.problem_class and int(row[3]) == inst.dimension
        assert float(row[5]) == pytest.approx(inst.reference.value, rel=1e-6)
        assert row[6] == inst.reference.method and row[8] == inst.split


def test_document_quotes_the_current_spec_digest():
    text = DOC.read_text(encoding="utf-8")
    quoted = re.findall(r"spec digest: `(sha256:[0-9a-f]{64})`", text)
    assert quoted == [spec.spec_digest()], "data changed: update the digest in the doc (before the first scored run)"
