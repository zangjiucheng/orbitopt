"""E14-10 experimental arms A-E and the budget every arm shares.

Comparability rules (all PROPOSED; frozen in ADR optimization-0001 by E14-01):

* every arm runs the same (instance, seed) cells, so every comparison is paired;
* every arm gets the same point-evaluation budget, counted as calls of the objective/constraint functions at one
  point x, whoever makes them (solver, agent probe, scoring of an LLM-proposed candidate);
* wall clock runs from cell start to the verifier's verdict and includes LLM latency, compilation and verification;
* the result an arm returns is always scored with the REFERENCE problem's callables, never with the arm's own
  (possibly mis-compiled) formulation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Budget:
    evaluations_per_dimension: int = 2000
    min_dimension: int = 2
    wall_clock_s: float = 600.0
    llm_tokens_max: int = 200_000  # input + output, per cell; arms A and B use 0
    llm_calls_max: int = 20

    def evaluations(self, dimension: int) -> int:
        return self.evaluations_per_dimension * max(dimension, self.min_dimension)


BUDGET = Budget()


@dataclass(frozen=True)
class Arm:
    id: str
    name: str
    problem_input: Literal["reference_problem", "nl_statement"]
    human_supplies: tuple[str, ...]
    llm_supplies: tuple[str, ...]
    held_fixed: tuple[str, ...]
    solver: str
    llm_calls_max: int


#: arm A's single solver for every class: it handles bounds, nonlinear constraints and integrality, takes a seed,
#: and needs no per-class choice. Named here as a proposal; ADR optimization-0001 may substitute the IR's default.
FIXED_SOLVER = "scipy.optimize.differential_evolution(seed=<cell seed>, polish=True, integrality=<from problem>)"

#: arm B's rule table: problem class -> solver. The rule may read only the declared problem class and variable type.
RULE_TABLE: dict[str, str] = {
    "lp": "scipy.optimize.linprog(method='highs')",
    "milp": "scipy.optimize.milp (HiGHS branch and bound)",
    "convex_qp": "scipy.optimize.minimize(method='trust-constr') from the box centre",
    "nlp": "scipy.optimize.minimize(method='SLSQP') multistart from seeded uniform points until the budget",
    "global": "scipy.optimize.differential_evolution(seed=<cell seed>, polish=True)",
    "black_box": "exhaustive enumeration in increasing cost when the integer grid fits the budget, else "
                 "differential_evolution with integrality",
    "multi_objective": "Tchebycheff scalarisation over 50 evenly spaced weights, each solved by the 'nlp' rule "
                       "with 1/50 of the budget",
}

_SHARED_FIXED = ("instance catalogue and split", "cell seeds", "evaluation budget", "wall-clock cap",
                 "scoring with the reference problem's callables", "verifier")

ARMS: tuple[Arm, ...] = (
    Arm("A", "hand-defined problem + fixed solver", "reference_problem",
        human_supplies=("problem (objective, constraints, bounds, variable types)",),
        llm_supplies=(),
        held_fixed=_SHARED_FIXED + ("solver: FIXED_SOLVER for every class",),
        solver="fixed", llm_calls_max=0),
    Arm("B", "hand-defined problem + rule-based solver selection", "reference_problem",
        human_supplies=("problem (objective, constraints, bounds, variable types)", "problem class label"),
        llm_supplies=(),
        held_fixed=_SHARED_FIXED + ("solver chosen by RULE_TABLE from the class label",),
        solver="rule_table", llm_calls_max=0),
    Arm("C", "LLM proposes candidate solutions directly", "nl_statement",
        human_supplies=("natural-language statement",),
        llm_supplies=("candidate points x; may read back each candidate's objective and violation",),
        held_fixed=_SHARED_FIXED + ("no solver: each candidate costs one evaluation",),
        solver="none", llm_calls_max=BUDGET.llm_calls_max),
    Arm("D", "LLM problem compiler + fixed solver", "nl_statement",
        human_supplies=("natural-language statement",),
        llm_supplies=("compiled problem in the IR (or an explicit rejection)",),
        held_fixed=_SHARED_FIXED + ("solver: FIXED_SOLVER, as arm A", "compile attempts: at most 3, schema-invalid "
                                    "output retried with the validator message"),
        solver="fixed", llm_calls_max=3),
    Arm("E", "LLM compiler + automatic iteration agent", "nl_statement",
        human_supplies=("natural-language statement", "list of hard constraints the agent may not change"),
        llm_supplies=("compiled problem", "solver choice, restarts and authorised reformulations between rounds"),
        held_fixed=_SHARED_FIXED + ("agent tool set and policy frozen before the first scored run",),
        solver="agent", llm_calls_max=BUDGET.llm_calls_max),
)

ARM_IDS: tuple[str, ...] = tuple(a.id for a in ARMS)

#: the pre-registered primary comparisons (treatment, control); Holm runs over these x metrics.PRIMARY_ENDPOINTS
PRIMARY_COMPARISONS: tuple[tuple[str, str], ...] = (("B", "A"), ("C", "A"), ("D", "A"), ("E", "D"), ("E", "B"))
