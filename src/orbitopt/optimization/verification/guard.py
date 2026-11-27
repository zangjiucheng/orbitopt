"""The independent verification guard (E14-06): derive each result status from evidence, never from claims.

``verify_result`` grades one ``OptimizationResult`` against the problem it
should answer. The result's four booleans are claims; the guard derives each
status itself and reports a claim it cannot confirm as a finding:

* ``solver_converged`` from the solver's termination reason;
* ``constraints_satisfied`` from its own re-evaluation of ``result.x``;
* ``independently_verified`` from caller-supplied evidence that names this
  problem and this result and comes from an allowed method other than the
  solver;
* ``optimality_certified`` from a certificate the guard computes itself.

Missing or mismatched evidence leaves a status not assessed, never passed,
and an error inside the guard is recorded and never read as a pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Any, Callable, Mapping, Sequence

from ..problem import STATUSES, OptimizationProblem, VerificationPolicy, _plain, schema_errors, validate_problem
from ..result import RESULT_SCHEMA, OptimizationResult, Termination
from . import certificates
from ._point import PointCheck, check_point
from .evidence import (
    CERTIFICATE_KIND,
    INDEPENDENT_KINDS,
    AlternateSolverCheck,
    AnalyticalOptimum,
    EvidenceItem,
    LowerBoundCertificate,
    ReferenceSolution,
    result_digest,
)
from .report import Finding, FindingCode, Outcome, StatusOutcome, VerificationReport, decide, unassessed_report

Tools = Mapping[str, Callable[..., Any]]
RE_EVALUATION = "guard:re-evaluation"
KKT = "guard:kkt"


@dataclass(frozen=True)
class Provenance:
    """How the result was produced, as recorded by the runtime that ran the solver.

    ``solved_problem`` is the problem the solver actually ran when it differs
    from the graded one (an agent revision, a compiled form). ``cache_hit``
    and ``cache_key`` record a cache lookup and the problem digest the cached
    entry was stored under.
    """

    solved_problem: OptimizationProblem | None = None
    cache_hit: bool = False
    cache_key: str | None = None


def verify_result(problem: OptimizationProblem, result: OptimizationResult | Mapping[str, Any], *,
                  policy: VerificationPolicy | None = None, evidence: Sequence[EvidenceItem] = (),
                  artifact_digest: str | None = None, provenance: Provenance | None = None,
                  tools: Tools | None = None, seed: int = 0) -> VerificationReport:
    """Grade ``result`` against ``problem`` and return the four statuses, the findings and a verdict.

    ``policy`` defaults to ``problem.verification``. ``result`` may be the
    dict form, which is schema-checked first. ``artifact_digest`` is the
    digest recorded for the result when it was stored; it must equal
    ``result_digest(result)``. ``tools`` resolves ``ToolCall`` objectives and
    constraints. ``seed`` fixes the samples a lower-bound certificate is
    checked on. Never raises: an internal error yields a failed verdict with
    a ``guard_error`` finding.
    """
    policy = problem.verification if policy is None else policy
    try:
        return _Guard(problem, policy, tuple(evidence), provenance or Provenance(), tools, seed).run(
            result, artifact_digest)
    except Exception as exc:  # the guard's own failure is never a pass
        return unassessed_report(f"guard error: {exc!r}", [Finding(FindingCode.GUARD_ERROR, "/", repr(exc))],
                                 tuple(getattr(policy, "required", ())))


class _Guard:
    def __init__(self, problem: OptimizationProblem, policy: VerificationPolicy, evidence: tuple,
                 provenance: Provenance, tools: Tools | None, seed: int):
        self.problem, self.policy, self.evidence = problem, policy, evidence
        self.provenance, self.tools, self.seed = provenance, tools, seed
        self.findings: list[Finding] = []

    def add(self, code: FindingCode, path: str, message: str) -> None:
        self.findings.append(Finding(code, path, message))

    def report(self, statuses: dict[str, StatusOutcome], digest: str | None, rdigest: str | None):
        found = tuple(self.findings)
        required = self.policy.required
        if not required:
            self.add(FindingCode.NO_REQUIREMENT, "/verification/required", "the policy requires no status")
            found = tuple(self.findings)
        return VerificationReport(decide(statuses, required, found), statuses, required, found, digest, rdigest)

    def run(self, raw: Any, artifact_digest: str | None) -> VerificationReport:
        required = tuple(self.policy.required)
        for error in validate_problem(_with_policy(self.problem, self.policy)):
            self.add(FindingCode.INVALID_PROBLEM, error.path, f"[{error.code.value}] {error.message}")
        if self.findings:
            return unassessed_report("the graded problem or policy is invalid", self.findings, required)
        digest = self.problem.digest()
        result = self.parse(raw)
        if result is None:
            return unassessed_report("the result is malformed", self.findings, required, digest)
        rdigest = result_digest(result)
        if artifact_digest is not None and artifact_digest != rdigest:
            self.add(FindingCode.DIGEST_MISMATCH, "/artifact_digest",
                     f"recorded digest {artifact_digest!r} does not match the result's digest {rdigest!r}")
        self.check_identity(result, digest)
        point = self.guarded("constraints_satisfied", lambda: self.point(result))
        statuses = {
            "solver_converged": self.status(lambda: self.converged(result)),
            "constraints_satisfied": self.status(lambda: _status(point)),
            "independently_verified": self.status(lambda: self.independent(result, digest, rdigest, point)),
            "optimality_certified": self.status(lambda: self.optimality(result, digest, rdigest, point)),
        }
        for name in STATUSES:
            if getattr(result, name) is True and statuses[name].outcome is not Outcome.PASSED:
                self.add(FindingCode.CLAIM_CONTRADICTED, f"/{name}",
                         f"the result claims {name} but the guard finds it {statuses[name].outcome.value}: "
                         f"{statuses[name].reason}")
        return self.report(statuses, digest, rdigest)

    def guarded(self, name: str, compute: Callable[[], Any]) -> Any:
        try:
            return compute()
        except Exception as exc:
            self.add(FindingCode.GUARD_ERROR, f"/{name}", f"guard error: {exc!r}")
            return StatusOutcome(Outcome.NOT_ASSESSED, f"guard error: {exc!r}")

    def status(self, compute: Callable[[], StatusOutcome]) -> StatusOutcome:
        try:
            return compute()
        except Exception as exc:
            self.add(FindingCode.GUARD_ERROR, "/", f"guard error: {exc!r}")
            return StatusOutcome(Outcome.NOT_ASSESSED, f"guard error: {exc!r}")

    # ------------------------------------------------------------------------------------------- result shape
    def parse(self, raw: Any) -> OptimizationResult | None:
        if isinstance(raw, Mapping):
            errors = schema_errors(dict(raw), RESULT_SCHEMA)
            for error in errors:
                self.add(FindingCode.MALFORMED_RESULT, error.path, error.message)
            if errors:
                return None
            raw = OptimizationResult.from_dict(raw)
        if not isinstance(raw, OptimizationResult):
            self.add(FindingCode.MALFORMED_RESULT, "/", f"expected an OptimizationResult, got {type(raw).__name__}")
            return None
        bad = [name for name in STATUSES if not isinstance(getattr(raw, name), bool)]
        bad += [name for name in ("x", "objective_values", "constraint_residuals")
                if not isinstance(getattr(raw, name), Mapping)]
        if not isinstance(raw.problem_digest, str) or not isinstance(raw.solver_id, str):
            bad.append("problem_digest/solver_id")
        for name in bad:
            self.add(FindingCode.MALFORMED_RESULT, f"/{name}", "field has the wrong type")
        try:
            result_digest(raw)
        except (TypeError, ValueError) as exc:
            self.add(FindingCode.MALFORMED_RESULT, "/", f"the result cannot be digested: {exc}")
            return None
        return None if bad else raw

    # ------------------------------------------------------------------------------------------- identity
    def check_identity(self, result: OptimizationResult, digest: str) -> None:
        solved = self.provenance.solved_problem
        claimed = digest if solved is None else solved.digest()
        if result.problem_digest != claimed:
            which = "the graded problem" if solved is None else "the problem the solver ran"
            self.add(FindingCode.PROBLEM_MISMATCH, "/problem_digest",
                     f"the result answers {result.problem_digest!r}, not {which} ({claimed!r})")
        if solved is not None and claimed != digest:
            compare_formulations(self.problem, solved, self.add)
        if self.provenance.cache_hit and self.provenance.cache_key != claimed:
            self.add(FindingCode.STALE_CACHE, "/provenance/cache_key",
                     f"cache hit on an entry stored under {self.provenance.cache_key!r}, not {claimed!r}")

    # ------------------------------------------------------------------------------------------- statuses
    def converged(self, result: OptimizationResult) -> StatusOutcome:
        if result.termination is Termination.CONVERGED:
            return StatusOutcome(Outcome.PASSED, "the solver reports termination 'converged'", ("result.termination",))
        self.add(FindingCode.NOT_CONVERGED, "/termination", f"termination is {result.termination.value!r}")
        return StatusOutcome(Outcome.FAILED, f"termination is {result.termination.value!r}", ("result.termination",))

    def point(self, result: OptimizationResult) -> PointCheck:
        check = check_point(self.problem, result.x, self.policy, self.tools, "/x", self.add)
        if check.objectives is not None:
            self.compare_claims(result, check)
        return check

    def compare_claims(self, result: OptimizationResult, check: PointCheck) -> None:
        """Claimed objective values and residuals must match the guard's own; NaN claims are findings."""
        by_constraint = {c.name: c for c in self.problem.constraints}
        groups = (("objective_values", check.objectives, lambda name, ref: self.policy.objective_rtol * max(1.0, abs(ref))),
                  ("constraint_residuals", check.residuals,
                   lambda name, ref: self.policy.constraint_tol * max(1.0, abs(by_constraint[name].rhs))))
        for field_name, computed, tolerance in groups:
            code = FindingCode.OBJECTIVE_MISMATCH if field_name == "objective_values" else FindingCode.RESIDUAL_MISMATCH
            for name, claim in getattr(result, field_name).items():
                path = f"/{field_name}/{name}"
                if name not in computed:
                    self.add(FindingCode.MALFORMED_RESULT, path, "names nothing in the graded problem")
                elif not isinstance(claim, (int, float)) or isinstance(claim, bool) or not math.isfinite(claim):
                    self.add(FindingCode.NON_FINITE, path, f"claimed value {claim!r} is not a finite number")
                elif math.isfinite(computed[name]) and abs(claim - computed[name]) > tolerance(name, computed[name]):
                    self.add(code, path, f"claimed {claim:.12g}, re-evaluated {computed[name]:.12g}")

    def independent(self, result: OptimizationResult, digest: str, rdigest: str, point: Any) -> StatusOutcome:
        for i, item in enumerate(result.evidence):
            if item.kind in INDEPENDENT_KINDS or item.kind == CERTIFICATE_KIND:
                self.add(FindingCode.SELF_ATTESTATION, f"/result/evidence/{i}",
                         f"{item.kind} evidence supplied by the result itself is a claim, not verification")
        agree: dict[str, list[str]] = {}
        disagree = []
        for item in self.evidence:
            if getattr(item, "kind", None) not in INDEPENDENT_KINDS:
                continue
            verdict = self.check_independent(item, result, digest, rdigest, point)
            if verdict is True:  # the item named a valid objective, or the problem has only one
                objective = item.objective if item.objective is not None else self.problem.objectives[0].name
                agree.setdefault(objective, []).append(item.id)
            elif verdict is False:
                disagree.append(item.id)
        if disagree:
            return StatusOutcome(Outcome.FAILED, "independent evidence contradicts the result", tuple(disagree))
        uncovered = [o.name for o in self.problem.objectives if o.name not in agree]
        ids = tuple(i for ids in agree.values() for i in ids)
        if not uncovered:
            return StatusOutcome(Outcome.PASSED, "independent evidence agrees with the re-evaluated result", ids)
        self.add(FindingCode.MISSING_EVIDENCE, "/evidence",
                 f"no evidence from an allowed independent method validates objectives {uncovered} of this result")
        return StatusOutcome(Outcome.NOT_ASSESSED, f"no independent evidence validates objectives {uncovered}", ids)

    def check_independent(self, item, result, digest, rdigest, point) -> bool | None:
        """True when the item validates and agrees, False when it validates and disagrees, None otherwise."""
        path = f"/evidence/{getattr(item, 'id', '?')}"
        try:
            if not self.pinned(item, result, digest, rdigest, path):
                return None
            objective = self.objective_name(item, path)
            if objective is None:
                return None
            if isinstance(item, AlternateSolverCheck):
                reference = self.alternate_value(item, result, digest, objective, path)
            else:
                reference = self.stated_value(item, objective, path)
            if reference is None:
                return None
            value = _objective(point, objective)
            if value is None:
                self.add(FindingCode.EVIDENCE_INCONCLUSIVE, path, "the result's objective could not be re-evaluated")
                return None
            tol = self.policy.objective_rtol * max(1.0, abs(reference))
            if isinstance(item, ReferenceSolution) and item.tolerance is not None:
                tol = min(tol, float(item.tolerance))
            if abs(value - reference) <= tol:
                return True
            better = _sense(self.problem, objective) * (value - reference) < 0
            if isinstance(item, AlternateSolverCheck) and better:
                self.add(FindingCode.EVIDENCE_INCONCLUSIVE, path,
                         f"the alternate solver found {reference:.12g}, worse than the result's {value:.12g}")
                return None
            self.add(FindingCode.EVIDENCE_DISAGREES, path,
                     f"{item.kind} gives {reference:.12g}, the re-evaluated result {value:.12g} (tolerance {tol:.3g})")
            return False
        except Exception as exc:  # an evidence item that raises never counts
            self.add(FindingCode.EVIDENCE_ERROR, path, f"checking the evidence raised {exc!r}")
            return None

    def pinned(self, item, result: OptimizationResult, digest: str, rdigest: str, path: str) -> bool:
        """The item's kind is allowed, its digests name this problem and result, and it is not the solver's."""
        if item.kind not in self.policy.independent_methods and item.kind != CERTIFICATE_KIND:
            self.add(FindingCode.EVIDENCE_NOT_ALLOWED, path, f"the policy does not allow {item.kind!r} evidence")
            return False
        for name, expected in (("problem_digest", digest), ("result_digest", rdigest)):
            value = getattr(item, name, None)
            if not value:
                self.add(FindingCode.EVIDENCE_MISMATCH, f"{path}/{name}", "missing digest")
                return False
            if value != expected:
                self.add(FindingCode.EVIDENCE_MISMATCH, f"{path}/{name}", f"names {value!r}, not {expected!r}")
                return False
        if item.producer == result.solver_id:
            self.add(FindingCode.FORGED_VERIFICATION, f"{path}/producer",
                     f"the evidence comes from the result's own solver {result.solver_id!r}")
            return False
        return True

    def objective_name(self, item, path: str) -> str | None:
        names = [o.name for o in self.problem.objectives]
        name = item.objective if item.objective is not None else (names[0] if len(names) == 1 else None)
        if name not in names:
            self.add(FindingCode.EVIDENCE_INVALID, f"{path}/objective",
                     f"names no objective of the problem (objectives: {names})")
            return None
        return name

    def stated_value(self, item: AnalyticalOptimum | ReferenceSolution, objective: str, path: str) -> float | None:
        value = float(item.value)
        if not math.isfinite(value):
            self.add(FindingCode.EVIDENCE_INVALID, f"{path}/value", f"{item.value!r} is not finite")
            return None
        if isinstance(item, ReferenceSolution):
            if not (isinstance(item.source, str) and item.source.strip()):
                self.add(FindingCode.EVIDENCE_INVALID, f"{path}/source", "a reference needs a source")
                return None
            if item.tolerance is not None and not (math.isfinite(float(item.tolerance)) and item.tolerance >= 0):
                self.add(FindingCode.EVIDENCE_INVALID, f"{path}/tolerance", "tolerance must be finite and >= 0")
                return None
            return value
        if not (isinstance(item.derivation, str) and item.derivation.strip()):
            self.add(FindingCode.EVIDENCE_INVALID, f"{path}/derivation", "an analytical optimum needs a derivation")
            return None
        if item.x is not None:
            check = check_point(self.problem, item.x, self.policy, self.tools, f"{path}/x", _ignore)
            attained = _objective(check, objective)
            if check.outcome is not Outcome.PASSED or attained is None or \
                    abs(attained - value) > self.policy.objective_rtol * max(1.0, abs(value)):
                self.add(FindingCode.EVIDENCE_INVALID, f"{path}/x",
                         "the derivation's optimiser is infeasible or does not attain the stated value")
                return None
        return value

    def alternate_value(self, item: AlternateSolverCheck, result: OptimizationResult, digest: str,
                        objective: str, path: str) -> float | None:
        if item.solver is None or not callable(item.solver):
            self.add(FindingCode.EVIDENCE_INVALID, f"{path}/solver", f"no solver supplied for {item.solver_id!r}")
            return None
        other = item.solver(self.problem)  # the guard runs it; a pasted-in result is never read
        if not isinstance(other, OptimizationResult):
            self.add(FindingCode.EVIDENCE_INVALID, path, f"the alternate solver returned {type(other).__name__}")
            return None
        if other.problem_digest != digest:
            self.add(FindingCode.EVIDENCE_MISMATCH, path, "the alternate solver answered another problem")
            return None
        if other.solver_id == result.solver_id or other.solver_id != item.solver_id:
            self.add(FindingCode.FORGED_VERIFICATION, path,
                     f"the alternate result comes from {other.solver_id!r}, which is not independent of "
                     f"{result.solver_id!r} or is not the declared {item.solver_id!r}")
            return None
        check = check_point(self.problem, other.x, self.policy, self.tools, f"{path}/x", _ignore)
        value = _objective(check, objective)
        if check.outcome is not Outcome.PASSED or value is None:
            self.add(FindingCode.EVIDENCE_INVALID, path, f"the alternate solver's point is not feasible: {check.reason}")
            return None
        return value

    def optimality(self, result: OptimizationResult, digest: str, rdigest: str, point: Any) -> StatusOutcome:
        bounds = [e for e in self.evidence if getattr(e, "kind", None) == CERTIFICATE_KIND]
        if not ("optimality_certified" in self.policy.required or result.optimality_certified is True or bounds):
            return StatusOutcome(Outcome.NOT_ASSESSED, "not requested by the policy, claimed or evidenced")
        if not isinstance(point, PointCheck) or point.outcome is not Outcome.PASSED:
            outcome = Outcome.FAILED if getattr(point, "outcome", None) is Outcome.FAILED else Outcome.NOT_ASSESSED
            return StatusOutcome(outcome, "optimality needs a feasible point", (RE_EVALUATION,))
        for cert in bounds:
            refuted = self.check_bound(cert, result, digest, rdigest, point)
            if refuted:
                return StatusOutcome(Outcome.FAILED, f"lower-bound certificate refuted: {refuted}", (cert.id,))
        kkt = certificates.kkt_certificate(self.problem, result.x, self.policy)
        if kkt.outcome is Outcome.NOT_ASSESSED:
            self.add(FindingCode.CERTIFICATE_UNSUPPORTED, "/optimality_certified", kkt.reason)
        elif kkt.outcome is Outcome.FAILED:
            self.add(FindingCode.CERTIFICATE_FAILED, "/optimality_certified", kkt.reason)
        return StatusOutcome(kkt.outcome, kkt.reason, (KKT,))

    def check_bound(self, cert: LowerBoundCertificate, result, digest, rdigest, point: PointCheck) -> str | None:
        path = f"/evidence/{getattr(cert, 'id', '?')}"
        try:
            if not self.pinned(cert, result, digest, rdigest, path):
                return None
            objective = self.objective_name(cert, path)
            if objective is None:
                return None
            if not (isinstance(cert.derivation, str) and cert.derivation.strip()):
                self.add(FindingCode.EVIDENCE_INVALID, f"{path}/derivation", "a bound needs a derivation")
                return None
            refuted = certificates.refute_lower_bound(self.problem, cert, objective, result.x,
                                                      point.objectives[objective], self.policy, self.tools, self.seed)
        except Exception as exc:
            self.add(FindingCode.EVIDENCE_ERROR, path, f"checking the certificate raised {exc!r}")
            return None
        if refuted:
            self.add(FindingCode.CERTIFICATE_FAILED, path, refuted)
        return refuted


def compare_formulations(reference: OptimizationProblem, solved: OptimizationProblem,
                         add: Callable[[FindingCode, str, str], None]) -> None:
    """Report how ``solved`` departs from ``reference`` in what the hard-constraint fingerprint protects.

    A widened domain, a dropped or unlocked user constraint or a loosened
    right-hand side is ``unauthorized_relaxation``; any other change to a
    user objective, locked constraint, variable or parameter is
    ``formulation_mismatch``. Revisions that keep the fingerprint (an added
    tightening constraint, a new initial guess) are noted, not condemned.
    """
    if reference.hard_constraint_fingerprint() == solved.hard_constraint_fingerprint():
        add(FindingCode.REVISED_PROBLEM, "/provenance/solved_problem",
            "the solver ran a revision that keeps every hard constraint; the point is graded against the original")
        return
    reported = []

    def report(code: FindingCode, path: str, message: str) -> None:
        reported.append(code)
        add(code, path, message)

    sv = {v.name: v for v in solved.variables}
    for v in reference.variables:
        other = sv.get(v.name)
        if other is None:
            report(FindingCode.FORMULATION_MISMATCH, f"/variables/{v.name}", "variable dropped")
        else:
            change = _domain_change(v, other)
            if change:
                report(change[0], f"/variables/{v.name}", change[1])
    for name in sorted(set(sv) - {v.name for v in reference.variables}):
        report(FindingCode.FORMULATION_MISMATCH, f"/variables/{name}", "variable added")
    sc = {c.name: c for c in solved.constraints}
    for c in (c for c in reference.constraints if c.locked):
        other = sc.get(c.name)
        if other is None:
            report(FindingCode.UNAUTHORIZED_RELAXATION, f"/constraints/{c.name}", "locked user constraint dropped")
            continue
        mine, theirs = _without_description(c), _without_description(other)
        if mine == theirs:
            continue
        loosened = (c.sense == "<=" and other.rhs > c.rhs) or (c.sense == ">=" and other.rhs < c.rhs)
        if {k for k in mine if mine[k] != theirs[k]} <= {"rhs", "locked"} and (loosened or not other.locked):
            report(FindingCode.UNAUTHORIZED_RELAXATION, f"/constraints/{c.name}",
                   f"locked user constraint relaxed: {c.sense} {c.rhs!r} became {other.sense} {other.rhs!r}"
                   + ("" if other.locked else ", unlocked"))
        else:
            report(FindingCode.FORMULATION_MISMATCH, f"/constraints/{c.name}", "locked user constraint changed")
    so = {o.name: o for o in solved.objectives}
    for o in (o for o in reference.objectives if o.origin == "user"):
        if o.name not in so or _without_description(o) != _without_description(so[o.name]):
            report(FindingCode.FORMULATION_MISMATCH, f"/objectives/{o.name}", "user objective replaced or changed")
    sp = {p.name: p for p in solved.parameters}
    for p in reference.parameters:
        if p.name not in sp or _without_description(p) != _without_description(sp[p.name]):
            report(FindingCode.FORMULATION_MISMATCH, f"/parameters/{p.name}", "parameter changed")
    if not reported:
        add(FindingCode.FORMULATION_MISMATCH, "/provenance/solved_problem", "the hard-constraint fingerprint differs")


_RELAXES = {"binary": ("integer", "continuous"), "integer": ("continuous",), "discrete": ("continuous",)}


def _domain_change(ref, other) -> tuple[FindingCode, str] | None:
    if ref.unit != other.unit:
        return FindingCode.FORMULATION_MISMATCH, f"unit {ref.unit!r} became {other.unit!r}"
    ref_lower, ref_upper = (0.0, 1.0) if ref.kind == "binary" else (ref.lower, ref.upper)
    lower, upper = (0.0, 1.0) if other.kind == "binary" else (other.lower, other.upper)
    widened = (ref_lower is not None and (lower is None or lower < ref_lower)) or \
              (ref_upper is not None and (upper is None or upper > ref_upper))
    if ref.kind == "discrete" and other.kind == "discrete":
        widened = not set(other.choices) <= set(ref.choices)
    if widened or other.kind in _RELAXES.get(ref.kind, ()):
        return FindingCode.UNAUTHORIZED_RELAXATION, (
            f"domain relaxed: {ref.kind} [{ref_lower}, {ref_upper}] became {other.kind} [{lower}, {upper}]")
    if (ref.kind, ref_lower, ref_upper, tuple(ref.choices)) != (other.kind, lower, upper, tuple(other.choices)):
        return FindingCode.FORMULATION_MISMATCH, "domain changed"
    return None


def _without_description(item) -> dict:
    return {f.name: _plain(getattr(item, f.name)) for f in fields(item) if f.name != "description"}


def _with_policy(problem: OptimizationProblem, policy: VerificationPolicy) -> OptimizationProblem:
    return problem if policy is problem.verification else replace(problem, verification=policy)


def _status(point: Any) -> StatusOutcome:
    if isinstance(point, StatusOutcome):
        return point
    return StatusOutcome(point.outcome, point.reason, (RE_EVALUATION,))


def _objective(check: Any, name: str) -> float | None:
    values = getattr(check, "objectives", None)
    if values is None or not math.isfinite(values.get(name, math.nan)):
        return None
    return values[name]


def _sense(problem: OptimizationProblem, objective: str) -> float:
    return 1.0 if next(o for o in problem.objectives if o.name == objective).sense == "minimize" else -1.0


def _ignore(*_: Any) -> None:
    return None
