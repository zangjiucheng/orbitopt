"""E10-06 (task #303): redesigned joint-chance admission and ranking.

The E7-05 diagnosis (task #276, ``docs/benchmarks/
e7_05_out_of_sample_reliability.md``) found the E7-01/E3-02 feasibility
gate -- reject every handoff whose worst normalized joint violation
exceeds ``JointChanceSpec.violation_tolerance`` -- rejecting 99 of 168
training-grid worlds of which 20 landed anyway under FL-305, on every
rejection channel (divert 51, fuel 48, probability 0): the binary gate
treats a 1 % divert overhang and a 60 % one alike, and the raw
``p_land`` it grades is itself miscalibrated (hold-out Brier skill
-0.619). This module replaces the gate with a *graded admission rule*
built on two things the old gate threw away:

* a **calibrated landing probability** :class:`LandingCalibration` --
  a ridge-regularized logistic model on the *physical* feasibility
  margins (fuel quantile z-margin ``phi_inv(p_fuel)`` and the divert
  violation ``g_div``), fit by Newton IRLS on the recorded E7-04 joint
  arm (seeds 1-4, ``docs/benchmarks/e7_04_failure_driven.json``). Two
  physical features only: monotone in each argument by fit (the E7-01
  constraint semantics -- no plan buys a higher probability by spending
  margin -- hold for the shipped coefficients), and it stays sane when
  the dispersion bands move (band-x1.5 ECE 0.033 against the raw
  model's 0.425), unlike a model keyed on ``logit p_raw``, which
  inherits the training band's overconfidence.
* **physical margin gates** in place of the single worst-violation
  cut: deep violations are rejected exactly as before (the task keeps
  them), but a violation is also rejected only when a *second* channel
  is at least borderline (two independent constraints binding is not a
  boundary artifact), or when the calibrated landing probability is
  below the campaign climatology (the row cannot beat a random draw of
  the same campaign, so the constraint earns nothing by relaxing).

Admission is conservative with respect to the old rule: every world the
old gate accepted is still accepted (the rule only ever un-rejects
rows it rejected), all 51 deep violations of the training grid stay
rejected, and the false-reject count (rejected but FL-305 landed)
falls 20 -> 10 with the 15 newly admitted worlds going 10 landed / 5
failed under the paired FL-305 baseline -- a 0.67 landing rate on
admitted violations against the 0.208 campaign climatology the floor
equals.

:class:`AdmissionPolicy` is the single decision surface:
:meth:`AdmissionPolicy.decide` grades a recorded row (reconstructing
its margins via :mod:`orbitopt.flightlab.recovery.reliability`),
:meth:`AdmissionPolicy.admits_verdict` grades a live
:class:`~orbitopt.flightlab.recovery.chance.JointVerdict` for FL-302
(``plan_boostback_entry(..., joint_admission=policy)``).
:func:`candidate_rank` ranks candidates by admissibility first, then a
hybrid probability -- the raw ``p_land`` where the old rule already
flew the burn and the raw surrogate is the validated quantity
(E7-02: consistent above 0.8), the calibrated ``p_hat`` exactly where
the raw surrogate is known to be wrong (E7-05: predicted 0.853,
achieved 0.185 in its [0.8, 0.9) band).

Determinism: :meth:`LandingCalibration.fit` is hand-rolled Newton IRLS
(no scipy), fixed iteration budget, tolerance below double noise, so
the coefficients are reproducible from the recorded data; the shipped
:data:`DEFAULT_CALIBRATION` is the exact fit on the recorded E7-04
``e7_04_joint_base`` arm (168 rows, features ``(fuel_z,
divert_violation)``, ridge 1e-2).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .chance import JointChanceSpec, JointVerdict, phi_inv

# reliability.joint_risk transitively import boostback, so the recorded-row
# reconstruction is imported lazily inside landing_achievable_metrics --
# this module is on boostback's own import chain (joint_admission).

__all__ = [
    "ADMISSION_GATES", "AdmissionPolicy", "DEFAULT_CALIBRATION",
    "DEFAULT_MIN_ADMISSION_PROBABILITY",
    "FEAT_DIVERT_VIOLATION", "FEAT_FUEL_Z", "FEAT_P_RAW_LOGIT",
    "FULL_FEATURES", "GATE_ADMITTED", "GATE_DEEP", "GATE_FLOOR",
    "GATE_SATISFIES", "GATE_SECOND", "LandingCalibration", "PHYSICAL_FEATURES",
    "auc", "candidate_rank", "landing_achievable_metrics", "regrade",
]

#: calibrated-model feature names
FEAT_FUEL_Z = "fuel_z"                     # phi_inv(p_fuel): sigma of fuel loss
FEAT_DIVERT_VIOLATION = "divert_violation"  # g_div, normalized, >= 0
FEAT_P_RAW_LOGIT = "p_raw_logit"           # logit of the raw p_land surrogate
PHYSICAL_FEATURES: tuple[str, ...] = (FEAT_FUEL_Z, FEAT_DIVERT_VIOLATION)
FULL_FEATURES: tuple[str, ...] = PHYSICAL_FEATURES + (FEAT_P_RAW_LOGIT,)

#: The admission probability floor. DESIGN ASSUMPTION (E10-06): the
#: recorded joint-arm climatology of the training campaign -- 35/168
#: = 0.208 landings in ``docs/benchmarks/e7_04_failure_driven.json``
#: arm ``e7_04_joint_base``. A violating world is admitted only if it
#: beats a random draw of the same campaign, so relaxing the
#: constraint cannot lower the expected success count.
DEFAULT_MIN_ADMISSION_PROBABILITY = 0.208

# admission gates (decide()["gate"]), in decision order
GATE_SATISFIES = "satisfies"               # old rule accepted: unchanged
GATE_DEEP = "deep_violation"               # worst > deep: rejected as before
GATE_SECOND = "second_channel"             # a second channel also binding
GATE_FLOOR = "probability_floor"           # calibrated p_hat below climatology
GATE_ADMITTED = "admitted"                 # violation tolerated by the rule
ADMISSION_GATES: tuple[str, ...] = (GATE_SATISFIES, GATE_DEEP, GATE_SECOND,
                                    GATE_FLOOR, GATE_ADMITTED)

_EPS = 1e-15  # keep phi_inv / logit away from their poles


# ---------------------------------------------------------------------------
# the calibrated landing probability
# ---------------------------------------------------------------------------


def _solve_small(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting (k <= 4 systems)."""
    k = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-300:
            raise ZeroDivisionError("singular Newton system")
        m[col], m[piv] = m[piv], m[col]
        f = m[col][col]
        m[col] = [v / f for v in m[col]]
        for r in range(k):
            if r != col and m[r][col] != 0.0:
                g = m[r][col]
                m[r] = [vr - g * vc for vr, vc in zip(m[r], m[col])]
    return [m[i][k] for i in range(k)]


@dataclass(frozen=True)
class LandingCalibration:
    """Ridge-regularized logistic map from feasibility margins to the
    probability the joint campaign records a landing, ``P(landed)``.

    ``coefficients`` is ``[intercept, b_f1, b_f2, ...]`` in the order of
    ``features``. :meth:`fit` is deterministic: Newton IRLS, ridge on
    every coefficient, linear predictor clamped to +/-30, 100
    iterations and a 1e-10 step tolerance (convergence far below double
    rounding at these data sizes).
    """
    features: tuple[str, ...]
    coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.features:
            raise ValueError("at least one feature is required")
        if len(self.coefficients) != len(self.features) + 1:
            raise ValueError("coefficients must carry an intercept plus one weight per feature")
        unknown = [f for f in self.features if f not in FULL_FEATURES]
        if unknown:
            raise ValueError(f"unknown calibration features: {unknown}")

    # -- scoring ----------------------------------------------------------

    def probability(self, metrics: Mapping[str, float]) -> float:
        eta = float(self.coefficients[0])
        for c, f in zip(self.coefficients[1:], self.features):
            eta += c * float(metrics[f])
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, eta))))

    # -- fitting ----------------------------------------------------------

    @classmethod
    def fit(cls, rows: Iterable[Mapping[str, Any]], spec: JointChanceSpec,
            features: Sequence[str] = PHYSICAL_FEATURES, *,
            ridge: float = 1e-2, max_iter: int = 100, tol: float = 1e-10,
            ) -> "LandingCalibration":
        """Fit on joint-mode recorded rows (``constraint_violations``
        reconstructable, ``landing_probability`` present). Rows the
        reconstruction rejects are skipped; labels are the recorded
        ``outcome`` (``landed`` = 1). The label is deliberately the
        *campaign-recorded* outcome, censoring by the current gate
        included: the decision the policy makes is "does this world
        land under campaign grading", and the conservatism that adds
        to violating rows is in the admission direction (a violating
        world's true achievable probability is its paired-FL-305 rate,
        measured at 0.67 on newly admitted rows -- above the floor).
        """
        feats = tuple(features)
        for f in feats:
            if f not in FULL_FEATURES:
                raise ValueError(f"unknown calibration feature: {f!r}")
        X: list[list[float]] = []
        y: list[float] = []
        for row in rows:
            m = landing_achievable_metrics(row, spec)
            if m is None:
                continue
            X.append([1.0] + [m[f] for f in feats])
            y.append(1.0 if row.get("outcome") == "landed" else 0.0)
        if not X:
            raise ValueError("fit needs at least one reconstructable joint-mode row")
        if all(v == 0.0 for v in y) or all(v == 1.0 for v in y):
            raise ValueError("fit needs both landed and failed rows")
        k = len(X[0])
        w = [0.0] * k
        for _ in range(max_iter):
            grad = [0.0] * k
            hess = [[0.0 if a != b else ridge for b in range(k)] for a in range(k)]
            for xi, yi in zip(X, y):
                eta = max(-30.0, min(30.0, sum(a * b for a, b in zip(w, xi))))
                p = 1.0 / (1.0 + math.exp(-eta))
                wt = p * (1.0 - p) + 1e-6
                r = yi - p
                for a in range(k):
                    grad[a] += xi[a] * r
                    for b in range(k):
                        hess[a][b] += xi[a] * xi[b] * wt
            grad = [g - ridge * wa for g, wa in zip(grad, w)]
            step = _solve_small(hess, grad)
            w = [wa + s for wa, s in zip(w, step)]
            if max(abs(s) for s in step) < tol:
                break
        return cls(features=feats, coefficients=tuple(float(v) for v in w))

    # -- (de)serialization --------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {"features": list(self.features),
                "coefficients": [float(c) for c in self.coefficients]}

    @staticmethod
    def from_mapping(d: Mapping[str, Any]) -> "LandingCalibration":
        return LandingCalibration(features=tuple(str(f) for f in d["features"]),
                                  coefficients=tuple(float(c) for c in d["coefficients"]))


#: Shipped default: :meth:`LandingCalibration.fit` on the 168 recorded
#: ``e7_04_joint_base`` rows (docs/benchmarks/e7_04_failure_driven.json),
#: features ``("fuel_z", "divert_violation")``, ridge 1e-2. Both physical
#: weights carry their constraint-sane signs (+fuel_z: more fuel margin
#: can only help; -divert_violation: divert overrun can only hurt).
#: Regenerating this line from the recorded file reproduces every digit;
#: the E10-06 benchmark refits it and asserts agreement.
DEFAULT_CALIBRATION = LandingCalibration(
    features=PHYSICAL_FEATURES,
    coefficients=(-4.120438063536452, 2.733026209007225, -22.256139996971008))


def landing_achievable_metrics(record: Mapping[str, Any],
                               spec: JointChanceSpec) -> dict[str, Any] | None:
    """Joint-feasibility view of one recorded row: the three normalized
    violations (via :func:`orbitopt.flightlab.recovery.reliability.
    constraint_violations`), the calibration feature values, and the
    admission-gate inputs. ``None`` for non-joint rows (the E7-05
    reconstruction refuses them)."""
    from .reliability import constraint_violations
    v = constraint_violations(record, spec)
    if v is None:
        return None
    g_fuel, g_div, g_prob = v["violations"]
    p_fuel = min(max(v["p_fuel"], _EPS), 1.0 - _EPS)
    p_raw = min(max(float(record["landing_probability"]), _EPS), 1.0 - _EPS)
    worst = v["worst"]
    # second-highest *including duplicates*: two channels tying at the
    # worst value are two independent channels at that level
    second = sorted(v["violations"], reverse=True)[1]
    return {
        "p_fuel": v["p_fuel"], "p_divert": v["p_divert"], "p_raw": p_raw,
        "violations": v["violations"], "worst": worst, "second_worst": second,
        "consistent": v["consistent"],
        FEAT_FUEL_Z: phi_inv(p_fuel),
        FEAT_DIVERT_VIOLATION: g_div,
        FEAT_P_RAW_LOGIT: math.log(p_raw / (1.0 - p_raw)),
    }


# ---------------------------------------------------------------------------
# the admission policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdmissionPolicy:
    """Graded replacement for the binary ``satisfies()`` gate (E10-06).

    In decision order, applied only to a crossed handoff the old gate
    would reject (worst violation > ``spec.violation_tolerance``):

    1. ``worst > deep_violation``    -> reject (E7-05: the deep layer is
       the spec-vs-band disagreement, not a boundary call; all 51 deep
       training-grid rejects stay rejected).
    2. ``second_worst > second_violation_max`` -> reject: two channels
       at least borderline means the handoff fails the joint constraint
       in two independent ways.
    3. ``p_hat < min_admission_probability`` -> reject: the calibrated
       landing probability does not beat the campaign climatology the
       floor equals.
    4. otherwise admit.

    Rows the old gate accepted are untouched (gate ``satisfies``), so
    the policy can only ever turn rejections into acceptances, never
    the reverse.
    """
    calibration: LandingCalibration = DEFAULT_CALIBRATION
    # mirrors reliability.DEFAULT_DEEP_VIOLATION / DEFAULT_BORDERLINE_VIOLATION
    deep_violation: float = 0.5
    second_violation_max: float = 0.15
    min_admission_probability: float = DEFAULT_MIN_ADMISSION_PROBABILITY

    def _gate(self, spec: JointChanceSpec, metrics: Mapping[str, Any]) -> dict[str, Any]:
        worst = float(metrics["worst"])
        second = float(metrics["second_worst"])
        p_hat = self.calibration.probability(metrics)
        if worst <= spec.violation_tolerance:
            gate, admitted = GATE_SATISFIES, True
        elif worst > self.deep_violation:
            gate, admitted = GATE_DEEP, False
        elif second > self.second_violation_max:
            gate, admitted = GATE_SECOND, False
        elif p_hat < self.min_admission_probability:
            gate, admitted = GATE_FLOOR, False
        else:
            gate, admitted = GATE_ADMITTED, True
        return {"admitted": admitted, "gate": gate, "worst": worst,
                "second_worst": second, "calibrated_probability": p_hat,
                "p_raw": float(metrics["p_raw"])}

    def decide(self, record: Mapping[str, Any],
               spec: JointChanceSpec) -> dict[str, Any] | None:
        """Grade one recorded joint-mode row; ``None`` for rows the joint
        reconstruction refuses (not joint-mode data)."""
        m = landing_achievable_metrics(record, spec)
        if m is None:
            return None
        return self._gate(spec, m)

    def admits_verdict(self, spec: JointChanceSpec,
                       verdict: JointVerdict) -> bool:
        """Live-plan counterpart of :meth:`decide`, for FL-302: grade a
        crossed handoff's :class:`JointVerdict` under the policy."""
        p_fuel = min(max(verdict.p_fuel, _EPS), 1.0 - _EPS)
        p_raw = min(max(verdict.p_land, _EPS), 1.0 - _EPS)
        second = sorted(verdict.violations, reverse=True)[1]
        metrics = {"worst": verdict.worst_violation, "second_worst": second,
                   "p_raw": p_raw, FEAT_FUEL_Z: phi_inv(p_fuel),
                   FEAT_DIVERT_VIOLATION: verdict.violations[1]}
        return bool(self._gate(spec, metrics)["admitted"])

    # -- (de)serialization --------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {"calibration": self.calibration.to_dict(),
                "deep_violation": float(self.deep_violation),
                "second_violation_max": float(self.second_violation_max),
                "min_admission_probability": float(self.min_admission_probability)}

    @staticmethod
    def from_option(option: Any) -> "AdmissionPolicy":
        """Coerce the ``joint_admission`` plan option: an
        :class:`AdmissionPolicy` passes through; a mapping carries the
        dataclass fields (``calibration`` as a mapping or
        :class:`LandingCalibration`); ``False``/``None`` is handled by
        the caller (no policy = the old binary gate)."""
        if isinstance(option, AdmissionPolicy):
            return option
        d = dict(option)
        calib = d.pop("calibration", None)
        if calib is None:
            calib = DEFAULT_CALIBRATION
        elif not isinstance(calib, LandingCalibration):
            calib = LandingCalibration.from_mapping(calib)
        known = {"deep_violation", "second_violation_max", "min_admission_probability"}
        unknown = [k for k in d if k not in known]
        if unknown:
            raise ValueError(f"unknown joint_admission options: {sorted(unknown)}")
        return AdmissionPolicy(calibration=calib, **d)


# ---------------------------------------------------------------------------
# offline analysis: regrade, ranking, AUC
# ---------------------------------------------------------------------------


def _baseline_map(baseline_rows: Iterable[Mapping[str, Any]] | None
                  ) -> dict[tuple[Any, Any], Mapping[str, Any]]:
    return {} if baseline_rows is None else \
        {(r.get("seed"), r.get("idx")): r for r in baseline_rows}


def regrade(joint_rows: Sequence[Mapping[str, Any]], spec: JointChanceSpec,
            policy: AdmissionPolicy, *, reject_reason: str = "JOINT_CHANCE_VIOLATED",
            baseline_rows: Sequence[Mapping[str, Any]] | None = None,
            ) -> dict[str, Any]:
    """Before/after of :class:`AdmissionPolicy` over a recorded joint
    campaign. A *recorded rejection* is a row whose governing reason is
    ``reject_reason`` (rows rejected by an earlier physical check are
    not joint-gate decisions and are unaffected). With
    ``baseline_rows`` (the paired no-gate FL-305 campaign, keyed by
    ``(seed, idx)``) the count of *false rejections* -- joint-rejected
    rows that landed under FL-305 -- is reported before and after,
    with the newly admitted rows split by that paired outcome. Also
    proves the superset property: ``previously_accepted_now_rejected``
    counts rows the old gate accepted that the policy would reject
    (structurally 0; asserted)."""
    base = _baseline_map(baseline_rows)
    old_rejected: list[Mapping[str, Any]] = []
    newly_admitted: list[Mapping[str, Any]] = []
    gates: Counter[str] = Counter()
    severity: Counter[str] = Counter()
    now_rejected = 0
    unjoint = 0
    for row in joint_rows:
        decision = policy.decide(row, spec)
        if decision is None:
            unjoint += 1
            continue
        if decision["worst"] <= spec.violation_tolerance:
            if not decision["admitted"]:
                now_rejected += 1
            continue
        if row.get("reason") != reject_reason:
            continue  # governed by an earlier physical check
        old_rejected.append(row)
        gates[decision["gate"]] += 1
        w = decision["worst"]
        severity["deep" if w > policy.deep_violation
                 else ("mid" if w > policy.second_violation_max else "near")] += 1
        if decision["admitted"]:
            newly_admitted.append(row)

    def _fl305(row: Mapping[str, Any]) -> str | None:
        b = base.get((row.get("seed"), row.get("idx")))
        return None if b is None else str(b.get("outcome"))

    def _false(rows: Iterable[Mapping[str, Any]]) -> int:
        return sum(1 for r in rows if _fl305(r) == "landed")

    admitted_set = {id(r) for r in newly_admitted}
    remaining = [r for r in old_rejected if id(r) not in admitted_set]
    out: dict[str, Any] = {
        "old_rejections": len(old_rejected),
        "new_rejections": len(remaining),
        "newly_admitted": len(newly_admitted),
        "previously_accepted_now_rejected": now_rejected,
        "non_joint_rows": unjoint,
        "rejection_gates": {g: int(gates[g]) for g in ADMISSION_GATES if gates[g]},
        "new_rejection_severity": {s: int(severity[s]) for s in ("near", "mid", "deep")
                                   if severity[s]},
        "deep_rejections_kept": int(gates[GATE_DEEP]),
    }
    if baseline_rows is not None:
        old_false = _false(old_rejected)
        new_false = _false(remaining)
        out["false_rejections_old"] = old_false
        out["false_rejections_new"] = new_false
        out["newly_admitted_by_paired_baseline"] = {
            "landed": _false(newly_admitted),
            "failed": sum(1 for r in newly_admitted if _fl305(r) == "failed"),
        }
        out["false_rejections_kept_by_gate"] = {
            gate: sum(1 for r in remaining
                      if policy.decide(r, spec)["gate"] == gate and _fl305(r) == "landed")
            for gate in (GATE_DEEP, GATE_SECOND, GATE_FLOOR)}
    return out


def candidate_rank(rows: Sequence[Mapping[str, Any]], spec: JointChanceSpec,
                   policy: AdmissionPolicy) -> list[dict[str, Any]]:
    """Rank joint-mode candidates the E10-06 way: admissible first,
    then by the hybrid probability -- the raw ``p_land`` where the old
    rule already flew the burn (E7-02 validated it there) and the
    calibrated ``p_hat`` exactly where the old rule rejected the world
    and E7-05 measured the surrogate wrong -- then by how close the
    handoff is to the corridor and by the fuel quantile margin.

    Rows the joint reconstruction refuses are skipped. Ties break
    deterministically on ``(index)``.
    """
    entries: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        decision = policy.decide(row, spec)
        if decision is None:
            continue
        m = landing_achievable_metrics(row, spec)
        assert m is not None
        score = decision["p_raw"] if decision["gate"] == GATE_SATISFIES \
            else decision["calibrated_probability"]
        entries.append({"index": i, "seed": row.get("seed"), "idx": row.get("idx"),
                        "admitted": decision["admitted"], "gate": decision["gate"],
                        "probability": score,
                        "p_hat": decision["calibrated_probability"],
                        "p_raw": decision["p_raw"],
                        "worst": decision["worst"],
                        "fuel_z_margin": m[FEAT_FUEL_Z]})
    entries.sort(key=lambda e: (not e["admitted"], -e["probability"], e["worst"],
                                -e["fuel_z_margin"], e["index"]))
    return entries


def auc(scores: Sequence[float], labels: Sequence[int | float]) -> float:
    """Mann-Whitney U AUC with midrank ties; ``labels`` are 0/1."""
    n = len(scores)
    if n != len(labels) or not n:
        raise ValueError("scores and labels must align and be non-empty")
    order = sorted(range(n), key=lambda i: scores[i])
    rank = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            rank[order[k]] = avg
        i = j + 1
    n1 = sum(int(l) for l in labels)
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        raise ValueError("AUC needs both classes present")
    s1 = sum(rank[i] for i in range(n) if int(labels[i]) == 1)
    return (s1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)
