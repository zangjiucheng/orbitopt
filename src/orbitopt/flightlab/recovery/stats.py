"""R-06 (task #238): statistical reading of the recovery Monte Carlo.

FL-304 reports a success rate plus a reason-code histogram, which answers *how
many* failed but not *how much you should trust that number* nor *which
perturbation is doing the damage*. R-06 adds three readings on top of the very
same per-sample records, so nothing here changes what is flown and a fixed
seed still reproduces the report byte-for-byte:

* **failure classes** -- every non-landed sample is attributed to one family:
  the planner found no feasible plan (``planning_infeasible``), the plan was
  feasible but flying it broke down (``execution_failure``), the booster
  touched down outside the landing limits (``landing_accuracy``), or the
  sample ran short of propellant (``fuel_insufficient``). A reason code the
  classifier does not know lands in ``unclassified``, which is reported as
  ``0`` like any other class -- never silently dropped (acceptance: every
  failure carries a class);
* **confidence interval** -- a Wilson score interval for the success rate.
  Samples are independent by construction (one ``[seed, idx]`` stream per
  sample), so the binomial interval is the honest statement of what a run of
  ``n`` samples supports;
* **perturbation sensitivity** -- per source, how strongly that source's own
  draw (in sigma-equivalents of its declared distribution) associates with
  failure. This is an association over the joint dispersion, not a variance
  decomposition: sources are dispersed together, so a confounded source can
  inherit its neighbour's effect. The one-at-a-time reading (each source
  re-flown alone, see
  :func:`~orbitopt.flightlab.recovery.montecarlo.run_monte_carlo`'s
  ``sensitivity="solo"``) is what isolates a source.

Determinism: everything is a pure function of the record list and the
declared model, computed with fixed numpy/stdlib arithmetic -- no RNG use.

    from orbitopt.flightlab.recovery.stats import classify_failure
    classify_failure("INSUFFICIENT_PROPELLANT_BOOSTBACK")   # 'fuel_insufficient'
    classify_failure("LANDING_POSITION_OUT_OF_TOLERANCE")   # 'landing_accuracy'
"""
from __future__ import annotations

from math import sqrt
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np

from orbitopt.flightlab.reasons import normalize_code
from orbitopt.flightlab.recovery.uncertainty import SOURCE_ORDER, UncertaintyModel

__all__ = [
    "FAILURE_CLASS_FUEL_INSUFFICIENT",
    "FAILURE_CLASS_LANDING_ACCURACY",
    "FAILURE_CLASS_PLANNING_INFEASIBLE",
    "FAILURE_CLASS_EXECUTION_FAILURE",
    "FAILURE_CLASS_UNCLASSIFIED",
    "FAILURE_CLASSES",
    "classify_failure",
    "failure_breakdown",
    "failure_class_title",
    "solo_models",
    "source_sensitivity",
    "statistics_summary",
    "success_rate_interval",
]

FAILURE_CLASS_PLANNING_INFEASIBLE = "planning_infeasible"
FAILURE_CLASS_EXECUTION_FAILURE = "execution_failure"
FAILURE_CLASS_LANDING_ACCURACY = "landing_accuracy"
FAILURE_CLASS_FUEL_INSUFFICIENT = "fuel_insufficient"
FAILURE_CLASS_UNCLASSIFIED = "unclassified"

#: Display order, and the full vocabulary :func:`failure_breakdown` reports
#: (an absent class reads as 0, never as "did not happen").
FAILURE_CLASSES: tuple[str, ...] = (
    FAILURE_CLASS_PLANNING_INFEASIBLE,
    FAILURE_CLASS_EXECUTION_FAILURE,
    FAILURE_CLASS_LANDING_ACCURACY,
    FAILURE_CLASS_FUEL_INSUFFICIENT,
    FAILURE_CLASS_UNCLASSIFIED,
)

_TITLES: Mapping[str, str] = {
    FAILURE_CLASS_PLANNING_INFEASIBLE: "Planning infeasible",
    FAILURE_CLASS_EXECUTION_FAILURE: "Execution failure",
    FAILURE_CLASS_LANDING_ACCURACY: "Landing accuracy",
    FAILURE_CLASS_FUEL_INSUFFICIENT: "Insufficient propellant",
    FAILURE_CLASS_UNCLASSIFIED: "Unclassified",
}

#: Every reason code the recovery chain can hand a Monte Carlo sample, mapped
#: to its family. Propellant shortfalls win over the generic infeasibility of
#: the plan that reported them -- "add reserve" is the actionable reading of
#: ``INSUFFICIENT_PROPELLANT_BOOSTBACK``, not "the planner gave up".
_CODE_CLASS: Mapping[str, str] = {
    # planning could not produce a flyable chain
    "RECOVERY_INFEASIBLE": FAILURE_CLASS_PLANNING_INFEASIBLE,
    "CORRIDOR_NOT_REACHED": FAILURE_CLASS_PLANNING_INFEASIBLE,
    "CORRIDOR_MISSED": FAILURE_CLASS_PLANNING_INFEASIBLE,
    "LANDING_INFEASIBLE": FAILURE_CLASS_PLANNING_INFEASIBLE,
    # the plan flew, but the flight did not complete
    "SAMPLE_ERROR": FAILURE_CLASS_EXECUTION_FAILURE,
    "RECOVERY_LANDING_NOT_ASSESSED": FAILURE_CLASS_EXECUTION_FAILURE,
    "RECOVERY_LANDING_NOT_ATTEMPTED": FAILURE_CLASS_EXECUTION_FAILURE,
    "LANDING_NO_TOUCHDOWN": FAILURE_CLASS_EXECUTION_FAILURE,
    "ENTRY_Q_LIMIT_EXCEEDED": FAILURE_CLASS_EXECUTION_FAILURE,
    "ENTRY_G_LIMIT_EXCEEDED": FAILURE_CLASS_EXECUTION_FAILURE,
    "HANDOFF_STATE_DISCONTINUITY": FAILURE_CLASS_EXECUTION_FAILURE,
    "HANDOFF_MASS_NOT_CONSERVED": FAILURE_CLASS_EXECUTION_FAILURE,
    # it landed, outside the acceptance limits
    "LANDING_FAILED": FAILURE_CLASS_LANDING_ACCURACY,
    "LANDING_POSITION_OUT_OF_TOLERANCE": FAILURE_CLASS_LANDING_ACCURACY,
    "LANDING_VERTICAL_SPEED_EXCEEDED": FAILURE_CLASS_LANDING_ACCURACY,
    "LANDING_LATERAL_SPEED_EXCEEDED": FAILURE_CLASS_LANDING_ACCURACY,
    "LANDING_TILT_EXCEEDED": FAILURE_CLASS_LANDING_ACCURACY,
    "LANDING_MAX_G_EXCEEDED": FAILURE_CLASS_LANDING_ACCURACY,
    # propellant, wherever it was reported short
    "INSUFFICIENT_PROPELLANT_BOOSTBACK": FAILURE_CLASS_FUEL_INSUFFICIENT,
    "INSUFFICIENT_PROPELLANT_ENTRY_BURN": FAILURE_CLASS_FUEL_INSUFFICIENT,
    "INSUFFICIENT_PROPELLANT_RESERVE": FAILURE_CLASS_FUEL_INSUFFICIENT,
    "LANDING_PROPELLANT_RESERVE_LOW": FAILURE_CLASS_FUEL_INSUFFICIENT,
    "PROPELLANT_RESERVE_LOW": FAILURE_CLASS_FUEL_INSUFFICIENT,
    "PROPELLANT_EXHAUSTED": FAILURE_CLASS_FUEL_INSUFFICIENT,
    "INFEASIBLE_PROPELLANT_RESERVE": FAILURE_CLASS_FUEL_INSUFFICIENT,
}

#: Reason codes carrying a runtime detail (``SAMPLE_ERROR:ZeroDivisionError``).
_PREFIX_CLASS: tuple[tuple[str, str], ...] = (("SAMPLE_ERROR",
                                               FAILURE_CLASS_EXECUTION_FAILURE),)


def failure_class_title(cls: str) -> str:
    """Short human title of an R-06 failure class; an unknown class name
    comes back as-is."""
    return _TITLES.get(cls, cls)


def classify_failure(reason: str) -> str:
    """The R-06 family of one failing sample's governing reason code. Total
    by construction: a code outside the mapping (including the success code
    ``LANDED`` and an empty code, neither of which is a failure) is reported
    as ``unclassified`` rather than dropped -- :func:`failure_breakdown` only
    asks this for non-landed records."""
    code = normalize_code(reason)
    if not code:
        return FAILURE_CLASS_UNCLASSIFIED
    cls = _CODE_CLASS.get(code)
    if cls is None:
        cls = next((c for p, c in _PREFIX_CLASS if code.startswith(p)), None)
    if cls is None and "PROPELLANT" in code:
        cls = FAILURE_CLASS_FUEL_INSUFFICIENT
    if cls is None and code.startswith("INFEASIBLE"):
        cls = FAILURE_CLASS_PLANNING_INFEASIBLE
    return cls or FAILURE_CLASS_UNCLASSIFIED


def _is_landed(rec: Mapping[str, Any]) -> bool:
    return rec.get("outcome") == "landed" or rec.get("reason") == "LANDED"


def failure_breakdown(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Classify every non-landed record: counts per family (always all five
    classes, so a zero is an explicit zero) plus the reason codes behind each
    family. Every failure is counted exactly once, so ``n_failures`` equals
    ``n - landed`` and the classes sum to it."""
    classes = {c: 0 for c in FAILURE_CLASSES}
    codes: dict[str, dict[str, int]] = {}
    n_fail = 0
    for rec in samples:
        if _is_landed(rec):
            continue
        code = str(rec.get("reason") or "") or "(no reason code)"
        cls = classify_failure(code)
        classes[cls] += 1
        n_fail += 1
        per = codes.setdefault(cls, {})
        per[code] = per.get(code, 0) + 1
    return {"n_samples": len(samples), "n_failures": n_fail,
            "classes": classes,
            "codes_by_class": {c: dict(sorted(codes[c].items()))
                               for c in FAILURE_CLASSES if c in codes}}


def success_rate_interval(successes: int, n: int,
                          level: float = 0.95) -> dict[str, Any]:
    """Wilson score interval for a binomial success rate -- the reading R-06
    reports instead of a bare percentage. Chosen over the normal (Wald)
    interval because it stays inside [0, 1] and stays informative at
    ``k = 0`` and ``k = n``, which a small recovery Monte Carlo hits often.
    ``standard_error`` is the plain Wald SE, reported alongside so a reader
    can see how much of the interval is small-sample Wilson widening."""
    n = int(n)
    k = int(successes)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not 0 <= k <= n:
        raise ValueError(f"successes must be within [0, n], got {k} of {n}")
    if not 0.0 < float(level) < 1.0:
        raise ValueError(f"level must be inside (0, 1), got {level}")
    z = NormalDist().inv_cdf(1.0 - (1.0 - float(level)) / 2.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    lower, upper = centre - half, centre + half
    if k == 0:
        lower = 0.0  # exact for k = 0; floating point would leave ~1e-17 there
    if k == n:
        upper = 1.0
    return {"method": "wilson_score", "confidence_level": float(level),
            "successes": k, "n": n, "point": p,
            "standard_error": sqrt(p * (1.0 - p) / n),
            "lower": max(0.0, lower), "upper": min(1.0, upper)}


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank) -- the tie handling Spearman
    needs, and the reason the result is stable rather than order-dependent."""
    order = np.argsort(values, kind="stable")
    ranked = np.empty(values.size, dtype=np.float64)
    ordered = values[order]
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and ordered[j + 1] == ordered[i]:
            j += 1
        ranked[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranked


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    """Spearman rank correlation, or None when either side is constant (no
    information to correlate -- reported as ``uninformative``, never as 0.0,
    which would read as "measured and found nothing")."""
    if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return None
    rx, ry = _ranks(x), _ranks(y)
    sx, sy = float(rx.std()), float(ry.std())
    if sx <= 0.0 or sy <= 0.0:
        return None
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


#: |z| above which a sample counts as "perturbed" for the tail split.
TAIL_SIGMA = 1.0


def _z_scores(sample: Mapping[str, Any], scales: Sequence[Any]) -> float | None:
    """The sample's perturbation of one source in sigma-equivalents (RMS over
    the source's random components), or None when a needed draw is missing."""
    acc, count = 0.0, 0
    for comp in scales:
        if comp.sigma_eq is None:
            continue
        raw = sample.get(comp.draw_key)
        if raw is None:
            return None
        z = (float(raw) - comp.neutral) / comp.sigma_eq
        acc += z * z
        count += 1
    return sqrt(acc / count) if count else None


def _signed_z(sample: Mapping[str, Any], scales: Sequence[Any]) -> float | None:
    """Directional z for a single-component source (a negative thrust factor
    and a positive one do not do the same thing); None for multi-component
    sources, whose direction is not one number."""
    movers = [c for c in scales if c.sigma_eq is not None]
    if len(movers) != 1:
        return None
    raw = sample.get(movers[0].draw_key)
    if raw is None:
        return None
    return (float(raw) - movers[0].neutral) / movers[0].sigma_eq


def source_sensitivity(samples: Sequence[Mapping[str, Any]],
                       model: UncertaintyModel,
                       threshold_sigma: float = TAIL_SIGMA) -> dict[str, Any]:
    """Per perturbation source, how much its own draw explains failure across
    the flown samples (module docstring: association under joint dispersion).

    Per active source: mean |z| overall / among landed / among failed (a
    failure population with a larger mean |z| than the landed one is a source
    whose extremes hurt), the failure rate above and below one sigma-equivalent
    of it, and the Spearman rank correlation between its |z| and the failure
    indicator (plus the signed one for the single-axis factor sources).
    ``ranking`` orders the sources by that rank correlation, ties broken by
    source name so the report is stable."""
    scales = model.component_scales()
    landed = np.array([_is_landed(s) for s in samples], dtype=bool)
    failed = (~landed).astype(np.float64)
    out: dict[str, Any] = {"state": "uninformative",
                           "method": "Spearman rank correlation of the "
                                     "sample's |z| (sigma-equivalents of its "
                                     "declared distribution) against failure, "
                                     "under joint dispersion",
                           "tail_threshold_sigma": float(threshold_sigma),
                           "sources": {}, "ranking": []}
    scored: list[tuple[float, str]] = []
    for name in SOURCE_ORDER:
        comps = scales[name]
        if not any(c.sigma_eq is not None for c in comps):
            out["sources"][name] = {"state": "inactive_or_bias_only",
                                    "n": 0}
            continue
        zs, signed = [], []
        kept = []
        for i, rec in enumerate(samples):
            z = _z_scores(rec, comps)
            if z is None:
                continue
            zs.append(z)
            sz = _signed_z(rec, comps)
            signed.append(sz)
            kept.append(i)
        idx = np.asarray(kept, dtype=int)
        entry: dict[str, Any] = {"n": int(idx.size), "components":
                                 sum(1 for c in comps if c.sigma_eq is not None)}
        if idx.size == 0:
            entry["state"] = "no_sample_draws"
            out["sources"][name] = entry
            continue
        z = np.asarray(zs, dtype=np.float64)
        y = failed[idx]
        hit = z >= float(threshold_sigma)
        entry["failure_rate"] = float(y.mean())
        entry["mean_abs_z"] = float(z.mean())
        entry["mean_abs_z_landed"] = (float(z[landed[idx]].mean())
                                      if landed[idx].any() else None)
        entry["mean_abs_z_failed"] = (float(z[~landed[idx]].mean())
                                      if (~landed[idx]).any() else None)
        entry["tail"] = {
            "threshold_sigma": float(threshold_sigma),
            "high": {"n": int(hit.sum()),
                     "failure_rate": (float(y[hit].mean()) if hit.any() else None)},
            "low": {"n": int((~hit).sum()),
                    "failure_rate": (float(y[~hit].mean())
                                     if (~hit).any() else None)}}
        rho = _spearman(z, y)
        entry["rank_correlation"] = rho
        if len(signed) == idx.size and all(s is not None for s in signed):
            entry["signed_rank_correlation"] = _spearman(
                np.asarray(signed, dtype=np.float64), y)
        entry["state"] = "assessed" if rho is not None else "uninformative"
        out["sources"][name] = entry
        if rho is not None:
            scored.append((rho, name))
    scored.sort(key=lambda t: (-t[0], t[1]))
    out["ranking"] = [name for _rho, name in scored]
    if scored:
        out["state"] = "assessed"
    elif not np.any(failed):
        out["state"] = "all_landed"
    elif np.all(failed):
        out["state"] = "all_failed"
    return out


def solo_models(model: UncertaintyModel) -> dict[str, UncertaintyModel]:
    """One model per active source, carrying that source's declaration alone
    (everything else off) -- the one-at-a-time lineup for the marginal
    sensitivity reading. Declaration order and the source's own distribution
    and parameters are preserved verbatim."""
    decl = model.to_dict()
    out: dict[str, UncertaintyModel] = {}
    for name in SOURCE_ORDER:
        if not model.sources[name].is_active():
            continue
        out[name] = UncertaintyModel.from_config(
            {name: decl[name],
             **{other: {"enabled": False}
                for other in SOURCE_ORDER if other != name}})
    return out


def statistics_summary(*, successes: int, n: int,
                       samples: Sequence[Mapping[str, Any]],
                       model: UncertaintyModel,
                       level: float = 0.95,
                       solo: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The ``statistics`` section of ``montecarlo.json`` (R-06): the
    confidence interval on the success rate, the failure-class breakdown, the
    joint-dispersion sensitivity, and -- when the caller ran the one-at-a-time
    passes -- the marginal per-source reading. Pure function of the flown
    samples plus ``solo``, so it is as reproducible as the samples."""
    out: dict[str, Any] = {
        "success_rate_ci": success_rate_interval(successes, n, level),
        "failure_classes": failure_breakdown(samples),
        "sensitivity": source_sensitivity(samples, model),
    }
    if solo is not None:
        out["solo_sensitivity"] = solo
    return out
