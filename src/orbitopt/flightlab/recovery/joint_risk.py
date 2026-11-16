"""E7-03 (task #279, epic #266): joint risk modeling of the fuel and divert
channels of the landing-probability constraint.

The written case
----------------
``chance.evaluate_joint_chance`` multiplies the channel probabilities,
``p_land = p_fuel * p_divert`` -- an independence assertion that was never
tested (the task this module answers). E7-02 (task #278) had shown the
surrogate to be consistent only above the 0.8 floor and overconfident by
4-9x below it, with one candidate explanation being exactly this product.
This module is the measurement and the model E7-03 chose; the full evidence
(with confidence intervals, sensitivity and the calibration comparison) is
`docs/benchmarks/e7_03_joint_risk.md` (JSON:
`docs/benchmarks/e7_03_joint_risk.json`, script
`benchmarks/bench_e7_03_joint_risk.py`) -- this docstring records the
decisions, the JSON carries the numbers.

Data and indicators (rules fixed *before* correlating, from the reason-code
vocabulary the sample itself carries -- :mod:`.boostback` / :mod:`.landing`
are the source of the codes and the 25 m position tolerance):

* **fuel channel fails** -- the realized corridor margin is negative
  (``corridor_fuel_margin_kg < 0``, recorded whenever the corridor was
  crossed), or the governing reason is ``LANDING_PROPELLANT_RESERVE_LOW``
  (the margin at handoff was fine and the landing-burn dispersion ate the
  reserve: a fuel-channel failure observed downstream);
* **divert channel fails** -- the corridor state window was missed
  (``CORRIDOR_MISSED``: the boostback divert closure did not deliver a
  usable insertion), or a touchdown happened outside the position
  tolerance (``LANDING_POSITION_OUT_OF_TOLERANCE``: the landing burn could
  not close the divert);
* **censored divert** -- the chain aborted for corridor propellant
  (``INSUFFICIENT_PROPELLANT_RESERVE``): the landing burn was never flown,
  so the divert outcome was never observed. Those rows are *excluded* from
  the contingency table (counting them either way is an assumption, not a
  measurement) and the report quotes the Frechet bounds they induce;
* the ``LANDING_LATERAL_SPEED_EXCEEDED`` rows touch down inside the
  position tolerance: neither channel failed, landing still did not happen.
  A third mode the two-channel model does not carry -- counted as
  no-channel-failure here and flagged in the report, *not* folded into the
  divert channel (the p_land surrogate never claims to model it).

Findings (n = 168 per arm; per-arm phi with bootstrap CI, pooled with the
censored rows excluded; Fisher-exact for significance): realized fuel and
divert failures are **negatively associated** (pooled phi = -0.127, Fisher
p = 0.010): they *compete* -- a sample that burns through its reserve tends
to reach the corridor near its centre (short boostback, low miss), and a
big-divert sample arrives with margin. Co-failure is therefore *rarer* than
independence says (pooled observed joint-failure 0.027 vs 0.048 expected),
which means "at least one channel fails" is *more* likely than the product
admits -- **the product overstates p_land**, the very direction E7-02
measured as the surrogate's overconfidence. A Gaussian copula with the
fitted correlation (fit *out of sample*: pooled from the three non-joint
arms, graded on the joint-chance arm -- ``rho = -0.27``) pulls the forecast
down where E7-02 said it should come down and improves the calibration
metrics (Brier 0.222 -> 0.214, skill ~0 -> +0.04, ECE 0.313 -> 0.301, MCE
0.581 -> 0.563): coupling explains a first, quantified chunk of the
mid-band optimism -- not all of it (the 0.5-0.8 bin remains overconfident;
E7-02's monotone-recalibration decision stands on top of this one).

Why this model (the acceptance criterion "the chosen method has a written
basis"):

* **joint Monte Carlo** (the task's named alternative) *is* the E3-03
  sample -- it is what we fit from, not what we plan with: the planner needs
  p_land at every corridor handoff of every Nelder-Mead step, and re-flying
  the campaign per evaluation is exactly the cost the chance constraint
  exists to avoid. A copula is the two-parameter summary of that joint
  sample the chance module can evaluate analytically.
* **Gaussian copula**: preserves the two declared marginals exactly, is
  monotone in each argument (no plan ever buys a *higher* p_land by
  spending margin -- the E7-01/E7-02 constraint semantics survive), is
  Frechet-bounded by construction, and reduces to the shipped product at
  rho = 0, so the coupling is additive to the existing spec, not a rewrite.
  The copula correlation is fitted by matching the observed joint-failure
  frequency at the observed marginals (one scalar, one moment -- the data
  supports one dependence parameter, not a family).
* the tempting **linear Bernoulli coupling** ``P(both fail) = (1-lam)
  pF*pD + lam*min(...)`` was rejected: it is non-monotone in p_land near the
  corners for the lam the data wants, which would let the penalized optimizer
  game the constraint.
* **negative rho, honestly labelled**: failures compete, so coupling *lowers*
  p_land below the product (bivariate normal with rho < 0 is
  negative-quadrant-dependent for every marginal pair). This corrects the
  forecast in the direction E7-02 asked for -- but only part of the way:
  after coupling the 0.5-0.8 bin is still overconfident, and the e7-02 rule
  (never trade low-p_land samples on the forecast) still stands.

``FITTED_LANDING_DEPENDENCE_RHO`` below is the value carried to the planner
spec (``JointChanceSpec.landing_dependence_rho``); the default stays 0.0
(product) until the epic flips it with a decision -- see the e7-03 report's
verdict.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

from orbitopt.flightlab.recovery.boostback import (CORRIDOR_MISSED,
                                                  INSUFFICIENT_PROPELLANT_RESERVE)
from orbitopt.flightlab.recovery.chance import phi_inv
from orbitopt.flightlab.recovery.landing import DEFAULT_LANDING_LIMITS

__all__ = [
    "FITTED_LANDING_DEPENDENCE_RHO",
    "back_out_component_probabilities",
    "bootstrap_phi_ci",
    "contingency_table",
    "coupled_pairs_from_records",
    "failure_channels",
    "fisher_exact_two_sided_p",
    "fit_gaussian_copula_rho",
    "frechet_phi_bounds",
    "implied_loss_sigma_kg",
    "pearson_correlation",
    "phi_coefficient",
    "phi_report",
]

#: Corridor propellant abort -- divert censored (module docstring). Kept as
#: the imported constant; a row's governing reason is the string the sample
#: carries (:mod:`.montecarlo` writes ``outcome.reason_codes[0]``).
CENSORED_DIVERT_REASON = INSUFFICIENT_PROPELLANT_RESERVE

#: Lateral-speed landings: touchdown inside the position tolerance, so
#: neither channel failed -- a third failure mode, neither fuel nor divert
#: (module docstring).
SPEED_FAILURE_REASON = "LANDING_LATERAL_SPEED_EXCEEDED"

#: Position tolerance the recorded sample was graded with -- the landing
#: verdict's own limit, not a knob (``LANDING_POSITION_OUT_OF_TOLERANCE`` is
#: the same test one level up).
DEFAULT_POSITION_TOL_M = float(DEFAULT_LANDING_LIMITS.position_tol_m)

#: E7-03 fitted Gaussian-copula correlation between the fuel and divert
#: *channels* (same sign for failures and successes: the Gaussian copula is
#: radially symmetric). Fitted out of sample -- pooled contingency of the
#: three arms that did NOT optimize the joint chance, applied to the arm
#: that did; the moment match is :func:`fit_gaussian_copula_rho`.
#: DESIGN ASSUMPTION anchored on docs/benchmarks/e7_03_joint_risk.json
#: (module docstring: the measurement, the report: the case). Negative --
#: realized failures compete -- so this value *lowers* coupled p_land below
#: the product (the correction E7-02's reliability table asked for, partial
#: as it is); see the e7-02 decision before quoting p_land below 0.8.
FITTED_LANDING_DEPENDENCE_RHO = -0.273

#: Bootstrap replicates / seed for :func:`bootstrap_phi_ci` (deterministic
#: by repo rule; 2000 replicates is resolution-limited at n ~ 400, matching
#: the interval widths the Wilson tables quote).
DEFAULT_BOOTSTRAP_TRIALS = 2000
DEFAULT_BOOTSTRAP_SEED = 20251115


# -- channel indicators ------------------------------------------------------

def failure_channels(record: Mapping[str, Any], *,
                     position_tol_m: float = DEFAULT_POSITION_TOL_M,
                     ) -> Tuple[Optional[bool], Optional[bool]]:
    """(fuel_fail, divert_fail) for one recorded Monte Carlo sample row, with
    ``None`` = the channel's outcome was never observed (module docstring's
    rules: a row that never crossed the corridor evaluates neither channel;
    a corridor propellant abort never flies the landing burn, censoring the
    divert; every other row carries the fields the rules read)."""
    if record.get("outcome") == "error":
        return (None, None)
    reason = record.get("reason") or ""
    margin = record.get("corridor_fuel_margin_kg")
    if margin is None and reason not in ("LANDING_PROPELLANT_RESERVE_LOW",):
        fuel: Optional[bool] = None
    else:
        fuel = bool((margin is not None and float(margin) < 0.0)
                    or reason == "LANDING_PROPELLANT_RESERVE_LOW")
    pos = record.get("position_error_m")
    if reason == CORRIDOR_MISSED:
        divert: Optional[bool] = True
    elif pos is not None:
        divert = bool(float(pos) > float(position_tol_m))
    else:
        divert = None
    return (fuel, divert)


def contingency_table(records: Iterable[Mapping[str, Any]], *,
                      position_tol_m: float = DEFAULT_POSITION_TOL_M,
                      ) -> dict[str, Any]:
    """2x2 table of the realized channel outcomes over the recorded rows,
    with the censored/unobserved rows counted separately by reason (never
    silently dropped): cells ``both``/``fuel_only``/``divert_only``/
    ``neither``, ``n_observed``, and the reason histograms ``censored_divert``
    (fuel observed, divert not) and ``not_evaluated`` (neither)."""
    both = fuel_only = divert_only = neither = 0
    censored = censored_fuel_fail = 0
    censored_reasons: Counter[str] = Counter()
    not_eval: Counter[str] = Counter()
    for rec in records:
        fuel, divert = failure_channels(rec, position_tol_m=position_tol_m)
        reason = str(rec.get("reason") or "UNKNOWN")
        if fuel is None or divert is None:
            if fuel is None:
                not_eval[reason] += 1
            else:
                censored += 1
                censored_reasons[reason] += 1
                if fuel:
                    censored_fuel_fail += 1
            continue
        if fuel and divert:
            both += 1
        elif fuel:
            fuel_only += 1
        elif divert:
            divert_only += 1
        else:
            neither += 1
    return {"both": both, "fuel_only": fuel_only, "divert_only": divert_only,
            "neither": neither, "n_observed": both + fuel_only + divert_only + neither,
            "censored_divert": dict(censored_reasons), "censored_divert_total": censored,
            "censored_divert_fuel_fail": censored_fuel_fail,
            "not_evaluated": dict(not_eval), "not_evaluated_total": sum(not_eval.values())}


def _contingency_from_censored(table: Mapping[str, Any], *, divert_fails: bool,
                               ) -> tuple[int, int, int, int]:
    """Cells with every censored-divert row resolved one way (Frechet-style
    sensitivity): ``divert_fails=True`` pushes all censored rows into the
    divert-failure column, ``False`` into the success column; their fuel
    status (observed) decides the row."""
    both, fuel_only = table["both"], table["fuel_only"]
    divert_only, neither = table["divert_only"], table["neither"]
    c = int(table.get("censored_divert_total", 0))
    cf = int(table.get("censored_divert_fuel_fail", 0))
    if divert_fails:
        return (both + cf, fuel_only, divert_only + (c - cf), neither)
    return (both, fuel_only + cf, divert_only, neither + (c - cf))


# -- association statistics ----------------------------------------------------

def phi_coefficient(both: int, fuel_only: int, divert_only: int, neither: int,
                    ) -> Optional[float]:
    """phi correlation of the two failure indicators over the 2x2 table (the
    Pearson correlation of the indicators; equals Cramer's V for a 2x2):
    ``(ad - bc) / sqrt((a+b)(c+d)(a+c)(b+d))``. ``None`` when a margin is
    degenerate -- the correlation is then undefined, not zero."""
    n = both + fuel_only + divert_only + neither
    row1, row0 = both + fuel_only, divert_only + neither
    col1, col0 = both + divert_only, fuel_only + neither
    if n == 0 or min(row1, row0, col1, col0) == 0:
        return None
    den = math.sqrt(float(row1) * row0 * col1 * col0)
    return (both * neither - fuel_only * divert_only) / den


def _log_choose(n: int, k: int) -> float:
    return (math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0))


def fisher_exact_two_sided_p(both: int, fuel_only: int, divert_only: int,
                             neither: int) -> float:
    """Exact two-sided Fisher test of channel independence on the 2x2 table
    (probability method: sum of the hypergeometric probabilities of all
    tables at most as likely as the observed one). No normal approximation:
    the co-failure cell is single digits."""
    a, b, c, d = (both, fuel_only, divert_only, neither)
    n = a + b + c + d
    row1, col1 = a + b, a + c
    log_den = _log_choose(n, row1)
    lo, hi = max(0, row1 + col1 - n), min(row1, col1)
    pmfs = []
    for t in range(lo, hi + 1):
        pmfs.append(math.exp(_log_choose(col1, t) + _log_choose(n - col1, row1 - t)
                             - log_den))
    p_obs = pmfs[a - lo]
    tol = p_obs * (1.0 + 1e-9) + 1e-300
    return min(1.0, sum(p for p in pmfs if p <= tol))


def _fisher_z_ci(r: float, n: int, level: float = 0.95) -> Optional[Tuple[float, float]]:
    """Fisher-z interval for a correlation (any Pearson-type r, phi
    included): ``tanh(atanh(r) +- z/ sqrt(n-3))``. ``None`` when undefined
    (n <= 3, |r| >= 1)."""
    if n <= 3 or not -1.0 < r < 1.0:
        return None
    z = phi_inv(0.5 + level / 2.0)
    w = math.atanh(r)
    return (math.tanh(w - z / math.sqrt(n - 3)), math.tanh(w + z / math.sqrt(n - 3)))


def bootstrap_phi_ci(both: int, fuel_only: int, divert_only: int, neither: int, *,
                     trials: int = DEFAULT_BOOTSTRAP_TRIALS,
                     seed: int = DEFAULT_BOOTSTRAP_SEED,
                     level: float = 0.95) -> Optional[Tuple[float, float]]:
    """Percentile bootstrap CI for phi under the multinomial row model of
    the 2x2 table (the row draws of one arm are independent; this is the
    same interval the report quotes per arm). Deterministic on ``seed``."""
    n = both + fuel_only + divert_only + neither
    if n == 0:
        return None
    probs = [both / n, fuel_only / n, divert_only / n, neither / n]
    rng = random.Random(seed)

    def _binom(m: int, p: float) -> int:
        if p <= 0.0:
            return 0
        if p >= 1.0:
            return m
        return sum(1 for _ in range(m) if rng.random() < p)

    phis = []
    for _ in range(trials):
        x1 = _binom(n, probs[0])
        rem = n - x1
        x2 = _binom(rem, probs[1] / (1.0 - probs[0])) if rem and probs[1] else 0
        rem -= x2
        x3 = _binom(rem, probs[2] / max(1e-15, 1.0 - probs[0] - probs[1])) \
            if rem and probs[2] else 0
        x4 = rem - x3
        phis.append(phi_coefficient(x1, x2, x3, x4))
    phis = sorted(p for p in phis if p is not None)
    if len(phis) < trials // 2:
        return None
    alpha = (1.0 - level) / 2.0
    lo = phis[max(0, int(alpha * len(phis)))]
    hi = phis[min(len(phis) - 1, int(math.ceil((1.0 - alpha) * len(phis))) - 1)]
    return (lo, hi)


def phi_report(table: Mapping[str, Any], *, trials: int = DEFAULT_BOOTSTRAP_TRIALS,
               seed: int = DEFAULT_BOOTSTRAP_SEED) -> dict[str, Any]:
    """Association summary of a contingency table: phi, its Fisher-z and
    bootstrap CIs, the exact Fisher p, the independent-model expectation it
    is compared against, and the Frechet bounds the censored rows induce
    (:func:`frechet_phi_bounds`)."""
    cells = (table["both"], table["fuel_only"], table["divert_only"],
             table["neither"])
    n = sum(cells)
    phi_hat = phi_coefficient(*cells)
    marg_f = (cells[0] + cells[1]) / n if n else 0.0
    marg_d = (cells[0] + cells[2]) / n if n else 0.0
    joint_obs = cells[0] / n if n else 0.0
    return {"cells": list(cells), "n_observed": n, "phi": phi_hat,
            "ci95_fisher_z": _fisher_z_ci(phi_hat, n) if phi_hat is not None else None,
            "ci95_bootstrap": bootstrap_phi_ci(*cells, trials=trials, seed=seed),
            "fisher_exact_p": fisher_exact_two_sided_p(*cells),
            "marginal_fuel_fail": marg_f, "marginal_divert_fail": marg_d,
            "joint_failure_rate_observed": joint_obs,
            "joint_failure_rate_independent": marg_f * marg_d,
            "phi_bounds_censored_frechet": frechet_phi_bounds(table)}


def frechet_phi_bounds(table: Mapping[str, Any]) -> Optional[Tuple[float, float]]:
    """phi with the censored-divert rows resolved to divert-failure vs
    divert-success (both extreme readings of the ``INSUFFICIENT_PROPELLANT_
    RESERVE`` censoring; module docstring). ``None`` when nothing is
    censored or phi is degenerate on either resolution."""
    if not table.get("censored_divert_total"):
        return None
    hi_cells = _contingency_from_censored(table, divert_fails=True)
    lo_cells = _contingency_from_censored(table, divert_fails=False)
    hi, lo = phi_coefficient(*hi_cells), phi_coefficient(*lo_cells)
    if hi is None or lo is None:
        return None
    return (min(hi, lo), max(hi, lo))


def pearson_correlation(xs: Sequence[float], ys: Sequence[float],
                        level: float = 0.95) -> Optional[dict[str, Any]]:
    """Sample Pearson correlation with n and Fisher-z CI; ``None`` when a
    side is degenerate. The continuous companion of phi: corridor margin vs
    corridor miss, the same competition seen without binning."""
    n = min(len(xs), len(ys))
    if n < 3:
        return None
    mx = sum(float(v) for v in xs[:n]) / n
    my = sum(float(v) for v in ys[:n]) / n
    sxx = syy = sxy = 0.0
    for x, y in zip(xs[:n], ys[:n]):
        dx, dy = float(x) - mx, float(y) - my
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    if sxx <= 0.0 or syy <= 0.0:
        return None
    r = sxy / math.sqrt(sxx * syy)
    return {"r": r, "n": n, "ci95": _fisher_z_ci(r, n, level)}


# -- the copula fit ------------------------------------------------------------

def fit_gaussian_copula_rho(p_fail_fuel: float, p_fail_divert: float,
                            observed_joint_failure: float, *,
                            tol: float = 1e-4) -> Optional[float]:
    """Copula correlation matching the observed joint-failure frequency at
    the observed marginals under a Gaussian copula: solve
    ``coupled_joint_probability(pF, pD; rho) = observed`` by bisection (the
    copula joint is strictly monotone in rho). ``None`` when a marginal is
    0/1 or the target sits exactly on a Frechet bound (rho -> -1/+1 --
    unidentifiable from one table); clamped to |rho| <= 0.999 at the edges."""
    from orbitopt.flightlab.recovery.chance import coupled_joint_probability
    if not 0.0 < p_fail_fuel < 1.0 or not 0.0 < p_fail_divert < 1.0:
        return None
    lo = max(0.0, p_fail_fuel + p_fail_divert - 1.0)
    hi = min(p_fail_fuel, p_fail_divert)
    if not lo + 1e-9 < observed_joint_failure < hi - 1e-9:
        return None
    a, b = -0.999, 0.999
    for _ in range(64):
        mid = 0.5 * (a + b)
        if coupled_joint_probability(p_fail_fuel, p_fail_divert, mid) \
                < observed_joint_failure:
            a = mid
        else:
            b = mid
        if b - a < tol:
            break
    return 0.5 * (a + b)


def back_out_component_probabilities(record: Mapping[str, Any],
                                     spec: Any) -> Optional[Tuple[float, float]]:
    """Recover the (p_fuel, p_divert) factors of a recorded row of a
    product-mode run: the sample stores only the product
    ``landing_probability``, but the handoff miss (``corridor_miss_m``)
    pins p_divert through the declared model ``spec.p_divert``, and the
    product then pins p_fuel exactly (division by the same factor the
    planner multiplied by -- a back-out, not a re-prediction). ``None`` when
    the row carries no joint verdict (not a joint-mode row) or is
    inconsistent beyond float tolerance. Raises if ``spec`` itself couples
    (the back-out inverts the product, not a copula)."""
    if float(getattr(spec, "landing_dependence_rho", 0.0)) != 0.0:
        raise ValueError("back-out inverts the product model; spec must be "
                         "landing_dependence_rho = 0.0")
    p_land = record.get("landing_probability")
    miss = record.get("corridor_miss_m")
    if p_land is None or miss is None:
        return None
    p_div = float(spec.p_divert(float(miss)))
    if p_div <= 0.0:
        return None
    p_fuel = float(p_land) / p_div
    if p_fuel < -1e-9 or p_fuel > 1.0 + 1e-9:
        return None
    return (min(max(p_fuel, 0.0), 1.0), p_div)


def implied_loss_sigma_kg(record: Mapping[str, Any], spec: Any) -> Optional[float]:
    """The propellant-loss 1-sigma implied by a recorded row: the back-out
    p_fuel read through its quantile, ``margin / phi_inv(p_fuel)``. A
    consistency diagnostic of the declared loss model on the real sample
    (the rows should all imply one sigma); ``None`` when the row is not a
    product-mode joint row or the quantile is at a numerical edge."""
    comp = back_out_component_probabilities(record, spec)
    margin = record.get("corridor_fuel_margin_kg")
    if comp is None or margin is None:
        return None
    p_fuel = comp[0]
    if not 1e-15 < p_fuel < 1.0 - 1e-15:
        return None
    z = phi_inv(p_fuel)
    if z <= 0.0:
        return None
    return float(margin) / z


def coupled_pairs_from_records(records: Iterable[Mapping[str, Any]], spec: Any,
                               rho: float) -> list[tuple[float, float]]:
    """(coupled p_land, landed 0/1) pairs from recorded product-mode rows
    for a calibration grading: back out the factors per row, couple them at
    ``rho``. Rows without a joint verdict are skipped (they never entered
    the forecast the comparison grades)."""
    from orbitopt.flightlab.recovery.chance import coupled_joint_probability
    pairs: list[tuple[float, float]] = []
    for rec in records:
        comp = back_out_component_probabilities(rec, spec)
        if comp is None:
            continue
        p_land = coupled_joint_probability(comp[0], comp[1], rho)
        pairs.append((p_land, 1.0 if rec.get("outcome") == "landed" else 0.0))
    return pairs
