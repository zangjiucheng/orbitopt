"""E7-02 (task #278): probability calibration of the predicted ``p_land``.

The E3-02 joint chance planner carries an analytic landing-probability
surrogate (``JointVerdict.p_land``, recorded per sample as
``landing_probability``), and ADR flightlab-0005 already labels it "a
surrogate, not a guaranteed flight success rate". E7-02 measures the gap
instead of asserting it: the E3-02 pilot reading (seed 1, n=32) averaged a
predicted 0.705 against an actual 0.531, and the low-probability bins looked
overconfident, but 32 samples cannot say anything. E3-03 (#260) provides the
pooled, multi-seed sample this needs (168 paired rows, seeds 1-4).

The reading is the standard one for probabilistic classifiers -- the model
is *calibrated* when, among all samples predicted ``p``, the landed fraction
is also ``p``:

* **reliability table** -- per predicted-probability bin: the bin size, the
  mean prediction, the observed conditional success rate with its
  :func:`~orbitopt.flightlab.recovery.stats.success_rate_interval` Wilson
  score interval (the same small-sample-honest interval R-06 quotes), the
  gap, and the bin's Brier contribution;
* **verdict per bin** -- a bin is ``consistent`` when the mean prediction
  lies inside the observed Wilson interval (the data cannot tell the
  prediction and the flight record apart), ``overconfident`` /
  ``underconfident`` when it lies strictly above / below it, and
  ``insufficient_data`` when the bin is too small to grade;
* **Brier score** -- the mean squared (prediction - outcome) error, reported
  against the climatology reference (a constant forecast at the pooled
  success rate), so the number says whether the per-sample prediction
  carries any information at all, not just how large the error is;
* **ECE / MCE** -- the size-weighted mean, and the maximum, of the per-bin
  |observed - mean prediction| gaps (computed against the mean prediction
  of the bin, not its edge-centre, so unequal-width bins are graded on
  what they actually forecast).

Everything here is a pure function of the ``pairs`` iterable and the
declared bins -- no RNG, no re-flying, no physics. A fixed evidence file
reproduces the report byte-for-byte.
"""
from __future__ import annotations

from math import fsum
from typing import Any, Iterable, Mapping, Sequence

from orbitopt.flightlab.recovery.stats import success_rate_interval

__all__ = [
    "DEFAULT_CALIBRATION_BINS",
    "VERDICT_CONSISTENT",
    "VERDICT_OVERCONFIDENT",
    "VERDICT_UNDERCONFIDENT",
    "VERDICT_INSUFFICIENT_DATA",
    "brier_score",
    "pairs_from_records",
    "reliability_table",
    "calibration_report",
]

#: The E3-02 pilot's grouping, refined by splitting the trusted high band
#: (the pilot put >=0.9 in one bin with 4 samples; the pooled reading
#: separates >=0.95, where the E3-02 default floor lives, from 0.9-0.95).
DEFAULT_CALIBRATION_BINS: tuple[float, ...] = (0.0, 0.5, 0.8, 0.9, 0.95, 1.0)

VERDICT_CONSISTENT = "consistent"
VERDICT_OVERCONFIDENT = "overconfident"
VERDICT_UNDERCONFIDENT = "underconfident"
VERDICT_INSUFFICIENT_DATA = "insufficient_data"


def _check_pairs(pairs: Iterable[Sequence[float]]) -> list[tuple[float, float]]:
    """Validate the (prediction, outcome) pairs: prediction inside [0, 1],
    outcome exactly 0 or 1. Returns the pairs as plain float tuples."""
    out: list[tuple[float, float]] = []
    for i, pair in enumerate(pairs):
        try:
            p, y = float(pair[0]), float(pair[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError(f"pair {i} is not a (probability, 0/1) pair: {pair!r}"
                             ) from exc
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"pair {i} prediction outside [0, 1]: {p}")
        if y not in (0.0, 1.0):
            raise ValueError(f"pair {i} outcome must be 0 or 1, got {pair[1]!r}")
        out.append((p, y))
    if not out:
        raise ValueError("calibration needs at least one (prediction, outcome) pair")
    return out


def _check_bins(bin_edges: Sequence[float]) -> list[float]:
    edges = [float(e) for e in bin_edges]
    if len(edges) < 2:
        raise ValueError(f"need at least two bin edges, got {edges}")
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError(f"bin edges must strictly increase, got {edges}")
    if edges[0] != 0.0 or edges[-1] != 1.0:
        raise ValueError(f"bin edges must span [0, 1], got {edges}")
    return edges


def brier_score(pairs: Iterable[Sequence[float]]) -> dict[str, Any]:
    """Mean Brier score of (prediction, outcome) pairs, plus the climatology
    reference (the constant forecast at the pooled success rate) and the
    derived skill. ``score > reference`` means the per-sample predictions are
    *worse* than ignoring them and quoting the population rate once."""
    ps = _check_pairs(pairs)
    n = len(ps)
    # Keep scores and derived skill independent of sample ordering.
    score = fsum((p - y) ** 2 for p, y in ps) / n
    rate = fsum(y for _, y in ps) / n
    reference = rate * (1.0 - rate)
    return {"n": n, "score": score,
            "reference_climatology": reference,
            "skill": 1.0 - score / reference if reference > 0.0 else None,
            "success_rate": rate}


def reliability_table(pairs: Iterable[Sequence[float]],
                      bin_edges: Sequence[float] | None = None,
                      level: float = 0.95,
                      min_bin_n: int = 5) -> list[dict[str, Any]]:
    """Per-bin reliability reading of the (prediction, outcome) pairs.

    Bins are half-open ``[lo, hi)`` with the last bin closed at 1.0 (a
    certain forecast belongs in the top bin, not nowhere). Every bin of the
    declared partition is reported, empty ones included -- an empty high bin
    is itself a finding (the planner never claimed high confidence), and is
    graded ``insufficient_data``, never dropped. The Wilson interval is over
    the *observed* successes; the verdict compares the mean *prediction*
    against it."""
    ps = _check_pairs(pairs)
    edges = _check_bins(bin_edges if bin_edges is not None
                        else DEFAULT_CALIBRATION_BINS)
    table: list[dict[str, Any]] = []
    for lo, hi in zip(edges, edges[1:]):
        last = hi == edges[-1]
        sel = [(p, y) for p, y in ps
               if (lo <= p < hi) or (last and p == hi)]
        n = len(sel)
        row: dict[str, Any] = {"bin": [lo, hi], "n": n}
        if n == 0:
            row.update({"successes": 0, "observed_rate": None,
                        "mean_predicted": None, "gap": None,
                        "bin_brier": None, "ci": None,
                        "verdict": VERDICT_INSUFFICIENT_DATA})
            table.append(row)
            continue
        k = int(sum(y for _, y in sel))
        obs = k / n
        pred = fsum(p for p, _ in sel) / n
        ci = success_rate_interval(k, n, level)
        gap = obs - pred
        if n < min_bin_n:
            verdict = VERDICT_INSUFFICIENT_DATA
        elif pred > ci["upper"]:
            verdict = VERDICT_OVERCONFIDENT
        elif pred < ci["lower"]:
            verdict = VERDICT_UNDERCONFIDENT
        else:
            verdict = VERDICT_CONSISTENT
        row.update({"successes": k, "observed_rate": obs,
                    "mean_predicted": pred, "gap": gap,
                    "bin_brier": fsum((p - y) ** 2 for p, y in sel) / n,
                    "ci": ci, "verdict": verdict})
        table.append(row)
    return table


def calibration_report(pairs: Iterable[Sequence[float]],
                       bin_edges: Sequence[float] | None = None,
                       level: float = 0.95,
                       min_bin_n: int = 5) -> dict[str, Any]:
    """The full E7-02 reading of the (prediction, outcome) pairs: Brier score
    against the climatology reference, the reliability table (with Wilson
    intervals and per-bin verdicts), the size-weighted ECE and the maximum
    calibration error over graded bins, and the lists of bin ranges that are
    trustworthy (``consistent``) versus in need of correction
    (``overconfident`` / ``underconfident``)."""
    ps = _check_pairs(pairs)
    table = reliability_table(ps, bin_edges, level, min_bin_n)
    n = len(ps)
    graded = [r for r in table if r["n"] >= 1]
    ece = fsum(r["n"] * abs(r["gap"]) for r in graded if r["gap"] is not None) / n
    mce = max((abs(r["gap"]) for r in graded if r["gap"] is not None),
              default=0.0)
    briers = brier_score(ps)
    return {
        "n": n,
        "confidence_level": float(level),
        "bin_edges": [float(e) for e in (bin_edges or DEFAULT_CALIBRATION_BINS)],
        "brier": briers,
        "ece": ece,
        "mce": mce,
        "reliability": table,
        "reliable_bins": [[r["bin"][0], r["bin"][1]]
                          for r in table if r["verdict"] == VERDICT_CONSISTENT],
        "correctable_bins": [[r["bin"][0], r["bin"][1]] for r in table
                             if r["verdict"] in (VERDICT_OVERCONFIDENT,
                                                VERDICT_UNDERCONFIDENT)],
    }


def pairs_from_records(records: Iterable[Mapping[str, Any]],
                       prob_key: str = "landing_probability",
                       landed_key: str = "outcome",
                       landed_value: str = "landed") -> list[tuple[float, float]]:
    """Extract (predicted p_land, landed 0/1) pairs from Monte Carlo sample
    rows, skipping rows without a recorded prediction (arms that do not run
    the joint chance surrogate). Row order and filtering are left to the
    caller."""
    pairs = []
    for row in records:
        p = row.get(prob_key)
        if p is None:
            continue
        pairs.append((float(p),
                      1.0 if row.get(landed_key) == landed_value else 0.0))
    return pairs
