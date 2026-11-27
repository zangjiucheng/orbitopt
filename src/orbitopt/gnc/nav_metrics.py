"""Consistency, observability and cluster statistics for relative navigation
(E13-02, task #419) -- offline evaluation, kept apart from the filter so no
truth-side quantity lives in :mod:`orbitopt.gnc.ekf`.

The acceptance vocabulary, with the exact definitions this project gates on
(the declared alternatives must be stated, because "NEES passes" without a
dof and an interval is a mood, not a test):

* **NEES** (normalized estimation error squared), per epoch:
  ``e = x_true - x_est``, ``nees = e' P^-1 e ~ chi2(6)`` when the filter is
  consistent and the errors are the Gaussian ones it assumes.
* **NIS** (normalized innovation squared), per update: ``y' S^-1 y ~
  chi2(m)`` with ``m`` the measurement dof ( :class:`ekf.UpdateReport.nis`).
* **Chi-square interval gate**: a statistic passes when it lies inside the
  central ``confidence`` band of ``chi2(dof)``, i.e. between
  ``ppf((1-confidence)/2)`` and ``ppf((1+confidence)/2)`` -- the
  ``interval`` every report row carries, so a reader can recompute the rate.
* **Coverage**: per-axis, the fraction of epochs with
  ``|error_i| <= k * sqrt(P_ii)`` for the declared ``k`` (a 3-sigma
  Gaussian bound covers 99.73 % of trials; the benchmark declares its
  nominal coverage and gates the measured one against it).
* **Observability**: linearized along a nominal trajectory -- the stacked
  observability matrix ``O = stack_k( H_k Phi(t_k, t_0) )`` (measurements
  linearized at the nominal state, one block row per update epoch), with
  rank and singular values reported. A combination is called observable
  when the rank is full (6) above the declared relative threshold.
* **Cluster statistics**: trials group into cells (scenario x measurement
  x geometry); every scalar metric is summarized per cell by mean / median /
  q05 / q95 / count, because a single average across regimes hides exactly
  the failures the mismatch study is about.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import chi2


def chi_square_interval(dof: int, confidence: float) -> tuple:
    """Central chi-square interval for ``dof`` at ``confidence`` (e.g.
    ``(3.97, 20.96)`` for dof 6 at 0.90): NEES/NIS statistics pass when
    inside it."""
    dof = int(dof)
    c = float(confidence)
    if dof < 1:
        raise ValueError(f"dof must be >= 1, got {dof!r}")
    if not (0.0 < c < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")
    return (float(chi2.ppf((1.0 - c) / 2.0, dof)), float(chi2.ppf((1.0 + c) / 2.0, dof)))


def chi_square_cdf(value: float, dof: int) -> float:
    """P(chi2(dof) <= value) -- for reporting where inside/outside the gate
    a statistic sits, not just which side."""
    return float(chi2.cdf(float(value), int(dof)))


def nees(x_true, x_est, cov) -> float:
    """Normalized estimation error squared for one epoch (error 6-vector,
    6x6 covariance); consistent filters yield chi2(6) samples."""
    e = np.asarray(x_true, dtype=np.float64).reshape(6) - np.asarray(x_est, dtype=np.float64).reshape(6)
    p = np.asarray(cov, dtype=np.float64).reshape(6, 6)
    return float(e @ np.linalg.solve(p, e))


def nis(innovation, innovation_cov) -> float:
    """``y' S^-1 y`` for one update (dimension defines the dof)."""
    y = np.asarray(innovation, dtype=np.float64).reshape(-1)
    s = np.asarray(innovation_cov, dtype=np.float64).reshape(y.size, y.size)
    return float(y @ np.linalg.solve(s, y))


@dataclass(frozen=True)
class ConsistencyGate:
    """A declared consistency test: central chi-square band for one dof."""

    dof: int
    confidence: float

    @property
    def interval(self) -> tuple:
        return chi_square_interval(self.dof, self.confidence)

    def passes(self, value: float) -> bool:
        lo, hi = self.interval
        return lo <= float(value) <= hi

    def to_dict(self) -> dict:
        lo, hi = self.interval
        return {"dof": self.dof, "confidence": self.confidence,
                "interval": [lo, hi]}


def sigma_coverage(errors, sigmas, k: float = 3.0) -> float:
    """Fraction of rows of ``errors`` (N, m) within ``k`` times ``sigmas``
    (N, m) elementwise."""
    e = np.abs(np.asarray(errors, dtype=np.float64))
    s = np.asarray(sigmas, dtype=np.float64)
    if e.shape != s.shape or e.ndim != 2:
        raise ValueError(f"sigma_coverage: errors{e.shape} and sigmas{s.shape} must be the "
                         "same (N, m)")
    kk = float(k)
    if not (math.isfinite(kk) and kk > 0.0):
        raise ValueError(f"sigma_coverage: k must be finite > 0, got {k!r}")
    return float(np.mean(e <= kk * s))


def rmse(errors) -> float:
    """Root mean square of the norm of row errors (N, m) -- the norm is
    taken per epoch, then RMS over epochs (position/velocity RMSE as the
    acceptance states it)."""
    e = np.asarray(errors, dtype=np.float64)
    if e.ndim != 2:
        raise ValueError(f"rmse: expect (N, m) errors, got shape {e.shape}")
    return float(np.sqrt(np.mean(np.einsum("ij,ij->i", e, e))))


def rmse_per_axis(errors) -> tuple:
    e = np.asarray(errors, dtype=np.float64)
    return tuple(float(v) for v in np.sqrt(np.mean(e * e, axis=0)))


def cluster_stats(values: Sequence[float]) -> dict:
    """Deterministic per-cell summary of a scalar metric across trials."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        raise ValueError("cluster_stats: empty cluster")
    return {
        "count": int(v.size),
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "q05": float(np.quantile(v, 0.05)),
        "q95": float(np.quantile(v, 0.95)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
    }


def pass_rate(values: Sequence[float], gate: ConsistencyGate) -> dict:
    """Fraction of statistics inside the gate's central chi-square band."""
    flags = [gate.passes(v) for v in values]
    n = len(flags)
    k = sum(flags)
    return {"n": n, "passes": k, "rate": (k / n if n else 0.0), "gate": gate.to_dict()}


# ---------------------------------------------------------------------------
# observability
# ---------------------------------------------------------------------------


def stacked_observability(h_rows, phi_prefixes, *, r_weights=None) -> np.ndarray:
    """Stacked observability matrix from per-epoch measurement Jacobians and
    state-transition prefixes. ``h_rows[k]`` is H_k (m_k, 6) linearized at
    the nominal state of epoch k, ``phi_prefixes[k]`` is Phi(t_k, t_0)
    (6, 6); the stack is ``stack_k(H_k Phi_k)`` optionally left-weighted by
    R_k^-1/2 (so the singular values carry measurement weighting -- the
    version a filter actually gets to see)."""
    if len(h_rows) != len(phi_prefixes) or not h_rows:
        raise ValueError("stacked_observability: need one H and one Phi per epoch")
    blocks = []
    for k, (h, phi) in enumerate(zip(h_rows, phi_prefixes)):
        h = np.asarray(h, dtype=np.float64)
        block = h @ np.asarray(phi, dtype=np.float64)
        if r_weights is not None:
            block = np.asarray(r_weights[k], dtype=np.float64) @ block
        blocks.append(block)
    return np.vstack(blocks)


def observability_rank(matrix, rtol: float = 1.0e-8) -> dict:
    """Rank (relative threshold), singular values and the ratio of the
    smallest to largest for one observability matrix."""
    m = np.asarray(matrix, dtype=np.float64)
    s = np.linalg.svd(m, compute_uv=False)
    if s.size == 0 or s[0] == 0.0:
        rank = 0
    else:
        rank = int(np.sum(s > rtol * s[0]))
    return {"rank": rank, "n_columns": int(m.shape[1]),
            "singular_values": [float(v) for v in s],
            "smallest_to_largest": float(s[-1] / s[0]) if s.size and s[0] > 0 else 0.0}
