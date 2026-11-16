"""Benchmark: E7-02 probability calibration of the predicted p_land.

The E3-02 joint chance planner forecasts a landing probability per sample;
E7-02 asks whether those forecasts are *true* -- do the samples predicted at
p actually land at rate p? The pilot reading (seed 1, n=32) averaged a
predicted 0.705 against an actual 0.531 and flagged the low bins as
overconfident, but 32 samples settle nothing. This grades the forecast on the
E3-03 pooled, multi-seed sample instead:

* **source data, no re-flying** -- the recorded paired rows of
  ``docs/benchmarks/e3_03_robust_montecarlo.json`` (four seeds, 168 pooled
  samples, identical draws across arms); only the joint-chance arm carries a
  per-sample ``landing_probability``, and the acceptance floor of >=168
  pooled samples is enforced here, not assumed;
* **the standard classifier reading** -- reliability table per predicted
  bin with Wilson 95 % intervals on the observed rate, per-bin verdicts
  (consistent / overconfident / underconfident), Brier score against the
  climatology reference, ECE and MCE
  (:mod:`orbitopt.flightlab.recovery.calibration`);
* **the reliability diagram** -- the table drawn as observed-rate points
  with Wilson error bars against the diagonal, plus the histogram of
  predicted probabilities below.

Pure post-processing of recorded evidence (stdlib + matplotlib), seconds not
CPU-hours:

    PYTHONPATH=src python benchmarks/bench_e7_02_probability_calibration.py \
        -o docs/benchmarks/e7_02_probability_calibration.json \
        --plot docs/assets/e7_02_reliability.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from orbitopt.flightlab.recovery.calibration import (  # noqa: E402
    DEFAULT_CALIBRATION_BINS,
    VERDICT_CONSISTENT,
    calibration_report,
    pairs_from_records,
)

DEFAULT_INPUT = REPO / "docs" / "benchmarks" / "e3_03_robust_montecarlo.json"

#: The pooled-sample floor the task acceptance sets: the pilot's n=32 must be
#: replaced by the E3-03 multi-seed pooled sample.
MIN_POOLED = 168


def load_pairs(path: Path, arm: str, min_pooled: int,
               min_seeds: int) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    """Read the recorded E3-03 report, pool the arm's paired rows and return
    the calibration pairs plus a provenance block. Raises on any acceptance
    floor not met -- the calibration must not be quoted on less data than the
    acceptance names."""
    report = json.loads(path.read_text())
    if report.get("kind") != "e3-03-robust-recovery-montecarlo":
        raise SystemExit(f"{path} is not an E3-03 robust Monte Carlo report")
    runs = report.get("runs", {})
    if arm not in runs:
        raise SystemExit(f"arm {arm!r} not in {path.name} (have: "
                         f"{', '.join(sorted(runs))})")
    rows = runs[arm]["paired_rows"]
    pairs = pairs_from_records(rows)
    seeds = sorted({int(r["seed"]) for r in rows})
    if len(pairs) < min_pooled:
        raise SystemExit(f"pooled sample is {len(pairs)} rows, acceptance needs "
                         f">= {min_pooled}; re-run the E3-03 benchmark with a "
                         f"larger n-per-seed before calibrating")
    if len(seeds) < min_seeds:
        raise SystemExit(f"pooled sample covers seeds {seeds}, acceptance needs "
                         f"a multi-seed pool (>= {min_seeds})")
    provenance = {
        "source": str(path.relative_to(REPO) if path.is_relative_to(REPO)
                      else path),
        "arm": arm,
        "seeds": seeds,
        "n_pooled_rows": len(rows),
        "n_with_prediction": len(pairs),
    }
    return pairs, provenance


def plot_reliability(report: dict[str, Any], provenance: dict[str, Any],
                     path: Path) -> None:
    """Reliability diagram: observed rate with Wilson bars per bin against
    the perfect-calibration diagonal, plus the prediction histogram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = [r for r in report["reliability"] if r["n"] > 0]
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(6.4, 7.2), height_ratios=[2, 1],
        gridspec_kw={"hspace": 0.35})
    diag = [0.0, 1.0]
    ax.plot(diag, diag, color="0.5", lw=1.0, ls="--", label="perfect")
    for row in bins:
        x = row["mean_predicted"]
        obs = row["observed_rate"]
        colour = ("#2f8f4e" if row["verdict"] == VERDICT_CONSISTENT
                  else "#c0392b")
        ax.errorbar(x, obs,
                    yerr=[[obs - row["ci"]["lower"]],
                          [row["ci"]["upper"] - obs]],
                    fmt="o", color=colour, capsize=3, ms=6)
        ax.text(x, row["observed_rate"], f"  n={row['n']}", va="center",
                fontsize=8, color=colour)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted $p_{land}$ per bin")
    ax.set_ylabel("observed landing rate")
    ax.set_title(f"E7-02 reliability: joint-chance $p_{{land}}$ "
                 f"(n={report['n']}, seeds "
                 f"{', '.join(str(s) for s in provenance['seeds'])})")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    edges = report["bin_edges"]
    counts = [r["n"] for r in report["reliability"]]
    ax2.bar(range(len(counts)), counts, color="#4a6fa5", alpha=0.8)
    ax2.set_xticks(range(len(counts)))
    ax2.set_xticklabels([f"{e0:g}-{e1:g}" for e0, e1 in zip(edges, edges[1:])],
                        fontsize=8)
    ax2.set_ylabel("samples")
    ax2.set_title("predicted-$p_{land}$ histogram", fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-i", "--input", type=Path, default=DEFAULT_INPUT,
                    help="recorded E3-03 report (default: %(default)s)")
    ap.add_argument("--arm", default="e3_02_joint_chance",
                    help="arm whose per-sample p_land is graded "
                         "(default: %(default)s)")
    ap.add_argument("--bins", type=float, nargs="+",
                    default=list(DEFAULT_CALIBRATION_BINS),
                    help="bin edges spanning [0,1] (default: %(default)s)")
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--min-pooled", type=int, default=MIN_POOLED)
    ap.add_argument("--min-seeds", type=int, default=2)
    ap.add_argument("--plot", type=Path, default=None,
                    help="optional reliability diagram (PNG path)")
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args(argv)

    pairs, provenance = load_pairs(args.input, args.arm, args.min_pooled,
                                   args.min_seeds)
    report = calibration_report(pairs, args.bins, args.level)
    out = {"kind": "e7-02-probability-calibration",
           "provenance": provenance, **report}
    text = json.dumps(out, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    else:
        print(text)
    if args.plot:
        plot_reliability(report, provenance, args.plot)
        print(f"diagram -> {args.plot}")

    b = report["brier"]
    skill = b["skill"]
    skill_txt = "n/a" if skill is None else f"{skill:+.4f}"
    print(f"n={report['n']} seeds={provenance['seeds']} "
          f"success={b['success_rate']:.3f} brier={b['score']:.4f} "
          f"(climatology {b['reference_climatology']:.4f}, "
          f"skill {skill_txt}) ece={report['ece']:.4f} "
          f"mce={report['mce']:.4f}")
    for row in report["reliability"]:
        if row["n"]:
            print(f"  {row['verdict']:<18} [{row['bin'][0]:g},{row['bin'][1]:g}) "
                  f"n={row['n']:>3} pred={row['mean_predicted']:.3f} "
                  f"obs={row['observed_rate']:.3f} "
                  f"wilson=[{row['ci']['lower']:.3f},{row['ci']['upper']:.3f}]")
        else:
            print(f"  {row['verdict']:<18} [{row['bin'][0]:g},{row['bin'][1]:g}) "
                  f"n=  0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
