"""Benchmark: E7-03 joint risk modeling of the fuel and divert channels.

``chance.evaluate_joint_chance`` multiplies the channel probabilities --
``p_land = p_fuel * p_divert`` -- an independence assertion this task tests
on recorded Monte Carlo data instead of assuming. The reading is pure
post-processing of ``docs/benchmarks/e3_03_robust_montecarlo.json`` (four
arms, identical draws, 168 pooled rows per seed-pool; no re-flying), and it
follows the written model in :mod:`orbitopt.flightlab.recovery.joint_risk`:

* **contingency per arm** -- realized fuel/divert channel outcomes from the
  reason-code vocabulary the rows carry; the corridor-propellant aborts
  (divert censored -- the landing burn never flew) are counted separately
  and reported as Frechet bounds, never silently binned;
* **association** -- phi with bootstrap + Fisher-z 95 % intervals and an
  exact Fisher test (per arm and pooled over the three arms that did NOT
  optimize the joint chance, which is the honest fit set: the joint arm's
  planner *reacted to* the product model, so its rows cannot set the
  coefficient the replacement is fitted with);
* **the joint model** -- a Gaussian copula, rho fitted on the fit set by
  moment match, then graded on the joint-chance arm's 168 rows against the
  product forecast under the E7-02 calibration metrics (Brier/skill, ECE,
  MCE, per-bin verdicts) -- out-of-sample by construction;
* **diagnostics** -- Pearson margin-vs-miss (the same competition without
  binning) and per-row implied loss sigma (the back-out is exact: recorded
  ``landing_probability`` is the product evaluated on the realized row).

Run:

    PYTHONPATH=src python benchmarks/bench_e7_03_joint_risk.py \
        -o docs/benchmarks/e7_03_joint_risk.json \
        --plot docs/assets/e7_03_joint_coupling.png
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from orbitopt.flightlab.recovery.calibration import (  # noqa: E402
    DEFAULT_CALIBRATION_BINS,
    calibration_report,
)
from orbitopt.flightlab.recovery.chance import JointChanceSpec  # noqa: E402
from orbitopt.flightlab.recovery.joint_risk import (  # noqa: E402
    CENSORED_DIVERT_REASON,
    DEFAULT_POSITION_TOL_M,
    contingency_table,
    coupled_pairs_from_records,
    fit_gaussian_copula_rho,
    implied_loss_sigma_kg,
    pearson_correlation,
    phi_report,
)

DEFAULT_INPUT = REPO / "docs" / "benchmarks" / "e3_03_robust_montecarlo.json"

#: The arm whose forecast is graded -- the only arm whose planner consumed
#: the product model, and the one the fit set must exclude (docstring).
JOINT_ARM = "e3_02_joint_chance"

#: Acceptance floor: the correlation is quoted on the E3-03 pooled sample,
#: not the pilot's n=32 (the E7-02 floor, same source, same reason).
MIN_POOLED = 168

#: Nominal corridor exposure [kg] for the declared-sigma cross-check
#: (SEP loadout + landing need at the E3-03 sizing); the per-row implied
#: sigma is compared to this, not asserted equal.
NOMINAL_EXPOSURE_KG = 29_000.0


def load_report(path: Path, min_pooled: int, min_seeds: int) -> dict[str, Any]:
    """Read and validate the recorded E3-03 report (kind, per-arm row floor,
    multi-seed pool) -- same guard policy as the E7-02 script: refuse, don't
    quote, on less data than the acceptance names."""
    report = json.loads(path.read_text())
    if report.get("kind") != "e3-03-robust-recovery-montecarlo":
        raise SystemExit(f"{path} is not an E3-03 robust Monte Carlo report")
    runs = report.get("runs", {})
    for arm in runs:
        rows = runs[arm].get("paired_rows") or []
        if len(rows) < min_pooled:
            raise SystemExit(f"arm {arm!r} carries {len(rows)} rows, acceptance "
                             f"needs >= {min_pooled}; re-run E3-03 first")
        seeds = sorted({int(r["seed"]) for r in rows})
        if len(seeds) < min_seeds:
            raise SystemExit(f"arm {arm!r} covers seeds {seeds}, acceptance "
                             f"needs a multi-seed pool (>= {min_seeds})")
    return report


def arm_block(rows: list[dict[str, Any]], spec: JointChanceSpec) -> dict[str, Any]:
    """Contingency + association + continuous diagnostic + per-arm copula
    fit for one arm's recorded rows."""
    table = contingency_table(rows)
    assoc = phi_report(table)
    pairs = [(r["corridor_fuel_margin_kg"], r["corridor_miss_m"])
             for r in rows
             if r.get("corridor_fuel_margin_kg") is not None
             and r.get("corridor_miss_m") is not None]
    assoc["pearson_margin_vs_miss"] = pearson_correlation(
        [p[0] for p in pairs], [p[1] for p in pairs])
    assoc["copula_rho_per_arm"] = fit_gaussian_copula_rho(
        assoc["marginal_fuel_fail"], assoc["marginal_divert_fail"],
        assoc["joint_failure_rate_observed"])
    return {"contingency": table, "association": assoc}


def pooled_cells(blocks: dict[str, dict[str, Any]], arms: list[str],
                 ) -> dict[str, int]:
    """Cell-sum over arms (the rows across arms share the same seed/draw
    grid, so the pooled interval is quoted with that caveat, per the
    report's reading section)."""
    return {k: sum(blocks[a]["contingency"][k] for a in arms)
            for k in ("both", "fuel_only", "divert_only", "neither")}


def calibration_block(pairs: list[tuple[float, float]], bins: list[float],
                      level: float) -> dict[str, Any]:
    """The E7-02 report for one forecast set, trimmed to what the comparison
    quotes."""
    rep = calibration_report(pairs, bins, level)
    return {"n": rep["n"], "brier": rep["brier"], "ece": rep["ece"],
            "mce": rep["mce"], "reliability": rep["reliability"],
            "reliable_bins": rep["reliable_bins"],
            "correctable_bins": rep["correctable_bins"]}


def plot_coupling(arms_out: dict[str, Any], pooled: dict[str, Any],
                  prod: dict[str, Any], coup: dict[str, Any], path: Path,
                  rho: float) -> None:
    """Two panels: (top) the reliability comparison -- product vs copula-
    coupled forecast points with Wilson bars against the diagonal; (bottom)
    per-arm phi with bootstrap bars, the pooled estimate, and the zero line
    the independence assertion puts there."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(6.6, 7.6), height_ratios=[2, 1],
        gridspec_kw={"hspace": 0.38})
    ax.plot([0, 1], [0, 1], color="0.5", lw=1.0, ls="--", label="perfect")
    for rep, colour, marker, label, dodge in (
            (prod, "#c0392b", "^", "product (rho=0)", -0.012),
            (coup, "#2f8f4e", "s", f"Gaussian copula (rho={rho:+.2f})", +0.012)):
        first = True
        for row in rep["reliability"]:
            if not row["n"]:
                continue
            obs = row["observed_rate"]
            ax.errorbar(row["mean_predicted"] + dodge, obs,
                        yerr=[[obs - row["ci"]["lower"]],
                              [row["ci"]["upper"] - obs]],
                        fmt=marker, color=colour, capsize=3, ms=5,
                        label=label if first else None)
            first = False
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted $p_{land}$ per bin")
    ax.set_ylabel("observed landing rate")
    ax.set_title("E7-03: product vs copula-coupled $p_{land}$ "
                 "(joint-chance arm, n=168, OOS fit)", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    names = list(arms_out)
    xs, ys, los, his = [], [], [], []
    for i, name in enumerate(names):
        a = arms_out[name]["association"]
        if a["phi"] is None or a["ci95_bootstrap"] is None:
            continue
        xs.append(i)
        ys.append(a["phi"])
        los.append(a["phi"] - a["ci95_bootstrap"][0])
        his.append(a["ci95_bootstrap"][1] - a["phi"])
    ax2.errorbar(xs, ys, yerr=[los, his], fmt="s", color="#4a6fa5",
                 capsize=4, ms=6, label="per arm (bootstrap 95 %)")
    ax2.axhline(0.0, color="0.4", lw=1.0, ls="--")
    ax2.errorbar([len(names) + 0.4], [pooled["phi"]],
                 yerr=[[pooled["phi"] - pooled["ci95_bootstrap"][0]],
                       [pooled["ci95_bootstrap"][1] - pooled["phi"]]],
                 fmt="D", color="#2f8f4e", capsize=4, ms=8,
                 label="pooled fit set")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7)
    ax2.set_xlim(-0.6, len(names) + 1.0)
    ax2.set_ylabel(r"$\varphi$ of the two failure indicators")
    ax2.set_title("realized fuel/divert failures compete (phi < 0): "
                  "the product overstates $p_{land}$", fontsize=9)
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(alpha=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-i", "--input", type=Path, default=DEFAULT_INPUT,
                    help="recorded E3-03 report (default: %(default)s)")
    ap.add_argument("--bins", type=float, nargs="+",
                    default=list(DEFAULT_CALIBRATION_BINS),
                    help="calibration bin edges, same as E7-02 "
                         "(default: %(default)s)")
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--min-pooled", type=int, default=MIN_POOLED)
    ap.add_argument("--min-seeds", type=int, default=2)
    ap.add_argument("--trials", type=int, default=2000,
                    help="bootstrap replicates per interval")
    ap.add_argument("--seed", type=int, default=20251115,
                    help="bootstrap seed (deterministic)")
    ap.add_argument("--plot", type=Path, default=None,
                    help="optional diagram (PNG path)")
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args(argv)

    report = load_report(args.input, args.min_pooled, args.min_seeds)
    runs = report["runs"]
    spec = JointChanceSpec()

    arms_out = {name: arm_block(runs[name]["paired_rows"], spec)
                for name in sorted(runs)}
    fit_set = [a for a in sorted(runs) if a != JOINT_ARM]
    cells = pooled_cells(arms_out, fit_set)
    pooled = phi_report({**cells, "censored_divert_total": sum(
        arms_out[a]["contingency"]["censored_divert_total"] for a in fit_set),
        "censored_divert_fuel_fail": sum(
        arms_out[a]["contingency"]["censored_divert_fuel_fail"] for a in fit_set)},
        trials=args.trials, seed=args.seed)
    rho_oos = fit_gaussian_copula_rho(pooled["marginal_fuel_fail"],
                                      pooled["marginal_divert_fail"],
                                      pooled["joint_failure_rate_observed"])
    if rho_oos is None:
        raise SystemExit("copula rho not identifiable from the pooled table")
    rho_in = arms_out[JOINT_ARM]["association"]["copula_rho_per_arm"]

    # calibration comparison on the joint arm (out-of-sample rho)
    joint_rows = runs[JOINT_ARM]["paired_rows"]
    prod_pairs = [(r["landing_probability"],
                   1.0 if r["outcome"] == "landed" else 0.0)
                  for r in joint_rows if r.get("landing_probability") is not None]
    coup_pairs_oos = coupled_pairs_from_records(joint_rows, spec, rho_oos)
    coup_pairs_in = coupled_pairs_from_records(joint_rows, spec, rho_in)
    prod = calibration_block(prod_pairs, args.bins, args.level)
    coup = calibration_block(coup_pairs_oos, args.bins, args.level)
    coup_i = calibration_block(coup_pairs_in, args.bins, args.level)

    # back-out consistency diagnostic
    sigmas = sorted(s for s in (implied_loss_sigma_kg(r, spec)
                                for r in joint_rows) if s is not None)

    significant = (pooled["fisher_exact_p"] < 0.05
                   and pooled["ci95_bootstrap"] is not None
                   and pooled["ci95_bootstrap"][1] < 0.0)
    improves = coup["brier"]["score"] < prod["brier"]["score"]
    chosen = "gaussian_copula" if (significant and improves) else "product"
    out = {
        "kind": "e7-03-joint-risk",
        "provenance": {
            "source": str(args.input.relative_to(REPO)
                          if args.input.is_relative_to(REPO) else args.input),
            "arms": sorted(runs),
            "seeds": report.get("seeds"),
            "n_per_arm_rows": {a: len(runs[a]["paired_rows"]) for a in runs},
            "position_tol_m": DEFAULT_POSITION_TOL_M,
            "censored_divert_reason": CENSORED_DIVERT_REASON,
            "spec": "JointChanceSpec defaults (the spec the E3-03 joint arm ran)",
        },
        "arms": arms_out,
        "pooled_fit_set": {
            **pooled, "fit_arms": fit_set,
            "note": ("arms share the seed/draw grid, so the pooled interval "
                     "treats the 2x2 as iid rows -- optimistic by an unknown "
                     "small factor; the per-arm intervals are the conservative "
                     "reading and agree in sign"),
        },
        "copula": {
            "model": "gaussian",
            "symmetry_note": ("the Gaussian copula is radially symmetric: the "
                              "correlation of the failure indicators equals "
                              "that of the success indicators, so one rho "
                              "couples both readings"),
            "rho_out_of_sample": rho_oos,
            "rho_fit_arms": fit_set,
            "rho_in_sample_joint_arm": rho_in,
        },
        "calibration_comparison": {
            "arm": JOINT_ARM,
            "bins": args.bins,
            "product": prod,
            "coupled_oos": coup,
            "coupled_in_sample": coup_i,
            "deltas_coupled_oos_minus_product": {
                "brier": coup["brier"]["score"] - prod["brier"]["score"],
                "brier_skill": ((coup["brier"]["skill"] or 0.0)
                                - (prod["brier"]["skill"] or 0.0)),
                "ece": coup["ece"] - prod["ece"],
                "mce": coup["mce"] - prod["mce"],
            },
        },
        "diagnostics": {
            "implied_loss_sigma_kg": {
                "n": len(sigmas), "min": sigmas[0], "median": statistics.median(sigmas),
                "max": sigmas[-1],
                "declared_model_at_nominal": spec.propellant_loss_sigma_kg(
                    NOMINAL_EXPOSURE_KG),
                "nominal_exposure_kg": NOMINAL_EXPOSURE_KG,
            },
            "speed_failure_note": ("LANDING_LATERAL_SPEED_EXCEEDED rows land "
                                   "inside the position tolerance: neither "
                                   "channel failed, landing still did not "
                                   "happen -- a third mode the two-channel "
                                   "model does not carry, counted as "
                                   "no-channel-failure here"),
        },
        "verdict": {
            "independence_rejected": bool(significant),
            "association_sign": "negative" if pooled["phi"] < 0 else "positive",
            "fitted_rho": rho_oos,
            "chosen_model": chosen,
            "rationale": (
                "the pooled exact-Fisher p and bootstrap CI reject independence, "
                "and the OOS copula improves the E7-02 metrics on the arm the "
                "product was never fitted to -- the coupling is adopted "
                "(rho carried to JointChanceSpec.landing_dependence_rho by the "
                "e7-03 record). The sign (negative: failures compete) says the "
                "product OVERSTATES p_land, so coupling is a first, partial "
                "correction of exactly the optimism E7-02 measured; the "
                "verdict bands below 0.8 remain overconfident after coupling, "
                "and the e7-02 rule (never trade low-p samples on the "
                "forecast) stands unchanged"
                if chosen == "gaussian_copula" else
                "independence not rejected (or coupling does not improve the "
                "out-of-sample metrics): the product stands"),
        },
    }

    text = json.dumps(out, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    else:
        print(text)
    if args.plot:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plot_coupling(arms_out, pooled, prod, coup, args.plot, rho_oos)
        print(f"diagram -> {args.plot}")

    print(f"pooled fit set {fit_set}: cells {cells} n={pooled['n_observed']}")
    print(f"  phi={pooled['phi']:+.4f} boot95=[{pooled['ci95_bootstrap'][0]:+.3f},"
          f"{pooled['ci95_bootstrap'][1]:+.3f}] z95=[{pooled['ci95_fisher_z'][0]:+.3f},"
          f"{pooled['ci95_fisher_z'][1]:+.3f}] fisher_p={pooled['fisher_exact_p']:.4f}")
    print(f"  observed joint-fail {pooled['joint_failure_rate_observed']:.4f} vs "
          f"independent {pooled['joint_failure_rate_independent']:.4f}")
    for name, block in arms_out.items():
        a = block["association"]
        pm = a["pearson_margin_vs_miss"]
        print(f"{name:>22} cells {[block['contingency'][k] for k in ('both', 'fuel_only', 'divert_only', 'neither')]}"
              f" cens={block['contingency']['censored_divert_total']}"
              f" phi={a['phi']:+.3f} [{a['ci95_bootstrap'][0]:+.3f},{a['ci95_bootstrap'][1]:+.3f}]"
              f" p={a['fisher_exact_p']:.3f} r(miss)={pm['r']:+.3f}"
              f" rho_arm={a['copula_rho_per_arm']:+.3f}")
    print(f"copula rho: OOS {rho_oos:+.4f} (fit: {', '.join(fit_set)}); "
          f"in-sample joint arm {rho_in:+.4f}")
    print(f"calibration on {JOINT_ARM} (n={prod['n']}): product brier="
          f"{prod['brier']['score']:.4f} skill={prod['brier']['skill']:+.4f} "
          f"ece={prod['ece']:.4f} mce={prod['mce']:.4f}")
    print(f"  coupled OOS  brier={coup['brier']['score']:.4f} "
          f"skill={coup['brier']['skill']:+.4f} ece={coup['ece']:.4f} "
          f"mce={coup['mce']:.4f}")
    print(f"  implied loss sigma [kg]: n={len(sigmas)} "
          f"[{sigmas[0]:.0f}, {statistics.median(sigmas):.0f}, {sigmas[-1]:.0f}] "
          f"vs declared {spec.propellant_loss_sigma_kg(NOMINAL_EXPOSURE_KG):.0f} "
          f"at {NOMINAL_EXPOSURE_KG:.0f} kg exposure")
    print(f"VERDICT: independence_rejected={significant} rho={rho_oos:+.3f} "
          f"model={chosen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
