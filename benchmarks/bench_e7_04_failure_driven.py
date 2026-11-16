"""Benchmark: E7-04 failure-driven optimization -- fly the fix, re-pair it.

E3-03 put the finger on the failures (docs/benchmarks/e3_03_robust_
montecarlo.md): ``initial_state`` ranked first in *every* arm (Spearman
0.38-0.47) and the E3-02 joint-chance objective regressed exactly the two
categories its handoff exposes -- ``CORRIDOR_MISSED`` 5 -> 22 and
``LANDING_LATERAL_SPEED_EXCEEDED`` 24 -> 32. E7-04 attacks them through library knobs
(all off-by-default, every preset bit-exact):

* ``landing.tvc_lag_compensation`` -- the guidance fix, E3-01's command-side
  lag compensation. The lateral-speed failures ARE the declared 0.2 s TVC
  lag: the terminal divert limit-cycles against a gimbal the command cannot
  track (E3-01 measured lateral failures 8 -> 5 from it), and E3-03's
  fairness pin deliberately flew WITHOUT compensation -- so the identified
  failure mode is one the guidance law can already see but was not allowed
  to correct. This arm turns it back on.
* ``landing.cutoff_lateral_speed_m_s`` -- the terminal gate. The FL-305
  cutoff checks only the *vertical* ballistic arrival while the referee
  grades the *lateral* touchdown speed (limit 1.0 m/s); across the
  sub-second cutoff hop the lateral speed is ballistic, so this gate keeps
  the engine lit (the PD divert law still in charge) until the hop is
  laterally soft before the one-way shutdown. Measured here as a
  no-harm/no-gain tail safety in the E3-03 world (gate-only re-flew the
  baseline histogram exactly, 45/168): failures arrive at the 3 m gate
  window with 2-4 m/s lateral, too much for the window to null -- they are
  born in the divert, upstream, which is why compensation carries the fix.
* ``corridor_aim_margin_m`` -- the planning margin, opt-in (default 0 here).
  Grading the miss against ``radius - margin`` aims deeper; measured on
  this suite it trades ``INSUFFICIENT_PROPELLANT_RESERVE`` down for
  ``LANDING_LATERAL_SPEED_EXCEEDED`` up (shorter divert saves propellant
  but hands the landing a faster handoff), so it is a dial, not the fix.

Pairing. The FL-305-semantics arms pair against the RECORDED E3-03 rows
(docs/benchmarks/e3_03_robust_montecarlo.json): the draw stream is keyed
``[seed, idx]``, so sample (s, i) of a new flight saw the identical physical
world, and the gate-only re-fly reproducing the recorded histogram bit for
bit proves it. The joint-chance arm does NOT: E7-01's feasibility
semantics landed after the E3-03 report was flown, and the recorded
``e3_02_joint_chance`` rows are stale for pairing (a plain re-fly of that
arm on current HEAD rejects 27/42 with a reason code the recorded report
never contains). The joint fix therefore pairs against a FRESH current-HEAD
re-fly of the joint arm -- ``e7_04_joint_base`` -- on the identical grid.
Arms flown under E3-03's exact dispersions, actuator limits, seeds 1-4
(42/seed, 168 pooled):

* ``e7_04_guidance_fix`` -- FL-305 objective + lag compensation + gate,
  paired against recorded ``fl305_baseline``;
* ``e7_04_joint_base`` -- the joint arm re-flown on current HEAD (baseline);
* ``e7_04_joint_fix`` -- the joint arm + lag compensation + gate, paired
  against ``e7_04_joint_base``.

Every comparison is the Holm-corrected per-category exact McNemar family
of :mod:`orbitopt.flightlab.recovery.failure_driven` (the E3-03 lesson:
raw p = 0.019 does not survive a family), plus the recorded-vs-flown
``initial_state`` sensitivity shift the task demands reporting. Real
physics, ~10-15 s per sample per core (only the joint baseline is
re-flown; the FL-305 baseline is read from the recorded report):

    PYTHONPATH=src python benchmarks/bench_e7_04_failure_driven.py \
        --workers 8 -o docs/benchmarks/e7_04_failure_driven.json \
        --md docs/benchmarks/e7_04_failure_driven.md \
        --plot docs/assets/e7_04_failure_driven.png

    # smoke run of the machinery (small n; comparisons then under-powered):
    PYTHONPATH=src python benchmarks/bench_e7_04_failure_driven.py \
        --seeds 1 --n-per-seed 4 --workers 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))  # the E2E ascent fixture builder

from orbitopt.flightlab.contract import load_target  # noqa: E402
from orbitopt.flightlab.recovery.failure_driven import (  # noqa: E402
    category_pair_table,
    holm_family,
    sensitivity_shift,
    verdict_summary,
)
from orbitopt.flightlab.recovery.montecarlo import (  # noqa: E402
    MC_PLAN_OPTIONS,
    Dispersions,
    run_monte_carlo,
)
from orbitopt.flightlab.recovery.stats import (  # noqa: E402
    failure_breakdown,
    source_sensitivity,
    success_rate_interval,
)
from orbitopt.flightlab.recovery.uncertainty import UncertaintyModel  # noqa: E402

#: E7-04's library knobs (defaults here, overridable on the CLI): the
#: lateral gate at 80 % of the 1.0 m/s referee limit -- shut the engine
#: down only while the ballistic hop is guaranteed-gradeable -- and the
#: corridor planning inset, which this suite measured as a propellant /
#: lateral-speed trade and therefore ships at 0 (off) here. The library
#: ships both neutral (0 = off); these are the values *this benchmark*
#: proposes, not defaults.
DEFAULT_CUTOFF_GATE_M_S = 0.8
DEFAULT_AIM_MARGIN_M = 0.0

FL305_OBJECTIVE = {"landing_divert_budget_m": None, "landing_reserve_margin_kg": 0.0}
E302_CHANCE = {**FL305_OBJECTIVE, "joint_chance": {}}
#: the E3-03 fairness pin: 20 deg/s slew + 0.2 s lag in every arm, lag
#: SUFFERED not compensated (the recorded arms all flew naive -- including
#: the baseline and the joint arm; the E3-01 recorded arm flew compensated)
DECLARED_ACTUATOR = {"tvc_rate_limit_deg_s": 20.0, "tvc_lag_tau_s": 0.2}
#: the fix: the same declared chain with E3-01's command-side compensation
COMPENSATED = {**DECLARED_ACTUATOR, "tvc_lag_compensation": True}

DEFAULT_E3_03_REPORT = REPO / "docs" / "benchmarks" / "e3_03_robust_montecarlo.json"

#: candidate arms, their pairing (a recorded E3-03 arm or a freshly flown
#: baseline in this same run), and what each adds.
CONFIGS: dict[str, dict[str, Any]] = {
    "e7_04_guidance_fix": {
        "options": lambda gate, margin: {**MC_PLAN_OPTIONS, **FL305_OBJECTIVE,
                                         "corridor_aim_margin_m": margin,
                                         "landing": {**COMPENSATED,
                                                     "cutoff_lateral_speed_m_s": gate}},
        "vs": ["fl305_baseline"],
        "what": "FL-305 objective + E3-01 lag compensation + the E7-04 lateral cutoff gate",
    },
    "e7_04_joint_base": {
        "options": lambda gate, margin: {**MC_PLAN_OPTIONS, **E302_CHANCE,
                                         "landing": dict(DECLARED_ACTUATOR)},
        "vs": [],
        "what": ("joint-chance arm re-flown on current HEAD -- the fresh paired "
                 "baseline for joint_fix (the recorded joint rows predate E7-01's "
                 "feasibility semantics and are stale for pairing)"),
    },
    "e7_04_joint_fix": {
        "options": lambda gate, margin: {**MC_PLAN_OPTIONS, **E302_CHANCE,
                                         "corridor_aim_margin_m": margin,
                                         "landing": {**COMPENSATED,
                                                     "cutoff_lateral_speed_m_s": gate}},
        "vs": ["e7_04_joint_base"],
        "what": "joint chance + lag compensation + lateral cutoff gate",
    },
}

#: zero-flight-cost evidence pairs already inside the recorded report: the
#: compensation-only pair (the fix's core, measured at no new cost) and the
#: regression the task names
RECORDED_PAIRS: list[tuple[str, str, str]] = [
    ("e3_01_declared_chain", "fl305_baseline",
     "lag compensation alone, FL-305 objective (recorded, zero re-flight)"),
    ("e3_02_joint_chance", "fl305_baseline",
     "the E3-02 regression this task must undo (recorded context)"),
]

TARGETS = ("CORRIDOR_MISSED", "LANDING_LATERAL_SPEED_EXCEEDED")


def _landed(row: dict[str, Any]) -> bool:
    return row["outcome"] == "landed"


def _paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        out.append({"seed": r["seed"], "idx": r["idx"], "outcome": r["outcome"],
                    "reason": r["reason"],
                    "corridor_miss_m": r.get("corridor_miss_m"),
                    "corridor_fuel_margin_kg": r.get("corridor_fuel_margin_kg"),
                    "position_error_m": r.get("position_error_m"),
                    "touchdown_east_m": r.get("touchdown_east_m"),
                    "touchdown_north_m": r.get("touchdown_north_m"),
                    "landing_probability": r.get("landing_probability"),
                    "joint_violation": r.get("joint_violation")})
    return out


def _seed_list(text: str) -> list[int]:
    vals = [int(t) for t in text.replace(",", " ").split()]
    if not vals or any(v < 1 for v in vals):
        raise argparse.ArgumentTypeError("seeds must be positive integers")
    return vals


def _arm_summary(rows: list[dict[str, Any]], model: UncertaintyModel) -> dict[str, Any]:
    landed = sum(1 for r in rows if _landed(r))
    return {"success_rate": landed / len(rows) if rows else None,
            "success_rate_ci": success_rate_interval(landed, len(rows)),
            "failure_classes": failure_breakdown(rows),
            "sensitivity": source_sensitivity(rows, model)}


def _comparison(candidate_rows: list[dict[str, Any]], base_rows: list[dict[str, Any]],
                *, base_sens: dict[str, Any], cand_sens: dict[str, Any],
                allow_partial: bool = False
                ) -> dict[str, Any]:
    table = category_pair_table(candidate_rows, base_rows)
    if table["n_pairs"] != len(candidate_rows) or (table["n_base_only"]
                                                   and not allow_partial):
        raise SystemExit(
            f"pairing broken: {table['n_pairs']} shared coordinates of "
            f"{len(candidate_rows)} flown, {table['n_base_only']} recorded "
            f"rows unmatched -- the recorded arm and the new flight must "
            f"cover the identical [seed, idx] grid (use --allow-partial to "
            f"compare on the common subset anyway, smoke runs only)")
    fam = holm_family(table)
    return {"pairing": ("exact [seed, idx] grid" if not table["n_base_only"]
                        else f"partial: {table['n_pairs']} shared coordinates"),
            "family": fam,
            "sensitivity_initial_state": sensitivity_shift(cand_sens, base_sens),
            "verdict": verdict_summary(fam, targets=list(TARGETS))}


def _md(report: dict[str, Any]) -> str:
    d = report

    def comp_lines(c: dict[str, Any]) -> list[str]:
        kind = "recorded" if c.get("baseline_kind") == "recorded" else "fresh re-fly"
        out = [f"### `{c['candidate']}` vs {kind} `{c['baseline']}`"
               + (f" -- {c['note']}" if c.get("note") else ""), "",
               f"landed pair: exact McNemar p = {c['landed_p']:.4g}"
               f"{' (Holm-significant)' if c['landed_reject'] else ' (not significant)'}, "
               f"discordants {c['landed_b']}/{c['landed_c']} (a-only/b-only), "
               f"raw rate {c['candidate_rate']:.3f} vs baseline {c['baseline_rate']:.3f}.",
               "",
               "| category | baseline | candidate | Δ | b/c | raw p | Holm p | verdict |",
               "|---|---|---|---|---|---|---|---|"]
        for row in c["family"]["categories"]:
            m = row["mcnemar"]
            out.append(f"| `{row['code']}` | {row['baseline_count']} | "
                       f"{row['candidate_count']} | {row['delta']:+d} | "
                       f"{m['successes_a_only']}/{m['successes_b_only']} | "
                       f"{m['exact']['p_value']:.4g} | {row['holm']['adjusted_p']:.4g} | "
                       f"{row['direction']} |")
        s = c["sensitivity_initial_state"]
        out += ["",
                f"`initial_state` sensitivity: |rho| {s['baseline']['abs_rank_correlation']:.3f} "
                f"-> {s['candidate']['abs_rank_correlation']:.3f} "
                f"({s['abs_rank_correlation_delta']:+.3f}), rank position "
                f"{s['baseline']['rank_position']} -> {s['candidate']['rank_position']} "
                f"(ranking {s['candidate']['ranking']}), low->high tail failure-gap shift "
                f"{s['tail_failure_gap_delta']:+.3f}.", "",
                f"Verdict: **{c['verdict']['acceptance']}**"
                + (f" -- improved: {', '.join(c['verdict']['targets_improved'])}"
                   if c["verdict"]["targets_improved"] else "")
                + (f"; counts fell: {', '.join(c['verdict']['targets_count_decreased'])}"
                   if c["verdict"]["targets_count_decreased"]
                   and not c["verdict"]["targets_improved"] else "")
                + (f"; non-target regressions: "
                   f"{', '.join(c['verdict']['non_target_regressions'])}"
                   if c["verdict"]["non_target_regressions"] else "")
                + (f" (raw non-target increases: "
                   f"{', '.join(c['verdict']['non_target_count_increases'])})"
                   if c["verdict"]["non_target_count_increases"] else "")
                + ".", ""]
        return out

    lines = ["# E7-04 failure-driven optimization: paired re-fly report",
             "",
             f"Knobs: lateral cutoff gate `{d['gate_m_s']}` m/s (referee limit "
             f"1.0 m/s), E3-01 command-side TVC lag compensation ON in the fix "
             f"arms, corridor aim margin `{d['aim_margin_m']}` m (corridor "
             f"radius {d['corridor_radius_m']} m, 0 = off). Seeds {d['seeds']}, "
             f"{d['n_per_seed']}/seed, {d['n_pooled']} pooled -- E3-03's exact "
             "dispersions and declared actuator (20 deg/s + 0.2 s lag), the "
             "identical [seed, idx] draw streams.",
             "",
             f"_Pairing notes: {d['notes']}_", "",
             "| arm | landed | reason histogram |", "|---|---|---|"]
    for arm in d["arms"]:
        lines.append(f"| `{arm['name']}` | {arm['landed']}/{arm['n']} | "
                     f"{arm['reason_histogram']} |")
    lines += ["", "## Paired comparisons (Holm-corrected per-category McNemar)", ""]
    for c in d["comparisons"]:
        lines += comp_lines(c)
    if d.get("recorded_pairs"):
        lines += ["## Evidence from the recorded report (zero re-flight)", ""]
        for c in d["recorded_pairs"]:
            lines += comp_lines(c)
    return "\n".join(lines) + "\n"


def _plot(report: dict[str, Any], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comps = report["comparisons"] + report.get("recorded_pairs", [])
    fig, axes = plt.subplots(1, len(comps) + 1, figsize=(5.0 * (len(comps) + 1), 3.6))
    for ax, c in zip(axes, comps):
        cats = [r["code"] for r in c["family"]["categories"]]
        base = [r["baseline_count"] for r in c["family"]["categories"]]
        cand = [r["candidate_count"] for r in c["family"]["categories"]]
        x = range(len(cats))
        ax.bar([i - 0.2 for i in x], base, width=0.4, label="baseline")
        ax.bar([i + 0.2 for i in x], cand, width=0.4, label="candidate")
        ax.set_xticks(list(x))
        ax.set_xticklabels([c_.replace("_", "\n", 1).replace("_", "_")
                            for c_ in cats], fontsize=7)
        ax.set_ylabel(f"count / {report['n_pooled']}")
        kind = "recorded" if c.get("baseline_kind") == "recorded" else "fresh"
        ax.set_title(f"{c['candidate']}\nvs {kind} {c['baseline']}", fontsize=9)
        ax.legend(fontsize=8)
    s_ax = axes[-1]
    deltas = [c["sensitivity_initial_state"]["abs_rank_correlation_delta"]
              for c in comps]
    s_ax.barh([f"{c['candidate']}\nvs {c['baseline']}" for c in comps], deltas,
              color=["#2f7d4f" if dv <= 0 else "#b3423a" for dv in deltas])
    s_ax.axvline(0.0, color="k", lw=0.8)
    s_ax.set_title("initial_state |Spearman rho| shift\n(negative = less state-sensitive)",
                   fontsize=9)
    s_ax.tick_params(labelsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    from test_flightlab_simulate_recovery import (  # noqa: E402
        TARGET, VEHICLE, _ascent_with_textbook_separation)

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=_seed_list, default=[1, 2, 3, 4], metavar="LIST",
                    help="fixed seed list (default 1,2,3,4 -- E3-03's frozen "
                         "list; the recorded baselines only exist on it)")
    ap.add_argument("--n-per-seed", type=int, default=42,
                    help="samples per seed and arm (default 42: E3-03's grid, "
                         "the only one the recorded baselines exist on)")
    ap.add_argument("--arms", default="guidance,joint",
                    help="which candidate arms to fly: guidance, joint (flies "
                         "joint_fix against a fresh joint_base), or both")
    ap.add_argument("--gate-m-s", type=float, default=DEFAULT_CUTOFF_GATE_M_S,
                    help=f"lateral cutoff gate (default {DEFAULT_CUTOFF_GATE_M_S})")
    ap.add_argument("--aim-margin-m", type=float, default=DEFAULT_AIM_MARGIN_M,
                    help=f"corridor aim margin (default {DEFAULT_AIM_MARGIN_M})")
    ap.add_argument("--e3-03-report", type=Path, default=DEFAULT_E3_03_REPORT)
    ap.add_argument("--allow-partial", action="store_true",
                    help="compare on the [seed, idx] intersection even when the "
                         "flown grid does not cover the recorded report's (smoke "
                         "runs; the full 4x42 grid never needs it)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--plot", type=Path, default=None)
    ap.add_argument("--md", type=Path, default=None)
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args(argv)

    wanted: list[str] = []
    if "guidance" in args.arms:
        wanted.append("e7_04_guidance_fix")
    if "joint" in args.arms:
        wanted += ["e7_04_joint_base", "e7_04_joint_fix"]
    if not wanted:
        raise SystemExit("--arms must name at least one of guidance,joint")

    e3 = json.loads(args.e3_03_report.read_text(encoding="utf-8"))
    if not args.allow_partial and (list(e3["seeds"]) != list(args.seeds)
            or int(e3["n_per_seed"]) != int(args.n_per_seed)):
        raise SystemExit(f"pairing against the recorded E3-03 report requires its "
                         f"grid: report says seeds {e3['seeds']}, "
                         f"n_per_seed {e3['n_per_seed']}, CLI says {args.seeds} "
                         f"and {args.n_per_seed}")

    target = load_target(TARGET)
    ascent = _ascent_with_textbook_separation()
    disp = Dispersions()
    model = UncertaintyModel.from_dispersions(disp)
    n_total = args.n_per_seed * len(args.seeds)

    report: dict[str, Any] = {
        "kind": "e7-04-failure-driven-paired-refly",
        "gate_m_s": float(args.gate_m_s), "aim_margin_m": float(args.aim_margin_m),
        "corridor_radius_m": 8000.0,
        "seeds": list(args.seeds), "n_per_seed": int(args.n_per_seed),
        "n_pooled": int(n_total),
        "dispersions": disp.to_dict(),
        "ascent": "tests/test_flightlab_simulate_recovery.py::"
                  "_ascent_with_textbook_separation",
        "baselines_recorded_in": str(args.e3_03_report.relative_to(REPO)),
        "fairness": "identical ascent, dispersions, [seed, idx] draw streams and "
                    "declared actuator (20 deg/s + 0.2 s lag) as the recorded "
                    "E3-03 arms; the ONLY differences are the E7-04 knobs named "
                    "in gate_m_s / lag compensation / aim_margin_m; FL-305-mode "
                    "baselines are the recorded rows, never re-flown -- sample "
                    "(s, i) is the same physical world",
        "notes": ("pairing validity: a gate-only re-fly of the FL-305 objective "
                  "reproduced the recorded fl305_baseline histogram exactly "
                  "(45/168, every reason count), so FL-305-mode recorded rows "
                  "are bit-valid baselines on current HEAD; the JOINT-mode "
                  "recorded rows are NOT (E7-01 feasibility semantics landed "
                  "after them -- a plain re-fly rejects with JOINT_CHANCE_"
                  "VIOLATED, a reason the recorded report never contains), hence "
                  "the fresh e7_04_joint_base; the gate alone re-flew baseline "
                  "verdicts unchanged (no-harm/no-gain tail safety), so the "
                  "guidance fix rides on E3-01's lag compensation; the aim "
                  "margin trades INSUFFICIENT_PROPELLANT_RESERVE down for "
                  "LANDING_LATERAL_SPEED_EXCEEDED up and ships at 0 here"),
        "arms": [], "comparisons": [], "recorded_pairs": []}

    flown: dict[str, list[dict[str, Any]]] = {}
    for name in wanted:
        cfg = CONFIGS[name]
        opts = cfg["options"](args.gate_m_s, args.aim_margin_m)
        rows: list[dict[str, Any]] = []
        t0 = time.perf_counter()
        for seed in args.seeds:
            res = run_monte_carlo(target, VEHICLE, ascent, dispersions=disp,
                                  seed=seed, n=args.n_per_seed,
                                  workers=args.workers, plan_options=opts)
            rows.extend({"seed": int(seed), **dict(rec)} for rec in res.samples)
        landed = sum(1 for r in rows if _landed(r))
        hist = dict(sorted(_hist(rows).items()))
        flown[name] = rows
        report["arms"].append({
            "name": name, "what": cfg["what"],
            "plan_options": {k: (list(v) if isinstance(v, tuple) else v)
                             for k, v in opts.items()},
            "n": len(rows), "landed": landed, "reason_histogram": hist,
            "pooled": _arm_summary(rows, model),
            "paired_rows": _paired_rows(rows)})
        print(f"{name}: success {landed / n_total:.3f} ({landed}/{n_total}) "
              f"{hist} ({time.perf_counter() - t0:.0f} s)", flush=True)

    def _rows_of(name: str) -> list[dict[str, Any]]:
        if name in flown:
            return flown[name]
        return e3["runs"][name]["paired_rows"]

    def _sens_of(name: str) -> dict[str, Any]:
        if name in flown:
            return next(a["pooled"]["sensitivity"] for a in report["arms"]
                        if a["name"] == name)
        return e3["runs"][name]["pooled"]["sensitivity"]

    def _rate_of(name: str) -> float:
        if name in flown:
            return next(a["landed"] / a["n"] for a in report["arms"]
                        if a["name"] == name)
        return e3["runs"][name]["pooled"]["success_rate"]

    def _pair_entry(cand: str, base: str, kind: str,
                    note: str = "") -> dict[str, Any]:
        comp = _comparison(_rows_of(cand), _rows_of(base),
                           base_sens=_sens_of(base), cand_sens=_sens_of(cand),
                           allow_partial=args.allow_partial)
        fam = comp["family"]
        return {"candidate": cand, "baseline": base, "baseline_kind": kind,
                "note": note,
                "pairing": comp["pairing"],
                "candidate_rate": _rate_of(cand),
                "baseline_rate": _rate_of(base),
                "landed_p": fam["landed_pair"]["exact"]["p_value"],
                "landed_reject": fam["landed_pair"]["holm"]["reject"],
                "landed_b": fam["landed_pair"]["successes_a_only"],
                "landed_c": fam["landed_pair"]["successes_b_only"],
                "family": fam,
                "sensitivity_initial_state": comp["sensitivity_initial_state"],
                "verdict": comp["verdict"]}

    for name in wanted:
        for base in CONFIGS[name]["vs"]:
            kind = "flown" if base in flown else "recorded"
            entry = _pair_entry(name, base, kind)
            report["comparisons"].append(entry)
            v = entry["verdict"]
            print(f"paired {name} vs {base} ({kind}): "
                  f"acceptance={v['acceptance']!r} "
                  f"improved={v['targets_improved']} "
                  f"regressions={v['non_target_regressions']} "
                  f"landed p={entry['landed_p']:.4g}", flush=True)

    report["recorded_pairs"] = []
    for cand, base, note in RECORDED_PAIRS:
        if cand in e3["runs"] and base in e3["runs"]:
            entry = _pair_entry(cand, base, "recorded", note)
            report["recorded_pairs"].append(entry)
            v = entry["verdict"]
            print(f"recorded pair {cand} vs {base}: improved={v['targets_improved']} "
                  f"regressions={v['non_target_regressions']}", flush=True)

    text = json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    if args.md is not None:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(_md(report), encoding="utf-8")
    if args.plot is not None:
        _plot(report, args.plot)
    if args.out is None and args.md is None:
        print(text)
    return 0


def _hist(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[r["reason"]] = out.get(r["reason"], 0) + 1
    return out


if __name__ == "__main__":
    sys.exit(main())
