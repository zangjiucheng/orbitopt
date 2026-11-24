"""E12-00 release gate for E11-06: the archived nominal record.

The continuous entry+landing burn must be TWO ignitions in the flown track -- not a plan-level
promise -- with ``engines_lit`` agreeing with the engine state (throttle) and the mass flow. The
archive under tests/data/mission/reference/e11-06-starbase-continuous-burn{,-reports} is what the
gate grades; nothing here relaxes a graded limit: a check the design does not pass stays recorded
as failing (see the reports README).
"""
from __future__ import annotations

import json
from pathlib import Path

from orbitopt.flightlab.contract import read_run
from orbitopt.flightlab.engine_status import burns_from_extra_columns

ARCHIVE = Path(__file__).resolve().parents[1] / "tests/data/mission/reference/e11-06-starbase-continuous-burn"
REPORTS = Path(f"{ARCHIVE}-reports")
RUN = ARCHIVE / "mission" / "recovery" / "run"


def _booster():
    return {t.body_id: t for t in read_run(RUN).tracks}["booster"]


def test_the_staged_track_carries_an_engine_count_column():
    lit = _booster().extra_columns["engines_lit"]
    assert set(int(round(float(v))) for v in lit) <= {0, 3, 5, 13}


def test_the_flown_continuous_recovery_really_ignites_twice():
    booster = _booster()
    burns = burns_from_extra_columns(booster.t_s, booster.extra_columns, booster.mass_kg)
    assert len(burns) == 2                                            # boostback; entry+landing as ONE burn
    bb, burn = burns
    assert bb.engines_known and bb.engines_text == "13"               # the whole set, for the whole boostback
    assert bb.t_end_s < burn.t_start_s                                # one coast between the two ignitions
    counts = [n for _t, n in burn.stages]
    assert counts[0] == 13                                             # the burn IGNITES on the full set
    assert all(a >= b for a, b in zip(counts, counts[1:]))             # monotone: a shut engine stays shut
    assert counts[-1] < counts[0]                                      # staged down INSIDE the burn
    thr = {float(a): float(b) for a, b in zip(booster.t_s, booster.extra_columns["throttle"])}
    assert all(thr[t] > 0.0 for t, _n in burn.stages[:-1])             # every shutdown step is still powered
    assert burn.t_end_s > burn.t_start_s + 10.0                        # a real continuous burn, not two clipped ones


def test_engines_lit_matches_the_throttle_and_the_mass_flow():
    booster = _booster()
    lit = [int(round(float(v))) for v in booster.extra_columns["engines_lit"]]
    thr = [float(v) for v in booster.extra_columns["throttle"]]
    mass = [float(v) for v in booster.mass_kg]
    t = [float(v) for v in booster.t_s]
    assert all((l > 0) == (w > 0.0) for l, w in zip(lit, thr))        # lit <=> a burn command; silent <=> no flow
    assert all(m1 >= m2 for m1, m2 in zip(mass, mass[1:]))            # mass never grows
    for i in range(len(t) - 1):
        if lit[i] > 0 and lit[i + 1] == lit[i]:
            assert mass[i] > mass[i + 1]                               # engines lit: propellant is flowing
    # engine state and flow are consistent: wherever the engine count and the commanded throttle hold
    # (within 2%) the specific consumption is stable, and it rises with both engines and throttle --
    # flow could not come from an engine the track says was shut down
    from collections import defaultdict
    bins: dict[tuple[int, float], list[float]] = defaultdict(list)
    for i in range(len(t) - 1):
        if thr[i] > 0.0 and lit[i] == lit[i + 1] > 0 and abs(thr[i + 1] - thr[i]) <= 0.02 * max(thr[i], thr[i + 1]):
            bins[(lit[i], round(thr[i], 1))].append(((mass[i] - mass[i + 1]) / (t[i + 1] - t[i])) / (thr[i] * lit[i]))
    assert len(bins) >= 3
    for (n, w), v in bins.items():
        med = sorted(v)[len(v) // 2]
        assert all(abs(r - med) / med < 0.05 for r in v), (n, w)
    # for a fixed engine set the specific consumption does not depend on the throttle level --
    # the flow follows the ENGINE STATE, not a free-floating number
    by_n: dict[int, dict[float, float]] = defaultdict(dict)
    for (n, w), v in bins.items():
        by_n[n][w] = sorted(v)[len(v) // 2]
    for n, per_w in by_n.items():
        assert len(per_w) == len({w: 0 for w in per_w})
        if len(per_w) > 1:
            vals = sorted(per_w.values())
            assert (vals[-1] - vals[0]) / vals[0] < 0.10, n


def test_the_nominal_flight_landed_and_the_landing_gates_are_graded_green():
    doc = json.loads((RUN / "verification.json").read_text())
    by = {c["id"]: c for c in doc["checks"]}
    landing = by["recovery-landing"]
    assert landing["passed"] and by["landing-position"]["value"] is not None   # assessed AND green, not a placeholder
    assert "NOT_ASSESSED" not in str(landing.get("reason", ""))
    assert by["recovery-propellant-reserve"]["passed"]
    fly = json.loads((REPORTS / "fly.json").read_text())
    assert fly["plan_feasible"] and fly["ignitions_used"] == 2
    assert all(fly["checks"].values()) and len(fly["checks"]) == 8       # the eight graded landing/recovery checks
    assert fly["touchdown"]["landed"]


def test_the_mission_record_agrees_with_the_archived_report():
    mdoc = json.loads((ARCHIVE / "mission" / "mission.verification.json").read_text())
    mchecks = json.loads((REPORTS / "mission-checks.json").read_text())
    assert mdoc["status"]["result"] == "passed" and mdoc["status"]["passed"]
    assert mchecks["checks"] == mdoc["status"]["checks"]                 # the dump is the report, not a copy of convenience
    assert all(mchecks["checks"].values()) and len(mchecks["checks"]) == 71
    for k in ("recovery.landed", "recovery.propellant_reserve", "recovery.handoff-landing-touchdown",
              "recovery.handoff-mass-terminal", "recovery.joint-propellant-corridor-budget"):
        assert mchecks["checks"][k] is True, k
    stale = {"recovery-propellant-reserve", "recovery-landing"}          # never assessed in E5-08; stay red, recorded
    for name in ("orbit-refly.json", "hs-orbit-refly.json"):
        refly = json.loads((REPORTS / name).read_text())
        assert {k for k, v in refly["checks"].items() if not v} == stale, name
        assert "insertion-periapsis-within-tolerance" in refly["checks"]  # the re-fly grades insertion


def test_the_robustness_reports_carry_the_sample_count_and_a_confidence_interval():
    for name in ("mc-sealed-pinned.json", "mc-sealed-refine.json", "mc-hs-pinned.json"):
        row = json.loads((REPORTS / name).read_text())
        assert row["seeds"] == [8, 9, 10] and row["n"] == 36
        assert sum(v["n"] for v in row["per_seed"].values()) == 36
        assert sum(v["successes"] for v in row["per_seed"].values()) == row["successes"]
        assert abs(row["rate"] - row["successes"] / 36) < 1e-9
        lo, hi = row["cluster_ci95"]
        assert lo <= row["rate"] <= hi                                   # a rate outside its own CI is a lie
