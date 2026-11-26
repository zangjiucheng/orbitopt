"""E11R-01 (task #414): robust candidate selection -- the committed campaign is internally consistent.

The campaign (``tests/data/mission/reference/e11r-01-robust-candidate-selection/campaign.json``, produced by
``benchmarks/bench_e11r_01_robust_candidate_selection.py``, ~hours on a 40-worker box) ranks each search seed's
harvested feasible pool on training seeds only and flies the argmax pinned on the sealed 36-sample hold-out. These
gates do not re-run the campaign; they re-derive every number the report states from the data it ships, so the
write-up cannot drift from the record, and they lock what the campaign does NOT show:

* the selected arms are NOT significantly better than the sealed 33/36 champion (Holm-corrected, all p = 1) -- the
  significant result is selected vs the same run's first-feasible stop;
* the pooled 179/180 counts the same 36 hold-out samples once per search seed, so its cluster CI only reflects
  search-seed variation, not hold-out sampling.
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO / "tests/data/mission/reference/e11r-01-robust-candidate-selection/campaign.json"

#: the sealed E11-06 landing gates, which this campaign must not relax
SEALED_LIMITS = {"max_touchdown_speed_m_s": 3.0, "max_touchdown_lateral_speed_m_s": 1.0,
                 "min_propellant_reserve_kg": 100000.0, "entry_q_max_pa": 250000.0, "entry_g_max": 12.0}


@pytest.fixture(scope="module")
def rep() -> dict:
    return json.loads(CAMPAIGN.read_text(encoding="utf-8"))


def _mcnemar_exact(a_only: int, b_only: int) -> float:
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n)


def test_sealed_champion_is_reproduced(rep):
    assert rep["sealed_reproduced"] == {"matches_record_33_36": True, "n": 36, "successes": 33}


def test_selection_uses_training_seeds_disjoint_from_the_holdout(rep):
    manifest = rep["seed_manifest"]
    assert not set(manifest["train_seeds"]) & set(manifest["holdout_seeds"])
    assert not set(manifest["search_seeds"]) & set(manifest["holdout_seeds"])
    assert "training seeds only" in rep["protocol"]["ranking"]


def test_each_selected_arm_is_the_training_argmax_of_its_pool(rep):
    """landings, then mean corridor fuel margin, then (lower) mean miss, then harvest order."""
    for row in rep["per_seed"]:
        seed = row["seed"]
        pool = {arm: s for arm, s in rep["scores"].items() if arm.startswith(f"cand-s{seed}-")}
        assert len(pool) == row["pool_kept"]
        best = min(pool, key=lambda arm: (-pool[arm]["successes"], -pool[arm]["mean_fuel_margin_kg"],
                                          pool[arm]["mean_miss_m"], int(arm.rsplit("-", 1)[1])))
        assert best == row["selected_arm"], (seed, best, row["selected_arm"])


def test_pooled_and_dispersion_statistics_match_the_per_seed_rows(rep):
    selected = [row["holdout_selected"]["successes"] for row in rep["per_seed"]]
    stop = [row["holdout_stop"]["successes"] for row in rep["per_seed"]]
    pooled = rep["verdict"]["pooled_selected"]
    assert (pooled["successes"], pooled["n"], pooled["clusters"]) == (sum(selected), 180, 5) == (179, 180, 5)
    rates_sel = [k / 36 for k in selected]
    rates_stop = [k / 36 for k in stop]
    d = rep["verdict"]
    assert d["dispersion_selected"]["range"] == pytest.approx(max(rates_sel) - min(rates_sel), abs=1e-4)
    assert d["dispersion_stop"]["range"] == pytest.approx(max(rates_stop) - min(rates_stop), abs=1e-4)
    assert d["dispersion_selected"]["range"] < d["dispersion_stop"]["range"]
    # the stop points are the archived E12-00 first-feasible table
    assert stop == d["archived_table"]["successes"]


def test_paired_p_values_are_the_exact_mcnemar_of_their_tables(rep):
    for key in ("paired_vs_sealed", "paired_selected_vs_stop"):
        for pair in rep[key]:
            t = pair["table"]
            assert t["both"] + t["a_only"] + t["b_only"] + t["neither"] == pair["n_pairs"] == 36
            assert pair["diff"] == t["a_only"] - t["b_only"]
            assert pair["p_raw"] == pytest.approx(_mcnemar_exact(t["a_only"], t["b_only"]), rel=1e-9)


def test_holm_correction_over_the_sealed_comparisons(rep):
    pairs = rep["paired_vs_sealed"]
    m = len(pairs)
    order = sorted(range(m), key=lambda i: pairs[i]["p_raw"])
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pairs[i]["p_raw"]))
        assert pairs[i]["p_holm"] == pytest.approx(running, rel=1e-9)


def test_selected_arms_are_not_claimed_better_than_the_sealed_champion(rep):
    """The honest reading: 35-36/36 vs 33/36 is not significant after Holm (p = 1 here)."""
    assert all(not pair["significant"] for pair in rep["paired_vs_sealed"])
    assert all(pair["p_holm"] == 1.0 for pair in rep["paired_vs_sealed"])


def test_selection_never_loses_a_sample_the_first_feasible_stop_landed(rep):
    for pair in rep["paired_selected_vs_stop"]:
        assert pair["table"]["b_only"] == 0
        assert pair["diff"] >= 0


def test_graded_limits_and_the_fin_assumption_are_not_relaxed(rep):
    audit = rep["constraint_audit"]
    assert audit["graded_limits"] == SEALED_LIMITS
    assert audit["grid_fin_drag_is_design_assumption"] is True
