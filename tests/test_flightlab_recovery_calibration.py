"""Checks for orbitopt.flightlab.recovery.calibration (E7-02, task #278).

The acceptance criteria this file pins:

* Brier score with known reference values, including the climatology
  comparison (a forecast that cannot beat the pooled rate has zero skill);
* the reliability table bins every pair exactly once -- the bin sizes sum to
  n, boundary predictions land in the right bin, empty bins are reported,
  never dropped;
* per-bin verdicts read the prediction against the observed Wilson
  interval: a forecast well outside the flight record's interval is
  over/underconfident, a forecast inside it is consistent, and a bin too
  small to grade says so;
* ECE/MCE are the (size-weighted) gaps of the same table, and the whole
  report is a pure function of the pairs.
"""
from __future__ import annotations

import math

import pytest

from orbitopt.flightlab.recovery.calibration import (
    DEFAULT_CALIBRATION_BINS,
    VERDICT_CONSISTENT,
    VERDICT_INSUFFICIENT_DATA,
    VERDICT_OVERCONFIDENT,
    VERDICT_UNDERCONFIDENT,
    brier_score,
    calibration_report,
    pairs_from_records,
    reliability_table,
)


def _pairs(pred: float, landed: int, n: int) -> list[tuple[float, float]]:
    return [(float(pred), float(landed))] * n


class TestBrierScore:
    def test_known_reference_case(self):
        got = brier_score([(0.5, 1), (0.5, 0)])
        assert got["score"] == pytest.approx(0.25)
        assert got["reference_climatology"] == pytest.approx(0.25)
        assert got["skill"] == pytest.approx(0.0)
        assert got["n"] == 2 and got["success_rate"] == 0.5

    def test_perfect_and_worst(self):
        perfect = brier_score(_pairs(1.0, 1, 4) + _pairs(0.0, 0, 4))
        assert perfect["score"] == pytest.approx(0.0)
        worst = brier_score(_pairs(1.0, 0, 4) + _pairs(0.0, 1, 4))
        assert worst["score"] == pytest.approx(1.0)

    def test_climatology_reference_is_the_pooled_variance(self):
        # 3 landed of 12: constant-0.25 forecast Brier = p(1-p) = 0.1875.
        got = brier_score(_pairs(0.75, 0, 9) + _pairs(0.75, 1, 3))
        assert got["success_rate"] == pytest.approx(0.25)
        assert got["reference_climatology"] == pytest.approx(0.1875)

    def test_zero_variance_reference_has_no_skill_definition(self):
        got = brier_score(_pairs(1.0, 1, 3))
        assert got["reference_climatology"] == pytest.approx(0.0)
        assert got["skill"] is None


class TestValidation:
    def test_prediction_must_be_a_probability(self):
        with pytest.raises(ValueError, match="outside"):
            brier_score([(1.2, 1)])
        with pytest.raises(ValueError, match="outside"):
            reliability_table([(-0.1, 0)])

    def test_outcome_must_be_binary(self):
        with pytest.raises(ValueError, match="outcome"):
            brier_score([(0.5, 2)])
        with pytest.raises(ValueError, match="not a"):
            brier_score([(0.5, "landed")])

    def test_empty_input_is_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            calibration_report([])

    def test_bin_edges_must_span_and_increase(self):
        with pytest.raises(ValueError, match="span"):
            reliability_table([(0.5, 1)], [0.1, 0.5, 0.9])
        with pytest.raises(ValueError, match="increase"):
            reliability_table([(0.5, 1)], [0.0, 0.8, 0.4, 1.0])
        with pytest.raises(ValueError, match="two bin edges"):
            reliability_table([(0.5, 1)], [0.5])


class TestReliabilityTable:
    def test_every_pair_is_binned_exactly_once(self):
        pairs = ([(0.0, 1)] + [(0.4999, 0)] * 3 + [(0.5, 1)] * 5
                 + [(0.85, 1)] * 8 + [(0.92, 0)] * 2 + [(0.95, 1)]
                 + [(1.0, 1)] * 7)
        table = reliability_table(pairs)
        assert [r["n"] for r in table] == [4, 5, 8, 2, 8]
        assert sum(r["n"] for r in table) == len(pairs)
        assert table[0]["bin"] == [0.0, 0.5] and table[-1]["bin"] == [0.95, 1.0]

    def test_default_bins_match_the_e302_pilot_grouping_with_split_high_band(self):
        assert DEFAULT_CALIBRATION_BINS[:4] == (0.0, 0.5, 0.8, 0.9)
        assert DEFAULT_CALIBRATION_BINS[-2:] == (0.95, 1.0)

    def test_empty_bins_are_reported_not_dropped(self):
        table = reliability_table(_pairs(0.97, 1, 6))
        assert len(table) == len(DEFAULT_CALIBRATION_BINS) - 1
        empty = [r for r in table if r["n"] == 0]
        assert len(empty) == len(DEFAULT_CALIBRATION_BINS) - 2
        assert all(r["verdict"] == VERDICT_INSUFFICIENT_DATA for r in empty)
        assert all(r["observed_rate"] is None for r in empty)

    def test_row_carries_wilson_interval_of_observed_successes(self):
        # 8 landed of 10 predicted at 0.9: the same Wilson number the R-06
        # interval gives for 8/10 at 95 % (0.490-0.943).
        row, = [r for r in reliability_table(_pairs(0.9, 1, 8)
                                            + _pairs(0.9, 0, 2),
                                            [0.0, 1.0])]
        assert row["ci"]["method"] == "wilson_score"
        assert row["ci"]["lower"] == pytest.approx(0.4901, abs=1e-3)
        assert row["ci"]["upper"] == pytest.approx(0.9434, abs=1e-3)
        assert row["observed_rate"] == pytest.approx(0.8)
        assert row["mean_predicted"] == pytest.approx(0.9)
        assert row["gap"] == pytest.approx(-0.1)
        assert row["bin_brier"] == pytest.approx((8 * 0.01 + 2 * 0.81) / 10)

    def test_verdicts_read_prediction_against_observed_interval(self):
        # pred 0.9, only 5/100 land: prediction far above the interval.
        over, = [r for r in reliability_table(_pairs(0.9, 1, 5) + _pairs(0.9, 0, 95))
                 if r["n"]]
        assert over["verdict"] == VERDICT_OVERCONFIDENT
        # pred 0.1, 95/100 land: prediction far below the interval.
        under, = [r for r in reliability_table(_pairs(0.1, 1, 95) + _pairs(0.1, 0, 5))
                  if r["n"]]
        assert under["verdict"] == VERDICT_UNDERCONFIDENT
        # pred 0.5, 50/100 land: inside.
        ok, = [r for r in reliability_table(_pairs(0.5, 1, 50) + _pairs(0.5, 0, 50))
               if r["n"]]
        assert ok["verdict"] == VERDICT_CONSISTENT

    def test_small_bins_are_not_graded(self):
        # pred 0.9, 1 landed of 3: Wilson interval wide enough to matter,
        # but the min_bin_n floor says insufficient_data either way.
        row, = [r for r in reliability_table(_pairs(0.9, 1, 1) + _pairs(0.9, 0, 2))
                if r["n"]]
        assert row["verdict"] == VERDICT_INSUFFICIENT_DATA
        # ... and the very same data IS graded when the floor is lowered.
        row, = [r for r in reliability_table(_pairs(0.9, 1, 1) + _pairs(0.9, 0, 2),
                                            min_bin_n=1) if r["n"]]
        assert row["verdict"] == VERDICT_OVERCONFIDENT


class TestCalibrationReport:
    def test_ece_is_size_weighted_gap_mce_is_the_maximum(self):
        # bin [0,0.5): 10 samples pred 0.25, none land -> gap 0.25;
        # bin [0.5,1]: 40 samples pred 0.75, 30 land -> gap 0.0.
        pairs = _pairs(0.25, 0, 10) + _pairs(0.75, 1, 30) + _pairs(0.75, 0, 10)
        rep = calibration_report(pairs, [0.0, 0.5, 1.0])
        assert rep["n"] == 50
        assert rep["ece"] == pytest.approx((10 * 0.25 + 40 * 0.0) / 50)
        assert rep["mce"] == pytest.approx(0.25)

    def test_reliable_and_correctable_bin_lists(self):
        pairs = (_pairs(0.97, 1, 90) + _pairs(0.97, 0, 4)   # consistent high
                 + _pairs(0.6, 1, 2) + _pairs(0.6, 0, 48))  # overconfident mid
        rep = calibration_report(pairs)
        assert [0.95, 1.0] in rep["reliable_bins"]
        assert [0.5, 0.8] in rep["correctable_bins"]

    def test_report_reuses_the_brier_reference(self):
        pairs = _pairs(0.5, 1, 3) + _pairs(0.5, 0, 3)
        rep = calibration_report(pairs)
        assert rep["brier"] == brier_score(pairs)

    def test_report_is_a_pure_function_of_the_pairs(self):
        pairs = ([(0.3, 1)] * 4 + [(0.55, 0)] * 60 + [(0.85, 1)] * 20
                 + [(0.97, 1)] * 40)
        a = calibration_report(list(pairs))
        b = calibration_report(list(reversed(pairs)))
        assert a == b  # binning is order-free


class TestPairsFromRecords:
    def test_rows_without_a_prediction_are_skipped(self):
        rows = [{"outcome": "landed", "landing_probability": 0.9},
                {"outcome": "landed"},
                {"outcome": "failed", "reason": "X", "landing_probability": None},
                {"outcome": "failed", "landing_probability": 0.8}]
        assert pairs_from_records(rows) == [(0.9, 1.0), (0.8, 0.0)]

    def test_column_names_are_overridable(self):
        rows = [{"out": "yes", "p": 0.5}]
        assert pairs_from_records(rows, prob_key="p", landed_key="out",
                                 landed_value="yes") == [(0.5, 1.0)]
