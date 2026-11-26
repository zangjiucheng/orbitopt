"""E11R-02 (task #415): the E11-06 entry peak-q discrepancy is attributed, not widened.

E11-06 recorded the independent twin 18.2 % off the main model's entry peak q against the
unchanged 12 % band (``reflight.entry_peak_q``) with no mechanism named. This task's contract
is that the gap gets an explanation and the tolerance does NOT move. These gates read the
committed budget ``docs/benchmarks/e11r_02_peak_q_error_budget.json`` (regenerate with
``E11_ENGINE_STAGES=1 E11_FINS_FIXED=1 python
benchmarks/bench_e11r_02_peak_q_error_budget.py --out
docs/benchmarks/e11r_02_peak_q_error_budget.json``, ~1 min) and re-check the mechanism it
names against the live code, so a future change to either atmosphere model has to move the
attribution too.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from orbitopt.entry import _ATMOSPHERE, density_kg_m3 as main_density
from orbitopt.verify import full_recovery_reflight as fr

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "docs/benchmarks/e11r_02_peak_q_error_budget.json"
ARCHIVE = (REPO / "tests/data/mission/reference"
           / "e11-06-starbase-continuous-burn-reports/twin.json")
SEGMENT_BASE_M = 25_000.0     # first base height of the published table below 100 km


@pytest.fixture(scope="module")
def rep() -> dict:
    return json.loads(BENCH.read_text(encoding="utf-8"))


def _twin_rho(h_m: float) -> float:
    return float(fr._twin_us76(h_m)[1])


def _ratio(h_m: float) -> float:
    return main_density(h_m) / _twin_rho(h_m)


class TestBandWasNotWidened:
    def test_the_graded_band_is_still_twelve_percent(self, rep):
        assert fr.BANDS["reflight.entry_peak_q"][1] == pytest.approx(12.0)
        assert rep["verdict"]["band_entry_peak_q_pct_UNCHANGED"] == pytest.approx(12.0)

    def test_the_gate_is_still_red_and_says_so(self, rep):
        v = rep["verdict"]
        assert v["inside_band"] is False
        assert abs(v["measured_delta_pct"]) > 12.0
        assert v["implementation_differences_in_this_band"] == []

    def test_the_reported_delta_is_the_number_the_gate_archived(self, rep):
        archived = json.loads(ARCHIVE.read_text(encoding="utf-8"))
        assert float(archived["delta"]["peak_q_pct"]) > 12.0          # red in the archive
        assert rep["archive"]["delta_pct"] == pytest.approx(
            float(archived["delta"]["peak_q_pct"]), abs=0.01)
        assert rep["verdict"]["reproduces_archive"] is True
        assert abs(rep["verdict"]["measured_delta_pct"]
                   - rep["archive"]["delta_pct"]) < 1.0               # env drift, not a new gap


class TestBudgetCloses:
    def test_the_two_lines_are_the_delta(self, rep):
        b, v = rep["budget_pts_of_main_peak_q"], rep["verdict"]
        assert rep["closure_error_pts"] < 1e-9
        assert b["density_profile_form"] + b["everything_else_unexplained"] == pytest.approx(
            v["measured_delta_pct"], abs=0.02)                         # rounded report lines

    def test_the_profile_form_line_carries_the_whole_gap(self, rep):
        b = rep["budget_pts_of_main_peak_q"]
        assert b["density_profile_form"] > abs(b["everything_else_unexplained"]) * 3
        assert abs(b["density_profile_form"]) > rep["verdict"]["measured_delta_pct"]
        # the line over-shoots: the rest of the twin pulls the OTHER way, so the gap
        # cannot be closed by any other single difference either
        assert b["everything_else_unexplained"] < 0.0

    def test_step_size_and_gravity_are_not_the_mechanism(self, rep):
        b = rep["budget_pts_of_main_peak_q"]
        for line in ("numerics_step_halving", "gravity_j2",
                     "density_height_convention_geopotential", "entry_window_definition"):
            assert abs(b[line]) < 1.0, line

    def test_the_form_line_is_one_segment_of_the_profile(self, rep):
        b, v = rep["budget_pts_of_main_peak_q"], rep["verdict"]
        assert v["share_of_form_line_inside_0_to_25km_segment_pct"] > 95.0
        assert abs(b["density_profile_form_above_25km_segment"]) < 1.0
        assert b["density_profile_form_below_25km_segment"] == pytest.approx(
            b["density_profile_form"], abs=1.0)


class TestMechanismIsLive:
    def test_the_main_table_flies_one_exponential_over_the_whole_troposphere(self):
        # the root of the line: the published piecewise-exponential table has NO base
        # height between the ground and 25 km, so 0-25 km is a single exponential with
        # scale height 7.249 km, while US76 has layer bases at 11 and 20 km
        bases = [float(r[0]) for r in _ATMOSPHERE if float(r[0]) < 100.0]
        assert bases == [0.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
        assert float(_ATMOSPHERE[0][2]) == pytest.approx(7.249)

    def test_the_two_profiles_agree_at_sea_level(self):
        assert main_density(0.0) == pytest.approx(_twin_rho(0.0), rel=2e-3)

    def test_the_two_profiles_split_inside_the_segment_and_agree_above_it(self):
        below = np.linspace(2_000.0, 22_000.0, 21)
        above = np.linspace(25_000.0, 100_000.0, 16)
        worst_below = max(abs(1.0 - _ratio(float(h))) for h in below)
        worst_above = max(abs(1.0 - _ratio(float(h))) for h in above)
        assert worst_below > 0.20                       # where the entry peak lives
        assert worst_above < 0.10                       # and q is negligible there

    def test_the_profiles_split_where_the_peak_actually_is(self, rep):
        for key in ("twin_entry_peak_q_height_m", "main_entry_peak_q_height_m"):
            h = float(rep["rerun"][key])
            assert 0.0 < h < SEGMENT_BASE_M, key         # inside the disagreeing segment
        assert _ratio(float(rep["rerun"]["twin_entry_peak_q_height_m"])) < 0.85
        # the twin's own peak-q locus is now recorded, which is what proves the peak is
        # inside the segment instead of asserted
        assert float(rep["rerun"]["variants"]["twin"]["entry_peak_q_height_m"]) > 0.0
        assert rep["rerun"]["twin_entry_peak_q_t_s"] > 0.0

    def test_the_peak_lives_in_the_segment_the_line_is_blamed_for(self, rep):
        v = rep["verdict"]
        assert 0.0 < v["peak_q_height_twin_m"] < SEGMENT_BASE_M
        assert 0.0 < v["peak_q_height_main_m"] < SEGMENT_BASE_M
        assert v["peak_q_height_twin_m"] > v["peak_q_height_main_m"]   # denser air -> earlier peak
