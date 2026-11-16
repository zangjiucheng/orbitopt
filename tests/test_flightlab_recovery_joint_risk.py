"""Checks for orbitopt.flightlab.recovery.joint_risk (E7-03, task #279).

The acceptance criteria this file pins:

* the bivariate normal CDF is exact where an identity exists
  (Phi_2(h, h; rho) = 1/4 + asin(rho)/(2 pi)), reduces to the product at
  rho = 0, and respects the Frechet bounds everywhere -- the coupled
  ``p_land`` is therefore a probability, not a heuristic;
* ``JointChanceSpec.landing_dependence_rho = 0`` (the default) keeps
  ``evaluate_joint_chance`` byte-identical to the shipped product, and a
  negative rho raises p_land above it (failures compete: the product
  understates the joint success -- the direction the e7-03 evidence
  measures);
* the channel indicator rules read the recorded reason-code vocabulary as
  documented: censored (never-flown) diverts are NOT counted as successes,
  corridor-never-reached rows evaluate neither channel;
* the contingency, association and copula-fit primitives: phi/Fisher
  round-trips on hand-computable tables, deterministic bootstrap, a rho
  fit that inverts its own moment match;
* the per-row back-out is exact on product-mode records and refuses
  coupled specs; the fitted constant agrees with the recorded e7-03
  verdict JSON.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from orbitopt.flightlab.recovery.chance import (
    JointChanceSpec,
    bivariate_normal_cdf,
    coupled_joint_probability,
    evaluate_joint_chance,
    phi,
)
from orbitopt.flightlab.recovery.joint_risk import (
    FITTED_LANDING_DEPENDENCE_RHO,
    back_out_component_probabilities,
    bootstrap_phi_ci,
    contingency_table,
    coupled_pairs_from_records,
    failure_channels,
    fit_gaussian_copula_rho,
    fisher_exact_two_sided_p,
    frechet_phi_bounds,
    implied_loss_sigma_kg,
    pearson_correlation,
    phi_coefficient,
    phi_report,
)

REPO = Path(__file__).resolve().parents[1]


class TestBivariateNormal:
    def test_corner_identity(self):
        # P(X <= 0, Y <= 0) = 1/4 + asin(rho)/(2 pi) -- exact for every rho.
        for rho in (-0.9, -0.273, -0.05, 0.3, 0.95):
            got = bivariate_normal_cdf(0.0, 0.0, rho)
            want = 0.25 + math.asin(rho) / (2.0 * math.pi)
            assert got == pytest.approx(want, abs=2e-6)

    def test_reduces_to_product_at_zero(self):
        assert bivariate_normal_cdf(-0.4, 1.1, 0.0) == pytest.approx(
            phi(-0.4) * phi(1.1), abs=1e-12)

    def test_monotone_and_frechet_bounded(self):
        prev = -1.0
        for rho in (-0.95, -0.5, -0.1, 0.2, 0.7, 0.95):
            v = bivariate_normal_cdf(-0.3, 0.8, rho)
            assert v >= prev - 1e-9
            prev = v
            assert max(0.0, phi(-0.3) + phi(0.8) - 1.0) - 1e-9 <= v \
                <= min(phi(-0.3), phi(0.8)) + 1e-9

    def test_rejects_unit_rho(self):
        with pytest.raises(ValueError, match="rho"):
            bivariate_normal_cdf(0.0, 0.0, 1.0)


class TestCoupledProbability:
    def test_rho_zero_is_the_product_exactly(self):
        assert coupled_joint_probability(0.31, 0.77, 0.0) == 0.31 * 0.77

    def test_negative_dependence_lowers_the_joint_success(self):
        # NQD: bivariate normal with rho < 0 satisfies Phi_2 <= uv for ALL
        # (u, v), so a negative coupling lowers the joint success below the
        # product (failures compete -> "at least one fails" is MORE likely
        # than the product admits -- the direction E7-02 measured).
        p1, p2 = 0.8, 0.6
        assert coupled_joint_probability(p1, p2, -0.4) < p1 * p2
        assert coupled_joint_probability(p1, p2, +0.4) > p1 * p2
        # but never below the Frechet lower bound (0.8+0.6-1 = 0.4 here)
        assert coupled_joint_probability(p1, p2, -0.99) > 0.4

    def test_symmetric_and_frechet_clipped(self):
        a = coupled_joint_probability(0.11, 0.9, -0.3)
        b = coupled_joint_probability(0.9, 0.11, -0.3)
        assert a == pytest.approx(b, abs=1e-9)
        assert max(0.0, 0.11 + 0.9 - 1.0) - 1e-12 <= a <= min(0.11, 0.9) + 1e-12

    def test_endpoint_probabilities(self):
        assert coupled_joint_probability(0.0, 0.7, -0.3) == pytest.approx(0.0, abs=1e-12)
        assert coupled_joint_probability(1.0, 0.7, 0.3) == pytest.approx(0.7, abs=1e-9)

    def test_rejects_non_probabilities(self):
        with pytest.raises(ValueError, match="marginal"):
            coupled_joint_probability(1.2, 0.5, 0.1)


class TestSpecCoupling:
    def test_default_is_the_product_unchanged(self):
        spec = JointChanceSpec()
        assert spec.landing_dependence_rho == 0.0
        v = evaluate_joint_chance(spec, miss_m=351.0, margin_kg=1662.0,
                                  exposure_kg=20_000.0)
        assert v.p_land == pytest.approx(v.p_fuel * v.p_divert, abs=0.0)

    def test_coupled_spec_evaluates_below_product_for_negative_rho(self):
        coupled = JointChanceSpec(landing_dependence_rho=-0.5)
        plain = JointChanceSpec()
        a = evaluate_joint_chance(coupled, miss_m=351.0, margin_kg=1662.0,
                                  exposure_kg=20_000.0)
        b = evaluate_joint_chance(plain, miss_m=351.0, margin_kg=1662.0,
                                  exposure_kg=20_000.0)
        assert a.p_fuel == b.p_fuel and a.p_divert == b.p_divert
        assert a.p_land < b.p_land
        assert max(0.0, a.p_fuel + a.p_divert - 1.0) <= a.p_land

    def test_rho_domain(self):
        with pytest.raises(ValueError, match="landing_dependence_rho"):
            JointChanceSpec(landing_dependence_rho=1.0)
        with pytest.raises(ValueError, match="landing_dependence_rho"):
            JointChanceSpec(landing_dependence_rho=-1.5)
        assert JointChanceSpec.from_option(
            {"landing_dependence_rho": -0.273}).landing_dependence_rho == -0.273


class TestChannelIndicators:
    def test_landed_row_is_two_channel_success(self):
        rec = {"outcome": "landed", "reason": "LANDED",
               "corridor_fuel_margin_kg": 1200.0, "corridor_miss_m": 300.0,
               "position_error_m": 6.0}
        assert failure_channels(rec) == (False, False)

    def test_corridor_miss_is_divert_failure_margin_still_observed(self):
        rec = {"outcome": "failed", "reason": "CORRIDOR_MISSED",
               "corridor_fuel_margin_kg": -50.0, "corridor_miss_m": 9000.0}
        assert failure_channels(rec) == (True, True)
        rec["corridor_fuel_margin_kg"] = 500.0
        assert failure_channels(rec) == (False, True)

    def test_corridor_abort_censors_the_divert_channel(self):
        rec = {"outcome": "failed", "reason": "INSUFFICIENT_PROPELLANT_RESERVE",
               "corridor_fuel_margin_kg": -800.0, "corridor_miss_m": 120.0}
        assert failure_channels(rec) == (True, None)

    def test_reserve_low_at_touchdown_is_fuel_failure(self):
        rec = {"outcome": "failed", "reason": "LANDING_PROPELLANT_RESERVE_LOW",
               "corridor_fuel_margin_kg": 40.0, "position_error_m": 12.0}
        assert failure_channels(rec) == (True, False)

    def test_position_tolerance_is_the_grading_rule(self):
        base = {"outcome": "failed", "reason": "X",
                "corridor_fuel_margin_kg": 500.0}
        rec = dict(base, position_error_m=40.0)
        assert failure_channels(rec)[1] is True
        rec = dict(base, position_error_m=25.0)   # exactly at the limit: inside
        assert failure_channels(rec)[1] is False

    def test_speed_failure_is_neither_channel(self):
        rec = {"outcome": "failed", "reason": "LANDING_LATERAL_SPEED_EXCEEDED",
               "corridor_fuel_margin_kg": 500.0, "position_error_m": 8.0}
        assert failure_channels(rec) == (False, False)

    def test_no_corridor_no_channels(self):
        rec = {"outcome": "failed", "reason": "INSUFFICIENT_PROPELLANT_BOOSTBACK"}
        assert failure_channels(rec) == (None, None)
        assert failure_channels({"outcome": "error"}) == (None, None)


class TestContingencyAndAssociation:
    TABLE = {"both": 2, "fuel_only": 12, "divert_only": 54, "neither": 88,
             "censored_divert": {"INSUFFICIENT_PROPELLANT_RESERVE": 12},
             "censored_divert_total": 12, "censored_divert_fuel_fail": 12,
             "not_evaluated": {}, "not_evaluated_total": 0}

    def _rows(self):
        rows = ([{"outcome": "failed", "reason": "CORRIDOR_MISSED",
                  "corridor_fuel_margin_kg": -1.0, "corridor_miss_m": 9000.0}] * 2
                + [{"outcome": "failed", "reason": "LANDING_PROPELLANT_RESERVE_LOW",
                    "corridor_fuel_margin_kg": 20.0, "position_error_m": 10.0,
                    "corridor_miss_m": 100.0}] * 12
                + [{"outcome": "failed", "reason": "INSUFFICIENT_PROPELLANT_RESERVE",
                    "corridor_fuel_margin_kg": -500.0, "position_error_m": None,
                    "corridor_miss_m": 100.0}] * 12
                + [{"outcome": "failed", "reason": "X", "corridor_fuel_margin_kg": 500.0,
                    "position_error_m": 80.0, "corridor_miss_m": 2000.0}] * 54
                + [{"outcome": "landed", "reason": "LANDED",
                    "corridor_fuel_margin_kg": 900.0, "position_error_m": 5.0,
                    "corridor_miss_m": 300.0}] * 88
                + [{"outcome": "failed", "reason": "INSUFFICIENT_PROPELLANT_BOOSTBACK"}])
        return rows

    def test_table_counts_and_partitions(self):
        t = contingency_table(self._rows())
        assert [t[k] for k in ("both", "fuel_only", "divert_only", "neither")] \
            == [2, 12, 54, 88]
        assert t["n_observed"] == 156
        assert t["censored_divert_total"] == 12
        assert t["censored_divert_fuel_fail"] == 12
        assert t["not_evaluated_total"] == 1
        assert sum(t[k] for k in ("both", "fuel_only", "divert_only", "neither",
                                  "censored_divert_total",
                                  "not_evaluated_total")) == len(self._rows())

    def test_phi_known_values(self):
        assert phi_coefficient(10, 0, 0, 10) == pytest.approx(1.0)
        assert phi_coefficient(0, 10, 10, 0) == pytest.approx(-1.0)
        assert phi_coefficient(25, 25, 25, 25) == pytest.approx(0.0)
        assert phi_coefficient(5, 0, 0, 0) is None       # degenerate column

    def test_fisher_extremes(self):
        assert fisher_exact_two_sided_p(10, 0, 0, 10) < 1e-4
        assert fisher_exact_two_sided_p(25, 25, 25, 25) > 0.9

    def test_bootstrap_deterministic_and_brackets(self):
        a = bootstrap_phi_ci(2, 12, 54, 88, trials=400, seed=7)
        b = bootstrap_phi_ci(2, 12, 54, 88, trials=400, seed=7)
        assert a == b
        assert a[0] < a[1]
        phi_hat = phi_coefficient(2, 12, 54, 88)
        assert a[0] <= phi_hat <= a[1]

    def test_phi_report_fields(self):
        rep = phi_report(self.TABLE, trials=400, seed=11)
        assert rep["n_observed"] == 156
        assert rep["phi"] == pytest.approx(
            phi_coefficient(2, 12, 54, 88), abs=1e-9)
        lo, hi = rep["ci95_fisher_z"]
        assert lo <= rep["phi"] <= hi
        assert rep["joint_failure_rate_observed"] == pytest.approx(2 / 156)
        assert rep["joint_failure_rate_independent"] == pytest.approx(
            (14 / 156) * (56 / 156))
        blo, bhi = rep["phi_bounds_censored_frechet"]
        assert blo <= bhi

    def test_frechet_bounds_none_without_censoring(self):
        plain = dict(self.TABLE, censored_divert_total=0,
                     censored_divert_fuel_fail=0)
        assert frechet_phi_bounds(plain) is None

    def test_frechet_bounds_span_the_observed_phi(self):
        # resolving the 12 censored rows either way moves phi from -0.141
        # toward (and past) zero on both sides: the censoring is material
        # but does not change the sign of the measurement.
        bounds = frechet_phi_bounds(self.TABLE)
        observed = phi_coefficient(self.TABLE["both"], self.TABLE["fuel_only"],
                                   self.TABLE["divert_only"],
                                   self.TABLE["neither"])
        assert bounds is not None and bounds[0] < bounds[1]
        assert bounds[0] <= observed <= bounds[1]

    def test_pearson_degenerate_and_perfect(self):
        xs = [1.0, 2.0, 3.0, 4.0]
        r = pearson_correlation(xs, [10.0, 8.0, 6.0, 4.0])
        assert r["r"] == pytest.approx(-1.0)
        assert r["n"] == 4
        assert pearson_correlation(xs, [2.0, 2.0, 2.0, 2.0]) is None
        assert pearson_correlation([1.0], [1.0]) is None


class TestCopulaFit:
    def test_round_trip(self):
        target = coupled_joint_probability(0.14, 0.34, -0.4)
        rho = fit_gaussian_copula_rho(0.14, 0.34, target)
        assert rho == pytest.approx(-0.4, abs=2e-3)
        # the fitted joint reproduces the observed frequency
        assert coupled_joint_probability(0.14, 0.34, rho) == pytest.approx(
            target, abs=1e-4)

    def test_independent_data_fits_zero(self):
        assert fit_gaussian_copula_rho(0.3, 0.5, 0.15) == pytest.approx(
            0.0, abs=2e-2)

    def test_unidentifiable_cases(self):
        assert fit_gaussian_copula_rho(0.0, 0.5, 0.0) is None
        # on the Frechet lower bound (mutually exclusive events): rho -> -1,
        # not identifiable from one table.
        assert fit_gaussian_copula_rho(0.3, 0.5, 0.0) is None


class TestBackOut:
    SPEC = JointChanceSpec()

    def _row(self, margin_kg: float, miss_m: float):
        sigma = self.SPEC.propellant_loss_sigma_kg(20_000.0)
        p_fuel = phi(margin_kg / sigma)
        p_div = self.SPEC.p_divert(miss_m)
        return {"outcome": "failed", "reason": "X",
                "corridor_fuel_margin_kg": margin_kg, "corridor_miss_m": miss_m,
                "landing_probability": p_fuel * p_div}, p_fuel, p_div

    def test_recovers_the_factors_exactly(self):
        row, p_fuel, p_div = self._row(1662.0, 351.0)
        got = back_out_component_probabilities(row, self.SPEC)
        assert got[0] == pytest.approx(p_fuel, abs=1e-9)
        assert got[1] == pytest.approx(p_div, abs=1e-9)

    def test_implied_sigma_recovers_the_model_sigma(self):
        row, *_ = self._row(1662.0, 351.0)
        sigma = self.SPEC.propellant_loss_sigma_kg(20_000.0)
        assert implied_loss_sigma_kg(row, self.SPEC) == pytest.approx(
            sigma, rel=1e-9)

    def test_rows_without_a_forecast_back_out_to_none(self):
        assert back_out_component_probabilities(
            {"outcome": "failed", "corridor_miss_m": 100.0}, self.SPEC) is None

    def test_inconsistent_rows_back_out_to_none(self):
        row = {"outcome": "failed", "corridor_miss_m": 20_000.0,
               "corridor_fuel_margin_kg": 1000.0, "landing_probability": 0.9}
        assert back_out_component_probabilities(row, self.SPEC) is None

    def test_coupled_specs_refuse_the_back_out(self):
        coupled = JointChanceSpec(landing_dependence_rho=-0.273)
        with pytest.raises(ValueError, match="product"):
            back_out_component_probabilities(
                {"landing_probability": 0.5, "corridor_miss_m": 100.0}, coupled)

    def test_coupled_pairs_reproduce_the_product_at_rho_zero(self):
        rows = []
        for m, miss in [(1662.0, 351.0), (200.0, 900.0), (-300.0, 500.0)]:
            row, *_ = self._row(m, miss)
            row["outcome"] = "landed"
            rows.append(row)
        pairs = coupled_pairs_from_records(rows, self.SPEC, 0.0)
        assert len(pairs) == 3
        for (p_c, landed), row in zip(pairs, rows):
            assert p_c == pytest.approx(row["landing_probability"], abs=1e-12)
            assert landed == 1.0


class TestEvidenceAgreement:
    """The recorded e7-03 run is the fitted constants' source: the module
    must carry exactly the value the benchmark recorded (and the joint arm's
    table is the one documented in the report)."""

    REPORT = REPO / "docs" / "benchmarks" / "e7_03_joint_risk.json"
    SOURCE = REPO / "docs" / "benchmarks" / "e3_03_robust_montecarlo.json"

    def test_fitted_constant_matches_the_recorded_verdict(self):
        rep = json.loads(self.REPORT.read_text())
        assert rep["kind"] == "e7-03-joint-risk"
        assert rep["verdict"]["chosen_model"] == "gaussian_copula"
        assert FITTED_LANDING_DEPENDENCE_RHO == pytest.approx(
            rep["verdict"]["fitted_rho"], abs=5e-4)
        assert rep["verdict"]["fitted_rho"] < 0.0   # failures compete

    def test_joint_arm_contingency_reproduces(self):
        source = json.loads(self.SOURCE.read_text())
        rows = source["runs"]["e3_02_joint_chance"]["paired_rows"]
        t = contingency_table(rows)
        recorded = json.loads(self.REPORT.read_text())
        got = recorded["arms"]["e3_02_joint_chance"]["contingency"]
        assert [t[k] for k in ("both", "fuel_only", "divert_only", "neither")] \
            == [got[k] for k in ("both", "fuel_only", "divert_only", "neither")]
        assert t["censored_divert_total"] == got["censored_divert_total"]
