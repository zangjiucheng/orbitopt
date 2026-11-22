"""Checks for orbitopt.flightlab.recovery.redesign (E10-06 graded admission).

Tolerances and expectations -- declared up front, against the recorded
campaigns (docs/benchmarks/e7_04_failure_driven.json training grid and
e3_03_robust_montecarlo.json FL-305 pairing, both committed artifacts):

    REFIT_IS_SHIPPED      LandingCalibration.fit over the recorded
                          e7_04_joint_base rows reproduces
                          DEFAULT_CALIBRATION within floating-point
                          roundoff. Repeated fits on one platform are
                          bit-identical (deterministic Newton IRLS).
    GRADED_REGRADE        The recorded training grid regrades 99 -> 84
                          joint rejections, false rejections (joint-
                          rejected, FL-305 landed) 20 -> 10, newly
                          admitted 15 rows going 10 landed / 5 failed
                          under the paired baseline, all 51 deep
                          violations kept, zero previously-accepted rows
                          re-rejected.
    GATE_ORDER            A violating row is rejected by the deepest
                          applicable gate: deep > second-channel >
                          probability-floor > admitted.
    PLANNER_SMOKE         The pinned-script joint rollout rejects under
                          the binary gate and under floor/deep policies
                          that must reject it, and comes back feasible
                          only under a policy whose gates admit a
                          shallow single-channel violation.

Determinism (refit twice) and the stdlib-only import surface of the
module itself are checked too; the shipped coefficient literals are the
ones the E10-06 benchmark asserts.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from orbitopt.flightlab.recovery.chance import JointChanceSpec, JointVerdict
from orbitopt.flightlab.recovery.redesign import (
    DEFAULT_CALIBRATION, DEFAULT_MIN_ADMISSION_PROBABILITY, GATE_ADMITTED,
    GATE_DEEP, GATE_FLOOR, GATE_SATISFIES, GATE_SECOND, AdmissionPolicy,
    LandingCalibration, PHYSICAL_FEATURES, FULL_FEATURES, auc, candidate_rank,
    landing_achievable_metrics, regrade)

REPO_ROOT = Path(__file__).resolve().parents[1]
E7_04 = REPO_ROOT / "docs" / "benchmarks" / "e7_04_failure_driven.json"
E3_03 = REPO_ROOT / "docs" / "benchmarks" / "e3_03_robust_montecarlo.json"

SPEC = JointChanceSpec()


@pytest.fixture(scope="module")
def grid() -> dict:
    e704 = json.loads(E7_04.read_text())
    arm = next(a for a in e704["arms"] if a["name"] == "e7_04_joint_base")
    e303 = json.loads(E3_03.read_text())
    base = {(r["seed"], r["idx"]): r
            for r in e303["runs"]["fl305_baseline"]["paired_rows"]}
    return {"rows": arm["paired_rows"], "base": base}


# -- the calibrated model --------------------------------------------------


def test_default_calibration_is_the_recorded_refit(grid):
    fitted = LandingCalibration.fit(grid["rows"], SPEC)
    assert fitted.features == DEFAULT_CALIBRATION.features
    assert fitted.coefficients == pytest.approx(DEFAULT_CALIBRATION.coefficients, rel=1e-14)
    assert LandingCalibration.fit(grid["rows"], SPEC) == fitted


def test_calibration_is_monotone_in_the_physical_margins():
    cal = DEFAULT_CALIBRATION
    p0 = cal.probability({PHYSICAL_FEATURES[0]: 1.5, PHYSICAL_FEATURES[1]: 0.0})
    p_fuel = cal.probability({PHYSICAL_FEATURES[0]: 2.5, PHYSICAL_FEATURES[1]: 0.0})
    p_div = cal.probability({PHYSICAL_FEATURES[0]: 1.5, PHYSICAL_FEATURES[1]: 0.02})
    assert p_fuel > p0 > p_div  # E7-01 semantics: no plan buys probability
    # ... and the fit weights carry exactly those signs
    assert cal.coefficients[1] > 0.0 and cal.coefficients[2] < 0.0


def test_fit_rejects_degenerate_inputs(grid):
    rows = [r for r in grid["rows"] if r["outcome"] == "landed"]
    with pytest.raises(ValueError, match="both landed and failed"):
        LandingCalibration.fit(rows, SPEC)
    with pytest.raises(ValueError, match="unknown calibration feature"):
        LandingCalibration.fit(grid["rows"], SPEC, ("nope",))
    with pytest.raises(ValueError):
        LandingCalibration(features=(), coefficients=(0.0,))


def test_metrics_and_probability_round_trip(grid):
    for r in grid["rows"][:25]:
        m = landing_achievable_metrics(r, SPEC)
        assert m is not None and m["consistent"]
        p = DEFAULT_CALIBRATION.probability(m)
        assert 0.0 < p < 1.0
    assert landing_achievable_metrics({"outcome": "landed"}, SPEC) is None


# -- the admission policy ---------------------------------------------------


def _row_by_gate(grid) -> dict:
    found: dict[str, dict] = {}
    pol = AdmissionPolicy()
    for r in grid["rows"]:
        d = pol.decide(r, SPEC)
        if d and d["gate"] not in found and d["gate"] != GATE_SATISFIES:
            found[d["gate"]] = (r, d)
        if len(found) == 4:
            break
    return found


def test_every_gate_fires_on_the_recorded_grid_in_order(grid):
    found = _row_by_gate(grid)
    assert set(found) == {GATE_DEEP, GATE_SECOND, GATE_FLOOR, GATE_ADMITTED}
    for gate, (r, d) in found.items():
        assert d["admitted"] == (gate == GATE_ADMITTED)
        v = landing_achievable_metrics(r, SPEC)
        if gate == GATE_DEEP:
            assert v["worst"] > AdmissionPolicy().deep_violation
        elif gate == GATE_SECOND:
            assert v["worst"] <= AdmissionPolicy().deep_violation
            assert v["second_worst"] > AdmissionPolicy().second_violation_max
        elif gate == GATE_FLOOR:
            assert v["second_worst"] <= AdmissionPolicy().second_violation_max
            assert d["calibrated_probability"] < DEFAULT_MIN_ADMISSION_PROBABILITY
        else:
            assert d["calibrated_probability"] >= DEFAULT_MIN_ADMISSION_PROBABILITY


def test_admission_only_ever_unrejects(grid):
    pol = AdmissionPolicy()
    for r in grid["rows"]:
        d = pol.decide(r, SPEC)
        if d["worst"] <= SPEC.violation_tolerance:
            assert d["gate"] == GATE_SATISFIES and d["admitted"]


def test_admits_verdict_gate_ladder():
    pol = AdmissionPolicy()

    def verdict(p_fuel: float, violations: tuple) -> JointVerdict:
        return JointVerdict(p_fuel=p_fuel, p_divert=0.9,
                            p_land=0.9 * p_fuel, loss_sigma_kg=1000.0,
                            required_margin_kg=1000.0,
                            divert_budget_m=SPEC.divert_budget_m(),
                            violations=tuple(float(v) for v in violations))

    # rows the old gate accepted are untouched
    assert pol.admits_verdict(SPEC, verdict(0.99, (0.0, 0.0, 0.0)))
    # deep divert violation: rejected regardless of margins
    assert not pol.admits_verdict(SPEC, verdict(0.999, (0.0, 0.6, 0.0)))
    # second channel binding (worst 0.3 fuel, second 0.2 probability) while
    # the calibrated probability is high: the SECOND gate is what rejects
    assert not pol.admits_verdict(SPEC, verdict(0.99, (0.3, 0.0, 0.2)))
    # shallow single-channel violation with thin margins: the FLOOR rejects
    assert not pol.admits_verdict(SPEC, verdict(0.6, (0.1, 0.0, 0.0)))
    # shallow single-channel violation with fat margins: admitted
    shallow = verdict(0.99, (0.1, 0.0, 0.0))
    assert pol.admits_verdict(SPEC, shallow)
    # the same shallow violation fails a floor set above its p_hat
    assert not AdmissionPolicy(min_admission_probability=0.999).admits_verdict(
        SPEC, shallow)


def test_policy_option_coercion_and_rejects():
    pol = AdmissionPolicy(calibration=DEFAULT_CALIBRATION, deep_violation=0.4)
    assert AdmissionPolicy.from_option(pol) is pol
    rt = AdmissionPolicy.from_option(pol.to_dict())
    assert rt == pol
    assert AdmissionPolicy.from_option(
        {"deep_violation": 0.3, "calibration": DEFAULT_CALIBRATION.to_dict()}
    ).deep_violation == 0.3
    with pytest.raises(ValueError, match="unknown joint_admission"):
        AdmissionPolicy.from_option({"tolerance": 0.1})


# -- the recorded regrade ----------------------------------------------------


def test_regrade_recorded_numbers(grid):
    rows, base = grid["rows"], list(grid["base"].values())
    r = regrade(rows, SPEC, AdmissionPolicy(), baseline_rows=base)
    assert r["old_rejections"] == 99 and r["new_rejections"] == 84
    assert r["rejection_gates"] == {GATE_DEEP: 51, GATE_SECOND: 15,
                                    GATE_FLOOR: 18, GATE_ADMITTED: 15}
    assert r["false_rejections_old"] == 20 and r["false_rejections_new"] == 10
    assert r["newly_admitted"] == 15
    assert r["newly_admitted_by_paired_baseline"] == {"landed": 10, "failed": 5}
    assert r["deep_rejections_kept"] == 51
    assert r["previously_accepted_now_rejected"] == 0
    assert r["new_rejection_severity"] == {"near": 18, "mid": 30, "deep": 51}


def test_regrade_tighter_policy_rejects_more(grid):
    rows = grid["rows"]
    loose = regrade(rows, SPEC, AdmissionPolicy())
    tight = regrade(rows, SPEC, AdmissionPolicy(second_violation_max=0.0))
    assert tight["new_rejections"] > loose["new_rejections"]
    never = regrade(rows, SPEC, AdmissionPolicy(min_admission_probability=1.0))
    assert never["new_rejections"] == 99  # every violation floor-rejected


# -- ranking -----------------------------------------------------------------


def test_candidate_rank_admissible_first(grid):
    rows = grid["rows"]
    rank = candidate_rank(rows, SPEC, AdmissionPolicy())
    assert len(rank) == len(rows)
    assert sorted(e["index"] for e in rank) == list(range(len(rows)))
    admitted = [not e["admitted"] for e in rank]
    assert admitted == sorted(admitted)  # no rejected row above an admissible one
    viol_admitted = [e for e in rank
                     if e["admitted"] and e["gate"] == GATE_ADMITTED
                     and rows[e["index"]].get("reason") == "JOINT_CHANCE_VIOLATED"]
    assert len(viol_admitted) == 15  # matches the recorded regrade


def test_auc_hand_computed_and_edge_cases():
    assert auc([0.2, 0.4, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert auc([0.9, 0.8, 0.4, 0.2], [0, 0, 1, 1]) == 0.0
    assert abs(auc([0.5, 0.5, 0.1, 0.9], [0, 1, 0, 1]) - 0.875) < 1e-12
    with pytest.raises(ValueError, match="both classes"):
        auc([0.1, 0.2], [1, 1])


# -- FL-302 planner wiring -----------------------------------------------------

import sys  # noqa: E402
sys.path.insert(0, str(REPO_ROOT / "tests"))
from test_flightlab_recovery_boostback import (  # noqa: E402
    PAD, VEHICLE_YAML, _SEED_PLAN, _sep_state)
from orbitopt.flightlab.recovery import (  # noqa: E402
    BoosterModel, RecoveryTarget, plan_boostback_entry)

SEP = dict(alt=40_000.0, speed=250.0, fpa_deg=2.0, down_m=8_000.0, mass=65_000.0)
# the pinned script's handoff violates the divert channel by 0.38; relaxing
# the fuel-quantile and probability bars to 0.5 leaves it a SHALLOW,
# single-channel violation -- exactly what the graded rule exists to judge
SMOKE_SPEC = JointChanceSpec(scatter_budget_m=1400.0, p_fuel_min=0.5,
                             min_landing_probability=0.5)


@pytest.fixture(scope="module")
def vehicle():
    return BoosterModel.load(str(VEHICLE_YAML))


@pytest.fixture(scope="module")
def pad_target():
    return RecoveryTarget(kind="rtls", name="launch_site", latitude_deg=PAD[0],
                          longitude_deg=PAD[1], altitude_m=PAD[2])


def _smoke_plan(vehicle, pad_target, **admission):
    return plan_boostback_entry(_sep_state(**SEP), vehicle, pad_target,
                                initial_guess=_SEED_PLAN, max_fevals=1,
                                joint_chance=SMOKE_SPEC,
                                landing_divert_budget_m=None,
                                landing_reserve_margin_kg=0.0, **admission)


def test_planner_admits_shallow_violation_only_with_policy(vehicle, pad_target):
    """Pinned script, spec rebuilt so its 0.38 divert violation is shallow
    and single-channel: the binary gate rejects it (E7-01 semantics
    unchanged by default), the default policy also rejects it (the
    calibrated probability of a 0.38 divert overhang sits below the floor
    -- the policy is not a rubber stamp), a floor-0 policy admits it, and
    tightening the deep bar below 0.38 rejects it again."""
    assert _smoke_plan(vehicle, pad_target).reason_codes == ("JOINT_CHANCE_VIOLATED",)
    assert not _smoke_plan(vehicle, pad_target,
                           joint_admission=AdmissionPolicy()).feasible
    assert _smoke_plan(vehicle, pad_target,
                       joint_admission={"min_admission_probability": 0.0}).feasible
    assert not _smoke_plan(vehicle, pad_target,
                           joint_admission={"deep_violation": 0.05,
                                            "min_admission_probability": 0.0}).feasible


def test_planner_joint_admission_requires_joint_chance(vehicle, pad_target):
    with pytest.raises(ValueError, match="requires joint_chance"):
        plan_boostback_entry(_sep_state(**SEP), vehicle, pad_target,
                             initial_guess=_SEED_PLAN, max_fevals=1,
                             joint_admission=AdmissionPolicy())
