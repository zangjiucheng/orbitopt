"""E13-07 acceptance gate -- independent closed-loop verification of a sealed
GNC run. Grades the E13-01 reference run with the E13-07 tool and attacks the
verdict: the four error classes must be listed SEPARATELY, a missing
independent backend must be recorded (never imputed), a doctored covariance or
a swapped actuator record must be caught from the record itself, and the
verification digest must bind this verdict to this run."""
import copy
import dataclasses
import importlib.util

import numpy as np
import pytest

import orbitopt.verify.gnc_closed_loop as gl
from orbitopt.gnc import (ClockSpec, ControllerSpec, EstimatorSpec, ProportionalLaw, SensorNoise,
                          SensorSpec, simulate)
from orbitopt.mission.contract import Epoch
from orbitopt.mission.multispacecraft import MultiSpacecraftMission, SpacecraftSpec
from orbitopt.verify.report import read_report

T0 = Epoch(9300.0, "TDB")
HAS_TUDATPY = importlib.util.find_spec("tudatpy") is not None


def make_clock():
    return ClockSpec(
        t0=T0, dt_dynamics_s=5.0,
        sensors=(SensorSpec("navcam", period_s=2.0, phase_s=0.5, latency_s=0.3,
                            dropped_ticks=(3, 9)),),
        estimator=EstimatorSpec(period_s=1.0, phase_s=0.0),
        controller=ControllerSpec(period_s=4.0, phase_s=0.0, latency_s=1.0))


def mission_doc():
    m = MultiSpacecraftMission(
        mission_id="e13-07-sim", title="independent closed-loop verification", frame="GCRS",
        epoch=T0, spacecraft=(SpacecraftSpec("tgt", "TARGET"), SpacecraftSpec("chaser", "CHASER")))
    return m.to_dict()


def sim_log(run_id="e13-07-run", seed=7):
    return simulate(
        run_id=run_id, clock=make_clock(), mission=mission_doc(),
        frame="GCRS", target_id="tgt", chaser_id="chaser",
        chief_r_m=(7000000.0, 0.0, 0.0), chief_v_m_s=(0.0, 7546.1, 0.0),
        deputy_r_m=(7000040.0, 0.0, 15.0), deputy_v_m_s=(0.0, 7546.1, 0.02),
        laws={"chaser": ProportionalLaw(kp_per_s=0.01, kd=0.2)},
        noise={"navcam": SensorNoise(std_position_m=1.0, std_velocity_m_s=0.02, seed=seed)},
        t_end_s=80.0, execution_scale=0.95, execution_rtol=0.05, execution_seed=3)


@pytest.fixture(scope="module")
def log():
    return sim_log()


@pytest.fixture(scope="module")
def report(log, tmp_path_factory):
    path = tmp_path_factory.mktemp("e13-07") / "gnc-closed-loop.json"
    rep = gl.verify_gnc_closed_loop(log, out_path=path)
    rep["_path"] = str(path)  # module fixture keeps the write target for the round-trip test
    return rep


# --------------------------------------------------------------------------- report shape

def test_four_error_classes_listed_separately(report):
    lines = report["inputs"]["error_classes"]
    assert set(lines) == set(gl.ERROR_CLASSES)
    for cls, line in lines.items():
        assert line["result"] in ("passed", "failed", "not_checked")
        assert line["checks"] and all(n.startswith(cls + ".") for n in line["checks"])
    # never blended into one flag: the four verdicts are independent lines
    assert len({l["result"] for l in lines.values()}) >= 2


def test_evidence_types_are_the_required_ones(report):
    types = {r["evidence"]["type"] for r in report["residuals"]}
    assert types <= {"numerical_consistency", "analytical", "independent_model"}
    for r in report["residuals"]:
        assert r["evidence"]["cannot_prove"]  # every line carries its limits (task #263)


def test_numerical_line_converges_and_matches_truth(report):
    line = report["inputs"]["error_classes"]["numerical_error"]
    assert line["result"] == "passed"
    errs = report["inputs"]["sections"]["numerical"]["errors"]
    pos = [e["max_position_error_m"] for e in errs]
    assert all(b <= max(a, gl.REPLAY_FLOOR_POSITION_M) * 1.05 for a, b in zip(pos, pos[1:]))
    assert pos[-1] <= gl.REPLAY_TOL_POSITION_M
    assert errs[-1]["max_velocity_error_m_s"] <= gl.REPLAY_TOL_VELOCITY_M_S
    # an honest flat sweep says so instead of inventing an order
    assert report["inputs"]["sections"]["numerical"]["at_float_floor"] in (True, False)


def test_missing_reflight_backend_is_recorded_never_imputed(report):
    line = report["inputs"]["error_classes"]["model_error"]
    if HAS_TUDATPY:
        assert line["result"] == "passed"
        assert report["status"]["independent_validation"]["checks_total"] >= 1
    else:
        assert line["result"] == "not_checked"
        assert line["reason"] == gl.REFLIGHT_BACKEND_MISSING
        # an absent independent line must not masquerade as validation
        assert report["status"]["independent_validation"]["checks_total"] == 0


# --------------------------------------------------------------------------- navigation review

def test_declared_covariance_of_the_sensor_stream_survives_the_nis_gate(log):
    nav = gl.navigation_consistency(log)
    assert nav["checks"]["navigation_inconsistency.observation_nis"] is True
    assert nav["observations"]["gate_violations"] == 0
    assert 0.5 <= nav["observations"]["mean_nis_per_dof"] <= 2.0  # ~chi2/dof = 1


def test_understated_covariance_is_caught(log):
    """A filter lying about its uncertainty -- same numbers, tighter sigma --
    must move the statistic even though every record still digests."""
    recs = tuple(dataclasses.replace(
        ob, covariance_position_m2=tuple(v * 1.0e-6 for v in ob.covariance_position_m2))
        for ob in log.records["observation"])
    liar = dataclasses.replace(log, records={**log.records, "observation": recs})
    nav = gl.navigation_consistency(liar)
    assert nav["checks"]["navigation_inconsistency.observation_nis"] is False
    assert nav["observations"]["gate_violations"] > 0


def test_the_reference_estimators_stale_across_burns_nees_is_flagged(log):
    """The E13-01 reference filter is a contract demo (raw latest observation,
    no propagation across actuations); the NEES review must say so, not wave
    it through because the digest verifies."""
    nav = gl.navigation_consistency(log)
    assert nav["checks"]["navigation_inconsistency.estimate_nees"] is False
    assert nav["estimates"]["gate_violations"] > 0


def test_rank_deficient_and_non_psd_covariances_are_scored_honestly():
    nis, dof = gl.nis_quadratic_form([1.0, 0.0, 0.0], [1.0, 0, 0, 0, 1.0, 0, 0, 0, 0.0])
    assert dof == 2 and pytest.approx(nis, rel=1e-12) == 1.0
    nis, dof = gl.nis_quadratic_form([1.0, 0.0, 0.0], np.zeros(9))
    assert dof == 0 and np.isnan(nis)
    with pytest.raises(ValueError, match="not PSD"):
        gl.nis_quadratic_form([1.0, 0.0, 0.0], [-1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0])


def test_chi2_gate_is_a_sane_upper_quantile():
    q = gl.chi2_upper_quantile(0.9995, 6)
    assert 20.0 < q < 25.0  # tabulated chi2(6, .9995) = 22.458
    assert gl.chi2_upper_quantile(0.95, 6) < q


# --------------------------------------------------------------------------- control / actuation review

def test_swapped_actuator_record_is_caught_from_the_record_itself(log):
    recs = list(log.records["control_event"])
    i = next(i for i, ev in enumerate(recs) if ev.executed)
    recs[i] = dataclasses.replace(recs[i], realized_delta_v_m_s=tuple(
        3.0 * x for x in recs[i].realized_delta_v_m_s))
    lied = dataclasses.replace(log, records={**log.records, "control_event": tuple(recs)})
    assert gl.maneuver_execution(log)["checks"][
        "optimizer_control_nonconvergence.execution_within_declared_bounds"] is True
    assert gl.maneuver_execution(lied)["checks"][
        "optimizer_control_nonconvergence.execution_within_declared_bounds"] is False


def test_producer_convergence_assertion_is_cross_checked_not_trusted(log):
    base = gl.maneuver_execution(log)
    assert base["convergence_assertion"] is None  # the sim declares nothing
    assert base["checks"]["optimizer_control_nonconvergence.convergence_assertion"] is None
    agree = dataclasses.replace(log, config={**log.config, "convergence":
                                            {"converged": base["settled"]}})
    assert gl.maneuver_execution(agree)["checks"][
        "optimizer_control_nonconvergence.convergence_assertion"] is True
    contradict = dataclasses.replace(log, config={**log.config, "convergence":
                                                 {"converged": not base["settled"]}})
    assert gl.maneuver_execution(contradict)["checks"][
        "optimizer_control_nonconvergence.convergence_assertion"] is False


def test_step_sweep_does_not_accept_a_flat_large_residual(log, monkeypatch):
    monkeypatch.setattr(gl, "closed_loop_replay",
                        lambda *_a, **_k: {"available": True, "max_position_error_m": 60_000.0,
                                            "max_velocity_error_m_s": 65.0})
    result = gl.numerical_convergence(log)
    assert result["observed_order"] == 0.0
    assert not any(result["checks"].values())


# --------------------------------------------------------------------------- evidence binding

def test_verification_digest_round_trips_and_rebinds(report, log):
    gl.verify_evidence_binding(report, log)
    reread = read_report(report["_path"])
    assert reread["digest"] == report["digest"]
    gl.verify_evidence_binding(reread, log)


def test_digest_binds_the_verdict_to_the_run(report, log):
    other = sim_log(run_id="e13-07-run", seed=8)
    with pytest.raises(ValueError, match="run_digest"):
        gl.verify_evidence_binding(report, other)
    flipped = copy.deepcopy(report)
    key = next(k for k, v in flipped["status"]["checks"].items()
               if k.endswith("replay_step_convergence"))
    flipped["status"]["checks"][key] = not flipped["status"]["checks"][key]
    with pytest.raises(ValueError, match="verification_digest"):
        gl.verify_evidence_binding(flipped, log)


def test_missing_truth_is_recorded_not_scored():
    short = sim_log()
    truth = tuple(r for r in short.records["truth"] if r.t_s == 0.0)
    short = dataclasses.replace(short, records={**short.records, "truth": truth})
    assert gl.numerical_convergence(short)["reason"] == gl.TRUTH_TRACK_TOO_SHORT
    rep = gl.verify_gnc_closed_loop(short)
    line = rep["inputs"]["error_classes"]["numerical_error"]
    assert line["result"] == "not_checked"
    assert line["reason"] == gl.TRUTH_TRACK_TOO_SHORT


@pytest.mark.skipif(not HAS_TUDATPY, reason="the independent re-flight needs tudatpy")
def test_reflight_compares_a_burn_instant_sample_before_the_impulse(log):
    """A truth sample taken at an actuation instant is the PRE-impulse state (the run's event
    priority, as closed_loop_replay states). Reading the dense output unpinned at a seam gave the
    post-impulse state and reported the burn's own delta-v (~0.45 m/s here) as a model error."""
    burns = [e for e in log.records["control_event"] if e.executed]
    on_grid = [e for e in burns if any(abs(r.t_s - e.t_exec_s) < 1e-9 for r in log.records["truth"])]
    assert on_grid  # the scenario must put a burn on a truth sample, or this test checks nothing
    fly = gl.independent_reflight(log)
    assert fly["available"]
    biggest_burn = max(float(np.linalg.norm(e.realized_delta_v_m_s)) for e in on_grid)
    assert biggest_burn > 10 * fly["max_relative_velocity_error_m_s"]
    assert fly["within_tolerance"]
