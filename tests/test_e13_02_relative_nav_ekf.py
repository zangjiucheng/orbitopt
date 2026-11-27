"""E13-02 (task #419): relative-navigation EKF -- unit math, contract seam,
and the gates on the committed Monte-Carlo evidence
(docs/benchmarks/e13_02_relative_nav_ekf.json).

Unit layer: the process models reproduce the E12-02 propagators, every
measurement Jacobian matches finite differences, the filter keeps P a valid
covariance, zero-noise tracking converges, the contract seam converts and
publishes correctly, and the observability helpers rank what theory says.
Evidence layer: RMSE / NEES / NIS / coverage pass in the declared chi-square
bands for the Cartesian cells, the drag-mismatch cell FAILS consistency (the
mismatch must be visible), and the rank/null-space claims hold. Regenerate
with

    PYTHONPATH=src python benchmarks/bench_e13_02_relative_nav_ekf.py \
        -o docs/benchmarks/e13_02_relative_nav_ekf.json

(~1 min, CPU only)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from orbitopt.gnc import (NATIVE_COVARIANCE_KEY, NATIVE_MEASUREMENT_KEY,
                          CwProcess, GNCContractError, NonlinearRelativeProcess,
                          RangeAngle, RangeRate, RelativeNavEKF, RelativePosition,
                          RelativeVelocity, Stacked, chi_square_interval,
                          chi_square_cdf, estimate_from_document, frame_covariance,
                          is_valid_record_document, nees, nis, observability_rank,
                          rmse, sigma_coverage, stacked_observability,
                          white_noise_acceleration_q)
from orbitopt.gnc.contracts import Estimate, Observation
from orbitopt.mission.contract import Epoch
from orbitopt.mission.multispacecraft import RelativeState
from orbitopt.relative.cw import propagate_cw
from orbitopt.relative.frames import lvlh_to_eci
from orbitopt.relative.propagation import EARTH_MU, propagate_pair

DOC = Path(__file__).resolve().parents[1] / "docs/benchmarks/e13_02_relative_nav_ekf.json"

EPOCH = Epoch(9300.0, "TDB")
CR0 = np.array([7071000.0, 0.0, 0.0])
CV0 = np.array([0.0, np.sqrt(EARTH_MU / 7071000.0), 0.0])
N_REF = float(np.sqrt(EARTH_MU / 7071000.0 ** 3))
X0 = np.array([300.0, 12.0, -6.0, -0.01, 0.04, 0.005])
U0 = np.array([0.0, 1.0e-3, 0.0])


def _chief_pair(times):
    tr = propagate_pair(CR0, CV0, CR0.copy(), CV0.copy(), times, frame="GCRS")
    return tr.chief_r[0, :, :], tr.chief_v[0, :, :]


def _cartesian_obs(dr, dv, p_cov, v_cov, rec="obs-1", sensor="native-cart"):
    rs = RelativeState(target_id="tgt", chaser_id="chaser", epoch=EPOCH, frame="GCRS",
                       r_rel_m=tuple(float(v) for v in dr),
                       v_rel_m_s=tuple(float(v) for v in dv))
    return Observation(record_id=rec, sensor_id=sensor, t_sample_s=10.0, t_available_s=10.0,
                       latency_s=0.0, relative_state=rs,
                       covariance_position_m2=tuple(float(v) for v in np.asarray(p_cov).reshape(9)),
                       covariance_velocity_m2_s2=tuple(float(v) for v in np.asarray(v_cov).reshape(9)))


def _native_obs(z, r, rec="obs-2", sensor="range-angle"):
    rs = RelativeState(target_id="tgt", chaser_id="chaser", epoch=EPOCH, frame="GCRS",
                       r_rel_m=(100.0, 0.0, 0.0), v_rel_m_s=(0.0, 0.0, 0.0))
    return Observation(record_id=rec, sensor_id=sensor, t_sample_s=10.0, t_available_s=10.0,
                       latency_s=0.0, relative_state=rs,
                       covariance_position_m2=tuple(float(v) for v in np.eye(3).reshape(9)),
                       metadata={NATIVE_MEASUREMENT_KEY: tuple(z),
                                 NATIVE_COVARIANCE_KEY: tuple(float(v) for v in
                                                              np.asarray(r).reshape(-1))})


# ---------------------------------------------------------------------------
# process models == E12-02
# ---------------------------------------------------------------------------


def test_cw_process_reuses_the_e12_cw_propagator():
    # the impulse is B=[0;I]: it lands on the velocity block before the CW step
    proc = CwProcess(N_REF)
    dt = 45.0
    assert np.allclose(proc.f(X0, U0, dt), np.concatenate(propagate_cw(X0[:3], X0[3:] + U0, N_REF, dt)))
    assert np.allclose(proc.f(X0, np.zeros(3), dt), np.concatenate(propagate_cw(X0[:3], X0[3:], N_REF, dt)))


def test_cw_jacobian_matches_fd_of_f():
    proc = CwProcess(N_REF)
    fd = np.empty((6, 6))
    for i in range(6):
        h = 1e-6 * max(abs(X0[i]), 1.0)
        xp, xm = X0.copy(), X0.copy()
        xp[i] += h
        xm[i] -= h
        fd[:, i] = (proc.f(xp, U0, 60.0) - proc.f(xm, U0, 60.0)) / (2.0 * h)
    assert np.max(np.abs(proc.jacobian(X0, U0, 60.0) - fd)) < 1e-3


def test_nonlinear_process_matches_propagate_pair_with_impulse():
    proc = NonlinearRelativeProcess(CR0, CV0, frame="GCRS", j2=1.08263e-3)
    x1 = proc.f(X0, U0, 60.0)
    dr0, dv0 = lvlh_to_eci(CR0, CV0, X0[:3], X0[3:] + U0, "GCRS")
    tr = propagate_pair(CR0, CV0, CR0 + dr0, CV0 + dv0, [0.0, 60.0], frame="GCRS",
                        j2=1.08263e-3)
    ref = tr.rel_lvlh[0, -1, :]
    assert np.allclose(x1, ref, rtol=1e-6, atol=1e-4)
    # advance() moves the reference chief forward by the same step (the LVLH
    # frame itself rotates, so f is time-varying -- not a fixed map)
    proc.advance(U0, 60.0)
    tr2 = propagate_pair(CR0, CV0, CR0.copy(), CV0.copy(), [0.0, 60.0], frame="GCRS",
                         j2=1.08263e-3)
    assert np.allclose(proc.chief_r, tr2.chief_r[0, -1, :], atol=1e-3)
    assert np.allclose(proc.chief_v, tr2.chief_v[0, -1, :], atol=1e-6)


def test_nonlinear_jacobian_matches_fd_of_f():
    proc = NonlinearRelativeProcess(CR0, CV0, frame="GCRS")
    jac = proc.jacobian(X0, U0, 30.0)
    err = 0.0
    for i in range(6):
        h = 1e-5 * max(abs(X0[i]), 1.0)
        xp, xm = X0.copy(), X0.copy()
        xp[i] += h
        xm[i] -= h
        col = (proc.f(xp, U0, 30.0) - proc.f(xm, U0, 30.0)) / (2.0 * h)
        err = max(err, np.max(np.abs(col - jac[:, i]) / np.maximum(np.abs(col), 1e-2)))
    assert err < 1e-3   # FD-vs-FD agreement of the same scheme


def test_q_block_is_the_piecewise_constant_acceleration_integral():
    dt = 60.0
    q = white_noise_acceleration_q(2.0, dt)
    assert q.shape == (6, 6)
    assert np.allclose(q[:3, :3], 2.0 * dt**3 / 3.0 * np.eye(3))
    assert np.allclose(q[3:, 3:], 2.0 * dt * np.eye(3))
    assert np.allclose(q[:3, 3:], 2.0 * dt**2 / 2.0 * np.eye(3))


# ---------------------------------------------------------------------------
# measurement models
# ---------------------------------------------------------------------------

MODELS = [RelativePosition(), RelativeVelocity(), RangeRate(), RangeAngle(),
          Stacked([RelativePosition(), RangeAngle()])]


@pytest.mark.parametrize("model", MODELS, ids=lambda m: type(m).__name__)
def test_measurement_jacobians_match_finite_differences(model):
    jac = model.jacobian(X0)
    fd = np.empty((model.dof, 6))
    for i in range(6):
        h = 1e-6 * max(abs(X0[i]), 1.0)
        xp, xm = X0.copy(), X0.copy()
        xp[i] += h
        xm[i] -= h
        fd[:, i] = (model.h(xp) - model.h(xm)) / (2.0 * h)
    assert np.max(np.abs(jac - fd)) < 1e-7


def test_component_orders_and_stacked_dof():
    assert (RelativePosition().dof, RelativeVelocity().dof, RangeRate().dof,
            RangeAngle().dof) == (3, 3, 2, 3)
    assert RangeRate(("rate",)).dof == 1 and RangeAngle(("azimuth",)).dof == 1
    assert Stacked([RelativePosition(), RangeAngle()]).dof == 6
    with pytest.raises(ValueError):
        RangeRate(("range", "bogus"))


def test_range_model_geometry_honours_h():
    # along +x: range = x, rate = xdot, angles zero
    x = np.array([250.0, 0.0, 0.0, 0.0, -3.0, 0.0])
    rr = RangeRate().h(x)
    ra = RangeAngle().h(x)
    assert np.allclose(rr, [250.0, 0.0], atol=1e-12)
    assert np.allclose(ra[:1], [250.0], atol=1e-12) and abs(ra[1]) < 1e-12 and abs(ra[2]) < 1e-12


# ---------------------------------------------------------------------------
# the filter
# ---------------------------------------------------------------------------


def _run_filter(proc, q, x0, P0, meas, epochs, dt=60.0):
    filt = RelativeNavEKF(proc, x0, P0, q_accel_m2_s3=q)
    for k in range(epochs):
        if k:
            filt.predict(dt)
        filt.update(*meas(k, filt))
    return filt


def test_predict_update_keeps_p_a_valid_covariance():
    proc = CwProcess(N_REF)
    filt = _run_filter(proc, 1e-9, np.array([310.0, 5.0, 2.0, 0.0, 0.0, 0.0]),
                       np.diag([900.0, 900.0, 900.0, 0.01, 0.01, 0.01]),
                       lambda k, f: (np.array([300.0, 4.0, 1.0]),
                                     np.diag([1.0, 1.0, 1.0]), RelativePosition()),
                       8)
    sym = np.max(np.abs(filt.P - filt.P.T))
    assert sym < 1e-9
    assert np.min(np.linalg.eigvalsh(filt.P)) > 0.0
    # position knowledge improved over the run but never collapsed below R
    assert filt.P[0, 0] < 900.0
    assert filt.P[0, 0] > 0.0
    assert filt.n_predict == 7 and filt.n_update == 8


def test_zero_noise_tracking_converges_cw_and_nonlinear():
    for proc, tol in ((CwProcess(N_REF), 1e-6),
                      (NonlinearRelativeProcess(CR0, CV0, frame="GCRS"), 1e-3)):
        x = X0 + np.array([20.0, 10.0, 5.0, 0.05, 0.05, 0.01])
        filt = RelativeNavEKF(proc, x.copy(), np.diag([900.0, 900.0, 900.0, 0.01, 0.01,
                                                       0.01]), q_accel_m2_s3=0.0)
        for k in range(10):
            if k:
                filt.predict(30.0)
                if isinstance(proc, CwProcess):
                    x = np.concatenate(propagate_cw(x[:3], x[3:], N_REF, 30.0))
                else:
                    x = proc.f(x, np.zeros(3), 30.0)
            filt.update(x[:3], np.diag([1e-4, 1e-4, 1e-4]), RelativePosition())
        assert np.linalg.norm(filt.x[:3] - x[:3]) < tol


def test_update_report_carries_innovation_nis_and_sources():
    proc = CwProcess(N_REF)
    filt = RelativeNavEKF(proc, X0, np.diag([900.0] * 3 + [0.01] * 3), q_accel_m2_s3=1e-9)
    obs1 = _cartesian_obs(*lvlh_to_eci(CR0, CV0, X0[:3] + 1.0, X0[3:], "GCRS"),
                          np.diag([1.0, 1.0, 1.0]), np.diag([1e-4, 1e-4, 1e-4]),
                          rec="obs-a")
    rep = filt.update_from_observation(obs1, RelativePosition(), CR0, CV0)
    assert rep.dof == 3 and rep.observation_ids == ("obs-a",)
    assert rep.nis == pytest.approx(nis(rep.innovation,
                                        np.asarray(rep.innovation_cov).reshape(3, 3)))
    assert filt.source_observation_ids == ("obs-a",)
    obs2 = _native_obs(RangeAngle().h(X0), np.diag([0.25, 2.5e-9, 2.5e-9]), rec="obs-b")
    filt.update_from_observation(obs2, RangeAngle())
    assert filt.source_observation_ids == ("obs-a", "obs-b")


def test_native_extract_reads_metadata_not_the_cartesian_field():
    model = RangeAngle()
    z, _ = model.extract(_native_obs(model.h(X0), np.diag([0.25, 2.5e-9, 2.5e-9])))
    assert np.allclose(z, model.h(X0))
    with pytest.raises(GNCContractError, match="metadata"):
        model.extract(_cartesian_obs(np.array([100.0, 0, 0]), np.zeros(3),
                                     np.eye(3), np.eye(3)))
    with pytest.raises(GNCContractError, match="components"):
        model.extract(_native_obs([1.0, 2.0], np.eye(3)))


def test_cartesian_extract_rotates_through_the_chief():
    dr, dv = lvlh_to_eci(CR0, CV0, X0[:3], X0[3:], "GCRS")
    obs = _cartesian_obs(dr, dv, np.diag([1.0, 1.0, 1.0]), np.diag([1e-4, 1e-4, 1e-4]))
    z, r = RelativePosition().extract(obs, CR0, CV0)
    assert np.allclose(z, X0[:3], atol=1e-9)
    assert np.allclose(r, np.diag([1.0, 1.0, 1.0]))
    with pytest.raises(GNCContractError, match="chief"):
        RelativePosition().extract(obs)


def test_stacked_extract_each_blocks_covariances():
    stack = Stacked([RelativePosition(), RangeAngle()])
    dr, dv = lvlh_to_eci(CR0, CV0, X0[:3], X0[3:], "GCRS")
    obs_by_model = {
        "RelativePosition": _cartesian_obs(dr, dv, np.diag([1.0, 1.0, 1.0]),
                                           np.diag([1e-4, 1e-4, 1e-4])),
        "RangeAngle": _native_obs(RangeAngle().h(X0), np.diag([0.25, 2.5e-9, 2.5e-9])),
    }
    z, r = stack.extract_each(obs_by_model, CR0, CV0)
    assert z.shape == (6,) and r.shape == (6, 6)
    assert np.allclose(z[:3], X0[:3], atol=1e-9) and np.allclose(z[3:], RangeAngle().h(X0))
    assert np.allclose(r[:3, 3:], 0.0)


def test_make_estimate_publishes_valid_frame_mapped_estimate():
    proc = CwProcess(N_REF)
    filt = RelativeNavEKF(proc, X0, np.diag([900.0] * 3 + [0.01] * 3), q_accel_m2_s3=1e-9)
    dr, dv = lvlh_to_eci(CR0, CV0, X0[:3], X0[3:], "GCRS")
    filt.update_from_observation(_cartesian_obs(dr, dv, np.diag([1.0, 1.0, 1.0]),
                                                np.diag([1e-4, 1e-4, 1e-4])),
                                 RelativePosition(), CR0, CV0)
    est = filt.make_estimate(record_id="est-1", estimator_id="ekf", epoch=EPOCH,
                             target_id="tgt", chaser_id="chaser", inertial_frame="GCRS",
                             chief_r=CR0, chief_v=CV0)
    assert isinstance(est, Estimate)
    assert est.source_observation_ids == ("obs-1",)
    assert filt.source_observation_ids == ()
    doc = est.to_document()
    assert is_valid_record_document(doc)
    back = estimate_from_document(doc)
    assert np.allclose(np.asarray(back.relative_state.r_rel_m),
                       np.asarray(est.relative_state.r_rel_m))
    assert np.allclose(np.asarray(back.covariance_rel_m2),
                       np.asarray(est.covariance_rel_m2))
    p_i = np.asarray(est.covariance_rel_m2, dtype=np.float64).reshape(6, 6)
    assert np.allclose(p_i, p_i.T)
    assert np.linalg.eigvalsh(p_i).min() > 0.0


def test_frame_covariance_matches_finite_differences_of_the_mapping():
    p_lvlh = np.diag([900.0, 400.0, 100.0, 0.04, 0.09, 0.16])

    def map_state(delta6):
        dr, dv = lvlh_to_eci(CR0, CV0, X0[:3] + delta6[:3], X0[3:] + delta6[3:], "GCRS")
        return np.concatenate((dr - lvlh_to_eci(CR0, CV0, X0[:3], X0[3:], "GCRS")[0],
                               dv - lvlh_to_eci(CR0, CV0, X0[:3], X0[3:], "GCRS")[1]))

    j = np.empty((6, 6))
    for i in range(6):
        h = 1e-4 * max(abs(X0[i]), 1.0)
        dp, dm = np.zeros(6), np.zeros(6)
        dp[i] = h
        dm[i] = -h
        j[:, i] = (map_state(dp) - map_state(dm)) / (2.0 * h)
    manual = j @ p_lvlh @ j.T
    assert np.allclose(manual, frame_covariance(p_lvlh, CR0, CV0, "GCRS"), atol=1e-2)


def test_consistency_helpers_agree_with_scipy_bands():
    from scipy.stats import chi2
    for dof in (2, 3, 6):
        lo, hi = chi_square_interval(dof, 0.9)
        assert lo == pytest.approx(chi2.ppf(0.05, dof))
        assert hi == pytest.approx(chi2.ppf(0.95, dof))
        assert chi_square_cdf(hi, dof) == pytest.approx(0.95)
    z = np.array([3.0, -1.0, 0.5])
    assert nees(np.concatenate((z, np.zeros(3))), np.zeros(6),
                np.diag([16.0, 4.0, 1.0, 9.0, 9.0, 9.0])) == pytest.approx(9.0 / 16 + 1.0 / 4 + 1.0 / 4)
    assert nis(np.array([1.0, 2.0]), np.diag([2.0, 2.0])) == pytest.approx(2.5)
    assert rmse([[3.0], [-4.0]]) == pytest.approx(np.sqrt(12.5))
    assert sigma_coverage([[2.0], [4.0]], [[1.0], [1.0]], k=3.0) == 0.5
    assert sigma_coverage([[4.0]], [[1.0]], k=3.0) == 0.0


def test_observability_ranks_theory_cases():
    # instantaneous position: rank 3; range/rate on radial LOS: rank 2
    x = np.array([300.0, 1.0, 0.5, 0.0, 0.0, 0.0])
    rp = observability_rank(stacked_observability([RelativePosition().jacobian(x)],
                                                  [np.eye(6)]))
    assert rp["rank"] == 3
    rr = observability_rank(stacked_observability([RangeRate().jacobian(x)], [np.eye(6)]))
    assert rr["rank"] == 2 and rr["n_columns"] == 6
    # over a CW arc, position measurements gain velocity through curvature -> 6
    phi = np.eye(6)
    proc = CwProcess(N_REF)
    h_rows, phi_rows = [], []
    for _ in range(8):
        h_rows.append(RelativePosition().jacobian(x))
        phi_rows.append(phi)
        phi = proc.propagate_with_jacobian(x, np.zeros(3), 60.0)[1] @ phi
    arc = observability_rank(stacked_observability(h_rows, phi_rows))
    assert arc["rank"] == 6


# ---------------------------------------------------------------------------
# evidence gates
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def doc():
    assert DOC.exists(), f"run the benchmark first: {DOC}"
    return json.loads(DOC.read_text(encoding="utf-8"))


def cell(doc, process, truth, meas, geom):
    for r in doc["cells"]:
        if (r["process"], r["truth"], r["meas"], r["geom"]) == (process, truth, meas, geom):
            return r
    raise KeyError((process, truth, meas, geom))


def obsv(doc, meas, geom):
    for o in doc["observability"]:
        if (o["meas"], o["geom"]) == (meas, geom):
            return o
    raise KeyError((meas, geom))


def test_evidence_every_gate_passes(doc):
    assert doc["schema"] == "orbitopt.e13-02.relative-nav-ekf.v1"
    assert all(g["pass"] for g in doc["gates"]), \
        [g for g in doc["gates"] if not g["pass"]]


def test_evidence_consistency_on_cartesian_matched_cells(doc):
    lo, hi = doc["setup"]["declared_gates"]["nees"]["interval"]
    assert hi / lo > 4.0 and lo > 1.0
    for r in doc["cells"]:
        if r["matched"] and r["meas"] in ("pos", "pos_vel"):
            assert r["rmse_pos_m"] < 3.0, r
            assert r["rmse_vel_m_s"] < 0.05, r
            assert r["nees_pass_rate"] >= 0.70, r
            assert r["nis_pass_rate"] >= 0.70, r
            assert r["coverage_pos"] >= 0.98, r
            assert r["nees_interval_90"] == [pytest.approx(lo), pytest.approx(hi)]
            assert r["clusters"]["rmse_pos_m"]["count"] == doc["setup"]["seeds_per_cell"]
            assert len(r["per_seed_rmse_pos_m"]) == doc["setup"]["seeds_per_cell"]


def test_evidence_model_mismatch_is_detected_not_hidden(doc):
    # 5x drag, unmodelled: the filter MUST lose consistency (NEES band) while
    # staying bounded -- a mismatch-blind gate here would be worse than none
    for geom in ("radial", "along_track"):
        cw = cell(doc, "cw", "drag", "pos", geom)
        nl = cell(doc, "nonlinear", "drag", "pos", geom)
        for r in (cw, nl):
            assert 3.0 < r["rmse_pos_m"] <= 80.0, r
            assert r["nees_pass_rate"] <= 0.5, r
            assert r["nis_pass_rate"] <= 0.5, r
        # J2 across 300 m is nothing at this separation: stays consistent
        mm = cell(doc, "cw", "j2", "pos", geom)
        assert mm["rmse_pos_m"] < 3.0 and mm["nees_pass_rate"] >= 0.70


def test_evidence_weak_measurements_behave_as_observability_predicts(doc):
    # range/rate alone: null space, RMSE degrades orders of magnitude
    for geom in ("radial", "along_track"):
        rr = cell(doc, "cw", "cw", "range_rate", geom)
        assert rr["rmse_pos_m"] > 20.0, rr
        assert obsv(doc, "range_rate", geom)["rank"] < 6
        ra = obsv(doc, "range_angle", geom)
        assert ra["rank"] == 6 and ra["smallest_to_largest"] < 1e-3
        assert obsv(doc, "pos_vel", geom)["rank"] == 6
        pos = cell(doc, "cw", "cw", "pos", geom)
        ang = cell(doc, "cw", "cw", "range_angle", geom)
        assert ang["rmse_pos_m"] < 3.0          # position well pinned
        assert ang["nees_pass_rate"] < pos["nees_pass_rate"]   # but the linearisation bites
