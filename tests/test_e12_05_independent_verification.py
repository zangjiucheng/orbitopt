"""E12-05 (task #395): independent verification of LEO-RDV-001 by a TudatPy
two-spacecraft re-flight.

Two graded things. First the *honest degradation* contract (task #261 rule):
with no tudatpy backend reachable, the report still assembles and validates,
the CW and optimizer lines are graded from their environment-free
measurements, and the independent-model / Kepler / convergence lines are
``not_checked`` with reason ``REFLY_BACKEND_MISSING`` -- absent evidence is
recorded, never passed. Second the *separation* contract (the task's own
acceptance): the error sources appear as separate residual groups and as
separate entries of ``inputs["error_sources"]``, no single blended ``passed``
stands in for them, and no check rides on ``historical_telemetry``.

The cross-env marker test is the real thing: a TudatPy subprocess re-flies
the pair, seals the evidence, and the headline comparison must pass on
evidence ``independent_model``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from orbitopt.verify import rendezvous_refly as rr


def _backend_free() -> bool:
    backend, _ = rr.resolve_refly_backend()
    return backend is None


@pytest.fixture(scope="module")
def degraded(tmp_path_factory):
    """One full degraded run of the whole pipeline (no backend), shared."""
    if not _backend_free():
        pytest.skip("a tudatpy backend is reachable here; degraded mode is not the subject")
    out = tmp_path_factory.mktemp("e12-05-degraded")
    return rr.verify_rendezvous_refly(out_dir=out, stationarity=False)


def _load_written(report_path) -> dict:
    from orbitopt.verify.report import read_report, validate_report

    doc = read_report(report_path)
    validate_report(doc)  # schema + digest round trip, fail-closed
    return doc


class TestDegradedReport:
    def test_written_report_validates(self, degraded):
        assert _load_written(degraded["report_path"]) == degraded["report"]

    def test_backend_absence_recorded_not_passed(self, degraded):
        rep = degraded["report"]
        assert rep["inputs"]["headline_acceptance"] == "not_checked"
        assert any(rr.REFLY_BACKEND_MISSING in r for r in rep["inputs"]["not_checked_reasons"])
        src = rep["inputs"]["error_sources"]
        assert src["independent_model_agreement"]["result"] == "not_checked"
        assert src["numerical_propagation"]["result"] == "not_checked"

    def test_environment_free_lines_still_graded(self, degraded):
        src = degraded["report"]["inputs"]["error_sources"]
        assert src["approximate_model_cw"]["result"] in ("passed", "failed")
        assert src["optimizer_nonconvergence"]["result"] == "passed"
        assert src["approximate_model_cw"]["measured"]
        assert src["optimizer_nonconvergence"]["measured"]

    def test_error_sources_are_separate_lines(self, degraded):
        rep = degraded["report"]
        src = rep["inputs"]["error_sources"]
        assert set(src) == {"numerical_propagation", "approximate_model_cw",
                            "optimizer_nonconvergence", "independent_model_agreement"}
        results = {k: v["result"] for k, v in src.items()}
        assert len(set(results.values())) > 1, \
            f"error sources collapsed onto one verdict: {results}"
        # and the per-category / per-evidence rollups keep them apart too
        cats = rep["status"]["categories"]
        assert cats["model_agreement"]["result"] == "passed"          # the CW line
        assert cats["numerical_consistency"]["result"] == "passed"    # the optimizer line
        assert rep["status"]["independent_validation"]["result"] == "not_checked"

    def test_no_historical_telemetry_evidence(self, degraded):
        rep = degraded["report"]
        for r in rep["residuals"]:
            if "within_tolerance" in r:
                assert r["evidence"] in ("independent_model", "analytical",
                                         "numerical_consistency", "design_input")
        assert rep["status"]["evidence"]["historical_telemetry"]["checks_total"] == 0

    def test_cli_reports_degraded_with_distinct_exit_code(self, tmp_path):
        if not _backend_free():
            pytest.skip("backend reachable; degraded exit code is not the subject")
        rc = rr.main(["--no-stationarity", "-o", str(tmp_path)])
        assert rc == 2  # 0 passed / 1 failed / 2 headline not_checked


class TestStageWiring:
    def test_runner_registered_everywhere(self):
        from orbitopt.crossenv.plan import KNOWN_RUNNERS
        from orbitopt.crossenv.stages import STAGE_RUNNERS

        assert rr.REFLY_STAGE_RUNNER in KNOWN_RUNNERS
        assert rr.REFLY_STAGE_RUNNER in STAGE_RUNNERS
        schema = json.loads((Path(rr.__file__).parents[1] / "schemas"
                             / "crossenv-plan-1.0.json").read_text(encoding="utf-8"))
        assert rr.REFLY_STAGE_RUNNER in schema["properties"]["stages"]["items"]["properties"][
            "runner"]["enum"]

    def test_packaged_refly_plan_loads(self):
        from orbitopt.crossenv.plan import load_plan

        plan_path = Path(rr.__file__).parents[1] / "data" / "crossenv_e12_05_plan.yaml"
        plan = load_plan(plan_path)
        assert [(st.id, st.runner) for st in plan.stages] == [("refly", rr.REFLY_STAGE_RUNNER)]
        assert plan.scenario.name == "leo_rdv_001.yaml"

    @pytest.mark.skipif(not _backend_free(), reason="worker raises only without tudatpy here")
    def test_worker_refuses_without_backend(self):
        from orbitopt.rendezvous.config import load_rendezvous_config

        cfg_path = Path(rr.__file__).parents[1] / "data" / "leo_rdv_001.yaml"
        with pytest.raises(rr.ReflyBackendError, match=rr.REFLY_BACKEND_MISSING):
            rr.run_refly_in_current_environment(load_rendezvous_config(cfg_path))


class TestSweepSlicing:
    """E12-09 (task #411): a sweep history row is the concatenated pair state
    ``[r_c, v_c, r_d, v_d]``, so the graded position error is the norm over
    BOTH crafts' position columns and the velocity error over both crafts'
    velocity columns. The original slicing took ``d[:6]`` / ``d[6:]``, mixing
    position with velocity in one norm and reporting the deputy position
    difference as a velocity in m/s."""

    EPOCHS = (0.0, 100.0)
    REF = (7.0e6, 0.0, 0.0, 0.0, 7546.0, 0.0, 7.0e6 + 100.0, 0.0, 0.0, 0.0, 7556.0, 0.0)

    @staticmethod
    def _run(offset):
        import numpy as np

        hist = np.tile(np.array(TestSweepSlicing.REF) + np.array(offset), (2, 1))
        return {"tag": "coarse"}, np.array(TestSweepSlicing.EPOCHS), hist

    @staticmethod
    def _ref_run():
        import numpy as np

        return {"tag": "ref"}, np.array(TestSweepSlicing.EPOCHS), np.tile(
            np.array(TestSweepSlicing.REF), (2, 1))

    def _table(self, offset):
        table, epoch = rr.sweep_table([self._run(offset), self._ref_run()],
                                      leg_end_epoch=100.0, epoch_anchor=0.0)
        assert epoch == 100.0
        return table[0]

    def test_known_position_and_velocity_offsets(self):
        # chief position (3, 4, 0) and deputy position (12, 0, 0) m: the one
        # position metric is the joint norm, sqrt(9 + 16 + 144) = 13 m; chief
        # velocity (0, 0.03, 0) and deputy velocity (0, 0, -0.09) m/s: joint
        # norm sqrt(0.009) = 0.09487 m/s. The old slicing reported
        # sqrt(9+16+0.09+0.0009) and sqrt(144+0.0081) -- position mixed with
        # velocity, deputy position dressed as velocity.
        row = self._table((3.0, 4.0, 0.0, 0.0, 0.03, 0.0, 12.0, 0.0, 0.0, 0.0, 0.0, -0.09))
        assert row["position_error_m"] == pytest.approx(13.0)
        assert row["velocity_error_m_s"] == pytest.approx(0.09486832980505138)

    def test_velocity_only_offset_is_not_a_position_error(self):
        row = self._table((0.0, 0.0, 0.0, 0.03, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        assert row["position_error_m"] == pytest.approx(0.0)
        assert row["velocity_error_m_s"] == pytest.approx(0.05)

    def test_position_only_offset_is_not_a_velocity_error(self):
        row = self._table((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 30.0, 40.0, 0.0, 0.0, 0.0, 0.0))
        assert row["position_error_m"] == pytest.approx(50.0)
        assert row["velocity_error_m_s"] == pytest.approx(0.0)


class TestCwReplay:
    """The CW replay applies the plan's own pulses: its delta-v total is the
    plan's, and the residual that remains is model error, not plan error."""

    def test_cw_total_delta_v_equals_plan(self, degraded):
        plan = degraded["plan"]
        assert abs(degraded["cw"]["total_delta_v_m_s"]
                   - plan.total_delta_v_m_s) < 1e-12

    def test_optimizer_landed_well_inside_tolerance(self, degraded):
        measured = degraded["report"]["inputs"]["error_sources"][
            "optimizer_nonconvergence"]["measured"]["optimizer.landing_residual"]
        assert measured <= 1.0e-3  # the config's newton_tol_m


@pytest.mark.cross_env
@pytest.mark.skipif(not (os.environ.get("ORBITOPT_TUDAT_PYTHON")
                         and Path(os.environ.get("ORBITOPT_TUDAT_PYTHON", "")).exists()),
                    reason="needs ORBITOPT_TUDAT_PYTHON pointing at a tudatpy interpreter")
def test_independent_refly_headline_passes(tmp_path):
    """The acceptance test, run in an environment that has the backend: the
    TudatPy re-flight (crossenv stage, sealed + revalidated exchange) must
    agree with the main model on all four acceptance quantities, the sweep
    bounds must hold, and the report's independent-validation rollup must be
    the pass a reader of a pass checks (task #263)."""
    res = rr.verify_rendezvous_refly(out_dir=tmp_path, stationarity=True)
    rep = res["report"]
    src = rep["inputs"]["error_sources"]
    assert src["independent_model_agreement"]["result"] == "passed"
    assert src["numerical_propagation"]["result"] == "passed"
    assert src["approximate_model_cw"]["result"] == "passed"
    assert src["optimizer_nonconvergence"]["result"] == "passed"
    assert rep["status"]["independent_validation"]["result"] == "passed"
    assert rep["status"]["evidence"]["independent_model"]["checks_passed"] >= 3
    # sealed-exchange provenance rode into the report
    cx = rep["inputs"]["crossenv"]
    assert cx["envelope_digest"].startswith("sha256:")
    assert len(cx["envelope_digest"].split(":", 1)[1]) == 64
    assert cx["stage_report"]["status"] == "completed"
    # the worker re-verified the plan in its own environment and it passed
    assert res["refly"]["mission_in_refly_env"]["passed"] is True
    # both models flew the same plan (float reproducibility across envs)
    assert res["main_quantities"]["maneuver_identity_m_s"] <= rr.REFLY_VS_MAIN_TOLERANCES[
        "maneuver_sequence_m_s"]
