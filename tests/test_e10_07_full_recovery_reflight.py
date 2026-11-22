"""E10-07 (task #304): full-recovery independent re-flight gates.

Fast gates on the committed benchmark JSON
(docs/benchmarks/e10_07_full_recovery_reflight.json): the handoff digest is
recomputed from the recorded separation state, every comparison band is
checked against the module's live band table (no silent widening), the main
model verdict matches the E10-01 record, and the landing-phase probe
verdict is the honest one. Regenerate the JSON with
``python benchmarks/bench_e10_07_full_recovery_reflight.py --out
docs/benchmarks/e10_07_full_recovery_reflight.json`` (~3 min).
"""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pytest

from orbitopt.flightlab.recovery.propagator import BoosterState
from orbitopt.flightlab.handoff import HandoffState, handoff_digest
from orbitopt.verify import full_recovery_reflight as fr
from orbitopt.verify.report import EVIDENCE_INDEPENDENT_MODEL, report_digest

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "docs/benchmarks/e10_07_full_recovery_reflight.json"
DC8 = REPO / "src/orbitopt/data/flightlab/vehicles/starship_class_rtls_dc8.yaml"
E1002 = REPO / "docs/benchmarks/e10_02_hot_staging.json"


@pytest.fixture(scope="module")
def rep() -> dict:
    return json.loads(BENCH.read_text(encoding="utf-8"))


class TestReportIntegrity:
    def test_protocol_and_digest(self, rep):
        assert rep["protocol"] == fr.PROTOCOL
        assert rep["digest"] == report_digest(rep)

    def test_handoff_digest_recomputed_from_recorded_state(self, rep):
        st = rep["handoff"]["state"]
        sep = BoosterState(
            t_s=float(st["t_s"]),
            epoch_utc=dt.datetime.fromisoformat(st["epoch_utc"]),
            frame=st["frame"],
            position_m=tuple(float(v) for v in st["position_m"]),
            velocity_m_s=tuple(float(v) for v in st["velocity_m_s"]),
            mass_kg=float(st["mass_kg"]))
        from orbitopt.flightlab.recovery.propagator import BoosterModel
        dry = BoosterModel.load(DC8).dry_mass_kg
        hs = HandoffState.of(sep, from_phase="separation",
                             to_phase="boostback", dry_mass_kg=dry)
        assert handoff_digest(hs) == rep["handoff"]["digest"]

    def test_handoff_is_the_e10_02_hot_staging_separation(self, rep):
        e1002 = json.loads(E1002.read_text(encoding="utf-8"))
        evs = {k: v for k, v in rep["inputs"]["ascent_run"]["events"]}
        assert float(evs["separation"]) == pytest.approx(125.0, abs=0.01)
        assert float(evs["impact"]) == pytest.approx(
            e1002["dc8"]["events_s"]["debris_impact"], abs=0.01)
        assert float(evs["seco"]) == pytest.approx(
            e1002["dc8"]["events_s"]["seco"], abs=0.01)

    def test_script_is_the_sealed_e10_01_keep_vector(self, rep):
        assert rep["inputs"]["script"] == [21.2, -2.0, 1.0, 0.0, 0.0]


class TestBands:
    def test_bands_match_the_module_table(self, rep):
        for cid, (unit, limit, _basis) in fr.BANDS.items():
            assert rep["comparison"]["bands"][cid]["unit"] == unit
            assert rep["comparison"]["bands"][cid]["limit"] == pytest.approx(limit)

    def test_every_check_passes_on_independent_model_evidence(self, rep):
        st = rep["status"]
        assert st["result"] == "passed" and st["passed"]
        assert set(st["checks"]) == set(fr.BANDS)
        assert all(ok for ok in st["checks"].values())
        assert all(ev == EVIDENCE_INDEPENDENT_MODEL
                   for ev in st["check_evidence"].values())

    def test_deltas_stay_inside_the_measured_bands(self, rep):
        for cid, (unit, limit, _b) in fr.BANDS.items():
            assert rep["comparison"]["deltas"][cid] <= limit

    def test_residual_after_the_budget_lines_stays_inside_band(self, rep):
        for cid in ("reflight.entry_peak_q", "reflight.impact_speed",
                    "reflight.impact_position"):
            assert rep["comparison"]["residual"][cid] <= fr.BANDS[cid][1]

    def test_density_form_is_the_dominant_peak_q_line(self, rep):
        shares = rep["comparison"]["budget_shares"]["reflight.entry_peak_q"]
        assert shares["atmosphere"] > max(shares["numerical"], shares["gravity"])


class TestMissionVerdictHonesty:
    def test_main_plan_infeasible_matching_e10_01_finding(self, rep):
        mm = rep["main_model"]
        assert mm["plan_feasible"] is False
        assert set(mm["reason_codes"]) == {"CORRIDOR_MISSED",
                                           "ENTRY_Q_LIMIT_EXCEEDED"}

    def test_landing_burn_unplannable_from_the_real_crossed_handoff(self, rep):
        lp = rep["landing_phase"]
        assert lp["crossed"] is True
        assert lp["feasible"] is False and lp["landed"] is False
        assert lp["reason_codes"] == ["LANDING_INFEASIBLE"]
        assert lp["crossed_state"]["speed_m_s"] > 1000.0

    def test_corridor_start_check_is_recorded_as_design_input(self, rep):
        joined = " ".join(rep["findings"]) + " " + rep["landing_phase"]["note"]
        assert "design_input" in joined
        assert "corridor" in joined.lower()

    def test_no_declared_corridor_state_used(self, rep):
        assert "actual" in rep["findings"][0]
        assert rep["handoff"]["state"]["t_s"] == 125.0


class TestTwinPhysicsUnits:
    def test_us76_layers_continuous_at_layer_bases(self):
        for zb in (11_000.0, 20_000.0, 32_000.0, 47_000.0, 51_000.0, 71_000.0):
            # +-1 m geodetic straddles the layer base (geopotential offset
            # ~2 m here -- far inside the 0.1% continuity tolerance)
            _p1, rho1, _a1 = fr._twin_us76(zb - 1.0)
            _p2, rho2, _a2 = fr._twin_us76(zb + 1.0)
            assert abs(rho1 - rho2) / rho2 < 1e-3

    def test_us76_monotone_below_stratopause(self):
        hs = np.linspace(0.0, 80_000.0, 161)
        rhos = [fr._twin_us76(float(h))[1] for h in hs]
        assert all(b <= a for a, b in zip(rhos, rhos[1:]))

    def test_twin_geodesy_agrees_with_the_propagator(self):
        # Bowring's closed form (the propagator's geodetic read-out); the
        # propagator's height_m is a declared first-order approximation and
        # is deliberately not the reference here
        from orbitopt.flightlab.recovery.boostback import _geodetic_fast
        rng = np.random.default_rng(304)
        for _ in range(25):
            r = rng.normal(0.0, 6.6e6, 3)
            r = r / np.linalg.norm(r) * (6371000.0 + 1e5 * rng.random())
            assert abs(fr._twin_geodetic(r)[2] - _geodetic_fast(r)[2]) < 1e-3

    def test_twin_constants_are_own_wgs84(self):
        # the twin must not import the propagator's gravity constants
        import inspect
        src = inspect.getsource(fr)
        assert "J2_EARTH" not in src and "_G0 = 9.80665" in src
