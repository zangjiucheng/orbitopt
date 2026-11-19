"""E6-05 (task #292): the matrix's upper stage on JSBSim, and the law's
generalisation is what the committed matrix says it is.

Fast half: the E6-05 spec extension is default-off (the packaged F-07
scenario is untouched), its fields validate, the generated JSBSim airframe
really carries a cell's dry mass / inertia / engine / servo parameters
(A0-no-tvc gets its clamps closed), and the E6-02 upper-powered attitude
band exists with its two rows.

Slow half (``@pytest.mark.slow``, skips without the ``jsbsim`` binding):
the committed comparison JSON is fully checked, and one cell is re-flown
in process -- handoff residuals inside the FL-402 staging tolerances,
propellant parity inside the engine-parity tolerance, and the insertion
verdict of the JSBSim leg matching the native record it is compared
against.
"""
from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from orbitopt.flightlab.guidance.upper_stage import (ParkingOrbitTarget,
                                                     UpperStageGuidanceSpec)
from orbitopt.flightlab.sixdof.guidance_matrix import (GUIDANCE_MATRIX,
                                                       matrix_scenario)
from orbitopt.flightlab.sixdof.guided_ascent import (E602_PHASE_CONSTRAINTS,
                                                     UPPER_POWERED)
from orbitopt.flightlab.sixdof.jsbsim_matrix import (ATT_ERR_MAX_LIMIT_DEG,
                                                    upper_airframe_xml)

REPO = Path(__file__).resolve().parents[1]
BENCH_JSON = REPO / "docs" / "benchmarks" / "e6_05_jsbsim_matrix.json"


def _cell(cid: str):
    return next(c for c in GUIDANCE_MATRIX if c.cell_id == cid)


class TestSpecExtension:
    def test_defaults_keep_the_packaged_scenario(self):
        spec = UpperStageGuidanceSpec.from_vehicle(
            _cell("a220-i28.5-V1-A1").vehicle_pair_upper
            if hasattr(_cell("a220-i28.5-V1-A1"), "vehicle_pair_upper")
            else matrix_scenario(_cell("a220-i28.5-V1-A1")).pair.upper,
            target=ParkingOrbitTarget(altitude_m=220e3), ignition_t_s=180.0)
        assert spec.transfer_target is False
        assert spec.climb_time_to_go_gain == 0.0
        assert spec.cutoff_insertion_box is False
        assert spec.apogee_ceiling_gain == 0.0

    @pytest.mark.parametrize("kwargs", [
        {"climb_time_to_go_gain": -0.5},
        {"cutoff_periapsis_margin_m": -1.0},
        {"apogee_ceiling_gain": -1.0e-4},
    ])
    def test_new_fields_validate(self, kwargs):
        sc = matrix_scenario(_cell("a220-i28.5-V1-A1"))
        with pytest.raises(ValueError):
            replace(sc.upper_guidance, **kwargs)

    def test_matrix_flies_the_generalised_law(self):
        ug = matrix_scenario(_cell("a220-i28.5-V1-A1")).upper_guidance
        assert ug.transfer_target is True
        assert ug.climb_time_to_go_gain == 0.0
        assert ug.cutoff_insertion_box is True
        assert ug.apogee_ceiling_gain > 0.0


class TestGeneratedAirframe:
    def test_cell_numbers_land_in_the_xml(self):
        cell = _cell("a300-i28.5-V2-A1")
        xml = upper_airframe_xml(cell, matrix_scenario(cell))
        assert '<aircraft name="spike_upper_f07"' in xml
        assert f'<emptywt unit="KG"> {cell.vehicle.dry_mass_kg:.6f} </emptywt>' in xml
        assert '<engine file="spike_upper_engine">' in xml
        assert '<thruster file="spike_upper_nozzle">' in xml
        assert f'<lag> {1.0 / cell.actuator.lag_tau_s:.6f} </lag>' in xml
        assert f'{cell.actuator.slew_rate_deg_s * math.pi / 180.0:.9f} </rate_limit>' in xml
        travel = math.radians(cell.actuator.travel_deg)
        assert f'<min> {-travel:.9f} </min>' in xml

    def test_no_tvc_cell_gets_closed_clamps(self):
        cell = next(c for c in GUIDANCE_MATRIX if c.actuator_id == "A0-no-tvc")
        xml = upper_airframe_xml(cell, matrix_scenario(cell))
        assert "<min> -0.000000000 </min><max> 0.000000000 </max>" in xml

    def test_jsbsim_loads_the_generated_model(self):
        pytest.importorskip("jsbsim")
        from orbitopt.flightlab.sixdof import jsbsim_matrix as jm
        from orbitopt.flightlab.sixdof.jsbsim_driver import SixDofSim
        cell = _cell("a220-i28.5-V1-A1")
        root = jm._airframe_root(upper_airframe_xml(cell, matrix_scenario(cell)))
        try:
            SixDofSim(0.01, model="spike_upper_f07", root=root)
        finally:
            import shutil
            shutil.rmtree(root, ignore_errors=True)


class TestE602UpperPoweredBand:
    def test_both_rows_declared(self):
        assert E602_PHASE_CONSTRAINTS[(UPPER_POWERED, "att_err_max_deg")][0] \
            == ATT_ERR_MAX_LIMIT_DEG
        assert E602_PHASE_CONSTRAINTS[(UPPER_POWERED, "att_err_rms_deg")][0] == 7.0
        for key in ((UPPER_POWERED, "att_err_max_deg"),
                    (UPPER_POWERED, "att_err_rms_deg")):
            assert "E6-05" in E602_PHASE_CONSTRAINTS[key][3]


@pytest.mark.slow
class TestCommittedComparison:
    def test_json_is_checked_and_complete(self):
        pytest.importorskip("jsbsim")
        if not BENCH_JSON.exists():
            pytest.skip("run benchmarks/bench_e6_05_jsbsim_matrix.py first")
        bench = json.loads(BENCH_JSON.read_text())
        assert bench["meta"]["all_checks_passed"] is True
        assert len(bench["checks"]) >= 7 * len(bench["cells"])
        for rec in bench["cells"]:
            assert rec["all_checks_passed"] is True
            for key in ("d_periapsis_jsbsim_minus_native_m",
                        "d_apoapsis_jsbsim_minus_native_m",
                        "d_inclination_jsbsim_minus_native_deg",
                        "verdict_agrees"):
                assert key in rec["comparison"]
            assert rec["jsbsim"]["handoff_residuals"]["position_err_m"] < 1e-3
            assert rec["jsbsim"]["n_samples"] > 10

    def test_a_cell_reflies(self):
        pytest.importorskip("jsbsim")
        from orbitopt.flightlab.sixdof.jsbsim_matrix import (
            jsbsim_matrix_record, matrix_handoff)
        cell = _cell("a220-i28.5-V1-A1")
        rec = jsbsim_matrix_record(cell, matrix_handoff(cell), dt_s=0.01)
        h = rec["handoff_residuals"]
        assert h["position_err_m"] < 1e-3
        assert h["velocity_err_m_s"] < 1e-6
        assert h["attitude_err_rad"] < 1e-6
        assert abs(h["mass_err_kg"]) < 1e-3
        prop = rec["propellant_at_ignition_kg"]
        assert 0.0 < rec["propellant_burned_kg"] < prop + 1e-6
        assert rec["n_samples"] > 100
        assert rec["attitude_err_max_deg"] < ATT_ERR_MAX_LIMIT_DEG
