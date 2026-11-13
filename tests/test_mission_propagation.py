"""Tests for M-05 (task #223): TudatPy propagation of the injected orbit.

The pure tests run against a fake tudatpy (identity SPICE rotation, Keplerian
history generator) installed only into ``sys.modules`` -- no compute stack
needed. The single real-stack integration test is ``slow`` + ``importorskip``.
"""
from __future__ import annotations

import json
import math
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from orbitopt.flightlab.checks import orbital_elements
from orbitopt.flightlab.coast import coast_arc
from orbitopt.mission.contract import Epoch, HandoffState
from orbitopt.mission.propagation import (
    DragConfig, InjectedOrbitPropagation, MU_EARTH, OrbitPropagationSegment,
    PropagationForceModel, analytic_two_body_comparison, element_history,
    handoff_to_tudat_initial_state, orbit_propagation_report,
    propagate_injected_orbit, propagation_from_arrays)

ET0 = 7.0e8  # arbitrary ephemeris second (a 2022 date)


def leo_state(frame="GCRS", epoch=None):
    r = np.array([7_000_000.0, 0.0, 0.0])
    v = np.array([0.0, math.sqrt(MU_EARTH / 7_000_000.0), 0.0])
    return HandoffState(
        epoch=epoch or Epoch(mjd2000=9400.0, timescale="UTC"), frame=frame,
        r_m=tuple(r), v_m_s=tuple(v), mass_kg=12_000.0,
        stage_config={"upper_stage": "FV-1"}, metadata={"task": "223"})


def kepler_history(state6, et0, duration, n=1201):
    times, positions, velocities = coast_arc(state6[:3], state6[3:], duration,
                                             samples=n, mu=MU_EARTH)
    epochs = np.asarray(times) + et0
    states = np.column_stack([np.asarray(positions), np.asarray(velocities)])
    return epochs, states


@pytest.fixture
def fake_tudat(monkeypatch):
    """Install fake ``tudatpy`` modules (identity SPICE) and a Keplerian
    ``_propagate_single_coast``, so the whole propagation pipeline runs
    against a two-body truth without the compute stack."""
    modules = {}

    def mod(name, **attrs):
        m = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(m, key, value)
        modules[name] = m
        return m

    identity = np.eye(3)
    spice = mod("tudatpy.interface.spice",
                load_standard_kernels=lambda: None,
                compute_rotation_matrix_between_frames=MagicMock(return_value=identity))
    mod("tudatpy.interface", spice=spice)
    env = MagicMock(name="environment_setup")
    prop = MagicMock(name="propagation_setup")
    mod("tudatpy.numerical_simulation", environment_setup=env, propagation_setup=prop)
    mod("tudatpy", constants=MagicMock(), numerical_simulation=modules["tudatpy.numerical_simulation"],
        interface=modules["tudatpy.interface"])
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    import orbitopt.verify.tudat_propagate as tp

    calls = {"propagate": [], "overrides": []}

    def fake_propagate(bodies, accel, central, state6, et0, duration, step,
                       integrator="dop853", rtol=1e-11, atol=1e-6):
        calls["propagate"].append({"central": central, "state": np.asarray(state6),
                                  "et0": et0, "duration": duration, "step": step,
                                  "rtol": rtol, "atol": atol})
        return kepler_history(np.asarray(state6, dtype=float), float(et0), float(duration))

    def fake_override(body_settings, central_body, gravity):
        calls["overrides"].append({"central": central_body, "gravity": dict(gravity)})
        return {"gravitational_parameter": gravity["gravitational_parameter"],
                "reference_radius": gravity["reference_radius"], "j2": gravity.get("j2", 0.0),
                "normalized_c20": -gravity.get("j2", 0.0) / math.sqrt(5.0),
                "associated_reference_frame": f"IAU_{central_body}"}

    monkeypatch.setattr(tp, "_SPICE_LOADED", True)
    monkeypatch.setattr(tp, "_propagate_single_coast", fake_propagate)
    monkeypatch.setattr(tp, "_override_central_gravity", fake_override)
    yield {"env": env, "prop": prop, "spice": spice, "calls": calls}




# ---------------------------------------------------------------- force model


class TestForceModel:
    def test_drag_defaults_and_validation(self):
        d = DragConfig()
        assert d.drag_coefficient == 2.2 and d.scale_height_m > 0
        with pytest.raises(ValueError, match="reference_area_m2"):
            DragConfig(reference_area_m2=0.0)
        with pytest.raises(ValueError, match="finite"):
            DragConfig(scale_height_m=float("nan"))

    def test_force_model_bodies_and_validation(self):
        fm = PropagationForceModel()
        assert fm.j2 and fm.drag is None and fm.third_bodies == ()
        assert fm.bodies == ("Earth",)
        fm = PropagationForceModel(third_bodies=("Sun", "Moon", "Sun"))
        assert fm.third_bodies == ("Sun", "Moon") and fm.bodies == ("Earth", "Sun", "Moon")
        with pytest.raises(ValueError, match="central body"):
            PropagationForceModel(third_bodies=("Earth",))
        with pytest.raises(ValueError, match="non-empty"):
            PropagationForceModel(third_bodies=("",))

    def test_force_model_to_dict_rounds_trip(self):
        fm = PropagationForceModel(drag=DragConfig())
        doc = fm.to_dict()
        assert doc["central_gravity"]["gravitational_parameter"] == MU_EARTH
        assert doc["drag"]["model"] == "exponential_atmosphere_constant_cd"
        assert doc["third_bodies"] == []


# ---------------------------------------------------------------- handoff -> tudat


class TestHandoffConversion:
    def test_gcrs_state_rotated_at_handoff_epoch(self, fake_tudat):
        state = leo_state()
        state6_ecl, et = handoff_to_tudat_initial_state(state)
        assert et == pytest.approx(state.epoch.to_tdb_seconds())
        assert np.allclose(state6_ecl, np.concatenate([state.r_m, state.v_m_s]))
        fake_tudat["spice"].compute_rotation_matrix_between_frames.assert_called_once()
        src, dst, called_et = (
            fake_tudat["spice"].compute_rotation_matrix_between_frames.call_args.args)
        assert src == "GCRS" and dst == "ECLIPJ2000"
        assert called_et == pytest.approx(et)

    def test_ecef_state_is_bridged_inside_the_segment(self, fake_tudat, monkeypatch):
        rotated = np.array([[0.0, 7_000_000.0, 0.0]])
        vrot = np.array([[-math.sqrt(MU_EARTH / 7_000_000.0), 0.0, 0.0]])
        monkeypatch.setattr("orbitopt.flightlab.frames.to_gcrs",
                            lambda r, v, frame, epoch, t: (rotated, vrot))
        state = leo_state(frame="ECEF")
        state6_ecl, et = handoff_to_tudat_initial_state(state)
        assert np.allclose(state6_ecl[:3], rotated[0])
        assert np.allclose(state6_ecl[3:], vrot[0])
        assert et == pytest.approx(state.epoch.to_tdb_seconds())

    def test_rejects_non_handoff_state(self, fake_tudat):
        with pytest.raises(ValueError, match="HandoffState"):
            handoff_to_tudat_initial_state({"r_m": [7e6, 0, 0]})


# -------------------------------------------------- propagation (fake backend)


class TestPropagationFake:
    def test_default_run_is_two_j2_revolutions(self, fake_tudat):
        state = leo_state()
        prop = propagate_injected_orbit(state, revolutions=2.0)
        period = orbital_elements(state.r_m, state.v_m_s)["period_s"]
        call = fake_tudat["calls"]["propagate"][0]
        assert call["duration"] == pytest.approx(2.0 * period)
        assert call["central"] == "Earth"
        assert prop.duration_s == pytest.approx(2.0 * period)
        assert prop.revolutions == 2.0
        assert prop.states_gcrs.shape == prop.states_ecliptic.shape
        assert prop.states_gcrs[-1] == pytest.approx(prop.final_state_gcrs)

    def test_environment_is_built_from_the_configured_model(self, fake_tudat):
        from orbitopt.mission.propagation import EARTH_GRAVITY

        fm = PropagationForceModel(third_bodies=("Sun", "Moon"), drag=DragConfig())
        propagate_injected_orbit(leo_state(), fm)
        env = fake_tudat["env"]
        bodies, central, frame = env.get_default_body_settings.call_args.args
        assert bodies == ["Earth", "Sun", "Moon"] and central == "Earth"
        assert frame == "ECLIPJ2000"
        assert fake_tudat["calls"]["overrides"][-1]["gravity"] == EARTH_GRAVITY
        # drag demands a mass: the spacecraft body is created empty and given
        # the handoff mass.
        env.create_system_of_bodies.return_value.create_empty_body.assert_called_with("Spacecraft")
        assert env.create_system_of_bodies.return_value.get.return_value.mass == 12_000.0

    def test_point_mass_run_matches_analytic_within_gate(self, fake_tudat):
        prop = propagate_injected_orbit(leo_state(), PropagationForceModel(j2=False))
        assert prop.point_mass_only
        assert prop.analytic["max_position_m"] < 10.0
        assert prop.analytic["max_velocity_m_s"] < 0.001

    def test_rejects_bad_inputs(self, fake_tudat):
        with pytest.raises(ValueError, match="revolutions"):
            propagate_injected_orbit(leo_state(), revolutions=0.0)
        with pytest.raises(ValueError, match="revolutions"):
            propagate_injected_orbit(leo_state(), revolutions=float("nan"))
        flat = HandoffState(epoch=Epoch(mjd2000=9400.0), frame="GCRS",
                            r_m=(8_000_000.0, 0.0, 0.0), v_m_s=(0.0, 12_000.0, 0.0),
                            mass_kg=1000.0)
        with pytest.raises(ValueError, match="closed"):
            propagate_injected_orbit(flat)

    def test_element_history_brackets_the_arc(self, fake_tudat):
        prop = propagate_injected_orbit(leo_state())
        rows = prop.elements_history
        assert rows[0]["t_s"] == pytest.approx(0.0)
        assert rows[-1]["t_s"] == pytest.approx(prop.duration_s)
        for row in rows:
            assert row["period_s"] == pytest.approx(prop.period_s, rel=1e-9)
            assert row["semi_major_axis_m"] == pytest.approx(7_000_000.0, rel=1e-9)

    def test_rerun_hook_honours_verify_overrides(self, fake_tudat):
        fm = PropagationForceModel(third_bodies=("Sun",))
        prop = propagate_injected_orbit(leo_state(), fm)
        result = prop.rerun(perturbing_bodies=("Earth",), earth_spherical_harmonic=None)
        assert hasattr(result, "final_state") and hasattr(result, "segments")
        call = fake_tudat["calls"]["propagate"][-1]
        assert call["rtol"] == pytest.approx(1e-11)  # unchanged by this override
        result = prop.rerun(rtol=1e-13, atol=1e-8)
        call = fake_tudat["calls"]["propagate"][-1]
        assert call["rtol"] == pytest.approx(1e-13) and call["atol"] == pytest.approx(1e-8)


# ----------------------------------------------- recorded-history assembly


class TestRecordedHistory:
    def _recorded(self, n=601, duration=5828.5):
        state = leo_state()
        state6 = np.concatenate([state.r_m, state.v_m_s])
        epochs, states = kepler_history(state6, ET0, duration, n=n)
        return propagation_from_arrays(state, PropagationForceModel(j2=False), 1.0,
                                       epochs, states, ET0,
                                       orbital_elements(state.r_m, state.v_m_s)["period_s"],
                                       rotate_to_gcrs=False)

    def test_propagation_from_arrays_relative_epochs(self):
        prop = self._recorded()
        assert prop.epochs_s[0] == pytest.approx(0.0)
        assert prop.epochs_s[-1] == pytest.approx(5828.5, abs=1.0)
        assert prop.et0 == ET0
        assert np.allclose(prop.final_state_gcrs, prop.states_ecliptic[-1])

    def test_analytic_comparison_on_recorded_two_body_history(self):
        prop = self._recorded(n=1201)
        cmp = analytic_two_body_comparison(prop, samples=64)
        assert cmp["frame"] == "ECLIPJ2000"
        assert cmp["max_position_m"] < 10.0  # Hermite interpolation error only
        assert cmp["max_velocity_m_s"] < 1e-3

    def test_hermite_error_bounded_for_coarse_history(self):
        prop = self._recorded(n=25)  # ~243 s steps: sagitta would wreck a linear interp
        cmp = analytic_two_body_comparison(prop, samples=16)
        assert cmp["max_position_m"] < 200.0  # cubic Hermite keeps it metres-to-100m

    def test_element_history_sampling_includes_ends(self):
        prop = self._recorded()
        rows = element_history(prop, samples=13)
        assert len(rows) >= 10
        assert rows[0]["t_s"] == pytest.approx(0.0)
        assert rows[-1]["t_s"] == pytest.approx(prop.epochs_s[-1])

    def test_rejects_misshapen_history(self):
        state = leo_state()
        with pytest.raises(ValueError, match="at least two"):
            propagation_from_arrays(state, PropagationForceModel(), 1.0,
                                   np.array([ET0]), np.zeros((1, 6)), ET0, 100.0,
                                   rotate_to_gcrs=False)
        with pytest.raises(ValueError, match="line up"):
            propagation_from_arrays(state, PropagationForceModel(), 1.0,
                                    np.array([ET0, ET0 + 10.0]), np.zeros((3, 6)),
                                    ET0, 100.0, rotate_to_gcrs=False)


# ---------------------------------------------------------------- report


class TestReport:
    def _prop(self, fake_tudat, fm=None, revolutions=1.0):
        return propagate_injected_orbit(
            leo_state(), fm or PropagationForceModel(j2=False), revolutions,
            max_step_s=120.0)

    def test_point_mass_report_passes_gated(self, fake_tudat, tmp_path):
        prop = self._prop(fake_tudat)
        path = tmp_path / "m05.json"
        report = orbit_propagation_report(prop, label="task-223", write_path=path)
        assert report["kind"] == "orbit-propagation"
        assert report["status"]["result"] == "passed"
        checks = report["status"]["checks"]
        assert checks["residual.analytic_two_body_coast"]
        assert checks["task-223.step_convergence"]
        assert checks["task-223.two_body_conservation"] or checks["task-223.model_conservation"]
        assert json.loads(path.read_text())["digest"] == report["digest"]
        doc = json.loads(path.read_text())
        history = doc["inputs"]["orbit_propagation"]["elements_history"]
        assert len(history) >= 10
        assert history[-1]["t_s"] == pytest.approx(doc["inputs"]["duration_s"], rel=1e-3)
        assert doc["inputs"]["analytic_comparison"]["gated"] is True
        assert doc["inputs"]["force_model"]["j2"] is False

    def test_perturbed_report_records_without_gating(self, fake_tudat):
        prop = self._prop(fake_tudat, fm=PropagationForceModel(j2=True,
                                                              third_bodies=("Sun", "Moon")))
        report = orbit_propagation_report(prop)
        assert "residual.analytic_two_body_coast" not in report["status"]["checks"]
        assert report["inputs"]["analytic_comparison"]["gated"] is False
        assert report["inputs"]["analytic_comparison"]["max_position_m"] >= 0.0
        forces = report["propagations"][0]["dynamics"]["forces"]
        assert sum(f["type"] == "gravity" for f in forces) == 3
        assert report["propagations"][0]["dynamics"]["summary"]["earth_J2"]

    def test_drag_shows_up_in_the_inventory(self, fake_tudat):
        prop = self._prop(fake_tudat, fm=PropagationForceModel(j2=False, drag=DragConfig()))
        report = orbit_propagation_report(prop)
        dyn = report["propagations"][0]["dynamics"]
        assert any(f["type"] == "aerodynamic_drag" for f in dyn["forces"])
        assert dyn["summary"]["atmospheric_drag"] is True
        assert dyn["summary"]["central_point_mass_only"] is False
        assert not any("atmospheric_drag" in line for line in dyn["not_modelled"])
        propagated = report["propagations"][0]["conservation"]["propagated_model"]
        assert propagated["applicable"] is False
        # the residual stays ungated even though the gravity field is a point mass
        assert "residual.analytic_two_body_coast" not in report["status"]["checks"]

    def test_handoff_elements_and_history_in_inputs(self, fake_tudat):
        prop = self._prop(fake_tudat)
        report = orbit_propagation_report(prop)
        handoff = report["inputs"]["handoff"]
        assert handoff["frame"] == "GCRS" and handoff["mass_kg"] == 12_000.0
        assert handoff["initial_elements"]["semi_major_axis_m"] == pytest.approx(7e6, rel=1e-9)
        states = report["inputs"]["orbit_propagation"]["state_history_gcrs"]
        assert {"t_s", "position_m", "velocity_m_s"} <= set(states[0])


# ---------------------------------------------------------------- segment


class TestSegment:
    def test_run_hands_off_propagated_state_and_writes_report(self, fake_tudat, tmp_path):
        path = tmp_path / "reports" / "m05.json"
        segment = OrbitPropagationSegment(force_model=PropagationForceModel(j2=True),
                                         revolutions=1.5, report_path=path)
        state = leo_state()
        result = segment.run(state)
        assert result.segment == "tudatpy-injected-orbit-propagation"
        final = result.final_state
        assert final.frame == "GCRS"
        assert final.epoch.timescale == state.epoch.timescale
        period = orbital_elements(state.r_m, state.v_m_s)["period_s"]
        dt = state.epoch.delta_seconds(final.epoch)
        assert dt == pytest.approx(1.5 * period, rel=1e-6)
        assert final.mass_kg == state.mass_kg
        assert final.stage_config == state.stage_config
        assert result.events[0]["kind"] == "orbit_propagation"
        meta = final.metadata["m05_propagation"]
        assert meta["revolutions"] == 1.5
        assert meta["report_result"] in ("passed", "not_checked", "failed")
        assert path.exists()
        assert result.diagnostics["report_digest"] == json.loads(path.read_text())["digest"]
        prop = result.diagnostics["propagation"]
        assert isinstance(prop, InjectedOrbitPropagation)
        assert np.linalg.norm(np.asarray(final.r_m)) == pytest.approx(
            7_000_000.0, rel=1e-3)

    def test_segment_name_and_defaults(self):
        segment = OrbitPropagationSegment()
        assert segment.name == "tudatpy-injected-orbit-propagation"
        assert segment.revolutions == 1.0 and segment.report_path is None


# ---------------------------------------------------------------- package hygiene


def test_contract_package_does_not_drag_propagation_in():
    import subprocess
    import sys

    code = ("import orbitopt.mission as m, sys; "
            "assert not hasattr(m, 'propagation'); "
            "assert 'orbitopt.mission.propagation' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)


# ---------------------------------------------------------------- real tudatpy


@pytest.mark.slow
def test_tudatpy_real_short_arc_j2():
    pytest.importorskip("tudatpy")
    state = leo_state(epoch=Epoch(mjd2000=9400.5, timescale="UTC"))
    prop = propagate_injected_orbit(
        state, PropagationForceModel(j2=True, third_bodies=("Sun", "Moon")),
        revolutions=0.25, max_step_s=60.0, comparison_samples=48)
    report = orbit_propagation_report(prop, label="task-223-real")
    assert report["inputs"]["orbit_propagation"]["elements_history"][-1]["t_s"] > 0
    # J2 drags the node and shifts osculating elements; the analytic two-body
    # difference is physics now -- recorded, not gated.
    assert "residual.analytic_two_body_coast" not in report["status"]["checks"]
    assert report["inputs"]["analytic_comparison"]["max_position_m"] > 10.0
    elements = report["inputs"]["orbit_propagation"]["elements_history"]
    assert math.isfinite(elements[-1]["period_s"])
    assert abs(elements[-1]["period_s"] - prop.period_s) / prop.period_s < 1e-6
