"""Verification report (task #137): residuals, integrator tolerance + step
convergence, energy / angular-momentum error and the perturbation-model
inventory, as reproducible machine-readable JSON. Pure numpy/scipy: scipy's
DOP853 stands in for tudatpy behind the same duck-typed PropagationResult
interface (``model`` / ``segments`` / ``rerun``) tudat_propagate provides.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from orbitopt.verify import report as vr

MU_EARTH = 3.986004418e14
R_GEO = 42164.17e3
J2 = 1.08262668e-3
R_EARTH = 6378.137e3


@dataclass
class FakeResult:
    epochs: np.ndarray
    states: np.ndarray
    final_state: np.ndarray
    model: dict
    segments: list
    rerun: Optional[Callable] = field(default=None, repr=False)


def _rhs(j2: bool):
    def f(_t, y):
        r = y[:3]
        rn = np.linalg.norm(r)
        a = -MU_EARTH * r / rn**3
        if j2:
            z2 = (r[2] / rn) ** 2
            k = 1.5 * J2 * MU_EARTH * R_EARTH**2 / rn**5
            a = a + k * r * (5 * z2 - 1) - 2 * k * np.array([0.0, 0.0, r[2]])
        return np.concatenate([y[3:], a])
    return f


def propagate(y0, arcs, perturbing_bodies=("Earth",), earth_spherical_harmonic=None,
              rtol=1e-11, atol=1e-6, step_size=600.0, integrator="dop853"):
    j2 = earth_spherical_harmonic is not None
    t, y = 0.0, np.asarray(y0, dtype=float)
    E, S, segs, n = [], [], [], 0
    for arc in arcs:
        if arc["type"] == "impulsive_burn":
            y = y.copy()
            y[3:] += arc["delta_v"]
            continue
        sol = solve_ivp(_rhs(j2), (t, t + arc["duration"]), y, method="DOP853",
                        rtol=rtol, atol=atol, max_step=step_size)
        E.append(sol.t)
        S.append(sol.y.T)
        segs.append((n, n + len(sol.t)))
        n += len(sol.t)
        t, y = sol.t[-1], sol.y[:, -1]
    model = {
        "function": "fake", "central_body": "Earth",
        "perturbing_bodies": list(perturbing_bodies),
        "earth_spherical_harmonic": None if not j2 else list(earth_spherical_harmonic),
        "integrator": {"name": "dop853", "method": "scipy DOP853", "adaptive": True, "rtol": rtol,
                       "atol": atol, "step_s": None, "min_step_s": None, "max_step_s": step_size},
        "initial_state": np.asarray(y0, dtype=float), "initial_epoch_ephemeris_s": 0.0,
        "duration_s": float(t), "arcs": arcs,
    }

    def rerun(**ov):
        kw = dict(perturbing_bodies=perturbing_bodies, earth_spherical_harmonic=earth_spherical_harmonic,
                  rtol=rtol, atol=atol, step_size=step_size)
        kw.update(ov)
        return propagate(y0, arcs, **kw)

    return FakeResult(np.concatenate(E), np.concatenate(S), y, model, segs, rerun)


def _geo_state():
    v = np.sqrt(MU_EARTH / R_GEO)
    return np.array([R_GEO, 0.0, 0.0, 0.0, v * np.cos(0.01), v * np.sin(0.01)])


def _report(result, **kw):
    section = vr.propagation_section(result, "coast", mu=MU_EARTH, **kw)
    return vr.build_report(
        kind="test", label="geo", propagations=[section],
        residuals=[vr.state_residual("final_vs_initial", "one period later", state=result.final_state,
                                     reference_state=_geo_state(), tolerance={"position_m": 1e9})],
        inputs={"x": 1}, targets={"converged": True},
    )


def test_inventory_names_j2_j22_and_third_bodies():
    inv = vr.perturbation_inventory("Earth", ("Earth", "Moon", "Sun"), (2, 2))
    earth = inv["forces"][0]
    assert earth["model"] == "spherical_harmonic" and "J2" in earth["terms"]
    assert any("J22" in t for t in earth["terms"])
    s = inv["summary"]
    assert s["earth_J2"] and s["earth_J22"] and s["sun_gravity"] and s["moon_gravity"]
    assert s["third_bodies"] == ["Moon", "Sun"] and not s["central_point_mass_only"]
    assert any("solar_radiation_pressure" in x for x in inv["not_modelled"])

    pm = vr.perturbation_inventory("Sun", ("Sun", "Earth", "Mars", "Jupiter"))
    assert not pm["summary"]["earth_J2"] and pm["forces"][0]["model"] == "point_mass"
    assert all(f["ephemeris"] for f in pm["forces"][1:])
    assert vr.perturbation_inventory("Earth", ("Earth",))["summary"]["central_point_mass_only"]


def test_spherical_harmonic_terms():
    assert vr.spherical_harmonic_terms(2, 0) == ["J2"]
    assert vr.spherical_harmonic_terms(2, 2) == ["J2", "C21/S21", "C22/S22 (J22)"]


def test_two_body_report_is_conservative_and_converged():
    period = 2 * np.pi * np.sqrt(R_GEO**3 / MU_EARTH)
    res = propagate(_geo_state(), [{"type": "coast", "duration": period}])
    rep = _report(res)
    vr.validate_report(rep)
    p = rep["propagations"][0]
    assert p["integrator"]["convergence"]["converged"]
    assert p["integrator"]["convergence"]["delta_position_m"] < 1.0
    assert p["integrator"]["steps"]["n_steps"] > 10
    assert p["integrator"]["steps"]["max_s"] <= 600.0 + 1e-9
    pm = p["conservation"]["propagated_model"]
    assert pm["applicable"] and pm["within_tolerance"]
    assert pm["energy_rel_drift_max"] < 1e-9
    assert rep["status"]["passed"]
    assert rep["residuals"][0]["position_residual_m"] < 10.0  # closes after one period


def test_perturbed_model_flags_conservation_as_diagnostic_and_runs_two_body_reference():
    period = 2 * np.pi * np.sqrt(R_GEO**3 / MU_EARTH)
    res = propagate(_geo_state(), [{"type": "coast", "duration": period}], earth_spherical_harmonic=(2, 0))
    rep = _report(res)
    vr.validate_report(rep)
    cons = rep["propagations"][0]["conservation"]
    assert not cons["propagated_model"]["applicable"]
    assert cons["propagated_model"]["angular_momentum_rel_drift_max"] > 1e-7  # J2 precession is physical
    assert cons["two_body_reference"]["applicable"] and cons["two_body_reference"]["within_tolerance"]
    assert rep["propagations"][0]["dynamics"]["summary"]["earth_J2"]


def test_burns_split_conservation_segments():
    period = 2 * np.pi * np.sqrt(R_GEO**3 / MU_EARTH)
    arcs = [{"type": "coast", "duration": period / 2},
            {"type": "impulsive_burn", "delta_v": np.array([0.0, 50.0, 0.0])},
            {"type": "coast", "duration": period / 2}]
    res = propagate(_geo_state(), arcs)
    p = vr.propagation_section(res, "c", mu=MU_EARTH)
    assert len(p["conservation"]["propagated_model"]["segments"]) == 2
    assert p["conservation"]["propagated_model"]["within_tolerance"]  # burn not counted as drift
    assert p["integrator"]["steps"]["n_segments"] == 2


def test_loose_tolerance_fails_step_convergence():
    period = 2 * np.pi * np.sqrt(R_GEO**3 / MU_EARTH)
    res = propagate(_geo_state(), [{"type": "coast", "duration": 20 * period}], rtol=1e-4, atol=1e2, step_size=1e5)
    rep = _report(res)
    assert not rep["propagations"][0]["integrator"]["convergence"]["converged"]
    assert not rep["status"]["passed"]


def test_report_is_reproducible_json(tmp_path):
    period = 2 * np.pi * np.sqrt(R_GEO**3 / MU_EARTH)
    a = _report(propagate(_geo_state(), [{"type": "coast", "duration": period}]))
    b = _report(propagate(_geo_state(), [{"type": "coast", "duration": period}]))
    assert a["digest"] == b["digest"]
    pa = vr.write_report(a, tmp_path / "a.json")
    pb = vr.write_report(b, tmp_path / "b.json")
    assert pa.read_bytes() == pb.read_bytes()
    loaded = vr.read_report(pa)
    vr.validate_report(loaded)
    assert loaded["digest"] == vr.report_digest(loaded)
    loaded["residuals"][0]["position_residual_m"] += 1.0
    with pytest.raises(ValueError):
        vr.validate_report(loaded)


def test_non_finite_values_become_null():
    assert json.loads(vr.canonical_json({"x": float("nan"), "y": np.array([1.0, np.inf])})) == {"x": None, "y": [1.0, None]}


def test_finalize_writes_to_env_dir_and_scene_roundtrip(tmp_path, monkeypatch):
    period = 2 * np.pi * np.sqrt(R_GEO**3 / MU_EARTH)
    rep = _report(propagate(_geo_state(), [{"type": "coast", "duration": period}]))
    monkeypatch.setenv(vr.REPORT_DIR_ENV, str(tmp_path))
    vr.finalize_report(rep)
    written = list(tmp_path.glob("*.verification.json"))
    assert [p.name for p in written] == ["test-geo.verification.json"]

    scene = {"id": "s"}
    vr.attach_to_scene(scene, rep)
    vr.attach_to_scene(scene, rep)
    reports = vr.pop_from_scene(scene)
    assert set(scene) == {"id", "subtitle"} and len(reports) == 2
    assert scene["subtitle"].count("Verification report test:geo: PASSED") == 2
    bundle = vr.bundle(reports, "s")
    vr.validate_report(bundle)
    assert bundle["status"]["passed"]


def test_cli_run_writes_verification_report_next_to_scene(tmp_path, monkeypatch, capsys):
    """`orbitopt run` pops the reports a mission adapter attached to the scene
    and writes them (bundled if several) as <scene>.verification.json."""
    import orbitopt.mission_config as mc
    import orbitopt.scene_format as sf
    from orbitopt.cli import main

    period = 2 * np.pi * np.sqrt(R_GEO**3 / MU_EARTH)
    rep = _report(propagate(_geo_state(), [{"type": "coast", "duration": period}]))
    written = {}

    def fake_build(config):
        return vr.attach_to_scene({"bodies": []}, rep)

    def fake_write(scene, path):
        written["scene"] = scene

    monkeypatch.setattr(mc, "build_scene_from_config", fake_build)
    monkeypatch.setattr(sf, "write_scene", fake_write)
    monkeypatch.setattr("orbitopt.cli._load_and_validate_config",
                        lambda path: {"mission": {"id": "m", "kind": "geo-raising"},
                                      "output": {"scene_file": str(tmp_path / "scene.json")}})
    main(["run", "cfg.yaml"])
    assert vr.SCENE_REPORTS_KEY not in written["scene"]
    out = tmp_path / "scene.verification.json"
    assert out.exists()
    doc = vr.read_report(out)
    vr.validate_report(doc)
    assert doc["digest"] == rep["digest"]
    assert "verification report" in capsys.readouterr().out


def test_fixed_step_convergence_compares_states_at_same_epoch():
    """Acceptance regression: a 1000 s request with RK4 stops at 1020 s
    (20 s step) and at 1005 s (10 s step). The report used to diff those raw
    final states and call ~113 km of ordinary orbital motion integration
    error. With an exact (analytic) history the error must be ~0."""
    n_rate = np.sqrt(MU_EARTH / R_GEO**3)
    v = np.sqrt(MU_EARTH / R_GEO)

    ends = {20.0: 1020.0, 10.0: 1005.0}  # observed overshoots of a 1000 s request

    def exact(step):
        t = np.r_[np.arange(0.0, ends[step] - 0.5 * step, step), ends[step]]
        th = n_rate * t
        s = np.c_[R_GEO * np.cos(th), R_GEO * np.sin(th), 0 * t, -v * np.sin(th), v * np.cos(th), 0 * t]
        model = {
            "function": "fake", "central_body": "Earth", "perturbing_bodies": ["Earth"],
            "earth_spherical_harmonic": None,
            "integrator": {"name": "rk4", "method": "RK4", "adaptive": False, "rtol": None, "atol": None,
                           "step_s": step, "min_step_s": None, "max_step_s": step},
            "initial_state": s[0], "initial_epoch_ephemeris_s": 0.0, "duration_s": 1000.0,
            "arcs": [{"type": "coast", "duration": 1000.0}],
        }
        return FakeResult(t, s, s[-1], model, [(0, len(t))], rerun=lambda step_size=None, **_: exact(step_size))

    nominal = exact(20.0)
    assert nominal.epochs[-1] == 1020.0 and exact(10.0).epochs[-1] == 1005.0
    assert np.linalg.norm(nominal.final_state[:3] - exact(10.0).final_state[:3]) > 4e4  # the old false error (15 s of GEO motion)
    conv = vr.propagation_section(nominal, "rk4", mu=MU_EARTH)["integrator"]["convergence"]
    assert conv["compared_at_epoch_s"] == 1005.0
    assert conv["delta_position_m"] < 1e-3
    assert conv["converged"]


def test_status_is_tri_state():
    """Task #162: no checks is ``not_checked``, never an implicit pass."""
    empty = vr.status_from_checks({})
    assert empty == {"result": "not_checked", "passed": False, "checks": {}}
    assert vr.status_from_checks({"a": True, "b": True})["result"] == "passed"
    failed = vr.status_from_checks({"a": True, "b": False})
    assert failed["result"] == "failed" and not failed["passed"]
    assert vr.format_status(empty).startswith("NOT CHECKED")
    # Legacy documents without ``result`` are re-derived from their checks.
    assert vr.status_result({"passed": True, "checks": {}}) == "not_checked"


def test_report_without_checks_is_not_checked_and_validates():
    rep = vr.build_report("test", "empty", propagations=[])
    vr.validate_report(rep)
    assert rep["status"]["result"] == "not_checked"
    assert not rep["status"]["passed"]


def test_bundle_tri_state():
    passed = vr.build_report("t", "ok", propagations=[], residuals=[
        vr.state_residual("r", "ref", position=[0, 0, 0], reference_position=[0, 0, 0],
                          tolerance={"position_m": 1.0})])
    failed = vr.build_report("t", "bad", propagations=[], residuals=[
        vr.state_residual("r", "ref", position=[10, 0, 0], reference_position=[0, 0, 0],
                          tolerance={"position_m": 1.0})])
    empty = vr.build_report("t", "none", propagations=[])
    assert passed["status"]["result"] == "passed" and failed["status"]["result"] == "failed"

    for reports, expected in [([passed, passed], "passed"), ([passed, failed], "failed"),
                              ([passed, empty], "not_checked"), ([failed, empty], "failed"),
                              ([], "not_checked")]:
        b = vr.bundle(reports, "b")
        vr.validate_report(b)
        assert b["status"]["result"] == expected, (reports, expected)
        assert b["status"]["passed"] == (expected == "passed")
    assert "t:none" not in vr.bundle([passed, empty], "b")["status"]["checks"]


def test_attach_to_scene_shows_status_in_subtitle():
    empty = vr.build_report("t", "none", propagations=[])
    scene = vr.attach_to_scene({"subtitle": "S"}, empty)
    assert scene["subtitle"].splitlines() == ["S", "Verification report t:none: NOT CHECKED (0/0 checks passed)"]


def test_cli_warns_on_not_checked_report(tmp_path, monkeypatch, capsys):
    import orbitopt.mission_config as mc
    import orbitopt.scene_format as sf
    from orbitopt.cli import main

    rep = vr.build_report("t", "none", propagations=[])
    monkeypatch.setattr(mc, "build_scene_from_config", lambda config: vr.attach_to_scene({"bodies": []}, rep))
    monkeypatch.setattr(sf, "write_scene", lambda scene, path: None)
    monkeypatch.setattr("orbitopt.cli._load_and_validate_config",
                        lambda path: {"mission": {"id": "m", "kind": "geo-raising"},
                                      "output": {"scene_file": str(tmp_path / "scene.json")}})
    main(["run", "cfg.yaml"])
    out = capsys.readouterr().out
    assert "WARNING: wrote verification report" in out and "NOT CHECKED" in out
