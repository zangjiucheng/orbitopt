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


def _run_cli_with_fake_builder(monkeypatch, tmp_path, report):
    """`orbitopt run` now goes through orbitopt.pipeline: register a fake
    geo-raising builder (pipeline signature) that attaches ``report`` to a
    valid scene, feed a schema-complete config, and stub the scene writer.
    Returns the dict the written scene lands in."""
    import sys

    import orbitopt.mission_config as mc
    import orbitopt.scene_format as sf
    from orbitopt.cli import main
    from orbitopt.viz.scene import body_entry, scene_document

    def fake_builder(config, recorder=None):
        trail = {"times": [0.0, 1.0], "positions": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]}
        scene = scene_document("m", "M", [body_entry("spacecraft", "Probe", "#ffffff", "spacecraft", trail=trail)],
                               distance_unit="km")
        return vr.attach_to_scene(scene, report)

    written = {}

    def fake_write(scene, path, validate=True):
        written["scene"] = scene

    monkeypatch.setattr(sys.modules[__name__], "_fake_cli_builder", fake_builder, raising=False)
    monkeypatch.setitem(mc._KIND_BUILDERS, "geo-raising", (__name__, "_fake_cli_builder"))
    monkeypatch.setattr(sf, "write_scene", fake_write)
    monkeypatch.setattr("orbitopt.cli._load_and_validate_config",
                        lambda path: {"schemaVersion": "1.0",
                                      "mission": {"id": "m", "title": "M", "kind": "geo-raising"},
                                      "output": {"scene_file": str(tmp_path / "scene.json")}})
    main(["run", "cfg.yaml"])
    return written


def test_cli_run_writes_verification_report_next_to_scene(tmp_path, monkeypatch, capsys):
    """`orbitopt run` pops the reports a mission adapter attached to the scene
    and writes them (bundled if several) as <scene>.verification.json."""
    period = 2 * np.pi * np.sqrt(R_GEO**3 / MU_EARTH)
    rep = _report(propagate(_geo_state(), [{"type": "coast", "duration": period}]))
    written = _run_cli_with_fake_builder(monkeypatch, tmp_path, rep)
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
    assert empty["result"] == "not_checked" and not empty["passed"] and empty["checks"] == {}
    # Task #243: the three categories always appear, empty ones as not_checked.
    assert empty["check_categories"] == {}
    assert set(empty["categories"]) == set(vr.CHECK_CATEGORIES)
    assert all(r == {"result": "not_checked", "checks_passed": 0, "checks_total": 0}
               for r in empty["categories"].values())
    assert vr.status_from_checks({"a": True}, {"a": vr.CHECK_NUMERICAL})["result"] == "passed"
    failed = vr.status_from_checks({"a": True, "b": False},
                                   {"a": vr.CHECK_NUMERICAL, "b": vr.CHECK_MODEL})
    assert failed["result"] == "failed" and not failed["passed"]
    assert failed["categories"][vr.CHECK_NUMERICAL]["result"] == "passed"
    assert failed["categories"][vr.CHECK_MODEL] == {"result": "failed", "checks_passed": 0, "checks_total": 1}
    assert vr.format_status(empty).startswith("NOT CHECKED")
    # Legacy documents without ``result`` are re-derived from their checks.
    assert vr.status_result({"passed": True, "checks": {}}) == "not_checked"


def test_every_check_must_be_categorised():
    """Task #243: a check without a category (or with an unknown one) is a
    producer bug, refused where the status is built."""
    with pytest.raises(ValueError):
        vr.status_from_checks({"a": True})
    with pytest.raises(ValueError):
        vr.status_from_checks({"a": True}, {"a": "instrumental_drift"})
    with pytest.raises(ValueError):
        vr.status_from_checks({"a": True}, {"a": vr.CHECK_MODEL, "b": vr.CHECK_MODEL})
    with pytest.raises(ValueError):  # hand-written residual dict without category
        vr.build_report("t", "x", propagations=[],
                        residuals=[{"name": "r", "reference": "ref", "within_tolerance": True}])


def test_auto_checks_are_categorised_by_kind():
    """Step convergence / conservation / targeting convergence are numerical;
    a residual is model agreement unless its producer says otherwise."""
    period = 2 * np.pi * np.sqrt(R_GEO**3 / MU_EARTH)
    rep = _report(propagate(_geo_state(), [{"type": "coast", "duration": period}]))
    vr.validate_report(rep)
    cats = rep["status"]["check_categories"]
    assert set(cats) == set(rep["status"]["checks"])
    assert cats["coast.step_convergence"] == vr.CHECK_NUMERICAL
    assert cats["coast.two_body_conservation"] == vr.CHECK_NUMERICAL
    assert cats["coast.model_conservation"] == vr.CHECK_NUMERICAL
    assert cats["targeting.converged"] == vr.CHECK_NUMERICAL
    assert cats["residual.final_vs_initial"] == vr.CHECK_MODEL  # state_residual default
    roll = rep["status"]["categories"]
    assert roll[vr.CHECK_NUMERICAL]["result"] == "passed"
    assert roll[vr.CHECK_MODEL]["result"] == "passed"
    assert roll[vr.CHECK_HISTORICAL] == {"result": "not_checked", "checks_passed": 0, "checks_total": 0}


def test_default_evidence_follows_the_category():
    """Task #263: with no explicit evidence, the type defaults from the
    category -- so the numerical checks stay numerical, the residual is
    treated as leaning on an independent model (the strongest reading a
    category alone supports), and ``independent_validation`` is ``passed``
    because that default counts the residual as validation-grade."""
    period = 2 * np.pi * np.sqrt(R_GEO**3 / MU_EARTH)
    rep = _report(propagate(_geo_state(), [{"type": "coast", "duration": period}]))
    ev = rep["status"]["check_evidence"]
    assert ev["coast.step_convergence"] == vr.EVIDENCE_NUMERICAL
    assert ev["coast.two_body_conservation"] == vr.EVIDENCE_NUMERICAL
    assert ev["coast.model_conservation"] == vr.EVIDENCE_NUMERICAL
    assert ev["targeting.converged"] == vr.EVIDENCE_NUMERICAL
    assert ev["residual.final_vs_initial"] == vr.EVIDENCE_INDEPENDENT_MODEL
    roll = rep["status"]["evidence"]
    assert roll[vr.EVIDENCE_NUMERICAL]["result"] == "passed"
    assert roll[vr.EVIDENCE_INDEPENDENT_MODEL]["result"] == "passed"
    assert roll[vr.EVIDENCE_DESIGN_INPUT]["result"] == "not_checked"
    assert roll[vr.EVIDENCE_ANALYTICAL]["result"] == "not_checked"
    assert roll[vr.EVIDENCE_HISTORICAL]["result"] == "not_checked"
    assert rep["status"]["independent_validation"]["result"] == "passed"


def test_design_input_evidence_is_never_independent_validation():
    """Task #263: a passing design_input check is a green check -- internal
    consistency with what the solution aimed at -- but a report built only on
    design-input and numerical evidence must say so in ``independent_validation``:
    ``not_checked``, never a spurious ``passed``."""
    rep = vr.build_report(
        "t", "aim-point only", propagations=[],
        residuals=[vr.state_residual("periapsis_vs_target", "the design aim point",
                                     position=[0, 0, 0], reference_position=[0, 0, 0],
                                     tolerance={"position_m": 1.0},
                                     evidence=vr.EVIDENCE_DESIGN_INPUT,
                                     scope="the design aim point (not a real-world reference)",
                                     phases=["insertion"])],
    )
    vr.validate_report(rep)
    assert rep["status"]["result"] == "passed"          # the aim point was hit
    assert rep["status"]["check_evidence"]["residual.periapsis_vs_target"] == vr.EVIDENCE_DESIGN_INPUT
    assert rep["status"]["evidence"][vr.EVIDENCE_DESIGN_INPUT]["result"] == "passed"
    assert rep["status"]["evidence"][vr.EVIDENCE_DESIGN_INPUT]["independent"] is False
    assert rep["status"]["independent_validation"]["result"] == "not_checked"
    assert rep["residuals"][0]["evidence"]["type"] == vr.EVIDENCE_DESIGN_INPUT
    assert rep["residuals"][0]["evidence"]["scope"] == "the design aim point (not a real-world reference)"
    assert rep["residuals"][0]["evidence"]["phases"] == ["insertion"]
    # And the cannot_prove text names the trap the acceptance is guarding
    # against -- a design-input match must never be read as historical proof.
    assert "NOT independent validation" in rep["status"]["evidence"][vr.EVIDENCE_DESIGN_INPUT]["cannot_prove"]


def test_independent_validation_honours_analytical_and_numerical_only_reports():
    """An all-analytic or all-numerical report has no *validation-grade*
    evidence, so its ``independent_validation`` rollup is ``not_checked`` --
    an analytic closed form of the same model is not external validation."""
    rep = vr.build_report(
        "t", "analytic coast", propagations=[],
        residuals=[vr.state_residual("parking_coast_vs_analytic", "secular J2 closed form",
                                     position=[0, 0, 0], reference_position=[0, 0, 0],
                                     tolerance={"position_m": 1.0},
                                     evidence=vr.EVIDENCE_ANALYTICAL)],
    )
    vr.validate_report(rep)
    assert rep["status"]["result"] == "passed"
    assert rep["status"]["evidence"][vr.EVIDENCE_ANALYTICAL]["result"] == "passed"
    assert rep["status"]["evidence"][vr.EVIDENCE_ANALYTICAL]["independent"] is False
    assert rep["status"]["independent_validation"]["result"] == "not_checked"


def test_historical_and_independent_model_evidence_are_validation_grade():
    """Both historical telemetry and an independent model count as validation:
    they reach outside the model under test. Mixed with a passing design-input
    check, the report's ``independent_validation`` rollup reflects only the
    validation-grade checks."""
    rep = vr.build_report(
        "t", "mixed evidence", propagations=[],
        residuals=[
            vr.state_residual("touchdown_vs_telemetry", "published flight telemetry",
                              position=[0, 0, 0], reference_position=[0, 0, 0],
                              tolerance={"position_m": 1.0}, category=vr.CHECK_HISTORICAL),
            vr.state_residual("vs_independent_backend", "an independently written propagator",
                              position=[0, 0, 0], reference_position=[0, 0, 0],
                              tolerance={"position_m": 1.0}),
            vr.state_residual("aim_point_check", "the design aim",
                              position=[0, 0, 0], reference_position=[0, 0, 0],
                              tolerance={"position_m": 1.0}, evidence=vr.EVIDENCE_DESIGN_INPUT),
        ],
    )
    vr.validate_report(rep)
    roll = rep["status"]["evidence"]
    assert roll[vr.EVIDENCE_HISTORICAL]["result"] == "passed"
    assert roll[vr.EVIDENCE_INDEPENDENT_MODEL]["result"] == "passed"
    assert roll[vr.EVIDENCE_DESIGN_INPUT]["result"] == "passed"
    assert roll[vr.EVIDENCE_HISTORICAL]["independent"] is True
    assert roll[vr.EVIDENCE_INDEPENDENT_MODEL]["independent"] is True
    assert roll[vr.EVIDENCE_DESIGN_INPUT]["independent"] is False
    iv = rep["status"]["independent_validation"]
    assert iv["result"] == "passed" and iv["checks_passed"] == 2 and iv["checks_total"] == 2


def test_independent_validation_fails_when_a_validation_check_fails():
    """The independent_validation rollup is a rollup, not an existence check:
    a failing historical-telemetry check flips it to ``failed`` even when
    every other kind is green."""
    rep = vr.build_report(
        "t", "failed re-fly", propagations=[],
        residuals=[
            vr.state_residual("vs_flown", "published flight telemetry",
                              position=[10, 0, 0], reference_position=[0, 0, 0],
                              tolerance={"position_m": 1.0}, category=vr.CHECK_HISTORICAL),
            vr.state_residual("aim", "design aim",
                              position=[0, 0, 0], reference_position=[0, 0, 0],
                              tolerance={"position_m": 1.0}, evidence=vr.EVIDENCE_DESIGN_INPUT),
        ],
    )
    assert rep["status"]["result"] == "failed"
    assert rep["status"]["evidence"][vr.EVIDENCE_HISTORICAL]["result"] == "failed"
    assert rep["status"]["independent_validation"]["result"] == "failed"


def test_status_rejects_unknown_evidence_type():
    with pytest.raises(ValueError):
        vr.status_from_checks({"a": True}, {"a": vr.CHECK_MODEL},
                              {"a": "published_internet_forum_post"})
    with pytest.raises(ValueError):
        vr.status_from_checks({"a": True}, {"a": vr.CHECK_MODEL},
                              {"b": vr.EVIDENCE_ANALYTICAL})   # for a check that does not exist


def test_evidence_block_qualification_is_complete():
    """Every evidence type carries label + independence flag + a proof claim
    + a limitation, so a reader of a raw JSON report sees what the pass covers
    and what it does not (task #263 acceptance)."""
    for et in vr.EVIDENCE_TYPES:
        q = vr.EVIDENCE_QUALIFICATION[et]
        assert set(q) >= {"label", "independent", "proves", "cannot_prove"}
        assert q["proves"] and q["cannot_prove"]
    assert vr.INDEPENDENT_EVIDENCE == (vr.EVIDENCE_HISTORICAL, vr.EVIDENCE_INDEPENDENT_MODEL)
    for et in vr.INDEPENDENT_EVIDENCE:
        assert vr.EVIDENCE_QUALIFICATION[et]["independent"] is True
    for et in (vr.EVIDENCE_ANALYTICAL, vr.EVIDENCE_DESIGN_INPUT, vr.EVIDENCE_NUMERICAL):
        assert vr.EVIDENCE_QUALIFICATION[et]["independent"] is False
    # The design_input entry names the trap it guards against explicitly.
    assert "NOT independent validation" in vr.EVIDENCE_QUALIFICATION[vr.EVIDENCE_DESIGN_INPUT]["cannot_prove"]


def test_historical_residuals_carry_their_category_and_bundle_rolls_up():
    period = 2 * np.pi * np.sqrt(R_GEO**3 / MU_EARTH)
    res = propagate(_geo_state(), [{"type": "coast", "duration": period}])
    section = vr.propagation_section(res, "coast", mu=MU_EARTH)
    flown = vr.build_report(
        kind="t", label="flown", propagations=[section],
        residuals=[vr.state_residual("touchdown_vs_telemetry", "published flight telemetry",
                                     position=res.final_state[:3], reference_position=_geo_state()[:3],
                                     tolerance={"position_m": 1e9}, category=vr.CHECK_HISTORICAL)],
    )
    vr.validate_report(flown)
    cats = flown["status"]["check_categories"]
    assert cats["residual.touchdown_vs_telemetry"] == vr.CHECK_HISTORICAL
    assert cats["coast.step_convergence"] == vr.CHECK_NUMERICAL
    roll = flown["status"]["categories"]
    assert roll[vr.CHECK_HISTORICAL] == {"result": "passed", "checks_passed": 1, "checks_total": 1}
    assert roll[vr.CHECK_MODEL] == {"result": "not_checked", "checks_passed": 0, "checks_total": 0}
    # Task #263: the historical residual is also historical-telemetry evidence,
    # which makes this report one of the rare ones whose independent_validation
    # rollup is actually ``passed``.
    assert flown["status"]["check_evidence"]["residual.touchdown_vs_telemetry"] == vr.EVIDENCE_HISTORICAL
    assert flown["status"]["evidence"][vr.EVIDENCE_HISTORICAL]["result"] == "passed"
    assert flown["status"]["independent_validation"]["result"] == "passed"
    b = vr.bundle([flown], "b")
    vr.validate_report(b)
    broll = b["status"]["categories"]
    assert broll[vr.CHECK_HISTORICAL] == {"result": "passed", "checks_passed": 1, "checks_total": 1}
    assert broll[vr.CHECK_NUMERICAL]["result"] == "passed" and broll[vr.CHECK_NUMERICAL]["checks_total"] == 3
    assert broll[vr.CHECK_MODEL]["result"] == "not_checked"
    # Task #263: the bundle carries the same evidence rollup + independent-
    # validation rollup as any report; a bundle inherits a sub-report's
    # validation-grade evidence.
    assert b["status"]["evidence"][vr.EVIDENCE_HISTORICAL]["result"] == "passed"
    assert b["status"]["independent_validation"]["result"] == "passed"
    assert vr.status_result(b["status"]) == "passed"


def test_bundle_evidence_rollup_and_independent_validation():
    """A bundle of a design-input-only report plus a numerical-only report is
    ``passed`` overall but its independent_validation is ``not_checked`` -- the
    bundle honestly reports having no validation-grade evidence (task #263)."""
    design = vr.build_report("t", "aim", propagations=[], residuals=[
        vr.state_residual("r1", "the design aim", position=[0, 0, 0], reference_position=[0, 0, 0],
                          tolerance={"position_m": 1.0}, evidence=vr.EVIDENCE_DESIGN_INPUT)])
    numeric = vr.build_report("t", "self", propagations=[], residuals=[
        vr.state_residual("r2", "same model re-run", position=[0, 0, 0], reference_position=[0, 0, 0],
                          tolerance={"position_m": 1.0}, category=vr.CHECK_NUMERICAL)])
    b = vr.bundle([design, numeric], "b")
    vr.validate_report(b)
    assert b["status"]["result"] == "passed"
    er = b["status"]["evidence"]
    assert er[vr.EVIDENCE_DESIGN_INPUT]["result"] == "passed"
    assert er[vr.EVIDENCE_NUMERICAL]["result"] == "passed"
    assert er[vr.EVIDENCE_INDEPENDENT_MODEL]["result"] == "not_checked"
    assert er[vr.EVIDENCE_HISTORICAL]["result"] == "not_checked"
    assert b["status"]["independent_validation"]["result"] == "not_checked"


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
    rep = vr.build_report("t", "none", propagations=[])
    _run_cli_with_fake_builder(monkeypatch, tmp_path, rep)
    out = capsys.readouterr().out
    assert "WARNING: wrote verification report" in out and "NOT CHECKED" in out


# ------------------------------------------------------------- provenance (task #295)

def test_provenance_nulls_only_a_tool_that_is_not_installed():
    rep = vr.provenance()
    assert rep["python"] and rep["numpy"] == np.__version__
    for tool in ("tudatpy", "pykep"):
        import importlib.util

        if importlib.util.find_spec(tool) is None:
            assert rep[tool] is None, f"{tool} is not importable but provenance says {rep[tool]!r}"
        else:
            assert rep[tool], f"{tool} is importable, so its provenance is never null"


def test_provenance_reads_a_conda_record_when_the_package_has_no_dist_info(tmp_path, monkeypatch):
    """The report asked ``importlib.metadata`` and nothing else, so a conda
    build without dist-info (tudatpy) read as null -- the same false absence
    the FL-601 toolchain record once carried. It shares that lookup now."""
    import importlib.metadata
    import json as _json

    from orbitopt.flightlab import toolchain as tc

    meta = tmp_path / "conda-meta"
    meta.mkdir()
    (meta / "tudatpy-1.1.0-py310hfeedface_0.json").write_text(_json.dumps({}))
    monkeypatch.setattr(tc.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(importlib.metadata, "version",
                        lambda _name: (_ for _ in ()).throw(importlib.metadata.PackageNotFoundError))
    assert vr._version("tudatpy") == "1.1.0"
    assert vr._version("no-such-package-at-all-xyz") is None
