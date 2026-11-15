"""E1-01 (task #250): backend identity & physics parity.

Three things are graded here:

1. the identity vocabulary (:mod:`orbitopt.flightlab.backends`) -- one
   canonical name per physics engine, case-insensitive resolution, and the
   old ambiguous ``3DOF``/``6DOF`` accepted ONLY as per-entry-point deprecated
   aliases (and never written into an artefact);
2. every producer records its identity: run manifest, verification report,
   scene metadata, and the CLI carries the canonical flags;
3. the aligned spike-stage scenario is parameter-identical across the
   backends (mass / propellant / inertia / mount / Isp exact; thrust within
   the documented derived-value band; aero geometry shared, JSBSim aero
   stays zero) via the shared parity source.

Nothing here needs the optional ``jsbsim`` package or the OT binary: parity
is asserted against the packaged XMLs and the shared parameter document, and
the JSBSim-backed identity wiring is covered by the existing
``test_flightlab_sixdof_backend.py`` (which ``importorskip``s jsbsim).
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

# The tests that FLY an F-07 mission (real RK4 propagations) carry
# @pytest.mark.slow individually; the identity/parity checks are fast.

from orbitopt.flightlab.backends import (
    BACKENDS,
    BACKEND_JSBSIM,
    BACKEND_NATIVE_POINTMASS,
    BACKEND_NATIVE_RIGIDBODY,
    BACKEND_OPENTSIOLOVSKY,
    FIDELITY,
    canonical_backend,
    describe,
)
from orbitopt.flightlab.contract import read_run, write_run
from orbitopt.flightlab.parity import (
    check_parity,
    jsbsim_declared_params,
    load_parity_scenario,
    native_booster,
    native_stage_pair,
    parity_report,
)
from orbitopt.flightlab.sixdof.mission_output import (
    BACKEND_3DOF,
    BACKEND_6DOF,
    BACKENDS as MISSION_BACKENDS,
    f07_scenario,
    mission_comparison_report,
    run_mission,
    to_flight_run_mission,
    write_mission_outputs,
)

SHORT = dict(separation_t_s=70.0, ignition_delay_s=8.0, t_end_s=150.0,
             dt_s=0.1, sample_every=10)


# ---------------------------------------------------------------------------
# 1. the identity vocabulary
# ---------------------------------------------------------------------------

def test_one_canonical_identity_per_engine():
    assert BACKENDS == ("opentsiolkovsky", "native-pointmass", "jsbsim",
                        "native-rigidbody")
    assert set(FIDELITY) == set(BACKENDS)
    for b in BACKENDS:
        assert describe(b) and b in describe(b)
    assert describe(None) == ""


def test_resolution_is_case_insensitive_and_entry_point_scoped():
    # simulate's entry point: 3DOF/6DOF resolve to THIS entry point's pair
    assert canonical_backend("6dof",
                             aliases={"3dof": BACKEND_OPENTSIOLOVSKY,
                                      "6dof": BACKEND_JSBSIM}) == BACKEND_JSBSIM
    assert canonical_backend(" 3DOF ",
                             aliases={"3dof": BACKEND_OPENTSIOLOVSKY,
                                      "6dof": BACKEND_JSBSIM}) == BACKEND_OPENTSIOLOVSKY
    # mission's entry point: the same alias means something else, by design
    assert canonical_backend("6DOF",
                             aliases={"3dof": BACKEND_NATIVE_POINTMASS,
                                      "6dof": BACKEND_NATIVE_RIGIDBODY},
                             allowed=MISSION_BACKENDS) == BACKEND_NATIVE_RIGIDBODY
    # an identity another entry point owns is not accepted when restricted
    with pytest.raises(ValueError, match="not supported"):
        canonical_backend("opentsiolkovsky", allowed=MISSION_BACKENDS)
    for bad in ("8DOF", "rocketpy", ""):
        with pytest.raises(ValueError, match="not supported"):
            canonical_backend(bad)


def test_simulate_default_is_the_canonical_opentsiolkovsky():
    from orbitopt.flightlab.simulate import simulate

    default = inspect.signature(simulate).parameters["backend"].default
    assert default == BACKEND_OPENTSIOLOVSKY
    # mission's restricted set does not accept simulate's jsbsim identity
    with pytest.raises(ValueError, match="not supported"):
        canonical_backend("jsbsim", allowed=MISSION_BACKENDS)


@pytest.mark.slow
def test_mission_rejects_unknown_and_accepts_deprecated_alias():
    m = run_mission(f07_scenario(**SHORT), backend="native-rigidbody")
    assert m.backend == BACKEND_NATIVE_RIGIDBODY
    assert run_mission(f07_scenario(**SHORT), backend="6dof").backend == m.backend
    assert run_mission(f07_scenario(**SHORT),
                       backend="3DOF").backend == BACKEND_NATIVE_POINTMASS


# ---------------------------------------------------------------------------
# 2. identity in every artefact (run dir, report, scene, CLI)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def native_runs(tmp_path_factory):
    out = tmp_path_factory.mktemp("identity")
    runs = {}
    for backend in MISSION_BACKENDS:
        m = run_mission(f07_scenario(**SHORT), backend=backend)
        paths = write_mission_outputs(m, out / backend)
        runs[backend] = (to_flight_run_mission(m), paths)
    return runs


@pytest.mark.slow
def test_run_manifest_records_the_backend_and_round_trips(native_runs, tmp_path):
    for backend, (run, paths) in native_runs.items():
        assert run.backend == backend
        manifest = json.loads((Path(paths["run_dir"]) / "manifest.json")
                              .read_text(encoding="utf-8"))
        assert manifest["backend"] == backend
        back = read_run(paths["run_dir"])
        assert back.backend == backend
    # a run written without an identity (legacy) stays valid and None
    run = native_runs[MISSION_BACKENDS[0]][0]
    from dataclasses import replace as dreplace

    plain = dreplace(run, backend=None)
    write_run(tmp_path / "legacy", plain)
    assert read_run(tmp_path / "legacy").backend is None


@pytest.mark.slow
def test_verification_report_inputs_carry_identity_and_fidelity(native_runs):
    for backend, (_, paths) in native_runs.items():
        report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
        assert report["inputs"]["backend"] == backend
        assert backend in report["inputs"]["backend_fidelity"]


@pytest.mark.slow
def test_scene_flightrun_block_carries_identity(native_runs):
    for backend, (_, paths) in native_runs.items():
        scene = json.loads(Path(paths["scene"]).read_text(encoding="utf-8"))
        assert scene["flightRun"]["backend"] == backend


@pytest.mark.slow
def test_comparison_report_names_backends_canonically(native_runs):
    m3 = run_mission(f07_scenario(**SHORT), backend=BACKEND_3DOF)
    m6 = run_mission(f07_scenario(**SHORT), backend=BACKEND_6DOF)
    report = mission_comparison_report(m3, m6)
    assert set(report["inputs"]["backends"]) == {BACKEND_3DOF, BACKEND_6DOF}
    assert report["label"].endswith("native-rigidbody-vs-native-pointmass")
    prefixes = {k.split(".", 1)[0] for k in report["inputs"]["checks"]
                if "." in k and not k.startswith("check.")}
    assert prefixes == {BACKEND_3DOF, BACKEND_6DOF}


def test_cli_flags_take_canonical_names_and_aliases():
    from orbitopt.cli import build_parser

    p = build_parser()
    a = p.parse_args(["flight", "simulate", "t.yaml", "--vehicle", "v.yaml",
                      "-o", "r", "--backend", "JSBSIM"])
    assert a.backend == BACKEND_JSBSIM
    a = p.parse_args(["flight", "simulate", "t.yaml", "--vehicle", "v.yaml",
                      "-o", "r"])
    assert a.backend == BACKEND_OPENTSIOLOVSKY
    a = p.parse_args(["flight", "mission", "--backend", "native-pointmass"])
    assert a.backend == BACKEND_NATIVE_POINTMASS
    a = p.parse_args(["flight", "mission", "--backend", "6dof"])  # deprecated alias
    assert a.backend == BACKEND_NATIVE_RIGIDBODY
    for bad in ("8DOF", "rocketpy"):
        with pytest.raises(SystemExit):
            p.parse_args(["flight", "mission", "--backend", bad])
        with pytest.raises(SystemExit):
            p.parse_args(["flight", "simulate", "t.yaml", "--vehicle", "v",
                          "-o", "r", "--backend", bad])


@pytest.mark.slow
def test_no_artefact_carries_an_ambiguous_backend_label(native_runs):
    # the string that used to name two engines appears nowhere in outputs
    for backend, (_, paths) in native_runs.items():
        blob = ""
        for key in ("run_dir", "report", "scene"):
            path = Path(paths[key])
            blob += path.read_text(encoding="utf-8") if path.is_file() else ""
            if path.is_dir():
                blob += "".join(p.read_text(encoding="utf-8", errors="replace")
                                for p in path.rglob("*.json"))
        for token in ('"backend": "3DOF"', '"backend": "6DOF"',
                      '"backend": "3dof"', '"backend": "6dof"'):
            assert token not in blob


# ---------------------------------------------------------------------------
# 3. physics parity of the aligned scenario
# ---------------------------------------------------------------------------

def test_parity_checks_all_green():
    checks = check_parity()
    assert checks, "parity checks disappeared"
    failed = [c for c in checks if not c.passed]
    assert not failed, [(c.id, c.value, c.limit) for c in failed]
    assert parity_report()["all_passed"] is True


def test_jsbsim_xmls_declare_what_the_parity_source_states():
    jb = jsbsim_declared_params()
    p = load_parity_scenario()
    assert jb["empty_mass_kg"] == float(p.jsbsim["empty_mass_kg"])
    assert jb["propellant_mass_kg"] == pytest.approx(
        float(p.jsbsim["propellant_mass_kg"]), rel=0.0)
    assert tuple(jb["inertia_kg_m2"]) == tuple(
        float(v) for v in p.jsbsim["inertia_kg_m2"])
    assert jb["cg_x_m"] == pytest.approx(float(p.jsbsim["cg_x_m"]))
    assert jb["thruster_x_m"] - jb["cg_x_m"] == pytest.approx(
        float(p.stage1_engine["mount_x_cg_m"]))
    assert jb["isp_s"] == float(p.stage1_engine["isp_vacuum_s"])
    assert jb["propflowmax_kg_s"] == pytest.approx(
        float(p.jsbsim["propflowmax_kg_s"]), rel=1e-5)
    assert jb["nozzle_exit_area_m2"] == pytest.approx(
        float(p.jsbsim["nozzle_exit_area_m2"]), rel=1e-4)
    assert jb["nozzle_exit_area_m2"] == pytest.approx(
        float(p.stage1_engine["nozzle_exit_area_m2"]), rel=1e-4)
    assert jb["aero_declared"] is False


def test_f06_scenario_consumes_the_shared_source():
    from orbitopt.flightlab.sixdof.two_stage import f06_two_stage_scenario

    p = load_parity_scenario()
    pair = native_stage_pair(p)
    study = f06_two_stage_scenario()
    for side in ("stack", "booster", "upper"):
        got = getattr(study.pair, side)
        want = getattr(pair, side)
        assert (got.dry_mass_kg, got.propellant_mass_kg, got.inertia_kg_m2) == \
            (want.dry_mass_kg, want.propellant_mass_kg, want.inertia_kg_m2)
        assert got.engine.thrust_vacuum_n == want.engine.thrust_vacuum_n
        assert got.engine.isp_vacuum_s == want.engine.isp_vacuum_s
        assert got.engine.mount_x_cg_m == want.engine.mount_x_cg_m
    assert study.pair.separation_impulse_ns == pair.separation_impulse_ns
    # a call-level impulse override stays possible (schedule, not vehicle)
    assert f06_two_stage_scenario(
        separation_impulse_ns=30_000.0).pair.separation_impulse_ns == 30_000.0


def test_f05_booster_and_parity_booster_are_the_same_vehicle():
    from orbitopt.flightlab.sixdof.gravity_turn import f05_scenario

    v = f05_scenario().vehicle
    b = native_booster(load_parity_scenario(), name="spike_stage")
    assert v.name == b.name
    assert (v.dry_mass_kg, v.propellant_mass_kg, v.inertia_kg_m2) == \
        (b.dry_mass_kg, b.propellant_mass_kg, b.inertia_kg_m2)
    assert v.engine.thrust_vacuum_n == b.engine.thrust_vacuum_n


def test_thrust_band_is_the_documented_derived_value_gap():
    """E6-01 (#273): propflowmax is DERIVED from the canonical pair (choked
    flow), so the derived-vs-canonical gap is the XML's lb/s rounding, not the
    ~0.15% E4 spike left behind."""
    p = load_parity_scenario()
    jb = jsbsim_declared_params()
    derived = jb["isp_s"] * float(p.jsbsim["g0_m_s2"]) * jb["propflowmax_kg_s"]
    rel = abs(derived - float(p.stage1_engine["thrust_vacuum_n"])) / derived
    assert rel == pytest.approx(3.1e-8, abs=1e-8)   # lb/s rounding, nothing else
    assert rel <= p.thrust_rel_tol


def test_pad_thrust_is_parity_checked_not_assumed():
    """The shared exit area means the ambient debit, not just the vacuum
    thrust, is a parity parameter: at ISA sea level both engines must report
    thrust_vacuum_n - 2 m^2 * 101325 Pa = 1.6774 MN."""
    p = load_parity_scenario()
    jb = jsbsim_declared_params()
    area = float(p.stage1_engine["nozzle_exit_area_m2"])
    pad = float(p.stage1_engine["thrust_vacuum_n"]) - area * 101325.0
    assert pad == pytest.approx(1.67735e6, rel=1e-5)
    by_id = {c.id: c for c in check_parity(p)}
    area_check = by_id["parity-jsbsim-nozzle-area-vs-source"]
    pad_check = by_id["parity-engine-pad-thrust-vs-jsbsim"]
    assert area_check.passed and pad_check.passed
    assert pad_check.value <= pad_check.limit


def test_run_result_schema_enum_matches_the_vocabulary():
    from orbitopt.flightlab.contract import _schema

    schema = _schema("flight-result-1.0.json")
    assert set(schema["properties"]["backend"]["enum"]) == set(BACKENDS)
