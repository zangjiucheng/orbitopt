"""Formal 6-DOF Flight Lab backend (F-01, task #226): config-driven
``backend="6DOF"`` on the FL-401/402/403 spike drivers.

The migrated acceptance claims:

* every packaged scenario config loads and flies a FlightResult v1 run dir
  that round-trips through ``read_run`` (physics = the backend's own
  checks);
* vehicle / initial conditions / integration step switch with config edits
  alone -- no code touched (the dataclass-replace and re-digest tests);
* the FL-402 handoff bookkeeping and the FL-403 GO bounds are per-run
  checks with the config's declared limits (the same thresholds the spike
  tests asserted; closed-loop runs are ``slow``);
* ``simulate(backend="6DOF")`` dispatches without disturbing the 3-DOF
  default, and the CLI accepts ``--backend``;
* repeated runs of one config are byte-identical (determinism).

Needs the opt-in ``jsbsim==1.3.3`` extra (``pip install orbitopt[sixdof]``);
everything skips without it.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

import pytest
import yaml

jsbsim = pytest.importorskip("jsbsim", reason="6-DOF backend is opt-in (orbitopt[sixdof])")

from orbitopt.flightlab.adapters.sixdof import (  # noqa: E402
    BACKEND,
    JSBSIM_VERSION,
    SixDofConfig,
    SixDofConfigError,
    load_config,
    packaged_config,
    run_ascent,
)
from orbitopt.flightlab.contract import load_target, read_run  # noqa: E402
from orbitopt.flightlab.simulate import simulate  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SCENARIOS = ("fl401_vertical_hop", "fl402_staging_handoff", "fl403_program_tracking")
FL403_REFERENCE = (REPO / "tests/fixtures/flightlab/opentsiolkovsky"
                   / "fl403_spike/ot_out/trajectory.csv")


def _files_digest(run_dir: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(run_dir.rglob("*")):
        if p.is_file():
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def _minimal_target(tmp_path: Path) -> "object":
    doc = yaml.safe_load((REPO / "src/orbitopt/data/flightlab/scenarios"
                          / "leo_two_stage.yaml").read_text(encoding="utf-8"))
    doc["recovery"] = {"kind": "none"}
    doc["constraints"] = {}
    path = tmp_path / "sixdof-min-target.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return load_target(path)


# --------------------------------------------------------------------------
# config document
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", SCENARIOS)
def test_packaged_scenarios_load(name):
    cfg = load_config(packaged_config(name))
    assert cfg.name and cfg.model and cfg.digest
    assert cfg.dt_s > 0.0
    if name == "fl401_vertical_hop":
        assert cfg.control_mode == "hold" and cfg.staging is None
        assert cfg.duration_s == 30.0
    if name == "fl402_staging_handoff":
        assert cfg.staging is not None and cfg.staging.coast_s >= 60.0
    if name == "fl403_program_tracking":
        assert cfg.control_mode == "pid" and cfg.pid.meco_t_s == 100.0
        assert cfg.duration_for_run_s() == 105.0  # pid derives duration from MECO+coast


def test_config_requires_name_model_and_duration():
    with pytest.raises(SixDofConfigError, match="'name'"):
        load_config({"model": "spike_stage", "duration_s": 1.0})
    with pytest.raises(SixDofConfigError, match="duration_s"):
        load_config({"name": "x", "model": "spike_stage"})
    with pytest.raises(SixDofConfigError, match="provenance"):
        load_config({"name": "x", "model": "m", "duration_s": 1.0,
                     "provenance": "flight_certified"})


def test_config_rejects_typos_naming_the_key():
    with pytest.raises(SixDofConfigError, match="ic has unknown key"):
        load_config({"name": "x", "model": "m", "duration_s": 1.0,
                     "ic": {"altitude_m": 1000.0}})  # alt_m, not altitude_m
    with pytest.raises(SixDofConfigError, match="control.mode"):
        load_config({"name": "x", "model": "m", "duration_s": 1.0,
                     "control": {"mode": "lqr"}})
    with pytest.raises(SixDofConfigError, match="pid.*attitude_program|attitude_program"):
        load_config({"name": "x", "model": "m", "control": {"mode": "pid",
                                                            "meco_t_s": 10.0}})
    with pytest.raises(SixDofConfigError, match="staging is supported with"):
        load_config({"name": "x", "model": "m", "control": {"mode": "pid",
                                                            "meco_t_s": 10.0,
                                                            "attitude_program": [
                                                                {"t_s": 0.0}]},
                     "staging": {"at_t_s": 5.0}})
    with pytest.raises(SixDofConfigError, match="booster_dry_fraction"):
        load_config({"name": "x", "model": "m", "duration_s": 10.0,
                     "staging": {"at_t_s": 5.0, "booster_dry_fraction": 1.4}})
    with pytest.raises(SixDofConfigError, match="step_order"):
        load_config({"name": "x", "model": "m", "duration_s": 10.0,
                     "staging": {"at_t_s": 5.0, "step_order": "simultaneous"}})
    with pytest.raises(SixDofConfigError, match="exceeds"):
        load_config({"name": "x", "model": "m", "duration_s": 4.0,
                     "staging": {"at_t_s": 5.0}})
    with pytest.raises(SixDofConfigError, match="not a number"):
        load_config({"name": "x", "model": "m", "duration_s": "thirty seconds"})


# --------------------------------------------------------------------------
# FL-401 lineage: hold mode through the adapter and the Gate-B pipeline
# --------------------------------------------------------------------------

def test_hold_run_roundtrips_and_is_deterministic(tmp_path):
    cfg = load_config(packaged_config("fl401_vertical_hop"))
    run = run_ascent(cfg, tmp_path / "run-a")
    assert run.verification.physics_verified is True
    assert run.verification.mission_constraints_passed is None  # not run_ascent's call
    assert all(t.frame == "ECEF" for t in run.tracks)
    assert run.engines.get("jsbsim") == JSBSIM_VERSION
    assert run.created_utc == cfg.epoch_utc  # no wall-clock in the run
    kinds = {e.kind for e in run.events}
    assert "ignition" in kinds
    back = read_run(tmp_path / "run-a")
    assert back.run_id == run.run_id
    assert [t.body_id for t in back.tracks] == [cfg.body_id]
    # byte-determinism: the same config twice writes identical run dirs
    run2 = run_ascent(cfg, tmp_path / "run-b")
    assert run2.run_id == run.run_id
    assert _files_digest(tmp_path / "run-a") == _files_digest(tmp_path / "run-b")


def test_vehicle_and_ic_switch_is_config_only(tmp_path):
    """F-01 acceptance: another vehicle + another IC needs only a config
    edit -- here the packaged FL-401 document with model/ic/dt overrides."""
    base = dict(load_config(packaged_config("fl401_vertical_hop")).raw)
    swapped = copy.deepcopy(base)
    swapped.update(model="spike_booster", dt_s=0.005, duration_s=5.0)
    swapped["ic"] = dict(base["ic"], alt_m=100000.0, pitch_deg=85.0)
    r_orig = run_ascent(load_config(base), tmp_path / "orig")
    r_alt = run_ascent(load_config(swapped), tmp_path / "alt")
    assert r_alt.run_id != r_orig.run_id       # identity tracks the document
    assert r_alt.vehicle_sha256 != r_orig.vehicle_sha256
    assert r_alt.verification.physics_verified is True
    assert r_alt.engines["aircraft"] == "spike_booster"
    # the IC really flew: track starts at the configured 100 km geodetic alt
    t0 = r_alt.tracks[0]
    r0 = math.sqrt(sum(c * c for c in t0.position_m[0]))
    assert abs(r0 - 6_378_137.0 - 100_000.0) < 5_000.0  # equator, spherical-ish


def test_simulate_dispatches_backend_without_touching_the_default(tmp_path):
    target = _minimal_target(tmp_path)
    run = simulate(target, packaged_config("fl401_vertical_hop"),
                   tmp_path / "gateb6", backend="6dof", refly=False)
    v = run.verification
    assert v.physics_verified is True   # backend checks
    # a vertical hop never reaches orbit: the insertion checks are there and
    # honest, and the 3-DOF-only omissions are recorded
    assert v.mission_constraints_passed is False
    assert any(c.id == "insertion-above-atmosphere" for c in v.checks)
    assert any("no tudatpy insertion re-fly" in a for a in run.assumptions)
    read_run(tmp_path / "gateb6")  # the final run dir validates
    with pytest.raises(ValueError, match="8DOF"):
        simulate(target, packaged_config("fl401_vertical_hop"),
                 tmp_path / "bad", backend="8DOF")


def test_cli_accepts_backend_flag_case_insensitively():
    from orbitopt.cli import build_parser
    args = build_parser().parse_args(
        ["flight", "simulate", "t.yaml", "--vehicle", "v.yaml", "-o", "run",
         "--backend", "6dof"])
    assert args.backend == BACKEND
    default = build_parser().parse_args(
        ["flight", "simulate", "t.yaml", "--vehicle", "v.yaml", "-o", "run"])
    assert default.backend == "3DOF"
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["flight", "simulate", "t.yaml", "--vehicle", "v.yaml", "-o", "run",
             "--backend", "rocketpy"])


# --------------------------------------------------------------------------
# FL-402 lineage: the two-instance handoff, three bodies, bookkeeping
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_staging_run_carries_the_handoff_as_checks(tmp_path):
    run = run_ascent(packaged_config("fl402_staging_handoff"), tmp_path / "stg")
    assert run.verification.physics_verified is True
    assert [t.body_id for t in run.tracks] == ["stack", "booster", "upper"]
    assert run.tracks[1].parent_body_id == run.tracks[2].parent_body_id == "stack"
    by_id = {t.body_id: t for t in run.tracks}
    for child in ("booster", "upper"):
        assert by_id[child].t_end_s - by_id[child].t_start_s >= 61.0 - 0.01 - 1e-9
    checks = {c.id: c for c in run.verification.checks}
    for cid in ("sixdof-mass-closure", "sixdof-momentum-residual",
                "sixdof-position-continuity", "sixdof-attitude-continuity",
                "sixdof-velocity-continuity"):
        assert checks[cid].passed, (cid, checks[cid].value, checks[cid].limit)
    seps = [e for e in run.events if e.kind == "separation"]
    assert {e.body_id for e in seps} == {"stack", "booster", "upper"}


# --------------------------------------------------------------------------
# FL-403 lineage: closed-loop TVC, GO bounds from the config
# --------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not FL403_REFERENCE.is_file(),
                    reason="FL-403 3-DOF reference fixture not recorded")
def test_tracking_run_meets_the_configured_go_bounds(tmp_path):
    run = run_ascent(packaged_config("fl403_program_tracking"), tmp_path / "trk")
    checks = {c.id: c for c in run.verification.checks}
    assert run.verification.physics_verified is True
    # the config's bounds are the limits the checks carry (2 deg settled
    # max / 0.6 deg RMS / 0 % saturation / 500 m / 20 m/s / 0.2 deg)
    cfg = load_config(packaged_config("fl403_program_tracking"))
    assert checks["sixdof-att-tracking-settled-max"].limit == cfg.tracking.settled_max_deg
    assert checks["sixdof-att-tracking-settled-rms"].limit == cfg.tracking.settled_rms_deg
    assert checks["sixdof-gimbal-saturation"].limit == cfg.tracking.gimbal_sat_fraction_max
    for cid in ("sixdof-att-tracking-full-max", "sixdof-att-tracking-settled-max",
                "sixdof-att-tracking-settled-rms", "sixdof-gimbal-saturation",
                "sixdof-meco-altitude-deviation", "sixdof-meco-speed-deviation",
                "sixdof-meco-fpa-deviation"):
        assert checks[cid].passed, (cid, checks[cid].value, checks[cid].limit)
    assert {e.kind for e in run.events} == {"ignition", "meco"}
    # the reference fixture resolved: deviations were assessed, not skipped
    assert not any(c.id == "sixdof-meco-deviation-not-assessed"
                   for c in run.verification.checks)


@pytest.mark.slow
@pytest.mark.skipif(not FL403_REFERENCE.is_file(),
                    reason="FL-403 3-DOF reference fixture not recorded")
def test_tracking_dt_halving_agrees(tmp_path):
    """Same scenario at dt/2: the integration is converged (FL-403's dt
    claim now expressed as a check on two config-driven runs)."""
    base = dict(load_config(packaged_config("fl403_program_tracking")).raw)
    r1 = run_ascent(base, tmp_path / "dt01")
    r2 = run_ascent(dict(base, dt_s=0.005), tmp_path / "dt005")
    c1 = {c.id: c for c in r1.verification.checks}
    c2 = {c.id: c for c in r2.verification.checks}
    assert r1.verification.physics_verified and r2.verification.physics_verified
    assert abs(c1["sixdof-meco-altitude-deviation"].value
               - c2["sixdof-meco-altitude-deviation"].value) < 100.0
    assert abs(c1["sixdof-att-tracking-full-max"].value
               - c2["sixdof-att-tracking-full-max"].value) < 0.1


def test_missing_reference_reports_not_assessed(tmp_path):
    """The MECO-deviation checks are conditional on the reference CSV -- a
    config pointing at a missing file says so instead of silently passing."""
    trk = dict(load_config(packaged_config("fl403_program_tracking")).raw)
    trk["reference_csv"] = "does/not/exist/trajectory.csv"
    run = run_ascent(trk, tmp_path / "noref")
    skip = next(c for c in run.verification.checks
                if c.id == "sixdof-meco-deviation-not-assessed")
    assert skip.passed is False
    assert skip.reason_code == "REFLY_BACKEND_MISSING"
    assert not any(c.id == "sixdof-meco-altitude-deviation" for c in run.verification.checks)
    assert run.verification.physics_verified is False  # not-assessed is a failed check


def test_run_ascent_accepts_config_object(tmp_path):
    cfg = load_config(packaged_config("fl401_vertical_hop"))
    assert isinstance(cfg, SixDofConfig)
    run = run_ascent(cfg, tmp_path / "obj")
    assert run.verification.physics_verified is True
