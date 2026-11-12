"""Gate G (FL-304): `simulate --recovery` end-to-end + Monte Carlo.

Fast tests pin the honest stub verdict (the packaged ascent separates with no
recoverable propellant: the gate must fail with the FL-302 floor reason, the
debris track must be gone, nothing may be whitewashed) and the stdlib-side
verdict plumbing (declared recovery without a flown booster = failed
data-missing gate; the landing ids in checks.py mirror the FL-303 set).

The @slow tests fly the real thing. The textbook-RTLS ascent (separation
8 km up, 40 km down, 250 m/s) drives the full simulate -> FL-302 -> FL-303
chain and must come back *green* with FL-305's closed-loop terminal guidance
(ADR flightlab-0002): FL-302 hands the corridor off at ~3 km / ~306 m/s,
the closed loop lands it inside every FL-303 touchdown limit and the
propellant reserve -- no substituted landing anywhere. The same handoff
flown by FL-303's single open-loop plan (``landing: {mode: single}``) must
still come back *honest red* with the governing landing reason code
(landing.py "G3 note": under-thrust near the ground) -- explained, never
whitewashed. Two wiring-only tests (labelled as such) feed FL-303's
certified nominal verdict into simulate() / the Monte Carlo to pin the merge
and statistics plumbing independently of the physics. A small-N Monte Carlo
must reproduce itself byte-for-byte from a seed across worker layouts, and
every failure in it must carry a reason code.
"""
from __future__ import annotations

import math
import shutil
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from orbitopt.flightlab import checks
from orbitopt.flightlab.contract import (
    BodyTrack,
    Check,
    Event,
    FlightRun,
    Verification,
    load_target,
    read_run,
)
from orbitopt.flightlab.frames import geodetic_to_ecef

FIX = Path(__file__).parent / "fixtures" / "flightlab" / "opentsiolkovsky"
REPO = Path(__file__).resolve().parents[1]
VEHICLE = REPO / "src" / "orbitopt" / "data" / "flightlab" / "vehicles" / "synthetic_two_stage.yaml"
TARGET = REPO / "src" / "orbitopt" / "data" / "flightlab" / "scenarios" / "leo_two_stage.yaml"

PAD = (28.5619, -80.5772, 3.0)  # the scenario launch site
EPOCH_UTC = "2026-11-10T12:00:00Z"
SEP_T_S = 207.0


@pytest.fixture
def stub(monkeypatch, tmp_path):
    path = tmp_path / "otcli"
    shutil.copy(FIX / "stub_cli.py", path)
    path.chmod(0o755)
    monkeypatch.setenv("ORBITOPT_OPENTSIOLKOVSKY", str(path))
    monkeypatch.setenv("OT_STUB_FIXTURE", str(FIX / "synthetic_out"))
    monkeypatch.setenv("OT_STUB_SRC", str(REPO / "src" / "orbitopt"))
    return path


@pytest.fixture
def target():
    return load_target(TARGET)


# ---------------------------------------------------------------------------
# fast: the honest stub verdict, merged into the Gate-B run
# ---------------------------------------------------------------------------

def test_simulate_recovery_gate_fails_honestly_on_the_stub_scenario(stub, target, tmp_path):
    """The packaged ascent separates with every drop of stage-1 propellant
    burnt: the FL-302 floor must report INSUFFICIENT_PROPELLANT_BOOSTBACK
    within seconds, the ballistic debris track must be replaced by a flown
    (ballistic) booster track, and the reserve must read as measured 0 kg --
    a red recovery, fully explained, never a silent or faked one."""
    from orbitopt.flightlab.simulate import simulate

    run = simulate(target, VEHICLE, tmp_path / "run", timeout_s=60, refly=False)
    by_id = {c.id: c for c in run.verification.checks}
    gate = by_id[checks.RECOVERY_GATE_CHECK_ID]
    assert not gate.passed
    assert gate.reason_code == "INSUFFICIENT_PROPELLANT_BOOSTBACK"
    assert run.verification.mission_constraints_passed is False
    bodies = {t.body_id for t in run.tracks}
    assert "booster" in bodies and "stage1_debris" not in bodies
    reserve = by_id["recovery-propellant-reserve"]
    assert not reserve.passed and reserve.value == pytest.approx(0.0, abs=1e-6)
    assert "PROPELLANT_RESERVE_LOW" == reserve.reason_code
    # the merged run round-trips through the contract reader
    assert read_run(tmp_path / "run").verification == run.verification


def test_booster_track_carries_the_ascent_profile_columns(stub, target, tmp_path):
    """The flown booster track carries the geodetic altitude / lat / lon and
    the launch-site downrange the OpenTsiolkovsky ascent tracks do, so the
    profile charts (Compare Runs, replay) draw every body in one frame.
    Without them the booster alone fell back to inertial downrange and
    spherical altitude: its line began ~70 km off stage 1's separation point
    and ended ~8 km below ground."""
    from orbitopt.flightlab.compare import trajectory_profiles
    from orbitopt.flightlab.simulate import simulate

    run = simulate(target, VEHICLE, tmp_path / "run", timeout_s=60, refly=False)
    booster = next(t for t in run.tracks if t.body_id == "booster")
    assert {"altitude_m", "latitude_deg", "longitude_deg", "downrange_m"} <= set(booster.extra_columns)
    profiles = dict(trajectory_profiles(read_run(tmp_path / "run")))
    s1_end, s2_start = profiles["stage1"][-1], profiles["stage2"][0]
    b_start, b_end = profiles["booster"][0], profiles["booster"][-1]
    # separation sits between stage 1's last and stage 2's first sample
    assert s1_end[0] - 2.0 <= b_start[0] <= s2_start[0] + 2.0
    assert abs(b_start[1] - s1_end[1]) < 10.0
    assert b_end[1] == pytest.approx(0.0, abs=0.05)  # the impact is on the ground


def test_evaluate_recovery_gate_is_a_declared_constraint_when_missing(target):
    """A clean-orbit run that never attempted the landing cannot pass a
    recovery-declared mission: the gate lands as failed data-missing, not as
    silence."""
    from orbitopt.flightlab.checks import evaluate

    clean = _clean_orbit_run()
    t = replace(target, constraints=replace(
        target.constraints, max_dynamic_pressure_pa=None, max_specific_acceleration_g=None))
    v = evaluate(clean, t)
    assert not v.mission_constraints_passed
    gate = {c.id: c for c in v.checks}[checks.RECOVERY_GATE_CHECK_ID]
    assert not gate.passed
    assert gate.reason_code == checks.RECOVERY_LANDING_NOT_ASSESSED


def test_checks_module_landing_ids_mirror_fl303():
    """checks.py keeps its constraint-id list stdlib-only by mirroring the
    FL-303 touchdown check ids -- the mirror must not drift."""
    from orbitopt.flightlab.recovery import landing

    assert checks.LANDING_CONSTRAINT_CHECK_IDS == frozenset(landing.TOUCHDOWN_CHECK_IDS)
    assert checks.RECOVERY_GATE_CHECK_ID in checks.CONSTRAINT_CHECK_IDS
    assert checks.LANDING_CONSTRAINT_CHECK_IDS <= checks.CONSTRAINT_CHECK_IDS


def test_dispersions_needs_three_live_classes():
    from orbitopt.flightlab.recovery.montecarlo import Dispersions

    with pytest.raises(ValueError, match="at least 3 active"):
        Dispersions(thrust_pct=1.0, isp_pct=0.0, density_pct=0.0, cd_pct=0.0,
                    sep_altitude_m=0.0, sep_downrange_m=0.0, sep_crossrange_m=0.0,
                    sep_speed_m_s=0.0, sep_heading_deg=0.0)
    # the shipped default covers five classes
    live = sum(1 for v in vars(Dispersions()).values() if v)
    assert live >= 3


def test_dispersed_separation_state_moves_along_enu_axes():
    """alt/down/cross displace the state along local up / along-track /
    right-hand normal by the requested amounts; speed and heading do what
    their names say."""
    from orbitopt.flightlab.recovery import BoosterState, dispersed_separation_state
    from orbitopt.flightlab.recovery.boostback import _enu_basis

    e, n, u = (np.array(c) for c in _enu_basis(PAD[0], PAD[1]))
    r0 = np.array(geodetic_to_ecef(*PAD)) + 40_000.0 * u
    v0 = 250.0 * n
    st = BoosterState(t_s=SEP_T_S, epoch_utc=datetime(2026, 11, 10, 12, 0, 0),
                      frame="ECEF", position_m=tuple(r0), velocity_m_s=tuple(v0),
                      mass_kg=65_000.0)
    d = dispersed_separation_state(st, altitude_m=500.0, downrange_m=-300.0,
                                   speed_m_s=10.0)
    assert float(np.linalg.norm(np.array(d.position_m) - r0)) == pytest.approx(
        math.hypot(500.0, 300.0), abs=1.0)
    assert float(np.dot(np.array(d.position_m) - r0, u)) == pytest.approx(500.0, abs=0.5)
    assert float(np.linalg.norm(d.velocity_m_s)) == pytest.approx(260.0, abs=0.01)
    h = dispersed_separation_state(st, heading_deg=90.0)
    hv = np.array(h.velocity_m_s)
    assert float(np.dot(hv, e)) == pytest.approx(250.0, abs=0.01)  # right of north


# ---------------------------------------------------------------------------
# the hand-built ascent whose separation IS recoverable
# ---------------------------------------------------------------------------

def _ascent_with_textbook_separation() -> FlightRun:
    """Ascent run with the FL-302 textbook separation: stage-1 (30 t dry)
    released 8 km east of the pad at 40 km, 250 m/s, FPA +2 deg, 65 t, and a
    400 km circular GCRS insertion for stage 2 -- everything else
    illustrative but contract-valid."""
    from orbitopt.flightlab.recovery.boostback import _enu_basis

    e, n, u = (np.array(c) for c in _enu_basis(PAD[0], PAD[1]))
    r = np.array(geodetic_to_ecef(*PAD)) + 8_000.0 * e + 40_000.0 * u
    fpa = math.radians(2.0)
    v = 250.0 * (math.cos(fpa) * e + math.sin(fpa) * u)  # heading 90 deg, east
    pad_r = geodetic_to_ecef(*PAD)

    r_orb = checks.EARTH_RADIUS_M + 400_000.0
    v_orb = math.sqrt(checks.EARTH_MU / r_orb)
    inc = math.radians(51.6)
    vel = (0.0, v_orb * math.cos(inc), v_orb * math.sin(inc))

    return FlightRun(
        run_id="fl304-e2e-ascent", target_sha256="0" * 64, vehicle_sha256="0" * 64,
        git_sha="f8f7c77", created_utc=EPOCH_UTC,
        tracks=(
            BodyTrack(body_id="stage1", parent_body_id=None, frame="GCRS",
                      epoch_utc=EPOCH_UTC, provenance="illustrative",
                      t_s=(0.5, SEP_T_S), mass_kg=(380_000.0, 30_000.0),
                      position_m=(pad_r, pad_r), velocity_m_s=((0.0, 0.0, 0.0),) * 2),
            BodyTrack(body_id="stage2", parent_body_id="stage1", frame="GCRS",
                      epoch_utc=EPOCH_UTC, provenance="illustrative",
                      t_s=(620.0, 622.0), mass_kg=(55_000.0, 55_000.0),
                      position_m=((r_orb, 0.0, 0.0),) * 2, velocity_m_s=(vel,) * 2),
            BodyTrack(body_id="stage1_debris", parent_body_id="stage1", frame="ECEF",
                      epoch_utc=EPOCH_UTC, provenance="illustrative",
                      t_s=(SEP_T_S, SEP_T_S + 1.0), mass_kg=(65_000.0, 65_000.0),
                      position_m=(tuple(float(c) for c in r),) * 2,
                      velocity_m_s=(tuple(float(c) for c in v),) * 2),
        ),
        events=(
            Event(t_s=1.0, kind="separation", body_id="stage1_debris",
                  data={"from": "stage1"}),
            Event(t_s=620.0, kind="seco", body_id="stage2", data={}),
        ),
        verification=Verification(checks=(
            Check(id="max-q-within-limit", passed=True, value=36690.0,
                  limit=60_000.0, unit="Pa"),
            Check(id="max-accel-within-limit", passed=True, value=4.1,
                  limit=5.0, unit="g"),
        )),
    )


def _clean_orbit_run() -> FlightRun:
    ascent = _ascent_with_textbook_separation()
    return replace(ascent, tracks=tuple(
        t for t in ascent.tracks if t.body_id != "stage1_debris"),
        events=tuple(e for e in ascent.events if e.body_id != "stage1_debris"))


def _nominal_green_landing():
    """WIRING-ONLY substitute: FL-303's own nominal handoff state (low,
    slow, near: 200 m up, 80 m over, 10 m/s down, 54 t) landed by the real
    ``fly_landing_burn`` on the scenario pad -- the green verdict FL-303's
    acceptance suite certifies -- returned in place of whatever corridor
    state FL-302 hands over. Used only by the tests labelled wiring-only;
    the physical green E2E (FL-305) substitutes nothing."""
    from orbitopt.flightlab.recovery import BoosterModel, BoosterState
    from orbitopt.flightlab.recovery.boostback import LandingBurnEnu, RecoveryTarget, _enu_basis
    from orbitopt.flightlab.recovery.landing import fly_landing_burn

    e, n, u = (np.array(c) for c in _enu_basis(PAD[0], PAD[1]))
    bas = np.vstack([e, n, u])
    pos, vel, m0 = (80.0, 0.0, 200.0), (-10.0, 0.0, -10.0), 54_000.0
    enu = LandingBurnEnu(t_s=SEP_T_S, position_enu_m=pos, velocity_enu_m=vel,
                         mass_kg=m0)
    r = np.array(geodetic_to_ecef(*PAD)) + bas.T @ np.asarray(pos, dtype=float)
    v = bas.T @ np.asarray(vel, dtype=float)
    ecef = BoosterState(t_s=SEP_T_S, epoch_utc=datetime(2026, 11, 10, 12, 0, 0),
                        frame="ECEF", position_m=tuple(float(c) for c in r),
                        velocity_m_s=tuple(float(c) for c in v), mass_kg=m0)
    site = RecoveryTarget(kind="rtls", name="launch_site", latitude_deg=PAD[0],
                          longitude_deg=PAD[1], altitude_m=PAD[2])
    res = fly_landing_burn(enu, ecef, site, BoosterModel.load(VEHICLE),
                           solver="gfold", tof_s=25.0, n=16, glide_slope_deg=20.0)
    assert res.landed and res.reason_codes == (), "FL-303 nominal went red"

    def landing_override(state_enu, *args, **kwargs):
        """Same real verdict, mission clock shifted to where the pipeline
        hands the corridor over (the textbook plan, not the nominal tour,
        sets the handover time; the merged track must stay time-ordered)."""
        dt = float(state_enu.t_s) - SEP_T_S
        track = res.track
        shifted = replace(
            track,
            times_s=track.times_s + dt,
            events=tuple(replace(ev, t_s=ev.t_s + dt) for ev in track.events),
            impact=replace(track.impact, t_s=track.impact.t_s + dt)
            if track.impact is not None else None,
        )
        return replace(res, track=shifted,
                       impact=shifted.impact if shifted.impact is not None else res.impact)

    return landing_override


# ---------------------------------------------------------------------------
# slow: the honest chain, the green wiring, self-reproducing Monte Carlo
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_slow_e2e_nominal_recovery_is_green_through_the_real_chain(monkeypatch, target,
                                                                   tmp_path):
    """FL-305 acceptance: the textbook separation flown by the real
    simulate() -> FL-302 boostback/entry -> FL-303 landing burn under the
    FL-305 closed-loop terminal guidance, nothing substituted. The gate is
    green because FL-303's touchdown checks -- vertical / lateral speed,
    tilt, pad distance, max-g and the propellant reserve -- all pass on the
    propagated track, and the run round-trips through the contract reader."""
    import orbitopt.flightlab.simulate as sim
    from orbitopt.flightlab.recovery.landing import TOUCHDOWN_CHECK_IDS
    from orbitopt.flightlab.recovery.pipeline import NOMINAL_PLAN_GUESS
    from orbitopt.flightlab.simulate import simulate

    pytest.importorskip("gfold")
    ascent = _ascent_with_textbook_separation()
    monkeypatch.setattr(sim, "run_ascent", lambda *a, **k: ascent)

    out = tmp_path / "green"
    run = simulate(target, VEHICLE, out, refly=False,
                   recovery_options={"initial_guess": NOMINAL_PLAN_GUESS,
                                     "max_fevals": 18})
    by_id = {c.id: c for c in run.verification.checks}
    gate = by_id[checks.RECOVERY_GATE_CHECK_ID]
    assert gate.passed, f"gate failed: {gate.reason_code}"
    assert run.verification.mission_constraints_passed is True
    for check_id in TOUCHDOWN_CHECK_IDS:
        assert by_id[check_id].passed, (check_id, by_id[check_id].value)
    assert abs(by_id["landing-vertical-speed"].value) <= 3.0
    assert by_id["landing-position"].value < 25.0
    assert by_id["landing-tilt"].value <= 40.0
    assert by_id["landing-propellant-reserve"].value >= 0.0
    assert by_id["recovery-propellant-reserve"].passed
    touchdown = [e for e in run.events if e.kind == "touchdown"]
    assert len(touchdown) == 1 and touchdown[0].body_id == "booster"
    assert any("FL-305 closed-loop" in a for a in run.assumptions)
    bodies = {t.body_id for t in run.tracks}
    assert "booster" in bodies and "stage1_debris" not in bodies
    assert read_run(out).verification == run.verification


@pytest.mark.slow
def test_slow_e2e_missing_landing_solver_reads_not_assessed(monkeypatch, target,
                                                           tmp_path):
    """The locked conda env has no gfold (pip add-on layer). The physically
    green chain above must then read "landing not assessed" -- a failed
    gate, as every declared-but-unassessed check is -- and never the physics
    verdict LANDING_INFEASIBLE, nor a fake landing-solver check."""
    import sys

    import orbitopt.flightlab.simulate as sim
    from orbitopt.flightlab.recovery.pipeline import NOMINAL_PLAN_GUESS
    from orbitopt.flightlab.simulate import simulate

    monkeypatch.setitem(sys.modules, "gfold", None)
    ascent = _ascent_with_textbook_separation()
    monkeypatch.setattr(sim, "run_ascent", lambda *a, **k: ascent)

    run = simulate(target, VEHICLE, tmp_path / "no-solver", refly=False,
                   recovery_options={"initial_guess": NOMINAL_PLAN_GUESS,
                                     "max_fevals": 18})
    by_id = {c.id: c for c in run.verification.checks}
    gate = by_id[checks.RECOVERY_GATE_CHECK_ID]
    assert not gate.passed
    assert gate.reason_code == checks.RECOVERY_LANDING_NOT_ASSESSED
    assert "landing-solver" not in by_id
    assert all(c.reason_code != "LANDING_INFEASIBLE" for c in run.verification.checks)
    assert run.verification.mission_constraints_passed is False
    assert any("gfold is not installed" in a for a in run.assumptions)
    assert run.engines.get("gfold") == "not-run"


@pytest.mark.slow
def test_slow_e2e_single_plan_landing_is_honest_red_with_a_landing_reason_code(
        monkeypatch, target, tmp_path):
    """The same physical chain with FL-303's single open-loop G-FOLD plan
    (``landing: {mode: single}``): FL-302 feasible, FL-303 flies, the
    re-integration lands hard, and every verdict is explained. The corridor
    handoff (~300 m/s descent at h0) is beyond an open-loop flat-model plan
    under real drag and ambient thrust (FL-303's G3 note), so the mission
    must come back RED with the governing landing reason -- never silent,
    never whitewashed."""
    pytest.importorskip("gfold")  # a real landing burn (the locked env has none: envs/flightlab-landing.txt)
    import orbitopt.flightlab.simulate as sim
    from orbitopt.flightlab.recovery.pipeline import NOMINAL_PLAN_GUESS
    from orbitopt.flightlab.simulate import simulate

    ascent = _ascent_with_textbook_separation()
    monkeypatch.setattr(sim, "run_ascent", lambda *a, **k: ascent)

    out = tmp_path / "red"
    run = simulate(target, VEHICLE, out, refly=False,
                   recovery_options={"initial_guess": NOMINAL_PLAN_GUESS,
                                     "max_fevals": 18,
                                     "landing": {"mode": "single"}})
    by_id = {c.id: c for c in run.verification.checks}
    gate = by_id[checks.RECOVERY_GATE_CHECK_ID]
    assert not gate.passed
    assert gate.reason_code and gate.reason_code.startswith("LANDING")
    assert run.verification.mission_constraints_passed is False
    vertical = by_id["landing-vertical-speed"]
    assert not vertical.passed and vertical.value is not None and vertical.value < -3.0
    # the booster really flew: track plus start and terminal event
    bodies = {t.body_id for t in run.tracks}
    assert "booster" in bodies and "stage1_debris" not in bodies
    kinds = {e.kind for e in run.events}
    assert "landing_start" in kinds and kinds & {"touchdown", "impact"}
    assert read_run(out).verification == run.verification


@pytest.mark.slow
def test_slow_wiring_only_substituted_green_landing_travels_simulate(monkeypatch, target,
                                                                    tmp_path):
    """WIRING-ONLY (not a physics result -- the FL-305 green E2E above is):
    with FL-303's certified nominal verdict substituted for the landing step
    (``landing: {mode: single}`` + a patched ``fly_landing_burn``, see
    _nominal_green_landing), the real simulate() recovery branch must merge
    the booster run, turn the gate green, lift mission_constraints_passed,
    and round-trip through the contract reader."""
    pytest.importorskip("gfold")  # _nominal_green_landing flies FL-303's real nominal burn
    import orbitopt.flightlab.simulate as sim
    from orbitopt.flightlab.recovery import pipeline as pipe
    from orbitopt.flightlab.recovery.pipeline import NOMINAL_PLAN_GUESS
    from orbitopt.flightlab.simulate import simulate

    green = _nominal_green_landing()
    monkeypatch.setattr(pipe, "fly_landing_burn", green)
    ascent = _ascent_with_textbook_separation()
    monkeypatch.setattr(sim, "run_ascent", lambda *a, **k: ascent)

    out = tmp_path / "green"
    run = simulate(target, VEHICLE, out, refly=False,
                   recovery_options={"initial_guess": NOMINAL_PLAN_GUESS,
                                     "max_fevals": 18,
                                     "landing": {"mode": "single"}})
    by_id = {c.id: c for c in run.verification.checks}
    gate = by_id[checks.RECOVERY_GATE_CHECK_ID]
    assert gate.passed, f"gate failed: {gate.reason_code}"
    assert run.verification.mission_constraints_passed is True
    assert by_id["recovery-propellant-reserve"].passed
    touchdown = [e for e in run.events if e.kind == "touchdown"]
    assert len(touchdown) == 1 and touchdown[0].body_id == "booster"
    assert touchdown[0].data["position_error_m"] < 25.0
    bodies = {t.body_id for t in run.tracks}
    assert "booster" in bodies and "stage1_debris" not in bodies
    assert read_run(out).verification == run.verification


@pytest.mark.slow
def test_slow_monte_carlo_is_deterministic_per_seed(tmp_path):
    """Same seed -> same statistics, same worker layout or not: the two
    small-N runs (serial and 2-wide pool) must agree sample-for-sample and
    write byte-identical montecarlo.json; every sample carries an explaining
    reason code."""
    from orbitopt.flightlab.recovery.montecarlo import Dispersions, run_monte_carlo
    from orbitopt.flightlab.recovery.pipeline import NOMINAL_PLAN_GUESS

    target = load_target(TARGET)
    ascent = _ascent_with_textbook_separation()
    disp = Dispersions(thrust_pct=4.0, density_pct=8.0, sep_altitude_m=500.0,
                       sep_downrange_m=1_000.0, sep_speed_m_s=6.0)
    opts = {"initial_guess": NOMINAL_PLAN_GUESS, "max_fevals": 18}
    a = run_monte_carlo(target, VEHICLE, ascent, dispersions=disp, seed=5, n=2,
                        workers=1, plan_options=opts, out_path=tmp_path / "a.json")
    b = run_monte_carlo(target, VEHICLE, ascent, dispersions=disp, seed=5, n=2,
                        workers=2, plan_options=opts, out_path=tmp_path / "b.json")
    assert a.to_dict() == b.to_dict()
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    assert len(a.samples) == 2
    assert sum(a.reason_histogram.values()) == 2
    for s in a.samples:
        assert s["reason"], s
    assert 0.0 <= a.success_rate <= 1.0
    text = (tmp_path / "a.json").read_text(encoding="utf-8")
    assert '"success_rate"' in text and '"ellipse"' in text


@pytest.mark.slow
def test_slow_wiring_only_monte_carlo_green_source_yields_rate_percentiles_and_ellipse(
        monkeypatch, tmp_path):
    """WIRING-ONLY (statistics plumbing, not a physics result -- the FL-305
    before/after Monte Carlo lives in docs/benchmarks/
    fl305_recovery_montecarlo.json): with the FL-303 nominal verdict
    substituted for every landing, the Monte Carlo must report a full
    success rate, the touchdown dispersion ellipse over the landed subset,
    and the metric percentiles."""
    pytest.importorskip("gfold")  # _nominal_green_landing flies FL-303's real nominal burn
    from orbitopt.flightlab.recovery import pipeline as pipe
    from orbitopt.flightlab.recovery.montecarlo import Dispersions, run_monte_carlo
    from orbitopt.flightlab.recovery.pipeline import NOMINAL_PLAN_GUESS

    green = _nominal_green_landing()
    monkeypatch.setattr(pipe, "fly_landing_burn", green)
    target = load_target(TARGET)
    disp = Dispersions(thrust_pct=4.0, density_pct=8.0, sep_altitude_m=500.0,
                       sep_downrange_m=1_000.0, sep_speed_m_s=6.0)
    res = run_monte_carlo(target, VEHICLE, _ascent_with_textbook_separation(),
                          dispersions=disp, seed=9, n=3, workers=1,
                          plan_options={"initial_guess": NOMINAL_PLAN_GUESS,
                                        "max_fevals": 18,
                                        "landing": {"mode": "single"}},
                          out_path=tmp_path / "montecarlo.json")
    assert res.success_rate == 1.0
    assert res.reason_histogram == {"LANDED": 3}
    assert res.ellipse["state"] == "assessed" and res.ellipse["n"] == 3
    stats = res.landing_stats["position_error_m"]
    assert stats["n"] == 3 and 0.0 <= stats["p95"] < 25.0
