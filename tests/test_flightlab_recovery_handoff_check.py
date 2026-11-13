"""R-02 (task #234): the recovery handoff / joint-constraint report.

The fast tests fly a *dry* booster (separation mass just above dry) through
FL-302's floor path -- a real, fully ballistic FL-301 rollout, no optimizer
-- and grade it with ``build_handoff_report``: every handoff boundary the
chain never reached must be *reported unassessed*, never silently green;
the ballistic touchdown must be graded and named as belonging to the landing
phase; and injecting a mass bump / an IC jump into the flown track must
light the continuity and mass-conservation checks with the two new reason
codes and the right phase attribution. The ``simulate`` plumbing test pins
that merging the report into the run is opt-in (Gate B reference runs stay
byte-identical without the flag).

The @slow tests fly the real textbook-RTLS chain closed-loop (FL-305, gfold)
and demand a *green* report with all four boundary states present -- and a
deliberately tightened joint q budget must produce ``joint-max-q-entry``
attributed to phase "entry" in the report.
"""
from __future__ import annotations

import math
import shutil
from dataclasses import replace
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from orbitopt.flightlab.contract import load_target, read_run
from orbitopt.flightlab.frames import geodetic_to_ecef
from orbitopt.flightlab.handoff import (
    PHASE_BOOSTBACK, PHASE_ENTRY, PHASE_LANDING, PHASE_SEPARATION, PHASE_TOUCHDOWN,
)
from orbitopt.flightlab.recovery import BoosterModel, BoosterState
from orbitopt.flightlab.recovery.boostback import RecoveryTarget, plan_boostback_entry
from orbitopt.flightlab.recovery.handoff_check import (
    HANDOFF_MASS_NOT_CONSERVED,
    HANDOFF_STATE_DISCONTINUITY,
    JointConstraints,
    build_handoff_report,
    recovery_handoff_states,
)

FIX = Path(__file__).parent / "fixtures" / "flightlab" / "opentsiolkovsky"
REPO = Path(__file__).resolve().parents[1]
VEHICLE = REPO / "src" / "orbitopt" / "data" / "flightlab" / "vehicles" / "synthetic_two_stage.yaml"
TARGET = REPO / "src" / "orbitopt" / "data" / "flightlab" / "scenarios" / "leo_two_stage.yaml"

PAD = (28.5619, -80.5772, 3.0)
EPOCH = datetime(2026, 11, 10, 12, 0, 0)
SEP_T_S = 207.0


def _pad_basis():
    from orbitopt.flightlab.recovery.boostback import _enu_basis
    return tuple(np.array(c) for c in _enu_basis(PAD[0], PAD[1]))


@pytest.fixture
def stub(monkeypatch, tmp_path):
    path = tmp_path / "otcli"
    shutil.copy(FIX / "stub_cli.py", path)
    path.chmod(0o755)
    monkeypatch.setenv("ORBITOPT_OPENTSIOLKOVSKY", str(path))
    monkeypatch.setenv("OT_STUB_FIXTURE", str(FIX / "synthetic_out"))
    monkeypatch.setenv("OT_STUB_SRC", str(REPO / "src" / "orbitopt"))
    return path


# ---------------------------------------------------------------------------
# the dry ballistic chain: a real FL-301 rollout with nothing to burn
# ---------------------------------------------------------------------------


def _dry_plan():
    booster = BoosterModel.load(VEHICLE)
    e, n, u = _pad_basis()
    r0 = np.array(geodetic_to_ecef(*PAD)) + 8_000.0 * u + 2_000.0 * e
    st = BoosterState(t_s=SEP_T_S, epoch_utc=EPOCH, frame="ECEF",
                      position_m=tuple(float(c) for c in r0),
                      velocity_m_s=(250.0 * 0.7, 0.0, 10.0),
                      mass_kg=booster.dry_mass_kg + 100.0)
    site = RecoveryTarget.from_scenario(load_target(TARGET).to_dict())
    plan = plan_boostback_entry(st, booster, site)
    assert not plan.feasible
    assert plan.landing_burn_initial_state is None
    return st, booster, site, plan


@lru_cache(maxsize=1)
def _dry_report():
    st, booster, site, plan = _dry_plan()
    rep = build_handoff_report(st, plan, None, booster, recovery_target=site)
    assert rep.to_dict()  # serialisable, schema-shaped
    return rep


def test_dry_ballistic_chain_is_graded_not_whitewashed():
    rep = _dry_report()
    by_id = {c.id: c for c in rep.checks}
    ids = set(by_id)
    ua_ids = {u.id for u in rep.unassessed}
    # boundaries the chain never reached: reported, never graded green
    assert "handoff-boostback-entry" in ua_ids
    assert "handoff-entry-landing" in ua_ids
    assert "joint-propellant-corridor-budget" in ua_ids
    assert not (ids & ua_ids)
    # what DID fly is graded
    assert by_id["handoff-separation-boostback"].passed
    assert by_id["handoff-landing-touchdown"].passed  # impact record == track end
    assert by_id["handoff-mass-monotonic-boostback"].passed
    # 100 kg of propellant is not a landing reserve: honest red, landing phase
    reserve = by_id["joint-propellant-reserve-touchdown"]
    assert not reserve.passed and reserve.reason_code == "PROPELLANT_RESERVE_LOW"
    assert rep.phase_of(reserve.id) == PHASE_LANDING
    # a ballistic impact 100+ km downrange is not a landing
    assert by_id["joint-touchdown-position"].reason_code == "LANDING_POSITION_OUT_OF_TOLERANCE"
    assert rep.phase_of("joint-touchdown-position") == PHASE_LANDING
    assert not rep.passed
    note = rep.assumption_note()
    assert note.startswith("R-02 handoff report")
    assert "landing=LANDING_POSITION_OUT_OF_TOLERANCE" in note


def test_dry_chain_states_and_digests():
    rep = _dry_report()
    boundaries = [s.boundary for s in rep.states]
    assert (PHASE_SEPARATION, PHASE_BOOSTBACK) in boundaries
    assert (PHASE_LANDING, PHASE_TOUCHDOWN) in boundaries  # it did hit the ground
    assert (PHASE_ENTRY, PHASE_LANDING) not in boundaries  # corridor never crossed
    doc = rep.to_dict()
    assert doc["kind"] == "orbitopt.flightlab.recovery-handoff-report"
    assert {s["from_phase"] for s in doc["states"]} <= {"separation", "boostback",
                                                        "entry", "landing"}
    for c in doc["checks"]:
        assert c["phase"] in ("boostback", "entry", "landing")
        assert "reason_code" in c and "value" in c


def test_unassessed_entries_carry_a_phase_and_a_reason():
    rep = _dry_report()
    for u in rep.unassessed:
        assert u.note
        assert u.phase in ("boostback", "entry", "landing")
    assert {u.id: u.phase for u in rep.unassessed}["handoff-entry-landing"] == PHASE_LANDING


@pytest.mark.parametrize("bump_field,delta,check_id,phase", [
    ("mass", 5.0, "handoff-mass-monotonic-boostback", PHASE_BOOSTBACK),
    ("pos", 6_000.0, "handoff-separation-boostback", PHASE_BOOSTBACK),
])
def test_injected_track_fault_lights_the_named_check(bump_field, delta, check_id, phase):
    """The checks must be sensitive, not decorative: nudge the flown track
    (a consumer that re-derived its IC, a propagator that invented 5 kg)
    and the one check that owns it fails with its reason code."""
    st, booster, site, plan = _dry_plan()
    tr = plan.track
    if bump_field == "mass":
        mass = np.array(tr.mass_kg, dtype=np.float64)
        mass[len(mass) // 2] += delta
        bumped = replace(tr, mass_kg=mass)
    else:
        pos = np.array(tr.position_ecef_m, dtype=np.float64)
        pos[0] = pos[0] + np.array([0.0, 0.0, delta])
        bumped = replace(tr, position_ecef_m=pos)
    rep = build_handoff_report(st, replace(plan, track=bumped), None, booster,
                               recovery_target=site)
    by_id = {c.id: c for c in rep.checks}
    fired = by_id[check_id]
    assert not fired.passed
    expect = HANDOFF_MASS_NOT_CONSERVED if bump_field == "mass" \
        else HANDOFF_STATE_DISCONTINUITY
    assert fired.reason_code == expect
    assert rep.phase_of(check_id) == phase
    assert f"{phase}={expect}" in rep.assumption_note()


def test_joint_constraints_defaults_track_the_declared_limits():
    cons = JointConstraints.for_recovery()
    assert cons.touchdown_position_tol_m == 25.0
    assert cons.max_dynamic_pressure_pa == 150_000.0
    with pytest.raises(ValueError):
        JointConstraints(max_dynamic_pressure_pa=0.0)
    with pytest.raises(ValueError):
        JointConstraints(min_propellant_reserve_kg=-1.0)


# ---------------------------------------------------------------------------
# simulate plumbing: the merge is opt-in
# ---------------------------------------------------------------------------


def test_handoff_checks_merge_into_the_run_only_on_request(stub, tmp_path):
    from orbitopt.flightlab.simulate import simulate

    target = load_target(TARGET)
    quiet = simulate(target, VEHICLE, tmp_path / "quiet", timeout_s=60, refly=False)
    ids = {c.id for c in quiet.verification.checks}
    assert not any(i.startswith(("handoff-", "joint-max", "joint-touchdown",
                                 "joint-propellant")) for i in ids)
    assert not any(a.startswith("R-02") for a in quiet.assumptions)

    loud = simulate(target, VEHICLE, tmp_path / "loud", timeout_s=60, refly=False,
                    recovery_options={"handoff_check": True})
    loud_ids = {c.id for c in loud.verification.checks}
    assert "handoff-separation-boostback" in loud_ids
    assert "joint-propellant-reserve-touchdown" in loud_ids
    assert any(a.startswith("R-02 handoff report") for a in loud.assumptions)
    # still a valid, readable contract run with the extra checks in
    assert read_run(tmp_path / "loud").verification == loud.verification


# ---------------------------------------------------------------------------
# slow: the green closed-loop chain, and a tightened budget naming its phase
# ---------------------------------------------------------------------------


def _ascent_with_textbook_separation():
    from orbitopt.flightlab.contract import BodyTrack, Check, Event, FlightRun, Verification
    e, n, u = _pad_basis()
    r = np.array(geodetic_to_ecef(*PAD)) + 8_000.0 * e + 40_000.0 * u
    fpa = math.radians(2.0)
    v = 250.0 * (math.cos(fpa) * e + math.sin(fpa) * u)
    pad_r = geodetic_to_ecef(*PAD)
    from orbitopt.flightlab import checks
    r_orb = checks.EARTH_RADIUS_M + 400_000.0
    v_orb = math.sqrt(checks.EARTH_MU / r_orb)
    inc = math.radians(51.6)
    vel = (0.0, v_orb * math.cos(inc), v_orb * math.sin(inc))
    return FlightRun(
        run_id="r02-e2e-ascent", target_sha256="0" * 64, vehicle_sha256="0" * 64,
        git_sha="f8f7c77", created_utc="2026-11-10T12:00:00Z",
        tracks=(
            BodyTrack(body_id="stage1", parent_body_id=None, frame="GCRS",
                      epoch_utc="2026-11-10T12:00:00Z", provenance="illustrative",
                      t_s=(0.5, SEP_T_S), mass_kg=(380_000.0, 30_000.0),
                      position_m=(pad_r, pad_r), velocity_m_s=((0.0, 0.0, 0.0),) * 2),
            BodyTrack(body_id="stage2", parent_body_id="stage1", frame="GCRS",
                      epoch_utc="2026-11-10T12:00:00Z", provenance="illustrative",
                      t_s=(620.0, 622.0), mass_kg=(55_000.0, 55_000.0),
                      position_m=((r_orb, 0.0, 0.0),) * 2, velocity_m_s=(vel,) * 2),
            BodyTrack(body_id="stage1_debris", parent_body_id="stage1", frame="ECEF",
                      epoch_utc="2026-11-10T12:00:00Z", provenance="illustrative",
                      t_s=(SEP_T_S, SEP_T_S + 1.0), mass_kg=(65_000.0, 65_000.0),
                      position_m=(tuple(float(c) for c in r),) * 2,
                      velocity_m_s=(tuple(float(c) for c in v),) * 2),
        ),
        events=(Event(t_s=1.0, kind="separation", body_id="stage1_debris",
                      data={"from": "stage1"}),
                Event(t_s=620.0, kind="seco", body_id="stage2", data={})),
        verification=Verification(checks=()),
    )


@lru_cache(maxsize=1)
def _green_outcome():
    from orbitopt.flightlab.recovery.pipeline import NOMINAL_PLAN_GUESS, fly_recovery
    outcome = fly_recovery(
        _ascent_with_textbook_separation(), load_target(TARGET), VEHICLE,
        plan_options={"initial_guess": NOMINAL_PLAN_GUESS, "max_fevals": 18,
                      "handoff_check": True})
    assert outcome.landing is not None
    return outcome


@pytest.mark.slow
def test_slow_green_closed_loop_chain_gets_a_green_report():
    pytest.importorskip("gfold")
    outcome = _green_outcome()
    assert outcome.success, f"chain red: {outcome.reason_codes}"
    rep = outcome.handoff
    assert rep is not None and rep.passed
    assert [s.boundary for s in rep.states] == [
        (PHASE_SEPARATION, PHASE_BOOSTBACK), (PHASE_BOOSTBACK, PHASE_ENTRY),
        (PHASE_ENTRY, PHASE_LANDING), (PHASE_LANDING, PHASE_TOUCHDOWN)]
    merged = {c.id for c in outcome.run.verification.checks}
    assert {c.id for c in rep.checks} <= merged
    assert any(a.startswith("R-02 handoff report") for a in outcome.run.assumptions)
    # the corridor handoff state is FL-302's own landing-burn IC
    corridor = next(s for s in rep.states if s.boundary == (PHASE_ENTRY, PHASE_LANDING))
    ic = outcome.plan.landing_burn_initial_state
    assert corridor.t_s == pytest.approx(ic.t_s)
    assert corridor.mass_kg == pytest.approx(ic.mass_kg, abs=1e-6)


@pytest.mark.slow
def test_slow_tightened_joint_budget_names_the_entry_phase():
    pytest.importorskip("gfold")
    outcome = _green_outcome()
    sep = outcome.handoff.states[0]
    booster = BoosterModel.load(VEHICLE)
    site = RecoveryTarget.from_scenario(load_target(TARGET).to_dict())
    tight = JointConstraints(max_dynamic_pressure_pa=25_000.0)
    rep = build_handoff_report(sep, outcome.plan, outcome.landing, booster,
                               recovery_target=site, constraints=tight)
    by_id = {c.id: c for c in rep.checks}
    q_entry = by_id["joint-max-q-entry"]
    assert not q_entry.passed
    assert q_entry.reason_code == "ENTRY_Q_LIMIT_EXCEEDED"
    assert rep.phase_of("joint-max-q-entry") == PHASE_ENTRY
    assert by_id["joint-max-q-boostback"].passed  # the same budget leaves the
    # boostback window green: the violation is the entry phase's, not a global one
    assert rep.phase_of("joint-max-q-boostback") == PHASE_BOOSTBACK
    assert "entry=ENTRY_Q_LIMIT_EXCEEDED" in rep.assumption_note()
    doc = rep.to_dict()
    entry_row = next(c for c in doc["checks"] if c["id"] == "joint-max-q-entry")
    assert entry_row["phase"] == "entry"
    assert entry_row["passed"] is False
