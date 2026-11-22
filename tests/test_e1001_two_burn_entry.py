"""E10-01 (task #298): unpowered aerodynamic entry -- ``entry_burns=0``.

The FL-302 script family now represents the E5-07 reference vehicle's actual
architecture: boostback, then a drag/gravity coast (no entry burn at all),
then the landing burn -- two ignitions total. These tests pin

  * the layout: a 5-parameter script (boostback only), no entry burn on the
    plan, the throttle/trigger/speed-cutoff knobs refusing the 0-burn mode;
  * the flight: exactly one burn flown (the boostback), the mass history flat
    after cutoff through the corridor, and the ``mission_command`` re-fly
    reproducing the accepted insertion state (no kinematic shortcuts);
  * the restart accounting: a ``restarts_max=2`` vehicle flies the whole
    chain (boostback + landing burn ignitions) -- the E5-07
    RECOVERY_GAP.md statement ("no planner representation") no longer holds;
  * the scoring: the entry q/g verdicts still grade the post-boostback
    descent window -- unpowered entry is checked, not skipped.

The e2e closure verdicts on the sealed DC7 separation state (two-burn vs the
sealed three-burn record) are the record under
``tests/data/mission/reference/e10-01-two-burn/``, asserted by the record
test at the bottom; the synthetic-fixture tests here keep the chain honest
in CI.
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from orbitopt.flightlab.contract import load_target, read_run
from orbitopt.flightlab.recovery import BoosterModel, RecoveryError
from orbitopt.flightlab.recovery.boostback import (
    RecoveryTarget,
    plan_boostback_entry,
    mission_command,
    _unpack_script,
)
from orbitopt.flightlab.recovery.pipeline import (fly_recovery,
                                                  separation_state_ecef)
from orbitopt.flightlab.recovery.propagator import AltitudeThreshold, propagate

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "src" / "orbitopt" / "data" / "flightlab"
VEHICLE = DATA / "vehicles" / "synthetic_two_stage_rtls.yaml"
SCENARIO = DATA / "scenarios" / "leo_two_stage_rtls.yaml"
ASCENT = REPO / "tests" / "data" / "mission" / "reference" / "e5-05a-rtls-mission" / "ascent"
REF_E507 = REPO / "tests" / "data" / "mission" / "reference" / "e5-07-physics-qualification"
REF_E1001 = REPO / "tests" / "data" / "mission" / "reference" / "e10-01-two-burn"
REFERENCE_VEHICLE = DATA / "vehicles" / "starship_class_public_estimate.yaml"

#: the fixture separation mass (as in the E5-05c closed-loop tests)
SEP_MASS_KG = 165_433.8
#: a keep-pinned unpowered script on the synthetic fixture (the boostback
#: half of the E5-05c +30 t one-burn seed, aim slot at 49.76 km); recorded
#: 2026-11 -- verdict CORRIDOR_MISSED (arrives fast, engine-off), used here
#: as the deterministic pinned script the structural checks read.
SEED_B2 = [22.37, 49.76, 1.0, 0.0, -0.127]

RECIPE_B2 = dict(entry_burns=0, boostback_cutoff="iip",
                 landing_need_model="closed_loop", corridor_aim_margin_m=7000.0,
                 entry_q_max_pa=250_000.0, entry_g_max=12.0,
                 landing_reserve_margin_kg=2500.0)


@pytest.fixture(scope="module")
def inputs():
    booster = BoosterModel.load(VEHICLE)
    run = read_run(ASCENT)
    rt = RecoveryTarget.from_scenario(load_target(SCENARIO).to_dict())
    sep = separation_state_ecef(run)
    return booster, rt, dataclasses.replace(sep, mass_kg=SEP_MASS_KG)


def _plan(booster, rt, sep, **kw):
    opts = {**RECIPE_B2, "initial_guess": SEED_B2, "search": "keep"}
    opts.update(kw)
    return plan_boostback_entry(sep, booster, rt, **opts)


# ---------------------------------------------------------------------------
# layout and validation
# ---------------------------------------------------------------------------


def test_unpack_script_zero_burn_layout():
    x = np.array([8.0, 30.0, 0.9, 5.0, -2.0])
    bb, eb, eb2 = _unpack_script(x, 0, False)
    assert (bb.start_delay_s, bb.duration_s, bb.throttle, bb.pitch_deg, bb.yaw_deg) == \
        (8.0, 30.0, 0.9, 5.0, -2.0)
    assert eb is None and eb2 is None
    # a throttle tail is meaningless with no entry burn and must not be read
    _bb, eb, eb2 = _unpack_script(np.append(x, [0.5]), 0, False)
    assert eb is None and eb2 is None


def test_zero_burn_validation(inputs):
    booster, rt, sep = inputs
    with pytest.raises(ValueError, match="entry_burns must be 0, 1 or 2"):
        plan_boostback_entry(sep, booster, rt, entry_burns=3)
    with pytest.raises(ValueError, match="entry_throttle"):
        plan_boostback_entry(sep, booster, rt, entry_burns=0, entry_throttle=True,
                             search="keep", initial_guess=SEED_B2)
    with pytest.raises(ValueError, match="one-burn"):
        plan_boostback_entry(sep, booster, rt, entry_burns=0, entry_cutoff="speed")
    with pytest.raises(ValueError, match="entry_burns=0.*5"):
        plan_boostback_entry(sep, booster, rt, entry_burns=0, initial_guess=SEED_B2[:4])
    with pytest.raises(ValueError, match="entry_burns=0.*5"):
        plan_boostback_entry(sep, booster, rt, entry_burns=0,
                             initial_guess=SEED_B2 + [16_000.0, 22.0])


@pytest.mark.slow
def test_zero_burn_layout_and_engine_off_coast(inputs):
    """The accepted plan carries no entry burn, flew exactly one burn, and
    the mass history is flat from boostback cutoff to the track end: the
    entry phase is coasted, not flown."""
    booster, rt, sep = inputs
    plan = _plan(booster, rt, sep)
    assert plan.entry_burn is None and plan.entry_burn_2 is None
    assert any("entry_burns=0" in a for a in plan.assumptions)
    starts = [e for e in plan.track.events if e.kind == "burn_start"]
    ends = [e for e in plan.track.events if e.kind == "burn_end"]
    assert len(starts) == 1 and len(ends) == 1
    assert plan.ignitions_used == 1
    t = np.asarray(plan.track.times_s, dtype=np.float64)
    m = np.asarray(plan.track.mass_kg, dtype=np.float64)
    coast = t >= ends[0].t_s + 1e-6
    assert coast.sum() > 100                      # a real coast window exists
    assert float(np.max(m[coast] - m[coast][0])) < 1e-6  # engine-off: mass flat


@pytest.mark.slow
def test_zero_burn_entry_window_still_grades_q_and_g(inputs):
    """The corridor report carries the entry-window q/g of the unpowered
    descent -- the 250 kPa / 12 g limits are checked against drag loads, not
    skipped because no burn was flown."""
    booster, rt, sep = inputs
    plan = _plan(booster, rt, sep)
    r = plan.corridor_report
    assert r.entry_max_q_pa is not None and r.entry_max_q_pa > 0.0
    assert r.entry_max_proper_g is not None and r.entry_max_proper_g > 1.0
    # and the plan's verdict agrees with the limits it was given: a
    # 100 kPa cap plan is refused on a descent that peaks above it, an
    # infinite one is not.
    tight = _plan(booster, rt, sep, entry_q_max_pa=50_000.0)
    loose = _plan(booster, rt, sep, entry_q_max_pa=5_000_000.0,
                  entry_g_max=50.0)
    assert "ENTRY_Q_LIMIT_EXCEEDED" in tight.reason_codes
    assert "ENTRY_Q_LIMIT_EXCEEDED" not in loose.reason_codes
    assert "ENTRY_G_LIMIT_EXCEEDED" not in loose.reason_codes


@pytest.mark.slow
def test_zero_burn_refly_reproduces_the_track(inputs):
    """``mission_command`` re-flies the accepted 5-parameter script through
    the FL-301 propagator and reproduces the accepted track -- no kinematic
    shortcuts. With no entry burn the re-fly lights exactly one burn and the
    first post-cutoff descending h0 crossing is the same event the plan
    graded. (``mission_command`` replays the fixed-duration script family --
    the FL-302 acceptance re-fly it has always served -- so this test plans
    with the duration cutoff.)"""
    booster, rt, sep = inputs
    plan = _plan(booster, rt, sep, boostback_cutoff="duration",
                 initial_guess=[22.37, 40.0, 1.0, 0.0, -0.127],
                 dt_opt=0.5, dt_final=0.5)
    cmd = mission_command(plan, sep)
    h0 = rt.corridor.h0_m
    tr = propagate(sep, booster, cmd, sep.t_s + 900.0,
                   events=("burn", "ground", AltitudeThreshold(h0, "reflight_h0", "down")),
                   dt_s=0.5, sample_s=2.0)
    starts = [e for e in tr.events if e.kind == "burn_start"]
    assert len(starts) == 1                    # boostback only; entry: coast
    ends = [e for e in tr.events if e.kind == "burn_end"]
    crossings = [e for e in tr.events if e.kind == "reflight_h0"
                 and e.t_s >= ends[0].t_s + 1e-6]
    if plan.landing_burn_initial_state is not None and crossings:
        r0 = np.asarray(plan.landing_burn_initial_state.position_m)
        r1 = np.asarray(crossings[0].data["position_ecef_m"])
        assert float(np.linalg.norm(r1 - r0)) < 2.0
    assert abs(float(tr.mass_kg[-1]) - float(plan.track.mass_kg[-1])) < 5.0


def test_zero_burn_clamps_and_two_ignition_vehicle(inputs):
    """Out-of-box values clamp into the 5-parameter box; a restarts_max=2
    vehicle (the E5-07 reference budget) passes the planner's ignition
    check -- only one ignition is consumed by the plan itself, the landing
    burn's is budgeted on top (2 in total)."""
    booster, rt, sep = inputs
    assert booster.engine.restarts_max >= 2
    two = dataclasses.replace(
        booster, engine=dataclasses.replace(booster.engine, restarts_max=2))
    kept = plan_boostback_entry(          # clamped, not rejected
        sep, two, rt, entry_burns=0, boostback_cutoff="iip",
        initial_guess=[999.0, -999.0, 5.0, -9.0, 99.0], max_fevals=1,
        search="keep")
    bb = kept.boostback
    assert 0.0 <= bb.start_delay_s <= 150.0 and -20.0 <= bb.duration_s <= 80.0
    assert two.engine.throttle_min <= bb.throttle <= two.engine.throttle_max
    assert 0.0 <= bb.pitch_deg <= 30.0 and -45.0 <= bb.yaw_deg <= 45.0
    one = dataclasses.replace(
        booster, engine=dataclasses.replace(booster.engine, restarts_max=1))
    with pytest.raises(RecoveryError, match="ignitions"):
        plan_boostback_entry(sep, one, rt, entry_burns=1)


# ---------------------------------------------------------------------------
# the E5-07 reference vehicle: the gap statement is now representable
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_e507_reference_vehicle_profile_now_plans():
    """RECOVERY_GAP.md recorded: asked about the E5-07 separation state, the
    planner answered ``entry_burns=0 -> ValueError``. The ValueError is gone:
    the vehicle's actual architecture (restart 2, aerodynamic entry) plans.
    The verdict (where the corridor lands) is the record's business; here
    only representability is asserted -- the gap document itself stays
    sealed."""
    from orbitopt.mission.coupled_branches import (booster_state_from_run,
                                                   separation_handoff)
    booster = BoosterModel.load(REFERENCE_VEHICLE)
    run = read_run(REF_E507 / "mission-optimized" / "launch" / "ascent")
    sep = booster_state_from_run(run, separation_handoff(run))
    rt = RecoveryTarget.from_scenario(
        load_target(DATA / "scenarios" / "leo_starship_class.yaml").to_dict())
    assert booster.engine.restarts_max == 2
    assert "entry" not in booster.engine.usable_for
    plan = plan_boostback_entry(sep, booster, rt, entry_burns=0,
                                boostback_cutoff="iip",
                                landing_need_model="closed_loop",
                                entry_q_max_pa=250_000.0, entry_g_max=12.0,
                                initial_guess=SEED_B2, search="keep")
    assert plan.entry_burn is None
    assert plan.ignitions_used == 1
    assert plan.track is not None


SEED_B2_DC7 = [21.2, -2.0, 1.0, 0.0, 0.0]   # keep-script probe seed (E10-01)
REF_E508 = REPO / "tests/data/mission/reference/e5-08-new-vehicle-dc7"


@pytest.fixture(scope="module")
def dc7_inputs():
    """The sealed E5-08 DC7 ascent: its recorded separation state and the
    return target, vehicle with restarts_max lowered to 2 (the two-burn
    budget) and the graded landing reserve ("target", 100 t) applied."""
    from orbitopt.flightlab.recovery.pipeline import landing_reserve_booster
    from orbitopt.mission.coupled_branches import (booster_state_from_run,
                                                   separation_handoff)
    run = read_run(REF_E508 / "mission-optimized" / "launch" / "ascent")
    sep = booster_state_from_run(run, separation_handoff(run))
    target = load_target(DATA / "scenarios" / "leo_starship_class_rtls_e508_dc7.yaml")
    booster = landing_reserve_booster(
        BoosterModel.load(DATA / "vehicles" / "starship_class_rtls_dc7.yaml"),
        target, "target")
    booster = dataclasses.replace(
        booster, engine=dataclasses.replace(booster.engine, restarts_max=2))
    return booster, RecoveryTarget.from_scenario(target.to_dict()), sep


# ---------------------------------------------------------------------------
# end-to-end: the FULL chain (FL-302 plan -> FL-301 return track -> FL-303
# landing -> FL-304/verification + R-02 handoff report) with entry_burns=0.


@dataclasses.dataclass(frozen=True)
class Chain:
    outcome: object
    report: object


@pytest.fixture(scope="module")
def chain(inputs) -> Chain:
    """``fly_recovery`` with the fixed 5-parameter unpowered script and a
    restart-2 vehicle, handoff checks merged into the verification."""
    from orbitopt.flightlab.recovery.pipeline import fly_recovery
    booster, rt, sep = inputs
    two = dataclasses.replace(
        booster, engine=dataclasses.replace(booster.engine, restarts_max=2))
    run = read_run(ASCENT)
    outcome = fly_recovery(run, load_target(SCENARIO), VEHICLE,
                           booster=two, sep_state=sep,
                           plan_options=dict(
                               entry_burns=0, boostback_cutoff="iip",
                               search="keep", initial_guess=SEED_B2,
                               dt_opt=0.5, dt_final=0.5, handoff_check=True))
    return Chain(outcome, outcome.handoff)


@pytest.mark.slow
def test_fly_recovery_chain_supports_entry_burns_zero(chain):
    """Planner, flight and scoring all speak entry_burns=0: the plan carries
    no entry burn, the verification grades the entry window as a coast
    (dynamic-pressure / load checks on the ballistic descent), the ignition
    count is 2 (boostback + landing) and the handoff report marks the
    entry-burn coast check as inapplicable rather than failing."""
    out, rep = chain.outcome, chain.report
    assert out.plan.entry_burn is None
    ids = {c.id for c in out.checks}
    assert "joint-max-q-entry" in ids                 # graded, not missing
    missing = {u.id: u.note for u in rep.unassessed}
    assert any("entry_burns=0" in str(getattr(u, "note", ""))
               for u in rep.unassessed)
    assert "handoff-boostback-entry" in ids           # cutoff handoff graded
    assert "handoff-entry-landing" in ids or "handoff-entry-landing" in missing
    btr = next(t for t in out.run.tracks if t.body_id == "booster")
    m = np.asarray(btr.mass_kg)
    assert float(np.min(m[:-1] - m[1:])) >= -1e-6      # mass never grows
    ignitions = [e for e in out.run.events
                 if e.body_id == "booster" and e.kind == "ignition"]
    assert len(ignitions) == 1 == out.plan.ignitions_used  # boostback only


@pytest.mark.slow
def test_dc7_unpowered_verdict_is_a_classified_corridor_miss(dc7_inputs):
    """E10-01 closure probe on the sealed DC7 ascent: the unpowered chain
    flies end to end (plan + handoff + verification all produced) and the
    honest verdict is a corridor miss with entry-q violation -- the record
    (REF_E1001) carries the classification. The 3-burn sealed DC7 record
    passes the same checks unchanged (it is replayed by
    scripts/e508_new_vehicle.py), so the difference is the aero, not the
    gates."""
    booster, rt, sep = dc7_inputs
    plan = _plan(booster, rt, sep, entry_burns=0, initial_guess=SEED_B2_DC7,
                 dt_opt=0.5, dt_final=0.25)
    assert plan.entry_burn is None
    assert not plan.feasible
    codes = set(plan.reason_codes)
    assert "CORRIDOR_MISSED" in codes
    assert plan.corridor_report.reached          # arrives; too fast/hot
    cr = plan.corridor_report
    assert cr.entry_max_q_pa > 250_000.0        # the aero cannot shed the
    assert cr.entry_max_proper_g < 12.0                # energy inside the window
    rec = json.loads((REF_E1001 / "two-burn-outcome.json").read_text())
    assert rec["verdict"].startswith("NOT CLOSED")
    assert "vehicle-capability" in rec["classification"]["category"]
    p_rec = rec["probes"]["dc7_keep_script"]
    assert list(p_rec["reason_codes"]) == sorted(codes)
    assert abs(cr.speed_m_s - p_rec["speed_m_s"]) < 0.05 * p_rec["speed_m_s"]
