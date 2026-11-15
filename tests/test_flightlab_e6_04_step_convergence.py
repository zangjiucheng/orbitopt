"""E6-04 (#276): the sampled-data control loop and the RK4 plant stay on
their own grids.

The audit's three findings each get a lock:

* the control law is evaluated ONLY at integration step boundaries (never at
  an RK4 mid-stage state) and one sampled command is HELD across the four
  stages -- a spy law records the evaluation clock;
* with an unaligned ``dt_c_s`` every tick lands on an integration boundary
  and every tick period lies in ``[dt_c_s, dt_c_s + dt_s)`` -- the tick snaps
  up, never down, and no mid-stage estimate advances the tracker state; a
  ``dt_c_s`` below the plant step warns instead of silently decimating;
* a stateful guidance re-read at a time it has already answered returns the
  HELD command (the command-slew state is not re-seeded from the raw
  pointing) -- the defect the audit found and fixed.

Convergence itself is the benchmark's deliverable
(``benchmarks/bench_e6_04_step_convergence.py``): the committed JSON is
checked here for 4th-order integration convergence at a fixed control tick,
1st-order control convergence at a fixed plant step, and a default-step
error small enough to justify shipping (0.02 s / 0.02 s).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from orbitopt.flightlab.guidance.upper_stage import (
    ParkingOrbitTarget, UpperStageGuidanceSpec,
)
from orbitopt.flightlab.sixdof.aero import load_packaged_aero
from orbitopt.flightlab.sixdof.gravity_turn import (
    WaypointAttitudeProgram, propagate_point_mass,
)
from orbitopt.flightlab.sixdof.rigidbody import (
    ControlCommand, RigidBodyEngine, RigidBodyState, RigidBodyVehicle, propagate,
)
from orbitopt.flightlab.sixdof.tvc import AttitudeTvcLaw, run_closed_loop_ascent

REPO = Path(__file__).resolve().parents[1]
BENCH_JSON = REPO / "docs" / "benchmarks" / "e604_step_convergence.json"

pytestmark = pytest.mark.slow  # real RK4 closed-loop propagations


def vehicle() -> RigidBodyVehicle:
    engine = RigidBodyEngine(thrust_vacuum_n=1.88e6, isp_vacuum_s=320.0,
                             isp_sea_level_s=282.0, nozzle_exit_area_m2=2.0,
                             throttle_min=0.4, throttle_max=1.0, mount_x_cg_m=22.0)
    return RigidBodyVehicle(name="spike_stage", dry_mass_kg=10_000.0,
                            propellant_mass_kg=60_000.0,
                            inertia_kg_m2=(3.4e4, 1.7e6, 1.7e6), engine=engine,
                            aero=load_packaged_aero("spike_stage"))


def pad_state(**kw) -> RigidBodyState:
    kw.setdefault("lat_geod_deg", 28.5)
    kw.setdefault("lon_deg", -80.6)
    kw.setdefault("mass_kg", 70_000.0)
    return RigidBodyState.from_launch(**kw)


# ---------------------------------------------------------------------------
# the law never sees a mid-stage evaluation
# ---------------------------------------------------------------------------

class _ClockSpy:
    """A ControlLaw that stamps every evaluation with the clock and a fresh
    deflection -- any two evaluations sharing a returned command must share
    the sampled state, and every clock reading must be a step boundary."""

    def __init__(self) -> None:
        self.calls: list[tuple[float, float]] = []  # (t_s, gimbal_pitch)
        self._n = 0

    def __call__(self, t_s: float, state) -> ControlCommand:
        self._n += 1
        defl = 1e-4 * self._n
        self.calls.append((float(t_s), defl))
        return ControlCommand(throttle=1.0, gimbal_pitch_rad=defl,
                              gimbal_yaw_rad=0.0)


def test_law_evaluates_only_at_step_boundaries():
    dt, t_end = 0.02, 0.4
    spy = _ClockSpy()
    propagate(pad_state(pitch_deg=89.5), vehicle(), spy, t_end, dt_s=dt)
    n_steps = round(t_end / dt)
    # initial sample + one step-start sample per step + one sample per
    # boundary after the first -- and NONE in between.
    assert len(spy.calls) == 2 * n_steps + 1
    for t, _ in spy.calls:
        k = t / dt
        assert abs(k - round(k)) < 1e-9, f"mid-stage evaluation at t={t}"


def test_ticks_land_on_boundaries_with_an_unaligned_tick():
    law = AttitudeTvcLaw(vehicle=vehicle(),
                         program=WaypointAttitudeProgram.vertical(), dt_c_s=0.03)
    propagate(pad_state(pitch_deg=89.5), vehicle(), law, 1.0, dt_s=0.01)
    h = law.history()
    t_ticks = np.asarray(h["t_s"])[1:]
    dt_ticks = np.asarray(h["dt_s"])[1:]
    assert t_ticks.size > 25  # the tick ran
    grid = t_ticks / 0.01 - np.round(t_ticks / 0.01)
    assert np.max(np.abs(grid)) * 0.01 < 1e-9  # never fired at a stage time
    assert dt_ticks.min() >= 0.03 - 1e-9  # snaps UP to the grid, never early
    assert dt_ticks.max() < 0.03 + 0.01 - 1e-9  # ... and never late by >= dt_s


def test_control_rate_below_the_plant_step_warns():
    prog = WaypointAttitudeProgram.vertical()
    with pytest.warns(UserWarning, match="below the integration step"):
        run_closed_loop_ascent(pad_state(pitch_deg=89.5), vehicle(), prog, 0.1,
                               dt_s=0.05, dt_c_s=0.02)


# ---------------------------------------------------------------------------
# a stateful guidance answers a re-read with the held command
# ---------------------------------------------------------------------------

class _GuidanceSpy:
    def __init__(self) -> None:
        self.times: list[float] = []

    def command_from_state(self, t_s, r, v, m):
        self.times.append(float(t_s))
        return np.eye(3), 1.0


def test_point_mass_guidance_samples_only_at_step_boundaries():
    from orbitopt.flightlab.sixdof.gravity_turn import f05_scenario
    scn = f05_scenario()
    spy = _GuidanceSpy()
    veh = vehicle()
    propagate_point_mass(scn.state0, veh, scn.program, 2.0, dt_s=0.02,
                         throttle=scn.throttle, use_aero=False, guidance=spy)
    assert spy.times[0] == pytest.approx(scn.state0.t_s)
    for t in spy.times:
        k = (t - scn.state0.t_s) / 0.02
        assert abs(k - round(k)) < 1e-9, f"mid-stage guidance call at t={t}"


def _spec(**kwargs) -> UpperStageGuidanceSpec:
    kwargs.setdefault("target", ParkingOrbitTarget(altitude_m=400_000.0))
    kwargs.setdefault("dry_mass_kg", 1000.0)
    kwargs.setdefault("rated_thrust_vacuum_n", 440_000.0)
    kwargs.setdefault("isp_vacuum_s", 340.0)
    kwargs.setdefault("ignition_t_s", 100.0)
    kwargs.setdefault("command_slew_rate_deg_s", 3.0)
    return UpperStageGuidanceSpec(**kwargs)


def test_guidance_re_read_at_its_own_time_returns_the_held_command():
    from orbitopt.flightlab.frames import geodetic_to_ecef
    g = _spec().new_guidance()
    r = np.asarray(geodetic_to_ecef(28.5, -80.0, 120_000.0))
    v = np.array([0.0, 1000.0, 300.0])
    d1 = g.dcm_from_state(101.0, r, v, 8000.0)
    d_again = g.dcm_from_state(101.0, r, v * 1.7, 8000.0)  # re-read, new state
    assert np.allclose(d_again, d1, atol=1e-12)  # held, not the raw Lambda
    # and the limiter memory survived: the NEXT command slews from the held
    # one, not from the raw re-read state
    d2 = g.dcm_from_state(102.0, r, v * 1.7, 8000.0)
    ang = math.degrees(math.acos(float(np.clip((np.trace(d1 @ d2.T) - 1.0) / 2.0,
                                               -1.0, 1.0))))
    assert ang <= 2.0 * 3.0 + 1e-6  # one second at 3 deg/s on both axes


# ---------------------------------------------------------------------------
# the committed convergence curves
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bench() -> dict:
    if not BENCH_JSON.exists():
        pytest.skip("run benchmarks/bench_e6_04_step_convergence.py first")
    return json.loads(BENCH_JSON.read_text())


def _orders(rows: list[dict]) -> list[float]:
    return [row["order"] for row in rows if row["order"] is not None]


def test_plant_truncation_is_fourth_order(bench):
    rows = bench["integration"]["plant_truncation"]["rows"]
    orders = _orders(rows)
    assert len(orders) == 3
    # the last pair approaches the reference run's own float floor
    assert all(3.4 <= o <= 4.6 for o in orders[:2]), f"plant not 4th order: {orders}"
    assert rows[0]["magnitude"] > 1e3 * rows[-1]["magnitude"]


def test_closed_loop_output_is_flat_in_the_plant_step(bench):
    # E6-04: loop contraction absorbs the RK4 truncation transient -- the
    # loop output is step-invariant to the float floor (the pre-audit plant
    # leaked a 1st-order dt_s plateau instead; ADR-0008 records the numbers).
    rows = bench["integration"]["closed_loop"]["rows"]
    assert bench["integration"]["closed_loop"]["slew_limit_ticks_max"] == 0.0
    assert max(row["magnitude"] for row in rows) < 1e-7
    assert bench["integration"]["closed_loop"]["vertical_max_magnitude"] < 1e-7


def test_control_convergence_is_at_least_first_order(bench):
    control = bench["control"]
    assert control["slew_limit_ticks_max"] == 0.0
    orders = _orders(control["rows"])
    assert len(orders) == 4
    assert all(0.6 <= o <= 1.9 for o in orders), \
        f"control loop not ~1st order: {orders}"
    assert control["rows"][-1]["magnitude"] < 5e-3  # dt_c=0.01 on a 1 km ascent


def test_aligned_control_grid_is_step_invariant(bench):
    rows = {row["dt_s"]: row for row in bench["alignment"]["rows"]}
    a, b = rows[0.05], rows[0.025]
    assert a["n_ticks"] == b["n_ticks"]  # same tick schedule at dt_s = dt_c/2
    # residual: aero-table crossing times differ with the grid (ADR-0008 F5)
    assert abs(a["alt_m"] - b["alt_m"]) < 5e-3
    c = rows[0.02]
    assert c["n_ticks"] != a["n_ticks"]  # 0.05 is off the 0.02 grid: snaps
    assert abs(c["alt_m"] - a["alt_m"]) < 0.5  # snapped schedule stays sane


def test_default_steps_are_justified(bench):
    d = bench["default_vs_fine"]
    # dominant term is ZOH discretisation (1st order in dt_c, the sweep
    # shows +6.8 mm at dt_c=0.02 on a ~1 km ascent) -- not integration
    assert abs(d["d_alt_m"]) < 2e-2
    assert abs(d["d_tilt_deg"]) < 5e-3
    assert abs(d["d_max_abs_err_pitch_rad"]) < 2e-5
    assert bench["tick_bookkeeping"]["tick_period_in_band"]
    assert bench["tick_bookkeeping"]["tick_time_off_boundary_max_s"] < 1e-9

