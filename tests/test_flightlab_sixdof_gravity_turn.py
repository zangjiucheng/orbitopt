"""F-05 (task #230) regressions: the 3-DOF point-mass reference is right on
its own analytics, the five-rung ladder flies and behaves, and the headline
3-DOF/6-DOF difference metrics stay inside the committed thresholds.

Real RK4 physics (same budget as the F-04 closed-loop tests -- the full
180 s study is deliberately NOT re-run here; it lives in
``docs/benchmarks/fl605_gravity_turn_ascent.json``, which
``test_committed_benchmark_satisfies_the_regression_thresholds`` audits):

    PYTHONPATH=src python -m pytest tests/test_flightlab_sixdof_gravity_turn.py
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from orbitopt.entry import MU_EARTH_M3_S2
from orbitopt.flightlab import frames
from orbitopt.flightlab.atmosphere import G0_M_S2
from orbitopt.flightlab.sixdof.gravity_turn import (
    ATTRIBUTION_STEPS,
    DIFF_CHANNELS,
    RUNG_ORDER,
    ascent_channels,
    compare_ascent,
    f05_scenario,
    propagate_point_mass,
    run_f05_study,
)
from orbitopt.flightlab.sixdof.rigidbody import RigidBodyPhysics
from orbitopt.flightlab.sixdof.tvc import WaypointAttitudeProgram

pytestmark = pytest.mark.slow  # real RK4 propagations

REPO = Path(__file__).resolve().parents[1]
BENCHMARK_JSON = REPO / "docs" / "benchmarks" / "fl605_gravity_turn_ascent.json"

#: The acceptance's regression gates on the committed 180 s study (headline
#: 6dof_full - 3dof_aero, max |difference|). Calibrated at the committed
#: run's values (alt 2134 m, speed 2.1 m/s, fpa 0.47 deg, mass 0.01 kg) with
#: margin; growing past any of these means one side of the fidelity ladder
#: moved -- see docs/flightlab.md (F-05). The mass gate is the engine-law
#: consistency check: both models feed the same RigidBodyEngine, so the
#: columns may differ only through Isp's ambient-pressure dependence.
GATE_HEADLINE = {
    "altitude_m": 4_000.0,   # m
    "speed_m_s": 10.0,       # m/s
    "fpa_deg": 1.2,          # deg
    "mass_kg": 5.0,          # kg
}

#: Each attribution term must actually contribute something, on the channel
#: it is expected to show up in (the committed study's max |difference|:
#: aero 34 km of apogee, the rigid-body loop 2.1 km, the actuator lag a
#: milli-degree of FPA -- its trajectory effect under this gentle 0.65 deg/s
#: program is sub-decimetre, which is itself the finding; F-04's own tests
#: pin where the lag bites). If one collapses to ~0 the scenario silently
#: stopped exercising it.
GATE_ATTRIBUTION_MIN = {
    "aero_point_mass": ("altitude_m", 10_000.0),
    "attitude_dynamics": ("altitude_m", 1_000.0),
    "actuator_lag": ("fpa_deg", 3e-4),
    "aero_rigid_body": ("altitude_m", 10_000.0),
}


def test_point_mass_reference_matches_vacuum_ballistic_analytics():
    """Aero off, non-rotating point-mass Earth: a constant-attitude vertical
    boost is textbook Tsiolkovsky-with-gravity-loss, and the mass column is
    exact by construction. Pins the point-mass RHS and the shared engine law."""
    scn = f05_scenario()
    physics = RigidBodyPhysics(atmosphere=False, rotation=False, j2=False)
    run = propagate_point_mass(scn.state0, scn.vehicle,
                               WaypointAttitudeProgram.vertical(90.0), 30.0,
                               dt_s=0.05, sample_every=5, physics=physics,
                               use_aero=False)
    assert run.terminal == "t_end"
    eng = scn.vehicle.engine
    ve = eng.isp_vacuum_s * G0_M_S2                       # vacuum ambient => no correction
    mdot = eng.thrust_vacuum_n / ve
    m0 = scn.vehicle.mass_wet_kg
    m = run.samples["mass_kg"]
    t = run.samples["t_s"]
    assert np.max(np.abs(m - (m0 - mdot * t))) < 1e-6 * m0

    r0 = frames.geodetic_to_ecef(28.5, -80.6, 0.0).reshape(3)
    g = MU_EARTH_M3_S2 / float(r0 @ r0)                    # j2 off: g at (near) the pad radius
    v_an = ve * np.log(m0 / (m0 - mdot * t)) - g * t
    v = np.linalg.norm(np.stack([run.samples["vx_m_s"], run.samples["vy_m_s"],
                                 run.samples["vz_m_s"]], axis=-1), axis=1)
    assert abs(v[-1] - v_an[-1]) < 3e-3 * v_an[-1]
    # altitude: same analytic speed trapezoidated on a fine analytic grid
    fine = np.linspace(0.0, t[-1], 3001)
    v_an_fine = ve * np.log(m0 / (m0 - mdot * fine)) - g * fine
    h_an = np.concatenate([[0.0], np.cumsum(0.5 * (v_an_fine[1:] + v_an_fine[:-1])
                                            * np.diff(fine))])
    assert abs(run.samples["altitude_m"][-1] - h_an[-1]) < 5e-3 * h_an[-1]


def test_compare_ascent_against_itself_is_zero_and_recomputes_the_channels():
    scn = f05_scenario()
    run = propagate_point_mass(scn.state0, scn.vehicle, scn.program, 10.0,
                               dt_s=0.05, sample_every=5, use_aero=True)
    diff = compare_ascent(run, run)
    for ch in DIFF_CHANNELS:
        assert np.max(np.abs(diff.diff[ch])) == 0.0
    # channels are recomputed from the raw ECEF columns and agree with the
    # run's own sampled altitude (same ellipsoid definition)
    ch = ascent_channels(run)
    assert np.max(np.abs(ch["altitude_m"] - run.samples["altitude_m"])) < 1e-6


@pytest.fixture(scope="module")
def short_study():
    """All five rungs of a 40 s cut of the scenario (vertical rise + the
    first third of the pitch-over): the live behavioural regression."""
    scn = replace(f05_scenario(t_end_s=40.0, dt_s=0.05, sample_every=100),
                  program=WaypointAttitudeProgram(times_s=(0.0, 10.0, 40.0),
                                                  heading_deg=(90.0, 90.0, 90.0),
                                                  pitch_deg=(90.0, 90.0, 45.0),
                                                  roll_deg=(0.0, 0.0, 0.0)))
    return run_f05_study(scn)


def test_short_ladder_flies_and_tracks_the_program(short_study):
    for name in RUNG_ORDER:
        run = short_study.runs[name]
        r = run.run if hasattr(run, "run") else run
        assert r.terminal == "t_end", name
        assert float(r.samples["altitude_m"][-1]) > 5_000.0, name
    full = short_study.runs["6dof_full"]
    # the F-04 tracker flies the 1.5 deg/s ramp without touching its limits;
    # the residual 3-axis error / AoA are the trim angle plus the known
    # un-actuated roll-about-flight-axis DOF (F-04 scope), not a runaway --
    # trajectory fidelity itself is gated on the FPA difference below
    assert full.metrics.att_err_max_deg < 12.0
    assert full.run.alpha_max_deg < 20.0
    assert full.diagnostics.pitch_saturated_s == 0.0       # 1.5 deg/s ramp is gentle
    assert full.diagnostics.pitch_slew_limited_s == 0.0


def test_short_headline_difference_grows_after_the_pitch_over(short_study):
    d = short_study.headline
    t = d.t_s
    vertical = t <= 10.0
    late = t >= 25.0
    # before the pitch-over both models thrust along the same axis and the
    # tracker sits on its setpoint: the pair may not separate
    assert np.max(np.abs(d.diff["altitude_m"][vertical])) < 5.0
    # after it does: aero realisation + lag + tracker are in the loop, and
    # the divergence must be visibly growing, not noise
    assert np.max(np.abs(d.diff["altitude_m"][late])) > 3.0 * np.max(
        np.abs(d.diff["altitude_m"][vertical])) + 1.0
    assert 0.5 < np.max(np.abs(d.diff["speed_m_s"])) < 40.0
    # same engine law => same mass bookkeeping within the Isp-vs-ambient band
    assert np.max(np.abs(d.diff["mass_kg"])) < 2.0


def test_committed_benchmark_satisfies_the_regression_thresholds():
    assert BENCHMARK_JSON.exists(), (
        "run PYTHONPATH=src python benchmarks/bench_fl605_gravity_turn.py")
    doc = json.loads(BENCHMARK_JSON.read_text(encoding="utf-8"))
    assert doc["schema"] == "orbitopt-fl605-1"
    assert "#230" in doc["task"]
    # the committed study is re-runnable: the scenario block pins the knobs
    integ = doc["scenario"]["integration"]
    assert integ["t_end_s"] == 180.0 and integ["dt_s"] == 0.05
    assert doc["scenario"]["program"]["pitch_deg"] == [90.0, 90.0, 18.0]

    for name in RUNG_ORDER:
        assert doc["rungs"][name]["terminal"] == "t_end", name
    # one engine law on every rung: the records must agree on when it ran dry
    burn_times = [doc["rungs"][n]["burn_out_t_s"] for n in RUNG_ORDER]
    assert max(burn_times) - min(burn_times) < 1.0

    metrics = doc["headline"]["metrics"]
    for ch, gate in GATE_HEADLINE.items():
        unit = {"altitude_m": "m", "speed_m_s": "m_s", "fpa_deg": "deg",
                "mass_kg": "kg"}[ch]
        assert metrics[f"d_{ch}_max_abs_{unit}"] < gate, ch
        assert metrics[f"d_{ch}_rms_{unit}"] <= metrics[f"d_{ch}_max_abs_{unit}"]

    for name, desc in ATTRIBUTION_STEPS:
        step = doc["attribution"][name]["metrics"]
        ch, lo = GATE_ATTRIBUTION_MIN[name]
        unit = {"altitude_m": "m", "speed_m_s": "m_s", "fpa_deg": "deg",
                "mass_kg": "kg"}[ch]
        assert step[f"d_{ch}_max_abs_{unit}"] > lo, f"{name}: {desc}"

    # the curves are the metrics' own data, not a stale copy
    curves = doc["headline"]["curves"]
    t = np.asarray(curves["t_s"])
    assert np.all(np.diff(t) > 0.0)
    for ch in DIFF_CHANNELS:
        unit = {"altitude_m": "m", "speed_m_s": "m_s", "fpa_deg": "deg",
                "mass_kg": "kg"}[ch]
        curve = np.asarray(curves[f"d_{ch}"])
        assert curve.size == t.size
        assert abs(float(curve[-1]) - metrics[f"d_{ch}_terminal_{unit}"]) < 1e-9
        assert float(np.max(np.abs(curve))) == pytest.approx(
            metrics[f"d_{ch}_max_abs_{unit}"], rel=1e-9, abs=1e-12)


def test_committed_attribution_telescopes_to_the_headline_exactly():
    """``headline = attitude_dynamics + actuator_lag + aero_rigid_body -
    aero_point_mass`` holds SAMPLE-WISE, not just statistically: every curve
    shares the one 1-s grid and the rung pairs form a closed ladder, so the
    decomposition is algebraically exact. The strongest statement the
    acceptance's "difference sources" bullet can have -- and it pins the
    attribution scheme itself."""
    doc = json.loads(BENCHMARK_JSON.read_text(encoding="utf-8"))
    head = doc["headline"]["curves"]
    att = doc["attribution"]
    for ch in DIFF_CHANNELS:
        rhs = (np.asarray(att["attitude_dynamics"]["curves"][f"d_{ch}"])
               + np.asarray(att["actuator_lag"]["curves"][f"d_{ch}"])
               + np.asarray(att["aero_rigid_body"]["curves"][f"d_{ch}"])
               - np.asarray(att["aero_point_mass"]["curves"][f"d_{ch}"]))
        err = np.abs(np.asarray(head[f"d_{ch}"]) - rhs)
        assert float(np.max(err)) < 1e-3, ch


def test_headline_channels_agree_with_recomputed_rung_endpoints():
    """The committed headline's terminal values must equal the recomputed
    channel difference of its own rung endpoints -- i.e. the benchmark's
    headline really is 6dof_full minus 3dof_aero, not some other pair."""
    doc = json.loads(BENCHMARK_JSON.read_text(encoding="utf-8"))
    curves = doc["headline"]["curves"]
    metrics = doc["headline"]["metrics"]
    for ch in DIFF_CHANNELS:
        unit = {"altitude_m": "m", "speed_m_s": "m_s", "fpa_deg": "deg",
                "mass_kg": "kg"}[ch]
        diff_end = float(curves[f"six_{ch}"][-1]) - float(curves[f"ref_{ch}"][-1])
        assert abs(diff_end - metrics[f"d_{ch}_terminal_{unit}"]) < 1e-6
    assert curves["six_mass_kg"][-1] == pytest.approx(
        doc["rungs"]["6dof_full"]["mass_end_kg"], rel=1e-9)
    assert curves["ref_mass_kg"][-1] == pytest.approx(
        doc["rungs"]["3dof_aero"]["mass_end_kg"], rel=1e-9)
