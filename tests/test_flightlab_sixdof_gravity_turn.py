"""F-05 (task #230) regressions: the 3-DOF point-mass reference is right on
its own analytics, the five-rung ladder flies and behaves, and the headline
3-DOF/6-DOF difference metrics stay inside the committed thresholds.

E1-02 (task #251) added the powered/coast phase split to the same committed
report: the phase gates below pin where the attitude error and the angle of
attack happen (the 180 s study burns out at 99.85 s, so a whole-run maximum is
a mix of the loaded phase and an unactuated coast), that ``tvc.burn_s`` is the
burn and not the flight span, and that the report's own qualification block can
be recomputed from its own phase records and the constraints it declares.

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
    F05_PHASE_CONSTRAINTS,
    QUALIFIED_RUNG,
    REFERENCE_RUNG,
    RUNG_ORDER,
    ascent_channels,
    compare_ascent,
    f05_scenario,
    propagate_point_mass,
    run_f05_study,
)
from orbitopt.flightlab.sixdof.rigidbody import RigidBodyPhysics
from orbitopt.flightlab.sixdof.tvc import COAST, POWERED, WaypointAttitudeProgram

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

#: E1-02 (task #251) gates on the committed phase split of ``6dof_full``, with
#: margin over the committed run (powered att err 4.77 deg, AoA 14.42 deg,
#: q*alpha 482 kPa-deg, coast AoA 62.61 deg and about a tenth of a pascal;
#: re-recorded after E6-01/E6-04).
#: The powered lines are the acceptance's; the coast lines are the *load*
#: statement that licenses reading the big coast angles as geometry -- and the
#: coast AoA ceiling documents the magnitude of the artifact the task asks to
#: explain, so a future change that moves it is noticed rather than re-explained.
GATE_PHASES = {
    "powered_att_err_max_deg": ("<=", 5.0),
    "powered_att_err_rms_deg": ("<=", 3.0),
    "powered_alpha_max_deg": ("<=", 15.0),          # also the aero table's AoA edge
    "powered_q_alpha_max_pa_deg": ("<=", 520_000.0),
    "powered_max_q_pa": ("<=", 75_000.0),
    "powered_pitch_deflection_margin": (">=", 0.95),
    # burn_s, not the 180 s span: the burn is m_prop / mdot = 60 t at
    # T_vac / (Isp g0) = 599.1 kg/s = 100.15 s since E6-01's choked flow
    # (99.85 s before); the gate separates it from the span, not the 0.3 s.
    "powered_engine_s": ("<=", 101.0),
    "coast_engine_s": ("==", 0.0),
    "coast_alpha_max_deg": ("<=", 65.0),
    "coast_q_alpha_max_pa_deg": ("<=", 10.0),
    "coast_max_q_pa": ("<=", 1.0),
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
    # 60 t of propellant in a 40 s cut: the engine is lit until the very last
    # tick, so the phase split must report a powered phase and nothing else
    # (E1-02) -- the coast is not invented when the scenario has none
    assert set(full.phases) == {POWERED}
    assert full.diagnostics.burn_s == pytest.approx(40.0, rel=1e-6)
    assert full.diagnostics.coast_s == 0.0


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
    assert doc["schema"] == "orbitopt-fl605-2"
    assert "#230" in doc["task"] and "#251" in doc["task"]
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


def _flat_phases(doc: dict) -> dict:
    """``phases[rung][phase][metric]`` of the qualified rung, flattened under
    ``powered_`` / ``coast_`` prefixes, so a gate table can name a metric
    unambiguously."""
    qualified = doc["phases"][QUALIFIED_RUNG]
    return {f"{phase}_{k}": v for phase in qualified for k, v in qualified[phase].items()}


def test_committed_phase_split_meets_its_gates():
    """E1-02 (task #251): the powered phase is the qualification, the coast is
    reported and must stay unloaded, and neither may drift past its gate."""
    doc = json.loads(BENCHMARK_JSON.read_text(encoding="utf-8"))
    assert doc["schema"] == "orbitopt-fl605-2"
    qualified = doc["phases"][QUALIFIED_RUNG]
    assert set(qualified) == {POWERED, COAST}
    assert qualified[COAST]["t0_s"] > doc["rungs"][QUALIFIED_RUNG]["burn_out_t_s"]
    assert qualified[POWERED]["t1_s"] <= doc["rungs"][QUALIFIED_RUNG]["burn_out_t_s"]
    # a point-mass rung has no attitude to be in error about and no gimbal to
    # margin: its phase block is aero only, which is what makes the two backends
    # comparable rather than accidentally equal-looking
    for name in ("3dof_vacuum", "3dof_aero"):
        block = doc["phases"][name][POWERED]
        assert "att_err_max_deg" not in block and "pitch_deflection_margin" not in block
        assert "alpha_max_deg" in block and "max_q_pa" in block

    flat = _flat_phases(doc)
    for key, (op, gate) in GATE_PHASES.items():
        assert key in flat, key
        value = flat[key]
        ok = {"<=": value <= gate, ">=": value >= gate, "==": value == pytest.approx(gate)}[op]
        assert ok, f"{key} = {value} {op} {gate}"


def test_committed_burn_time_is_the_burn_not_the_flight_span():
    """The bug task #251 was opened for: the closed-loop rungs flew to 180 s on
    a 99.85 s burn and used to report ``tvc.burn_s = 180.0``. The ticker's own
    engine-on bookkeeping must put the burn and the coast back where they
    belong, and the two must partition the flight."""
    doc = json.loads(BENCHMARK_JSON.read_text(encoding="utf-8"))
    for name in ("6dof_ideal", "6dof_lag", "6dof_full"):
        tvc = doc["rungs"][name]["tvc"]
        burn_out = doc["rungs"][name]["burn_out_t_s"]
        t_end = doc["rungs"][name]["t_end_s"]
        assert abs(tvc["burn_s"] - burn_out) <= 0.01 * burn_out, name
        assert tvc["burn_s"] < 0.6 * t_end and tvc["coast_s"] > 0.0, name
        assert tvc["coast_s"] == pytest.approx(t_end - burn_out, rel=0.01), name
        assert tvc["burn_s"] + tvc["coast_s"] == pytest.approx(t_end, rel=1e-9), name
        assert doc["phases"][name][POWERED]["engine_s"] == pytest.approx(tvc["burn_s"]), name
        assert doc["phases"][name][COAST]["engine_s"] == 0.0, name
    for name in ("3dof_vacuum", "3dof_aero"):
        assert "tvc" not in doc["rungs"][name], name


def test_committed_alpha_max_is_a_coast_phase_artifact():
    """The written explanation of "6-DOF 62.8 deg vs 3-DOF 26.7 deg", as a
    gate: the 6-DOF whole-run peak belongs to the coast, at an altitude where
    the phase's own dynamic pressure is a fraction of a pascal, while the
    loaded phase of BOTH backends peaks at 14.3 deg. If the discrepancy ever
    moves into the powered phase, this fails and the explanation must be
    rewritten."""
    doc = json.loads(BENCHMARK_JSON.read_text(encoding="utf-8"))
    attr = doc["alpha_max_attribution"]
    assert attr["rung"] == QUALIFIED_RUNG
    assert attr["phase"] == COAST
    assert attr["t_s"] == pytest.approx(doc["scenario"]["integration"]["t_end_s"])
    assert attr["alpha_max_deg"] == pytest.approx(
        doc["rungs"][QUALIFIED_RUNG]["alpha_max_deg"], rel=1e-9)
    assert attr["alpha_max_deg"] > attr["powered_alpha_max_deg"]
    assert attr["altitude_m"] > 100_000.0 and attr["phase_max_q_pa"] < 1.0
    assert attr["phase_q_alpha_max_pa_deg"] < 10.0
    # loaded => the backends agree; that is the fidelity claim E1-02 keeps
    assert abs(attr["powered_alpha_max_deg"]
               - attr["reference_powered_alpha_max_deg"]) < 1.0
    assert attr["powered_alpha_max_t_s"] < doc["rungs"][QUALIFIED_RUNG]["burn_out_t_s"]


def test_committed_qualification_recomputes_from_its_own_numbers():
    """The committed verdict is not taken on trust: every constraint of
    :data:`F05_PHASE_CONSTRAINTS` is re-evaluated here against the report's own
    phase records (and the derived burn/agreement rows), and must reproduce the
    committed value and verdict -- so a stale or hand-edited block fails."""
    doc = json.loads(BENCHMARK_JSON.read_text(encoding="utf-8"))
    qualified, reference = (doc["phases"][QUALIFIED_RUNG], doc["phases"][REFERENCE_RUNG])
    burn_out = doc["rungs"][QUALIFIED_RUNG]["burn_out_t_s"]
    derived = {
        "run": {"tvc_burn_s_vs_burn_out_rel":
                abs(doc["rungs"][QUALIFIED_RUNG]["tvc"]["burn_s"] - burn_out) / burn_out},
        "study": {
            "powered_alpha_model_agreement_deg":
                abs(qualified[POWERED]["alpha_max_deg"] - reference[POWERED]["alpha_max_deg"]),
            "coast_minus_powered_alpha_max_deg":
                qualified[COAST]["alpha_max_deg"] - qualified[POWERED]["alpha_max_deg"],
        },
    }
    checks = {(c["phase"], c["metric"]): c for c in doc["qualification"]["checks"]}
    assert len(checks) == len(doc["qualification"]["checks"]) == len(F05_PHASE_CONSTRAINTS)
    for constraint in F05_PHASE_CONSTRAINTS:
        row = checks[(constraint.phase, constraint.metric)]
        source = (qualified[constraint.phase] if constraint.phase in (POWERED, COAST)
                  else derived[constraint.phase])
        value = source[constraint.metric]
        assert row["value"] == pytest.approx(value, rel=1e-9, abs=1e-12), constraint.metric
        assert row["op"] == constraint.op and row["limit"] == constraint.limit
        assert row["pass"] == {
            "<=": value <= constraint.limit, ">=": value >= constraint.limit,
            "==": abs(value - constraint.limit) <= 1e-9}[constraint.op], constraint.metric
    assert doc["qualification"]["all_passed"] is True
    assert all(row["pass"] for row in checks.values())
