"""E11-06: the booster's engine sequence -- the pure burn extraction, the exported track column, the telemetry
chart and the recovery panel's engine-sequence section."""
from __future__ import annotations

import numpy as np
import pytest

from orbitopt.flightlab.engine_status import EngineBurn, burns_from_extra_columns, engine_burns


def test_a_run_on_engine_is_one_burn_and_a_gap_is_a_new_ignition():
    t = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    lit = [13, 13, 0, 0, 13, 13, 5, 3, 0]
    burns = engine_burns(t, lit)
    assert [b.index for b in burns] == [1, 2]
    assert (burns[0].t_start_s, burns[0].t_end_s) == (0.0, 2.0)
    assert (burns[1].t_start_s, burns[1].t_end_s) == (4.0, 8.0)
    assert burns[1].engines_text == "13 → 5 → 3"        # shut down inside the burn, never relit
    assert burns[0].engines_text == "13"


def test_throttle_is_reported_for_the_engines_that_are_lit():
    t = [0, 1, 2, 3, 4]
    lit = [13, 13, 5, 5, 0]
    thr = [0.9, 0.6, 0.25, 0.2, 0.0]                 # fractions of the FULL set (5/13 = 0.385 at most)
    b = engine_burns(t, lit, thr)[0]
    assert b.throttle_max == pytest.approx(0.9)
    assert b.throttle_min == pytest.approx(0.2 / (5 / 13))           # 52 % of the five lit engines
    assert b.propellant_kg is None
    mass = [1000, 980, 960, 950, 945]
    assert engine_burns(t, lit, thr, mass)[0].propellant_kg == pytest.approx(55.0)


def test_no_engine_column_means_no_burns_and_a_track_that_never_fires_has_none():
    assert burns_from_extra_columns([0, 1], {"mach": [0.5, 0.5]}) == ()
    assert engine_burns([0, 1, 2], [0, 0, 0]) == ()
    assert engine_burns([], []) == ()
    assert isinstance(EngineBurn(1, 0.0, 1.0, ((0.0, 13),), 0.5, 1.0, None).duration_s, float)


def test_a_track_without_the_engine_column_still_shows_its_burn_timing():
    burns = burns_from_extra_columns([0, 1, 2, 3, 4], {"throttle": [0.0, 0.8, 0.5, 0.0, 0.0]})
    assert len(burns) == 1 and (burns[0].t_start_s, burns[0].t_end_s) == (1.0, 3.0)
    assert burns[0].engines_text == "\u2014" and not burns[0].engines_known


def test_an_unstaged_track_keeps_exactly_the_columns_it_always_had():
    from orbitopt.flightlab.recovery.propagator import BoosterModel, BoosterState, ThrustCmd, propagate
    from pathlib import Path
    import datetime as dt

    veh = BoosterModel.load(Path(__file__).resolve().parents[1] / "src/orbitopt/data/flightlab/vehicles/starship_class_rtls_starbase_fins.yaml")
    sep = BoosterState(t_s=0.0, epoch_utc=dt.datetime(2026, 11, 10), frame="ECEF",
                       position_m=(6.4e6, 0.0, 0.0), velocity_m_s=(300.0, 0.0, 0.0), mass_kg=600e3)
    track = propagate(sep, veh, lambda t, _s: ThrustCmd(0.6, (1.0, 0.0, 0.0)) if t < 2.0 else None, 4.0,
                      dt_s=0.1, sample_s=0.5)
    assert "engines_lit" not in track.to_body_track("ECEF").extra_columns       # whole set only: unchanged export


def test_the_flown_track_exports_the_engines_lit_column():
    from orbitopt.flightlab.recovery.propagator import BoosterModel, BoosterState, ThrustCmd, propagate
    from pathlib import Path
    import datetime as dt

    veh_path = Path(__file__).resolve().parents[1] / "src/orbitopt/data/flightlab/vehicles/starship_class_rtls_starbase_fins.yaml"
    veh = BoosterModel.load(veh_path)
    sep = BoosterState(t_s=0.0, epoch_utc=dt.datetime(2026, 11, 10), frame="ECEF",
                       position_m=(6.4e6, 0.0, 0.0), velocity_m_s=(300.0, 0.0, 0.0), mass_kg=600e3)

    def cmd(t, _s):                                    # 13 engines to 2 s, 3 engines 2-4 s, off after
        if t < 2.0:
            return ThrustCmd(0.6, (1.0, 0.0, 0.0))
        if t < 4.0:
            return ThrustCmd(0.12, (1.0, 0.0, 0.0), min_throttle_scale=3 / 13)
        return None

    track = propagate(sep, veh, cmd, 6.0, dt_s=0.1, sample_s=0.5)
    assert track.engine_count == veh.engine.count
    body = track.to_body_track("ECEF")
    lit = np.array(body.extra_columns["engines_lit"])
    assert set(np.round(lit, 3)) <= {0.0, 3.0, float(veh.engine.count)}
    assert lit[1] == veh.engine.count and lit[-1] == 0.0 and (lit == 3.0).any()   # sample 0 is the pre-command seed
    burns = burns_from_extra_columns(body.t_s, body.extra_columns, body.mass_kg)
    assert len(burns) == 1 and burns[0].engines_text == f"{veh.engine.count} → 3"


def test_the_replay_exposes_an_engines_lit_chart():
    from orbitopt.flightlab.ui.flight_panel import TELEMETRY_CHARTS

    assert ("engines_lit", "Engines lit", "engines") in TELEMETRY_CHARTS


def test_the_recovery_panel_lists_the_burns(monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from orbitopt.flightlab.compare import RecoverySummary
    from orbitopt.flightlab.ui.recovery_panel import RecoveryPanel

    QApplication.instance() or QApplication([])
    burns = engine_burns([0, 1, 2, 3, 4, 5, 6, 7], [13, 13, 0, 0, 13, 5, 3, 0], [0.9, 0.9, 0, 0, 0.7, 0.3, 0.15, 0],
                         [900e3, 890e3, 880e3, 880e3, 880e3, 850e3, 840e3, 835e3])
    summary = RecoverySummary(state="flown", kind="rtls", gate=None, checks=(), touchdown=None, explanations=(),
                              engine_burns=burns)
    panel = RecoveryPanel()
    panel.set_summary(summary)
    assert panel.engine_table.rowCount() == 2
    assert "2 ignitions" in panel.engine_label.text()
    assert panel.engine_table.item(1, 3).text() == "13 → 5 → 3"
    assert panel.engine_table.item(1, 3).toolTip().startswith("Engines lit over the burn")
    panel.clear()
    assert panel.engine_table.rowCount() == 0 and not panel.engine_table.isVisibleTo(panel)


def test_an_engine_count_axis_has_whole_number_ticks():
    from orbitopt.flightlab.ui.charts import _integer_ticks

    for lo, hi in ((-0.8, 13.8), (0.0, 3.0), (-0.3, 5.3), (0.0, 130.0)):
        ticks = _integer_ticks(lo, hi)
        assert ticks and all(abs(t - round(t)) < 1e-9 for t in ticks)
        assert all(lo - 1e-9 <= t <= hi + 1e-9 for t in ticks)


def test_a_stage_track_without_a_throttle_column_gets_an_estimated_throttle_curve_in_the_telemetry():
    from orbitopt.flightlab.contract import BodyTrack
    from orbitopt.flightlab.ui.telemetry import build_telemetry_channels

    class _Run:                                                  # the duck-typed run the builder needs
        pass

    t = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    mass = [1000.0, 800.0, 600.0, 500.0, 400.0, 400.0]            # 100 kg/s, 100, 50, 50, then off
    pos = tuple((6.4e6, 0.0, 0.0) for _ in t)
    vel = tuple((0.0, 0.0, 0.0) for _ in t)
    stage = BodyTrack(body_id="stage1", parent_body_id=None, frame="GCRS", epoch_utc="2026-11-10T12:00:00Z",
                      provenance="illustrative", t_s=tuple(t), mass_kg=tuple(mass), position_m=pos, velocity_m_s=vel,
                      extra_columns={})
    recorded = BodyTrack(body_id="booster", parent_body_id="stage1", frame="GCRS", epoch_utc="2026-11-10T12:00:00Z",
                         provenance="illustrative", t_s=tuple(t), mass_kg=tuple(mass), position_m=pos,
                         velocity_m_s=vel, extra_columns={"throttle": (0.0, 0.5, 0.5, 0.5, 0.0, 0.0)})
    run = _Run()
    run.tracks = (stage, recorded)
    channels = {c.id: c for c in build_telemetry_channels(run)}
    thr = {s.body_id: s for s in channels["throttle"].series}
    est = thr["stage1"]
    assert est.derived and "est." in est.label                       # Stage 1 now HAS a throttle series, labelled
    assert [round(v) for _t, v in est.points] == [100, 100, 50, 50, 0]
    assert est.points[0][0] == 0.0                                   # a burn starts where the mass starts to fall
    assert not thr["booster"].derived and thr["booster"].points[1][1] == 50.0   # a recorded throttle is untouched
