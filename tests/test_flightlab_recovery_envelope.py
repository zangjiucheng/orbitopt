"""R-07 (task #239): nominal recovery report + dispersion-envelope plumbing.

Fast tests pin the capture-format geometry (site-ENU projection, uniform row
down-selection), the envelope dataset bookkeeping (every captured track in
index order, R-06 class on every failure, the binned band bracketing the
population, samples without a track ignored), the nominal report over the
honest stub verdict, and the figures rendering to PNG. A @slow wiring test
(substituted green landing, labelled as such) pins the Monte Carlo plumbing:
capture rides on the same records, and a fixed seed reports the identical
statistics with and without ``capture_tracks``.
"""
from __future__ import annotations

import math
import shutil

from pathlib import Path

import numpy as np
import pytest

from orbitopt.flightlab.contract import load_target
from orbitopt.flightlab.frames import geodetic_to_ecef
from orbitopt.flightlab.recovery.envelope import (
    TRACK_COLUMNS,
    TRACK_SCHEMA,
    capture_track_rows,
    envelope_dataset,
    nominal_report,
    plot_comparison,
    plot_envelope,
    site_from_target,
    write_nominal_report,
)
from orbitopt.flightlab.recovery.propagator import BoosterTrack

REPO = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures" / "flightlab" / "opentsiolkovsky"
VEHICLE = REPO / "src" / "orbitopt" / "data" / "flightlab" / "vehicles" / "synthetic_two_stage.yaml"
TARGET = REPO / "src" / "orbitopt" / "data" / "flightlab" / "scenarios" / "leo_two_stage.yaml"
SITE = (28.5619, -80.5772, 3.0)


def _fake_track(n: int = 1200, *, east_from: float = 8000.0,
                alt_from: float = 40000.0) -> BoosterTrack:
    """A straight east-descending synthetic booster track (ECEF) starting at
    the pad + ``east_from`` east, for the capture-geometry tests."""
    e = np.array(geodetic_to_ecef(SITE[0], SITE[1], SITE[2]))
    lat_r, lon_r = math.radians(SITE[0]), math.radians(SITE[1])
    east_unit = np.array([-math.sin(lon_r), math.cos(lon_r), 0.0])
    up_unit = np.array([math.cos(lat_r) * math.cos(lon_r),
                        math.cos(lat_r) * math.sin(lon_r), math.sin(lat_r)])
    t = np.linspace(207.0, 320.0, n)
    frac = np.linspace(0.0, 1.0, n)
    r = (e[None, :] + (east_from * (1.0 - frac))[:, None] * east_unit[None, :]
         + (alt_from * (1.0 - frac))[:, None] * up_unit[None, :])
    return BoosterTrack(epoch_utc="2026-11-10T12:00:00", times_s=t,
                        mass_kg=np.linspace(65000.0, 30000.0, n),
                        position_ecef_m=r,
                        velocity_ecef_m_s=np.zeros_like(r),
                        height_m_arr=alt_from * (1.0 - frac),
                        throttle=np.ones(n), mach=np.ones(n) * 2.0,
                        dynamic_pressure_pa=np.ones(n) * 1e3,
                        events=(), impact=None, termination="impact",
                        max_proper_g=3.0, max_dynamic_pressure_pa=1e4,
                        ignitions_used=2, dry_mass_kg=30000.0,
                        landing_reserve=None)


def _sample(idx: int, outcome: str, reason: str, *, touchdown=None,
            seed: int = 0) -> dict:
    """A sample record carrying a capture track: a swept ground path whose
    last point is the touchdown (landed) or a crash point (failed)."""
    rng = np.random.default_rng(seed + idx)
    end = touchdown if touchdown is not None else (rng.uniform(-2000, 2000),
                                                   rng.uniform(-2000, 2000))
    rows = []
    for i in range(20):
        f = i / 19.0
        rows.append([207.0 + 120.0 * f,
                     (1.0 - f) * 8000.0 + f * end[0] + rng.normal(0, 10),
                     (1.0 - f) * 300.0 + f * end[1] + rng.normal(0, 10),
                     40000.0 * (1.0 - f) + 1000.0])
    rec = {"idx": idx, "outcome": outcome, "reason": reason,
           "track": {"schema": TRACK_SCHEMA, "columns": list(TRACK_COLUMNS),
                     "n_full": 900, "rows": rows}}
    if outcome == "landed" and touchdown is not None:
        rec["touchdown_east_m"] = float(end[0])
        rec["touchdown_north_m"] = float(end[1])
    return rec


# ---------------------------------------------------------------------------
# fast: capture geometry
# ---------------------------------------------------------------------------

def test_capture_projects_to_site_enu_and_downsamples():
    track = _fake_track(n=1200)
    cap = capture_track_rows(track, SITE, 240)
    assert cap["schema"] == TRACK_SCHEMA
    assert cap["columns"] == list(TRACK_COLUMNS)
    assert cap["n_full"] == 1200 and len(cap["rows"]) == 240
    first, last = cap["rows"][0], cap["rows"][-1]
    assert first[0] == pytest.approx(207.0) and last[0] == pytest.approx(320.0)
    # starts 8 km east, 40 km above the ellipsoid (up is measured from the
    # site altitude, 3 m here); ends at the site, on the ground
    assert first[1] == pytest.approx(8000.0, rel=1e-3)
    assert first[2] == pytest.approx(0.0, abs=1.0)
    assert first[3] == pytest.approx(40000.0 - SITE[2], rel=1e-3)
    assert last[1] == pytest.approx(0.0, abs=1.0)
    assert last[3] == pytest.approx(-SITE[2], abs=1e-6)


def test_capture_keeps_short_tracks_intact_and_rejects_tiny_budgets():
    track = _fake_track(n=50)
    cap = capture_track_rows(track, SITE, 240)
    assert len(cap["rows"]) == 50 and cap["n_full"] == 50
    with pytest.raises(ValueError, match="at least 2"):
        capture_track_rows(track, SITE, 1)


def test_site_from_target_is_the_launch_site():
    assert site_from_target(load_target(TARGET)) == pytest.approx(SITE)


# ---------------------------------------------------------------------------
# fast: envelope dataset bookkeeping
# ---------------------------------------------------------------------------

def _dataset(**kw):
    samples = [
        _sample(1, "landed", "LANDED", touchdown=(5.0, -4.0), seed=3),
        _sample(0, "failed", "LANDING_POSITION_OUT_OF_TOLERANCE", seed=4),
        _sample(2, "failed", "INSUFFICIENT_PROPELLANT_ENTRY_BURN", seed=5),
        {"idx": 3, "outcome": "landed", "reason": "LANDED"},  # uncaptured
    ]
    return envelope_dataset(samples, SITE, **kw)


def test_envelope_dataset_orders_classifies_and_ignores_uncaptured():
    data = _dataset()
    assert data["kind"] == "flightlab-recovery-envelope"
    assert data["n_captured"] == 3
    assert [t["idx"] for t in data["trajectories"]] == [0, 1, 2]
    by_idx = {t["idx"]: t for t in data["trajectories"]}
    assert by_idx[1]["failure_class"] is None
    assert by_idx[1]["outcome"] == "landed"
    assert by_idx[0]["failure_class"] == "landing_accuracy"
    assert by_idx[2]["failure_class"] == "fuel_insufficient"
    assert by_idx[1]["touchdown_enu_m"] == [5.0, -4.0]
    assert data["ground"]["max_touchdown_radius_m"] == pytest.approx(6.403, abs=1e-3)
    assert all(t["rows"] for t in data["trajectories"])


def test_envelope_band_brackets_the_population_and_gaps_sparse_tails():
    data = _dataset()
    band = data["band"]
    assert band["kind"] == "altitude_km_vs_range_to_site_km"
    assert len(band["range_km"]) == len(band["alt_min_km"]) == len(band["alt_max_km"])
    assert all(lo <= hi for lo, hi in zip(band["alt_min_km"], band["alt_max_km"]))
    # every captured track passes near the start bin, ~40 km of altitude
    assert any(x < 9.0 for x in band["range_km"])
    top = [h for x, h in zip(band["range_km"], band["alt_max_km"]) if x < 9.0]
    assert top and max(top) == pytest.approx(41.0, abs=2.0)
    # a single-sample population still reports a band (>=2 samples per bin:
    # the 3 tracks share the sweep) -- but a population with no rows at all
    # reports none
    assert envelope_dataset([{"idx": 0, "outcome": "failed", "reason": "X"}],
                            SITE)["band"] is None


def test_nominal_report_over_the_honest_stub_verdict(tmp_path, monkeypatch):
    """The stub ascent separates with no recoverable propellant: the nominal
    report must carry the red gate, the governing reason and the flown
    (ballistic) track, re-assessed through checks.evaluate -- the report
    shape holds whether the recovery passed or honestly failed."""
    from orbitopt.flightlab.simulate import simulate

    path = tmp_path / "otcli"
    shutil.copy(FIX / "stub_cli.py", path)
    path.chmod(0o755)
    monkeypatch.setenv("ORBITOPT_OPENTSIOLKOVSKY", str(path))
    monkeypatch.setenv("OT_STUB_FIXTURE", str(FIX / "synthetic_out"))
    monkeypatch.setenv("OT_STUB_SRC", str(REPO / "src" / "orbitopt"))

    run = simulate(load_target(TARGET), VEHICLE, tmp_path / "run",
                   timeout_s=60, refly=False, run_recovery=False)
    outcome = _outcome_from_stub_run(run)
    report = nominal_report(outcome, load_target(TARGET), max_track_rows=300)
    assert report["kind"] == "flightlab-recovery-nominal-report"
    assert report["recovered"] is False
    assert report["reason_codes"] == ["INSUFFICIENT_PROPELLANT_BOOSTBACK"]
    assert report["verdict"]["mission_constraints_passed"] is False
    gate = next(c for c in report["checks"] if c["id"] == "recovery-landing")
    assert gate["passed"] is False and gate["reason_title"]
    assert report["corridor"]["feasible"] is False
    assert report["corridor"]["reason_codes"] == ["INSUFFICIENT_PROPELLANT_BOOSTBACK"]
    assert report["landing"] is None
    assert report["handoff"] is not None  # R-02 report always built
    assert report["track"] is not None and report["track"]["schema"] == TRACK_SCHEMA
    assert len(report["track"]["rows"]) <= 300
    write_nominal_report(report, tmp_path / "nom.json")
    import json
    again = json.loads((tmp_path / "nom.json").read_text(encoding="utf-8"))
    write_nominal_report(again, tmp_path / "nom2.json")
    assert (tmp_path / "nom.json").read_bytes() == (tmp_path / "nom2.json").read_bytes()


def _outcome_from_stub_run(run):
    """Fly the stub run's recovery through the real pipeline (fast: FL-302's
    propellant floor fires within seconds on this separation state)."""
    from orbitopt.flightlab.recovery.pipeline import fly_recovery
    return fly_recovery(run, load_target(TARGET), VEHICLE)


def test_nominal_report_defaults_plan_options_to_the_mc_warm_start(monkeypatch):
    """nominal_recovery() with no plan_options must use the Monte Carlo's
    warm start, i.e. the nominal report and the dispersed population share
    one setup (acceptance: fair comparison, same conditions)."""
    from orbitopt.flightlab.recovery.envelope import nominal_recovery
    from orbitopt.flightlab.recovery.montecarlo import MC_PLAN_OPTIONS
    import orbitopt.flightlab.recovery.envelope as env

    seen = {}

    def fake_fly_recovery(*args, **kwargs):
        seen.update(kwargs["plan_options"])
        return "outcome"

    monkeypatch.setattr(env, "fly_recovery", fake_fly_recovery)
    nominal_recovery(load_target(TARGET), VEHICLE, None)
    assert seen == dict(MC_PLAN_OPTIONS)


# ---------------------------------------------------------------------------
# fast: rendering (matplotlib is in the locked CI env; skip if absent)
# ---------------------------------------------------------------------------

def test_plot_envelope_and_comparison_render(tmp_path):
    pytest.importorskip("matplotlib")
    data = _dataset(nominal=capture_track_rows(_fake_track(n=400), SITE, 600))
    png = tmp_path / "env.png"
    plot_envelope(data, png, title="unit envelope")
    assert png.stat().st_size > 10_000

    ci = {"low": 0.4, "high": 0.9}
    run_a = {"label": "baseline", "dataset": data, "nominal": data["nominal"],
             "success_rate": 0.5, "ci": ci}
    run_b = {"label": "stack", "dataset": _dataset(),
             "nominal": data["nominal"], "success_rate": 0.75, "ci": ci}
    png2 = tmp_path / "cmp.png"
    plot_comparison([run_a, run_b], png2)
    assert png2.stat().st_size > 10_000


# ---------------------------------------------------------------------------
# slow: Monte Carlo capture plumbing (substituted landing, wiring only)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_slow_monte_carlo_capture_rides_on_identical_statistics(monkeypatch):
    """WIRING-ONLY (capture plumbing, not a physics result): with the FL-303
    nominal verdict substituted for every landing, a fixed-seed run with
    ``capture_tracks`` must report the identical statistics to the run
    without it -- the captured tracks are records, never inputs."""
    pytest.importorskip("gfold")
    import sys
    sys.path.insert(0, str(REPO / "tests"))
    from orbitopt.flightlab.recovery import pipeline as pipe
    from orbitopt.flightlab.recovery.montecarlo import (
        MC_PLAN_OPTIONS, Dispersions, run_monte_carlo)
    from test_flightlab_simulate_recovery import (
        TARGET as TEST_TARGET, VEHICLE as TEST_VEHICLE,
        _ascent_with_textbook_separation, _nominal_green_landing)

    green = _nominal_green_landing()
    monkeypatch.setattr(pipe, "fly_landing_burn", green)
    target = load_target(TEST_TARGET)
    disp = Dispersions(thrust_pct=4.0, density_pct=8.0, sep_altitude_m=500.0,
                       sep_downrange_m=1_000.0, sep_speed_m_s=6.0)
    opts = dict(MC_PLAN_OPTIONS, landing={"mode": "single"},
                max_fevals=18, landing_divert_budget_m=None,
                landing_reserve_margin_kg=0.0)
    common = dict(dispersions=disp, seed=9, n=2, workers=1, plan_options=opts)
    plain = run_monte_carlo(target, TEST_VEHICLE,
                            _ascent_with_textbook_separation(), **common)
    caught = run_monte_carlo(target, TEST_VEHICLE,
                             _ascent_with_textbook_separation(),
                             capture_tracks=40, **common)
    assert caught.success_rate == plain.success_rate
    assert caught.reason_histogram == plain.reason_histogram
    assert caught.statistics == plain.statistics
    assert caught.ellipse == plain.ellipse
    assert len(caught.track_capture) > 0 and not plain.track_capture
    for a, b in zip(plain.samples, caught.samples):
        stripped = {k: v for k, v in b.items() if k != "track"}
        assert stripped == dict(a)
        assert len(b["track"]["rows"]) <= 40
        assert b["track"]["schema"] == TRACK_SCHEMA
