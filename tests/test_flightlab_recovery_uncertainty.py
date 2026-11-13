"""Checks for orbitopt.flightlab.recovery.uncertainty (R-04, task #236).

The acceptance criteria this file pins:

* fixed seed fully reproducible -- the draw stream is a fixed-order
  contract, so the same (config, [seed, idx]) reproduces every factor, and
  the legacy flat :class:`Dispersions` path consumes EXACTLY the RNG calls
  FL-304 always consumed (pre-R-04 fixed-seed results stay reproducible);
* distribution parameters recorded in the report -- the declaration is
  carried verbatim (round-trip tested) into ``MonteCarloResult.uncertainty``;
* every source individually switchable -- enabling/disabling one source
  changes only that source's draw (inactive sources stay neutral and consume
  no RNG).

Physics side (propagator): the wind defaults are zero and leave every force
bit-for-bit what it was; a steady ENU wind is equivalent to shifting the
ground velocity by minus the wind, tested through the real aero path.
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pytest
import yaml

from orbitopt.flightlab.frames import geodetic_to_ecef
from orbitopt.flightlab.recovery.montecarlo import Dispersions
from orbitopt.flightlab.recovery.propagator import (
    BoosterEngine,
    BoosterModel,
    BoosterPhysics,
    _aero_ecef,
    _wind_ecef,
)
from orbitopt.flightlab.recovery.uncertainty import (
    SOURCE_ORDER,
    UncertaintyModel,
)

EPOCH = datetime(2026, 11, 10, 12, 0, 0)
PAD = (28.5, -80.6)


def _full_config():
    return {
        "thrust": {"enabled": True, "distribution": "uniform_pct",
                   "half_width_pct": 5.0},
        "isp": {"enabled": True, "distribution": "normal_pct", "sigma_pct": 1.5},
        "density": {"enabled": True, "distribution": "uniform_pct",
                    "half_width_pct": 8.0},
        "drag": {"enabled": False},
        "wind": {"enabled": True, "distribution": "normal",
                 "sigma_east_m_s": 4.0, "sigma_north_m_s": 3.0,
                 "sigma_up_m_s": 0.5},
        "initial_state": {"enabled": True, "distribution": "normal",
                          "sigma_altitude_m": 300.0, "sigma_downrange_m": 800.0,
                          "sigma_crossrange_m": 800.0, "sigma_speed_m_s": 5.0,
                          "sigma_heading_deg": 0.2},
    }


def _vehicle():
    engine = BoosterEngine(
        count=1, usable_for=("boostback", "landing"), thrust_vacuum_n=1_800_000.0,
        isp_sea_level_s=282.0, isp_vacuum_s=311.0, nozzle_exit_area_m2=5.0,
        throttle_min=0.4, throttle_max=1.0, restarts_max=3)
    return BoosterModel(
        name="test-booster", provenance="illustrative", dry_mass_kg=30_000.0,
        landing_propellant_reserve_kg=0.0, reference_area_m2=10.7516,
        cd_mach=((0.0, 0.35), (0.8, 0.38), (1.1, 0.55), (1.5, 0.5),
                 (2.5, 0.4), (4.0, 0.32), (8.0, 0.26)),
        engine=engine)


# ---------------------------------------------------------------------------
# config parsing / recording
# ---------------------------------------------------------------------------


def test_config_roundtrip_and_declaration_is_verbatim(tmp_path):
    """from_file (YAML, nested under ``uncertainty:``) -> to_dict -> from_dict
    is stable, sources keep declaration order, and the flat view agrees with
    the declared widths."""
    path = tmp_path / "uncertainty.yaml"
    path.write_text(yaml.safe_dump({"uncertainty": _full_config()}),
                    encoding="utf-8")
    m = UncertaintyModel.from_file(path)
    assert m.active_sources() == ("thrust", "isp", "density", "wind",
                                  "initial_state")
    assert UncertaintyModel.from_dict(m.to_dict()) == m
    decl = m.to_dict()
    assert list(decl) == list(SOURCE_ORDER)
    assert decl["wind"]["sigma_east_m_s"] == 4.0
    assert decl["drag"] == {"enabled": False}
    view = m.dispersions_view()
    assert view["thrust_pct"] == 5.0 and view["isp_pct"] == 1.5
    assert view["cd_pct"] == 0.0 and view["sep_speed_m_s"] == 5.0
    # a bare flat mapping (no wrapper key) parses identically
    assert UncertaintyModel.from_config(_full_config()) == m


def test_switches_are_per_source():
    """Disabling one source zeroes exactly that source's draw; enabling wind
    on top of a wind-off config leaves thrust alone and only shifts what the
    wind consumed downstream (the fixed-order contract)."""
    base = _full_config()
    base["wind"] = {"enabled": False}
    m_off = UncertaintyModel.from_config(base)
    d_off = m_off.draw(np.random.default_rng([11, 0]))
    assert d_off["wind_east_m_s"] == d_off["wind_north_m_s"] == 0.0
    assert d_off["wind_up_m_s"] == 0.0
    assert d_off["thrust_factor"] != 1.0  # thrust stays active

    m_on = UncertaintyModel.from_config(_full_config())
    d_on = m_on.draw(np.random.default_rng([11, 0]))
    assert d_on["thrust_factor"] == d_off["thrust_factor"]  # drawn first, same
    assert (d_on["wind_east_m_s"], d_on["wind_north_m_s"],
            d_on["wind_up_m_s"]) != (0.0, 0.0, 0.0)

    only_thrust = UncertaintyModel.from_config(
        {"thrust": {"distribution": "uniform_pct", "half_width_pct": 2.0}})
    d = only_thrust.draw(np.random.default_rng([1, 2]))
    assert 0.98 <= d["thrust_factor"] <= 1.02
    assert d["isp_factor"] == d["density_scale"] == d["cd_factor"] == 1.0
    assert d["sep_heading_deg"] == 0.0


def test_fixed_seed_reproduces_and_order_is_the_contract():
    m = UncertaintyModel.from_config(_full_config())
    a = m.draw(np.random.default_rng([7, 3]))
    b = m.draw(np.random.default_rng([7, 3]))
    assert a == b
    c = m.draw(np.random.default_rng([7, 4]))
    assert c != a


def test_legacy_dispersions_draw_exactly_the_old_stream():
    """The R-04 translation of a flat Dispersions consumes exactly the RNG
    calls the pre-R-04 inline draw consumed -- byte-for-byte identical
    factors, wind all zero."""
    def legacy(rng, disp):
        def uniform(pct):
            return (1.0 + float(rng.uniform(-pct, pct)) / 100.0
                    if pct > 0.0 else 1.0)
        def normal(sigma):
            return float(rng.normal(0.0, sigma)) if sigma > 0.0 else 0.0
        return {"thrust_factor": uniform(disp.thrust_pct),
                "isp_factor": uniform(disp.isp_pct),
                "density_scale": uniform(disp.density_pct),
                "cd_factor": uniform(disp.cd_pct),
                "sep_altitude_m": normal(disp.sep_altitude_m),
                "sep_downrange_m": normal(disp.sep_downrange_m),
                "sep_crossrange_m": normal(disp.sep_crossrange_m),
                "sep_speed_m_s": normal(disp.sep_speed_m_s),
                "sep_heading_deg": normal(disp.sep_heading_deg)}

    setups = [Dispersions(),
              Dispersions(thrust_pct=0.0, isp_pct=2.0, density_pct=3.0,
                          cd_pct=0.0, sep_altitude_m=100.0,
                          sep_speed_m_s=1.0)]
    for disp in setups:
        for seed in (0, 3, 17, 99):
            a = legacy(np.random.default_rng([seed, 5]), disp)
            b = UncertaintyModel.from_dispersions(disp).draw(
                np.random.default_rng([seed, 5]))
            for key, val in a.items():
                assert b[key] == val, (key, val, b[key])
            assert b["wind_east_m_s"] == 0.0


@pytest.mark.parametrize("bad, msg", [
    ({"wind": {"enabled": True}}, "distribution"),
    ({"thrust": {"enabled": True, "distribution": "uniform_pct"}},
     "half_width_pct"),
    ({"gust": {"enabled": True}}, "unknown uncertainty source"),
    ({"thrust": {"distribution": "uniform_pct", "half_width_pct": 1,
                 "sigma_east_m_s": 2}}, "unknown"),
    ({"thrust": {"distribution": "triangular", "half_width_pct": 1}},
     "unknown distribution"),
    ({"thrust": {"distribution": "normal_pct", "sigma_pct": -1}}, ">= 0"),
    ({"thrust": {"enabled": False}}, "at least one active"),
])
def test_bad_configs_are_rejected(bad, msg):
    with pytest.raises(ValueError, match=msg):
        UncertaintyModel.from_config(bad)


# ---------------------------------------------------------------------------
# report wiring through run_monte_carlo (stubs, no physics)
# ---------------------------------------------------------------------------


def test_run_monte_carlo_records_the_declaration(tmp_path, monkeypatch):
    from orbitopt.flightlab.recovery import montecarlo as mc

    seen: list[dict] = []

    def fake_sample(job):
        seen.append(job["uncertainty"])
        return {"reason": "stub", "outcome": "stub"}

    monkeypatch.setattr(mc, "_run_sample", fake_sample)
    monkeypatch.setattr(mc, "write_run", lambda *a, **k: None)
    path = tmp_path / "unc.yaml"
    path.write_text(yaml.safe_dump(_full_config()), encoding="utf-8")

    result = mc.run_monte_carlo(None, "vehicle.yaml", None, uncertainty=path,
                                n=2, seed=3, workers=1)
    out = result.to_dict()
    assert out["schemaVersion"] == "1.3"
    assert out["uncertainty"]["wind"]["sigma_east_m_s"] == 4.0
    assert out["uncertainty"]["drag"] == {"enabled": False}
    assert out["dispersions"]["thrust_pct"] == 5.0
    assert any("R-04" in a for a in result.assumptions)
    assert all(s == seen[0] for s in seen)  # every job carries it verbatim

    # both knobs at once is a contradiction, not a silent precedence
    with pytest.raises(ValueError, match="not both"):
        mc.run_monte_carlo(None, "v.yaml", None, uncertainty=path,
                           dispersions=Dispersions(), n=1, seed=1)


# ---------------------------------------------------------------------------
# wind in the acceptance physics (FL-004 propagator, real aero path)
# ---------------------------------------------------------------------------


def test_zero_wind_is_bit_for_bit_identity():
    ph = BoosterPhysics()
    assert not ph.has_wind
    r = np.array(geodetic_to_ecef(*PAD, 20_000.0))
    assert _wind_ecef(r, ph).tolist() == [0.0, 0.0, 0.0]
    v = np.array((100.0, -250.0, -30.0))
    veh = _vehicle()
    plain = _aero_ecef(r, v, veh, ph)
    for w in (0.0, -0.0):  # explicit zeros must not fork the arithmetic
        f, spd, mach, q = _aero_ecef(r, v, veh,
                                     BoosterPhysics(wind_east_m_s=w,
                                                    wind_north_m_s=w,
                                                    wind_up_m_s=w))
        assert np.array_equal(f, plain[0]) and spd == plain[1]
        assert mach == plain[2] and q == plain[3]


def test_steady_wind_is_a_ground_velocity_shift():
    """Drag(q, mach, direction) with wind W at velocity v equals the no-wind
    aero at v - W: the wind enters exactly as air-relative velocity, nothing
    else."""
    veh = _vehicle()
    r = np.array(geodetic_to_ecef(0.0, 0.0, 15_000.0))
    # ENU basis at (lat 0, lon 0): east (0,1,0), north (0,0,1), up (1,0,0)
    wind = BoosterPhysics(wind_east_m_s=8.0, wind_north_m_s=-3.0,
                          wind_up_m_s=0.5)
    w_ecef = _wind_ecef(r, wind)
    assert w_ecef[0] == pytest.approx(0.5)    # up
    assert w_ecef[1] == pytest.approx(8.0)    # east
    assert w_ecef[2] == pytest.approx(-3.0)   # north

    v = np.array((0.0, 220.0, -60.0))
    f_w, s_w, m_w, q_w = _aero_ecef(r, v, veh, wind)
    f_0, s_0, m_0, q_0 = _aero_ecef(r, v - w_ecef, veh, BoosterPhysics())
    assert np.allclose(f_w, f_0, rtol=1e-12, atol=1e-9)
    assert s_w == pytest.approx(s_0, rel=1e-12)
    assert m_w == pytest.approx(m_0, rel=1e-12) and q_w == pytest.approx(q_0)

    # headwind along the air-relative motion increases that speed (with the
    # default co-rotating-air model the ECEF velocity IS air-relative, so the
    # wind shifts it by exactly its ENU vector)
    v_east = np.array((0.0, 220.0, 0.0))
    _, calm, _, _ = _aero_ecef(r, v_east, veh, BoosterPhysics())
    west_wind = BoosterPhysics(wind_east_m_s=-10.0)
    _, into, _, _ = _aero_ecef(r, v_east, veh, west_wind)
    assert into > calm + 9.9


def test_wind_requires_finite_fields():
    with pytest.raises(ValueError, match="wind"):
        BoosterPhysics(wind_east_m_s=float("nan"))
