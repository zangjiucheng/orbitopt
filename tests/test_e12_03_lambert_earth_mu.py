"""E12-03 (task #393): the Lambert Earth-mu regression.

``lambert/cpu.py``'s ``solve_lambert_single`` defaults to ``mu=pk.MU_SUN``:
correct for interplanetary scenes, catastrophically wrong for a 500 km
rendezvous. This test pins both halves of the defence:

1. handing a LEO pair to the *default* mu must fail loudly in the
   physics_io band check -- it must never silently return a solar-system
   trajectory through an Earth orbit;
2. every ``solve_lambert_single`` call the rendezvous planner makes carries
   an explicit mu that is the Earth value, to the magnitude order a central-
   body mixup would break (a 500 km orbit at solar mu would need a ~58-day
   time of flight; the mixup shows up as 5-6 orders of magnitude).

The config layer adds a third net (dynamics.mu must equal the exact packaged
Earth value -- tests/test_e12_03_rendezvous_planner.py).
"""
from __future__ import annotations

import math

import pytest

from orbitopt import physics_io as pio
from orbitopt.lambert import solve_lambert_single
from orbitopt.physics_io import KNOWN_MU, PhysicsIOError
from orbitopt.relative import EARTH_MU, EARTH_RE
from orbitopt.rendezvous import load_reference_config, plan_rendezvous


def _leo_pair():
    """Two LEO points 40 deg apart in a 51.6 deg plane: a plausible transfer
    the planner could be asked about."""
    r1 = EARTH_RE + 500e3
    u1, u2 = 0.2, 0.2 + math.radians(40.0)
    inc = math.radians(51.6)

    def st(u):
        p = r1 * math.cos(u)
        q = r1 * math.sin(u)
        x = p
        y = q * math.cos(inc)
        z = q * math.sin(inc)
        return (x, y, z)

    return st(u1), st(u2)


def test_default_mu_on_a_leo_pair_is_a_loud_failure():
    r1, r2 = _leo_pair()
    with pytest.raises(PhysicsIOError):
        solve_lambert_single(r1, r2, 1200.0)


def test_explicit_earth_mu_on_the_same_pair_works():
    r1, r2 = _leo_pair()
    sols = solve_lambert_single(r1, r2, 1200.0, mu=KNOWN_MU["earth"])
    v1 = sols[0][0]
    speed = math.sqrt(sum(x * x for x in v1))
    r = math.sqrt(sum(x * x for x in r1))
    escape = math.sqrt(2.0 * KNOWN_MU["earth"] / r)
    assert 3000.0 < speed < escape  # bound LEO-scale orbit, not a solar-system escape


def test_planner_never_lets_the_lambert_default_through(monkeypatch):
    """Every planner call into the Lambert solver must carry mu == Earth,
    explicitly and at Earth magnitude."""
    import orbitopt.rendezvous.planner as planner

    recorded = []
    real = planner.solve_lambert_single

    def spy(*args, **kwargs):
        recorded.append(kwargs.get("mu", "MISSING"))
        return real(*args, **kwargs)

    monkeypatch.setattr(planner, "solve_lambert_single", spy)
    plan_rendezvous(load_reference_config())

    assert recorded, "planner never invoked solve_lambert_single"
    for mu in recorded:
        assert mu != "MISSING", "planner called solve_lambert_single without an explicit mu"
        assert mu == KNOWN_MU["earth"], "planner passed a mu that is not the packaged Earth value"
        # order-of-magnitude guard: LEO dynamics live at 1e14, not 1e20
        assert 1e13 < mu < 1e16


def test_planner_explicit_mu_survives_a_narrow_window(monkeypatch):
    """The spy test above proves the reference mission; also pin it for a
    mission whose config mu differs from the default (solar) by 5 orders."""
    import orbitopt.rendezvous.planner as planner
    import yaml
    from importlib import resources

    from orbitopt.rendezvous import parse_rendezvous_config

    doc = yaml.safe_load(
        (resources.files("orbitopt") / "data" / "leo_rdv_001.yaml").read_text(encoding="utf-8")
    )
    doc["windows"]["transfer_tof_min_s"] = 1500.0
    doc["windows"]["transfer_tof_max_s"] = 1700.0

    recorded_mu = []
    real = planner.solve_lambert_single

    def spy(*args, **kwargs):
        recorded_mu.append(kwargs["mu"])
        return real(*args, **kwargs)

    monkeypatch.setattr(planner, "solve_lambert_single", spy)
    plan = plan_rendezvous(parse_rendezvous_config(doc))
    assert 1500.0 <= plan.tof_s <= 1700.0
    assert recorded_mu and all(m == EARTH_MU for m in recorded_mu)
