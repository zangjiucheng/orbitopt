"""3-DOF lifting entry model (numpy + scipy)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from orbitopt.entry import (
    MU_EARTH_M3_S2,
    R_EARTH_M,
    EntryVehicle,
    density_kg_m3,
    flat_plate_coefficients,
    fly_entry,
    height_m,
)


def test_density_table():
    assert density_kg_m3(0.0) == pytest.approx(1.225)
    assert density_kg_m3(120e3) == pytest.approx(2.438e-8)
    assert density_kg_m3(-500.0) == pytest.approx(1.225)  # clamped below ground
    h = np.linspace(0.0, 400e3, 200)
    rho = np.array([density_kg_m3(x) for x in h])
    assert np.all(np.diff(rho) < 0)  # strictly thinner with height


def test_flat_plate_lift_to_drag_is_cot_alpha():
    for alpha in (30.0, 45.0, 65.0):
        cd, cl = flat_plate_coefficients(alpha)
        assert cl / cd == pytest.approx(1.0 / math.tan(math.radians(alpha)))
        assert cd == pytest.approx(2.0 * math.sin(math.radians(alpha)) ** 3)


def test_height_on_the_ellipsoid():
    assert height_m(np.array([R_EARTH_M + 1000.0, 0.0, 0.0])) == pytest.approx(1000.0)
    polar_radius = R_EARTH_M * (1.0 - 1.0 / 298.257223563)
    assert height_m(np.array([0.0, 0.0, polar_radius + 500.0])) == pytest.approx(500.0, abs=1.0)


def _deorbited_state(dv=80.0):
    r = R_EARTH_M + 275e3
    v = math.sqrt(MU_EARTH_M3_S2 / r)
    return np.array([r, 0.0, 0.0, 0.0, v - dv, 0.0])


def test_entry_reaches_the_stop_height_with_plausible_loads():
    cd, cl = flat_plate_coefficients(65.0)
    vehicle = EntryVehicle(150_000.0, 400.0, cd, cl)
    lift_up = fly_entry(_deorbited_state(), vehicle, bank_deg=0.0)
    banked = fly_entry(_deorbited_state(), vehicle, bank_deg=60.0)
    for traj in (lift_up, banked):
        assert traj.ended_at_height and traj.heights_m[-1] == pytest.approx(1000.0, abs=1.0)
        assert 1.0 < traj.aero_g.max() < 6.0
        assert traj.first_crossing_s(120e3) is not None
    # Banking spends lift sideways: less range, harder deceleration.
    assert banked.times_s[-1] < lift_up.times_s[-1]
    assert banked.aero_g.max() > lift_up.aero_g.max()
    # Terminal (belly-flop) speed near the ground: tens of m/s, not orbital.
    v_air = lift_up.states[-1, 3:] - np.cross([0.0, 0.0, 7.2921150e-5], lift_up.states[-1, :3])
    assert 30.0 < np.linalg.norm(v_air) < 150.0


def test_no_atmosphere_no_burn_stays_in_orbit():
    vehicle = EntryVehicle(150_000.0, 400.0, 1.5, 0.7)
    traj = fly_entry(_deorbited_state(dv=0.0), vehicle, bank_deg=0.0, max_time_s=600.0)
    assert not traj.ended_at_height and traj.heights_m.min() > 250e3


def test_bank_reversal_and_density_scale():
    cd, cl = flat_plate_coefficients(65.0)
    vehicle = EntryVehicle(150_000.0, 400.0, cd, cl)
    right = fly_entry(_deorbited_state(), vehicle, bank_deg=45.0)
    # Reversed from the start = banked the other way: a mirror image in z.
    left = fly_entry(_deorbited_state(), vehicle, bank_deg=45.0, bank_reversal_s=0.0)
    mirror = fly_entry(_deorbited_state(), vehicle, bank_deg=-45.0)
    assert right.states[-1, 2] == pytest.approx(-left.states[-1, 2], rel=1e-6)
    assert left.states[-1, :3] == pytest.approx(mirror.states[-1, :3], abs=1.0)
    # Reversing part-way lands in between.
    half = fly_entry(_deorbited_state(), vehicle, bank_deg=45.0, bank_reversal_s=0.5 * right.times_s[-1])
    assert left.states[-1, 2] < half.states[-1, 2] < right.states[-1, 2] or \
        right.states[-1, 2] < half.states[-1, 2] < left.states[-1, 2]
    # Thinner air: the same entry comes down further along; another integrator agrees.
    thin = fly_entry(_deorbited_state(), vehicle, bank_deg=45.0, density_scale=0.8)

    def range_angle(traj):
        a, b = traj.states[0, :3], traj.states[-1, :3]
        return math.acos(a @ b / np.linalg.norm(a) / np.linalg.norm(b))

    assert range_angle(thin) > range_angle(right)
    rk45 = fly_entry(_deorbited_state(), vehicle, bank_deg=45.0, method="RK45", rtol=1e-11, atol=1e-5, max_step_s=10.0)
    assert np.linalg.norm(rk45.states[-1, :3] - right.states[-1, :3]) < 50.0
