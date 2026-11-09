"""solve_lambert_single(plane_normal=...): explicit transfer sense for
transfer planes that contain the z axis, where pykep's default choice
depends on the sign of rounding noise."""
from __future__ import annotations

import numpy as np
import pytest

pk = pytest.importorskip("pykep")

from orbitopt.lambert.cpu import solve_lambert_single

MU = 398600.4418e9
TOF = 3.0 * 86400.0


def _h(r, v):
    return np.cross(r, v)


def test_plane_normal_plus_z_matches_default_for_regular_geometry():
    r1 = np.array([7.0e6, 0.0, 1.0e5])
    r2 = np.array([-1.0e8, 3.0e8, 2.0e7])
    (v1_ref, v2_ref), = solve_lambert_single(r1, r2, TOF, mu=MU)[:1]
    (v1, v2), = solve_lambert_single(r1, r2, TOF, mu=MU, plane_normal=[0.0, 0.0, 1.0])[:1]
    np.testing.assert_allclose(v1, v1_ref, rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(v2, v2_ref, rtol=1e-9, atol=1e-6)


@pytest.mark.parametrize("z_noise", [0.0, 1e-9, -1e-9])
def test_polar_plane_follows_plane_normal_not_rounding_noise(z_noise):
    # Transfer plane is the x-z plane: r1 x r2 has a z component of ~0.
    r1 = np.array([7.0e6, z_noise, 0.0])
    r2 = np.array([-2.0e8, 0.0, 3.0e8])
    normal = np.array([0.0, -1.0, 0.0])  # r1 x r2 direction: short way
    (v1, _), = solve_lambert_single(r1, r2, TOF, mu=MU, plane_normal=normal)[:1]
    assert np.dot(_h(r1, v1), normal) > 0.0

    (v1_retro, _), = solve_lambert_single(r1, r2, TOF, mu=MU, prograde=False, plane_normal=normal)[:1]
    assert np.dot(_h(r1, v1_retro), normal) < 0.0
    assert np.linalg.norm(v1) < np.linalg.norm(v1_retro)  # 124 deg short way vs 236 deg long way


def test_plane_normal_parallel_to_r1_is_rejected():
    r1 = np.array([7.0e6, 0.0, 0.0])
    with pytest.raises(ValueError):
        solve_lambert_single(r1, np.array([0.0, 4.0e8, 0.0]), TOF, mu=MU, plane_normal=r1)
