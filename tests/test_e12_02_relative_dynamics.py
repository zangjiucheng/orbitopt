"""E12-02 (task #392): relative orbital dynamics.

Gates the ECI <-> LVLH/RTN kinematics (orbitopt.relative.frames), the closed
form CW model (orbitopt.relative.cw) against RK4 integration of its own
equations, the nonlinear two-spacecraft propagator
(orbitopt.relative.propagation) against conservation laws and convergence,
and the physics_io contract: LVLH/RTN are accepted where declared, and every
geocentric-frame check keeps rejecting them. The validity envelope itself is
gated in tests/test_e12_02_cw_validity.py against the committed benchmark
JSON.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from orbitopt import physics_io as pio
from orbitopt.physics_io import PhysicsIOError, UnitMismatchError
from orbitopt.relative import (
    EARTH_J2,
    EARTH_MU,
    EARTH_RE,
    cw_state_transition,
    eci_to_lvlh,
    lvlh_dcm,
    lvlh_to_eci,
    mean_motion_circular,
    propagate_cw,
    propagate_pair,
)

MU = EARTH_MU


def _rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def kepler_state(a, e, inc, raan, argp):
    """Geocentric equatorial state at perigee for classical elements (SI)."""
    r_p = a * (1.0 - e)
    v_p = math.sqrt(MU * (2.0 / r_p - 1.0 / a))
    pqw = np.array([[r_p, 0.0, 0.0], [0.0, v_p, 0.0]])
    rot = _rotz(raan) @ _rotx(inc) @ _rotz(argp)
    r, v = (rot @ pqw.T).T
    return r, v


def circular(a, inc, raan):
    return kepler_state(a, 0.0, inc, raan, 0.0)


CHIEF_ECC = [  # (label, a, e, inc, raan, argp)
    ("geo_circ", 42164000.0, 0.0, 0.0, 0.0, 0.0),
    ("leo45", 7071000.0, 0.0, math.radians(45), math.radians(30), 0.0),
    ("ecc01", 10000000.0, 0.1, math.radians(51.6), math.radians(120), math.radians(20)),
    ("ecc20", 26600000.0, 0.2, math.radians(28.5), math.radians(45), math.radians(300)),
    ("polar_e05", 7200000.0, 0.05, math.radians(98), 0.0, math.radians(80)),
]


class TestFrames:
    def test_dcm_is_right_orthonormal(self):
        for _, a, e, i, O, w in CHIEF_ECC:
            r, v = kepler_state(a, e, i, O, w)
            D = lvlh_dcm(r, v)
            assert np.abs(D @ D.T - np.eye(3)).max() < 1e-12
            assert np.linalg.det(D) == pytest.approx(1.0, abs=1e-12)

    def test_dcm_axes_geometry(self):
        # A known inclined circular chief: x must be radial, z must be h_hat.
        r, v = circular(7071000.0, math.radians(45), math.radians(30))
        D = lvlh_dcm(r, v)
        x, y, z = D
        assert x == pytest.approx(r / np.linalg.norm(r), abs=1e-12)
        h = np.cross(r, v)
        assert z == pytest.approx(h / np.linalg.norm(h), abs=1e-12)
        # y along-track: positive inertial rate component, in the plane
        assert np.dot(y, v) > 0
        assert abs(np.dot(y, x)) < 1e-12

    def test_deputy_directly_below_is_pure_minus_x(self):
        r, v = circular(7071000.0, math.radians(53), math.radians(40))
        n = mean_motion_circular(MU, np.linalg.norm(r))
        dr = -123.4 * (r / np.linalg.norm(r))
        # same inertial velocity: the position is pure -x, but the *rate* is
        # not zero -- a co-velocity deputy slides along-track at -omega x dr
        rho, rho_dot = eci_to_lvlh(r, v, dr, np.zeros(3), "GCRS")
        assert rho == pytest.approx(np.array([-123.4, 0.0, 0.0]), abs=1e-9)
        assert np.abs(rho_dot[[0, 2]]).max() < 1e-12
        assert rho_dot[1] == pytest.approx(123.4 * n, rel=1e-9)
        # truly stationary in LVLH needs the frame-rate velocity
        omega = np.cross(r, v) / np.dot(r, r)
        rho2, rho_dot2 = eci_to_lvlh(r, v, dr, np.cross(omega, dr), "GCRS")
        assert np.abs(rho_dot2).max() < 1e-12

    def test_roundtrip_is_exact_to_roundoff(self):
        rng = np.random.default_rng(392)
        for _, a, e, i, O, w in CHIEF_ECC:
            r, v = kepler_state(a, e, i, O, w)
            dr = rng.normal(0.0, 5000.0, 3)
            dv = rng.normal(0.0, 5.0, 3)
            rho, rho_dot = eci_to_lvlh(r, v, dr, dv, "EME2000")
            dr2, dv2 = lvlh_to_eci(r, v, rho, rho_dot, "EME2000")
            assert np.abs(dr2 - dr).max() < 1e-8 * max(1.0, np.abs(dr).max() / 1000)
            assert np.abs(dv2 - dv).max() < 1e-10

    def test_eccentric_chief_velocity_transform_carries_omega(self):
        # A deputy with a purely radial inertial separation and *zero* inertial
        # relative velocity does have a LVLH-frame rate: the frame rotates.
        _, r, v = "ecc20", *kepler_state(26600000.0, 0.2, math.radians(28.5),
                                         math.radians(45), math.radians(300))
        dr = r * (200.0 / np.linalg.norm(r))
        rho, rho_dot = eci_to_lvlh(r, v, dr, np.zeros(3), "GCRS")
        assert np.abs(rho[1:]).max() == pytest.approx(0.0, abs=1e-9)
        assert rho[0] == pytest.approx(200.0, abs=1e-9)
        assert rho_dot[1] != pytest.approx(0.0, abs=1e-6)  # along-track from frame rate
        dr2, dv2 = lvlh_to_eci(r, v, rho, rho_dot, "GCRS")
        assert np.abs(dr2 - dr).max() < 1e-8
        assert np.abs(dv2).max() < 1e-12

    def test_batch_broadcast(self):
        r, v = circular(7071000.0, math.radians(45), math.radians(30))
        rhat = r / np.linalg.norm(r)
        offs = np.outer([100.0, 1000.0, 10.0], rhat)  # pure radial separations
        rho, rho_dot = eci_to_lvlh(r, v, offs, np.zeros((3, 3)), "GCRS")
        assert rho.shape == (3, 3)
        assert np.abs(rho[:, 0] - np.array([100.0, 1000.0, 10.0])).max() < 1e-9
        assert np.abs(rho[:, 1:]).max() < 1e-9

    def test_rejects_degenerate_and_bad_units(self):
        r, v = circular(7071000.0, 0.3, 0.1)
        with pytest.raises(PhysicsIOError, match="angular momentum"):
            eci_to_lvlh(r, -r / np.linalg.norm(r) * 7500.0,
                        np.array([10.0, 0, 0]), np.zeros(3), "GCRS")
        with pytest.raises(PhysicsIOError):
            eci_to_lvlh(np.zeros(3), v, np.array([10.0, 0, 0]), np.zeros(3), "GCRS")
        with pytest.raises(PhysicsIOError):
            eci_to_lvlh(r, v, np.array([np.nan, 0, 0]), np.zeros(3), "GCRS")
        # beyond lunar distance: no longer a pair separation, out of band
        with pytest.raises(UnitMismatchError):
            eci_to_lvlh(r, v, np.array([5.0e8, 0, 0]), np.zeros(3), "GCRS")
        # rotating-frame chief rejected: no Earth-fixed labels in CHIEF_FRAMES
        with pytest.raises(PhysicsIOError, match="orbital|rotating|family|frame"):
            eci_to_lvlh(r, v, np.array([10.0, 0, 0]), np.zeros(3), "ECEF")
        # and a family name still names a family, not a frame
        with pytest.raises(PhysicsIOError):
            eci_to_lvlh(r, v, np.array([10.0, 0, 0]), np.zeros(3), "ECI")


class TestClohessyWiltshire:
    def test_identity_and_semigroup(self):
        n = mean_motion_circular(MU, 7071000.0)
        assert np.abs(cw_state_transition(n, 0.0) - np.eye(6)).max() < 1e-14
        t1, t2 = 1234.5, 4000.0
        p1, p2, p12 = (cw_state_transition(n, t) for t in (t1, t2, t1 + t2))
        assert np.abs(p12 - p2 @ p1).max() < 1e-7

    def test_matches_rk4_of_its_own_equations(self):
        n = mean_motion_circular(MU, 7071000.0)

        def rhs(s):
            x, y, z, xv, yv, zv = s
            return np.array([xv, yv, zv, 3 * n * n * x + 2 * n * yv, -2 * n * xv, -n * n * z])

        rng = np.random.default_rng(11)
        for t in (60.0, 1000.0, 5500.0, 50000.0):
            phi = cw_state_transition(n, t)
            dt = t / 40000
            cols = []
            for j in range(6):
                s = np.zeros(6)
                s[j] = 1.0
                steps = int(round(t / dt))
                h = t / steps
                for _ in range(steps):
                    k1 = rhs(s); k2 = rhs(s + 0.5 * h * k1)
                    k3 = rhs(s + 0.5 * h * k2); k4 = rhs(s + h * k3)
                    s = s + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
                cols.append(s)
            assert np.abs(phi - np.stack(cols, axis=1)).max() < 1e-6

    def test_batched_times_match_scalar_calls(self):
        n = mean_motion_circular(MU, 7071000.0)
        ts = np.array([0.0, 37.0, 5000.0, 62000.0])
        P = cw_state_transition(n, ts)
        assert P.shape == (4, 6, 6)
        for k, t in enumerate(ts):
            assert np.abs(P[k] - cw_state_transition(n, float(t))).max() < 1e-14

    def test_closed_ellipse_condition_and_cross_track_oscillation(self):
        n = mean_motion_circular(MU, 7071000.0)
        T = 2 * math.pi / n
        x0, z0 = 50.0, 7.0
        s0 = np.array([x0, 0.0, z0, 0.0, -2.0 * n * x0, 0.0])
        r1, v1 = propagate_cw(s0[:3], s0[3:], n, T)
        # bounded in the plane, and the cross-track axis oscillates exactly
        assert np.abs(r1[0] - x0).max() < 1e-8
        assert abs(r1[1] - 0.0) < 1e-8
        assert abs(r1[2] - z0 * math.cos(n * T)) < 1e-10
        # the *wrong* classic condition (y_dot0 = -1.5 n x0) drifts: CW y must
        # never be called bounded without the factor 2
        r2, _ = propagate_cw(s0[:3], np.array([0.0, -1.5 * n * x0, 0.0]), n, 20 * T)
        assert abs(r2[1]) > 100 * x0

    def test_superposition(self):
        n = mean_motion_circular(MU, 7071000.0)
        rng = np.random.default_rng(5)
        a = rng.normal(0, 30, 6); b = rng.normal(0, 30, 6)
        t = 900.0
        ra, va = propagate_cw(a[:3], a[3:], n, t)
        rb, vb = propagate_cw(b[:3], b[3:], n, t)
        rsum, vsum = propagate_cw(a[:3] + b[:3], a[3:] + b[3:], n, t)
        assert np.abs(rsum - (ra + rb)).max() < 1e-9
        assert np.abs(vsum - (va + vb)).max() < 1e-12

    def test_batch_states_and_time_series(self):
        n = mean_motion_circular(MU, 7071000.0)
        r0 = np.array([[10.0, 0, 0], [100.0, 0, 0]])
        v0 = np.zeros((2, 3))
        ts = np.linspace(0, 5000, 11)
        # one state, a grid of times -> one trace per time
        r, v = propagate_cw(r0[0], v0[0], n, ts)
        assert r.shape == ts.shape + (3,)
        assert np.abs(r[0] - r0[0]).max() < 1e-12
        # a batch of states, one duration each -> matches scalar calls
        durs = np.array([400.0, 4000.0])
        R, V = propagate_cw(r0, v0, n, durs)
        for k in range(2):
            rs, vs = propagate_cw(r0[k], v0[k], n, float(durs[k]))
            assert np.abs(R[k] - rs).max() < 1e-9
            assert np.abs(V[k] - vs).max() < 1e-9
        # ... and one shared duration matches per-row scalar calls
        R1, _ = propagate_cw(r0, v0, n, 1234.0)
        assert np.abs(R1[1] - propagate_cw(r0[1], v0[1], n, 1234.0)[0]).max() < 1e-12

    def test_guards(self):
        n = mean_motion_circular(MU, 7071000.0)
        with pytest.raises(UnitMismatchError):
            propagate_cw(np.array([5.0e8, 0, 0]), np.zeros(3), n, 100.0)
        with pytest.raises(ValueError):
            propagate_cw(np.array([10.0, 0, 0]), np.zeros(3), 0.0, 100.0)
        with pytest.raises(UnitMismatchError):
            propagate_cw(np.array([10.0, 0, 0]), np.zeros(3), n, 1e12)


class TestPairPropagation:
    def test_initial_state_and_shapes(self):
        rc, vc = circular(7071000.0, 0.4, 0.2)
        rd = rc + np.array([250.0, 0, 0])
        vd = vc + np.array([0.0, 0.02, 0.0])
        times = np.linspace(0.0, 1000.0, 5)
        tr = propagate_pair(rc, vc, rd, vd, times, frame="GCRS", sub_dt=2.0)
        assert tr.chief_r.shape == (1, 5, 3)
        assert tr.rel_lvlh.shape == (1, 5, 6)
        assert np.abs(tr.chief_r[0, 0] - rc).max() == 0.0
        assert np.abs(tr.deputy_v[0, 0] - vd).max() == 0.0
        rho0, rhod0 = eci_to_lvlh(rc, vc, rd - rc, vd - vc, "GCRS")
        assert np.abs(tr.rel_lvlh[0, 0] - np.concatenate((rho0, rhod0))).max() < 1e-9

    def test_converges_at_fourth_order(self):
        rc, vc = kepler_state(12000000.0, 0.08, 1.0, 0.5, 0.9)
        rd = rc + np.array([500.0, 100.0, 10.0])
        times = np.linspace(0.0, 9000.0, 9)
        coarse = propagate_pair(rc, vc, rd, vc.copy(), times, frame="EME2000", sub_dt=60.0)
        fine = propagate_pair(rc, vc, rd, vc.copy(), times, frame="EME2000", sub_dt=30.0)
        err_c = np.abs(coarse.deputy_r - fine.deputy_r).max()
        finer = propagate_pair(rc, vc, rd, vc.copy(), times, frame="EME2000", sub_dt=15.0)
        err_f = np.abs(fine.deputy_r - finer.deputy_r).max()
        assert err_f > 0
        assert 8.0 < err_c / err_f < 32.0

    def test_two_body_conservation(self):
        rc, vc = circular(7071000.0, 0.3, 0.1)
        rd = rc + np.array([100.0, 0, 0])
        times = np.linspace(0.0, 4 * 2 * math.pi / mean_motion_circular(MU, 7071000.0), 401)
        tr = propagate_pair(rc, vc, rd, vc.copy(), times, frame="GCRS")
        r = tr.deputy_r[0]
        v = tr.deputy_v[0]
        E = 0.5 * np.sum(v * v, axis=1) - MU / np.linalg.norm(r, axis=1)
        h = np.cross(r, v)
        assert (E.max() - E.min()) / abs(E[0]) < 1e-8
        assert np.abs(np.diff(h, axis=0)).max() / np.linalg.norm(h[0]) < 1e-8
        # zero-separation pair: deputy is a duplicate chief, they stay merged
        tr2 = propagate_pair(rc, vc, rc.copy(), vc.copy(), times, frame="GCRS")
        assert np.abs(tr2.deputy_r - tr2.chief_r).max() < 1e-9
        assert np.abs(tr2.rel_lvlh).max() < 1e-9

    def test_j2_nodal_drift_matches_secular_theory(self):
        inc = math.radians(51.6)
        a = 7000000.0
        rc, vc = circular(a, inc, 0.0)
        n = mean_motion_circular(MU, a)
        times = np.linspace(0.0, 30 * 2 * math.pi / n, 1201)
        tr = propagate_pair(rc, vc, rc + np.array([10.0, 0, 0]), vc.copy(), times,
                            frame="GCRS", j2=EARTH_J2, sub_dt=(2 * math.pi / n) / 1000)
        h = np.cross(tr.chief_r[0], tr.chief_v[0])
        raan = np.unwrap(np.arctan2(h[:, 1], h[:, 0]))
        ndot = -1.5 * EARTH_J2 * math.sqrt(MU) * EARTH_RE ** 2 / a ** 3.5 * math.cos(inc)
        measured = (raan[-1] - raan[0]) / (times[-1] - times[0])
        # short-period wobble in RAAN is O(J2); the mean drift is the target
        assert abs(measured - ndot) / abs(ndot) < 0.02
        # axisymmetric potential: h_z is conserved even with J2
        assert abs(h[:, 2].max() - h[0, 2]) / abs(h[0, 2]) < 1e-6

    def test_batch_equals_independent_runs(self):
        rcs = np.stack([circular(7071000.0 + k * 1e6, inc, raan)[0]
                        for k, (inc, raan) in enumerate([(0.3, 0.1), (1.2, 5.0)])])
        vcs = np.stack([circular(7071000.0 + k * 1e6, inc, raan)[1]
                        for k, (inc, raan) in enumerate([(0.3, 0.1), (1.2, 5.0)])])
        rds = rcs + np.array([[200.0, 0, 0], [-150.0, 30.0, 0]])
        times = np.linspace(0.0, 3000.0, 7)
        trb = propagate_pair(rcs, vcs, rds, vcs, times, frame="GCRS", sub_dt=5.0)
        assert trb.chief_r.shape == (2, 7, 3)
        for k in range(2):
            single = propagate_pair(rcs[k], vcs[k], rds[k], vcs[k], times, frame="GCRS", sub_dt=5.0)
            assert np.abs(trb.chief_r[k] - single.chief_r[0]).max() < 1e-9

    def test_cw_and_nonlinear_agree_inside_the_envelope(self):
        # The planning-model acceptance tie-in: circular chief, small
        # separation, quarter orbit -- CW and the nonlinear pair must stay
        # within centimetres. (The envelope numbers are the committed
        # benchmark; this is the runtime canary.)
        rc, vc = circular(7071000.0, math.radians(51.6), math.radians(40))
        n = mean_motion_circular(MU, 7071000.0)
        rd = rc + np.array([50.0, 0, 0])
        times = np.linspace(0.0, 0.5 * 2 * math.pi / n, 51)
        tr = propagate_pair(rc, vc, rd, vc.copy(), times, frame="GCRS")
        rho0, rhod0 = eci_to_lvlh(rc, vc, rd - rc, np.zeros(3), "GCRS")
        r_cw, _ = propagate_cw(rho0, rhod0, n, times)
        err = np.linalg.norm(tr.rel_lvlh[0, :, :3] - r_cw, axis=1)
        assert err[-1] < 0.05

    def test_guards(self):
        rc, vc = circular(7071000.0, 0.3, 0.1)
        rd = rc + np.array([100.0, 0, 0])
        times = np.linspace(0.0, 100.0, 3)
        with pytest.raises(PhysicsIOError):
            propagate_pair(rc, vc, rd, vc.copy(), times, frame="ECEF")
        with pytest.raises(PhysicsIOError):
            propagate_pair(rc, vc, rd, vc.copy(), times, frame="ECI")
        with pytest.raises(UnitMismatchError):
            propagate_pair(rc, vc, rd, vc.copy(), times, frame="GCRS", mu=398600.4415)
        with pytest.raises(UnitMismatchError):
            propagate_pair(rc * 1000, vc, rd * 1000, vc.copy(), times, frame="GCRS")
        with pytest.raises(ValueError):
            propagate_pair(rc, vc, rd, vc.copy(), np.array([0.0, 50.0, 40.0]), frame="GCRS")
        with pytest.raises(ValueError):
            propagate_pair(rc, vc, rd, vc.copy(), np.array([0.0]), frame="GCRS")
        with pytest.raises(UnitMismatchError):
            propagate_pair(rc, vc, rd, vc.copy(), np.array([0.0, 1e12]), frame="GCRS")


class TestPhysicsIoContract:
    def test_relative_frames_are_known_in_their_own_family(self):
        assert pio.FRAMES["LVLH"].family == "orbital"
        assert pio.FRAMES["RTN"].family == "orbital"
        assert "orbital" in pio.FRAME_FAMILIES
        assert set(pio.ROTATING_FRAMES) == {"ECEF"}
        assert "LVLH" not in pio.ROTATING_FRAMES

    def test_declared_interface_accepts_lvlh_and_rtn(self):
        assert set(pio.RELATIVE_STATE.frames) == {"LVLH", "RTN"}
        pio.check_frame("LVLH", pio.RELATIVE_STATE.frames)
        pio.check_frame("RTN", pio.RELATIVE_STATE.frames)
        with pytest.raises(PhysicsIOError, match="spelling"):
            pio.check_frame("lvlh", pio.RELATIVE_STATE.frames)

    def test_geocentric_checks_are_not_weakened(self):
        # every geocentric vocabulary still rejects the relative frames
        for spec in pio.PHYSICS_IO.values():
            if "LVLH" not in spec.frames:
                with pytest.raises(PhysicsIOError):
                    spec.check_frame("LVLH")
                with pytest.raises(PhysicsIOError):
                    spec.check_frame("RTN")
        # the family check is what enforces "no rotating/orbital frame here"
        with pytest.raises(pio.FrameMismatchError):
            pio.check_frame_family("LVLH", "inertial")
        with pytest.raises(pio.FrameMismatchError):
            pio.check_frame_family("RTN", "rotating")
        with pytest.raises(PhysicsIOError):
            pio.check_frame("VVLH", pio.RELATIVE_STATE.frames,
                            where="vvlh stays out of the accepted vocabulary")

    def test_ambiguity_hints_unchanged_or_improved(self):
        with pytest.raises(PhysicsIOError, match="'LVLH'"):
            pio.check_frame("VVLH", ("GCRS", "EME2000", "LVLH"))
        for name in ("ECI", "J2000", "ITRF"):
            with pytest.raises(PhysicsIOError):
                pio.check_frame(name, ("GCRS",))

    def test_specs_registered(self):
        assert set(pio.PHYSICS_IO) >= {"orbitopt.relative", "orbitopt.relative.propagation"}
        rel = pio.RELATIVE_STATE
        assert rel.units.length == "m" and rel.centre == "chief"
        assert rel.position_range == (0.0, 4e8)
        pair = pio.RELATIVE_PAIR
        assert set(pair.frames) == {"GCRS", "EME2000"}
        assert pair.centre == "earth"
        r = rel.check_positions(np.array([123.0, -40.0, 2.0]))
        assert r.shape == (3,)
        assert rel.check_positions(np.zeros(3)).shape == (3,)  # contact is legitimate
        with pytest.raises(UnitMismatchError):
            rel.check_positions(np.array([5.0e8, 0, 0]))  # 500 000 km: beyond lunar distance, out of band
