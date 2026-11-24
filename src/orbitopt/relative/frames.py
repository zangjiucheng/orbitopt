"""ECI <-> LVLH / RTN: the relative-frame kinematics of a spacecraft pair (E12-02).

A relative state is only meaningful *with respect to a chief*: LVLH ("local
vertical, local horizontal", the Hill frame of the Clohessy-Wiltshire
equations) and RTN ("radial-tangential-normal") are two names for the same
orthonormal frame built from the chief's orbit. This module is the one place
that frame is constructed, and the only way a state enters or leaves it.

The axes (right-handed, x-y-z == R-T-N)
---------------------------------------
``x`` (R) radially outward from the central body through the chief
``y`` (T) along-track: in the orbital plane, 90 deg ahead of the radius
``z`` (N) the orbit normal ``h_hat`` (completes the right-handed triple)

This is the frame in which the Clohessy-Wiltshire equations take their
standard form, so it is also the frame :mod:`orbitopt.relative.cw` assumes.
Note it is *not* nadir-x ("VVLH" in some spacecraft body books): that flips
the sign of the CW Coriolis and gravity-gradient terms. ``VVLH`` is rejected
by :func:`orbitopt.physics_io.check_frame` with a hint pointing here.

The velocity transformation
---------------------------
The frame spins with the chief at omega = (r x v) / |r|^2 about ``z`` (the
osculating orbit's angular velocity -- exact for any two-body orbit, J2's
node drift only enters through the chief state itself). So a relative
velocity is *not* just the rotated inertial difference:

    rho     = D (r_d - r_c)
    rho_dot = D (v_d - v_c - omega x (r_d - r_c))

with ``D`` the rows-of-axes matrix. Dropping the omega term is the same
category of error as dropping an omega x r term in an ECEF conversion -- at
700 km it is 1.5 cm/s per metre of cross-range separation, which closes a
rendezvous corridor before anyone notices. Both directions implemented here
carry the term, and the round trip is exact to double-precision round-off
for any chief state (eccentric, inclined -- tests/test_e12_02_relative_dynamics.py pins
that).

Conventions: metres and m/s (RELATIVE_STATE), the chief's inertial state in
any geocentric frame the caller names -- there is no default frame. Inputs
broadcast, so a single chief with a batch of deputy offsets is one call.
"""
from __future__ import annotations

import numpy as np

from orbitopt.physics_io import (
    EARTH_CENTRED_METRES,
    RELATIVE_STATE,
    PhysicsIOError,
    check_frame,
)

#: The inertial (or pseudo-inertial) Earth-centred frames a chief state may be
#: handed in: the transform is a pure vector rotation, so every orbitopt
#: Earth-centred label works identically. Rotating/Earth-fixed labels are
#: rejected -- a chief velocity that carries the Earth's rotation would feed
#: a bogus omega into the orbital-plane construction.
CHIEF_FRAMES = ("GCRS", "EME2000", "ECLIPJ2000", "TEME", "LAUNCH_ECI")


def _as_state(arr, what: str, where: str) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 0 or a.shape[-1:] != (3,):
        raise PhysicsIOError(
            f"{what} must be a (3,) or [...,3] array of vectors, got shape {a.shape}"
            + (f" ({where})" if where else "")
        )
    if not np.all(np.isfinite(a)):
        raise PhysicsIOError(f"{what} contains non-finite values{', ' + where if where else ''}.")
    return a


def lvlh_dcm(r_chief, v_chief, *, frame: str = "LVLH", where: str = "") -> np.ndarray:
    """The (...,3,3) rotation matrix whose **rows** are the LVLH/RTN axes
    (x radial outward, y along-track, z orbit normal) built from the chief's
    inertial state, i.e. ``rho = D @ dr`` maps an Earth-centred separation
    into the chief's orbital frame. ``frame`` selects the label and must be
    one of the interface's relative frames; the geometry is the same for
    ``RTN``.

    Raises for a degenerate chief: zero radius, or a radial (zero angular
    momentum) state, which has no orbital plane and hence no LVLH frame.
    """
    check_frame(frame, RELATIVE_STATE.frames, where=where or "lvlh_dcm")
    r = _as_state(r_chief, "r_chief", where)
    v = _as_state(v_chief, "v_chief", where)
    r, v = np.broadcast_arrays(r, v)
    rn = np.linalg.norm(r, axis=-1)
    if np.any(rn == 0.0):
        raise PhysicsIOError("chief position has zero magnitude; no orbital frame is "
                             "defined at the central body" + (f' ({where})' if where else ""))
    h = np.cross(r, v)
    hn = np.linalg.norm(h, axis=-1)
    # a *numerically* radial state has no plane either: a parallel v leaves
    # ~eps*|r||v| of round-off residue in r x v, not an exact zero
    if np.any(hn <= 1.0e-12 * rn * np.linalg.norm(v, axis=-1)):
        raise PhysicsIOError("chief has zero angular momentum (a radial state): no orbital "
                             "plane, no LVLH frame" + (f" ({where})" if where else ""))
    xhat = r / rn[..., None]
    zhat = h / hn[..., None]
    yhat = np.cross(zhat, xhat)  # unit: z is orthogonal to the radial x by construction
    return np.stack((xhat, yhat, zhat), axis=-2)


def _orbital_rate(r, v):
    """The LVLH frame's angular velocity vector (r x v)/|r|^2 -- about the
    orbit normal at the osculating rate of the radius vector."""
    return np.cross(r, v) / np.sum(r * r, axis=-1, keepdims=True)


def eci_to_lvlh(r_chief, v_chief, dr, dv, frame: str, *, where: str = ""):
    """Map an Earth-centred chief/deputy *difference* state into the chief's
    LVLH/RTN frame.

    ``frame`` is the label of the **inertial frame the difference state is
    given in** (one of :data:`CHIEF_FRAMES` -- no default, same rule as
    everywhere in orbitopt); ``r_chief``/``v_chief`` must be in that same
    frame. Returns ``(rho, rho_dot)``: position [m] and LVLH-frame rate [m/s]
    of deputy relative to chief, in chief-centred metres per
    :data:`orbitopt.physics_io.RELATIVE_STATE`. All inputs broadcast.
    """
    check_frame(frame, CHIEF_FRAMES, where=where or "eci_to_lvlh")
    r = _as_state(r_chief, "r_chief", where)
    v = _as_state(v_chief, "v_chief", where)
    d_r = _as_state(dr, "dr", where)
    d_v = _as_state(dv, "dv", where)
    r, v = np.broadcast_arrays(r, v)
    where = where or "eci_to_lvlh"
    EARTH_CENTRED_METRES.check_positions(r, where=where + ": chief")
    EARTH_CENTRED_METRES.check_velocities(v, where=where + ": chief")
    D = lvlh_dcm(r, v, frame="LVLH", where=where)
    dr_b, dv_b = np.broadcast_arrays(d_r, d_v)
    rho = np.einsum("...ij,...j->...i", D, dr_b)
    rho_dot = np.einsum("...ij,...j->...i", D, dv_b - np.cross(_orbital_rate(r, v), dr_b))
    RELATIVE_STATE.check_positions(rho, where=where or "eci_to_lvlh")
    RELATIVE_STATE.check_velocities(rho_dot, where=where or "eci_to_lvlh")
    return rho, rho_dot


def lvlh_to_eci(r_chief, v_chief, rho, rho_dot, frame: str, *, where: str = ""):
    """Inverse of :func:`eci_to_lvlh`: rebuild the Earth-centred
    chief->deputy difference state in ``frame`` from a LVLH/RTN relative
    state. The round trip ``lvlh_to_eci(..., *eci_to_lvlh(...))`` is exact to
    double-precision round-off for any chief state.
    """
    check_frame(frame, CHIEF_FRAMES, where=where or "lvlh_to_eci")
    r = _as_state(r_chief, "r_chief", where)
    v = _as_state(v_chief, "v_chief", where)
    rho = _as_state(rho, "rho", where)
    rho_dot = _as_state(rho_dot, "rho_dot", where)
    r, v = np.broadcast_arrays(r, v)
    D = lvlh_dcm(r, v, frame="LVLH", where=where)
    dt_b, dvt_b = np.broadcast_arrays(rho, rho_dot)
    dr = np.einsum("...ji,...j->...i", D, dt_b)
    dv = np.einsum("...ji,...j->...i", D, dvt_b) + np.cross(_orbital_rate(r, v), dr)
    return dr, dv
