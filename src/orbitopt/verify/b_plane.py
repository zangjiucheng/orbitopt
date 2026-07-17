"""B-plane (impact-parameter plane) geometry for a hyperbolic flyby -- the
standard way real mission designers parameterize and target a close
planetary/lunar encounter.

Why not just target a raw (spacecraft - body) Cartesian miss vector: that
vector's *direction* is only a meaningful, well-conditioned quantity once
the miss is already large compared to the flyby body's local gravitational
influence -- for a close, deep flyby (the useful kind), small changes in the
approach velocity produce disproportionately large changes in exactly where
and which direction the trajectory passes the body (real dynamical
sensitivity near closest approach, not numerical noise). B and v_infinity,
by contrast, are exact invariants of the *unperturbed* two-body arc through
a given (relative position, relative velocity) sample: they depend only on
that arc's conserved specific energy and angular momentum, not on where
along the arc the sample was taken, which keeps them well-conditioned
targets regardless of how close the flyby is.

This module is pure geometry -- it has no opinion on *how* a targeter uses
these quantities (see orbitopt.verify.differential_correction.target_lunar_flyby
for the existing Cartesian-miss-vector corrector, and
orbitopt.viz.mission_timeline's free-return search for a B-plane-informed
outer search layered on top of it, not a replacement for it).
"""
from __future__ import annotations

import numpy as np


def b_plane_state(r_rel_m, v_rel_m_s, mu_m3_s2):
    """Impact-parameter vector B and hyperbolic excess velocity vector
    v_infinity of the osculating two-body hyperbola implied by a relative
    state (r_rel_m, v_rel_m_s) -- e.g. spacecraft relative to the Moon at
    some sample epoch near a lunar flyby.

    Derived from first principles (energy + angular-momentum conservation,
    no true-anomaly/Kepler-equation solve needed):

    * specific energy eps = v^2/2 - mu/r and angular momentum h_vec = r x v
      are conserved along the arc; v_infinity = sqrt(2*eps) is the speed as
      r -> infinity.
    * the eccentricity vector e_vec = (v x h_vec)/mu - r_hat points along
      periapsis, with |e_vec| = e.
    * writing the outgoing asymptote direction as
      cos(theta_inf)*p_hat + sin(theta_inf)*q_hat in the (p_hat=e_hat,
      q_hat=h_hat x p_hat) in-plane basis and substituting the standard
      hyperbolic-asymptote identity cos(theta_inf) = -1/e,
      sin(theta_inf) = +sqrt(e^2-1)/e gives the closed-form v_infinity
      vector below.
    * B = (h_vec x v_infinity_vec) / v_infinity^2 is then the standard
      B-plane construction (perpendicular to v_infinity, magnitude
      h/v_infinity = the impact parameter).

    Returns (b_vec_m, v_inf_vec_m_s), or (None, None) if the local relative
    motion isn't hyperbolic (e<=1) or is degenerate (near-zero angular
    momentum) -- not expected for a real flyby candidate, but guards
    against a stray trial landing on a pathological sample.
    """
    r_rel_m = np.asarray(r_rel_m, dtype=float)
    v_rel_m_s = np.asarray(v_rel_m_s, dtype=float)
    r = np.linalg.norm(r_rel_m)
    v = np.linalg.norm(v_rel_m_s)

    specific_energy = 0.5 * v * v - mu_m3_s2 / r
    if specific_energy <= 0.0:
        return None, None

    h_vec = np.cross(r_rel_m, v_rel_m_s)
    h = np.linalg.norm(h_vec)
    if h < 1.0:
        return None, None  # near-radial relative trajectory; orbital plane undefined

    e_vec = np.cross(v_rel_m_s, h_vec) / mu_m3_s2 - r_rel_m / r
    e = np.linalg.norm(e_vec)
    if e <= 1.0:
        return None, None

    v_inf = np.sqrt(2.0 * specific_energy)
    h_hat = h_vec / h
    v_inf_vec = (v_inf / e**2) * (-e_vec + np.sqrt(e * e - 1.0) * np.cross(h_hat, e_vec))
    b_vec = np.cross(h_vec, v_inf_vec) / (v_inf * v_inf)
    return b_vec, v_inf_vec


def periapsis_distance_m(b_mag_m, v_inf_m_s, mu_m3_s2):
    """Convert a B-plane impact-parameter magnitude + hyperbolic excess
    speed to the corresponding periapsis (closest-approach) distance -- the
    two are related but *not* equal in general (they coincide only in the
    negligible-deflection, e->infinity limit): the exact relation for a
    two-body hyperbola is r_p^2 + 2*a*r_p = b^2 with a = mu/v_inf^2.
    """
    a_mag_m = mu_m3_s2 / (v_inf_m_s * v_inf_m_s)
    return np.sqrt(a_mag_m * a_mag_m + b_mag_m * b_mag_m) - a_mag_m


def b_plane_magnitude_for_periapsis(target_periapsis_m, v_inf_m_s, mu_m3_s2):
    """Inverse of periapsis_distance_m: the B-plane magnitude that yields
    exactly target_periapsis_m periapsis distance, given a (held-fixed)
    hyperbolic excess speed."""
    a_mag_m = mu_m3_s2 / (v_inf_m_s * v_inf_m_s)
    return np.sqrt(target_periapsis_m * (target_periapsis_m + 2.0 * a_mag_m))


def perpendicular_basis(s_hat):
    """Right-handed (T, R) orthonormal basis perpendicular to unit vector
    s_hat (the B-plane's own in-plane axes; S=s_hat, usually the
    v_infinity direction, is the implicit third, normal, axis). Any fixed
    reference vector not parallel to s_hat gives an equally valid,
    consistently-oriented basis -- the +Z/+X swap below only avoids the
    degenerate case where the reference happens to be (near-)parallel to
    s_hat itself.
    """
    s_hat = np.asarray(s_hat, dtype=float)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(s_hat, reference)) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    r_hat = np.cross(s_hat, reference)
    r_hat /= np.linalg.norm(r_hat)
    t_hat = np.cross(r_hat, s_hat)
    return t_hat, r_hat


def b_plane_angle_deg(b_vec_m, v_inf_vec_m_s):
    """B-plane angle (degrees, 0-360) of b_vec within the (T, R) basis
    perpendicular to v_inf_vec -- a single scalar "which side" descriptor
    of a flyby's aim point, holding |B| (and so periapsis distance, at
    fixed v_infinity) fixed. Two flybys with the same periapsis distance
    but different B-plane angle pass the body on geometrically different
    sides/planes -- exactly the degree of freedom a free-return search
    needs to steer the post-flyby trajectory without changing how close it
    passes the Moon.
    """
    v_inf = np.linalg.norm(v_inf_vec_m_s)
    s_hat = np.asarray(v_inf_vec_m_s, dtype=float) / v_inf
    t_hat, r_hat = perpendicular_basis(s_hat)
    b_t = float(np.dot(b_vec_m, t_hat))
    b_r = float(np.dot(b_vec_m, r_hat))
    return float(np.degrees(np.arctan2(b_r, b_t)) % 360.0)
