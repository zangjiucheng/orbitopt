"""Powered-gap bridge: the stack across a separation it is still thrusting through (hot staging).

The M-01 staging handoff compares the parent's final sample with the child's first. A backend samples each body on
its own clock, so the two are a grid step apart; the repo bridged that gap with a two-body + J2 coast, which is
right for a cold separation (the engines are off between cutoff and separation). With hot staging (E10-02) the
stack is still thrusting through the gap -- the booster's engines are throttled down but lit, the ship has
ignited -- so a coast leaves out a couple of seconds of 17 m/s^2 and the continuity checks fail by tens of metres
and of m/s (E11-06: 43 m, 43 m/s and 28 t on the Starbase hot-staging ascent).

OpenTsiolkovsky writes no row for the stack at the separation instant (only the ship's, after it), so the
separation state has to be bridged from the stack's last samples. This module does it from the OBSERVED motion:
the acceleration and the propellant flow of the last two intervals, extrapolated linearly across the gap and
integrated. No vehicle model is needed (a model of the engines at altitude would have to reproduce OT's own: the
observed flow here was 15.3 t/s, 25% above what the declared vacuum Isp gives).

The tolerances of a powered gap are not the cold-separation ones, because the bridge is not exact:

    position  1 m      (unchanged)
    velocity  1.0 m/s  (cold: 0.05)
    mass      0.05 %   of the stack (cold: 0.001 kg)

They are TWICE the residual this bridge leaves on the E11-06 Starbase hot-staging separation (0.20 m, 0.486 m/s,
258 kg = 0.011 %) -- ONE separation, so the calibration is one sample; the same bridge tried on the stack's
previous interval (which has no ignition in it) leaves 0.07 m / 0.096 m/s, so a back-test would NOT have predicted
the real gap's error. Both numbers, the method and the basis are written on every handoff document that uses it.
A cold separation never takes this path.
"""
from __future__ import annotations

from typing import Any, Sequence

#: tolerances of a powered gap (see the module docstring for their basis and its limits)
POWERED_POSITION_M = 1.0
POWERED_VELOCITY_M_S = 1.0
POWERED_MASS_FRACTION = 5.0e-4
POWERED_BRIDGE_METHOD = ("powered-gap bridge: linear extrapolation of the observed acceleration and mass flow "
                         "(tolerances 1 m / 1.0 m/s / 0.05 % = 2x the residual measured on the E11-06 Starbase "
                         "hot-staging separation, one sample)")


def is_powered_across_gap(run: Any, parent_body: str) -> bool:
    """True when ``parent_body`` is still firing after its last sample: its main-engine-cutoff event is later than
    the track's final time. A cold separation records MECO at (or before) the last sample, so it is never powered."""
    track = next((t for t in run.tracks if t.body_id == parent_body), None)
    if track is None or len(track.t_s) < 3:
        return False
    t_last = float(track.t_s[-1])
    return any(e.kind == "meco" and e.body_id == parent_body and float(e.t_s) > t_last + 1e-6 for e in run.events)


def powered_extrapolate(t: Sequence[float], r: Sequence[Sequence[float]], v: Sequence[Sequence[float]],
                        m: Sequence[float], gap_s: float) -> tuple[list[float], list[float], float]:
    """Position, velocity and mass ``gap_s`` after the last of three samples ``(t, r, v, m)``, from the observed
    acceleration and mass flow of the last two intervals, each extrapolated linearly (a(t) = a1 + slope (t - mid)).
    Exact for a constant-jerk motion and a linear mass flow."""
    if len(t) != 3 or not (t[0] < t[1] < t[2]) or gap_s <= 0.0:
        raise ValueError("powered_extrapolate needs three increasing samples and a positive gap")
    g = float(gap_s)
    d1, d0 = t[2] - t[1], t[1] - t[0]
    mid1, mid0 = 0.5 * (t[1] + t[2]), 0.5 * (t[0] + t[1])
    tau0 = t[2] - mid1
    k = 1.0 / (mid1 - mid0)
    v_out, r_out = [], []
    for i in range(3):
        a1 = (v[2][i] - v[1][i]) / d1
        a0 = (v[1][i] - v[0][i]) / d0
        slope = (a1 - a0) * k
        v_out.append(float(v[2][i]) + a1 * g + slope * (g * g / 2.0 + tau0 * g))
        r_out.append(float(r[2][i]) + float(v[2][i]) * g + a1 * g * g / 2.0 + slope * (g ** 3 / 6.0 + tau0 * g * g / 2.0))
    f1 = (m[1] - m[2]) / d1
    f0 = (m[0] - m[1]) / d0
    fs = (f1 - f0) * k
    m_out = float(m[2]) - (f1 * g + fs * (g * g / 2.0 + tau0 * g))
    return r_out, v_out, m_out


def powered_tolerances(base_position_m: float, base_velocity_m_s: float) -> tuple[float, float]:
    """(position, velocity) tolerances of a powered gap: never tighter than the cold-separation ones."""
    return max(float(base_position_m), POWERED_POSITION_M), max(float(base_velocity_m_s), POWERED_VELOCITY_M_S)


def powered_mass_tolerance_kg(stack_mass_kg: float, base_kg: float) -> float:
    """Mass tolerance of a powered gap: 0.05 % of the stack, never tighter than the cold one."""
    return max(float(base_kg), POWERED_MASS_FRACTION * abs(float(stack_mass_kg)))


#: longest gap this bridge will extrapolate over; a longer one is not a separation transient
MAX_POWERED_GAP_S = 10.0


def powered_parent_state(run: Any, parent_body: str, t_sep_s: float) -> dict | None:
    """The stack's state at the separation instant ``t_sep_s`` when it is still thrusting across the gap, else None.

    Returns ``{state, gap_s, mass_kg, method, tolerances: {position_m, velocity_m_s, mass_fraction}}``; ``state`` is
    a GCRS :class:`~orbitopt.mission.contract.HandoffState` at the separation epoch carrying the bridge on its
    metadata. Cold separations (and gaps beyond :data:`MAX_POWERED_GAP_S`) return None."""
    if not is_powered_across_gap(run, parent_body):
        return None
    from orbitopt.mission.ascent import epoch_at_ascent_time
    from orbitopt.mission.two_stage_leo import _sample_state

    track = next(t for t in run.tracks if t.body_id == parent_body)
    gap = float(t_sep_s) - float(track.t_s[-1])
    if not (0.0 < gap <= MAX_POWERED_GAP_S):
        return None
    cfg = {"body_id": track.body_id, "parent_body_id": track.parent_body_id or ""}
    n = len(track.t_s)
    states = [_sample_state(track, i, cfg) for i in (n - 3, n - 2, n - 1)]
    r, v, m = powered_extrapolate([float(track.t_s[i]) for i in (n - 3, n - 2, n - 1)],
                                  [s.r_m for s in states], [s.v_m_s for s in states],
                                  [s.mass_kg for s in states], gap)
    from dataclasses import replace

    last = states[-1]
    state = replace(last, epoch=epoch_at_ascent_time(track.epoch_utc, float(t_sep_s), "UTC"),
                    r_m=tuple(r), v_m_s=tuple(v), mass_kg=m,
                    metadata={**last.metadata, "t_s": float(t_sep_s), "staging_bridge": POWERED_BRIDGE_METHOD,
                              "bridge_gap_s": gap})
    return {"state": state, "gap_s": gap, "mass_kg": m, "method": POWERED_BRIDGE_METHOD,
            "tolerances": {"position_m": POWERED_POSITION_M, "velocity_m_s": POWERED_VELOCITY_M_S,
                           "mass_fraction": POWERED_MASS_FRACTION}}
