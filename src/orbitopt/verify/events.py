"""Event location on a propagated state history: periapsis / closest approach,
sphere-of-influence (SOI) crossings and entry-interface crossings.

The old verification path picked the *saved sample* with the smallest
distance (``find_closest_approach``) or linearly interpolated altitude
between samples (``find_altitude_crossing``). Both results therefore depended
on the propagator's output spacing: a 600 s RK4 step reported an 81,733 km
lunar closest approach for a trajectory whose converged value was 4,379 km
(see docs/engineering-notes.md).

This module removes that dependence by treating each event as the root of a
smooth scalar *event function* g(t) and solving g(t) = 0 with a bracketing
root finder (Brent) on a continuous representation of the trajectory:

* Dense output: a cubic Hermite interpolant per saved interval, built from
  the saved positions *and* velocities (so it is C1 and 4th-order accurate in
  position, unlike linear interpolation between samples).
* Optional local re-propagation (``refine``): once an event is bracketed, the
  bracketing interval is re-propagated from its left-end state with a much
  finer output spacing (by the same high-fidelity propagator) and the root is
  re-solved on that fresh dense output. Repeating this drives the result to
  the propagator's own accuracy, independent of the original sampling.

Event functions used here:

* periapsis / closest approach to a body:  g = (r_sc - r_b) . (v_sc - v_b),
  the (half) derivative of the squared range; a minimum is a -/+ crossing.
* SOI / sphere crossing:                   g = |r_sc - r_b| - R_soi
* entry interface (altitude crossing):     g = |r_sc| - (R_body + h_EI)

Everything here is pure numpy/scipy so it can be tested without tudatpy; body
ephemerides are injected as callables ``body_state_fn(absolute_epoch) ->
(6,)`` (SPICE-backed wrappers live in ``tudat_propagate``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.optimize import brentq

# Signature of a local re-propagator: (state0 (6,), epoch0, duration, max_step)
# -> (epochs (n,), states (n, 6)) with epochs[0] == epoch0, in the same epoch
# convention as the history being refined.
RefineFn = Callable[[np.ndarray, float, float, float], "tuple[np.ndarray, np.ndarray]"]
BodyStateFn = Callable[[float], np.ndarray]


def _zero_body_state(_t):
    return np.zeros(6)


class HermiteDenseOutput:
    """C1 piecewise cubic Hermite interpolant through saved states.

    Position uses the saved velocity as its derivative at each node (so the
    position error is O(h^4) per interval); velocity is the analytical
    derivative of the position cubic (O(h^3)). Epochs must be strictly
    increasing -- duplicate epochs (e.g. at an impulsive-burn seam in
    ``propagate_multi_arc``) are a velocity discontinuity: ``states`` keeps
    the *last* state at a repeated epoch (post-burn, where the next interval
    starts) while ``states_left`` keeps the *first* (pre-burn, where the
    previous interval ends). Each interval's Hermite cubic uses its own
    start state from ``states`` and end state from ``states_left``, so no
    interval ever interpolates toward a velocity from the other side of a
    burn.
    """

    def __init__(self, epochs, states):
        epochs = np.asarray(epochs, dtype=float)
        states = np.asarray(states, dtype=float)
        if epochs.ndim != 1 or states.shape != (epochs.size, 6):
            raise ValueError("epochs must be (n,) and states (n, 6)")
        if epochs.size < 2:
            raise ValueError("need at least two samples for dense output")
        # keep the last of any run of equal epochs
        if np.any(np.diff(epochs) < 0.0):
            raise ValueError("epochs must be non-decreasing")
        keep_last = np.ones(epochs.size, dtype=bool)
        keep_last[:-1] = np.diff(epochs) > 0.0
        keep_first = np.ones(epochs.size, dtype=bool)
        keep_first[1:] = np.diff(epochs) > 0.0
        if np.count_nonzero(keep_last) < 2:
            raise ValueError("need at least two distinct epochs for dense output")
        self.epochs = epochs[keep_last]
        self.states = states[keep_last]  # post-seam state at each node
        self.states_left = states[keep_first]  # pre-seam state at each node

    @property
    def t0(self):
        return float(self.epochs[0])

    @property
    def t1(self):
        return float(self.epochs[-1])

    def __call__(self, t, interval=None) -> np.ndarray:
        """State at ``t``. ``interval`` pins the saved interval used, so an
        endpoint shared with the next interval (an impulsive-burn seam)
        evaluates to the pre-burn state when asked from the left."""
        t = float(t)
        e = self.epochs
        if interval is None:
            i = int(np.clip(np.searchsorted(e, t, side="right") - 1, 0, e.size - 2))
        else:
            i = int(interval)
        ta, tb = e[i], e[i + 1]
        h = tb - ta
        s = (t - ta) / h
        ra, va = self.states[i, :3], self.states[i, 3:]
        rb, vb = self.states_left[i + 1, :3], self.states_left[i + 1, 3:]
        s2, s3 = s * s, s * s * s
        h00 = 2 * s3 - 3 * s2 + 1
        h10 = s3 - 2 * s2 + s
        h01 = -2 * s3 + 3 * s2
        h11 = s3 - s2
        r = h00 * ra + h10 * h * va + h01 * rb + h11 * h * vb
        d00 = (6 * s2 - 6 * s) / h
        d10 = 3 * s2 - 4 * s + 1
        d01 = (-6 * s2 + 6 * s) / h
        d11 = 3 * s2 - 2 * s
        v = d00 * ra + d10 * va + d01 * rb + d11 * vb
        return np.concatenate([r, v])


@dataclass
class Event:
    epoch: float  # same convention as the history's epochs
    state: np.ndarray  # (6,) spacecraft state at the event (history frame)
    direction: int  # +1: g crosses - -> +, -1: + -> -
    value: float  # event-specific scalar (distance km, altitude km, ...)


def _g_samples(epochs, states, g):
    return np.array([g(float(t), s) for t, s in zip(epochs, states)])


def _brackets(a, b, direction):
    """Indices i such that g changes sign from ``a[i]`` (interval start) to
    ``b[i]`` (interval end) in ``direction``."""
    up = (a < 0.0) & (b >= 0.0)
    down = (a > 0.0) & (b <= 0.0)
    if direction > 0:
        mask = up
    elif direction < 0:
        mask = down
    else:
        mask = up | down
    return np.nonzero(mask)[0]


def _solve_on(dense: HermiteDenseOutput, g, ta, tb, xtol, interval=None):
    f = lambda t: g(t, dense(t, interval))  # noqa: E731
    fa, fb = f(ta), f(tb)
    if fa == 0.0:
        return ta
    if fb == 0.0:
        return tb
    if np.sign(fa) == np.sign(fb):
        return None
    return brentq(f, ta, tb, xtol=xtol, rtol=4 * np.finfo(float).eps, maxiter=200)


def locate_events(
    epochs,
    states,
    g: Callable[[float, np.ndarray], float],
    direction: int = 0,
    refine: Optional[RefineFn] = None,
    refine_points: int = 64,
    refine_iterations: int = 2,
    xtol: float = 1e-3,
    first_only: bool = False,
) -> list:
    """Find every root of ``g(t, state)`` along a saved history.

    Sign changes of g between consecutive saved samples bracket the roots;
    each is then solved with Brent's method on the Hermite dense output. With
    ``refine`` given, the bracketing interval is re-propagated from its left
    sample with ``refine_points`` output points (``max_step = interval /
    refine_points``) and re-solved, ``refine_iterations`` times, each time
    shrinking the window to the two refined samples around the root.

    Note: an event whose g goes - -> + -> - entirely between two saved
    samples (e.g. a graze shorter than one output step) is invisible to sign
    bracketing; for closest approach, ``find_closest_approach_event`` also
    checks the sample-wise distance minimum to catch that case.
    """
    epochs = np.asarray(epochs, dtype=float)
    states = np.asarray(states, dtype=float)
    dense = HermiteDenseOutput(epochs, states)
    # g at each interval's own start/end state: across an impulsive-burn seam
    # the end value uses the pre-burn state, so a burn-induced jump in g is
    # never mistaken for a root of the continuous trajectory.
    ga = _g_samples(dense.epochs[:-1], dense.states[:-1], g)
    gb = _g_samples(dense.epochs[1:], dense.states_left[1:], g)
    events = []
    for i in _brackets(ga, gb, direction):
        ev = _locate_in_interval(dense, g, i, ga, refine, refine_points, refine_iterations, xtol)
        if ev is not None:
            events.append(ev)
            if first_only:
                break
    return events


def _locate_in_interval(dense, g, i, gv, refine, refine_points, refine_iterations, xtol):
    ta, tb = float(dense.epochs[i]), float(dense.epochs[i + 1])
    direction = 1 if gv[i] < 0.0 else -1  # gv = interval-start g values
    t_root = _solve_on(dense, g, ta, tb, xtol, i)
    k = i
    if t_root is None:
        return None
    local = dense
    if refine is not None:
        state_a = dense.states[i]
        for _ in range(int(refine_iterations)):
            span = tb - ta
            if span <= xtol:
                break
            r_epochs, r_states = refine(state_a.copy(), ta, span, span / float(refine_points))
            r_epochs = np.asarray(r_epochs, dtype=float)
            r_states = np.asarray(r_states, dtype=float)
            if r_epochs.size < 2:
                break
            cand = HermiteDenseOutput(r_epochs, r_states)
            rg = _g_samples(cand.epochs, cand.states, g)
            idx = _brackets(rg[:-1], rg[1:], direction)
            if idx.size == 0:
                break
            j = int(idx[0])
            ta2, tb2 = float(cand.epochs[j]), float(cand.epochs[j + 1])
            t_new = _solve_on(cand, g, ta2, tb2, xtol, j)
            if t_new is None:
                break
            # commit only a fully successful refinement step, so the returned
            # epoch and state always come from the same dense output
            local, k, t_root = cand, j, t_new
            ta, tb, state_a = ta2, tb2, cand.states[j]
    return Event(epoch=float(t_root), state=local(t_root, k), direction=direction, value=float("nan"))


# --------------------------------------------------------------------------
# Concrete event functions
# --------------------------------------------------------------------------

def range_rate_event(body_state_fn: BodyStateFn, epoch_offset: float = 0.0):
    """g = (r - r_b).(v - v_b): zero at periapsis/apoapsis w.r.t. the body.
    ``epoch_offset`` converts history epochs to the absolute epoch expected
    by ``body_state_fn``."""
    def g(t, s):
        b = np.asarray(body_state_fn(epoch_offset + t), dtype=float)
        return float(np.dot(s[:3] - b[:3], s[3:] - b[3:]))
    return g


def sphere_crossing_event(radius_m: float, body_state_fn: BodyStateFn = _zero_body_state, epoch_offset: float = 0.0):
    """g = |r - r_b| - radius. direction=-1 -> entering, +1 -> leaving."""
    def g(t, s):
        b = np.asarray(body_state_fn(epoch_offset + t), dtype=float)
        return float(np.linalg.norm(s[:3] - b[:3]) - radius_m)
    return g


def find_periapsides(
    epochs, states, body_state_fn: BodyStateFn = _zero_body_state, epoch_offset: float = 0.0,
    refine: Optional[RefineFn] = None, **kw,
) -> list:
    """All periapsis passages (range-rate - -> + crossings) relative to the
    body whose absolute state is ``body_state_fn(epoch_offset + t)`` (default:
    the history's own central body, i.e. the origin). ``value`` is the
    periapsis distance in km."""
    g = range_rate_event(body_state_fn, epoch_offset)
    events = locate_events(epochs, states, g, direction=+1, refine=refine, **kw)
    for ev in events:
        b = np.asarray(body_state_fn(epoch_offset + ev.epoch), dtype=float)
        ev.value = float(np.linalg.norm(ev.state[:3] - b[:3])) / 1000.0
    return events


def find_closest_approach_event(
    epochs, states, body_state_fn: BodyStateFn = _zero_body_state, epoch_offset: float = 0.0,
    refine: Optional[RefineFn] = None, **kw,
) -> Event:
    """Global minimum distance to a body over the history, located by root
    finding on the range rate rather than taken from the nearest sample.

    Candidates are every periapsis event plus the two history end points
    (a monotone approach that is cut off by the end of the window has its
    minimum on the boundary); the sample-wise minimum is also checked and,
    when not already bracketed by a periapsis root (a pass shorter than one
    sample interval), its two neighbouring intervals are searched with the
    dense output directly. ``value`` is the distance in km.
    """
    dense = HermiteDenseOutput(epochs, states)
    e, s = dense.epochs, dense.states

    def dist_at(t, st):
        b = np.asarray(body_state_fn(epoch_offset + t), dtype=float)
        return float(np.linalg.norm(st[:3] - b[:3]))

    periapsides = find_periapsides(epochs, states, body_state_fn, epoch_offset, refine=refine, **kw)
    candidates = list(periapsides)
    for k in (0, e.size - 1):
        candidates.append(Event(float(e[k]), s[k].copy(), 0, float("nan")))

    # Safety net for a sample-wise minimum that sign bracketing missed (only
    # if no located periapsis already lies in its neighbouring intervals --
    # otherwise an unrefined Hermite root could undercut the refined one).
    d_samples = np.array([dist_at(t, st) for t, st in zip(e, s)])
    k = int(np.argmin(d_samples))
    covered = any(e[max(k - 1, 0)] <= p.epoch <= e[min(k + 1, e.size - 1)] for p in periapsides)
    if 0 < k < e.size - 1 and not covered:
        g = range_rate_event(body_state_fn, epoch_offset)
        for i in (k - 1, k):
            t = _solve_on(dense, g, float(e[i]), float(e[i + 1]), kw.get("xtol", 1e-3), i)
            if t is not None:
                candidates.append(Event(float(t), dense(t, i), 1, float("nan")))
        candidates.append(Event(float(e[k]), s[k].copy(), 0, float("nan")))

    best = min(candidates, key=lambda ev: dist_at(ev.epoch, ev.state))
    best.value = dist_at(best.epoch, best.state) / 1000.0
    return best


def find_sphere_crossings(
    epochs, states, radius_km: float, body_state_fn: BodyStateFn = _zero_body_state,
    epoch_offset: float = 0.0, direction: int = 0, refine: Optional[RefineFn] = None, **kw,
) -> list:
    """Crossings of a sphere of ``radius_km`` around a body (SOI entry/exit,
    entry interface, ...). direction=-1 -> inbound only, +1 -> outbound only.
    ``value`` is the distance from the body centre in km (== radius_km)."""
    g = sphere_crossing_event(float(radius_km) * 1000.0, body_state_fn, epoch_offset)
    events = locate_events(epochs, states, g, direction=direction, refine=refine, **kw)
    for ev in events:
        b = np.asarray(body_state_fn(epoch_offset + ev.epoch), dtype=float)
        ev.value = float(np.linalg.norm(ev.state[:3] - b[:3])) / 1000.0
    return events


def soi_radius_km(semi_major_axis_km: float, mu_body: float, mu_primary: float) -> float:
    """Laplace sphere of influence r_SOI = a (mu_b / mu_p)^(2/5)."""
    return float(semi_major_axis_km) * (float(mu_body) / float(mu_primary)) ** 0.4


__all__: Sequence[str] = (
    "HermiteDenseOutput", "Event", "locate_events", "range_rate_event", "sphere_crossing_event",
    "find_periapsides", "find_closest_approach_event", "find_sphere_crossings", "soi_radius_km",
)
