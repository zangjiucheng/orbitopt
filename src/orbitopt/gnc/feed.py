"""The interface wall between what IS and what is KNOWN (E13-01).

The acceptance rule this module exists to make structural, not aspirational:
**plant truth must never reach guidance.** In a shared-process simulation
the rule is trivially violable by one attribute access, so isolation is not
a comment here -- it is a type boundary:

* truth flows into the run log (:class:`orbitopt.gnc.recording.GncRunLog`).
  The log is an append-only sink: the simulator writes truth, the run log
  and offline evaluators read it back, and *no guidance-side object can be
  constructed from it*.
* guidance is driven through :class:`GuidanceInput`, which accepts exactly
  one record type -- ``Estimate``. Feeding it anything else (a truth sample,
  a raw observation, a command) is a :class:`TypeError` naming the type it
  was handed, because the leak this guards is real, not hypothetical: an
  estimator stub that forwards truth is how a closed-loop demo starts
  looking perfect.
* a :class:`GuidanceLaw` receives a single ``Estimate`` and returns a
  commanded delta-v. It gets no plant handle, no ledger, no clock -- the
  call signature is the whole worldview.

This module deliberately contains no name containing ``truth`` (checked in
``tests/test_e13_01_truth_isolation.py`` by parsing this file's AST): the
guidance-facing surface has no vocabulary for plant state, so
``getattr``-style scavenging finds nothing either. The ledger lives on
purpose -- far from the guidance imports, in
:mod:`orbitopt.gnc.sim` / :mod:`orbitopt.gnc.recording`.
"""
from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable

from .contracts import Estimate, GNCContractError


@runtime_checkable
class GuidanceLaw(Protocol):
    """What the guidance/control side provides: from one navigation estimate,
    a commanded inertial delta-v [m/s] (the zero vector to coast). No plant
    handle in the signature -- a law that wants more is a law that wants to
    cheat."""

    def delta_v(self, estimate: Estimate) -> tuple:
        """Commanded delta-v given the latest estimate (zero vector = coast)."""
        ...


class GuidanceInput:
    """The ONLY object a guidance/control module ever sees of the run: the
    latest navigation estimate per chaser. Structural guarantees:

    * ``push`` accepts ``Estimate`` and nothing else (``TypeError``);
    * it holds estimates only -- there is no attribute, method or key path
      from this object to plant state (its slots are the whole object).
    """

    __slots__ = ("_latest",)

    def __init__(self) -> None:
        self._latest: dict = {}

    def push(self, estimate: Estimate) -> None:
        """Deliver a new navigation estimate. Not an ``Estimate``? That is
        the wall: refused, with the class actually handed in."""
        if not isinstance(estimate, Estimate):
            raise TypeError(
                f"GuidanceInput.push accepts Estimate records only, got "
                f"{type(estimate).__name__} -- guidance reads navigation "
                "estimates, never plant state or raw measurements; a value "
                "that reached this call as something else leaked upstream")
        key = (estimate.relative_state.chaser_id, estimate.relative_state.target_id)
        prev = self._latest.get(key)
        if prev is not None and estimate.t_update_s < prev.t_update_s:
            raise GNCContractError(
                f"GuidanceInput.push: estimate {estimate.record_id} at t_update_s "
                f"{estimate.t_update_s} is older than the current one ({prev.record_id}, "
                f"{prev.t_update_s}) -- out-of-order delivery would silently rewind the "
                "guidance view of the pair")
        self._latest[key] = estimate

    def latest(self, chaser_id: str, target_id: str | None = None) -> Estimate:
        """The newest estimate for one chaser (optionally pinned to one
        target). Raises when the filter has not reported yet -- "no state
        estimate yet" must be an explicit error, never a silent fallback to
        anything more accurate."""
        hits = [k for k in self._latest if k[0] == chaser_id and (target_id is None or k[1] == target_id)]
        if not hits:
            raise GNCContractError(
                f"GuidanceInput has no estimate for chaser {chaser_id!r}"
                + (f" target {target_id!r}" if target_id else "")
                + " -- the navigation filter has not updated yet; coasting is "
                  "a decision, not a fallback value")
        if target_id is None and len({k[1] for k in hits}) > 1:
            raise GNCContractError(
                f"GuidanceInput: chaser {chaser_id!r} has estimates for several targets "
                "-- name the target_id")
        return self._latest[hits[0] if target_id else max(hits, key=lambda k: self._latest[k].t_update_s)]

    def chaser_ids(self) -> tuple:
        """Chaser ids currently held (names, not states)."""
        return tuple(sorted({k[0] for k in self._latest}))

    def __len__(self) -> int:
        return len(self._latest)


def guidance_input_from(estimates) -> GuidanceInput:
    """Build a feed from an iterable of estimates (pushed in order; the feed
    enforces its own rules on the way in -- no privileged constructor)."""
    feed = GuidanceInput()
    for est in estimates:
        feed.push(est)
    return feed


def run_guidance_step(law: GuidanceLaw, feed: GuidanceInput, chaser_id: str,
                      target_id: str | None = None) -> tuple:
    """One control tick, the only sanctioned way to invoke a law: pull the
    latest estimate through the feed (type-checked twice -- the feed, then
    here) and call ``law.delta_v``. Returns ``(estimate, delta_v)`` so the
    harness can record the Command with its provenance."""
    est = feed.latest(chaser_id, target_id)
    if not isinstance(est, Estimate):  # belt: the feed is the wall, this is the padlock
        raise TypeError(f"feed returned {type(est).__name__}, not an Estimate")
    if not isinstance(law, GuidanceLaw):
        raise TypeError(f"guidance law must implement GuidanceLaw.delta_v, got {law!r}")
    dv = law.delta_v(est)
    if isinstance(dv, (str, bytes)) or not hasattr(dv, "__iter__"):
        raise TypeError(f"GuidanceLaw.delta_v must return a 3-vector, got {dv!r}")
    parts = tuple(dv)
    if len(parts) != 3 or not all(isinstance(x, (int, float)) and x == x for x in parts):
        raise TypeError(f"GuidanceLaw.delta_v must return 3 finite numbers, got {dv!r}")
    return est, tuple(float(x) for x in parts)
