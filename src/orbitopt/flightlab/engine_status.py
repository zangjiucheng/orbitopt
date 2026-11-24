"""Engine sequence of a flown body: which burns it made, on how many engines, at what throttle.

Reads the per-sample ``engines_lit`` column (number of engines lit; the landing burn can shut engines
down, 13 -> 5 -> 3 on the real booster) and the flown ``throttle`` (fraction of the FULL engine set) from a
track's extra columns. A burn is a maximal run of samples with at least one engine lit, so an entry burn
that runs on into the landing burn without a shutdown is ONE burn -- one ignition. Pure and Qt-free.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class EngineBurn:
    index: int                                   # 1-based burn number (= the ignition that started it)
    t_start_s: float
    t_end_s: float
    stages: tuple[tuple[float, int], ...]        # (t_s, engines lit) at the start and at every shutdown step
    throttle_min: float                          # of the engines lit at that moment, 0..1
    throttle_max: float
    propellant_kg: float | None                  # mass spent over the burn, when the track carries mass
    engines_known: bool = True                   # False: the track has no engine-count column (burn timing only)

    @property
    def duration_s(self) -> float:
        return self.t_end_s - self.t_start_s

    @property
    def engines_text(self) -> str:
        """``13 -> 5 -> 3``: the engine count at the start of the burn and at each shutdown step
        (``--`` when the track does not record how many engines were lit)."""
        if not self.engines_known:
            return "\u2014"
        counts: list[int] = []
        for _t, n in self.stages:
            if not counts or counts[-1] != n:
                counts.append(n)
        return " → ".join(str(n) for n in counts)


def engine_burns(t_s: Sequence[float], engines_lit: Sequence[float], throttle: Sequence[float] | None = None,
                 mass_kg: Sequence[float] | None = None, *, engines_known: bool = True) -> tuple[EngineBurn, ...]:
    """The burns of one body from its sampled engine state (``engines_known=False``: ``engines_lit`` is only
    an on/off flag, so the burn timing is real and the engine count is not)."""
    t = [float(x) for x in t_s]
    n = [int(round(float(x))) for x in engines_lit]
    if len(t) != len(n) or not t:
        return ()
    thr = None if throttle is None else [float(x) for x in throttle]
    mass = None if mass_kg is None else [float(x) for x in mass_kg]
    burns: list[EngineBurn] = []
    i = 0
    while i < len(t):
        if n[i] <= 0:
            i += 1
            continue
        j = i
        while j + 1 < len(t) and n[j + 1] > 0:
            j += 1
        stages = [(t[i], n[i])]
        for k in range(i + 1, j + 1):
            if n[k] != n[k - 1]:
                stages.append((t[k], n[k]))
        lo = hi = 0.0
        if thr is not None:
            per_set = [thr[k] / (n[k] / max(n[i:j + 1])) for k in range(i, j + 1) if n[k] > 0]
            lo, hi = (min(per_set), max(per_set)) if per_set else (0.0, 0.0)
        # the burn ends at the first sample with no engine lit (or the last sample of the track)
        t_end = t[j + 1] if j + 1 < len(t) else t[j]
        spent = None
        if mass is not None:
            spent = mass[i] - mass[min(j + 1, len(mass) - 1)]
        burns.append(EngineBurn(len(burns) + 1, t[i], t_end, tuple(stages), lo, hi, spent, engines_known))
        i = j + 1
    return tuple(burns)


def burns_from_extra_columns(t_s: Sequence[float], extra: Mapping[str, Sequence[float]],
                             mass_kg: Sequence[float] | None = None) -> tuple[EngineBurn, ...]:
    """Burns of a contract ``BodyTrack``: from ``engines_lit`` when it carries one (a staged landing burn),
    otherwise from the throttle alone."""
    cols = {k.lower(): v for k, v in extra.items()}
    lit = cols.get("engines_lit")
    if lit is not None:
        return engine_burns(t_s, lit, cols.get("throttle"), mass_kg)
    thr = cols.get("throttle")
    if thr is None:
        return ()
    # no engine-count column: the throttle still says when an engine was lit
    return engine_burns(t_s, [1.0 if float(x) > 0.0 else 0.0 for x in thr], thr, mass_kg, engines_known=False)
