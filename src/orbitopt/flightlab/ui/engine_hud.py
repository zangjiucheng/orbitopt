"""Engine HUD in the style of the SpaceX webcast: a circular map of the booster's engines (lit ones white,
shut-down ones dim) inside a throttle arc, next to a globe that shows the vehicle's orientation against north.

The geometry and the data lookups are plain functions (no Qt) so they can be tested; :class:`EngineHud` only
paints. The engine map follows the real Super Heavy layout -- 3 centre engines, a ring of 10, an outer ring of
20 -- and lights engines in the order they are last to shut down on the real vehicle: the centre three, then the
inner ring, then the outer ring (the landing burn's 13 -> 5 -> 3 is therefore 3 + 10, then 3 + 2, then 3).

What the data supports, and what it does not:
  * speed and propellant come from the telemetry series (propellant is the recorded value when the run has one,
    otherwise mass minus the declared dry mass, and the series label says which);
  * engines lit / throttle come from the flown track (``engines_lit`` when the run staged its engines, otherwise
    the throttle column says only whether an engine was lit, and the count is shown as not recorded);
  * the orientation is the thrust direction while an engine is lit when the track recorded it (thrust acts along
    the nose, so a landing burn reads vertical); otherwise the flight-path orientation from position and velocity
    (nose forward while ascending, tail first while falling) -- the HUD says which.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

#: Super Heavy: 3 centre engines, 10 in the inner ring, 20 in the outer ring.
RINGS = (3, 10, 20)
RING_RADII = (0.13, 0.40, 0.76)           # fraction of the map radius
ENGINE_RADIUS = 0.085                     # fraction of the map radius
MIN_THROTTLE = 0.4                        # Raptor floor, drawn as a notch on the arc


# ------------------------------------------------------------------ engine map

def engine_layout(total: int = 33) -> tuple[tuple[float, float, int], ...]:
    """(x, y, ring) of every engine, unit-circle coordinates, y up. Rings fill 3 -> 10 -> 20; any engines beyond
    33 go on the outer ring."""
    total = max(0, int(total))
    counts = []
    left = total
    for i, cap in enumerate(RINGS):
        n = left if i == len(RINGS) - 1 else min(cap, left)
        counts.append(n)
        left -= n
    out: list[tuple[float, float, int]] = []
    for ring, (n, radius) in enumerate(zip(counts, RING_RADII)):
        offset = math.pi / 2.0 if ring != 1 else math.pi / 2.0 + math.pi / max(n, 1)
        for k in range(n):
            a = offset + 2.0 * math.pi * k / max(n, 1)
            out.append((radius * math.cos(a), radius * math.sin(a), ring))
    return tuple(out)


def _spread(n: int) -> list[int]:
    """0..n-1 ordered so that any prefix is spread evenly around a ring (golden-ratio stepping)."""
    seen: list[int] = []
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    k = 0
    while len(seen) < n:
        idx = int((k * phi % 1.0) * n)
        if idx not in seen:
            seen.append(idx)
        k += 1
        if k > 100 * max(n, 1):
            seen += [i for i in range(n) if i not in seen]
    return seen


def lit_order(total: int = 33) -> tuple[int, ...]:
    """Indices into :func:`engine_layout` in the order engines are lit (and reverse of the order they are shut):
    centre first, then the inner ring spread evenly, then the outer ring spread evenly."""
    layout = engine_layout(total)
    order: list[int] = []
    for ring in range(len(RINGS)):
        members = [i for i, (_x, _y, r) in enumerate(layout) if r == ring]
        order.extend(members[j] for j in _spread(len(members)))
    return tuple(order)


def lit_engines(total: int, lit: int) -> frozenset[int]:
    """Which engine indices are lit when ``lit`` of ``total`` are."""
    lit = max(0, min(int(lit), int(total)))
    return frozenset(lit_order(total)[:lit])


# ------------------------------------------------------------------ orientation

@dataclass(frozen=True)
class Orientation:
    heading_deg: float        # heading of the nose, from north, clockwise
    elevation_deg: float      # flight-path elevation above the local horizon
    nose_elevation_deg: float  # where the nose points (nose forward when rising, tail first when falling)
    nose_enu: tuple = (0.0, 0.0, 1.0)   # unit vector of the nose in the local east / north / up frame


def nose_from_thrust(heading_deg: float, elevation_deg: float) -> tuple[float, float, float]:
    """Thrust acts along the nose: its (heading, elevation) as a unit vector in east / north / up."""
    h, e = math.radians(heading_deg), math.radians(elevation_deg)
    return (math.sin(h) * math.cos(e), math.cos(h) * math.cos(e), math.sin(e))


def orientation_from_nose(nose: Sequence[float], path_elevation_deg: float) -> Orientation:
    n = [float(c) for c in nose]
    ln = math.sqrt(sum(c * c for c in n)) or 1.0
    n = [c / ln for c in n]
    heading = math.degrees(math.atan2(n[0], n[1])) % 360.0
    return Orientation(heading, float(path_elevation_deg), math.degrees(math.asin(max(-1.0, min(1.0, n[2])))),
                       (n[0], n[1], n[2]))


def slerp(a: Sequence[float], b: Sequence[float], w: float) -> tuple[float, float, float]:
    """Unit vector a turning into b along the great circle, w in 0..1."""
    dot = max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))
    ang = math.acos(dot)
    if ang < 1e-9:
        return (float(a[0]), float(a[1]), float(a[2]))
    if ang > math.pi - 1e-3:                                    # opposite: turn through a perpendicular, not an undefined arc
        helper = (0.0, 0.0, 1.0) if abs(a[2]) < 0.9 else (1.0, 0.0, 0.0)
        perp = (a[1] * helper[2] - a[2] * helper[1], a[2] * helper[0] - a[0] * helper[2], a[0] * helper[1] - a[1] * helper[0])
        ln = math.sqrt(sum(c * c for c in perp)) or 1.0
        perp = tuple(c / ln for c in perp)
        return slerp(a, perp, 2.0 * w) if w < 0.5 else slerp(perp, b, 2.0 * w - 1.0)
    sa, sb = math.sin((1.0 - w) * ang) / math.sin(ang), math.sin(w * ang) / math.sin(ang)
    return tuple(sa * x + sb * y for x, y in zip(a, b))                           # type: ignore[return-value]


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


#: the vehicle turns between its burn attitude and its coast attitude over this long, either side of a burn
ATTITUDE_BLEND_S = 12.0
#: inside a burn the displayed attitude is a weighted average of the recorded thrust directions this far either side
BURN_SMOOTH_S = 2.0


EARTH_RATE_RAD_S = 7.2921159e-5
#: below this Earth-relative speed the vehicle is standing on the pad (or hovering): it points straight up
STANDING_SPEED_M_S = 30.0


def flight_path_orientation(position: Sequence[float], velocity: Sequence[float], *,
                            inertial: bool = False) -> Orientation | None:
    """Heading and elevation of the Earth-RELATIVE velocity against the local horizon at ``position``. The local
    frame comes from the position and the +z (spin) axis, so it works for ECEF and (to the precession) GCRS alike.
    In an inertial frame (``inertial=True``) Earth's rotation, omega x r, is removed first: otherwise a rocket
    standing on the pad would 'fly' east at 400 m/s and be drawn lying down. Under ``STANDING_SPEED_M_S`` the
    vehicle stands on its engines: straight up."""
    r = [float(c) for c in position]
    v = [float(c) for c in velocity]
    if inertial:
        v = [v[0] + EARTH_RATE_RAD_S * r[1], v[1] - EARTH_RATE_RAD_S * r[0], v[2]]      # v - omega x r
    rn = math.sqrt(sum(c * c for c in r))
    vn = math.sqrt(sum(c * c for c in v))
    if rn < 1.0:
        return None
    if vn < STANDING_SPEED_M_S:
        return Orientation(0.0, 90.0, 90.0, (0.0, 0.0, 1.0))
    up = [c / rn for c in r]
    east = [-up[1], up[0], 0.0]                       # z-hat x up
    en = math.sqrt(east[0] ** 2 + east[1] ** 2)
    if en < 1e-9:
        return None
    east = [c / en for c in east]
    north = [up[1] * east[2] - up[2] * east[1], up[2] * east[0] - up[0] * east[2], up[0] * east[1] - up[1] * east[0]]
    ve = sum(a * b for a, b in zip(v, east))
    vnorth = sum(a * b for a, b in zip(v, north))
    vu = sum(a * b for a, b in zip(v, up))
    heading = math.degrees(math.atan2(ve, vnorth)) % 360.0
    elevation = math.degrees(math.atan2(vu, math.hypot(ve, vnorth)))
    nose = elevation if elevation >= 0.0 else -elevation       # tail first while falling: the nose points up-range
    vhat = (ve / vn, vnorth / vn, vu / vn)
    nose_enu = vhat if elevation >= 0.0 else (-vhat[0], -vhat[1], -vhat[2])     # tail first while falling
    return Orientation(heading, elevation, nose, nose_enu)


# ------------------------------------------------------------------ data at a time

@dataclass(frozen=True)
class EngineState:
    t_s: float
    total: int
    lit: int | None               # None: the track does not record how many engines were lit
    on: bool                      # at least one engine firing
    throttle: float               # of the lit engines, 0..1
    orientation: Orientation | None
    burn_index: int | None = None
    count_known: bool = True
    throttle_est: bool = False    # the throttle is estimated from the mass flow (the track records none)


def _step_at(t: Sequence[float], values: Sequence[float], at: float) -> float | None:
    """Value of the last sample at or before ``at`` (a step), None before the first."""
    best = None
    for tt, vv in zip(t, values):
        if tt <= at + 1e-9:
            best = vv
        else:
            break
    return best


def _lerp_at(t: Sequence[float], values: Sequence[float], at: float) -> float | None:
    if not t or at < t[0] - 1e-9 or at > t[-1] + 1e-9:
        return None
    lo, hi = 0, len(t) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if t[mid] <= at:
            lo = mid
        else:
            hi = mid
    if t[hi] == t[lo]:
        return float(values[lo])
    f = (at - t[lo]) / (t[hi] - t[lo])
    return float(values[lo]) + f * (float(values[hi]) - float(values[lo]))


def engine_state_at(t_s: Sequence[float], extra: Mapping[str, Sequence[float]],
                    position_m: Sequence[Sequence[float]], velocity_m_s: Sequence[Sequence[float]],
                    at: float, *, total: int = 33, burns: Sequence = (), inertial: bool = False) -> EngineState | None:
    """The HUD's inputs at time ``at`` (None outside the track)."""
    t = [float(x) for x in t_s]
    if not t or at < t[0] - 1e-9 or at > t[-1] + 1e-9:
        return None
    cols = {k.lower(): v for k, v in extra.items()}
    thr_full = _step_at(t, cols["throttle"], at) if "throttle" in cols else None
    lit_raw = _step_at(t, cols["engines_lit"], at) if "engines_lit" in cols else None
    known = lit_raw is not None
    if known:
        lit = int(round(float(lit_raw)))
        on = lit > 0
    else:
        on = thr_full is not None and float(thr_full) > 0.0
        lit = None
    # throttle of the lit engines: the flown throttle is a fraction of the FULL set
    if thr_full is None or not on:
        throttle = 0.0
    elif known and lit:
        throttle = min(1.0, float(thr_full) / max(lit / max(max(float(x) for x in cols["engines_lit"]), 1.0), 1e-9))
    else:
        throttle = min(1.0, float(thr_full))
    k = min(range(len(t)), key=lambda i: abs(t[i] - at))
    orient = (flight_path_orientation(position_m[k], velocity_m_s[k], inertial=inertial)
              if position_m and velocity_m_s else None)
    burn = None
    for b in burns:
        if b.t_start_s - 1e-9 <= at <= b.t_end_s + 1e-9:
            burn = b.index
    return EngineState(float(at), int(total), lit, on, throttle, orient, burn, known)


# ------------------------------------------------------------------ the row's geometry

#: half-extents, in units of the ring radius r, of the four things in a row: speed gauge | engine map (its outer
#: ring is the widest part) | orientation globe | propellant column with its label
ROW_HALF_EXTENTS = (1.0, 1.1, 0.95, 0.9)
ROW_MARGIN = 10.0
ROW_MIN_GAP = 16.0


def row_layout(width: float, height: float, label_h: float = 30.0) -> tuple[float, tuple[float, ...], float]:
    """(r, centres x, gap): the ring radius and the four centres of a vehicle row. ``r`` is limited by the height
    AND by the width, so that however narrow the panel is there stays at least ``ROW_MIN_GAP`` of clear space
    between neighbouring things (the rings used to touch in a narrow panel); spare width widens the gaps."""
    need = 2.0 * sum(ROW_HALF_EXTENTS)                       # in units of r
    r_h = (height - label_h) / 2.0 - 14.0
    r_w = (width - 2.0 * ROW_MARGIN - 3.0 * ROW_MIN_GAP) / need
    r = max(12.0, min(r_h, r_w))
    gap = max(ROW_MIN_GAP, (width - 2.0 * ROW_MARGIN - need * r) / 3.0)
    xs: list[float] = []
    x = ROW_MARGIN
    for half in ROW_HALF_EXTENTS:
        xs.append(x + half * r)
        x += 2.0 * half * r + gap
    return r, tuple(xs), gap


# ------------------------------------------------------------------ two vehicles, one readout each

@dataclass(frozen=True)
class VehicleSpec:
    name: str
    kind: str                 # "superheavy" | "starship" | "generic"
    total: int                # engines in the map
    bodies: tuple[str, ...]   # track body ids to follow, in preference order, as the flight goes on
    attached: tuple[str, ...] = ()   # before its own track exists the vehicle rides on these (the stack)
    observed_only: bool = False


SUPER_HEAVY = VehicleSpec("SUPER HEAVY", "superheavy", 33, ("booster", "stage1", "stage1_debris"))
STARSHIP = VehicleSpec("STARSHIP", "starship", 6, ("stage2",), ("stage1",))


def stage_specs(tracks: Mapping[str, object], vehicle: Mapping | None) -> dict[int, VehicleSpec]:
    """Use the stage tracks and matching vehicle document to build the HUD rows."""
    vehicle = vehicle or {}
    starship_class = str(vehicle.get("name", "")).startswith("starship-class")
    stages = vehicle.get("stages") or ()
    numbers = {int(body[5:]) for body in tracks if body.startswith("stage") and body[5:].isdigit()}
    if "booster" in tracks or "stage1_debris" in tracks:
        numbers.add(1)
    specs = {}
    for number in sorted(numbers - {0}):
        track_id = f"stage{number}"
        bodies = ("booster", track_id, "stage1_debris") if number == 1 else (track_id,)
        attached = (f"stage{number - 1}",) if number > 1 else ()
        declared = stages[number - 1] if number <= len(stages) else {}
        engine = declared.get("engine") or {}
        count = declared.get("engine_count", engine.get("count"))
        peak = max((int(value) for body in bodies if body in tracks
                    for key, values in tracks[body].extra_columns.items() if key.lower() == "engines_lit"
                    for value in values if math.isfinite(value) and value >= 0), default=0)
        layout = {1: SUPER_HEAVY, 2: STARSHIP}.get(number) if starship_class else None
        total = count if type(count) is int and count > 0 else layout.total if layout else 0
        if layout is not None and total == layout.total and peak <= total:
            specs[number] = layout
        else:
            name = "BOOSTER / STAGE 1" if number == 1 and "booster" in tracks else f"STAGE {number}"
            specs[number] = VehicleSpec(name, "generic", max(total, peak), bodies, attached, peak > total)
    return specs


def starship_layout() -> tuple[tuple[float, float, float, str], ...]:
    """(x, y, radius, kind) of the six Starship engines on the unit circle: three big vacuum Raptors at the
    corners of a triangle and three small sea-level ones in the middle, as in the webcast."""
    out = []
    for k in range(3):
        a = math.pi / 2.0 + 2.0 * math.pi * k / 3.0
        out.append((0.52 * math.cos(a), 0.52 * math.sin(a), 0.27, "rvac"))
    for k in range(3):
        a = -math.pi / 2.0 + 2.0 * math.pi * k / 3.0
        out.append((0.13 * math.cos(a), 0.13 * math.sin(a), 0.09, "sl"))
    return tuple(out)


@dataclass(frozen=True)
class VehicleReadout:
    name: str
    body_id: str | None
    engines: EngineState | None
    speed_kmh: float | None
    speed_frac: float                      # of the fastest the vehicle goes in this run
    propellant_t: float | None
    propellant_frac: float | None          # of what it carried at the start of its track
    propellant_label: str                  # "recorded" | "est." | ""
    pointing: str                          # "thrust" | "flight path" | ""
    attached: bool = False                 # still part of the stack: no engines of its own are firing yet


def _series_at(points: Sequence[tuple[float, float]], at: float) -> float | None:
    return _lerp_at([p[0] for p in points], [p[1] for p in points], at) if points else None


def _is_inertial(track) -> bool:
    """Contract tracks are inertial (GCRS or the non-rotating launch frame) unless they say ECEF."""
    return str(getattr(track, "frame", "GCRS")).upper() not in ("ECEF", "ITRS")


def earth_relative_speed_series(track) -> list[tuple[float, float]]:
    """[(t, km/s)] of the speed against the ground: the inertial velocity minus omega x r on an inertial track,
    so a rocket on the pad reads 0 (the telemetry 'speed' channel is the inertial speed, ~0.47 km/s there)."""
    inertial = _is_inertial(track)
    out = []
    for t, r, v in zip(track.t_s, track.position_m, track.velocity_m_s):
        vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
        if inertial:
            vx, vy = vx + EARTH_RATE_RAD_S * float(r[1]), vy - EARTH_RATE_RAD_S * float(r[0])
        out.append((float(t), math.sqrt(vx * vx + vy * vy + vz * vz) / 1000.0))
    return out


def mass_flow_series(track) -> list[tuple[float, float]]:
    """[(t, kg/s)] of the propellant burn rate from the mass history (the engines' throttle on a track that
    records none: OpenTsiolkovsky tracks carry the mass but not the throttle)."""
    t, m = [float(x) for x in track.t_s], [float(x) for x in track.mass_kg]
    out = [(t[0], 0.0)] if t else []
    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]
        out.append((t[k], max(0.0, (m[k - 1] - m[k]) / dt) if dt > 0 else 0.0))
    return out


def pick_track(tracks: Mapping[str, object], bodies: Sequence[str], at: float):
    """The first of ``bodies`` whose track covers ``at`` (the booster after separation, stage 1 before)."""
    for b in bodies:
        tr = tracks.get(b)
        if tr is not None and tr.t_s and tr.t_s[0] - 1e-9 <= at <= tr.t_s[-1] + 1e-9:
            return tr
    return None


def _on_from_mass(t: Sequence[float], mass: Sequence[float], at: float) -> bool:
    """No throttle column: an engine is lit when the mass is falling around ``at``."""
    k = max(1, min(range(len(t)), key=lambda i: abs(t[i] - at)))
    return float(mass[k - 1]) - float(mass[k]) > 1e-3 * max(abs(float(mass[k])), 1.0) * 1e-3


def readout_at(spec: VehicleSpec, tracks: Mapping[str, object], speed: Mapping[str, Sequence],
               propellant: Mapping[str, tuple[str, Sequence]], at: float, burns: Sequence = (),
               flow: Mapping[str, Sequence] | None = None) -> VehicleReadout:
    """Everything one vehicle's HUD row shows at time ``at``. ``speed`` maps body id -> [(t, km/s)] and
    ``propellant`` maps body id -> (label, [(t, tonnes)])."""
    tr = pick_track(tracks, spec.bodies, at)
    if tr is None:
        carrier = pick_track(tracks, spec.attached, at) if spec.attached else None
        if carrier is None:
            return VehicleReadout(spec.name, None, None, None, 0.0, None, None, "", "")
        # still in the stack: the stack's speed and pointing, engines off, the propellant it already carries
        sp = speed.get(carrier.body_id)
        kmh = None if not sp or _series_at(sp, at) is None else _series_at(sp, at) * 3600.0
        top = max((v for _t, v in sp), default=0.0) * 3600.0 if sp else 0.0
        label, pts = propellant.get(spec.bodies[0], ("", ()))
        prop = _series_at(pts, at) if pts else None
        pfrac = None if prop is None or not pts or pts[0][1] <= 0.0 else max(0.0, min(1.0, prop / pts[0][1]))
        kk = min(range(len(carrier.t_s)), key=lambda i: abs(carrier.t_s[i] - at))
        orient = flight_path_orientation(carrier.position_m[kk], carrier.velocity_m_s[kk],
                                         inertial=_is_inertial(carrier))
        st = EngineState(float(at), spec.total, 0, False, 0.0, orient, None, True)
        return VehicleReadout(spec.name, spec.bodies[0], st, kmh, 0.0 if kmh is None or top <= 0 else min(1.0, kmh / top),
                              prop, pfrac, label, "flight path" if orient else "", True)
    st = engine_state_at(tr.t_s, tr.extra_columns, tr.position_m, tr.velocity_m_s, at,
                         total=spec.total, burns=burns if tr.body_id == "booster" else (),
                         inertial=_is_inertial(tr))
    cols = {k.lower(): v for k, v in tr.extra_columns.items()}
    pointing = "flight path" if st is not None and st.orientation is not None else ""
    if st is not None and "throttle" not in cols and "engines_lit" not in cols:
        fl = (flow or {}).get(tr.body_id)
        if fl:
            top = max(v for _t, v in fl) or 1.0
            # the flow at sample k is the burn over the interval ENDING at k: for a time between samples it is the next
            # sample's value (the last one, at the very end of the track)
            now = next((v for tt, v in fl if tt >= at - 1e-9), fl[-1][1])
            frac = 0.0 if now is None else max(0.0, min(1.0, now / top))
            on = frac > 0.02                                    # burning: the mass is draining
            st = EngineState(st.t_s, st.total, None, on, frac if on else 0.0, st.orientation, None, False, True)
        else:
            on = _on_from_mass(tr.t_s, tr.mass_kg, at)           # nothing but the mass says an engine burns
            st = EngineState(st.t_s, st.total, None, on, 0.0, st.orientation, None, False)
    # Attitude. Under power the vehicle points where the thrust points (thrust acts along the nose). Between burns it
    # is engines-first along the path after the first burn (the booster flips once after boostback), nose-forward
    # before it; and it TURNS between the two -- over ATTITUDE_BLEND_S after a burn and again before the next --
    # because the real attitude is continuous. Without thrust data the nose simply leads the path.
    base = st.orientation if st is not None else None
    if st is not None and base is not None:
        vhat = base.nose_enu if base.elevation_deg >= 0 else tuple(-c for c in base.nose_enu)
        nose = tuple(vhat)
        if "thrust_elev_deg" in cols and "thrust_heading_deg" in cols:
            lit_col = cols.get("engines_lit") or cols.get("throttle")
            times = [float(x) for x in tr.t_s]
            lit_idx = [i for i, v in enumerate(lit_col) if float(v) > 0.0] if lit_col is not None else []
            if lit_idx:
                def n_thr(i):
                    return nose_from_thrust(float(cols["thrust_heading_deg"][i]), float(cols["thrust_elev_deg"][i]))

                before = [i for i in lit_idx if times[i] <= at + 1e-9]
                after = [i for i in lit_idx if times[i] >= at - 1e-9]
                if st.on and before:
                    i0 = before[-1]
                    # inside a burn: a centred, triangular-weighted average of the recorded thrust directions of
                    # THIS burn within +-BURN_SMOOTH_S. The recorded direction swings between steering laws (the
                    # entry burn's, then the landing law's) and between 2 s samples; the display must not snap.
                    lit_set = set(lit_idx)
                    lo = hi = i0
                    while lo - 1 in lit_set:
                        lo -= 1
                    while hi + 1 in lit_set:
                        hi += 1
                    acc, wsum = [0.0, 0.0, 0.0], 0.0
                    for i in range(lo, hi + 1):
                        wgt = 1.0 - abs(times[i] - at) / BURN_SMOOTH_S
                        if wgt > 0.0:
                            nv = n_thr(i)
                            acc = [acc[0] + wgt * nv[0], acc[1] + wgt * nv[1], acc[2] + wgt * nv[2]]
                            wsum += wgt
                    if wsum <= 0.0:
                        nose = n_thr(i0)
                    else:
                        ln = math.sqrt(sum(c * c for c in acc))
                        nose = tuple(c / ln for c in acc) if ln > 1e-9 else n_thr(i0)
                    pointing = "thrust"
                else:
                    prev_i = before[-1] if before else None
                    next_i = after[0] if after else None
                    nose = tuple(-c for c in vhat) if prev_i is not None else vhat      # engines first once it has burned
                    w_out = 0.0
                    if prev_i is not None:
                        w_out = smoothstep(1.0 - (at - times[prev_i]) / ATTITUDE_BLEND_S)
                        nose = slerp(nose, n_thr(prev_i), w_out)
                    w_in = 0.0
                    if next_i is not None:
                        w_in = smoothstep(1.0 - (times[next_i] - at) / ATTITUDE_BLEND_S)
                        nose = slerp(nose, n_thr(next_i), w_in)
                    pointing = "thrust" if max(w_out, w_in) >= 0.5 else "flight path"
        st = EngineState(st.t_s, st.total, st.lit, st.on, st.throttle,
                         orientation_from_nose(nose, base.elevation_deg), st.burn_index, st.count_known, st.throttle_est)

    sp = speed.get(tr.body_id)
    kmh = None if not sp else (None if _series_at(sp, at) is None else _series_at(sp, at) * 3600.0)
    top = max((v for _t, v in sp), default=0.0) * 3600.0 if sp else 0.0
    frac = 0.0 if kmh is None or top <= 0.0 else max(0.0, min(1.0, kmh / top))
    label, pts = propellant.get(tr.body_id, ("", ()))
    prop = _series_at(pts, at) if pts else None
    start = pts[0][1] if pts else 0.0
    pfrac = None if prop is None or start <= 0.0 else max(0.0, min(1.0, prop / start))
    return VehicleReadout(spec.name, tr.body_id, st, kmh, frac, prop, pfrac, label, pointing)


# ------------------------------------------------------------------ the widget

try:                                                       # the module stays importable without Qt
    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
    from PySide6.QtWidgets import QScrollArea, QVBoxLayout, QWidget
except Exception:                                          # pragma: no cover
    QWidget = object  # type: ignore[assignment,misc]
    QScrollArea = object  # type: ignore[assignment,misc]

_DIM = "#2c2c2c"


def _font(widget, size: float, bold: bool = False):
    f = QFont(widget.font())
    f.setPointSizeF(size)
    f.setBold(bold)
    return f


def gauge_angle_deg(frac: float) -> float:
    """Screen angle (degrees, counter-clockwise from +x) of a gauge value 0..1: the 270 degree arc starts at the
    lower left (225) and runs clockwise over the top to the lower right (-45)."""
    return 225.0 - 270.0 * max(0.0, min(1.0, float(frac)))


def _gauge(p, c, r: float, frac: float, *, color="#ffffff", back="#2a2a2a", width=5.0,
           pointer: bool = False) -> None:
    """A 270 degree arc open at the bottom (the webcast gauge), filled clockwise from the lower left. With
    ``pointer`` a bright needle crosses the ring at the value, so the reading is the pointer's position."""
    rect = QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r)
    p.setPen(QPen(QColor(back), width, Qt.SolidLine, Qt.RoundCap))
    p.drawArc(rect, 225 * 16, -270 * 16)
    frac = max(0.0, min(1.0, frac))
    if frac > 0.0:
        p.setPen(QPen(QColor(color), width, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(rect, 225 * 16, int(-270 * 16 * frac))
    if pointer:
        a = math.radians(gauge_angle_deg(frac))
        inner, outer = r - 9.0, r + 9.0
        p.setPen(QPen(QColor(color), 3.0, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(c.x() + inner * math.cos(a), c.y() - inner * math.sin(a)),
                   QPointF(c.x() + outer * math.cos(a), c.y() - outer * math.sin(a)))


class VehicleHud(QWidget):                                 # type: ignore[misc,valid-type]
    """One vehicle's row: speed gauge | engine map in a throttle arc | orientation globe | propellant."""

    def __init__(self, spec: VehicleSpec, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.setObjectName("flightLabVehicleHud" + spec.kind.capitalize())
        self.setMinimumSize(280, 134)                     # keep the gauges legible in a short details pane
        self._r: VehicleReadout | None = None
        self._lit: frozenset[int] = frozenset()
        from orbitopt.flightlab.ui.plot_export import ExportButton

        self.export_button = ExportButton(self, lambda: self.spec.name)      # the card as a picture

    def resizeEvent(self, event) -> None:                  # noqa: N802
        super().resizeEvent(event)
        self.export_button.reposition()

    def set_readout(self, readout: VehicleReadout | None) -> None:
        self._r = readout
        st = readout.engines if readout else None
        if st is None or not st.on:
            self._lit = frozenset()
        elif st.lit is None:
            self._lit = frozenset()
        elif self.spec.kind == "generic":
            self._lit = frozenset(_spread(self.spec.total)[:st.lit])
        else:
            self._lit = lit_engines(self.spec.total, st.lit)
        self.update()

    @property
    def readout(self) -> VehicleReadout | None:
        return self._r

    @property
    def lit_indices(self) -> frozenset[int]:
        return self._lit

    @property
    def gauge_fractions(self) -> tuple[float, float]:
        """(propellant fraction on the outer ring, throttle on the inner ring), each 0..1."""
        rd = self._r
        prop = 0.0 if rd is None or rd.propellant_frac is None else max(0.0, min(1.0, rd.propellant_frac))
        st = rd.engines if rd else None
        thr = (0.0 if st is None or not st.on or not (st.count_known or st.throttle_est)
               else max(0.0, min(1.0, st.throttle)))
        return prop, thr

    # ---- painting
    def paintEvent(self, _event) -> None:                  # noqa: N802
        from orbitopt.viz.theme import HAIRLINE, INK_MUTED, INK_PRIMARY, INK_SECONDARY, PANEL

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QColor(PANEL))
        p.setPen(QPen(QColor(HAIRLINE), 1))                      # the row's divider, in the gap below it
        p.drawLine(QPointF(8, self.height() - 1.0), QPointF(self.width() - 8, self.height() - 1.0))
        w, h = float(self.width()), float(self.height())
        label_h = 30.0
        r, xs, _gap = row_layout(w, h, label_h)
        cy = (h - label_h) / 2.0 + 4.0
        rd = self._r
        self._paint_speed(p, QPointF(xs[0], cy), r, rd)
        self._paint_engines(p, QPointF(xs[1], cy), r * 0.92, rd)
        self._paint_globe(p, QPointF(xs[2], cy), r * 0.95, rd)
        self._paint_propellant(p, QPointF(xs[3], cy), r, rd)
        p.setPen(QColor(INK_PRIMARY)); p.setFont(_font(self, 11, True))
        p.drawText(QRectF(0, h - label_h, w, label_h), Qt.AlignCenter, self.spec.name)
        p.end()

    def _paint_speed(self, p, c, r, rd) -> None:
        from orbitopt.viz.theme import INK_MUTED, INK_PRIMARY, INK_SECONDARY

        _gauge(p, c, r, 0.0 if rd is None else rd.speed_frac)
        p.setPen(QColor(INK_SECONDARY)); p.setFont(_font(self, 8.5, True))
        p.drawText(QRectF(c.x() - r, c.y() - r * 0.62, 2 * r, 16), Qt.AlignCenter, "SPEED")
        p.setPen(QColor(INK_PRIMARY)); p.setFont(_font(self, 17, True))
        txt = "—" if rd is None or rd.speed_kmh is None else f"{rd.speed_kmh:,.0f}"
        p.drawText(QRectF(c.x() - r, c.y() - r * 0.28, 2 * r, 30), Qt.AlignCenter, txt)
        p.setPen(QColor(INK_SECONDARY)); p.setFont(_font(self, 8.5, True))
        p.drawText(QRectF(c.x() - r, c.y() + r * 0.28, 2 * r, 16), Qt.AlignCenter, "KM/H")

    def _paint_engines(self, p, c, r, rd) -> None:
        from orbitopt.viz.theme import HAIRLINE, INK_MUTED, INK_PRIMARY, INK_SECONDARY

        st = rd.engines if rd else None
        prop, thr = self.gauge_fractions
        # outer ring: the propellant left, as a fraction of what the vehicle started with (the number is the
        # column on the right); thin ring inside it: the throttle of the lit engines, read off a moving POINTER
        _gauge(p, c, r * 1.04, prop, width=6.0)
        _gauge(p, c, r * 0.90, thr, width=2.5, pointer=st is not None and st.on and (st.count_known or st.throttle_est))
        inner = r * 0.76
        p.setPen(Qt.NoPen)
        if self.spec.kind == "starship":
            for i, (x, y, rad, _k) in enumerate(starship_layout()):
                p.setBrush(QBrush(QColor(INK_PRIMARY if i in self._lit else _DIM)))
                p.drawEllipse(QPointF(c.x() + x * inner, c.y() - y * inner), rad * inner, rad * inner)
        elif self.spec.kind == "superheavy":
            er = ENGINE_RADIUS * inner * 1.3
            for i, (x, y, _ring) in enumerate(engine_layout(self.spec.total)):
                p.setBrush(QBrush(QColor(INK_PRIMARY if i in self._lit else _DIM)))
                p.drawEllipse(QPointF(c.x() + x * inner, c.y() - y * inner), er, er)
        else:
            total = self.spec.total
            er = inner * min(0.12, 1.7 / max(total, 1))
            ring = 0.0 if total == 1 else 0.68 * inner
            for i in range(total):
                a = -math.pi / 2 + 2 * math.pi * i / total
                p.setBrush(QBrush(QColor(INK_PRIMARY if i in self._lit else _DIM)))
                p.drawEllipse(QPointF(c.x() + ring * math.cos(a), c.y() + ring * math.sin(a)), er, er)
        p.setPen(QColor(INK_SECONDARY)); p.setFont(_font(self, 8.5))
        if st is None:
            line = "no data"
        else:
            lit = f"{st.lit}" if st.lit is not None else ("n/a" if st.on else "0")
            if rd.attached:
                line = "ATTACHED"
            elif st.on and not st.count_known:
                line = "FIRING" + (f"  THR ~{st.throttle * 100:.0f}%" if st.throttle_est else "")   # count not recorded
            else:
                count = f"{lit} lit" if self.spec.observed_only or not st.total else f"{lit}/{st.total}"
                line = count + (f"  THR {st.throttle * 100:.0f}%" if st.on else "")
        p.drawText(QRectF(c.x() - r * 1.2, c.y() + r * 1.1, r * 2.4, 14), Qt.AlignCenter, line)

    def _paint_globe(self, p, c, r, rd) -> None:
        from orbitopt.viz.theme import HAIRLINE, INK_MUTED, INK_SECONDARY

        st = rd.engines if rd else None
        o = st.orientation if st else None
        p.setPen(QPen(QColor(HAIRLINE), 1.4)); p.setBrush(Qt.NoBrush)
        p.drawEllipse(c, r, r)
        p.setPen(QPen(QColor("#3a3a3a"), 1))
        p.drawEllipse(c, r, r * 0.34); p.drawEllipse(c, r * 0.34, r); p.drawEllipse(c, r * 0.72, r)
        heading = o.heading_deg if o else 0.0
        th = math.radians(-heading - 60.0)
        nx, ny = c.x() + r * 0.98 * math.cos(th), c.y() + r * 0.34 * math.sin(th) - r * 0.05
        p.setPen(QColor(INK_SECONDARY)); p.setFont(_font(self, 10, True))
        p.drawText(QRectF(nx - 10, ny - 22, 20, 16), Qt.AlignCenter, "N")
        tilt = 0.0
        if o is not None:                                   # lean from the vertical, towards the nose's east side
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, o.nose_enu[2]))))
            tilt *= 1.0 if o.nose_enu[0] >= 0 else -1.0
        p.save(); p.translate(c); p.rotate(tilt)
        bw, bh = r * 0.12, r * 1.28
        path = QPainterPath()
        path.addRoundedRect(QRectF(-bw / 2, -bh * 0.42, bw, bh * 0.84), bw * 0.18, bw * 0.18)
        nose = QPolygonF([QPointF(-bw / 2, -bh * 0.42), QPointF(0, -bh * 0.58), QPointF(bw / 2, -bh * 0.42)])
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor("#d9d9d9" if o is not None else "#4a4a4a"))); p.drawPath(path)
        p.setBrush(QBrush(QColor("#1b1b1b" if o is not None else "#3a3a3a"))); p.drawPolygon(nose)
        p.setBrush(QBrush(QColor("#bdbdbd" if o is not None else "#444")))
        for sgn in (-1, 1):
            p.drawPolygon(QPolygonF([QPointF(sgn * bw / 2, bh * 0.30), QPointF(sgn * bw * 1.2, bh * 0.44),
                                     QPointF(sgn * bw / 2, bh * 0.42)]))
        p.restore()
        if rd is not None and rd.pointing and o is not None:
            p.setPen(QColor(INK_MUTED)); p.setFont(_font(self, 7.5))
            p.drawText(QRectF(c.x() - r * 1.3, c.y() + r * 1.1, r * 2.6, 14), Qt.AlignCenter,
                       f"{rd.pointing}  elev {int(round(o.nose_elevation_deg)):+d}°")

    def _paint_propellant(self, p, c, r, rd) -> None:
        from orbitopt.viz.theme import HAIRLINE, INK_MUTED, INK_PRIMARY, INK_SECONDARY

        bar_h, bar_w = r * 1.7, r * 0.34
        top = c.y() - bar_h / 2.0
        p.setPen(QPen(QColor(HAIRLINE), 1.2)); p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(QRectF(c.x() - bar_w / 2, top, bar_w, bar_h), 4, 4)
        frac = 0.0 if rd is None or rd.propellant_frac is None else rd.propellant_frac
        if frac > 0.0:
            fill = bar_h * frac
            p.setPen(Qt.NoPen); p.setBrush(QBrush(QColor(INK_PRIMARY)))
            p.drawRoundedRect(QRectF(c.x() - bar_w / 2 + 2, top + bar_h - fill + 1, bar_w - 4, max(0.0, fill - 2)), 3, 3)
        p.setPen(QColor(INK_SECONDARY)); p.setFont(_font(self, 8.5, True))
        p.drawText(QRectF(c.x() - r, top - 18, 2 * r, 14), Qt.AlignCenter, "PROPELLANT")
        p.setPen(QColor(INK_PRIMARY)); p.setFont(_font(self, 12, True))
        txt = "—" if rd is None or rd.propellant_t is None else f"{rd.propellant_t:,.1f} t"
        p.drawText(QRectF(c.x() - r * 1.1, c.y() + bar_h / 2 + 2, 2.2 * r, 18), Qt.AlignCenter, txt)
        p.setPen(QColor(INK_MUTED)); p.setFont(_font(self, 7.5))
        sub = ""
        if rd is not None and rd.propellant_frac is not None:
            sub = f"{rd.propellant_frac * 100:.0f}% of start" + (f" · {rd.propellant_label}" if rd.propellant_label else "")
        p.drawText(QRectF(c.x() - r * 1.3, c.y() + bar_h / 2 + 20, 2.6 * r, 12), Qt.AlignCenter, sub)


class EngineHud(QScrollArea):                              # type: ignore[misc,valid-type]
    """A row for each recorded launch stage, following the replay clock."""

    def __init__(self, parent=None):
        super().__init__(parent)
        from orbitopt.viz.theme import PANEL

        self.setObjectName("flightLabEngineHud")
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.NoFrame)
        content = QWidget()
        content.setObjectName("flightLabEngineContent")
        content.setAttribute(Qt.WA_StyledBackground)
        content.setStyleSheet(f"#flightLabEngineContent {{ background: {PANEL}; }}")
        self.setWidget(content)
        lay = self._rows = QVBoxLayout(content)
        lay.setContentsMargins(0, 6, 0, 6)
        lay.setSpacing(22)                                  # air between the two vehicles
        from orbitopt.flightlab.ui.grid_fin_hud import GridFinHud

        self.superheavy = VehicleHud(SUPER_HEAVY)
        self.starship = VehicleHud(STARSHIP)
        self._stage_huds = {1: self.superheavy, 2: self.starship}
        self._active_stages: tuple[int, ...] = ()
        self.gridfins = GridFinHud()                        # the booster's grid fins: layout, deflection, airflow
        lay.addWidget(self.superheavy, 1)
        lay.addWidget(self.starship, 1)
        lay.addWidget(self.gridfins, 1)
        self._tracks: dict = {}
        self._speed: dict = {}
        self._prop: dict = {}
        self._flow: dict = {}
        self._burns: tuple = ()
        self._t = 0.0

    def clear(self) -> None:
        self._tracks, self._speed, self._prop, self._flow, self._burns = {}, {}, {}, {}, ()
        self._active_stages = ()
        for hud in self._stage_huds.values():
            hud.set_readout(None)
            hud.hide()
        self.gridfins.set_readout(None)
        self.gridfins.hide()

    @property
    def has_stages(self) -> bool:
        return bool(self._active_stages)

    def set_run(self, tracks: Sequence, channels: Sequence, burns: Sequence = (),
                fin_count: int | None = None, fins_fixed: bool = True, *, vehicle: Mapping | None = None) -> None:
        """``tracks`` are the run's BodyTracks; ``channels`` the telemetry channels (speed and propellant);
        ``fin_count`` the booster's grid fins when its vehicle declares them (else the V3 three); ``fins_fixed``
        whether they never fold (Super Heavy's do not; a vehicle that stows them says ``stowable``)."""
        self.clear()
        self.gridfins.set_fin_count(fin_count if fin_count is not None else 3)
        self.gridfins.set_fixed(fins_fixed)
        self._tracks = {t.body_id: t for t in tracks}
        specs = stage_specs(self._tracks, vehicle)
        for position, (number, spec) in enumerate(specs.items()):
            if number not in self._stage_huds:
                self._stage_huds[number] = VehicleHud(spec)
            hud = self._stage_huds[number]
            hud.spec = spec
            self._rows.insertWidget(position, hud, 1)
            hud.show()
        self._active_stages = tuple(specs)
        booster = self._tracks.get("booster")
        self.gridfins.setVisible(booster is not None and (fin_count is not None or
                                 any("fin" in key.lower() for key in booster.extra_columns)))
        self._burns = tuple(burns)
        self._speed, self._prop, self._flow = {}, {}, {}
        for body_id, tr in self._tracks.items():                # speed against the ground, like the webcast
            self._speed[body_id] = earth_relative_speed_series(tr)
            self._flow[body_id] = mass_flow_series(tr)          # throttle for tracks that record none
        for ch in channels or ():
            for ser in ch.series:
                if ch.id == "propellant":
                    label = "recorded" if "recorded" in ser.label else "est." if "est." in ser.label else ""
                    self._prop.setdefault(ser.body_id, (label, ser.points))
        self.set_time(self._t)

    def set_time(self, t: float) -> None:
        self._t = float(t)
        if not self._tracks:
            return
        for number in self._active_stages:
            hud = self._stage_huds[number]
            hud.set_readout(readout_at(hud.spec, self._tracks, self._speed, self._prop, self._t, self._burns, self._flow))
        from orbitopt.flightlab.ui.grid_fin_hud import gridfin_readout_at

        booster = self._tracks.get("booster")
        self.gridfins.set_readout(None if booster is None
                                  else gridfin_readout_at(booster.t_s, booster.extra_columns, self._t))

    # kept for callers that only care about the booster row
    @property
    def state(self):
        r = self.superheavy.readout
        return None if r is None else r.engines

    @property
    def lit_indices(self) -> frozenset[int]:
        return self.superheavy.lit_indices
