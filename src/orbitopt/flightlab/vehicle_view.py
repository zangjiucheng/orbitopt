"""Read-only description of a Flight Lab vehicle, for the Vehicle view.

The Vehicle page of Flight Lab's sidebar draws what a vehicle *file* declares
-- stack, masses, thrust schedule, staging and recovery stage -- and, when a
flown run is selected, sets the declared values beside what the run flew.
Nothing here edits a file or simulates anything: it only reads the document
(:func:`orbitopt.flightlab.vehicle_catalog.load_validated`) and, for the
overlay, the run's events and tracks. Qt-free, so it imports on a headless
install and the numbers are testable.

Time conventions (the OT adapter's, checked against a flown run): stage 1's
frame starts at liftoff; stage *k* >= 2's frame starts when stage *k*-1
separates (``separation_time_s``). A stage ignites ``engine.burn_start_s``
into its frame; its ``burn_end_s``, its thrust *table* and any jettison
``time_s`` are all measured from the frame start (the table is not shifted by
``burn_start_s``: a flown stage 2 burns the table's value at ``burn_start_s``
from the first second). A ``restart:`` block carries absolute times (seconds
after liftoff) -- the declared schedule the overlay checks.

The derived numbers are **ideal**: the rocket equation with vacuum Isp over
the declared thrust schedule, no gravity, drag or steering losses. They exist
to read a vehicle at a glance, not to predict a flight -- the flown overlay
is the check.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from orbitopt.flightlab import vehicle_catalog as vc

__all__ = [
    "G0",
    "BoosterView",
    "GeometryView",
    "Jettison",
    "OverlayRow",
    "Segment",
    "StageView",
    "VehicleView",
    "build_vehicle_view",
    "file_sha256",
    "flown_overlay",
    "load_vehicle_view",
]

G0 = 9.80665
P0_PA = 101_325.0

#: a stage whose schematic length is not declared is drawn this many
#: diameters tall at most (keeps a 600 t booster from dwarfing the page)
_SCHEMATIC_MAX_ASPECT = 11.0


# --------------------------------------------------------------------------
# value objects
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Segment:
    """One engine interval of a stage: ``kind`` is ``burn``, ``coast`` (the
    gap of a restart schedule) or ``restart``."""

    stage: int
    kind: str
    t0_s: float
    t1_s: float


@dataclass(frozen=True)
class Jettison:
    name: str
    t_s: float            # absolute, seconds after liftoff
    mass_kg: float


@dataclass(frozen=True)
class StageView:
    index: int                                  # 1-based
    mass_initial_kg: float
    mass_dry_kg: float
    diameter_m: float | None
    isp_vac_s: float
    thrust_vac_n: float                         # peak of the declared schedule
    ignition_s: float
    burnout_s: float
    separation_s: float | None
    burned_kg: float                            # propellant the declared schedule burns
    delta_v_m_s: float                          # ideal, vacuum Isp, see module docstring
    thrust: tuple[tuple[float, float], ...]     # absolute (t, N) polyline
    segments: tuple[Segment, ...]
    jettisons: tuple[Jettison, ...]
    nozzle_exit_area_m2: float | None = None

    @property
    def propellant_kg(self) -> float:
        return self.mass_initial_kg - self.mass_dry_kg

    @property
    def left_at_cutoff_kg(self) -> float:
        """Loaded propellant the declared schedule does not burn (the RTLS
        reserve of a stage-1 that is cut off before depletion)."""
        return self.propellant_kg - self.burned_kg


@dataclass(frozen=True)
class BoosterView:
    dry_mass_kg: float
    landing_reserve_kg: float
    reference_area_m2: float
    engine_count: int | None
    thrust_vacuum_n: float
    isp_sea_level_s: float
    isp_vacuum_s: float
    throttle_min: float
    throttle_max: float | None
    restarts_max: int | None
    usable_for: tuple[str, ...]


@dataclass(frozen=True)
class GeometryView:
    """What the drawing needs. ``declared`` is True when the vehicle file has
    a ``geometry:`` block (lengths in metres); otherwise the lengths are
    relative drawing units from stage mass over diameter squared -- a
    schematic, never to be read in metres."""

    declared: bool
    stage_length: tuple[float, ...]
    stage_diameter: tuple[float, ...]
    fairing_length: float | None
    fairing_diameter: float | None


@dataclass(frozen=True)
class OverlayRow:
    label: str
    unit: str
    declared: float | None
    flown: float | None

    @property
    def delta(self) -> float | None:
        if self.declared is None or self.flown is None:
            return None
        return self.flown - self.declared


@dataclass(frozen=True)
class VehicleView:
    name: str
    provenance: str
    as_of: str
    disclaimer: str
    assumptions: tuple[str, ...]
    payload_kg: float
    stages: tuple[StageView, ...]
    booster: BoosterView | None
    geometry: GeometryView
    restart: Mapping[str, Any] | None
    markers: tuple[tuple[float, str, str], ...]      # (t_s, label, kind)
    liftoff_mass_kg: float
    liftoff_thrust_vac_n: float
    liftoff_thrust_sl_n: float | None
    notes: tuple[str, ...]
    document: Mapping[str, Any] = field(repr=False, default_factory=dict)
    source_path: str = ""

    @property
    def total_delta_v_m_s(self) -> float:
        return sum(s.delta_v_m_s for s in self.stages)

    @property
    def liftoff_twr_vac(self) -> float:
        return self.liftoff_thrust_vac_n / (self.liftoff_mass_kg * G0)

    @property
    def liftoff_twr_sl(self) -> float | None:
        if self.liftoff_thrust_sl_n is None:
            return None
        return self.liftoff_thrust_sl_n / (self.liftoff_mass_kg * G0)

    @property
    def last_burnout_s(self) -> float:
        return max(s.burnout_s for s in self.stages)

    def provenance_for(self, path: str):
        """The resolved :class:`~orbitopt.flightlab.vehicle_catalog.Provenance`
        of a document path, or ``None`` when the number carries no tag."""
        try:
            return vc.provenance_of(self.document, path)
        except (KeyError, ValueError):
            return None


# --------------------------------------------------------------------------
# document -> view
# --------------------------------------------------------------------------

def _num(node: Any, default: float | None = None) -> float | None:
    """A number, or the ``value`` of an inline provenance node."""
    if isinstance(node, Mapping):
        node = node.get("value")
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        return default
    return float(node)


def _table(node: Any) -> list[tuple[float, float]] | None:
    if isinstance(node, Mapping):
        node = node.get("value")
    if isinstance(node, (list, tuple)) and node and all(
            isinstance(p, (list, tuple)) and len(p) == 2 for p in node):
        return [(float(t), float(f)) for t, f in node]
    return None


def _window(points: Sequence[tuple[float, float]], t0: float,
            t1: float) -> list[tuple[float, float]]:
    """The polyline restricted to ``[t0, t1]``, the cut points interpolated."""
    def at(t: float) -> float:
        if t <= points[0][0]:
            return points[0][1]
        for (ta, fa), (tb, fb) in zip(points, points[1:]):
            if ta <= t <= tb:
                return fa + (fb - fa) * (t - ta) / ((tb - ta) or 1.0)
        return points[-1][1]

    inner = [(t, f) for t, f in points if t0 < t < t1]
    return [(t0, at(t0)), *inner, (t1, at(t1))]


def _truncate_impulse(points: Sequence[tuple[float, float]], limit: float
                      ) -> tuple[list[tuple[float, float]], float | None]:
    """The polyline up to the instant its impulse reaches ``limit`` (the
    stage runs dry); ``None`` for the time when it never does."""
    out = [points[0]]
    used = 0.0
    for (ta, fa), (tb, fb) in zip(points, points[1:]):
        seg = 0.5 * (fa + fb) * (tb - ta)
        if used + seg < limit or seg <= 0.0:
            out.append((tb, fb))
            used += max(seg, 0.0)
            continue
        need = limit - used
        k = (fb - fa) / ((tb - ta) or 1.0)
        # fa*x + k*x^2/2 = need
        x = need / fa if abs(k) < 1e-9 * max(fa, 1.0) else (
            (-fa + math.sqrt(max(fa * fa + 2.0 * k * need, 0.0))) / k)
        x = min(max(x, 0.0), tb - ta)
        out.append((ta + x, fa + k * x))
        return out, ta + x
    return out, None


def _impulse(points: Sequence[tuple[float, float]]) -> float:
    return sum(0.5 * (f0 + f1) * (t1 - t0)
               for (t0, f0), (t1, f1) in zip(points, points[1:]))


def _stage_thrust(stage: Mapping[str, Any], origin: float, ignition: float,
                  burnout: float, restart: Mapping[str, Any] | None
                  ) -> tuple[list[tuple[float, float]], list[Segment]]:
    """Absolute thrust polyline and the engine segments of one stage."""
    engine = stage.get("engine") or {}
    raw = engine.get("thrust_vac_n")
    table = _table(raw)
    if table is None:
        peak = _num(raw, 0.0) or 0.0
        table = [(0.0, peak), (max(burnout - origin, 1.0), peak)]
    absolute = [(origin + t, f) for t, f in table]
    if restart is None:
        return _window(absolute, ignition, burnout), [Segment(0, "burn", ignition, burnout)]
    cs = float(restart["coast_start_s"])
    rs = cs + float(restart["coast_duration_s"])
    re_ = rs + float(restart["restart_burn_s"])
    f_restart = float(restart["restart_thrust_n"])
    main = _window(absolute, ignition, cs)
    pts = main + [(cs, 0.0), (rs, 0.0), (rs, f_restart), (re_, f_restart)]
    return pts, [Segment(0, "burn", ignition, cs), Segment(0, "coast", cs, rs),
                 Segment(0, "restart", rs, re_)]


def _origins(stages: Sequence[Mapping[str, Any]]) -> list[float]:
    """Frame start of every stage (see the module docstring)."""
    out = [0.0]
    for prev in stages[:-1]:
        sep = _num(prev.get("separation_time_s"))
        if sep is None:   # no declared separation: the previous stage's burn end
            sep = out[-1] + (_num((prev.get("engine") or {}).get("burn_end_s"), 0.0) or 0.0)
        out.append(sep)
    return out


def _geometry(doc: Mapping[str, Any], stages: Sequence[Mapping[str, Any]]) -> GeometryView:
    diam = [_num((s.get("aero") or {}).get("diameter_m")) for s in stages]
    geo = doc.get("geometry")
    if isinstance(geo, Mapping) and geo.get("stages"):
        lengths, diameters = [], []
        for i, s in enumerate(stages):
            g = (geo["stages"][i] if i < len(geo["stages"]) else {}) or {}
            lengths.append(float(g["length_m"]) if "length_m" in g else 0.0)
            diameters.append(float(g.get("diameter_m") or diam[i] or 0.0))
        fair = geo.get("fairing") or {}
        top_d = diameters[-1] if diameters else 0.0
        return GeometryView(
            declared=True, stage_length=tuple(lengths), stage_diameter=tuple(diameters),
            fairing_length=float(fair["length_m"]) if "length_m" in fair else None,
            fairing_diameter=float(fair.get("diameter_m") or top_d) if fair else None)
    # schematic: equal-density tanks -> length ~ mass / d^2, bounded
    known = [d for d in diam if d]
    default_d = known[0] if known else 1.0
    diameters = [d or default_d for d in diam]
    raw = [float(s.get("mass_initial_kg", 0.0)) / (d * d) for s, d in zip(stages, diameters)]
    tallest = max(raw) or 1.0
    lengths = [diameters[i] * _SCHEMATIC_MAX_ASPECT * raw[i] / tallest for i in range(len(raw))]
    return GeometryView(declared=False, stage_length=tuple(lengths),
                        stage_diameter=tuple(diameters),
                        fairing_length=None, fairing_diameter=None)


def _booster(doc: Mapping[str, Any]) -> BoosterView | None:
    b = doc.get("booster")
    if not isinstance(b, Mapping):
        return None
    e = b.get("engines") or {}
    count = e.get("count")
    restarts = e.get("restarts_max")
    return BoosterView(
        dry_mass_kg=_num(b.get("dry_mass_kg"), 0.0) or 0.0,
        landing_reserve_kg=_num(b.get("landing_propellant_reserve_kg"), 0.0) or 0.0,
        reference_area_m2=_num(b.get("reference_area_m2"), 0.0) or 0.0,
        engine_count=int(count) if isinstance(count, (int, float)) else None,
        thrust_vacuum_n=_num(e.get("thrust_vacuum_n"), 0.0) or 0.0,
        isp_sea_level_s=_num(e.get("isp_sea_level_s"), 0.0) or 0.0,
        isp_vacuum_s=_num(e.get("isp_vacuum_s"), 0.0) or 0.0,
        throttle_min=_num(e.get("throttle_min"), 0.0) or 0.0,
        throttle_max=_num(e.get("throttle_max")),
        restarts_max=int(restarts) if isinstance(restarts, (int, float)) else None,
        usable_for=tuple(str(u) for u in (e.get("usable_for") or ())))


def build_vehicle_view(doc: Mapping[str, Any], *, source_path: str = "") -> VehicleView:
    """The Vehicle page's model of a vehicle document (validated or not: a
    field the file lacks simply drops out of the view)."""
    raw_stages = [s for s in (doc.get("stages") or []) if isinstance(s, Mapping)]
    if not raw_stages:
        raise ValueError("a vehicle needs at least one stage")
    payload = _num(doc.get("payload_kg"), 0.0) or 0.0
    restart = doc.get("restart") if isinstance(doc.get("restart"), Mapping) else None
    restart_stage = int(restart.get("stage", len(raw_stages))) if restart else None
    origins = _origins(raw_stages)
    notes: list[str] = []
    markers: list[tuple[float, str, str]] = []
    stages: list[StageView] = []

    for k, s in enumerate(raw_stages, start=1):
        eng = s.get("engine") or {}
        origin = origins[k - 1]
        ignition = origin + (_num(eng.get("burn_start_s"), 0.0) or 0.0)
        rs = restart if restart is not None and k == restart_stage else None
        if rs is not None:
            burnout = (float(rs["coast_start_s"]) + float(rs["coast_duration_s"])
                       + float(rs["restart_burn_s"]))
        else:
            end = _num(eng.get("burn_end_s"))
            if end is not None:
                burnout = origin + end
            else:
                tab = _table(eng.get("thrust_vac_n"))
                burnout = ignition + (tab[-1][0] if tab else 0.0)
        pts, segs = _stage_thrust(s, origin, ignition, burnout, rs)
        isp = _num(eng.get("isp_vac_s"), 0.0) or 0.0
        mi = float(_num(s.get("mass_initial_kg"), 0.0) or 0.0)
        md = float(_num(s.get("mass_dry_kg"), 0.0) or 0.0)
        loaded = max(mi - md, 0.0)
        burned = _impulse(pts) / (isp * G0) if isp > 0 else 0.0
        if isp > 0 and burned > loaded * 1.001:
            pts, dry_t = _truncate_impulse(pts, loaded * isp * G0)
            if dry_t is not None:
                notes.append(
                    f"Stage {k}: the declared thrust schedule would burn {burned / 1000:.1f} t "
                    f"but only {loaded / 1000:.1f} t is loaded; the stage runs dry at "
                    f"{dry_t:.1f} s (burn_end_s puts it at {burnout:.1f} s).")
                burnout = dry_t
                segs = [Segment(g.stage, g.kind, g.t0_s, min(g.t1_s, dry_t))
                        for g in segs if g.t0_s < dry_t]
                burned = loaded
        above = sum(float(_num(x.get("mass_initial_kg"), 0.0) or 0.0) for x in raw_stages[k:])
        m0 = mi + above + payload
        mf = m0 - min(burned, loaded)
        dv = isp * G0 * math.log(m0 / mf) if isp > 0 and mf > 0 and m0 > mf else 0.0
        sep = _num(s.get("separation_time_s"))
        jets = tuple(Jettison(str(j.get("name", "jettison")), origin + float(j.get("time_s", 0.0)),
                              float(j.get("mass_kg", 0.0)))
                     for j in (s.get("jettisons") or []) if isinstance(j, Mapping))
        stages.append(StageView(
            index=k, mass_initial_kg=mi, mass_dry_kg=md,
            diameter_m=_num((s.get("aero") or {}).get("diameter_m")),
            isp_vac_s=isp, thrust_vac_n=max((f for _, f in pts), default=0.0),
            ignition_s=ignition, burnout_s=burnout, separation_s=sep,
            burned_kg=burned, delta_v_m_s=dv, thrust=tuple(pts),
            segments=tuple(Segment(k, g.kind, g.t0_s, g.t1_s) for g in segs),
            jettisons=jets, nozzle_exit_area_m2=_num(eng.get("nozzle_exit_area_m2"))))
        if sep is not None:
            markers.append((sep, f"Stage {k} separation", "separation"))
        for j in jets:
            markers.append((j.t_s, f"{j.name.capitalize()} jettison", "jettison"))
        if sep is not None and sep < burnout - 1e-6 and k < len(raw_stages):
            notes.append(f"Stage {k} separates at {sep:.1f} s, before its burn ends at "
                         f"{burnout:.1f} s.")

    first = stages[0]
    stack = sum(s.mass_initial_kg for s in stages) + payload
    f0 = first.thrust[0][1] if first.thrust else first.thrust_vac_n
    ae = first.nozzle_exit_area_m2
    f_sl = max(f0 - P0_PA * ae, 0.0) if ae is not None else None
    twr_sl = f_sl / (stack * G0) if f_sl is not None else None
    if twr_sl is not None and twr_sl < 1.0:
        notes.append(f"Sea-level liftoff thrust-to-weight is {twr_sl:.2f} (< 1): "
                     "the declared stack cannot leave the pad.")
    for st in stages:
        if st.burned_kg and st.left_at_cutoff_kg > 0.02 * st.propellant_kg:
            notes.append(f"Stage {st.index} is cut off before depletion: "
                         f"{st.left_at_cutoff_kg / 1000:.1f} t of its "
                         f"{st.propellant_kg / 1000:.1f} t propellant stays unburned.")
    markers.sort()
    return VehicleView(
        name=str(doc.get("name", "vehicle")), provenance=str(doc.get("provenance", "")),
        as_of=str(doc.get("as_of", "")), disclaimer=str(doc.get("disclaimer", "")).strip(),
        assumptions=tuple(str(a).strip() for a in (doc.get("assumptions") or ())),
        payload_kg=payload, stages=tuple(stages), booster=_booster(doc),
        geometry=_geometry(doc, raw_stages), restart=dict(restart) if restart else None,
        markers=tuple(markers), liftoff_mass_kg=stack, liftoff_thrust_vac_n=f0,
        liftoff_thrust_sl_n=f_sl, notes=tuple(notes), document=doc, source_path=source_path)


def load_vehicle_view(vehicle: str | Path) -> VehicleView:
    """A packaged catalog name or any path -> :class:`VehicleView`."""
    doc = vc.load_vehicle_file(vehicle)
    path = vc.catalog_paths().get(str(vehicle)) if not isinstance(vehicle, Path) else vehicle
    return build_vehicle_view(doc, source_path=str(path or vehicle))


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# declared vs flown
# --------------------------------------------------------------------------

def _first(events, kind: str, stage: int | None = None, *, jettison: bool | None = None):
    for e in sorted(events, key=lambda e: float(e.t_s)):
        if e.kind != kind:
            continue
        data = e.data or {}
        if stage is not None and int(data.get("stage", -1)) != stage:
            continue
        if jettison is not None and (data.get("ot_event") == "jettison") != jettison:
            continue
        return e
    return None


def flown_overlay(view: VehicleView, run) -> list[OverlayRow]:
    """Declared-vs-flown rows for a :class:`~orbitopt.flightlab.contract.FlightRun`.
    A row keeps ``None`` on the side that does not exist (a flown-only
    event, or a declared value the run did not reach). Times in seconds
    after liftoff, masses in kg."""
    events = list(run.events)
    rows: list[OverlayRow] = []

    def add(label: str, unit: str, declared, flown) -> None:
        rows.append(OverlayRow(label, unit, declared, None if flown is None else float(flown)))

    liftoff = _first(events, "liftoff")
    t_lift = float(liftoff.t_s) if liftoff is not None else 0.0
    meco = _first(events, "meco")
    add("Stage 1 cutoff (MECO)", "s", view.stages[0].burnout_s,
        None if meco is None else float(meco.t_s) - t_lift)
    seps = [e for e in sorted(events, key=lambda e: float(e.t_s))
            if e.kind == "separation" and (e.data or {}).get("ot_event") != "jettison"]
    for k, stage in enumerate(view.stages[:-1]):
        if stage.separation_s is not None:
            add(f"Stage {k + 1} separation", "s", stage.separation_s,
                float(seps[k].t_s) - t_lift if k < len(seps) else None)
    for stage in view.stages[1:]:
        ign = _first(events, "ignition", stage.index)
        add(f"Stage {stage.index} ignition", "s", stage.ignition_s,
            None if ign is None else float(ign.t_s) - t_lift)
    jets = [e for e in sorted(events, key=lambda e: float(e.t_s))
            if (e.data or {}).get("ot_event") == "jettison"]
    declared_jets = [j for s in view.stages for j in s.jettisons]
    for i, j in enumerate(declared_jets):
        add(f"{j.name.capitalize()} jettison", "s", j.t_s,
            float(jets[i].t_s) - t_lift if i < len(jets) else None)
    seco = _first(events, "seco")
    if view.restart is not None:
        coast = (None if seco is None else _flown_coast(
            run, seco.body_id, view.stages[-1].ignition_s + t_lift, float(seco.t_s)))
        cs, rs = float(view.restart["coast_start_s"]), (
            float(view.restart["coast_start_s"]) + float(view.restart["coast_duration_s"]))
        add("Coast start (mass track)", "s", cs, None if coast is None else coast[0] - t_lift)
        add("Restart ignition (mass track)", "s", rs, None if coast is None else coast[1] - t_lift)
    add("Final cutoff (SECO)", "s", view.stages[-1].burnout_s,
        None if seco is None else float(seco.t_s) - t_lift)
    maxq = _first(events, "max_q")
    if maxq is not None:
        add("Max-Q", "s", None, float(maxq.t_s) - t_lift)
    tracks = list(run.tracks)
    if tracks:
        root = tracks[0]
        add("Liftoff mass", "kg", view.liftoff_mass_kg, float(root.mass_kg[0]))
        released = [t for t in tracks if t.parent_body_id == root.body_id]
        booster = next((t for t in released if t.body_id.endswith("_debris")
                        or t.body_id == "booster"), None)
        if booster is not None:
            add("Booster mass after separation", "kg", None, float(booster.mass_kg[0]))
    return rows


def _flown_coast(run, body_id: str, t_from: float, t_to: float) -> tuple[float, float] | None:
    """The longest stretch inside ``[t_from, t_to]`` over which ``body_id``'s mass
    stays constant (a ballistic coast between two burns), as
    ``(coast_start, restart)`` from the run's tracks. The edges are only as
    sharp as the track's output step. ``None`` without such a stretch."""
    best: tuple[float, float] | None = None
    for track in run.tracks:
        if track.body_id != body_id:
            continue
        t, m = list(track.t_s), list(track.mass_kg)
        i = 0
        while i < len(t) - 1:
            if not (t_from <= t[i] <= t_to) or abs(m[i + 1] - m[i]) > 0.5:
                i += 1
                continue
            j = i
            while j < len(t) - 1 and abs(m[j + 1] - m[j]) <= 0.5 and t[j + 1] <= t_to:
                j += 1
            if j > i and (best is None or t[j] - t[i] > best[1] - best[0]):
                best = (float(t[i]), float(t[j]))
            i = j + 1
    return best
