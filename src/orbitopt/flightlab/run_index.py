"""Run discovery and the launch summary for Mission Control's Flight Lab mode.

Flight Lab renders runs; it does not make them. Runs come from code and the
CLI (``orbitopt flight simulate -o DIR``, ``orbitopt launch -o DIR``, the
reference scenarios), so Mission Control finds them on disk the way the
Missions list finds missions:

* :func:`discover_flight_runs` walks a set of root folders for FlightResult
  run directories (a ``manifest.json`` with ``run_id`` and ``bodies``);
* :func:`default_run_roots` is where it looks without being told: every
  folder in ``$ORBITOPT_FLIGHT_RUNS`` (``os.pathsep``-separated) and
  ``./runs`` (Mission Control adds the folders picked with *Add folder…*);
* :func:`launch_summary` and :func:`key_events` are the "what flew"
  read-out of the Flight tab: site, liftoff epoch, vehicle, target and
  achieved orbit, payload, and the altitude/speed at each milestone.

Qt-free, so a headless install can list and summarise runs.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "KeyEvent",
    "RunEntry",
    "SummaryRow",
    "default_run_roots",
    "discover_flight_runs",
    "find_target",
    "is_run_dir",
    "key_events",
    "launch_summary",
    "run_label",
]

#: Folders never worth descending into while looking for runs.
_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache",
              ".pytest_cache", "trajectories", "opentsiolkovsky", "bodies", "vehicle"}
#: Run directories named for their role inside a bigger output folder: the
#: list labels them by the folder that holds them (``launch-iss/ascent``).
_ROLE_NAMES = {"ascent", "run", "flight", "result"}

EVENT_TITLES = {
    "liftoff": "Liftoff", "ignition": "Ignition", "max_q": "Max-Q", "meco": "MECO",
    "separation": "Separation", "seco": "SECO", "impact": "Impact",
    "touchdown": "Touchdown", "boostback": "Boostback", "entry_burn": "Entry burn",
    "landing_burn": "Landing burn", "apogee": "Apogee", "jettison": "Jettison",
}


@dataclass(frozen=True)
class RunEntry:
    """One discovered run directory, as the sidebar lists it."""

    path: Path
    label: str
    detail: str
    mtime: float
    verdict: bool | None


@dataclass(frozen=True)
class SummaryRow:
    section: str
    label: str
    value: str


@dataclass(frozen=True)
class KeyEvent:
    t_s: float
    kind: str
    title: str
    body_id: str
    altitude_km: float | None
    speed_km_s: float | None


# --------------------------------------------------------------------------- discovery


def _read_json(path: Path) -> Mapping[str, Any] | None:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, Mapping) else None


def is_run_dir(path: Path | str) -> bool:
    """A complete FlightResult run directory (an in-progress cache entry is not)."""
    path = Path(path)
    manifest = path / "manifest.json"
    if not manifest.is_file() or (path / "_designer-in-progress").exists():
        return False
    doc = _read_json(manifest)
    return bool(doc) and "run_id" in doc and "bodies" in doc


def run_label(path: Path | str) -> str:
    """The list label: the folder name, or the parent's for a role-named
    folder (``orbitopt launch -o iss-cape`` writes ``iss-cape/ascent``)."""
    path = Path(path)
    if path.name.lower() in _ROLE_NAMES and path.parent.name:
        return path.parent.name
    return path.name


def _verdict(path: Path) -> bool | None:
    doc = _read_json(path / "verification.json") or {}
    value = doc.get("mission_constraints_passed")
    return value if isinstance(value, bool) else None


def _entry(path: Path) -> RunEntry:
    manifest = _read_json(path / "manifest.json") or {}
    run_id = str(manifest.get("run_id", path.name))
    created = str(manifest.get("created_utc", ""))
    detail = " · ".join(part for part in (run_id, created.replace("T", " ").rstrip("Z")) if part)
    return RunEntry(path=path, label=run_label(path), detail=detail,
                    mtime=(path / "manifest.json").stat().st_mtime, verdict=_verdict(path))


def discover_flight_runs(roots: Iterable[Path | str], *, max_depth: int = 4) -> list[RunEntry]:
    """Every run directory under ``roots`` (a root may itself be a run),
    newest first, each listed once. Missing roots are skipped; the walk stops
    at a run directory (its subfolders are its own data) and ``max_depth``
    levels below a root."""
    found: dict[Path, RunEntry] = {}

    def walk(path: Path, depth: int) -> None:
        if is_run_dir(path):
            resolved = path.resolve()
            if resolved not in found:
                found[resolved] = _entry(path)
            return
        if depth >= max_depth:
            return
        try:
            children = sorted(p for p in path.iterdir() if p.is_dir())
        except OSError:
            return
        for child in children:
            if child.name.startswith(".") or child.name in _SKIP_DIRS:
                continue
            walk(child, depth + 1)

    for root in roots:
        root = Path(root).expanduser()
        if root.is_dir():
            walk(root, 0)
    return sorted(found.values(), key=lambda e: (-e.mtime, e.label))


def default_run_roots() -> list[Path]:
    """Where Flight Lab looks without being told (see the module docstring).
    The designer's result cache is not among them: its runs are named by
    input digest, and with no designer page they are reached by adding that
    folder like any other."""
    roots: list[Path] = []
    for part in os.environ.get("ORBITOPT_FLIGHT_RUNS", "").split(os.pathsep):
        if part.strip():
            roots.append(Path(part.strip()).expanduser())
    roots.append(Path.cwd() / "runs")
    return roots


# --------------------------------------------------------------------------- summary


def find_target(run_dir: Path | str):
    """The FlightTarget the run flew, when it sits beside the run (``target.yaml``
    in the run folder or its parent -- the ``orbitopt launch`` layout); None
    otherwise. The run itself records only the target's digest."""
    from orbitopt.flightlab.contract import load_target

    run_dir = Path(run_dir)
    for cand in (run_dir / "target.yaml", run_dir.parent / "target.yaml"):
        if cand.is_file():
            try:
                return load_target(cand)
            except Exception:  # noqa: BLE001 -- an unreadable target only drops its rows
                return None
    return None


def _first_sample_geodetic(run) -> tuple[float, float, float] | None:
    """Launch point (lat, lon, height) from the first sample of the first stage."""
    track = next((t for t in run.tracks if t.parent_body_id is None and t.t_s), None)
    if track is None:
        return None
    try:
        from orbitopt.flightlab.frames import ecef_to_geodetic, from_gcrs, to_gcrs

        epoch = datetime.fromisoformat(track.epoch_utc.replace("Z", "+00:00")).astimezone(timezone.utc)
        r, v, t = [track.position_m[0]], [track.velocity_m_s[0]], [float(track.t_s[0])]
        if track.frame != "GCRS":
            r, v = to_gcrs(r, v, track.frame, epoch, t)
        r_ecef, _ = from_gcrs(r, v, "ECEF", epoch, t)
        lat, lon, h = ecef_to_geodetic(r_ecef[0])
        return float(lat), float(lon), float(h)
    except Exception:  # noqa: BLE001 -- no astropy / odd frame: the row is simply omitted
        return None


def _fmt_latlon(lat: float, lon: float) -> str:
    return (f"{abs(lat):.3f}° {'N' if lat >= 0 else 'S'}, "
            f"{abs(lon):.3f}° {'E' if lon >= 0 else 'W'}")


def _fmt_orbit(peri_km: float | None, apo_km: float | None, inc_deg: float | None) -> str:
    if peri_km is None or apo_km is None:
        return "—"
    text = f"{peri_km:,.0f} × {apo_km:,.0f} km"
    return text + (f" @ {inc_deg:.2f}°" if inc_deg is not None else "")


def _fmt_kg(value: float | None) -> str:
    return "—" if value is None else f"{value:,.0f} kg"


def launch_summary(bundle, target=None) -> list[SummaryRow]:
    """Rows for the Flight tab: launch, vehicle, target, result, run."""
    from orbitopt.flightlab.compare import run_outputs

    run = bundle.run
    rows: list[SummaryRow] = []
    add = lambda section, label, value: rows.append(SummaryRow(section, label, str(value)))  # noqa: E731

    site = _first_sample_geodetic(run)
    if target is not None:
        add("Launch", "Site", target.launch_site.name)
        add("Launch", "Coordinates", _fmt_latlon(target.launch_site.latitude_deg,
                                                  target.launch_site.longitude_deg))
    elif site is not None:
        add("Launch", "Coordinates", _fmt_latlon(site[0], site[1]))
    epoch = next((t.epoch_utc for t in run.tracks), run.created_utc)
    add("Launch", "Liftoff (UTC)", epoch.replace("T", " ").rstrip("Z"))
    decision = (bundle.optimization or {}).get("decision_vector") or {}
    azimuth = decision.get("launch_azimuth_deg")
    if azimuth is None:
        program = (bundle.vehicle or {}).get("attitude_program") or []
        azimuth = program[0].get("azimuth_deg") if program else None
    if azimuth is not None:
        add("Launch", "Azimuth", f"{float(azimuth):.1f}°")

    vehicle = bundle.vehicle or {}
    if vehicle:
        add("Vehicle", "Name", vehicle.get("name", "—"))
        add("Vehicle", "Provenance", vehicle.get("provenance", "—"))
        add("Vehicle", "Stages", len(vehicle.get("stages") or []))
    outputs = run_outputs(bundle)

    def metric(key):
        m = outputs.get(key)
        return None if m is None or not isinstance(m.value, (int, float)) else float(m.value)

    add("Vehicle", "Payload", _fmt_kg(metric("outcome.payload_kg")))

    if target is not None:
        orbit = target.target_orbit
        add("Target", "Orbit", _fmt_orbit(orbit.periapsis_altitude_m / 1000.0,
                                          orbit.apoapsis_altitude_m / 1000.0, orbit.inclination_deg))
        add("Target", "Mission", target.mission.title)
    add("Result", "Insertion orbit", _fmt_orbit(metric("insertion.periapsis_km"),
                                                metric("insertion.apoapsis_km"),
                                                metric("insertion.inclination_deg")))
    t_ins = metric("outcome.insertion_t_s")
    if t_ins is not None:
        add("Result", "Insertion at", f"T+{t_ins:,.0f} s")
    add("Result", "Upper stage at insertion", _fmt_kg(metric("outcome.insertion_mass_kg")))
    opt = bundle.optimization or {}
    if opt:
        add("Result", "Optimizer", f"{opt.get('status', '—')} ({opt.get('objective', '—')})")
    for key, label in (("verdict.mission", "Mission constraints"), ("verdict.physics", "Physics"),
                       ("verdict.optimizer", "Optimizer converged")):
        add("Result", label, outputs[key].value)

    add("Run", "Run id", run.run_id)
    if run.engines:
        add("Run", "Backend", ", ".join(f"{k} {v}" for k, v in sorted(run.engines.items())))
    add("Run", "Folder", str(bundle.run_dir))
    return rows


def _series_value(channel, body_id: str, t: float) -> float | None:
    from orbitopt.flightlab.ui.telemetry import interpolated_value

    if channel is None:
        return None
    series = [s for s in channel.series if s.points and s.points[0][0] <= t + 1e-6 <= s.points[-1][0] + 1e-6]
    pick = next((s for s in series if s.body_id == body_id), series[0] if series else None)
    return None if pick is None else interpolated_value(pick.points, t)


def key_events(run, channels: Sequence | None = None) -> list[KeyEvent]:
    """The run's events in time order, each with the altitude [km] and speed
    [km/s] of its body at that moment (from the telemetry channels)."""
    by_id = {c.id: c for c in (channels or ())}
    out = []
    for event in sorted(run.events, key=lambda e: float(e.t_s)):
        t = float(event.t_s)
        kind = "jettison" if (event.data or {}).get("ot_event") == "jettison" else event.kind
        title = EVENT_TITLES.get(kind) or kind.replace("_", " ").title()
        out.append(KeyEvent(t_s=t, kind=event.kind, title=title, body_id=event.body_id,
                            altitude_km=_series_value(by_id.get("altitude"), event.body_id, t),
                            speed_km_s=_series_value(by_id.get("speed"), event.body_id, t)))
    return out
