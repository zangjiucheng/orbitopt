"""Optimizer candidates as scene data, for side-by-side comparison in the
viewer.

The Mars pipeline searches the departure window, keeps a ranked, diverse
short list (orbitopt.candidate_selection) and flies the list through tudatpy
until one passes verification. Only that one has a high-fidelity
trajectory; the others exist as (departure, time of flight, Lambert delta-v)
points plus the gate's verdict. This module turns each into a scene
``candidates`` entry with its numbers, its fate, and a *screening* path --
the two-body Lambert arc (Sun point mass, JPL low-precision ephemerides)
those numbers describe. The path is labeled as screening fidelity so a viewer
never presents it as a flown trajectory; the exported candidate additionally
names the body whose trail is its high-fidelity flight.

Each path also carries ``bodyPositions``: Earth's and Mars's positions at
every one of the path's own sample times, from the same ephemeris as the arc.
Candidates depart on other dates than the flown mission, outside the span the
scene's planet trails cover, so a viewer centering on a planet subtracts these
per-candidate positions (r_arc(t_i) - r_planet(t_i)) instead of the trails'.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pykep as pk

from orbitopt.bodies import mjd2000_to_ephemeris_seconds, planet
from orbitopt.lambert.gpu_batch import solve_lambert_batch

SCREENING_FIDELITY = "two-body Lambert (Sun point mass, JPL low-precision ephemerides)"
SCREENING_POINTS = 160

# Selection statuses that are distinct trajectories worth drawing. Duplicates
# repeat a better candidate's path; infeasible ones have no path.
_DRAWN_STATUSES = ("selected", "truncated")

# Bodies a viewer can center on (the Mars scene's planet ids, also pykep names).
FRAME_BODY_IDS = ("earth", "mars")


def body_positions_km(body_id: str, mjd2000: np.ndarray) -> np.ndarray:
    """(n, 3) heliocentric ecliptic positions [km] of ``body_id`` at each
    epoch in ``mjd2000`` -- the JPL low-precision ephemerides lambert_arc_km
    uses, so an arc starts exactly on Earth and ends on Mars in their frames."""
    body = planet(body_id)
    return np.array([body.eph(pk.epoch(float(t)))[0] for t in np.asarray(mjd2000, dtype=float)]) / 1000.0


def lambert_arc_km(t0_mjd2000: float, tof_days: float, n_points: int = SCREENING_POINTS) -> np.ndarray:
    """(n_points, 3) heliocentric ecliptic positions [km] along the
    Earth->Mars Lambert transfer leaving at ``t0_mjd2000`` with ``tof_days``,
    from departure to arrival inclusive."""
    earth, mars = planet("earth"), planet("mars")
    r1, _ = earth.eph(pk.epoch(t0_mjd2000))
    r2, _ = mars.eph(pk.epoch(t0_mjd2000 + tof_days))
    r1, r2 = np.asarray(r1, dtype=float), np.asarray(r2, dtype=float)
    tof_s = tof_days * pk.DAY2SEC
    v1 = solve_lambert_batch(r1[None, :], r2[None, :], np.array([tof_s]), pk.MU_SUN, use_gpu=False).v1[0]
    out = np.empty((n_points, 3))
    out[0] = r1
    for k, dt in enumerate(np.linspace(0.0, tof_s, n_points)[1:], start=1):
        r, _v = pk.propagate_lagrangian(r1.tolist(), v1.tolist(), float(dt), pk.MU_SUN)
        out[k] = r
    return out / 1000.0


def build_candidate_entries(
    selection_records: Sequence[dict],
    gate_candidates: Sequence[dict],
    exported_index: int | None,
    reference_et: float,
    flown_body_id: str,
    arc_fn: Callable[[float, float], np.ndarray] = lambert_arc_km,
    body_fn: Callable[[str, np.ndarray], np.ndarray] = body_positions_km,
    frame_body_ids: Sequence[str] = FRAME_BODY_IDS,
) -> list[dict]:
    """Scene ``candidates`` entries, best rank first.

    ``selection_records`` are the select stage's per-candidate records
    (``index``, ``status``, ``rank``, ``reason``, ``t0_mjd2000``,
    ``tof_days``, ``lambert_total_dv_m_s``); ``gate_candidates`` the
    Verification Gate's (``index``, ``verdict``, ``verification``,
    ``reasons``). With no selection records (a pinned transfer skips that
    stage) the gate's candidates stand in. Path times are on the scene
    timeline: days since ``reference_et``. ``body_fn(body_id, mjd2000s)``
    gives each of ``frame_body_ids`` at the path's own epochs (see
    ``screeningPath.bodyPositions``).
    """
    gate = {c["index"]: c for c in gate_candidates}
    records = [r for r in selection_records if r.get("status") in _DRAWN_STATUSES]
    if not records:
        records = [
            {"index": c["index"], "status": "selected", "rank": c.get("selection_rank", 0),
             "t0_mjd2000": c["t0_mjd2000"], "tof_days": c["tof_days"],
             "lambert_total_dv_m_s": c.get("lambert_total_dv_m_s"), "reason": c.get("source", "")}
            for c in gate_candidates if "t0_mjd2000" in c and "tof_days" in c
        ]
    records = sorted(records, key=lambda r: (r.get("rank") is None, r.get("rank") or 0, r["index"]))

    entries = []
    for rec in records:
        index = int(rec["index"])
        t0, tof = float(rec["t0_mjd2000"]), float(rec["tof_days"])
        path = arc_fn(t0, tof)
        start_days = (mjd2000_to_ephemeris_seconds(t0) - reference_et) / 86400.0
        offsets = np.linspace(0.0, tof, len(path))
        times = start_days + offsets
        body_positions = {
            body_id: np.round(np.asarray(body_fn(body_id, t0 + offsets), dtype=float), 1).tolist()
            for body_id in frame_body_ids
        }
        g = gate.get(index, {})
        dv = rec.get("lambert_total_dv_m_s", rec.get("cost"))
        entry = {
            "id": f"candidate-{index}",
            "label": f"Candidate #{index}",
            "searchIndex": index,
            "rank": rec.get("rank"),
            "selection": rec["status"],
            "selectionReason": rec.get("reason", ""),
            # None: never reached the gate (not on the short list).
            "verdict": g.get("verdict"),
            "verification": g.get("verification"),
            "verdictReasons": list(g.get("reasons", [])),
            "exported": index == exported_index,
            "bestEffort": bool(g.get("best_effort", False)),
            "departureMjd2000": t0,
            "tofDays": tof,
            "lambertTotalDvMs": None if dv is None else float(dv),
            "screeningPath": {
                "fidelity": SCREENING_FIDELITY,
                "times": np.round(times, 8).tolist(),
                "positions": np.round(path, 1).tolist(),
                "bodyPositions": body_positions,
            },
        }
        if index == exported_index:
            entry["flownBodyId"] = flown_body_id
        entries.append(entry)
    return entries
