"""R-07: nominal recovery report and the dispersed trajectory envelope.

The Monte Carlo (FL-304, R-04/R-06) answers *how often* the recovery lands;
this module answers *where and how* it lands:

* :func:`nominal_recovery` flies the unperturbed recovery once -- the same
  chain, the same plan options the Monte Carlo warm-starts from, zero
  dispersion -- and :func:`nominal_report` turns that flight into the
  nominal verification report: the gate verdict, every check with its
  plain-language reason, the corridor handoff numbers, the landing
  measurement, the R-02 handoff report and the full booster track
  (separation -> boostback -> entry -> landing burn -> touchdown) in the
  site-ENU frame;
* :func:`envelope_dataset` + :func:`plot_envelope` read the per-sample
  tracks a ``run_monte_carlo(capture_tracks=...)`` run recorded and draw the
  dispersion envelope: every flown ground track in the site-ENU plane, the
  altitude/range band the samples sweep through, and every failure sample
  annotated with its index and governing reason code, coloured by R-06
  failure class;
* :func:`plot_comparison` overlays two configurations (the FL-305 baseline
  and the current stack, typically) flown under identical conditions -- same
  ascent, seed, dispersions, warm start and referee -- so the picture shows
  exactly what the guidance changed.

Track capture is a down-sampled, plain-data view (uniform row selection,
fixed column layout ``[t_s, east_m, north_m, up_m]`` in the site-ENU frame);
it never enters the statistics -- the success rate, histogram and R-06
statistics are computed from the same records with or without it, so a
fixed seed still reproduces the report byte-for-byte.

    from orbitopt.flightlab.recovery.envelope import (
        nominal_recovery, nominal_report, write_nominal_report)
    outcome = nominal_recovery(target, vehicle, ascent_run)
    write_nominal_report(nominal_report(outcome, target), "nominal_report.json")

    mc = run_monte_carlo(target, vehicle, ascent_run, seed=1, n=32,
                         capture_tracks=240, out_path="montecarlo.json")
    data = envelope_dataset(mc.samples, site=site_from_target(target),
                            nominal=track_rows(outcome))
    plot_envelope(data, "envelope.png", title="R-07 dispersion envelope")
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from orbitopt.flightlab import checks, reasons
from orbitopt.flightlab.contract import Check, FlightRun, FlightTarget
from orbitopt.flightlab.frames import geodetic_to_ecef
from orbitopt.flightlab.recovery.boostback import _enu_basis
from orbitopt.flightlab.recovery.handoff_check import RecoveryHandoffReport
from orbitopt.flightlab.recovery.pipeline import fly_recovery
from orbitopt.flightlab.recovery.propagator import BoosterTrack
from orbitopt.flightlab.recovery.stats import FAILURE_CLASSES, classify_failure

__all__ = ["CLASS_COLORS", "TRACK_COLUMNS", "TRACK_SCHEMA",
           "capture_track_rows", "envelope_dataset", "nominal_recovery",
           "landed_touchdowns", "nominal_report", "plot_comparison", "plot_envelope",
           "site_from_target", "track_rows", "write_nominal_report"]

#: Fixed capture column layout -- the record's ``track.rows`` are
#: ``[t_s, east_m, north_m, up_m]`` in the site-ENU frame (up measured from
#: the site's geodetic altitude). Bump on any layout change; readers check it.
TRACK_COLUMNS: tuple[str, ...] = ("t_s", "east_m", "north_m", "up_m")
TRACK_SCHEMA = "r07-track-capture-1"

#: Envelope line colours, keyed by R-06 failure class (landed samples use
#: the neutral ``"landed"`` entry; the nominal flight is always black).
CLASS_COLORS: dict[str, str] = {
    "landed": "#6baed6",
    FAILURE_CLASSES[0]: "#d95f0e",
    FAILURE_CLASSES[1]: "#7570b3",
    FAILURE_CLASSES[2]: "#e7298a",
    FAILURE_CLASSES[3]: "#66a61e",
    FAILURE_CLASSES[4]: "#999999",
}


# ---------------------------------------------------------------------------
# track capture (shared with the Monte Carlo worker)
# ---------------------------------------------------------------------------

def site_from_target(target: FlightTarget) -> tuple[float, float, float]:
    """(lat_deg, lon_deg, alt_m) of the recovery site: the launch site for an
    RTLS target, else the launch site as display origin (the frame origin is
    cosmetic -- the envelope is the same set of tracks either way)."""
    site = target.launch_site
    return (float(site.latitude_deg), float(site.longitude_deg),
            float(site.altitude_wgs84_m))


def _enu_axes(site: Sequence[float]) -> np.ndarray:
    e, n, u = (np.array(c, dtype=np.float64) for c in _enu_basis(site[0], site[1]))
    return np.vstack([e, n, u])


def capture_track_rows(track: BoosterTrack, site: Sequence[float],
                       max_rows: int) -> dict[str, Any]:
    """The flown booster track as the capture format (module docstring):
    ECEF positions projected into the site-ENU frame, uniform down-selection
    to ``max_rows`` rows (first and last row always kept), plain-data rows
    rounded to millimetre resolution so the JSON stays diff-friendly."""
    if max_rows < 2:
        raise ValueError(f"capture needs at least 2 rows, got {max_rows}")
    r = np.asarray(track.position_ecef_m, dtype=np.float64)
    t = np.asarray(track.times_s, dtype=np.float64)
    pad = np.array(geodetic_to_ecef(site[0], site[1], site[2]), dtype=np.float64)
    enu = (r - pad) @ _enu_axes(site).T
    idx = np.unique(np.linspace(0.0, len(t) - 1.0,
                                min(int(max_rows), len(t))).round().astype(int))
    rows = [[round(float(t[i]), 3), round(float(enu[i, 0]), 3),
             round(float(enu[i, 1]), 3),
             round(float(track.height_m_arr[i] - site[2]), 3)] for i in idx]
    return {"schema": TRACK_SCHEMA, "columns": list(TRACK_COLUMNS),
            "n_full": int(len(t)), "rows": rows}


def track_rows(outcome: Any, site: Sequence[float],
               max_rows: int = 2000) -> dict[str, Any] | None:
    """The capture-format track of a ``RecoveryOutcome`` (the full flown
    booster track: FL-302 rollout to the corridor crossing + FL-303 landing
    re-integration when it ran). ``None`` when the plan never flew."""
    if getattr(outcome, "plan", None) is None:
        return None
    from orbitopt.flightlab.recovery.pipeline import _booster_track
    return capture_track_rows(_booster_track(outcome.plan, outcome.landing),
                              site, max_rows)


# ---------------------------------------------------------------------------
# the nominal verification report
# ---------------------------------------------------------------------------

def nominal_recovery(target: FlightTarget, vehicle_path: Path | str,
                     ascent_run: FlightRun, *,
                     plan_options: Mapping[str, Any] | None = None,
                     sep_state: Any = None) -> Any:
    """The nominal recovery flight: the full E3 chain for ``ascent_run`` with
    *no* dispersion. ``plan_options`` defaults to the Monte Carlo's warm
    start (:data:`~orbitopt.flightlab.recovery.montecarlo.MC_PLAN_OPTIONS`),
    so the nominal report and the dispersed population share one setup."""
    from orbitopt.flightlab.recovery.montecarlo import MC_PLAN_OPTIONS
    return fly_recovery(ascent_run, target, vehicle_path,
                        sep_state=sep_state,
                        plan_options=dict(plan_options
                                         if plan_options is not None
                                         else MC_PLAN_OPTIONS))


def _check_dict(c: Check) -> dict[str, Any]:
    out: dict[str, Any] = {"id": c.id, "passed": bool(c.passed),
                           "value": None if c.value is None else float(c.value),
                           "limit": None if c.limit is None else float(c.limit),
                           "unit": c.unit, "reason_code": c.reason_code}
    if c.reason_code:
        out["reason_title"] = reasons.explain(c.reason_code).title_en
    return out


def _handoff_dict(report: RecoveryHandoffReport | None) -> dict[str, Any] | None:
    if report is None:
        return None
    return {"passed": bool(report.passed),
            "phases": dict(report.phases),
            "checks": [_check_dict(c) for c in report.checks],
            "unassessed": [u.to_dict() for u in report.unassessed]}


def nominal_report(outcome: Any, target: FlightTarget, *,
                   max_track_rows: int = 2000) -> dict[str, Any]:
    """The R-07 nominal verification report for one flown recovery: verdict,
    every check (as merged into the run, re-assessed by
    :func:`orbitopt.flightlab.checks.evaluate`), corridor handoff, landing
    measurement, R-02 handoff report and the full booster track in capture
    format. Every field comes from the flight; nothing is summarised away."""
    site = site_from_target(target)
    verdict = checks.evaluate(outcome.run, target)
    plan, landing = outcome.plan, outcome.landing
    corridor = None
    if plan is not None:
        rep = plan.corridor_report
        corridor = {"feasible": bool(plan.feasible),
                    "reason_codes": list(plan.reason_codes),
                    "ignitions_used": int(plan.ignitions_used),
                    "reached": bool(rep.reached), "t_s": rep.t_s,
                    "miss_m": rep.miss_m, "tilt_deg": rep.tilt_deg,
                    "speed_m_s": rep.speed_m_s,
                    "propellant_available_kg": rep.propellant_available_kg,
                    "landing_need_kg": rep.landing_need_kg,
                    "reserve_kg": rep.reserve_kg,
                    "entry_max_q_pa": rep.entry_max_q_pa,
                    "entry_max_proper_g": rep.entry_max_proper_g,
                    "landing_divert_budget_m": rep.landing_divert_budget_m,
                    "min_landing_reserve_margin_kg":
                        rep.min_landing_reserve_margin_kg}
    land = None
    if landing is not None:
        land = {"feasible": bool(landing.feasible), "landed": bool(landing.landed),
                "reason_codes": list(landing.reason_codes),
                "position_error_m": landing.position_error_m,
                "vertical_speed_m_s": landing.vertical_speed_m_s,
                "lateral_speed_m_s": landing.lateral_speed_m_s,
                "touchdown_enu_m": (list(landing.touchdown_enu_m)
                                    if landing.touchdown_enu_m is not None else None),
                "max_tilt_deg": landing.max_tilt_deg,
                "max_proper_g": float(landing.max_proper_g),
                "ignitions_used": int(landing.ignitions_used)}
        if getattr(landing, "actuator_margin", None) is not None:
            land["actuator_margin"] = landing.actuator_margin.to_dict()
    return {
        "schemaVersion": "1.0",
        "kind": "flightlab-recovery-nominal-report",
        "recovered": bool(outcome.success),
        "reason_codes": list(outcome.reason_codes),
        "target": {"recovery_kind": target.recovery.kind,
                   "site": [site[0], site[1], site[2]]},
        "verdict": {
            "optimizer_converged": verdict.optimizer_converged,
            "physics_verified": verdict.physics_verified,
            "mission_constraints_passed": verdict.mission_constraints_passed,
        },
        "checks": [_check_dict(c) for c in outcome.run.verification.checks],
        "corridor": corridor,
        "landing": land,
        "handoff": _handoff_dict(outcome.handoff),
        "track": track_rows(outcome, site, max_rows=max_track_rows),
        "assumptions": list(outcome.assumptions),
    }


def write_nominal_report(report: Mapping[str, Any], path: Path | str) -> None:
    """Sorted-key JSON of :func:`nominal_report` -- byte-identical for
    identical flights (the FL-304 determinism contract, extended to R-07)."""
    Path(path).write_text(json.dumps(dict(report), sort_keys=True, indent=1)
                          + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# the dispersion envelope
# ---------------------------------------------------------------------------

def envelope_dataset(samples: Sequence[Mapping[str, Any]],
                     site: Sequence[float], *,
                     nominal: Mapping[str, Any] | None = None,
                     bins: int = 32) -> dict[str, Any]:
    """The plotting dataset for the dispersion envelope, built from Monte
    Carlo sample records that carry capture tracks (``capture_tracks=``).
    Pure data (no matplotlib): the captured rows and per-trajectory summaries
    (governing reason, R-06 failure class, touchdown point, max range), the
    binned min/max altitude band across the population, and the ground-view
    extents.

    ``bins`` grids the range-to-site axis for the altitude band; a bin needs
    at least two sample tracks inside it to report min/max, so sparse tails
    read as gaps rather than invented extremes."""
    tracks: list[dict[str, Any]] = []
    rows_by_track: list[np.ndarray] = []
    for rec in samples:
        cap = rec.get("track")
        if not cap or not cap.get("rows"):
            continue
        rows = np.asarray(cap["rows"], dtype=np.float64)
        en = rows[:, 1:3]
        touchdown = None
        if rec.get("touchdown_east_m") is not None:
            touchdown = [float(rec["touchdown_east_m"]),
                         float(rec["touchdown_north_m"])]
        tracks.append({
            "idx": int(rec["idx"]), "outcome": str(rec.get("outcome", "")),
            "reason": str(rec.get("reason", "")),
            "failure_class": (None if rec.get("outcome") == "landed"
                              else classify_failure(str(rec.get("reason", "")))),
            "touchdown_enu_m": touchdown,
            "last_enu_m": [float(rows[-1, 1]), float(rows[-1, 2])],
            "max_range_m": float(np.max(np.hypot(en[:, 0], en[:, 1]))),
            "min_up_m": float(np.min(rows[:, 3]))})
        rows_by_track.append(rows)
    order = sorted(range(len(tracks)), key=lambda i: tracks[i]["idx"])
    tracks = [tracks[i] for i in order]

    band = None
    if rows_by_track:
        all_r = np.concatenate([np.hypot(r[:, 1], r[:, 2]) for r in rows_by_track])
        edges = np.unique(np.quantile(all_r, np.linspace(0.0, 1.0, int(bins) + 1)))
        centers, lo, hi = [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            alt_in_bin: list[float] = []
            for r in rows_by_track:
                rr = np.hypot(r[:, 1], r[:, 2])
                m = ((rr >= a) & (rr < b)) if b < edges[-1] else ((rr >= a) & (rr <= b))
                if np.any(m):
                    alt_in_bin.extend(r[m, 3].tolist())
            if len(alt_in_bin) >= 2:
                centers.append(0.5 * (a + b) / 1000.0)
                lo.append(min(alt_in_bin) / 1000.0)
                hi.append(max(alt_in_bin) / 1000.0)
        band = {"kind": "altitude_km_vs_range_to_site_km",
                "range_km": centers, "alt_min_km": lo, "alt_max_km": hi}
    radii = [math.hypot(*t["touchdown_enu_m"]) for t in tracks
             if t["touchdown_enu_m"] is not None]
    return {"kind": "flightlab-recovery-envelope",
            "site": [float(site[0]), float(site[1]), float(site[2])],
            "n_captured": len(tracks),
            "trajectories": [{**t, "rows": _rows_by_idx(rows_by_track, order, i)}
                             for i, t in enumerate(tracks)],
            "band": band,
            "ground": {"max_touchdown_radius_m": max(radii, default=0.0)},
            "nominal": None if nominal is None else dict(nominal)}


def _rows_by_idx(rows_by_track: Sequence[np.ndarray], order: Sequence[int],
                 i: int) -> list[list[float]]:
    r = rows_by_track[order[i]]
    return [[round(float(x), 3) for x in row] for row in r]


def _nominal_xy(nominal: Mapping[str, Any] | None) -> np.ndarray | None:
    if not nominal or not nominal.get("rows"):
        return None
    return np.asarray(nominal["rows"], dtype=np.float64)


def plot_envelope(data: Mapping[str, Any], path: Path | str, *,
                  title: str = "Recovery dispersion envelope (R-07)") -> Path:
    """Render :func:`envelope_dataset`: left, every flown ground track in the
    site-ENU plane (nominal black, landed samples blue, failures coloured by
    R-06 class and annotated with ``idx REASON`` at their last flown point,
    plus the max-touchdown-radius ring); right, the altitude-vs-range-to-site
    band the population sweeps, with the nominal profile over it. Matplotlib
    is imported locally (Agg), so headless imports of this module never
    require it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axg, axp) = plt.subplots(1, 2, figsize=(13.0, 6.0))
    plotted_classes: dict[str, int] = {}
    for tr in data["trajectories"]:
        rows = np.asarray(tr["rows"], dtype=np.float64)
        landed = tr["outcome"] == "landed"
        cls = "landed" if landed else str(tr["failure_class"])
        color = CLASS_COLORS.get(cls, CLASS_COLORS["unclassified"])
        axg.plot(rows[:, 1], rows[:, 2], color=color, lw=0.7, alpha=0.65)
        plotted_classes[cls] = plotted_classes.get(cls, 0) + 1
        if landed and tr["touchdown_enu_m"] is not None:
            axg.plot(tr["touchdown_enu_m"][0], tr["touchdown_enu_m"][1],
                     ".", color=color, ms=4)
        else:
            axg.annotate(f"{tr['idx']} {tr['reason']}", tr["last_enu_m"],
                         fontsize=5.5, color=color,
                         textcoords="offset points", xytext=(2.5, 2.5))
            axg.plot(tr["last_enu_m"][0], tr["last_enu_m"][1], "x",
                     color=color, ms=4)
    nom = _nominal_xy(data.get("nominal"))
    extra_handles = [plt.Line2D([], [], marker="+", ls="", color="red",
                                ms=10, label="site")]
    if nom is not None:
        axg.plot(nom[:, 1], nom[:, 2], color="black", lw=1.6, label="nominal")
        extra_handles.append(plt.Line2D([], [], color="black", lw=1.6,
                                        label="nominal"))
    r_max = data["ground"]["max_touchdown_radius_m"]
    if r_max > 0.0:
        th = np.linspace(0.0, 2.0 * np.pi, 128)
        axg.plot(r_max * np.cos(th), r_max * np.sin(th), ls=":", lw=0.8,
                 color="black")
        extra_handles.append(plt.Line2D([], [], ls=":", lw=0.8, color="black",
                                        label=f"max touchdown radius {r_max:.0f} m"))
    axg.set_xlabel("east [m]")
    axg.set_ylabel("north [m]")
    axg.set_title("ground tracks (site ENU)")
    handles = [plt.Line2D([], [], color=c, lw=1.4,
                          label=f"{k} (n={plotted_classes[k]})")
               for k in FAILURE_CLASSES + ("landed",)
               if (k in plotted_classes) and (c := CLASS_COLORS.get(k))]
    axg.legend(handles=handles + extra_handles, fontsize=6.5, loc="best")

    band = data.get("band")
    if band and band["range_km"]:
        x = np.asarray(band["range_km"])
        axp.fill_between(x, band["alt_min_km"], band["alt_max_km"],
                         color="#6baed6", alpha=0.35, label="sample band")
        if nom is not None:
            r_km = np.hypot(nom[:, 1], nom[:, 2]) / 1000.0
            axp.plot(r_km, nom[:, 3] / 1000.0,
                     color="black", lw=1.6, label="nominal")
        axp.set_xlabel("range to site [km]")
        axp.set_ylabel("altitude above site [km]")
        axp.set_title("altitude / range envelope")
        axp.legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def landed_touchdowns(data: Mapping[str, Any]) -> list[list[float]]:
    """Site-ENU touchdown points of the *landed* samples only. A failed
    sample can still carry a ground-contact point (a crash or a miss far
    from the pad); those are failures, not part of the touchdown
    dispersion, so the comparison scatter and its sigma ellipses use this
    list and match the "landed samples" they are labelled with."""
    return [tr["touchdown_enu_m"] for tr in data["trajectories"]
            if tr["outcome"] == "landed" and tr["touchdown_enu_m"] is not None]


def plot_comparison(runs: Sequence[Mapping[str, Any]], path: Path | str, *,
                    title: str = "Recovery comparison under identical "
                                "conditions") -> Path:
    """Side-by-side view of two configurations flown under identical
    conditions (same ascent, seed, dispersions, warm start and referee).
    Each run entry: ``label``, ``dataset`` (:func:`envelope_dataset` output),
    ``nominal`` capture dict, ``success_rate`` and optional ``ci`` (the R-06
    Wilson interval). Left: both nominal tracks; right: both touchdown
    scatters with 1/3-sigma ellipses and the success-rate interval per run.
    The per-run failure annotation lives on the envelope figures
    (:func:`plot_envelope`); this panel compares."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    # Runs flown under identical dispersions often land on the very same
    # point; distinct marker shapes (hollow first) and line styles keep the
    # earlier run visible under the later one.
    markers = ["o", "x", "^", "s"]
    line_styles = ["-", "--", "-.", ":"]
    fig, (axg, axt) = plt.subplots(1, 2, figsize=(13.0, 6.0))
    for i, run in enumerate(runs):
        color = colors[i % len(colors)]
        nom = _nominal_xy(run.get("nominal"))
        if nom is not None:
            axg.plot(nom[:, 1], nom[:, 2], color=color, lw=1.6,
                     ls=line_styles[i % len(line_styles)],
                     label=f"nominal {run['label']}")
        pts = landed_touchdowns(run["dataset"])
        if pts:
            marker = markers[i % len(markers)]
            axt.plot([p[0] for p in pts], [p[1] for p in pts], marker,
                     ls="", ms=6 if marker == "o" else 5, mec=color,
                     mfc="none" if marker == "o" else color, mew=1.0,
                     label=f"{run['label']} touchdowns (n={len(pts)})")
        if len(pts) >= 3:
            p = np.asarray(pts)
            cov = np.cov(p.T)
            vals, vecs = np.linalg.eigh(cov)
            major = int(vals.argmax())
            ang = math.degrees(math.atan2(vecs[0, major], vecs[1, major]))
            for scale, ls in ((1.0, "-"), (3.0, "--")):
                axt.add_patch(Ellipse(
                    p.mean(axis=0),
                    2.0 * scale * math.sqrt(max(vals[1], 0.0)),
                    2.0 * scale * math.sqrt(max(vals[0], 0.0)),
                    angle=ang, fill=False, ls=ls, ec=color, lw=0.8))
        ci = run.get("ci") or {}
        lo = ci.get("lower", ci.get("low"))
        hi = ci.get("upper", ci.get("high"))
        rate = float(run.get("success_rate", float("nan")))
        txt = (f"{run['label']}: {rate:.2f}"
               + (f" 95% CI [{lo:.2f}, {hi:.2f}]" if lo is not None else ""))
        axt.annotate(txt, xy=(0.02, 0.97 - 0.07 * i),
                     xycoords="axes fraction", fontsize=7.5, color=color)
    axg.plot(0.0, 0.0, marker="+", color="black", ms=10)
    axg.set_xlabel("east [m]")
    axg.set_ylabel("north [m]")
    axg.set_title("nominal tracks")
    axg.legend(fontsize=7)
    axt.set_xlabel("east [m]")
    axt.set_ylabel("north [m]")
    axt.set_title("touchdown scatter (landed samples)")
    axt.legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
