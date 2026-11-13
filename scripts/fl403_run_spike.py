"""Regenerate the FL-403 report artifacts from the FORMAL 6-DOF backend (F-01, task #226).

The FL-403 spike (#208) used to fly through this script's own loop -- that
was an isolated experiment entry point. It no longer is one: the scenario
lives in the packaged formal config
(``orbitopt/data/flightlab/sixdof/fl403_program_tracking.yaml``,
``backend="6DOF"``), and this script is now a *report generator* that flies
that config through ``orbitopt.flightlab.adapters.sixdof.run_ascent`` at
dt = 0.01 and 0.005 s and renders the same artifacts from the resulting
run directories:

* ``docs/benchmarks/fl403_tracking_spike.json``  -- all metrics (both dts)
* ``docs/assets/fl403_tracking.png``             -- attitude error + MECO bars
* ``docs/assets/fl403_gimbal.png``               -- gimbal cmd/actual, authority
* ``docs/assets/fl403_meco_deviation.png``       -- 6-DOF vs 3-DOF at MECO

(The 3-DOF reference CSV itself is recorded by
``scripts/fl403_record_reference.py`` -- a fixture-regeneration tool, not a
flight path.)

Usage (needs ``jsbsim`` and ``matplotlib``)::

    PYTHONPATH=src python scripts/fl403_run_spike.py
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from orbitopt.flightlab.adapters.sixdof import (  # noqa: E402
    load_config,
    packaged_config,
    run_ascent,
)
from orbitopt.flightlab.sixdof.tracking import (  # noqa: E402
    MIN_THRUST_FRACTION,
    RATED_THRUST_N,
    AttitudeProgram,
    ReferenceTrajectory,
    chart_error_deg,
    flight_path_angle,
)

FIXTURE = Path("tests/fixtures/flightlab/opentsiolkovsky/fl403_spike")
ASSETS = Path("docs/assets")
OUT_JSON = Path("docs/benchmarks/fl403_tracking_spike.json")
GIMBAL_TRAVEL_RAD = 0.0873  # XML clipto, same constant tracking._metrics uses

D_ALT_TOL_M, D_SPEED_TOL_M_S, D_FPA_TOL_DEG = 6000.0, 200.0, 1.0


def _samples_from_track(track):
    """The run's BodyTrack (+ its preserved extra columns) back into the
    sample dicts the plot code consumes -- the same keys the spike's
    ``SixDofSim.samples`` carried."""
    rows = []
    for i, t in enumerate(track.t_s):
        s = {"t_s": t, "mass_kg": track.mass_kg[i],
             "x_m": track.position_m[i][0], "y_m": track.position_m[i][1],
             "z_m": track.position_m[i][2],
             "vx_m_s": track.velocity_m_s[i][0], "vy_m_s": track.velocity_m_s[i][1],
             "vz_m_s": track.velocity_m_s[i][2]}
        for key, values in track.extra_columns.items():
            s[key] = values[i]
        rows.append(s)
    return rows


class _Result:
    """Minimal stand-in for the spike's TrackingResult, rebuilt from a
    formal run directory (the plots and the JSON keys are unchanged)."""

    def __init__(self, run, program, reference, dt_s):
        self.checks = run.verification.checks  # the run is the metrics
        self.program = program
        self.reference = reference
        track = run.tracks[0]
        self.samples = _samples_from_track(track)
        self.dt_s = dt_s
        self._r0 = math.sqrt(sum(c * c for c in track.position_m[0]))
        self._meco_t = next(e.t_s for e in run.events if e.kind == "meco")

    def check(self, cid):
        return next(c for c in self.checks if c.id == cid)

    def alt_climb_m(self, s):
        return math.sqrt(s["x_m"] ** 2 + s["y_m"] ** 2 + s["z_m"] ** 2) - self._r0

    @property
    def meco_sample(self):
        return min(self.samples, key=lambda s: abs(s["t_s"] - self._meco_t))


def _metrics_dict(res):
    """The FL-403 report numbers, read back from the formal run's checks
    (signed deviation fields recomputed against the reference: the checks
    carry absolute values by design)."""
    meco = res.meco_sample
    ref = res.reference.at(res._meco_t)
    climb = res.alt_climb_m(meco)
    speed = math.sqrt(meco["vx_m_s"] ** 2 + meco["vy_m_s"] ** 2 + meco["vz_m_s"] ** 2)
    fpa = math.degrees(flight_path_angle(meco))
    settled = [chart_error_deg(s, res.program) for s in res.samples
               if s["t_s"] >= 60.0]
    burn_errs = [chart_error_deg(s, res.program) for s in res.samples
                 if 0.5 <= s["t_s"] <= res._meco_t]
    burn = [s for s in res.samples if s["t_s"] <= res._meco_t]
    sat = sum(1 for s in burn
              if abs(s["gimbal_pitch_actual_rad"]) >= GIMBAL_TRAVEL_RAD - 1e-6
              or abs(s["gimbal_yaw_actual_rad"]) >= GIMBAL_TRAVEL_RAD - 1e-6)
    return {
        "dt_s": res.dt_s,
        "meco_t_s": res._meco_t,
        "att_err_max_deg": res.check("sixdof-att-tracking-full-max").value,
        "att_err_rms_deg": math.sqrt(sum(e * e for e in burn_errs) / len(burn_errs))
                                 if burn_errs else 0.0,
        "gimbal_sat_fraction_of_burn": sat / len(burn) if burn else 0.0,
        "gimbal_max_deg": max(math.degrees(max(abs(s["gimbal_pitch_actual_rad"]),
                                               abs(s["gimbal_yaw_actual_rad"])))
                              for s in res.samples),
        "settled_max_deg": res.check("sixdof-att-tracking-settled-max").value,
        "settled_rms_deg": res.check("sixdof-att-tracking-settled-rms").value,
        "settled_window_t_ge_60s": {"att_err_max_deg": round(max(settled), 3),
                                    "att_err_rms_deg": round(
                                        math.sqrt(sum(e * e for e in settled)
                                                  / len(settled)), 3)},
        "alt_climb_6dof_m": climb, "alt_3dof_m": ref["altitude_m"],
        "d_alt_m": climb - ref["altitude_m"],
        "speed_6dof_m_s": speed, "speed_3dof_m_s": ref["speed_earth_m_s"],
        "d_speed_m_s": speed - ref["speed_earth_m_s"],
        "fpa_6dof_deg": fpa, "fpa_3dof_deg": ref["fpa_deg"],
        "d_fpa_deg": fpa - ref["fpa_deg"],
        "physics_verified": all(c.passed for c in res.checks),
    }


def _err_curve(res):
    ts, es = [], []
    for s in res.samples:
        if s["t_s"] >= 0.5:
            ts.append(s["t_s"])
            es.append(max(chart_error_deg(s, res.program), 1e-4))
    return ts, es


def _plot_tracking(results):
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(10, 3.6), dpi=150,
                                  gridspec_kw={"width_ratios": [2.6, 1]})
    for dt_key, style in (("dt0.01", "-"), ("dt0.005", "--")):
        ts, es = _err_curve(results[dt_key])
        ax.semilogy(ts, es, style, lw=1.0,
                    label=f"{dt_key} s (max {results[dt_key].check('sixdof-att-tracking-full-max').value:.1f}°)")
    settle = 60.0
    ax.axvline(settle, color="grey", lw=0.8)
    ax.text(settle + 0.7, 2e-3, "settled\nwindow", fontsize=7, color="grey")
    ax.axhline(2.0, color="tab:red", lw=0.8, ls=":")
    ax.text(2, 2.3, "max bound 2° (whole burn)", fontsize=7, color="tab:red")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 5)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("attitude error vs program [deg]")
    full = results["dt0.01"].check("sixdof-att-tracking-full-max").value
    ax.set_title("FL-403: PID gimbal tracking of the 3-DOF attitude program\n"
                 f"full profile max {full:.2f}° "
                 "(pitch-over start at t=8 s); bound 2° over the whole burn", fontsize=8)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)

    m = _metrics_dict(results["dt0.01"])
    names = ["d_alt [m]\n(tol ±%d)" % D_ALT_TOL_M, "d_speed [m/s]\n(tol ±%d)" % D_SPEED_TOL_M_S,
             "d_fpa [deg]\n(tol ±%.1f)" % D_FPA_TOL_DEG]
    vals = [m["d_alt_m"], m["d_speed_m_s"], m["d_fpa_deg"]]
    tols = [D_ALT_TOL_M, D_SPEED_TOL_M_S, D_FPA_TOL_DEG]
    bars = axb.bar(range(3), vals, color="tab:orange")
    for i, (b, tol) in enumerate(zip(bars, tols)):
        axb.hlines([tol, -tol], i - 0.4, i + 0.4, color="grey", lw=0.8)
        axb.text(i, vals[i] - (2500 if i == 0 else 0), f"{vals[i]:+.1f}",
                 ha="center", fontsize=7,
                 va="top" if i == 0 else "bottom")
    axb.axhline(0, color="black", lw=0.8)
    axb.set_xticks(range(3), names, fontsize=7)
    axb.set_title("MECO (t=100 s) deviation vs 3-DOF ref", fontsize=8)
    axb.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(ASSETS / "fl403_tracking.png")
    plt.close(fig)


def _plot_gimbal(res):
    ts = [s["t_s"] for s in res.samples]
    fig, (axp, axy, axa) = plt.subplots(3, 1, figsize=(10, 6), dpi=150, sharex=True)
    for ax, c, a, name in ((axp, "gimbal_pitch_cmd_rad", "gimbal_pitch_actual_rad", "pitch"),
                           (axy, "gimbal_yaw_cmd_rad", "gimbal_yaw_actual_rad", "yaw")):
        ax.plot(ts, [math.degrees(s[c]) for s in res.samples], lw=0.8, color="tab:orange",
                label=f"{name} cmd")
        ax.plot(ts, [math.degrees(s[a]) for s in res.samples], lw=0.8, color="black",
                label=f"{name} actual")
        ax.axhline(5.0, color="grey", ls=":", lw=0.8)
        ax.axhline(-5.0, color="grey", ls=":", lw=0.8)
        ax.set_ylabel("deg")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    m = _metrics_dict(res)
    axp.set_title(f"FL-403: TVC gimbals (dotted = ±5° travel) — saturation {m['gimbal_sat_fraction_of_burn']:.1%} "
                  f"of burn, max {m['gimbal_max_deg']:.2f}°", fontsize=9)
    thr = [s["thrust_n"] / RATED_THRUST_N for s in res.samples]
    axa.plot(ts, thr, lw=0.8, color="tab:green")
    axa.axhline(MIN_THRUST_FRACTION, color="grey", ls=":", lw=0.8)
    axa.axvspan(res._meco_t, ts[-1], color="grey", alpha=0.25)
    axa.text(102.5, 0.5, "MECO: no authority\n→ neutral", fontsize=7)
    axa.set_ylabel("thrust / rated")
    axa.set_xlabel("t [s]")
    axa.grid(alpha=0.3)
    axa.set_title("thrust authority for the gain schedule", fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "fl403_gimbal.png")
    plt.close(fig)


def _plot_meco(res):
    ref = res.reference
    meco = res.meco_sample
    ts = [s["t_s"] for s in res.samples]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(11, 3.4), dpi=150)
    r = [ref.at(t) for t in ts]
    a1.plot(ts, [res.alt_climb_m(s) for s in res.samples], lw=1.0, color="tab:orange",
            label="6-DOF climb |r|-|r0|")
    a1.plot(ts, [i["altitude_m"] for i in r], lw=1.0, ls="--", color="black", label="3-DOF (OT)")
    a1.set_xlabel("t [s]")
    a1.set_ylabel("climb above launch [m]")
    a1.set_title(f"altitude: 6-DOF {res.alt_climb_m(meco)/1000:.1f} km vs 3-DOF "
                 f"{ref.at(res._meco_t)['altitude_m']/1000:.1f} km", fontsize=8)
    a1.legend(fontsize=7)
    a2.plot(ts, [math.hypot(s["vx_m_s"], math.hypot(s["vy_m_s"], s["vz_m_s"])) for s in res.samples],
            lw=1.0, color="tab:orange", label="6-DOF (ECEF)")
    a2.plot(ts, [i["speed_earth_m_s"] for i in r], lw=1.0, ls="--", color="black", label="3-DOF (OT)")
    a2.set_xlabel("t [s]")
    a2.set_ylabel("speed [m/s]")
    a2.set_title("velocity", fontsize=8)
    m = _metrics_dict(res)
    fpa6 = [math.degrees(flight_path_angle(s)) for s in res.samples]
    a3.plot(ts, fpa6, lw=1.0, color="tab:orange", label="6-DOF")
    a3.plot(ts, [i["fpa_deg"] for i in r], lw=1.0, ls="--", color="black", label="3-DOF (OT)")
    a3.set_xlabel("t [s]")
    a3.set_ylabel("FPA [deg]")
    a3.set_title(f"flight-path angle @MECO: {m['fpa_6dof_deg']:.1f}° vs {m['fpa_3dof_deg']:.1f}°",
                 fontsize=8)
    for ax in (a1, a2, a3):
        ax.grid(alpha=0.3)
    a1.legend(fontsize=7, loc="lower right")
    a2.legend(fontsize=7, loc="lower right")
    a3.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(ASSETS / "fl403_meco_deviation.png")
    plt.close(fig)


def main() -> int:
    config_path = packaged_config("fl403_program_tracking")
    base_doc = load_config(config_path).raw
    ref = ReferenceTrajectory.from_csv(FIXTURE / "ot_out" / "trajectory.csv")

    results = {}
    for dt in (0.01, 0.005):
        doc = dict(base_doc, dt_s=dt)
        cfg = load_config(doc)
        with tempfile.TemporaryDirectory() as tmp:
            run = run_ascent(cfg, Path(tmp) / f"fl403-dt{dt}")
        prog = AttitudeProgram(
            times=tuple(p[0] for p in cfg.pid.program),
            azimuth_deg=tuple(p[1] for p in cfg.pid.program),
            elevation_deg=tuple(p[2] for p in cfg.pid.program))
        res = _Result(run, prog, ref, dt)
        results[dt] = res
        m = _metrics_dict(res)
        print(f"dt={dt}: max {m['att_err_max_deg']:.2f}°, "
              f"sat {m['gimbal_sat_fraction_of_burn']:.1%}, "
              f"d_alt {m['d_alt_m']:+.0f} m, physics_verified={m['physics_verified']}")

    out = {f"dt{k}": _metrics_dict(res) for k, res in results.items()}
    out["_meta"] = {
        "task": "FL-403 (#208), re-run through the formal 6-DOF backend (F-01, #226)",
        "config": "src/orbitopt/data/flightlab/sixdof/fl403_program_tracking.yaml",
        "model": "spike_stage",
        "reference": "OT 3-DOF fl403_spike fixture (e2=18 deg)",
        "gains": dict(base_doc["control"]["gains"]),
        "notes": "att_err_max covers t in [0.5,100]; settled window t>=60 s. "
                 "MECO now cuts the throttle exactly at meco_t_s (the spike's "
                 "nominal 5 s coast ran open throttle; metrics stop at MECO, so "
                 "the MECO-time numbers are unchanged).",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUT_JSON}")

    keyed = {f"dt{k}": res for k, res in results.items()}
    _plot_tracking(keyed)
    _plot_gimbal(results[0.01])
    _plot_meco(results[0.01])
    print(f"wrote {ASSETS}/fl403_tracking.png, fl403_gimbal.png, fl403_meco_deviation.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
