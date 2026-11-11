"""Re-run the FL-403 closed-loop tracking spike and regenerate its report artifacts (task #208).

Fly the JSBSim spike stage closed-loop on the 3-DOF attitude program at
dt = 0.01 and 0.005 s, then write:

* ``docs/benchmarks/fl403_tracking_spike.json``  -- all metrics (both dts)
* ``docs/assets/fl403_tracking.png``             -- attitude error + MECO bars
* ``docs/assets/fl403_gimbal.png``               -- gimbal cmd/actual, authority
* ``docs/assets/fl403_meco_deviation.png``       -- 6-DOF vs 3-DOF at MECO

Usage (needs ``jsbsim`` and ``matplotlib``; the 3-DOF reference fixture must
be recorded, see ``scripts/fl403_record_reference.py``)::

    PYTHONPATH=src python scripts/fl403_run_spike.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from orbitopt.flightlab.adapters.opentsiolkovsky import Vehicle  # noqa: E402
from orbitopt.flightlab.sixdof.tracking import (  # noqa: E402
    MIN_THRUST_FRACTION,
    RATED_THRUST_N,
    AttitudeProgram,
    ReferenceTrajectory,
    chart_error_deg,
    flight_path_angle,
    run_tracking_spike,
)

FIXTURE = Path("tests/fixtures/flightlab/opentsiolkovsky/fl403_spike")
ASSETS = Path("docs/assets")
OUT_JSON = Path("docs/benchmarks/fl403_tracking_spike.json")
SETTLE_T_S = 60.0
MECO_T_S = 100.0

D_ALT_TOL_M, D_SPEED_TOL_M_S, D_FPA_TOL_DEG = 6000.0, 200.0, 1.0


def _settled(res):
    errs = [chart_error_deg(s, res.program) for s in res.samples if s["t_s"] >= SETTLE_T_S]
    return {"att_err_max_deg": round(max(errs), 3),
            "att_err_rms_deg": round(math.sqrt(sum(e * e for e in errs) / len(errs)), 3)}


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
        ax.semilogy(ts, es, style, lw=1.0, label=f"{dt_key} s (max {results[dt_key].metrics.att_err_max_deg:.1f}°)")
    ax.axvline(SETTLE_T_S, color="grey", lw=0.8)
    ax.text(SETTLE_T_S + 0.7, 1e-2, "settled\nwindow", fontsize=7, color="grey")
    ax.axhline(2.0, color="tab:red", lw=0.8, ls=":")
    ax.text(2, 2.3, "settled max bound 2°", fontsize=7, color="tab:red")
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 40)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("attitude error vs program [deg]")
    ax.set_title("FL-403: PID gimbal tracking of the 3-DOF attitude program\n"
                 "early peak ~24° is the program-infeasibility transient near vertical (t<60 s); "
                 "settled ≤1.2°", fontsize=8)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)

    m = results["dt0.01"].metrics
    names = ["d_alt [m]\n(tol ±%d)" % D_ALT_TOL_M, "d_speed [m/s]\n(tol ±%d)" % D_SPEED_TOL_M_S,
             "d_fpa [deg]\n(tol ±%.1f)" % D_FPA_TOL_DEG]
    vals = [m.d_alt_m, m.d_speed_m_s, m.d_fpa_deg]
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
    m = res.metrics
    axp.set_title(f"FL-403: TVC gimbals (dotted = ±5° travel) — saturation {m.gimbal_sat_fraction_of_burn:.1%} "
                  f"of burn, max {m.gimbal_max_deg:.2f}°", fontsize=9)
    thr = [s["thrust_n"] / RATED_THRUST_N for s in res.samples]
    axa.plot(ts, thr, lw=0.8, color="tab:green")
    axa.axhline(MIN_THRUST_FRACTION, color="grey", ls=":", lw=0.8)
    axa.axvspan(MECO_T_S, ts[-1], color="grey", alpha=0.25)
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
    meco = min(res.samples, key=lambda s: abs(s["t_s"] - MECO_T_S))
    ts = [s["t_s"] for s in res.samples]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(11, 3.4), dpi=150)
    r = [ref.at(t) for t in ts]
    a1.plot(ts, [res.alt_climb_m(s) for s in res.samples], lw=1.0, color="tab:orange",
            label="6-DOF climb |r|-|r0|")
    a1.plot(ts, [i["altitude_m"] for i in r], lw=1.0, ls="--", color="black", label="3-DOF (OT)")
    a1.set_xlabel("t [s]")
    a1.set_ylabel("climb above launch [m]")
    a1.set_title(f"altitude: 6-DOF {res.alt_climb_m(meco)/1000:.1f} km vs 3-DOF "
                 f"{ref.at(MECO_T_S)['altitude_m']/1000:.1f} km", fontsize=8)
    a1.legend(fontsize=7)
    a2.plot(ts, [math.hypot(s["vx_m_s"], math.hypot(s["vy_m_s"], s["vz_m_s"])) for s in res.samples],
           lw=1.0, color="tab:orange", label="6-DOF (ECEF)")
    a2.plot(ts, [i["speed_earth_m_s"] for i in r], lw=1.0, ls="--", color="black", label="3-DOF (OT)")
    a2.set_xlabel("t [s]")
    a2.set_ylabel("speed [m/s]")
    a2.set_title("velocity", fontsize=8)
    m = res.metrics
    fpa6 = [math.degrees(flight_path_angle(s)) for s in res.samples]
    a3.plot(ts, fpa6, lw=1.0, color="tab:orange", label="6-DOF")
    a3.plot(ts, [i["fpa_deg"] for i in r], lw=1.0, ls="--", color="black", label="3-DOF (OT)")
    a3.set_xlabel("t [s]")
    a3.set_ylabel("FPA [deg]")
    a3.set_title(f"flight-path angle @MECO: {m.fpa_6dof_deg:.1f}° vs {m.fpa_3dof_deg:.1f}°", fontsize=8)
    for ax in (a1, a2, a3):
        ax.grid(alpha=0.3)
    a1.legend(fontsize=7, loc="lower right")
    a2.legend(fontsize=7, loc="lower right")
    a3.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(ASSETS / "fl403_meco_deviation.png")
    plt.close(fig)


def main() -> int:
    veh = Vehicle.load(FIXTURE / "ascent_vehicle.yaml")
    prog = AttitudeProgram.from_vehicle(veh)
    ref = ReferenceTrajectory.from_csv(FIXTURE / "ot_out" / "trajectory.csv")
    meco = float(veh.stages[0]["engine"]["burn_end_s"])

    results = {}
    for dt in (0.01, 0.005):
        results[dt] = run_tracking_spike(ref, prog, meco, dt_s=dt)
        print(f"dt={dt}: max {results[dt].metrics.att_err_max_deg:.2f}°, "
              f"sat {results[dt].metrics.gimbal_sat_fraction_of_burn:.1%}, "
              f"d_alt {results[dt].metrics.d_alt_m:+.0f} m, finite={results[dt].metrics.finite}")

    out = {f"dt{k}": vars(res.metrics) for k, res in results.items()}
    out["_meta"] = {
        "task": "FL-403 (#208)",
        "model": "spike_stage",
        "reference": "OT 3-DOF fl403_spike fixture (e2=18 deg)",
        "gains": {"kp": 3.0, "ki": 0.01, "kd": 4.0, "i_max_alpha_rad_s2": 0.05},
        "settled_window_t_ge_60s": {f"dt_{k}": _settled(res) for k, res in results.items()},
        "notes": "att_err_max/rms fields cover t in [0.5,100]; the large early-window numbers "
                 "are the documented program-infeasibility transient near vertical (el-rate "
                 "demanded at el~90 is unachievable by TVC at low speed).",
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
