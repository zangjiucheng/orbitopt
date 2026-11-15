"""Benchmark: E6-04 numerical control audit -- step-size convergence of the
closed-loop 6-DOF ascent (task #276).

The F-04 TVC loop is a sampled-data controller inside a fixed-step RK4 plant,
and E6-04 (task #276) pinned down *where* the two grids interact: the control
law is sampled once per integration step at the step boundary and the command
is held across the four RK4 stages (ZOH), so the only numerical errors left
are the textbook ones -- ODE truncation from ``dt_s`` and sample-and-hold
discretisation from ``dt_c_s``. This benchmark measures both and records what
the audit found about their *interaction* (ADR-0008):

* **plant truncation (open loop)** -- a constant command on the vacuum plant
  (smooth RHS; the aero tables are only C0), ``dt_s`` halved: textbook
  4th-order RK4 convergence (ratio 16 per halving);
* **closed-loop integration sweep** -- ``dt_c_s`` fixed, ``dt_s`` halved: the
  loop contraction absorbs the truncation transient, and the loop output is
  flat to the float floor at every step (the pre-audit plant instead showed a
  1st-order ``dt_s`` plateau, the k4-stage command leak);
* **control sweep** -- ``dt_s`` fixed fine, ``dt_c_s`` halved: ZOH
  discretisation, effective order between 1 and 2;
* **alignment** -- with ``dt_c_s`` on (or snapped to) the integration grid,
  one tick schedule for every consistent ``dt_s``;
* **default justification** -- the shipped (dt_s, dt_c_s) = (0.02, 0.02)
  versus a refined (0.0025, 0.0025) run;
* **tick bookkeeping** -- an unaligned ``dt_c_s`` snaps every tick to an
  integration boundary (never mid-stage), every period in
  [dt_c_s, dt_c_s + dt_s).

The loop scenario is a SINGLE ramp, pitch 90->60 deg over the full 12 s: an
interior slope kink in a waypoint program is an O(1) discontinuity of the
feedforward command at the kink tick -- boundary accumulated-time jitter of
one ulp samples the slope on either side of it -- and that plateau, not any
step-size error, dominates a two-segment program's convergence curve (the
audit documents it under "program kinks" in ADR-0008). The ramp ends at 60 deg
keep every run clear of the actuator travel/slew limits, so no hard limit
engages and the curves are pure numerics.

``tests/test_flightlab_e6_04_step_convergence.py`` checks the committed JSON
against regression thresholds; ``docs/adr/flightlab-0008-...md`` records the
audit and the default-step choice.

    PYTHONPATH=src python benchmarks/bench_e6_04_step_convergence.py \
        -o docs/benchmarks/e604_step_convergence.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from orbitopt.flightlab.sixdof.aero import load_packaged_aero  # noqa: E402
from orbitopt.flightlab.sixdof.rigidbody import (  # noqa: E402
    ControlCommand, RigidBodyEngine, RigidBodyState, RigidBodyVehicle,
    dcm_from_quaternion, propagate,
)
from orbitopt.flightlab.sixdof.tvc import AttitudeTvcLaw, WaypointAttitudeProgram  # noqa: E402

T_END = 12.0
DT_GRID = (0.04, 0.02, 0.01, 0.005)
DT_REF = 0.0025


def _vehicle() -> RigidBodyVehicle:
    engine = RigidBodyEngine(thrust_vacuum_n=1.88e6, isp_vacuum_s=320.0,
                             isp_sea_level_s=282.0, nozzle_exit_area_m2=2.0,
                             throttle_min=0.4, throttle_max=1.0, mount_x_cg_m=22.0)
    return RigidBodyVehicle(name="spike_stage", dry_mass_kg=10_000.0,
                            propellant_mass_kg=60_000.0,
                            inertia_kg_m2=(3.4e4, 1.7e6, 1.7e6), engine=engine,
                            aero=load_packaged_aero("spike_stage"))


def _pad_state(**kw) -> RigidBodyState:
    kw.setdefault("lat_geod_deg", 28.5)
    kw.setdefault("lon_deg", -80.0)
    kw.setdefault("mass_kg", 70_000.0)
    return RigidBodyState.from_launch(**kw)


def _ramp() -> WaypointAttitudeProgram:
    """One linear pitch-over, 90 -> 60 deg over the whole window (no interior
    kink, no limit engagement -- see the module docstring)."""
    return WaypointAttitudeProgram(times_s=(0.0, T_END),
                                   heading_deg=(90.0, 90.0),
                                   pitch_deg=(90.0, 60.0),
                                   roll_deg=(0.0, 0.0))


def _metrics(r, law) -> dict:
    s = r.samples
    up = np.array([s["x_m"][-1], s["y_m"][-1], s["z_m"][-1]])
    up = up / np.linalg.norm(up)
    q = np.array([s["q_w"][-1], s["q_x"][-1], s["q_y"][-1], s["q_z"][-1]])
    nose = dcm_from_quaternion(q).T @ np.array([1.0, 0.0, 0.0])
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(nose @ up)))))
    hist = law.history()
    return {"alt_m": float(s["altitude_m"][-1]), "tilt_deg": tilt,
            "max_abs_err_pitch_rad": float(np.max(np.abs(hist["err_pitch_rad"]))),
            "n_ticks": int(np.asarray(hist["t_s"]).size) - 1,
            "slew_limit_ticks": float(np.sum(np.asarray(hist["pitch_slew_limited"]))),
            "history": hist}


def _closed(dt_s: float, dt_c: float, prog: WaypointAttitudeProgram) -> dict:
    veh = _vehicle()
    law = AttitudeTvcLaw(vehicle=veh, program=prog, dt_c_s=dt_c)
    return _metrics(propagate(_pad_state(pitch_deg=89.5), veh, law, T_END,
                              dt_s=dt_s), law)


def _constant(dt_s: float) -> dict:
    """The same plant and ODE with a fixed command on the VACUUM plant: pure
    truncation error. (With aero the piecewise-table air model is only C0 and
    its crossings scatter the RK4 order -- a property of the air table, not
    of the integrator, so the integrator's curve is taken where the RHS is
    smooth.)"""
    veh = _vehicle()
    from dataclasses import replace
    veh = replace(veh, aero=None)
    seen: list = []

    def law(t_s, state):
        seen.append(t_s)
        return ControlCommand(throttle=1.0, gimbal_pitch_rad=0.05, gimbal_yaw_rad=0.0)
    r = propagate(_pad_state(pitch_deg=89.5), veh, law, T_END, dt_s=dt_s)
    return {"alt_m": float(r.samples["altitude_m"][-1]), "tilt_deg": float("nan"),
            "max_abs_err_pitch_rad": float("nan"), "n_ticks": -1,
            "slew_limit_ticks": 0.0, "history": {}}


def _diff(run: dict, ref: dict) -> dict:
    return {"d_alt_m": run["alt_m"] - ref["alt_m"],
            "d_tilt_deg": run["tilt_deg"] - ref["tilt_deg"],
            "d_max_abs_err_pitch_rad": run["max_abs_err_pitch_rad"]
                                       - ref["max_abs_err_pitch_rad"]}


def _sweep(runs, fixed_label, fixed_val, free_label, ref_free) -> dict:
    ref = runs[ref_free]
    rows, prev_mag = [], None
    for free in DT_GRID:
        d = _diff(runs[free], ref)
        mag = abs(d["d_alt_m"])
        rows.append({free_label: free, fixed_label: fixed_val, **d,
                     "magnitude": mag,
                     "order": (float(np.log2(prev_mag / mag))
                               if prev_mag and mag > 1e-300 and prev_mag > 1e-300
                               else None)})
        prev_mag = mag
    return {"reference": {free_label: ref_free}, "rows": rows,
            "slew_limit_ticks_max": max(r["slew_limit_ticks"] for r in runs.values())}


def _open_loop_sweep(fixed_label, fixed_val) -> dict:
    runs = {dt: _constant(dt) for dt in DT_GRID + (DT_REF,)}
    return _sweep(runs, fixed_label, fixed_val, "dt_s", DT_REF)


def _json_safe(obj):
    """Strict JSON: NaN/Inf (open-loop N/A metrics) become null."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def main(out_path: Path | None) -> None:
    t0 = time.time()
    prog = _ramp()
    vert = WaypointAttitudeProgram.vertical()

    plant = {"scenario": "open_loop_constant_command",
             **_open_loop_sweep("dt_c_s", float("nan"))}

    closed_runs = {dt: _closed(dt, 0.04, prog) for dt in DT_GRID + (DT_REF,)}
    closed = {"scenario": "single_ramp_closed_loop",
              **_sweep(closed_runs, "dt_c_s", 0.04, "dt_s", DT_REF)}
    closed_vert = {dt: _closed(dt, 0.04, vert) for dt in DT_GRID + (DT_REF,)}
    closed["vertical_max_magnitude"] = max(
        abs(_diff(closed_vert[dt], closed_vert[DT_REF])["d_alt_m"]) for dt in DT_GRID)

    control_runs = {dtc: _closed(0.005, dtc, prog)
                    for dtc in (0.16, 0.08, 0.04, 0.02, 0.01, DT_REF)}
    ref = control_runs[DT_REF]
    control = {"scenario": "single_ramp_closed_loop", "fixed": {"dt_s": 0.005},
               "reference": {"dt_c_s": DT_REF}, "rows": [],
               "slew_limit_ticks_max": max(r["slew_limit_ticks"]
                                           for r in control_runs.values())}
    prev = None
    for dtc in (0.16, 0.08, 0.04, 0.02, 0.01):
        d = _diff(control_runs[dtc], ref)
        mag = abs(d["d_alt_m"])
        control["rows"].append({"dt_c_s": dtc, "dt_s": 0.005, **d, "magnitude": mag,
                                "order": (float(np.log2(prev / mag))
                                          if prev and mag > 1e-300 else None)})
        prev = mag

    alignment = {"dt_c_s": 0.05, "rows": []}
    for dt_s in (0.05, 0.025, 0.02):
        r = _closed(dt_s, 0.05, prog)
        alignment["rows"].append({"dt_s": dt_s, "alt_m": r["alt_m"],
                                  "tilt_deg": r["tilt_deg"],
                                  "max_abs_err_pitch_rad": r["max_abs_err_pitch_rad"],
                                  "n_ticks": r["n_ticks"]})

    fine, dflt = _closed(DT_REF, DT_REF, prog), _closed(0.02, 0.02, prog)
    default_vs_fine = _diff(dflt, fine)

    r = _closed(0.01, 0.03, prog)
    tt = np.asarray(r["history"]["t_s"])[1:]
    dd = np.asarray(r["history"]["dt_s"])[1:]
    ticks = {"dt_s": 0.01, "dt_c_s": 0.03, "n_ticks": int(tt.size),
             "tick_time_off_boundary_max_s":
                 float(np.max(np.abs(tt / 0.01 - np.round(tt / 0.01))) * 0.01),
             "tick_period_min_s": float(dd.min()), "tick_period_max_s": float(dd.max()),
             "tick_period_in_band": bool(np.all((dd >= 0.03 - 1e-9) & (dd < 0.04 - 1e-9)))}

    payload = {
        "meta": {
            "task": "E6-04 numerical control audit (#276)",
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "t_end_s": T_END, "runtime_s": round(time.time() - t0, 1),
            "defaults": {"dt_s": 0.02, "dt_c_s": 0.02},
            "scenario": "single pitch-over ramp 90->60 deg over 12 s; the law "
                        "is sampled once per integration step at the step "
                        "boundary and held across the four RK4 stages (E6-04)",
        },
        "integration": {"plant_truncation": plant, "closed_loop": closed},
        "control": control,
        "alignment": alignment,
        "default_vs_fine": default_vs_fine,
        "tick_bookkeeping": ticks,
    }
    text = json.dumps(_json_safe(payload), indent=1)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n")
        print(f"wrote {out_path}")
    print("[plant truncation]  " + " | ".join(
        f"dt_s={r['dt_s']}: {r['magnitude']:.2e} ord~{r['order']:.2f}"
        for r in plant["rows"] if r["order"]))
    print("[closed loop, dt_c=0.04]  " + " | ".join(
        f"dt_s={r['dt_s']}: {r['magnitude']:.2e}" for r in closed["rows"]))
    print("[control, dt_s=0.005]  " + " | ".join(
        f"dt_c={r['dt_c_s']}: {r['magnitude']:.2e} ord~{r['order']:.2f}"
        for r in control["rows"] if r["order"]))
    print(f"[default vs fine] {default_vs_fine}")
    print(f"[ticks] {ticks}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args()
    main(args.out)
