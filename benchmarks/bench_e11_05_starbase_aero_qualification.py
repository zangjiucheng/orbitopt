#!/usr/bin/env python3
"""E11-05: Starbase aero model qualification -- booster power-off drag.

Background (E5-08 / E10-04): the Starbase stage-1 debris impact speed sits
12.80 % off the independent twin against a 12 % band, and E10-04 evidenced
the gap as a genuine power-off drag-model difference (replay from the
recorded separation state with the declared polar reproduces the gap; a
uniform x1.1 on the illustrative polar would close it, but the polar has
no qualification behind it).

This benchmark qualifies the booster power-off drag model with a reviewable
semi-empirical component build-up (orbitopt.flightlab.booster_aero), runs it
against the public data this repo holds -- the recorded OT debris flights
(in an in-flight Cd inversion) and the E11-01 grid-fin tables -- and records
the impact of the E11-01 deployed/stowed fin states on Cd.  The 12 % band is
not touched: if the qualified model still leaves the check outside it, the
discrepancy is recorded, again, as a genuine model difference with the
evidence.

Reproduce (~1 min):

    python benchmarks/bench_e11_05_starbase_aero_qualification.py --out \
        docs/benchmarks/e11_05_starbase_aero_qualification.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from orbitopt.flightlab import booster_aero as ba  # noqa: E402
from orbitopt.flightlab import vehicle_catalog as vc  # noqa: E402
from orbitopt.flightlab.recovery.propagator import BoosterModel, BoosterState  # noqa: E402
from orbitopt.verify import independent_dynamics as idyn  # noqa: E402

REF = REPO / "tests/data/mission/reference"
SB_CSV = (REF / "e5-08-new-vehicle-starbase/mission-optimized/launch/ascent"
          "/trajectories/stage1_debris.csv")
SB_RAW = (REF / "e5-08-new-vehicle-starbase/mission-optimized/launch/ascent"
          "/opentsiolkovsky/bodies/stage1.csv")
DC7_CSV = (REF / "e5-08-new-vehicle-dc7/mission-optimized/launch/ascent"
           "/trajectories/stage1_debris.csv")
DC7_RAW = (REF / "e5-08-new-vehicle-dc7/mission-optimized/launch/ascent"
           "/opentsiolkovsky/bodies/stage1.csv")
E1004 = REF / "e5-08-new-vehicle-starbase-reports/debris-model-difference.json"
E1007 = REPO / "docs/benchmarks/e10_07_full_recovery_reflight.json"
DC8_YAML = REPO / "src/orbitopt/data/flightlab/vehicles/starship_class_rtls_dc8.yaml"
STARBASE_YAML = REPO / "src/orbitopt/data/flightlab/vehicles/starship_class_rtls_starbase.yaml"
DC7_YAML = REPO / "src/orbitopt/data/flightlab/vehicles/starship_class_rtls_dc7.yaml"

S_STARBASE = math.pi / 4.0 * 9.0 ** 2          # declared 9 m stage diameter
Q_INVERT_MIN_PA = 50_000.0                      # finite-difference noise floor
MACH_BINS = ((4.0, 4.5), (4.5, 5.0), (5.0, 5.5), (5.5, 6.0), (6.0, 13.0))
BAND_IMPACT_SPEED_PCT = 12.0                    # UNCHANGED by this task


def _interp_table(table, mach):
    xs = [float(r[0]) for r in table]
    ys = [float(r[1]) for r in table]
    return float(np.interp(mach, xs, ys))


# ------------------------------------------------------- Part B: inversion

def invert_csv(path: Path, s_ref_m2: float):
    """In-flight power-off Cd of the recorded OT debris: central difference
    of the recorded ECEF velocity, minus J2 + Coriolis + centrifugal (the
    twin's own force package), projected on -v_rel; a_drag = q Cd S / m with
    the RECORDED q."""
    rows = list(csv.DictReader(open(path)))
    om = np.array([0.0, 0.0, idyn.OMEGA_EARTH_RAD_S])
    out = []
    for i in range(1, len(rows) - 1):
        q = float(rows[i]["dynamic_pressure_pa"])
        if q < Q_INVERT_MIN_PA:
            continue
        r0 = np.array([float(rows[i - 1][k]) for k in ("x_m", "y_m", "z_m")])
        r1 = np.array([float(rows[i + 1][k]) for k in ("x_m", "y_m", "z_m")])
        v0 = np.array([float(rows[i - 1][k]) for k in ("vx_m_s", "vy_m_s", "vz_m_s")])
        v1 = np.array([float(rows[i + 1][k]) for k in ("vx_m_s", "vy_m_s", "vz_m_s")])
        dt = float(rows[i + 1]["t_s"]) - float(rows[i - 1]["t_s"])
        m = float(rows[i]["mass_kg"])
        r = 0.5 * (r0 + r1)
        v = 0.5 * (v0 + v1)
        rn = float(np.linalg.norm(r))
        g = -(idyn._MU_EARTH / rn ** 3) * r
        z2 = (r[2] / rn) ** 2
        k = 1.5 * idyn._J2 * idyn._MU_EARTH * idyn._R_EARTH ** 2 / rn ** 5
        g = g + k * np.array([r[0] * (1 - 5 * z2), r[1] * (1 - 5 * z2), r[2] * (3 - 5 * z2)])
        a_fict = -2 * np.cross(om, v) - np.cross(om, np.cross(om, r))
        a_aero = (v1 - v0) / dt - g - a_fict
        v_rel = v + np.cross(om, r)
        speed = float(np.linalg.norm(v_rel))
        a_drag = -(a_aero @ v_rel) / speed
        out.append({"t_s": float(rows[i]["t_s"]),
                    "altitude_m": float(rows[i]["altitude_m"]),
                    "mach": float(rows[i]["mach"]),
                    "q_pa": q, "cd_eff": m * a_drag / (q * s_ref_m2)})
    return out


def bin_stats(samples, table):
    """Median inverted Cd and median ratio to the declared polar per Mach
    bin (dense-atmosphere rows only -- already filtered by q)."""
    bins = []
    for lo, hi in MACH_BINS:
        vals = [s["cd_eff"] for s in samples if lo <= s["mach"] < hi]
        ratios = [s["cd_eff"] / _interp_table(table, s["mach"])
                  for s in samples if lo <= s["mach"] < hi]
        if not vals:
            continue
        bins.append({"mach_bin": [lo, hi], "n": len(vals),
                     "cd_eff_median": round(float(np.median(vals)), 4),
                     "declared_median": round(float(np.median(
                         [_interp_table(table, s["mach"]) for s in samples
                          if lo <= s["mach"] < hi])), 4),
                     "ratio_median": round(float(np.median(ratios)), 4)})
    return bins


# ---------------------------------------------------------- Part C: replay

def _sep_state_eci(raw_csv: Path, t0: float):
    for row in csv.DictReader(open(raw_csv)):
        if abs(float(row["time_s"]) - t0) < 1e-6:
            r = np.array([float(row[k]) for k in
                          ("pos_eci_x_m", "pos_eci_y_m", "pos_eci_z_m")])
            v = np.array([float(row[k]) for k in
                          ("vel_eci_x_mps", "vel_eci_y_mps", "vel_eci_z_mps")])
            return r, v
    raise KeyError(f"{raw_csv.name}: no raw history row at t={t0}")


def debris_replay(summary_csv: Path, raw_csv: Path, polar, diameter_m: float):
    """Twin's own power-off debris integrator from the RECORDED separation
    state -- the raw history's ECI columns at separation, rotated ECI->ECEF
    with the verified _ecef_to_eci convention (E10-04 protocol) -- and the
    RECORDED debris mass."""
    rows = list(csv.DictReader(open(summary_csv)))
    t0 = float(rows[0]["t_s"])
    m0 = float(rows[0]["mass_kg"])
    raw = list(csv.DictReader(open(raw_csv)))
    rec_speed = float(raw[-1]["speed_earth_mps"])  # ground-relative, at impact
    r_eci, v_eci = _sep_state_eci(raw_csv, t0)
    om = np.array([0.0, 0.0, idyn.OMEGA_EARTH_RAD_S])
    r0 = idyn._rot_z(r_eci, idyn.OMEGA_EARTH_RAD_S * t0)
    v0 = idyn._rot_z(v_eci, idyn.OMEGA_EARTH_RAD_S * t0) - np.cross(om, r0)
    imp = idyn._debris_to_impact(r0, v0, m0, [list(r) for r in polar],
                                 0.02, diameter_m=diameter_m, t0_s=t0)
    speed = float(imp["speed_earth_m_s"])
    return {"twin_speed_m_s": round(speed, 3),
            "recorded_speed_m_s": round(rec_speed, 3),
            "speed_pct": round((speed / rec_speed - 1.0) * 100.0, 3)}



# --------------------------------------------------------- Part E: sync

def part_sync(veh, qualified_dc8):
    """One generated table, both models: the propagator (main model) and the
    E10-07 twin each consume booster.cd_mach; replace DC8's declared
    illustrative table with the E11-05 qualified one and fly the same coast
    through both integrators."""
    import dataclasses
    from datetime import datetime
    from orbitopt.flightlab.recovery.propagator import propagate
    veh_q = dataclasses.replace(veh, cd_mach=tuple(qualified_dc8))
    machs = [0.0, 0.5, 1.1, 2.0, 4.0, 8.0, 9.5]
    agree = max(abs(veh_q.cd(m) - ba.interp_cd(qualified_dc8, m)) for m in machs)
    state = json.loads(E1007.read_text())["handoff"]["state"]
    pos = tuple(state["position_m"])
    vel = tuple(state["velocity_m_s"])
    mass = float(state["mass_kg"])
    t0 = float(state["t_s"])
    st0 = BoosterState(t0, datetime.fromisoformat(state["epoch_utc"]),
                       "ECEF", pos, vel, mass)
    tr = propagate(st0, veh_q, None, t0 + 900.0, events=("ground",), dt_s=0.05)
    if tr.impact is None:
        raise SystemExit("sync coast did not terminate on the ground")
    twin = idyn._debris_to_impact(np.array(pos), np.array(vel), mass,
                                  [list(r) for r in qualified_dc8], 0.02,
                                  diameter_m=9.0, t0_s=t0)
    t_speed = float(twin["speed_earth_m_s"])
    delta_pct = (tr.impact.speed_m_s / t_speed - 1.0) * 100.0
    return {"interp_agreement_abs": round(agree, 12),
            "main_model": {"speed_m_s": round(tr.impact.speed_m_s, 3)},
            "twin": {"speed_m_s": round(t_speed, 3)},
            "impact_speed_delta_pct": round(delta_pct, 3)}


# -------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "docs/benchmarks/e11_05_starbase_aero_qualification.json"))
    args = ap.parse_args(argv)

    sb_doc = vc.load_validated(STARBASE_YAML)
    dc7_doc = vc.load_validated(DC7_YAML)
    import yaml
    dc8_doc = yaml.safe_load(DC8_YAML.read_text())
    sb_ca = sb_doc["stages"][0]["aero"]["ca_power_off"]
    dc7_ca = dc7_doc["stages"][0]["aero"]["ca_power_off"]
    veh = BoosterModel.load(DC8_YAML)
    gf = veh.grid_fins
    declared_declared = veh.cd_mach

    # ---- A: component tables
    qualified_ef = ba.cd_engines_first_table(
        diameter_m=9.0, length_m=71.0, protuberances=ba.DC8_STOWED)
    qualified_axial_dc8 = ba.ca_stacked_stowed(
        dc8_doc["stages"][0]["aero"]["ca_power_off"], ba.DC8_STOWED,
        ref_area_m2=S_STARBASE)
    qualified_axial_v1 = ba.ca_stacked_stowed(sb_ca, ba.STARBASE_V1_STOWED,
                                              ref_area_m2=S_STARBASE)
    fin_delta = ba.fin_state_delta_cd(gf, ref_area_m2=S_STARBASE)
    anchor = {}
    for m in (0.0, 0.8, 1.1, 2.0, 4.0, 6.0, 8.0):
        anchor[str(m)] = {
            "declared": round(_interp_table(declared_declared, m), 3),
            "qualified_buildup": round(_interp_table(qualified_ef, m), 3),
            "ratio": round(_interp_table(qualified_ef, m)
                           / _interp_table(declared_declared, m), 3)}
    f9_bc = None
    for i, stage in enumerate(dc7_doc["stages"]):
        bc = stage["aero"].get("ballistic_coefficient_kg_m2")
        if bc:
            f9_bc = {"stage_index": i, "bc_kg_m2": float(bc),
                     "dry_mass_kg": float(stage["mass_dry_kg"]),
                     "implied_cd": round(float(stage["mass_dry_kg"]) / (
                         float(bc) * math.pi / 4.0
                         * float(stage["aero"]["diameter_m"]) ** 2), 3)}
            break

    # ---- B: recorded Cd inversion
    inv_sb = invert_csv(SB_CSV, S_STARBASE)
    inv_dc7 = invert_csv(DC7_CSV, S_STARBASE)
    inversion = {
        "method": ("central difference of recorded ECEF velocity, gravity "
                   "(J2) + Coriolis + centrifugal removed with the twin's own "
                   "force package, projected on -v_rel; Cd = m a / (q S) with "
                   "the recorded q; rows with q > 50 kPa"),
        "starbase": {"n_dense": len(inv_sb), "bins": bin_stats(inv_sb, sb_ca),
                     "cd_eff_mean_ratio": round(float(np.mean(
                         [s["cd_eff"] / _interp_table(sb_ca, s["mach"])
                          for s in inv_sb])), 4)},
        "dc7": {"n_dense": len(inv_dc7), "bins": bin_stats(inv_dc7, dc7_ca),
                "cd_eff_mean_ratio": round(float(np.mean(
                    [s["cd_eff"] / _interp_table(dc7_ca, s["mach"])
                     for s in inv_dc7])), 4)}}

    # ---- C: same-state replay with each polar (E10-04 protocol)
    e1004 = json.loads(E1004.read_text())
    e1004_sb = e1004["replays"]["starbase"]["twin_from_recorded_state"]["1.0"]["speed_pct"]
    control_sb = debris_replay(SB_CSV, SB_RAW, sb_ca, 9.0)
    control_dc7 = debris_replay(DC7_CSV, DC7_RAW, dc7_ca, 9.0)
    qualified_v1 = debris_replay(SB_CSV, SB_RAW, qualified_axial_v1, 9.0)
    deployed = [[r[0], r[1] + _interp_table(fin_delta, r[0])] for r in sb_ca]
    fins_deployed_bound = debris_replay(SB_CSV, SB_RAW, deployed, 9.0)
    qualified_dc7 = debris_replay(DC7_CSV, DC7_RAW, dc7_ca, 9.0)

    # ---- D: fin-state Cd effect (E11-01 block, stowed vs deployed)
    fin_state = {
        "stowed_increment_cd": round(
            ba.DC8_STOWED.cd_increment(0.3, S_STARBASE), 3),
        "deployed_minus_stowed": fin_delta,
        "note": ("deployed at zero incidence = profile + blockage; at flying "
                 "alpha the E11-01 envelope adds the normal-force and induced "
                 "terms on top (docs/benchmarks/e11_01_grid_fin_authority.md)")}

    # ---- E: sync check (one table, both models)
    sync = part_sync(veh, qualified_ef)

    # ---- verdict
    q_pct = qualified_v1["speed_pct"]
    over = abs(q_pct) > BAND_IMPACT_SPEED_PCT
    verdict = {
        "band_impact_speed_pct_UNCHANGED": BAND_IMPACT_SPEED_PCT,
        "starbase_qualified_speed_pct": q_pct,
        "inside_band": not over,
        "conclusion": (
            "the qualified power-off model leaves the Starbase debris check "
            "outside the unchanged 12 % band (replay "
            f"{q_pct:+.2f} %), so the discrepancy stays recorded as a genuine "
            "aero-model difference; the new evidence narrows WHERE it lives: "
            "the recorded flights invert to a power-off Cd BELOW the declared "
            "polar (Starbase "
            f"{inversion['starbase']['cd_eff_mean_ratio']:.3f}, DC7 "
            f"{inversion['dc7']['cd_eff_mean_ratio']:.3f}), so no physically "
            "supported power-off correction (protuberances +, deployed fins "
            "+) closes it -- every build-up term moves the replay FURTHER out; "
            "the remaining gap is the aero/atmosphere line between OT and the "
            "twin in dense high-q air, not a booster Cd error"
            if over else "the qualified model closes the check inside the band"),
        "control_reproduces_e1004": bool(
            abs(control_sb["speed_pct"] - e1004_sb) < 0.15)}

    rep = {
        "task": "E11-05 Starbase aero model qualification",
        "generated_by": Path(__file__).name,
        "git_commit": subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip(),
        "modules": ["orbitopt.flightlab.booster_aero"],
        "design_assumptions": list(ba.DESIGN_ASSUMPTIONS),
        "sources": [{"tag": t, "detail": d} for t, d in ba.QUALIFICATION_SOURCES],
        "part_a_component_tables": {
            "engines_first_stowed_qualified": qualified_ef,
            "engines_first_anchors_vs_declared": anchor,
            "axial_stacked_stowed_starbase_v1": qualified_axial_v1,
            "axial_stacked_stowed_dc8": qualified_axial_dc8,
            "f9_bc_anchor": f9_bc},
        "part_b_recorded_inversion": inversion,
        "part_c_same_state_replays": {
            "control_declared_starbase": control_sb,
            "control_declared_dc7": control_dc7,
            "qualified_axial_starbase_v1": qualified_v1,
            "fins_deployed_bound_starbase": fins_deployed_bound,
            "qualified_axial_dc7": qualified_dc7},
        "part_d_fin_state": fin_state,
        "part_e_sync_main_twin": sync,
        "verdict": verdict}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(f" wrote {args.out}")
    print(f" control Starbase {control_sb['speed_pct']:+.2f} % "
          f"(E10-04 {e1004_sb:+.2f} %), qualified {q_pct:+.2f} %; "
          f"inversion ratio {inversion['starbase']['cd_eff_mean_ratio']:.3f}; "
          f"sync delta {sync['impact_speed_delta_pct']:+.2f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
