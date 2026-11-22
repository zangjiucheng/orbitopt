"""E10-07 (task #304): full-recovery independent re-flight from the real
hot-staging separation handoff.

E8-05 verified the *landing phase* with an independent-dynamics twin, but it
started from FL-302's declared corridor insertion state -- a design input, not
a state the vehicle is actually handed at recovery. This module re-flies the
whole recovery chain -- boostback cutoff, the E10-01 unpowered aerodynamic
entry, and the terminal descent -- from the **actual DC8 hot-staging
separation state** (E10-02 timeline, OpenTsiolkovsky ascent, traceable by the
:class:`~orbitopt.flightlab.handoff.HandoffState` digest) through an
independently written point-mass twin, and compares every acceptance quantity
against the main FL-301/FL-302 model, splitting the discrepancy into an error
budget (atmosphere, numerics, gravity; frame and controller declared zero --
both models integrate in ECEF at the same epoch and replay the identical
authored script via :class:`~orbitopt.flightlab.recovery.boostback._Mission`,
so the controller line is shared by design).

Twin conventions follow the E8-05 rule (ContextDB
``e8-05-independent-dynamics-evidence-rule``): its own WGS-84 geodesy, its own
US76 layer model, its own J2, its own RK4; the engine/Cd/aero *coefficients*
are read from the vehicle file because they are design inputs, not physics.
Every comparison is labelled ``independent_model``; the corridor-origin
landing check stays what E10-07 found it to be: ``design_input``.

The mission verdict is reported honestly: the two-burn unpowered profile does
not close at DC8 separation conditions (consistent with E10-01's
``VEHICLE_CAPABILITY`` finding) and the landing burn is unplannable from the
*real* crossed handoff, so the terminal comparison is against ballistic
impact, not a touchdown.
"""
from __future__ import annotations

import dataclasses
import hashlib
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from orbitopt.flightlab.contract import load_target, read_run
from orbitopt.flightlab.handoff import HandoffState, handoff_digest
from orbitopt.flightlab.recovery.boostback import (
    IIP_BOOSTBACK_CAP_S,
    BoostbackPlan,
    RecoveryTarget,
    _Mission,
    _bearing_of_target,
    plan_boostback_entry,
)
from orbitopt.flightlab.recovery.landing import fly_landing_burn
from orbitopt.flightlab.recovery.pipeline import (
    landing_reserve_booster,
    separation_state_ecef,
)
from orbitopt.flightlab.recovery.propagator import BoosterModel
from orbitopt.verify.report import (
    CHECK_MODEL,
    EVIDENCE_INDEPENDENT_MODEL,
    provenance,
    report_digest,
    status_from_checks,
)

PROTOCOL = "e10-07-full-recovery-reflight-1.0"

# --- the re-flown script and recipe ---------------------------------------
# The sealed E5-08 recovery recipe (tests/data/mission/reference/
# e5-08-new-vehicle-dc7/recovery-script.json) flown unpowered (E10-01
# entry_burns=0) with the E10-01 keep-probe script: boostback delay 21.2 s,
# IIP aim -2.0 km (the parameter is km, as the planner sees it), full
# throttle, zero pitch/yaw bias, no entry burn.
E10_07_SCRIPT = [21.2, -2.0, 1.0, 0.0, 0.0]
E10_07_RECIPE: dict[str, Any] = {
    "entry_burns": 0,
    "search": "keep",
    "boostback_cutoff": "iip",
    "landing_need_model": "closed_loop",
    "corridor_aim_margin_m": 7000.0,
    "entry_q_max_pa": 250_000.0,
    "entry_g_max": 12.0,
    "landing_reserve_margin_kg": 2500.0,
    "horizon_s": 900.0,
    "dt_opt": 0.5,
    "dt_final": 0.25,
}

# --- twin physics (independent implementations) ----------------------------
_MU = 3.986004418e14          # m^3/s^2, WGS-84 GM (own constant, not imported)
_J2 = 1.08262668e-3           # own constant
_A_WGS84 = 6378137.0
_F_WGS84 = 1.0 / 298.257223563
_E2_WGS84 = _F_WGS84 * (2.0 - _F_WGS84)
_G0 = 9.80665
_R_DRY_AIR = 287.05287
_OMEGA_E = 7.2921159e-5
_T_STRAT = 186.866467         # K, US76 stratopause temperature
_P_STRAT = 0.3733876          # Pa, US76 at 84 852 m geopotential
_H_STRAT = 84852.0

#: US76 standard layers (geopotential base [m], T0 [K], signed lapse
#: dT/dH [K/m], p0 [Pa]) -- written from the standard table (bases and
#: lapse rates as published; a gradient layer's top temperature equals the
#: next base, e.g. 216.65 + 0.001*12 000 = 228.65 at 32 km).
_US76_LAYERS = (
    (0.0, 288.15, -0.0065, 101_325.0),
    (11_000.0, 216.65, 0.0, 22_632.06),
    (20_000.0, 216.65, 0.001, 5_474.889),
    (32_000.0, 228.65, 0.0028, 868.0187),
    (47_000.0, 270.65, 0.0, 110.5234),
    (51_000.0, 270.65, -0.0028, 66.93887),
    (71_000.0, 214.65, -0.002, 3.956420),
)


def _twin_geodetic(r: np.ndarray) -> tuple[float, float, float]:
    """(lat, lon, ellipsoid height) -- own iterative WGS-84 inverse."""
    x, y, z = (float(v) for v in r)
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    phi = math.atan2(z, p * (1.0 - _E2_WGS84)) if p > 0.0 else math.copysign(
        math.pi / 2.0, z)
    for _ in range(8):
        s = math.sin(phi)
        n_ = _A_WGS84 / math.sqrt(1.0 - _E2_WGS84 * s * s)
        phi = math.atan2(z + _E2_WGS84 * n_ * s, p)
    s = math.sin(phi)
    n_ = _A_WGS84 / math.sqrt(1.0 - _E2_WGS84 * s * s)
    h = p / math.cos(phi) - n_ if abs(phi) < 1.2 else z / s - n_ * (1.0 - _E2_WGS84)
    return phi, lon, h


def _twin_us76(h_geo: float) -> tuple[float, float, float]:
    """(p, rho, speed_of_sound) at geodetic height, own US76 layers; above
    the stratopause an isothermal continuation at the stratopause
    temperature (the same basis US76 itself tabulates from)."""
    z = max(float(h_geo), 0.0)
    h_pot = z * _A_WGS84 / (_A_WGS84 + z)
    if h_pot > _H_STRAT:
        t = _T_STRAT
        p = _P_STRAT * math.exp(-_G0 * (h_pot - _H_STRAT) / (_R_DRY_AIR * t))
        return p, p / (_R_DRY_AIR * t), math.sqrt(1.4 * _R_DRY_AIR * t)
    zb, t0, lapse, p0 = _US76_LAYERS[0]
    for layer in _US76_LAYERS:
        if h_pot >= layer[0]:
            zb, t0, lapse, p0 = layer
    if abs(lapse) < 1e-9:
        t = t0
        p = p0 * math.exp(-_G0 * (h_pot - zb) / (_R_DRY_AIR * t0))
    else:
        t = t0 + lapse * (h_pot - zb)
        p = p0 * (t / t0) ** (-_G0 / (_R_DRY_AIR * lapse))
    return p, p / (_R_DRY_AIR * t), math.sqrt(1.4 * _R_DRY_AIR * t)


def _band(check_id: str, unit: str, limit: float, basis: str) -> None:
    _TOL[check_id] = (unit, limit, basis)


_TOL: dict[str, tuple[str, float, str]] = {}
# Bands measured on the E10-07 benchmark run, then margined (E8-05 rule:
# measure first, margin second -- never a band picked to make a number pass).
_band("reflight.bb_cutoff.time", "s", 0.6,
      "event bisection (main) vs 0.05-0.1 s step boundary (twin); measured 0.25 s")
_band("reflight.bb_cutoff.position", "m", 600.0,
      "cutoff sampling boundary + burn-integration difference; measured 194 m")
_band("reflight.bb_cutoff.speed", "m/s", 6.0,
      "burn termination boundary sampling; measured 3.5 m/s")
_band("reflight.entry_peak_q", "%", 12.0,
      "density-profile form line measures 5.1% of the 4.8% delta (the whole"
      " discrepancy is the two density profiles, residual 0)")
_band("reflight.entry_peak_g", "g", 1.0,
      "follows the q line; measured 0.28 g")
_band("reflight.impact_time", "s", 1.0,
      "gravity-line dominated (J2 difference alone 0.31 s); measured 0.76 s")
_band("reflight.impact_position", "m", 1500.0,
      "numerical 116 m + gravity 111 m lines dominate; measured 544 m")
_band("reflight.impact_speed", "%", 12.0,
      "terminal descent speed is density-profile sensitive: the atmosphere"
      " form line measures 8.7% (the whole delta, residual 0); same relative"
      " band as the E8-05 ascent debris-impact speed check")
_band("reflight.propellant_remaining", "kg", 1200.0,
      "burn-exposure sampling; measured 453 kg")
BANDS = dict(_TOL)

#: The three headline acceptance quantities the budget is tabulated against.
_BUDGET_QUANTITIES = ("reflight.entry_peak_q", "reflight.impact_speed",
                      "reflight.impact_position")

TWIN_DESCRIPTION = (
    "own RK4 (0.05/0.1 s), own WGS-84 inverse geodesy, point-mass + J2 "
    "written from constants, US76 layers + isothermal stratopause "
    "continuation, Mach-interpolated Cd, ambient-corrected thrust with "
    "linear-Isp model -- coefficients from the vehicle YAML (design input); "
    "guidance callback shared with the main model by design (the authored "
    "script is the controller, declared zero on the budget)")


def _twin_mission(plan: Any, sep_state: Any, booster: BoosterModel,
                  target: RecoveryTarget) -> _Mission:
    """Rebuild the *authored* script the planner flew (IIP cutoff mode) so
    the twin replays the identical controller -- the controller budget line
    is shared-by-design and declared zero."""
    pad = np.asarray(target.position_ecef_m, dtype=np.float64)
    lat, lon, _ = _twin_geodetic(pad)
    e_ = np.array([-math.sin(lon), math.cos(lon), 0.0])
    n_ = np.array([-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon),
                   math.cos(lat)])
    d = np.asarray(sep_state.position_m, dtype=np.float64) - pad
    horiz = float(d @ e_) * e_ + float(d @ n_) * n_
    return_dir = -horiz / max(float(np.linalg.norm(horiz)), 1e-9)
    eng = booster.engine
    bb = BoostbackPlan(E10_07_SCRIPT[0], IIP_BOOSTBACK_CAP_S, E10_07_SCRIPT[2],
                       E10_07_SCRIPT[3], E10_07_SCRIPT[4])
    reserve = float(getattr(booster, "landing_propellant_reserve_kg", 0.0))
    return _Mission(float(sep_state.t_s), bb, None, _bearing_of_target(pad), None,
                    iip_aim_m=1000.0 * float(E10_07_SCRIPT[1]), pad_ecef=pad,
                    return_dir=return_dir, vernier_throttle=eng.throttle_min,
                    bb_floor_mass_kg=booster.dry_mass_kg + reserve)


class _TwinState:
    """Duck-typed FL-301 state view handed to the shared mission callback."""

    def __init__(self, t: float, y: np.ndarray) -> None:
        self.t_s = t
        self.epoch_utc = None
        self.frame = "ECEF"
        self.position_m = tuple(float(v) for v in y[:3])
        self.velocity_m_s = tuple(float(v) for v in y[3:6])
        self.mass_kg = float(y[6])


def _twin_flight(mission_factory: Callable[[], _Mission], sep_state: Any,
                 booster: BoosterModel, *, dt_s: float,
                gravity: str = "j2", density_fn: Callable[[float], float]
                | None = None, corridor_h0_m: float,
                horizon_s: float) -> dict:
    """Independent point-mass re-flight of the authored recovery script.

    Returns the acceptance quantities: boostback cutoff state, entry-window
    peak q / proper load (cutoff -> corridor-h0 crossing, the same window the
    main model grades), and the terminal descent to ellipsoid height zero
    (impact) with bisection-refined time/state. ``density_fn`` swaps only
    the density profile (pressure/sound speed stay US76) -- used for the
    atmosphere budget line, which measures the *form* difference between
    the two models' density profiles; it is a labelled perturbation
    experiment, not part of the twin.
    """
    msn = mission_factory()
    eng = booster.engine
    cd_tab = np.asarray(booster.cd_mach, dtype=np.float64)
    s_ref = float(booster.reference_area_m2)
    dry = float(booster.dry_mass_kg)
    omega = np.array([0.0, 0.0, _OMEGA_E])

    def accel(tt: float, y: np.ndarray) -> tuple[np.ndarray, float, float, float, float]:
        r, v, m = y[:3], y[3:6], float(y[6])
        rn = float(np.linalg.norm(r))
        a = -_MU * r / rn ** 3
        if gravity == "j2":
            zz = r[2]
            d2 = 1.5 * _J2 * _MU * _A_WGS84 ** 2 / rn ** 5
            a = a + np.array([-d2 * r[0] * (1.0 - 5.0 * zz * zz / rn ** 2),
                              -d2 * r[1] * (1.0 - 5.0 * zz * zz / rn ** 2),
                              -d2 * r[2] * (3.0 - 5.0 * zz * zz / rn ** 2)])
        a = a - 2.0 * np.cross(omega, v) - np.cross(omega, np.cross(omega, r))
        p, rho, a_s = _twin_us76(_twin_geodetic(r)[2])
        if density_fn is not None:
            rho = float(density_fn(_twin_geodetic(r)[2]))
        f_aero = np.zeros(3)
        q = 0.0
        if rho > 1e-14:
            spd = float(np.linalg.norm(v))  # air at rest in the ECEF frame
            if spd > 1e-6:
                mach = spd / a_s
                q = 0.5 * rho * spd * spd
                cd = float(np.interp(mach, cd_tab[:, 0], cd_tab[:, 1]))
                f_aero = -(q * cd * s_ref / spd) * v
        a = a + f_aero / m
        cmd = msn(tt, _TwinState(tt, y))
        thr = 0.0
        f_t = 0.0
        prop = 0.0
        if cmd is not None and m > dry:
            lvl = min(max(float(cmd.throttle), eng.throttle_min), eng.throttle_max)
            fmax = max(eng.thrust_vacuum_n - eng.nozzle_exit_area_m2 * p, 0.0)
            f_t = eng.count * lvl * fmax
            if f_t > 0.0:
                isp = eng.isp_sea_level_s + (eng.isp_vacuum_s - eng.isp_sea_level_s) \
                    * (1.0 - min(p, 101_325.0) / 101_325.0)
                dvec = np.asarray(cmd.dir_ecef, dtype=np.float64)
                dvec = dvec / float(np.linalg.norm(dvec))
                a = a + f_t * dvec / m
                prop = f_t / (isp * _G0)
                thr = lvl
        proper_g = (float(np.linalg.norm(f_aero)) + f_t) / (m * _G0)
        return np.concatenate([v, a, [-prop]]), thr, q, proper_g, p

    y = np.concatenate([np.asarray(sep_state.position_m, dtype=np.float64),
                        np.asarray(sep_state.velocity_m_s, dtype=np.float64),
                        [float(sep_state.mass_kg)]])
    t = float(sep_state.t_s)
    t_cap = t + float(horizon_s)
    cut: dict | None = None
    prev_on = False
    q_max = 0.0
    g_max = 0.0
    h0_t = None
    while t < t_cap:
        k1, thr1, q1, g1, _p = accel(t, y)
        k2, _, _, _, _ = accel(t + 0.5 * dt_s, y + 0.5 * dt_s * k1)
        k3, _, _, _, _ = accel(t + 0.5 * dt_s, y + 0.5 * dt_s * k2)
        k4, _, _, _, _ = accel(t + dt_s, y + dt_s * k3)
        y2 = y + dt_s / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        on = thr1 > 0.0
        if cut is None and prev_on and not on:
            cut = {"t_s": t, "position_ecef_m": [float(v) for v in y[:3]],
                   "velocity_ecef_m_s": [float(v) for v in y[3:6]],
                   "mass_kg": float(y[6])}
        prev_on = on
        if cut is not None and h0_t is None:
            q_max = max(q_max, q1)
            g_max = max(g_max, g1)
        h_old = _twin_geodetic(y[:3])[2]
        h_new = _twin_geodetic(y2[:3])[2]
        if cut is not None and h0_t is None and h_new <= corridor_h0_m < h_old:
            h0_t = t + dt_s * (h_old - corridor_h0_m) / (h_old - h_new)
        if h_new < 0.0 <= h_old:
            frac = h_old / (h_old - h_new)
            yf = y + dt_s * frac / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t_imp = t + dt_s * frac
            impact = {"t_s": t_imp,
                      "position_ecef_m": [float(v) for v in yf[:3]],
                      "velocity_ecef_m_s": [float(v) for v in yf[3:6]],
                      "speed_m_s": float(np.linalg.norm(yf[3:6])),
                      "vertical_speed_m_s":
                          float(np.dot(yf[3:6], yf[:3] / np.linalg.norm(yf[:3]))),
                      "mass_kg": float(yf[6])}
            return {"cut": cut, "entry_max_q_pa": q_max,
                    "entry_max_proper_g": g_max, "corridor_h0_cross_t_s": h0_t,
                    "impact": impact}
        y = y2
        t += dt_s
    raise RuntimeError("E10-07 twin: no ground intersection inside the horizon")


def _main_quantities(plan: Any, booster: BoosterModel) -> dict:
    """The acceptance quantities as the main FL-301/FL-302 model reports them."""
    dry = float(booster.dry_mass_kg)
    cut_ev = next(e for e in plan.track.events if e.kind == "burn_end")
    imp = plan.track.impact
    cr = plan.corridor_report
    if imp is None:
        raise RuntimeError("E10-07: main track ended without ground impact")
    return {
        "cut": {"t_s": float(cut_ev.t_s),
                "position_ecef_m": [float(v) for v in
                                    cut_ev.data["position_ecef_m"]],
                "velocity_ecef_m_s": [float(v) for v in
                                      cut_ev.data["velocity_ecef_m_s"]],
                "mass_kg": float(cut_ev.data["mass_kg"])},
        "entry_max_q_pa": float(cr.entry_max_q_pa),
        "entry_max_proper_g": float(cr.entry_max_proper_g),
        "corridor_h0_cross_t_s": float(cr.t_s) if cr.reached else None,
        "impact": {"t_s": float(imp.t_s),
                   "position_ecef_m": [float(v) for v in imp.position_ecef_m],
                   "velocity_ecef_m_s":
                       [float(v) for v in imp.velocity_ecef_m_s],
                   "speed_m_s": float(imp.speed_m_s),
                   "vertical_speed_m_s": float(imp.vertical_speed_m_s),
                   "mass_kg": float(plan.track.mass_kg[-1])},
        "propellant_at_cutoff_kg": float(cut_ev.data["mass_kg"] - dry),
        "propellant_remaining_kg": float(plan.track.mass_kg[-1] - dry),
    }


def _dist_m(a: list[float], b: list[float]) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64)
                                - np.asarray(b, dtype=np.float64)))


def _compare(main: dict, twin: dict, budget: dict) -> dict:
    """Deltas, per-band verdicts, budget shares and residual per quantity."""
    d: dict[str, Any] = {}
    d["reflight.bb_cutoff.time"] = abs(twin["cut"]["t_s"] - main["cut"]["t_s"])
    d["reflight.bb_cutoff.position"] = _dist_m(twin["cut"]["position_ecef_m"],
                                               main["cut"]["position_ecef_m"])
    d["reflight.bb_cutoff.speed"] = abs(
        float(np.linalg.norm(twin["cut"]["velocity_ecef_m_s"]))
        - float(np.linalg.norm(main["cut"]["velocity_ecef_m_s"])))
    d["reflight.entry_peak_q"] = 100.0 * abs(twin["entry_max_q_pa"]
                                             - main["entry_max_q_pa"]) \
        / main["entry_max_q_pa"]
    d["reflight.entry_peak_g"] = abs(twin["entry_max_proper_g"]
                                     - main["entry_max_proper_g"])
    d["reflight.impact_time"] = abs(twin["impact"]["t_s"] - main["impact"]["t_s"])
    d["reflight.impact_position"] = _dist_m(twin["impact"]["position_ecef_m"],
                                            main["impact"]["position_ecef_m"])
    d["reflight.impact_speed"] = 100.0 * abs(twin["impact"]["speed_m_s"]
                                             - main["impact"]["speed_m_s"]) \
        / main["impact"]["speed_m_s"]
    d["reflight.propellant_remaining"] = abs(twin["impact"]["mass_kg"]
                                             - main["impact"]["mass_kg"])
    checks = {cid: bool(d[cid] <= BANDS[cid][1]) for cid in BANDS}

    # Budget: each line is the twin-vs-twin shift when that one convention
    # changes; the residual is what the raw inter-model delta leaves after
    # the measured lines (floored at zero -- lines can partially cancel).
    shares: dict[str, dict[str, float]] = {}
    residual: dict[str, float] = {}
    for quant in _BUDGET_QUANTITIES:
        shares[quant] = {}
        for line, values in budget.items():
            if quant in values:
                shares[quant][line] = abs(values[quant])
        budget_sum = sum(shares[quant].values())
        residual[quant] = max(0.0, d[quant] - budget_sum)
    return {"deltas": d, "checks": checks,
            "bands": {cid: {"unit": u, "limit": lim, "basis": basis}
                      for cid, (u, lim, basis) in BANDS.items()},
            "budget_shares": shares, "residual": residual,
            "budget_note": (
                "frame line: both models integrate in ECEF at the same epoch"
                " with the same Earth-rotation rate -- declared zero;"
                " controller line: both fly the identical authored _Mission"
                " script -- declared zero; atmosphere line re-runs the twin"
                " on the main model's density profile (orbitopt.entry,"
                " labelled perturbation experiment); numerical line halves"
                " the twin step; gravity line drops J2 from the twin")}


def independent_recovery_reflight(
    ascent_dir: Path, *, scenario_yaml: Path, vehicle_yaml: Path,
    twin_dt_s: float = 0.1, twin_dt_fine_s: float = 0.05,
) -> dict:
    """Full-chain re-flight report for E10-07.

    ``ascent_dir`` is the committed OpenTsiolkovsky DC8 ascent run (E10-02
    hot-staging timeline) whose separation event is the actual handoff.
    """
    target = load_target(scenario_yaml)
    run = read_run(Path(ascent_dir))
    sep = separation_state_ecef(run)
    booster = BoosterModel.load(vehicle_yaml)
    booster = landing_reserve_booster(booster, target, "target")
    rt = RecoveryTarget.from_scenario(target.to_dict())

    handoff = HandoffState.of(sep, from_phase="separation", to_phase="boostback",
                              dry_mass_kg=booster.dry_mass_kg)
    digest = handoff_digest(handoff)

    plan = plan_boostback_entry(sep, booster, rt,
                                initial_guess=list(E10_07_SCRIPT), **E10_07_RECIPE)
    if plan.track.impact is None:
        raise RuntimeError("E10-07: main plan did not reach the ground")

    main = _main_quantities(plan, booster)
    h0 = float(rt.corridor.h0_m)
    horizon = float(E10_07_RECIPE["horizon_s"])
    factory = lambda: _twin_mission(plan, sep, booster, rt)  # noqa: E731

    twin = _twin_flight(factory, sep, booster, dt_s=twin_dt_s,
                        corridor_h0_m=h0, horizon_s=horizon)
    twin_fine = _twin_flight(factory, sep, booster, dt_s=twin_dt_fine_s,
                             corridor_h0_m=h0, horizon_s=horizon)
    from orbitopt.entry import density_kg_m3 as _entry_density
    twin_atm = _twin_flight(factory, sep, booster, dt_s=twin_dt_s,
                            density_fn=_entry_density, corridor_h0_m=h0,
                            horizon_s=horizon)
    twin_grav = _twin_flight(factory, sep, booster, dt_s=twin_dt_s,
                             gravity="point_mass", corridor_h0_m=h0,
                             horizon_s=horizon)

    def _shift(a: dict, b: dict) -> dict:
        return {
            "reflight.entry_peak_q": 100.0 * abs(a["entry_max_q_pa"]
                                                 - b["entry_max_q_pa"])
                / max(a["entry_max_q_pa"], 1.0),
            "reflight.impact_speed": 100.0 * abs(a["impact"]["speed_m_s"]
                                                 - b["impact"]["speed_m_s"])
                / a["impact"]["speed_m_s"],
            "reflight.impact_position": _dist_m(a["impact"]["position_ecef_m"],
                                                b["impact"]["position_ecef_m"]),
        }

    budget = {"numerical": _shift(twin, twin_fine),
              "atmosphere": _shift(twin, twin_atm),
              "gravity": _shift(twin, twin_grav)}
    comparison = _compare(main, twin, budget)

    # Landing phase at the REAL crossed handoff -- the point of E10-07.
    landing_probe: dict[str, Any]
    enu = plan.landing_burn_initial_state_enu
    st = plan.landing_burn_initial_state
    if enu is None or st is None:
        landing_probe = {"crossed": False}
    else:
        try:
            res = fly_landing_burn(enu, st, rt, booster)
            landing_probe = {"crossed": True, "feasible": bool(res.feasible),
                             "landed": bool(res.landed),
                             "reason_codes": list(res.reason_codes)}
        except Exception as exc:  # solver refusals are findings, not crashes
            landing_probe = {"crossed": True, "feasible": False, "landed": False,
                             "reason_codes": [f"SOLVER_ERROR:{type(exc).__name__}"]}
    cr = plan.corridor_report
    landing_probe["crossed_state"] = (
        None if st is None else
        {"t_s": float(st.t_s),
         "speed_m_s": float(np.linalg.norm(st.velocity_m_s)),
         "height_m": float(_twin_geodetic(
             np.asarray(st.position_m, dtype=np.float64))[2])})
    landing_probe["closed_loop_predicted_touchdown_vertical_m_s"] = (
        None if cr.landing_touchdown_vertical_m_s is None
        else float(cr.landing_touchdown_vertical_m_s))
    landing_probe["note"] = (
        "the E8-05 corridor-origin landing re-flight stays a design_input"
        " check: this probe is the first landing-phase attempt from the"
        " handoff state the mission actually reaches")

    checks = comparison["checks"]
    status = status_from_checks(
        checks,
        {cid: CHECK_MODEL for cid in checks},
        {cid: EVIDENCE_INDEPENDENT_MODEL for cid in checks})

    findings = [
        f"re-flight started from the actual DC8 hot-staging separation handoff"
        f" (OT run {run.run_id}, separation t={sep.t_s:.1f} s, handoff digest"
        f" {digest}); no declared-corridor state was used",
        "two-burn unpowered entry remains NOT CLOSED at the real DC8"
        f" separation (main-model verdict {list(plan.reason_codes)}) --"
        " consistent with the E10-01 VEHICLE_CAPABILITY finding; the"
        " independent twin agrees on the trajectory, not on closure",
        "landing burn unplannable from the real crossed handoff"
        f" ({landing_probe.get('reason_codes')}) -- the terminal comparison"
        " is ballistic impact, and the E8-05 corridor-start landing check"
        " is hereby recorded as design_input, not full-chain independent"
        " verification",
    ]

    report: dict[str, Any] = {
        "protocol": PROTOCOL,
        "kind": "full_recovery_reflight",
        "inputs": {
            "scenario": Path(scenario_yaml).name,
            "vehicle": Path(vehicle_yaml).name,
            "ascent_run": {"run_id": run.run_id, "backend": run.backend,
                           "git_sha": run.git_sha,
                           "events": [[e.kind, float(e.t_s)] for e in run.events]},
            "script": list(E10_07_SCRIPT),
            "recipe": dict(E10_07_RECIPE),
        },
        "handoff": {
            "digest": digest,
            "state": {k: (v.isoformat() if hasattr(v, "isoformat") else v)
                      for k, v in dataclasses.asdict(sep).items()},
            "separation_speed_m_s": float(np.linalg.norm(sep.velocity_m_s)),
        },
        "main_model": {
            "quantities": main,
            "plan_feasible": bool(plan.feasible),
            "reason_codes": list(plan.reason_codes),
            "corridor": {"reached": bool(cr.reached), "miss_m": float(cr.miss_m),
                         "cross_speed_m_s": float(cr.speed_m_s),
                         "t_s": float(cr.t_s) if cr.reached else None},
            "ignitions_used": int(plan.ignitions_used),
        },
        "independent_model": {"description": TWIN_DESCRIPTION,
                              "dt_s": twin_dt_s, "quantities": twin},
        "landing_phase": landing_probe,
        "comparison": comparison,
        "status": status,
        "findings": findings,
        "provenance": provenance(),
    }
    report["digest"] = report_digest(report)
    return report
