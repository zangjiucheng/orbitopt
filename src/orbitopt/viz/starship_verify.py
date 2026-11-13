"""Verification of the Starship Flight 15 return (orbitopt.viz.starship_flight).

The return is *targeted* inside its own model -- 0.3 m from the landing
point it was solved for -- so flying the same model again would prove
nothing. These reports check what that number cannot tell:

1. **Re-fly** -- the solved burn and entry flown again by independent
   numerics: the orbit from the tighter-tolerance tudatpy propagation, the
   entry with a different integrator (RK45 instead of DOP853) and step limit.
   The landing point must not move: the 0.3 m is not an integration artefact.
2. **Dispersions** -- the burn as executed is off (time, delta-v) or the
   vehicle / air are not as assumed (density, L/D). Open loop, each moves
   the landing point by tens to hundreds of km, which is why entry is
   guided: for each case the bank angle is re-solved (and, if the ship
   would overshoot sideways, one bank reversal is allowed, as entry
   guidance does). Passed means the ship can still come down within reach
   of Starbase with a bank inside the guidance limit.
3. **Limits** -- loads and propellant: peak deceleration (also across the
   dispersions), dynamic pressure, deorbit and landing-burn delta-v.
4. **Timeline** -- the solved return against SpaceX's published timeline,
   which the model never uses as an input.

Every threshold is a named constant below. Most are *assumptions* (SpaceX
publishes no limits for Starship V3) and are labelled as such in the
report. The reports use the verification-report-1.0 schema like every other
mission's, with no propagation sections (the orbit's own step-convergence
check is the re-fly's first check); the numbers behind each check ride in
``targets["metrics"]`` for the Verification tab. In the report's categories
(task #243): the re-fly is numerical consistency, the dispersions and limits
model agreement, the timeline check historical validation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from orbitopt.ground_track import eci_to_ecef
from orbitopt.verify.report import (
    CHECK_HISTORICAL,
    CHECK_MODEL,
    CHECK_NUMERICAL,
    build_report,
    report_digest,
    state_residual,
    status_from_checks,
)
from orbitopt.viz import starship_ascent as ascent_mod
from orbitopt.viz import starship_flight as sf

# ------------------------------------------------------------ thresholds
REFLY_TOLERANCE_M = 100.0
"""The re-flown landing-burn start point must stay within this of the
nominal one (the solver's own target is 100 m)."""
PASS_TIME_TOLERANCE_S = 1.0
"""Tighter tudatpy tolerances (x0.01) may move the return pass by at most this."""
MAX_BANK_DEG = 70.0
"""Guidance limit on the bank a dispersed case may need (assumed). Above
~70 deg vertical lift runs out and the entry steepens fast."""
CROSSRANGE_TOLERANCE_KM = 5.0
"""Sideways miss a dispersed case may keep at the 1 km landing-burn point
(assumed): what the subsonic belly-flop glide and the flip can trim --
neither is in the 3-DOF model."""
MAX_ENTRY_G = 4.0
"""Peak aerodynamic deceleration (assumed limit; flown Starship entries
peaked near 2-3 g)."""
MAX_DYNAMIC_PRESSURE_KPA = 15.0
"""Peak dynamic pressure (assumed limit)."""
DEORBIT_DV_BUDGET_M_S = 150.0
"""Deorbit burn delta-v the header tanks are assumed to allow."""
LANDING_DV_BUDGET_M_S = 700.0
"""Landing burn delta-v (incl. gravity loss) the header tanks are assumed to
allow; a Raptor landing burn from terminal velocity uses a few hundred m/s."""
ENTRY_TIME_TOLERANCE_S = 300.0
"""Predicted entry interface and tower catch vs SpaceX's published times."""
ANNOUNCED_ORBITS = 14
"""SpaceX: 'approximately 14 orbits' before entry, read as the number of
*completed* orbits -- the ship enters part-way through the next one."""


@dataclass(frozen=True)
class Dispersion:
    key: str
    label: str
    burn_time_s: float = 0.0
    deorbit_dv_m_s: float = 0.0
    density_scale: float = 1.0
    lift_scale: float = 1.0


DISPERSIONS = (
    Dispersion("density_low", "Air density -20%", density_scale=0.8),
    Dispersion("density_high", "Air density +20%", density_scale=1.2),
    Dispersion("burn_late", "Deorbit burn 1 s late", burn_time_s=1.0),
    Dispersion("burn_early", "Deorbit burn 1 s early", burn_time_s=-1.0),
    Dispersion("dv_high", "Deorbit delta-v +1 m/s", deorbit_dv_m_s=1.0),
    Dispersion("dv_low", "Deorbit delta-v -1 m/s", deorbit_dv_m_s=-1.0),
    Dispersion("lift_low", "Lift-to-drag -10%", lift_scale=0.9),
    Dispersion("lift_high", "Lift-to-drag +10%", lift_scale=1.1),
)
"""Density +-20%: the spread of the real atmosphere against a standard
model at entry heights. Burn +-1 s / +-1 m/s: a coarse execution error for a
Raptor burn. L/D +-10%: the flat-plate aerodynamics is a first estimate."""


def _metric(name: str, value: str, limit: str, ok: bool | None, category: str = CHECK_MODEL) -> dict:
    return {"name": name, "value": value, "limit": limit, "ok": ok, "category": category}


def _report(kind: str, label: str, checks: dict, metrics: list, inputs: dict, residuals=(),
            category: str = CHECK_MODEL) -> dict:
    """One report: every check (and metric) belongs to ``category`` -- these
    reports check one thing each, so all their checks share a category."""
    report = build_report(kind, label, propagations=[], residuals=list(residuals), inputs=inputs,
                          targets={"metrics": metrics})
    report["status"] = status_from_checks(checks, dict.fromkeys(checks, category))
    report["digest"] = report_digest(report)
    return report


def _end_ecef(pred, burn_time_s: float, traj) -> np.ndarray:
    utc = pred.liftoff_mjd2000 + (burn_time_s + traj.times_s[-1]) / 86400.0
    return eci_to_ecef(traj.states[-1][None, :3], [utc])[0]


# ------------------------------------------------------------ 1. re-fly
def refly_report(pred, tight, sol) -> dict:
    nominal, _ = sf._fly_from(pred, sol.burn_time_s, sol.bank_deg, sol.dv_normal_m_s)
    again, _ = sf._fly_from(tight, sol.burn_time_s, sol.bank_deg, sol.dv_normal_m_s,
                            method="RK45", rtol=1e-11, atol=1e-5, max_step_s=10.0)
    residual = state_residual(
        "landing_burn_start", "nominal solve (tudatpy orbit, DOP853 entry)",
        position=_end_ecef(tight, sol.burn_time_s, again), reference_position=_end_ecef(pred, sol.burn_time_s, nominal),
        tolerance={"position_m": REFLY_TOLERANCE_M}, category=CHECK_NUMERICAL)
    miss_m = float(np.linalg.norm(sf._miss_vector_km(tight, sol.burn_time_s, again))) * 1000.0
    pass_dt = (abs(pred.return_pass.time_s - tight.return_pass.time_s)
               if pred.return_pass is not None and tight.return_pass is not None else math.inf)
    checks = {
        "orbit_step_convergence": pass_dt <= PASS_TIME_TOLERANCE_S,
        "refly_matches_nominal": bool(residual["within_tolerance"]),
        "refly_reaches_starbase": miss_m <= REFLY_TOLERANCE_M,
        "reaches_landing_burn_height": bool(again.ended_at_height),
    }
    metrics = [
        _metric("Return pass, tolerances x0.01", f"moves {pass_dt:.3f} s", f"<= {PASS_TIME_TOLERANCE_S:g} s",
                checks["orbit_step_convergence"], CHECK_NUMERICAL),
        _metric("Landing point, re-flown vs nominal", f"{residual['position_residual_m']:.1f} m",
                f"<= {REFLY_TOLERANCE_M:g} m", checks["refly_matches_nominal"], CHECK_NUMERICAL),
        _metric("Re-flown miss from Starbase", f"{miss_m:.1f} m", f"<= {REFLY_TOLERANCE_M:g} m",
                checks["refly_reaches_starbase"], CHECK_NUMERICAL),
    ]
    return _report("starship_return_refly", "independent re-fly", checks, metrics, residuals=[residual],
                   category=CHECK_NUMERICAL, inputs={
        "orbit": "tudatpy, rtol/atol x0.01 of the nominal propagation",
        "entry": "scipy RK45, rtol 1e-11, atol 1e-5, max step 10 s (nominal: DOP853, 1e-9 / 1e-6, 60 s)",
        "burn_time_s": sol.burn_time_s, "bank_deg": sol.bank_deg, "dv_normal_m_s": sol.dv_normal_m_s})


# ------------------------------------------------------------ 2. dispersions
def _case_flight(pred, sol, case: Dispersion, bank_deg: float, reversal_s: float | None = None):
    from orbitopt.entry import EntryVehicle

    veh = sf._entry_vehicle()
    vehicle = EntryVehicle(veh.mass_kg, veh.area_m2, veh.cd, veh.cl * case.lift_scale)
    burn_t = sol.burn_time_s + case.burn_time_s
    traj, _ = sf._fly_from(pred, burn_t, bank_deg, sol.dv_normal_m_s,
                           deorbit_dv_m_s=sf.DEORBIT_DV_M_S + case.deorbit_dv_m_s, vehicle=vehicle,
                           density_scale=case.density_scale, bank_reversal_s=reversal_s)
    return sf._miss_vector_km(pred, burn_t, traj), traj


def recover(pred, sol, case: Dispersion, toward: float) -> dict:
    """Re-solve the bank (same side as the nominal) so the dispersed case
    comes down at Starbase's downrange; if it then overshoots sideways, also
    solve one bank reversal. Returns the outcome."""
    from scipy.optimize import brentq

    side = math.copysign(1.0, sol.bank_deg)

    def bank_for_downrange(reversal_s=None):
        def f(b):
            return float(_case_flight(pred, sol, case, side * b, reversal_s)[0][0])
        lo, hi = 0.0, 85.0
        if f(lo) * f(hi) > 0.0:
            return None
        return float(brentq(f, lo, hi, xtol=2e-3))

    out = {"case": case, "bank_deg": None, "reversal_s": None, "cross_km": None, "peak_g": None, "peak_q_kpa": None}
    bank = bank_for_downrange()
    if bank is None:
        return out
    miss, traj = _case_flight(pred, sol, case, side * bank)
    if toward * miss[1] > CROSSRANGE_TOLERANCE_KM:
        # Carried past Starbase sideways: reverse the bank part-way, the
        # later the reversal the further sideways it lands.
        end_s = float(traj.times_s[-1])

        def overshoot(t_rev):
            b = bank_for_downrange(t_rev)
            return math.inf if b is None else toward * float(_case_flight(pred, sol, case, side * b, t_rev)[0][1])

        t_lo = 0.3 * end_s
        if overshoot(t_lo) < 0.0:
            t_rev = float(brentq(overshoot, t_lo, end_s, xtol=1.0))
            b = bank_for_downrange(t_rev)
            if b is not None:
                bank, out["reversal_s"] = b, t_rev
                miss, traj = _case_flight(pred, sol, case, side * bank, t_rev)
    out.update(bank_deg=bank, cross_km=float(miss[1]), down_km=float(miss[0]),
               peak_g=float(traj.aero_g.max()), peak_q_kpa=float(traj.dynamic_pressure_pa.max()) / 1000.0)
    return out


def dispersion_outcomes(pred, sol) -> list[dict]:
    # Which side of Starbase is "short": lift straight up, no cross-range
    # from the entry (the same convention as solve_return).
    m0, _ = sf._fly_from(pred, sol.burn_time_s, math.copysign(1.0, sol.bank_deg), sol.dv_normal_m_s)
    toward = -math.copysign(1.0, float(sf._miss_vector_km(pred, sol.burn_time_s, m0)[1]))
    return [recover(pred, sol, case, toward) for case in DISPERSIONS]


# ------------------------------------------------------------ inclination
INCLINATION_SEARCH_DEG = (27.0, 30.5)
"""Where to look: from just above Starbase's latitude (26.0 deg, the lowest
direct-ascent inclination) up to the assumed Flight 14 value."""


def entry_reach_inclination(pred, lift_scale: float = 1.0, xtol_deg: float = 0.02) -> float | None:
    """The highest inclination at which the entry *alone* (bank only, no
    out-of-plane burn) still reaches Starbase on the return pass -- with the
    assumed lift, or scaled by ``lift_scale``. Lower inclinations put the
    pass closer to Starbase (it passes to the north). Flight 15's inclination
    is unpublished; SpaceX can pick one in this window. None if the search
    range does not bracket it. Each evaluation is a tudatpy propagation plus
    a bank search (~15 s)."""
    from scipy.optimize import brentq

    from orbitopt.entry import EntryVehicle

    veh = sf._entry_vehicle()
    vehicle = EntryVehicle(veh.mass_kg, veh.area_m2, veh.cd, veh.cl * lift_scale)

    def reach(inc):
        p = sf.predict_return(pred.liftoff_mjd2000, inclination_deg=inc, altitude_km=pred.altitude_km)
        if p.return_pass is None:
            return -math.inf
        return sf._bank_search(p, vehicle).max_reach_km

    lo, hi = INCLINATION_SEARCH_DEG
    f_lo, f_hi = reach(lo), reach(hi)
    if not (f_lo > 0.0 > f_hi):
        return None
    return float(brentq(reach, lo, hi, xtol=xtol_deg))


def dispersion_report(sol, outcomes: list[dict], inclinations: dict | None = None) -> dict:
    checks, metrics = {}, []
    for o in outcomes:
        case = o["case"]
        if o["bank_deg"] is None:
            checks[f"{case.key}.recoverable"] = False
            metrics.append(_metric(case.label, "no bank reaches Starbase's downrange", "", False))
            continue
        ok = o["bank_deg"] <= MAX_BANK_DEG and abs(o["cross_km"]) <= CROSSRANGE_TOLERANCE_KM
        checks[f"{case.key}.recoverable"] = ok
        how = f"bank {o['bank_deg']:.1f} deg" + (
            f", reversal at +{o['reversal_s'] / 60.0:.1f} min" if o["reversal_s"] is not None else "")
        metrics.append(_metric(case.label, f"{how}; {abs(o['cross_km']):.1f} km sideways",
                               f"bank <= {MAX_BANK_DEG:g} deg, <= {CROSSRANGE_TOLERANCE_KM:g} km", ok))
    # Informational (no check): which inclinations would let the entry reach
    # Starbase without the out-of-plane burn the assumed one needs.
    for key, label in (("nominal", "Entry alone reaches Starbase"),
                       ("lift_low", "... also with lift-to-drag -10%")):
        inc = (inclinations or {}).get(key)
        if key in (inclinations or {}):
            metrics.append(_metric(label, "not in the searched range" if inc is None else f"inclination <= {inc:.2f} deg",
                                   f"assumed {sf.INCLINATION_DEG:g} deg (Flight 14)", None))
    return _report("starship_return_dispersions", "guided recovery of dispersed entries", checks, metrics, inputs={
        "nominal_bank_deg": sol.bank_deg,
        "method": "per case: burn as executed, bank re-solved for Starbase's downrange; "
                  "one bank reversal allowed if it overshoots sideways",
        "cases": [c.label for c in DISPERSIONS],
        "entry_only_inclination_limits_deg": inclinations or {}})


# ------------------------------------------------------------ 3. limits
def limits_report(sol, outcomes: list[dict]) -> dict:
    entry = sol.entry
    peak_g = float(entry.aero_g.max())
    peak_q = float(entry.dynamic_pressure_pa.max()) / 1000.0
    flown = [o for o in outcomes if o["peak_g"] is not None]
    worst_g = max([peak_g] + [o["peak_g"] for o in flown])
    worst_q = max([peak_q] + [o["peak_q_kpa"] for o in flown])
    deorbit = float(np.linalg.norm(sol.burn_dv_eq))
    checks = {
        "peak_deceleration": worst_g <= MAX_ENTRY_G,
        "peak_dynamic_pressure": worst_q <= MAX_DYNAMIC_PRESSURE_KPA,
        "deorbit_dv_budget": deorbit <= DEORBIT_DV_BUDGET_M_S,
        "landing_dv_budget": sol.landing_dv_m_s <= LANDING_DV_BUDGET_M_S,
    }
    metrics = [
        _metric("Peak deceleration (nominal / worst dispersed)", f"{peak_g:.2f} / {worst_g:.2f} g",
                f"<= {MAX_ENTRY_G:g} g (assumed)", checks["peak_deceleration"]),
        _metric("Peak dynamic pressure (nominal / worst)", f"{peak_q:.1f} / {worst_q:.1f} kPa",
                f"<= {MAX_DYNAMIC_PRESSURE_KPA:g} kPa (assumed)", checks["peak_dynamic_pressure"]),
        _metric("Deorbit burn", f"{deorbit:.0f} m/s ({abs(sol.dv_normal_m_s):.0f} out of plane)",
                f"<= {DEORBIT_DV_BUDGET_M_S:g} m/s (assumed)", checks["deorbit_dv_budget"]),
        _metric("Landing burn incl. gravity loss", f"{sol.landing_dv_m_s:.0f} m/s",
                f"<= {LANDING_DV_BUDGET_M_S:g} m/s (assumed)", checks["landing_dv_budget"]),
    ]
    return _report("starship_entry_limits", "loads and propellant", checks, metrics, inputs={
        "note": "SpaceX publishes no Starship V3 limits: every limit here is an assumption"})


# ------------------------------------------------------------ 4. timeline
def timeline_report(pred, sol) -> dict:
    ei = sol.entry_interface_s
    ei_off = math.inf if ei is None else ei - sf.ANNOUNCED_ENTRY_S
    catch_off = sol.landing_end_s - ascent_mod.SHIP_CATCH_ANNOUNCED_S
    orbits = pred.orbits_by_announced_entry
    checks = {
        "entry_interface_vs_announced": abs(ei_off) <= ENTRY_TIME_TOLERANCE_S,
        "tower_catch_vs_announced": abs(catch_off) <= ENTRY_TIME_TOLERANCE_S,
        "orbits_vs_announced": ANNOUNCED_ORBITS <= orbits < ANNOUNCED_ORBITS + 1,
    }
    tol = f"within ±{ENTRY_TIME_TOLERANCE_S / 60.0:g} min"
    metrics = [
        _metric("Entry interface vs SpaceX's T+22:10:46", "—" if ei is None else f"{ei_off / 60.0:+.1f} min", tol,
                checks["entry_interface_vs_announced"], CHECK_HISTORICAL),
        _metric("Tower catch vs SpaceX's T+22:32:34", f"{catch_off / 60.0:+.1f} min", tol,
                checks["tower_catch_vs_announced"], CHECK_HISTORICAL),
        _metric("Orbits to the announced entry", f"{orbits:.2f}", f"{ANNOUNCED_ORBITS} completed",
                checks["orbits_vs_announced"], CHECK_HISTORICAL),
    ]
    return _report("starship_timeline_consistency", "against SpaceX's published timeline", checks, metrics,
                   category=CHECK_HISTORICAL,
                   inputs={"note": "the published times are never inputs to the orbit or the return solve"})


def inclination_limits(pred) -> dict:
    """entry_reach_inclination for the assumed lift and for L/D -10% (a few minutes)."""
    return {"nominal": entry_reach_inclination(pred), "lift_low": entry_reach_inclination(pred, lift_scale=0.9)}


def verify_return(pred, tight, sol, inclinations: dict | None = None) -> list[dict]:
    """All four reports for the solved return (about a minute; plus a few
    for ``inclinations``, see inclination_limits)."""
    outcomes = dispersion_outcomes(pred, sol)
    return [refly_report(pred, tight, sol), dispersion_report(sol, outcomes, inclinations),
            limits_report(sol, outcomes), timeline_report(pred, sol)]
