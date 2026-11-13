"""Benchmark scenarios with public reference data or analytic solutions (task #245).

Three scenarios, each with a *frozen expected value* (a published number or
the closed-form solution of the same problem) and a *declared tolerance*:
the measurement is the physics engine's answer, never the reference.

1. **kepler_two_body** -- the analytic two-body solution (classical elements
   + Kepler's equation) of a defined initial state after 7500 s, against a
   numerically propagated one (scipy DOP853, rtol 1e-12). Also checks the
   numerical run's own energy drift: under a point mass the specific orbital
   energy is an exact integral, so anything but float noise is integration
   error. The reference is the two-body problem itself (Vallado,
   *Fundamentals of Astrodynamics*, Ch. 2 -- closed form).
2. **j2_secular_drift** -- secular node/perigee drift of two textbook J2
   scenarios over one day (osculating state integrated with the J2 point
   acceleration, DOP853): a 700 km sun-synchronous orbit against the public
   sun-synchronous reference (+360 deg of node drift per tropical year), and
   a critical-inclination (63.435 deg) MEO orbit whose perigee drift must be
   zero (the analytic frozen-perigee condition, 5cos^2 i = 1). Drift rates
   per Vallado, *Fundamentals of Astrodynamics*, 4th ed., Sec. 9.7.
3. **starship_booster_catch** -- the Super Heavy booster return of Starship
   Flight 15: the catch time solved by ``orbitopt.viz.starship_ascent``'s
   physics (boostback, ballistic fall under the entry atmosphere model,
   landing burn) against the *published* timeline (~T+6:35, booster caught
   over the launch tower at the pad). The announced time is never an input
   to the solve (same source note as ``starship_ascent``: Wikipedia's
   flight table / spacelaunchschedule, accessed 2026-11).

The scenarios are regression gates, so the expected values are frozen
constants below, not recomputed at run time; ``tests/test_verify_benchmarks.py``
re-runs every benchmark in CI (the fast suite, so the pipeline gate covers
them) and additionally verifies the frozen constants against their formulas,
which catches drift of the Earth constants themselves.

``benchmark_report()`` wraps the results as a verification-report-1.0
document (no new schema, per epic P1: extend the existing verification
machinery, don't parallelise it), so the gate result rides the same rails
as every other verification report.

Run from the command line (exit code 1 if any benchmark fails):

    python -m orbitopt.verify.benchmarks
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
from scipy.integrate import solve_ivp

from orbitopt.entry import J2_EARTH, MU_EARTH_M3_S2, R_EARTH_M
from orbitopt.verify.report import build_report, report_digest, status_from_checks

# Integrator for every measurement here: high-accuracy explicit RK, absolute
# tolerances in m / m s-1.
NUM_METHOD = "DOP853"
NUM_RTOL = 1e-12
NUM_ATOL = 1e-6


# ---------------------------------------------------------------------------
# Two-body analytic solution (classical elements <-> Cartesian state)
# ---------------------------------------------------------------------------

def _rot3(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot1(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])


def _solve_kepler(M: float, e: float) -> float:
    """Eccentric anomaly from mean anomaly (Newton, Stumpff start)."""
    E = M + e * math.sin(M)
    for _ in range(60):
        d = (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
        E -= d
        if abs(d) < 1e-14:
            break
    return E


def coe_to_ic(a: float, e: float, i_deg: float, raan_deg: float, argp_deg: float,
              M0_deg: float, mu: float = MU_EARTH_M3_S2) -> np.ndarray:
    """Classical elements (semi-major axis m, eccentricity, angles deg) to a
    Cartesian state [m, m/s] in an inertial frame."""
    inc, raan, argp = map(math.radians, (i_deg, raan_deg, argp_deg))
    E = _solve_kepler(math.radians(M0_deg), e)
    nu = 2.0 * math.atan2(math.sqrt(1.0 + e) * math.sin(E / 2.0),
                          math.sqrt(1.0 - e) * math.cos(E / 2.0))
    r = a * (1.0 - e * math.cos(E))
    vs = math.sqrt(mu / (a * (1.0 - e * e)))
    rp = np.array([r * math.cos(nu), r * math.sin(nu), 0.0])
    vp = np.array([-vs * math.sin(nu), vs * (e + math.cos(nu)), 0.0])
    R = _rot3(-raan) @ _rot1(-inc) @ _rot3(-argp)
    return np.concatenate([R @ rp, R @ vp])


def ic_to_coe(state, mu: float = MU_EARTH_M3_S2) -> dict:
    """Cartesian state -> classical elements (``a`` m, angles deg, mean
    anomaly in [0, 360))."""
    r = np.asarray(state[:3], dtype=float)
    v = np.asarray(state[3:], dtype=float)
    rn = float(np.linalg.norm(r))
    h = np.cross(r, v)
    hn = float(np.linalg.norm(h))
    evec = ((v @ v - mu / rn) * r - (r @ v) * v) / mu
    e = float(np.linalg.norm(evec))
    a = float(1.0 / (2.0 / rn - (v @ v) / mu))
    inc = float(math.degrees(math.acos(np.clip(h[2] / hn, -1.0, 1.0))))
    node = np.cross([0.0, 0.0, 1.0], h)
    nn = float(np.linalg.norm(node))
    if nn > 1e-9:
        raan = math.degrees(math.acos(np.clip(node[0] / nn, -1.0, 1.0)))
        if node[1] < 0.0:
            raan = 360.0 - raan
    else:
        raan = 0.0
    if e > 1e-9:
        argp = math.degrees(math.acos(np.clip(node @ evec / (nn * e), -1.0, 1.0)))
        if evec[2] < 0.0:
            argp = 360.0 - argp
    else:
        argp = 0.0
    nu = math.acos(np.clip(evec @ r / (e * rn), -1.0, 1.0))
    if (r @ v) < 0.0:
        nu = 2.0 * math.pi - nu
    E = 2.0 * math.atan2(math.sqrt(1.0 - e) * math.sin(nu / 2.0),
                         math.sqrt(1.0 + e) * math.cos(nu / 2.0))
    return {"a": a, "e": e, "i_deg": inc, "raan_deg": raan % 360.0,
            "argp_deg": argp % 360.0, "M_deg": math.degrees(E - e * math.sin(E)) % 360.0}


def propagate_kepler_analytic(state0, dt: float, mu: float = MU_EARTH_M3_S2) -> np.ndarray:
    """The two-body state after ``dt`` s, exactly: elements propagated by
    M += n dt and rebuilt (the analytic reference, not an approximation)."""
    c = ic_to_coe(state0, mu)
    n = math.sqrt(mu / c["a"] ** 3)
    M0 = (c["M_deg"] + math.degrees(n * dt)) % 360.0
    return coe_to_ic(c["a"], c["e"], c["i_deg"], c["raan_deg"], c["argp_deg"], M0, mu)


def _accel_point_mass(t, s):
    r = np.asarray(s[:3], dtype=float)
    return np.concatenate([s[3:], -MU_EARTH_M3_S2 * r / float(np.linalg.norm(r)) ** 3])


def _accel_j2(t, s):
    r = np.asarray(s[:3], dtype=float)
    rn = float(np.linalg.norm(r))
    z2 = r[2] * r[2] / (rn * rn)
    k = -1.5 * J2_EARTH * MU_EARTH_M3_S2 * R_EARTH_M ** 2 / rn ** 5
    a = k * np.array([r[0] * (1.0 - 5.0 * z2), r[1] * (1.0 - 5.0 * z2), r[2] * (3.0 - 5.0 * z2)])
    return np.concatenate([s[3:], a - MU_EARTH_M3_S2 * r / rn ** 3])


def _integrate(accel, state0, dt: float) -> np.ndarray:
    return solve_ivp(accel, (0.0, dt), np.asarray(state0, dtype=float),
                     method=NUM_METHOD, rtol=NUM_RTOL, atol=NUM_ATOL).y[:, -1]


# ---------------------------------------------------------------------------
# 1. kepler_two_body
# ---------------------------------------------------------------------------

KEPLER_COE = {"a": 7.0e6, "e": 0.02, "i_deg": 53.0, "raan_deg": 40.0,
              "argp_deg": 30.0, "M0_deg": 10.0}
KEPLER_DURATION_S = 7500.0
"""~1.28 revolutions of a 7000 km, e=0.02 orbit."""
KEPLER_EXPECTED_STATE = (-5999426.598011, -1879678.332516, 3206724.110395,
                         -996.680003, -5609.932284, -4852.742673)
"""Frozen analytic state [m, m/s] after ``KEPLER_DURATION_S`` of
``KEPLER_COE``: the closed-form two-body solution evaluated once
(2026-11) from the Earth constants in :mod:`orbitopt.entry`; the test suite
re-derives it from the same formula."""
KEPLER_POS_TOL_M = 0.5
"""Position the numerical propagation may keep from the analytic state."""
KEPLER_VEL_TOL_M_S = 5e-3
"""Velocity the numerical propagation may keep from the analytic state."""
KEPLER_ENERGY_TOL_REL = 1e-9
"""Max relative drift of specific orbital energy over the propagation: an
exact integral under a point mass, so this measures pure integration error."""


def run_kepler_benchmark() -> "BenchmarkResult":
    state0 = coe_to_ic(**KEPLER_COE)
    expected = np.array(KEPLER_EXPECTED_STATE)
    actual = _integrate(_accel_point_mass, state0, KEPLER_DURATION_S)
    energy = 0.5 * (state0[3:] @ state0[3:]) - MU_EARTH_M3_S2 / float(np.linalg.norm(state0[:3]))
    energy_end = 0.5 * (actual[3:] @ actual[3:]) - MU_EARTH_M3_S2 / float(np.linalg.norm(actual[:3]))
    dpos = float(np.linalg.norm(actual[:3] - expected[:3]))
    dvel = float(np.linalg.norm(actual[3:] - expected[3:]))
    denergy = abs((energy_end - energy) / energy)
    checks = (
        _check("final_position_m", "propagated state vs analytic closed form",
               0.0, dpos, KEPLER_POS_TOL_M, "m"),
        _check("final_velocity_m_s", "propagated state vs analytic closed form",
               0.0, dvel, KEPLER_VEL_TOL_M_S, "m/s"),
        _check("specific_energy_drift", "energy is exact under a point mass",
               0.0, denergy, KEPLER_ENERGY_TOL_REL, "relative"),
    )
    return BenchmarkResult(
        "kepler_two_body", "Kepler two-body against the analytic solution",
        "closed-form two-body propagation (Vallado Ch. 2) of the frozen state "
        f"below; measurement: scipy {NUM_METHOD} rtol={NUM_RTOL:g} atol={NUM_ATOL:g}",
        checks,
        inputs={"coe_m_deg": dict(KEPLER_COE), "duration_s": KEPLER_DURATION_S,
                "expected_state_m_m_s": list(expected),
                "actual_state_m_m_s": [float(x) for x in actual]},
    )


# ---------------------------------------------------------------------------
# 2. j2_secular_drift
# ---------------------------------------------------------------------------

SUNSYNC_RAAN_RATE_DEG_PER_DAY = 360.0 / 365.2422
"""Public reference: a sun-synchronous orbit's node must drift 360 deg per
tropical year (0.9856258 deg/day)."""
SSO_ALTITUDE_M = 700e3
SSO_ECCENTRICITY = 1.0011e-3
SSO_RAAN_TOL_DEG_PER_DAY = 0.01
"""Measured minus published node drift: the osculating drift deviates from
the secular mean by the (short-period) J2 wobble, well under 0.01 deg/day."""
CRITICAL_INCLINATION_DEG = math.degrees(math.acos(1.0 / math.sqrt(5.0)))
"""63.4349 deg: 5cos^2 i = 1 -- the analytic frozen-perigee condition."""
CRIT_SEMI_MAJOR_AXIS_M = 26600e3
CRIT_ECCENTRICITY = 0.01
CRIT_EXPECTED_RAAN_DRIFT_DEG_PER_DAY = -0.030087
"""Secular node drift of the critical-inclination scenario,
-3/2 J2 n (Re/p)^2 cos i evaluated once (2026-11) from the Earth constants
in :mod:`orbitopt.entry`."""
CRIT_RAAN_TOL_DEG_PER_DAY = 0.005
CRIT_ARGP_TOL_DEG_PER_DAY = 0.02
"""Perigee drift of a critical-inclination orbit is zero analytically; the
osculating measurement carries the J2 short-period wobble."""
J2_DRIFT_WINDOW_S = 86400.0


def _j2_drift(a, e, i_deg, argp_deg=80.0, M0_deg=0.0):
    state0 = coe_to_ic(a, e, i_deg, 0.0, argp_deg, M0_deg)
    final = _integrate(_accel_j2, state0, J2_DRIFT_WINDOW_S)
    c0, c1 = ic_to_coe(state0), ic_to_coe(final)
    wrap = lambda x: (x + 180.0) % 360.0 - 180.0  # noqa: E731
    return wrap(c1["raan_deg"] - c0["raan_deg"]), wrap(c1["argp_deg"] - c0["argp_deg"])


def _sso_inclination_deg(a: float, e: float) -> float:
    """Inclination whose *secular* J2 node drift equals the sun-synchronous
    reference: i = acos(-2 rate / (3 J2 n (Re/p)^2)), Vallado Sec. 9.7.3."""
    n = math.sqrt(MU_EARTH_M3_S2 / a ** 3)
    p = a * (1.0 - e * e)
    rate = math.radians(SUNSYNC_RAAN_RATE_DEG_PER_DAY) / 86400.0
    return math.degrees(math.acos(-2.0 * rate / (3.0 * J2_EARTH * n * (R_EARTH_M / p) ** 2)))


def run_j2_benchmark() -> "BenchmarkResult":
    i_sso = _sso_inclination_deg(R_EARTH_M + SSO_ALTITUDE_M, SSO_ECCENTRICITY)
    sso_raan, _ = _j2_drift(R_EARTH_M + SSO_ALTITUDE_M, SSO_ECCENTRICITY, i_sso)
    crit_raan, crit_argp = _j2_drift(CRIT_SEMI_MAJOR_AXIS_M, CRIT_ECCENTRICITY,
                                     CRITICAL_INCLINATION_DEG)
    checks = (
        _check("sun_sync_node_drift_deg_per_day", "node drift of the sun-sync orbit vs 360 deg/tropical year",
               SUNSYNC_RAAN_RATE_DEG_PER_DAY, sso_raan, SSO_RAAN_TOL_DEG_PER_DAY, "deg/day"),
        _check("critical_node_drift_deg_per_day", "node drift of the critical-inclination orbit vs secular formula",
               CRIT_EXPECTED_RAAN_DRIFT_DEG_PER_DAY, crit_raan, CRIT_RAAN_TOL_DEG_PER_DAY, "deg/day"),
        _check("critical_perigee_drift_deg_per_day", "frozen perigee: 5cos^2 i = 1 => zero secular drift",
               0.0, crit_argp, CRIT_ARGP_TOL_DEG_PER_DAY, "deg/day"),
    )
    return BenchmarkResult(
        "j2_secular_drift", "J2 secular node/perigee drift against analytic rates",
        "secular J2 drift rates (Vallado 4th ed. Sec. 9.7) and the public "
        "sun-synchronous reference (360 deg/tropical year); measurement: one "
        "day of osculating propagation with the J2 point acceleration",
        checks,
        inputs={"window_s": J2_DRIFT_WINDOW_S,
                "sun_sync_inclination_deg": i_sso,
                "sun_sync_altitude_m": SSO_ALTITUDE_M,
                "critical_inclination_deg": CRITICAL_INCLINATION_DEG,
                "critical_semi_major_axis_m": CRIT_SEMI_MAJOR_AXIS_M},
    )


# ---------------------------------------------------------------------------
# 3. starship_booster_catch
# ---------------------------------------------------------------------------

ASCENT_SITE_ID = "starbase"
ASCENT_AZIMUTH_DEG = 106.54
"""Direct-ascent azimuth of the Flight 15 profile (southeast branch into the
30.5 deg corridor), same value the starship tests use."""
ASCENT_EXPECTED_CATCH_S = 395.0
"""SpaceX's published Flight 15 booster-catch time, ~T+6:35."""
ASCENT_CATCH_TIME_TOL_S = 60.0
"""Same acceptance the existing ascent test declares for the announced plan
(outlets disagree by seconds; the model's MECO state is itself assumed)."""
ASCENT_CATCH_POSITION_TOL_M = 10.0
"""Horizontal distance from the pad the catch must keep: Flight 15's booster
came back into the tower at the launch site (public fact); the model solves
for it, it is not imposed."""


@lru_cache(maxsize=1)
def _booster_return():
    from orbitopt.ground_track import ecef_to_geodetic, surface_distance_km
    from orbitopt.launch_sites import get_launch_site
    from orbitopt.viz.starship_ascent import solve_booster_return

    site = get_launch_site(ASCENT_SITE_ID)
    b = solve_booster_return(site, ASCENT_AZIMUTH_DEG)
    lat, lon, _h = ecef_to_geodetic(b.r_ecef[-1][None, :])
    miss_m = float(surface_distance_km(lat, lon, site.latitude_deg, site.longitude_deg)[0]) * 1e3
    return b, miss_m


def run_starship_benchmark() -> "BenchmarkResult":
    b, miss_m = _booster_return()
    checks = (
        _check("booster_catch_time_s", "solved catch vs SpaceX's published T+6:35",
               ASCENT_EXPECTED_CATCH_S, float(b.catch_s), ASCENT_CATCH_TIME_TOL_S, "s"),
        _check("booster_catch_offset_m", "catch point vs the launch tower (Flight 15 was caught at the pad)",
               0.0, miss_m, ASCENT_CATCH_POSITION_TOL_M, "m"),
    )
    return BenchmarkResult(
        "starship_booster_catch", "Starship F15 booster catch against the published timeline",
        "SpaceX's published Flight 15 timeline (Wikipedia flight table / "
        "spacelaunchschedule, accessed 2026-11); the announcement is never an "
        "input to the return solve",
        checks,
        inputs={"site_id": ASCENT_SITE_ID, "azimuth_deg": ASCENT_AZIMUTH_DEG,
                "catch_s": float(b.catch_s), "pad_offset_m": miss_m,
                "apogee_m": float(b.apogee_m)},
    )


# ---------------------------------------------------------------------------
# Results, report, runner
# ---------------------------------------------------------------------------

def _check(name: str, meaning: str, expected: float, actual: float,
           tolerance: float, unit: str) -> dict:
    return {"name": name, "meaning": meaning, "expected": float(expected),
            "actual": float(actual), "tolerance": float(tolerance), "unit": unit,
            "abs_error": float(abs(float(actual) - float(expected))),
            "passed": bool(abs(float(actual) - float(expected)) <= float(tolerance))}


@dataclass(frozen=True)
class BenchmarkResult:
    key: str
    label: str
    reference: str
    checks: tuple
    inputs: dict

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c["passed"] for c in self.checks)

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label, "reference": self.reference,
                "passed": self.passed, "checks": [dict(c) for c in self.checks],
                "inputs": dict(self.inputs)}


BENCHMARKS: dict = {
    "kepler_two_body": run_kepler_benchmark,
    "j2_secular_drift": run_j2_benchmark,
    "starship_booster_catch": run_starship_benchmark,
}


def run_benchmark(key: str) -> BenchmarkResult:
    """Run one benchmark scenario by key (``BENCHMARKS``)."""
    try:
        producer = BENCHMARKS[key]
    except KeyError:
        raise ValueError(f"unknown benchmark {key!r}; have {sorted(BENCHMARKS)}") from None
    return producer()


def run_all_benchmarks(keys: Sequence[str] | None = None) -> list:
    """Run every benchmark (or the given subset), in declaration order."""
    keys = list(BENCHMARKS) if keys is None else list(keys)
    return [run_benchmark(k) for k in keys]


def benchmark_report(results: Sequence[BenchmarkResult] | None = None) -> dict:
    """The benchmark suite as a verification-report-1.0 document: one check
    per scenario check (``<benchmark>.<check>``), every expected/actual/tuple
    pair in ``targets["metrics"]``. Digest covers everything but itself, so
    the same code on the same library versions re-runs byte-identically."""
    results = run_all_benchmarks() if results is None else list(results)
    checks, metrics = {}, []
    for r in results:
        for c in r.checks:
            checks[f"{r.key}.{c['name']}"] = c["passed"]
            metrics.append({"name": f"{r.label}: {c['meaning']}",
                            "expected": c["expected"], "actual": c["actual"],
                            "tolerance": c["tolerance"], "unit": c["unit"], "ok": c["passed"]})
    report = build_report("benchmark_scenarios", "public reference data / analytic solutions",
                          propagations=[], inputs={"benchmarks": [r.to_dict() for r in results]},
                          targets={"metrics": metrics})
    report["status"] = status_from_checks(checks)
    report["digest"] = report_digest(report)
    return report


def main(argv=None) -> int:
    """Run the benchmark suite; print a line per check; exit 1 on failure."""
    argv = list(sys.argv[1:] if argv is None else argv)
    json_out = "--json" in argv
    keys = [a for a in argv if not a.startswith("-")] or None
    results = run_all_benchmarks(keys)
    if json_out:
        print(json.dumps([r.to_dict() for r in results], indent=2, sort_keys=True))
    else:
        for r in results:
            print(f"[{'PASS' if r.passed else 'FAIL'}] {r.label}")
            for c in r.checks:
                verdict = "ok" if c["passed"] else "FAIL"
                print(f"    {verdict:4} {c['name']}: actual {c['actual']:.6g} "
                      f"expected {c['expected']:.6g} +/- {c['tolerance']:g} {c['unit']} "
                      f"(|err| {c['abs_error']:.3g})")
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
