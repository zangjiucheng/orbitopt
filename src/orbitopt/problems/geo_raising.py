"""pygmo UDP for GPU-batched optimization of a multi-burn GTO -> GEO
orbit-raising / geostationary-insertion sequence -- the spacecraft-side half of
a GOES-style mission (launch vehicle delivers a geostationary transfer orbit;
the spacecraft's liquid apogee engine raises it to GEO over several apogee
burns). See ``docs/goes_gto_geo_mission_plan.md`` for the mission context.

Model (a coarse, closed-form *screening* model, the fast half of the
framework's two-stage pattern -- refine/verify the winner in tudatpy):

  * All ``n_burns`` impulsive burns happen at a shared apogee, taken to be the
    geostationary radius ``r_geo``. Each burn raises the perigee (rounding the
    ellipse) and removes part of the residual inclination. After the last burn
    the orbit is circular and equatorial at ``r_geo`` (GEO) *by construction*.
  * The burn cost is the velocity-triangle magnitude at apogee:
    ``dv = sqrt(v_before^2 + v_after^2 - 2 v_before v_after cos(di))`` with
    ``v_before``/``v_after`` the vis-viva apogee speeds before/after the perigee
    raise and ``di`` the plane change taken at that burn.

Why multiple burns at all: under *ideal impulse* the total dv does NOT improve
by splitting -- the optimum collapses to a single combined apogee burn (all the
plane change + perigee raise done together at the lowest-speed injection
apogee). The real driver is **finite thrust**: a liquid apogee engine can only
impart so much dv per apogee pass (bounded apogee dwell time x thrust/mass;
GOES-16 held each burn to <41 min). Pass ``max_dv_per_burn_ms`` to enforce that
per-pass cap as a soft penalty -- then the minimum feasible ``n_burns`` is
``ceil(total_dv / cap)`` (~5 for a GOES-like injection with a ~225 m/s cap,
matching the flown campaign).

Assumption / known floor (from the mission-plan critic): the real GOES injection
apogee sits ~500 km *below* GEO, so modelling the shared apogee at ``r_geo``
makes the total dv a slight *lower bound*. Fine for a global screening seed; the
tudatpy stage (see verify/) closes the gap with the real geometry + perturbations.

Units: km and km/s internally (matching dynamics/nbody_gpu); the fitness
objective is total delta-v (+ any penalty) in **m/s**. Hand off to the
SI-meters tudatpy stage with ``orbitopt.problems.free_return.as_meters``.
"""
from __future__ import annotations

import numpy as np

from orbitopt.core.gpu import get_array_module, to_numpy
from orbitopt.core.problem import OrbitOptProblem
from orbitopt.dynamics.nbody_gpu import MU_EARTH_KM3_S2

R_EARTH_KM = 6378.137
SIDEREAL_DAY_S = 86164.0905


def geostationary_radius_km(mu_km3_s2: float = MU_EARTH_KM3_S2,
                            period_s: float = SIDEREAL_DAY_S) -> float:
    """Radius of a circular orbit whose period is one sidereal day (~42,164 km)."""
    return (mu_km3_s2 * (period_s / (2.0 * np.pi)) ** 2) ** (1.0 / 3.0)


def single_impulse_geo_insertion_ms(gto_perigee_km: float, gto_apogee_km: float,
                                    gto_inclination_deg: float,
                                    mu_km3_s2: float = MU_EARTH_KM3_S2) -> float:
    """Closed-form dv (m/s) for doing the whole GTO->GEO insertion in one apogee
    burn (the ``n_burns == 1`` case, which has no free parameters to optimize).
    Useful as the top of the n_burns sweep in examples/10."""
    r_geo = geostationary_radius_km(mu_km3_s2)
    rp0 = R_EARTH_KM + gto_perigee_km
    a0 = 0.5 * (rp0 + r_geo)  # inject apogee modelled at r_geo
    v_apo = np.sqrt(mu_km3_s2 * (2.0 / r_geo - 1.0 / a0))
    v_geo = np.sqrt(mu_km3_s2 / r_geo)
    di = np.radians(gto_inclination_deg)
    dv = np.sqrt(v_apo ** 2 + v_geo ** 2 - 2.0 * v_apo * v_geo * np.cos(di))
    return float(dv * 1000.0)


class GeoRaisingProblem(OrbitOptProblem):
    """Decision vector (length ``2*(n_burns-1)``, all in [0, 1]):

        [ fi_0 .. fi_{N-2},  fr_0 .. fr_{N-2} ]

    ``fi_k`` = fraction of the *remaining* inclination removed at burn k (the
    last burn takes whatever is left, so the plane changes always sum to the
    initial inclination). ``fr_k`` = fraction of the *remaining* perigee gap
    (r_geo - current perigee) closed at burn k (the last burn closes to r_geo,
    circularizing). This encoding keeps every intermediate orbit feasible and
    monotonic with plain box bounds -- the terminal orbit is circular-equatorial
    -GEO for *every* point in the box, so no terminal penalty is ever needed.

    Objective (single): total impulsive delta-v in m/s, plus (if
    ``max_dv_per_burn_ms`` is set) a soft penalty for any burn exceeding the
    per-apogee-pass finite-burn cap.
    """

    def __init__(
        self,
        gto_perigee_km: float,
        gto_apogee_km: float,
        gto_inclination_deg: float,
        n_burns: int,
        wet_mass_kg: float = 5192.0,
        dry_mass_kg: float = 2857.0,
        isp_s: float = 324.0,
        thrust_n: float = 458.0,
        target_longitude_deg: float = -75.2,
        max_dv_per_burn_ms: float | None = None,
        penalty_weight: float = 20.0,
        mu_km3_s2: float = MU_EARTH_KM3_S2,
        use_gpu: bool | None = None,
    ):
        super().__init__(use_gpu=use_gpu)
        if int(n_burns) < 2:
            raise ValueError(
                "GeoRaisingProblem needs n_burns >= 2; the n_burns == 1 case is "
                "fully determined -- use single_impulse_geo_insertion_ms()."
            )
        self.n_burns = int(n_burns)
        self.mu = float(mu_km3_s2)
        self.r_geo = geostationary_radius_km(self.mu)
        self.rp0 = R_EARTH_KM + float(gto_perigee_km)
        self.ra_injection = R_EARTH_KM + float(gto_apogee_km)
        self.r_burn = self.r_geo  # model: shared apogee taken at GEO radius
        self.i0 = np.radians(float(gto_inclination_deg))
        self.wet_mass_kg = float(wet_mass_kg)
        self.dry_mass_kg = float(dry_mass_kg)
        self.isp_s = float(isp_s)
        self.thrust_n = float(thrust_n)
        self.target_longitude_deg = float(target_longitude_deg)
        self.max_dv_per_burn_ms = None if max_dv_per_burn_ms is None else float(max_dv_per_burn_ms)
        self.penalty_weight = float(penalty_weight)

    # --- pygmo UDP protocol -------------------------------------------------
    def get_bounds(self):
        n = 2 * (self.n_burns - 1)
        return [0.0] * n, [1.0] * n

    def fitness(self, dv):
        return self.batch_fitness(np.asarray(dv, dtype=float))

    def batch_fitness(self, dvs):
        xp = get_array_module(self.use_gpu)
        n = 2 * (self.n_burns - 1)
        dv_matrix = xp.asarray(np.asarray(dvs, dtype=float)).reshape(-1, n)
        burn_dv_ms = self._burn_dvs_kms(xp, dv_matrix) * 1000.0          # (M, N)
        objective = burn_dv_ms.sum(axis=1)                               # (M,)
        if self.max_dv_per_burn_ms is not None:
            over = xp.maximum(burn_dv_ms - self.max_dv_per_burn_ms, 0.0)
            objective = objective + self.penalty_weight * over.sum(axis=1)
        return to_numpy(objective).ravel()

    # --- physics ------------------------------------------------------------
    def _apogee_speed(self, xp, rp):
        """Vis-viva speed at apogee ``r_burn`` of an ellipse with perigee ``rp``."""
        a = 0.5 * (rp + self.r_burn)
        return xp.sqrt(self.mu * (2.0 / self.r_burn - 1.0 / a))

    def _burn_dvs_kms(self, xp, dv_matrix):
        """Per-burn delta-v (km/s), shape (M, n_burns)."""
        m = dv_matrix.shape[0]
        n = self.n_burns
        frac_i = dv_matrix[:, : n - 1]
        frac_r = dv_matrix[:, n - 1:]

        rem_i = xp.full((m,), self.i0)
        di = xp.zeros((m, n))
        for k in range(n - 1):
            take = rem_i * frac_i[:, k]
            di[:, k] = take
            rem_i = rem_i - take
        di[:, n - 1] = rem_i

        rp_prev = xp.full((m,), self.rp0)
        v_before = self._apogee_speed(xp, rp_prev)
        out = xp.zeros((m, n))
        for k in range(n):
            if k < n - 1:
                rp_k = rp_prev + frac_r[:, k] * (self.r_burn - rp_prev)
            else:
                rp_k = xp.full((m,), self.r_burn)
            v_after = self._apogee_speed(xp, rp_k)
            out[:, k] = xp.sqrt(v_before ** 2 + v_after ** 2
                                - 2.0 * v_before * v_after * xp.cos(di[:, k]))
            v_before = v_after
            rp_prev = rp_k
        return out

    # --- helpers for the example / verifier / viz ---------------------------
    def decode(self, dv):
        """Expand one decision vector into the per-burn maneuver schedule and
        the intermediate osculating orbit after each burn. Pure numpy (single
        candidate)."""
        n = self.n_burns
        dv = np.asarray(dv, dtype=float).ravel()
        frac_i = dv[: n - 1]
        frac_r = dv[n - 1:]

        di = np.zeros(n)
        rem_i = self.i0
        for k in range(n - 1):
            di[k] = rem_i * frac_i[k]
            rem_i -= di[k]
        di[n - 1] = rem_i

        rp_before = np.empty(n)
        rp_after = np.empty(n)
        dvk = np.empty(n)
        incl_after = np.empty(n)
        rp_prev = self.rp0
        incl = self.i0
        for k in range(n):
            rp_k = self.r_burn if k == n - 1 else rp_prev + frac_r[k] * (self.r_burn - rp_prev)
            vb = float(np.sqrt(self.mu * (2.0 / self.r_burn - 1.0 / (0.5 * (rp_prev + self.r_burn)))))
            va = float(np.sqrt(self.mu * (2.0 / self.r_burn - 1.0 / (0.5 * (rp_k + self.r_burn)))))
            rp_before[k] = rp_prev
            rp_after[k] = rp_k
            dvk[k] = float(np.sqrt(vb ** 2 + va ** 2 - 2.0 * vb * va * np.cos(di[k])))
            incl -= di[k]
            incl_after[k] = incl
            rp_prev = rp_k

        dvk_ms = dvk * 1000.0
        cap = self.max_dv_per_burn_ms
        return {
            "n_burns": n,
            "r_geo_km": self.r_geo,
            "r_burn_km": self.r_burn,
            "perigee_before_km": rp_before,
            "perigee_after_km": rp_after,
            "apogee_km": np.full(n, self.r_burn),
            "plane_change_deg": np.degrees(di),
            "inclination_after_deg": np.degrees(incl_after),
            "dv_per_burn_ms": dvk_ms,
            "dv_total_ms": float(dvk_ms.sum()),
            "max_burn_ms": float(dvk_ms.max()),
            "feasible": bool(cap is None or dvk_ms.max() <= cap + 1e-6),
        }

    def delta_v_total_ms(self, dv) -> float:
        """Total maneuver delta-v (m/s), excluding penalty."""
        return float(self.decode(dv)["dv_total_ms"])

    def tsiolkovsky_capacity_ms(self) -> float:
        """Total onboard dv the propellant can supply (Tsiolkovsky ceiling)."""
        g0 = 9.80665
        return float(g0 * self.isp_s * np.log(self.wet_mass_kg / self.dry_mass_kg))

    def max_dv_per_pass_ms(self, dwell_s: float) -> float:
        """Rough finite-burn cap = thrust/mass * apogee dwell (m/s), at wet mass."""
        return float(self.thrust_n / self.wet_mass_kg * dwell_s)
