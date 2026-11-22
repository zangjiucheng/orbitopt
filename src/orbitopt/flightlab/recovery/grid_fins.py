"""Grid-fin aerodynamics and steering for the unpowered entry (E11-01).

The reference vehicle's reentry is aerodynamic: four grid fins are the only
control surfaces between separation and the landing burn, so an ``entry_burns
= 0`` (two-burn) mission must steer with them. This module is the E11-01
addition to the FL-301 point-mass model:

* :class:`GridFinModel` -- deployed-state aerodynamics (lift, drag and pitch
  moment as functions of Mach and angle of attack) plus the actuator limits
  (deflection saturation, deflection rate, bank rate). Semi-empirical lattice
  fin aerodynamics: the fin normal-force slope is a perforated-plate slope
  scaled by the fin solid fraction and reduced with Mach in the Ackeret
  fashion; the body contributes a large-angle-of-attack cylinder normal
  force. Every number without a public source is tagged in the vehicle YAML
  block and listed in :data:`DESIGN_ASSUMPTIONS`.
* :class:`GridFinActuator` -- the two-state (stowed/deployed) actuator with
  deflection slew/saturation and bank-rate limits, sampled once per
  integration step exactly like the engine command.
* :class:`GridFinGuidance` -- the shared AoA/bank steering law (constant
  commanded angle of attack, bank command from the heading error to the pad,
  or an open-loop altitude-keyed bank schedule for capability envelopes).

Trim/authority model: at a commanded angle of attack the fins trim to
``delta = alpha / trim_effectiveness`` (aft stabiliser convention: the fin's
local angle of attack is ``alpha * (1 - 1/trim_effectiveness)``, so a fin
stalls at a vehicle alpha of ``stall_alpha_deg / (1 - 1/g)``); the
deflection saturation gives the alpha authority ``alpha_max =
trim_effectiveness * delta_max``, capped by ``alpha_max_deg``. The realised
lift is the trimmed balance ``(body + fin) * cos(alpha)``.

Sign conventions (flown ECEF; tested in ``tests/test_grid_fins_e11_01.py``):
angle of attack is positive nose-up, positive bank rolls the lift vector to
the RIGHT of the velocity, a pitch-moment coefficient is positive nose-up,
and a positive control deflection (beyond trim) commands nose-up.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from orbitopt.flightlab import frames

DEG = math.pi / 180.0

#: defaults for a Super-Heavy-class booster (S_ref ~ 64 m^2, 71 m long); a
#: vehicle block overrides anything it lists -- see DESIGN_ASSUMPTIONS below.
_DEFAULTS: dict[str, Any] = {
    "count": 4,
    "fin_area_m2": 4.0,               # public_estimate: SH grid fin ~2 x 2 m
    "cd_add_mach": ((0.1, 0.50), (0.8, 0.52), (1.2, 0.58),
                    (2.0, 0.50), (4.0, 0.36), (8.0, 0.25)),
    "cn_alpha_mach": ((0.1, 2.4), (0.8, 2.2), (1.2, 1.6),
                      (2.0, 1.1), (4.0, 0.6), (8.0, 0.3)),
    "body_cn_alpha_per_rad": 2.0,     # large-alpha cylinder (beam) normal force
    "trim_effectiveness": 1.8,        # DESIGN ASSUMPTION: fin control power vs alpha
    "delta_max_deg": 40.0,            # DESIGN ASSUMPTION: lattice fin travel
    "delta_rate_max_deg_s": 20.0,     # DESIGN ASSUMPTION: hydraulic rate
    "bank_max_deg": 60.0,             # DESIGN ASSUMPTION: steering limit
    "bank_rate_max_deg_s": 30.0,      # DESIGN ASSUMPTION: roll authority
    "alpha_max_deg": 60.0,            # vehicle/geometric alpha limit
    "stall_alpha_deg": 45.0,          # lattice fin local stall (lit. ~45-50 deg)
    "deploy_altitude_m": 70000.0,     # entry interface for the fins (public: ~70 km)
    "stow_altitude_m": 1500.0,        # stow before the landing burn
    "deploy_q_min_pa": 100.0,         # DESIGN ASSUMPTION: deploy when fins do work
    "induced_k": 0.28,                # DESIGN ASSUMPTION: low-aspect-ratio fin polar
    "length_m": 71.0,                 # pitch moment reference chord (vehicle length)
    "diameter_m": 9.0,                # roll moment reference
    "static_margin": 0.08,            # DESIGN ASSUMPTION: weathervane cp-cg margin
    "roll_lever_ratio": 0.5,          # fin radial station over vehicle diameter
}

#: DESIGN ASSUMPTIONS (task acceptance: list with uncertainty ranges). The
#: ranges are what the model could plausibly carry; they are recorded, not
#: flown, and the E11-01 report propagates the worst-case pair through the
#: envelope benchmark.
DESIGN_ASSUMPTIONS = (
    "fin area 4 x 4 m^2 (+-25%): geometry from public imagery only",
    "fin normal-force slope table (+-30%): perforated-plate/Ackeret scaling, "
    "no wind-tunnel data for this vehicle",
    "trim effectiveness 1.8, static margin 0.08 L (+-50%): stability derived "
    "from grid-fin station, no public CG/cp data",
    "delta_max 40 deg and 20 deg/s rate (+-50%): actuator travel/speed not "
    "published",
    "bank rate 30 deg/s (+-50%): aeroservoelastic realisation folded into "
    "one rate, actuator data not public",
)


def _as_pairs(node: Any, name: str) -> tuple[tuple[float, float], ...]:
    try:
        pts = tuple((float(p[0]), float(p[1])) for p in node)
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"grid_fins.{name} must be a list of [mach, value] pairs") from exc
    ms = [m for m, _ in pts]
    if len(ms) < 2 or any(b <= a for a, b in zip(ms, ms[1:])):
        raise ValueError(f"grid_fins.{name} needs >= 2 strictly increasing Mach entries")
    return pts


@dataclass(frozen=True)
class GridFinModel:
    """Validated grid-fin aerodynamics for one vehicle (see module
    docstring). ``reference_area_m2`` is the vehicle S_ref the coefficients
    are referred to (fin contributions carry the fin/vehicle area ratio
    internally)."""

    reference_area_m2: float
    count: int
    fin_area_m2: float
    cd_add_mach: tuple[tuple[float, float], ...]
    cn_alpha_mach: tuple[tuple[float, float], ...]
    body_cn_alpha_per_rad: float
    trim_effectiveness: float
    delta_max_deg: float
    delta_rate_max_deg_s: float
    bank_max_deg: float
    bank_rate_max_deg_s: float
    alpha_max_deg: float
    stall_alpha_deg: float
    deploy_altitude_m: float
    stow_altitude_m: float
    deploy_q_min_pa: float
    induced_k: float
    length_m: float
    diameter_m: float
    static_margin: float
    roll_lever_ratio: float
    provenance: str = "illustrative"
    source: str = ""
    assumptions: tuple[str, ...] = DESIGN_ASSUMPTIONS

    def __post_init__(self) -> None:
        if self.count < 2 or self.fin_area_m2 <= 0.0 or self.reference_area_m2 <= 0.0:
            raise ValueError("grid fins need count >= 2 and positive areas")
        if self.trim_effectiveness <= 1.0:
            raise ValueError("trim_effectiveness must exceed 1")
        for name in ("delta_max_deg", "bank_max_deg", "alpha_max_deg",
                     "stall_alpha_deg", "delta_rate_max_deg_s",
                     "bank_rate_max_deg_s"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.stow_altitude_m >= self.deploy_altitude_m:
            raise ValueError("grid fins: stow altitude must be below deploy altitude")

    @classmethod
    def from_mapping(cls, node: Mapping[str, Any], *,
                     reference_area_m2: float) -> "GridFinModel":
        """Build from a vehicle-YAML ``booster.grid_fins:`` block: bare keys
        override :data:`_DEFAULTS`; ``provenance``/``source``/``assumptions``
        describe the whole block. Any other key is an error (a typo in a
        control-law table must not fly as a default)."""
        known = set(_DEFAULTS) | {"provenance", "source", "assumptions"}
        unknown = sorted(set(node) - known)
        if unknown:
            raise ValueError(f"grid_fins: unknown keys {unknown}")
        merged = dict(_DEFAULTS)
        merged.update(node)
        for key in ("cd_add_mach", "cn_alpha_mach"):
            if isinstance(merged[key], Mapping) and "value" in merged[key]:
                merged[key] = merged[key]["value"]
        prov = str(node.get("provenance", "illustrative"))
        if prov not in ("official", "public_estimate", "illustrative"):
            raise ValueError("grid_fins.provenance must be "
                             "official/public_estimate/illustrative")
        return cls(
            reference_area_m2=float(reference_area_m2),
            count=int(merged["count"]),
            fin_area_m2=float(merged["fin_area_m2"]),
            cd_add_mach=_as_pairs(merged["cd_add_mach"], "cd_add_mach"),
            cn_alpha_mach=_as_pairs(merged["cn_alpha_mach"], "cn_alpha_mach"),
            body_cn_alpha_per_rad=float(merged["body_cn_alpha_per_rad"]),
            trim_effectiveness=float(merged["trim_effectiveness"]),
            delta_max_deg=float(merged["delta_max_deg"]),
            delta_rate_max_deg_s=float(merged["delta_rate_max_deg_s"]),
            bank_max_deg=float(merged["bank_max_deg"]),
            bank_rate_max_deg_s=float(merged["bank_rate_max_deg_s"]),
            alpha_max_deg=float(merged["alpha_max_deg"]),
            stall_alpha_deg=float(merged["stall_alpha_deg"]),
            deploy_altitude_m=float(merged["deploy_altitude_m"]),
            stow_altitude_m=float(merged["stow_altitude_m"]),
            deploy_q_min_pa=float(merged["deploy_q_min_pa"]),
            induced_k=float(merged["induced_k"]),
            length_m=float(merged["length_m"]),
            diameter_m=float(merged["diameter_m"]),
            static_margin=float(merged["static_margin"]),
            roll_lever_ratio=float(merged["roll_lever_ratio"]),
            provenance=prov,
            source=str(node.get("source", "")),
            assumptions=tuple(str(a) for a in (node.get("assumptions")
                                               or DESIGN_ASSUMPTIONS)),
        )

    # -- geometry ------------------------------------------------------------

    @property
    def fin_area_ratio(self) -> float:
        """Total fin planform over the vehicle reference area."""
        return self.count * self.fin_area_m2 / self.reference_area_m2

    @property
    def alpha_authority_deg(self) -> float:
        """Commandable alpha ceiling from the deflection saturation."""
        return min(self.alpha_max_deg, self.trim_effectiveness * self.delta_max_deg)

    # -- aerodynamic surfaces --------------------------------------------------

    def cd_add(self, mach: float) -> float:
        """Zero-alpha deployed drag increment on the vehicle Cd (S_ref)."""
        pts = np.asarray(self.cd_add_mach, dtype=np.float64)
        return float(np.interp(mach, pts[:, 0], pts[:, 1]))

    def fin_cn_alpha(self, mach: float) -> float:
        """Deployed fin normal-force slope, VEHICLE-area referred [1/rad]."""
        pts = np.asarray(self.cn_alpha_mach, dtype=np.float64)
        return float(np.interp(mach, pts[:, 0], pts[:, 1])) * self.fin_area_ratio

    def trim_delta_deg(self, alpha_deg: float) -> float:
        """Trimming fin deflection [deg] holding ``alpha_deg`` (saturation is
        the actuator's job)."""
        return alpha_deg / self.trim_effectiveness

    def alpha_from_delta(self, delta_deg: float) -> float:
        """Trimmed alpha reachable with deflection ``delta_deg``."""
        return delta_deg * self.trim_effectiveness

    def _cn_terms(self, alpha_rad: float, mach: float) -> tuple[float, float]:
        """(body, fin) normal-force coefficients at the TRIMMED fin deflection,
        fin term clipped at its local stall."""
        body = self.body_cn_alpha_per_rad * math.sin(alpha_rad)
        a_fin = abs(alpha_rad) * (1.0 - 1.0 / self.trim_effectiveness)
        clipped = min(a_fin, self.stall_alpha_deg * DEG)
        fin = self.fin_cn_alpha(mach) * math.sin(clipped) * (1.0 if alpha_rad >= 0 else -1.0)
        return body, fin

    def cl(self, alpha_deg: float, mach: float) -> float:
        """Trimmed deployed lift coefficient (S_ref): total normal force times
        cos(alpha) -- the point-mass "lift perpendicular to velocity" reading
        of a weathervane-trimmed booster."""
        a = min(alpha_deg, self.alpha_authority_deg) * DEG
        body, fin = self._cn_terms(a, mach)
        return (body + fin) * math.cos(a)

    def cd_increment(self, alpha_deg: float, mach: float) -> float:
        """Deployed drag increment over the stowed vehicle Cd: profile plus
        the normal-force turn plus low-aspect-ratio fin induced drag."""
        a = min(alpha_deg, self.alpha_authority_deg) * DEG
        body, fin = self._cn_terms(a, mach)
        cn = body + fin
        return self.cd_add(mach) + cn * math.sin(a) + self.induced_k * fin * fin

    # -- moments (authority / test surface; the point mass flies the forces) ----

    def cm_alpha(self, mach: float) -> float:
        """Pitch-moment-curve slope [1/rad, q*S*L referred], negative =
        restoring (weathervane)."""
        return -self.static_margin * (self.body_cn_alpha_per_rad + self.fin_cn_alpha(mach))

    def cm_delta(self, mach: float) -> float:
        """Nose-up pitch moment per radian of control deflection: by the trim
        law, ``g`` times as effective as alpha itself."""
        return self.trim_effectiveness * (-self.cm_alpha(mach))

    def pitch_moment_nm(self, alpha_deg: float, mach: float, q_pa: float,
                        delta_ctrl_deg: float = 0.0) -> float:
        """Pitch moment [N m], positive nose-up, at dynamic pressure ``q_pa``
        with fin deflection ``delta_ctrl_deg`` (positive nose-up); the moment
        vanishes at ``(alpha, trim_delta(alpha))`` by the trim law."""
        cm = (self.cm_alpha(mach) * alpha_deg * DEG
              + self.cm_delta(mach) * delta_ctrl_deg * DEG)
        return cm * q_pa * self.reference_area_m2 * self.length_m

    def roll_moment_nm(self, delta_aileron_deg: float, mach: float,
                       q_pa: float) -> float:
        """Roll moment [N m] for a fin-pair differential ``delta_aileron_deg``
        (positive = bank right)."""
        cl_delta = self.fin_cn_alpha(mach) * self.roll_lever_ratio
        return cl_delta * delta_aileron_deg * DEG * q_pa \
            * self.reference_area_m2 * self.diameter_m


@dataclass(frozen=True)
class GridFinCommand:
    """One steering sample: commanded angle of attack and bank, degrees."""

    alpha_cmd_deg: float
    bank_cmd_deg: float


@dataclass(frozen=True)
class GridFinState:
    """Actuator snapshot flown over one integration step (what the track
    records)."""

    deployed: bool
    alpha_deg: float           # realised (authority-limited) alpha
    bank_deg: float            # realised bank
    fin_delta_deg: float       # mean |deflection| over the fins, degrees
    cl: float
    cd_increment: float


STOWED = GridFinState(False, 0.0, 0.0, 0.0, 0.0, 0.0)


def _slew(cur: float, target: float, max_step: float) -> float:
    d = target - cur
    return cur + max(min(d, max_step), -max_step)


def _slew_arr(cur: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
    return cur + np.clip(target - cur, -max_step, max_step)


class GridFinActuator:
    """Deployed/stowed two-state actuator with deflection slew/saturation and
    bank-rate limiting. One :meth:`step` per integrator step (piecewise
    constant like the engine command); the returned snapshot drives the force
    evaluation for the whole step. Roll deflections alternate over the fin
    stations (an X-configuration: opposite pairs move together, so two
    fins saturate first under combined trim + roll demand)."""

    def __init__(self, model: GridFinModel, *, deployed: bool = False) -> None:
        self.model = model
        self.deployed = bool(deployed)
        self.bank_deg = 0.0
        self.alpha_deg = 0.0
        self.delta_deg = np.zeros(model.count, dtype=np.float64)
        self._prev_snap = STOWED

    @property
    def snapshot(self) -> GridFinState:
        return self._prev_snap

    def step(self, dt_s: float, cmd: GridFinCommand | None, *,
             height_m: float, dynamic_pressure_pa: float,
             mach: float) -> GridFinState:
        """Advance one ``dt_s`` sample towards ``cmd`` (``None`` = hold alpha
        at 0, bank to 0). Deploy below the deploy altitude with enough q;
        stow below the stow altitude -- the altitude gap is the hysteresis."""
        m = self.model
        dt = float(dt_s)
        if not self.deployed and height_m <= m.deploy_altitude_m \
                and dynamic_pressure_pa >= m.deploy_q_min_pa:
            self.deployed = True
        if self.deployed and height_m <= m.stow_altitude_m:
            self.deployed = False
        if not self.deployed:
            self.bank_deg = _slew(self.bank_deg, 0.0, m.bank_rate_max_deg_s * dt)
            self.alpha_deg = 0.0
            self.delta_deg = _slew_arr(self.delta_deg, np.zeros_like(self.delta_deg),
                                       m.delta_rate_max_deg_s * dt)
            snap = GridFinState(False, 0.0, float(self.bank_deg),
                                float(np.mean(np.abs(self.delta_deg))), 0.0, 0.0)
            self._prev_snap = snap
            return snap

        alpha_cmd = 0.0 if cmd is None else min(max(float(cmd.alpha_cmd_deg), 0.0),
                                               m.alpha_authority_deg)
        self.alpha_deg = alpha_cmd
        delta_trim = m.trim_delta_deg(alpha_cmd)     # alpha within authority => |d|<=dmax
        roll_cmd = 0.0 if cmd is None else max(min(float(cmd.bank_cmd_deg),
                                                   m.bank_max_deg), -m.bank_max_deg)
        self.bank_deg = _slew(self.bank_deg, roll_cmd, m.bank_rate_max_deg_s * dt)

        # roll differential scales with realised bank and uses the deflection
        # margin trim leaves; every fin stays saturated within +-delta_max
        da = (self.bank_deg / m.bank_max_deg) * (m.delta_max_deg - abs(delta_trim))
        sign = (1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0)[:m.count]
        target = np.clip(delta_trim + np.array(sign) * da, -m.delta_max_deg,
                         m.delta_max_deg)
        self.delta_deg = _slew_arr(self.delta_deg, target, m.delta_rate_max_deg_s * dt)

        snap = GridFinState(True, float(alpha_cmd), float(self.bank_deg),
                            float(np.mean(np.abs(self.delta_deg))),
                            m.cl(alpha_cmd, mach), m.cd_increment(alpha_cmd, mach))
        self._prev_snap = snap
        return snap


@dataclass(frozen=True)
class GridFinGuidance:
    """The shared steering law (E10-07 rule: guidance is part of the authored
    script, identical in the main model and the independent twin).

    ``mode="steer"``: constant alpha inside the fin window (deploy altitude
    down to ``alpha_floor_altitude_m``, where alpha ramps to 0 so the final
    approach is ballistic and the corridor tilt is taken near-vertical) and a
    bank command proportional to the heading error to the pad, saturating at
    ``bank_max_deg``.
    ``mode="schedule"``: open-loop keyed on the last altitude gate crossed
    from ``schedule=((h_m, bank_deg[, alpha_deg]), ...)`` -- gates DESCENDING,
    the optional alpha entry lets the schedule fly a glide phase and a
    terminal alpha-0 dive phase (E11-01 envelope work); ``alpha_deg`` is the
    gate default.
    """

    model: GridFinModel
    pad_lat_deg: float = 0.0
    pad_lon_deg: float = 0.0
    alpha_deg: float = 30.0
    bank_max_deg: float = 60.0
    mode: str = "steer"
    gain_per_deg: float = 6.0
    alpha_floor_altitude_m: float = 8000.0
    schedule: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in ("steer", "schedule"):
            raise ValueError(f"unknown grid-fin guidance mode {self.mode!r}")
        if self.mode == "schedule" and not self.schedule:
            raise ValueError("schedule guidance needs a bank schedule")
        if not 0.0 <= self.alpha_deg <= self.model.alpha_authority_deg:
            raise ValueError(
                f"alpha_deg {self.alpha_deg} outside fin authority "
                f"[0, {self.model.alpha_authority_deg}] deg")
        if abs(self.bank_max_deg) > self.model.bank_max_deg:
            raise ValueError("guidance bank_max beyond the actuator limit")
        if self.mode == "schedule":
            norm = []
            for g in self.schedule:
                g = tuple(float(x) for x in g)
                if len(g) not in (2, 3):
                    raise ValueError("schedule gates are (h, bank[, alpha])")
                if len(g) == 3 and not 0.0 <= g[2] <= self.model.alpha_authority_deg:
                    raise ValueError(
                        f"gate alpha {g[2]} outside fin authority "
                        f"[0, {self.model.alpha_authority_deg}] deg")
                if abs(g[1]) > self.bank_max_deg:
                    raise ValueError("schedule bank beyond the guidance limit")
                norm.append(g)
            if any(g[0] >= p[0] for p, g in zip(norm, norm[1:])):
                raise ValueError("schedule gates must DESCEND in altitude")
            object.__setattr__(self, "schedule", tuple(
                g if len(g) == 3 else (g[0], g[1], self.alpha_deg) for g in norm))

    def command(self, height_m: float, pos_ecef_m: Sequence[float],
                vel_ecef_m_s: Sequence[float]) -> GridFinCommand | None:
        """One guidance sample (``None`` = no command, fins glide stowed)."""
        if self.mode == "schedule":
            if height_m > self.model.deploy_altitude_m or height_m < \
                    self.model.stow_altitude_m:
                return None
            bank, alpha = 0.0, self.alpha_deg
            for h_gate, b, a in self.schedule:   # gates in DESCENDING order
                if height_m <= h_gate:
                    bank, alpha = b, a
            return GridFinCommand(alpha, bank)
        if height_m > self.model.deploy_altitude_m or height_m < \
                self.alpha_floor_altitude_m:
            return None
        err = self.heading_error_deg(pos_ecef_m, vel_ecef_m_s)
        bank = max(min(self.gain_per_deg * err, self.bank_max_deg), -self.bank_max_deg)
        return GridFinCommand(self.alpha_deg, float(bank))

    def heading_error_deg(self, pos_ecef_m: Sequence[float],
                          vel_ecef_m_s: Sequence[float]) -> float:
        """Signed heading error to the pad [deg], positive = the pad lies to
        the RIGHT of the velocity (bank-right command sense)."""
        r = np.asarray(pos_ecef_m, dtype=np.float64)
        v = np.asarray(vel_ecef_m_s, dtype=np.float64)
        lat, lon, _h = frames.ecef_to_geodetic(r)
        east, north, _up = _enu_basis(lat, lon)
        ve = float(v @ east)
        vn = float(v @ north)
        if math.hypot(ve, vn) < 1.0:
            return 0.0
        heading = math.atan2(ve, vn)
        # pad position on the local level surface at the vehicle's radius
        pr = _geodetic_to_ecef(self.pad_lat_deg, self.pad_lon_deg, _h)
        dr = pr - r
        bearing = math.atan2(float(dr @ east), float(dr @ north))
        err = (bearing - heading) / DEG
        return (err + 180.0) % 360.0 - 180.0


def _enu_basis(lat_deg: float, lon_deg: float):
    la, lo = math.radians(lat_deg), math.radians(lon_deg)
    sa, ca, so, co = math.sin(la), math.cos(la), math.sin(lo), math.cos(lo)
    return (np.array((-so, co, 0.0)),
            np.array((-sa * co, -sa * so, ca)),
            np.array((ca * co, ca * so, sa)))


def _geodetic_to_ecef(lat_deg: float, lon_deg: float, h_m: float) -> np.ndarray:
    la, lo = math.radians(lat_deg), math.radians(lon_deg)
    e2 = frames.WGS84_F * (2.0 - frames.WGS84_F)
    n_rm = frames.WGS84_A / math.sqrt(1.0 - e2 * math.sin(la) ** 2)
    cl = math.cos(la)
    return np.array(((n_rm + h_m) * cl * math.cos(lo),
                     (n_rm + h_m) * cl * math.sin(lo),
                     (n_rm * (1.0 - e2) + h_m) * math.sin(la)))


#: sign of each fin's roll differential for the X configuration (pairs move
#: together); exposed for the unit tests
X_CONFIG_SIGN = (1.0, -1.0, -1.0, 1.0)
