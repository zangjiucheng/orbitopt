"""JSBSim 6-DOF spike driver (FL-401, task #206).

Loads the minimal liquid stage ``spike_stage`` (see
``orbitopt/data/flightlab/jsbsim/aircraft/spike_stage/spike_stage.xml``),
sets geodetic/attitude initial conditions, steps the simulation with
throttle and TVC gimbal commands, samples JSBSim properties in SI units and
packs them into a flightlab-contract :class:`~orbitopt.flightlab.contract.BodyTrack`
(frame ``ECEF``) with extra columns for attitude, body rates, gimbal command
vs. actual deflection and thrust.

Conventions
-----------
* Positions/velocities: JSBSim ECEF [m, m/s] (Earth-fixed, so a vehicle at
  rest on the rotating Earth has zero velocity).
* Attitude quaternion ``q_w, q_x, q_y, q_z`` (scalar first): ``R(q)`` is
  JSBSim's ``Tec2b`` direction-cosine matrix (v_body = R(q) v_ecef), so it
  stays well defined for a vertical vehicle (no Euler gimbal lock).
* Body rates ``p/q/r_rad_s``: JSBSim body rates relative to ECEF, body axes
  x forward, y right, z down.
* Gimbal: ``fcs/{pitch,yaw}-gimbal-cmd-rad`` is what the driver commands;
  ``fcs/{pitch,yaw}-gimbal-actual-rad`` is the servo output after lag,
  slew-rate and travel limits. Positive pitch == nose-up moment, positive
  yaw == nose-right moment (the model's FCS maps this onto JSBSim's
  thruster angle sign).

``jsbsim`` and ``numpy`` are imported lazily, so importing this module (and
its constants) works on a base install; constructing :class:`SixDofSim`
without jsbsim raises :class:`ImportError` with an install hint.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Callable

from orbitopt.flightlab.contract import BodyTrack

# --------------------------------------------------------------------------
# Units and the spike model's declared parameters (mirrors the XML; the tests
# check the running model against these numbers).
# --------------------------------------------------------------------------

FT_M = 0.3048
LBF_N = 4.4482216152605
SLUG_KG = 14.593902937206364
SLUGFT2_KGM2 = SLUG_KG * FT_M**2
G0_M_S2 = 9.80665

MODEL_NAME = "spike_stage"
#: Spike stage parameters as declared in the XML (SI).
SPIKE_STAGE = {
    "dry_mass_kg": 10_000.0,
    "propellant_mass_kg": 60_000.0,
    "isp_s": 320.0,
    "mdot_max_kg_s": 600.0,
    "mixture_ratio": 2.5,
    "gimbal_lag_tau_s": 0.2,  # actuator <lag> 5 rad/s
    "gimbal_rate_limit_rad_s": 0.35,
    "gimbal_limit_rad": 0.0873,
}
SPIKE_STAGE["thrust_vac_n"] = SPIKE_STAGE["isp_s"] * G0_M_S2 * SPIKE_STAGE["mdot_max_kg_s"]

#: Extra BodyTrack columns produced by :meth:`SixDofSim.to_body_track`, in order.
EXTRA_COLUMNS = (
    "q_w", "q_x", "q_y", "q_z",
    "p_rad_s", "q_rad_s", "r_rad_s",
    "gimbal_pitch_cmd_rad", "gimbal_yaw_cmd_rad",
    "gimbal_pitch_actual_rad", "gimbal_yaw_actual_rad",
    "thrust_n", "throttle",
    "ixx_kg_m2", "iyy_kg_m2", "izz_kg_m2", "cg_x_m",
    "altitude_m", "pitch_moment_n_m", "yaw_moment_n_m",
)


def data_root() -> Path:
    """The JSBSim root directory shipped with orbitopt (``aircraft/``,
    ``engine/`` under it)."""
    return Path(str(resources.files("orbitopt") / "data" / "flightlab" / "jsbsim"))


def _import_jsbsim():
    try:
        import jsbsim  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only on base installs
        raise ImportError(
            "the Flight Lab 6-DOF spike needs jsbsim==1.3.3: "
            "pip install -r envs/sixdof-requirements.txt (or orbitopt[sixdof])"
        ) from exc
    return jsbsim


def quat_from_dcm(m) -> tuple[float, float, float, float]:
    """Scalar-first unit quaternion ``q`` with ``R(q) == m`` for a 3x3
    rotation matrix ``m`` (Shepperd's method; sign fixed to q_w >= 0). For
    ``m = Tec2b`` this is the ECEF->body transform: v_body = R(q) v_ecef."""
    m00, m01, m02 = (float(m[0, j]) for j in range(3))
    m10, m11, m12 = (float(m[1, j]) for j in range(3))
    m20, m21, m22 = (float(m[2, j]) for j in range(3))
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = 2.0 * math.sqrt(1.0 + tr)
        w, x, y, z = 0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
        w, x, y, z = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
        w, x, y, z = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
        w, x, y, z = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return (w / n, x / n, y / n, z / n)


@dataclass(frozen=True)
class InitialConditions:
    """Geodetic position + attitude (+ optional NED velocity) at t=0."""

    lat_geod_deg: float = 0.0
    lon_deg: float = 0.0
    alt_m: float = 0.0  # above sea level
    pitch_deg: float = 90.0  # theta; 90 = nose straight up
    heading_deg: float = 90.0  # psi (true)
    roll_deg: float = 0.0  # phi
    v_north_m_s: float = 0.0
    v_east_m_s: float = 0.0
    v_down_m_s: float = 0.0


class SixDofSim:
    """Thin stateful wrapper around one ``jsbsim.FGFDMExec`` instance."""

    def __init__(self, dt_s: float = 0.01, *, model: str = MODEL_NAME, root: Path | str | None = None):
        jsbsim = _import_jsbsim()
        self.dt_s = float(dt_s)
        self.fdm = jsbsim.FGFDMExec(str(root or data_root()))
        self.fdm.set_debug_level(0)
        # set_dt BEFORE load_model: FGFCSComponents (actuator lag/rate
        # filters) cache the executive's dt when they are constructed, so a
        # later set_dt silently leaves the FCS running at JSBSim's default
        # 1/120 s while the dynamics use the new step.
        self.fdm.set_dt(self.dt_s)
        if not self.fdm.load_model(model):
            raise RuntimeError(f"JSBSim failed to load model {model!r} from {root or data_root()}")
        self._samples: list[dict[str, float]] = []

    # -- properties -------------------------------------------------------
    def __getitem__(self, prop: str) -> float:
        return float(self.fdm[prop])

    def __setitem__(self, prop: str, value: float) -> None:
        self.fdm[prop] = float(value)

    # -- setup / commands -------------------------------------------------
    def initialize(self, ic: InitialConditions) -> None:
        f = self.fdm
        f["ic/lat-geod-deg"] = ic.lat_geod_deg
        f["ic/long-gc-deg"] = ic.lon_deg
        f["ic/h-sl-ft"] = ic.alt_m / FT_M
        f["ic/theta-deg"] = ic.pitch_deg
        f["ic/psi-true-deg"] = ic.heading_deg
        f["ic/phi-deg"] = ic.roll_deg
        f["ic/vn-fps"] = ic.v_north_m_s / FT_M
        f["ic/ve-fps"] = ic.v_east_m_s / FT_M
        f["ic/vd-fps"] = ic.v_down_m_s / FT_M
        f["ic/p-rad_sec"] = 0.0
        f["ic/q-rad_sec"] = 0.0
        f["ic/r-rad_sec"] = 0.0
        if not f.run_ic():
            raise RuntimeError("JSBSim run_ic() failed")
        self._samples = [self.sample()]

    def ignite(self, throttle: float = 1.0) -> None:
        self.fdm["propulsion/engine[0]/set-running"] = 1
        self.set_throttle(throttle)

    def set_throttle(self, throttle: float) -> None:
        self.fdm["fcs/throttle-cmd-norm[0]"] = throttle

    def set_gimbal(self, pitch_rad: float = 0.0, yaw_rad: float = 0.0) -> None:
        self.fdm["fcs/pitch-gimbal-cmd-rad"] = pitch_rad
        self.fdm["fcs/yaw-gimbal-cmd-rad"] = yaw_rad

    @property
    def t_s(self) -> float:
        return float(self.fdm["simulation/sim-time-sec"])

    # -- stepping ---------------------------------------------------------
    def step(self) -> dict[str, float]:
        if not self.fdm.run():
            raise RuntimeError(f"JSBSim run() stopped at t={self.t_s:.3f} s")
        s = self.sample()
        self._samples.append(s)
        return s

    def run_for(self, duration_s: float, control: Callable[[float, "SixDofSim"], None] | None = None) -> None:
        """Step until ``duration_s`` past the current time; ``control(t, sim)``
        is called before every step (set throttle/gimbal there)."""
        n = round(duration_s / self.dt_s)
        for _ in range(n):
            if control is not None:
                control(self.t_s, self)
            self.step()

    # -- sampling ---------------------------------------------------------
    def sample(self) -> dict[str, float]:
        f = self.fdm
        g = lambda p: float(f[p])  # noqa: E731
        with warnings.catch_warnings():
            # jsbsim's Python binding returns numpy.matrix (deprecated class).
            warnings.simplefilter("ignore", PendingDeprecationWarning)
            q = quat_from_dcm(f.get_propagate().get_Tec2b())
        return {
            "t_s": g("simulation/sim-time-sec"),
            "mass_kg": g("inertia/mass-slugs") * SLUG_KG,
            "x_m": g("position/ecef-x-ft") * FT_M,
            "y_m": g("position/ecef-y-ft") * FT_M,
            "z_m": g("position/ecef-z-ft") * FT_M,
            "vx_m_s": g("velocities/ecef-x-fps") * FT_M,
            "vy_m_s": g("velocities/ecef-y-fps") * FT_M,
            "vz_m_s": g("velocities/ecef-z-fps") * FT_M,
            "q_w": q[0], "q_x": q[1], "q_y": q[2], "q_z": q[3],
            "p_rad_s": g("velocities/p-rad_sec"),
            "q_rad_s": g("velocities/q-rad_sec"),
            "r_rad_s": g("velocities/r-rad_sec"),
            "gimbal_pitch_cmd_rad": g("fcs/pitch-gimbal-cmd-rad"),
            "gimbal_yaw_cmd_rad": g("fcs/yaw-gimbal-cmd-rad"),
            "gimbal_pitch_actual_rad": g("fcs/pitch-gimbal-actual-rad"),
            "gimbal_yaw_actual_rad": g("fcs/yaw-gimbal-actual-rad"),
            "thrust_n": g("propulsion/engine[0]/thrust-lbs") * LBF_N,
            "throttle": g("fcs/throttle-pos-norm[0]"),
            "ixx_kg_m2": g("inertia/ixx-slugs_ft2") * SLUGFT2_KGM2,
            "iyy_kg_m2": g("inertia/iyy-slugs_ft2") * SLUGFT2_KGM2,
            "izz_kg_m2": g("inertia/izz-slugs_ft2") * SLUGFT2_KGM2,
            # structural frame: x aft from the nose
            "cg_x_m": g("inertia/cg-x-in") * FT_M / 12.0,
            "altitude_m": g("position/h-sl-ft") * FT_M,
            "pitch_moment_n_m": g("moments/m-prop-lbsft") * LBF_N * FT_M,
            "yaw_moment_n_m": g("moments/n-prop-lbsft") * LBF_N * FT_M,
            # diagnostics (not exported as columns)
            "v_down_m_s": g("velocities/v-down-fps") * FT_M,
            "gravity_m_s2": g("accelerations/gravity-ft_sec2") * FT_M,
            "theta_deg": g("attitude/theta-deg"),
        }

    @property
    def samples(self) -> list[dict[str, float]]:
        return list(self._samples)

    def to_body_track(self, *, body_id: str = "spike_stage", epoch_utc: str = "2026-01-01T00:00:00Z",
                      every: int = 1) -> BodyTrack:
        """The samples so far as a contract ``BodyTrack`` (ECEF, provenance
        ``illustrative``), keeping every ``every``-th sample (+ the last)."""
        rows = self._samples[::max(1, int(every))]
        if rows[-1] is not self._samples[-1]:
            rows.append(self._samples[-1])
        return BodyTrack(
            body_id=body_id,
            parent_body_id=None,
            frame="ECEF",
            epoch_utc=epoch_utc,
            provenance="illustrative",
            t_s=tuple(r["t_s"] for r in rows),
            mass_kg=tuple(r["mass_kg"] for r in rows),
            position_m=tuple((r["x_m"], r["y_m"], r["z_m"]) for r in rows),
            velocity_m_s=tuple((r["vx_m_s"], r["vy_m_s"], r["vz_m_s"]) for r in rows),
            extra_columns={c: tuple(r[c] for r in rows) for c in EXTRA_COLUMNS},
        )
