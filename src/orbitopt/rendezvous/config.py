"""Two-impulse rendezvous mission configs (E12-03).

The data contract behind :mod:`orbitopt.rendezvous`: the reference mission
LEO-RDV-001 and anything a user adapts from it. Everything the planner must
not move during optimization -- the delta-v budget, the drift and transfer
time windows, the departure gate, the arrival tolerances and, pointedly, the
Earth gravitational parameter -- is declared here and fixed before
:func:`orbitopt.rendezvous.plan_rendezvous` ever runs.

Loading mirrors :mod:`orbitopt.mission_config`'s split: a document is
validated against the packaged schema
``orbitopt/schemas/rendezvous-mission-1.0.json``, then re-validated by the
cross-checks the schema cannot express (``mu`` equals the exact packaged
Earth value, gate/hold ordering, the coplanar phasing is geometrically
solvable, ...). ``mu`` is checked with absolute equality on purpose: an
Earth-orbit rendezvous config that carries anything else -- e.g. the solar
value ``solve_lambert_single`` defaults to -- is a units/central-body
mistake, and the cheapest place to catch it is the file that declares it.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import jsonschema
import yaml

from orbitopt.physics_io import KNOWN_MU, RELATIVE_PAIR

#: Which config-schema version :func:`parse_rendezvous_config` accepts.
SCHEMA_VERSION = "1.0"

#: mission.kind this loader accepts (and the schema pins).
KIND = "rendezvous-two-impulse"

_SCHEMA_FILENAME = "rendezvous-mission-1.0.json"
_REFERENCE_DATA = "leo_rdv_001.yaml"


class RendezvousConfigError(ValueError):
    """A rendezvous mission config is malformed or self-contradictory."""


@dataclass(frozen=True)
class DynamicsCfg:
    frame: str
    mu: float
    j2: bool


@dataclass(frozen=True)
class TargetCfg:
    altitude_m: float
    inclination_deg: float
    raan_deg: float
    argument_of_latitude_deg: float


@dataclass(frozen=True)
class ChaserCfg:
    altitude_m: float
    initial_separation_m: float


@dataclass(frozen=True)
class WindowsCfg:
    drift_max_s: float
    transfer_tof_min_s: float
    transfer_tof_max_s: float


@dataclass(frozen=True)
class PhasingCfg:
    departure_gate_y_m: float
    gate_scan_step_s: float


@dataclass(frozen=True)
class OptimizerCfg:
    newton_tol_m: float
    newton_max_iter: int
    jacobian_eps_m_s: float
    golden_max_iter: int


@dataclass(frozen=True)
class SafetyCfg:
    """E12-04 hard safety constraints, fixed in the mission config before
    optimization -- the planner treats a violation as infeasibility and
    never relaxes a bound at runtime. Checked on the whole trajectory
    (drift, transfer coast between burns, post-insertion hover) by
    :func:`orbitopt.rendezvous.verify_plan_safety`. ``speed_range_knots_m``
    / ``speed_cap_m_s`` give the range-dependent relative-speed ceiling as
    a piecewise-linear profile, constant below the first knot and above the
    last."""

    keep_out_radius_m: float
    cross_track_max_m: float
    cone_half_angle_deg: float
    cone_range_m: float
    speed_range_knots_m: tuple[float, ...]
    speed_cap_m_s: tuple[float, ...]
    passive_contingency_s: float
    check_sample_dt_s: float


@dataclass(frozen=True)
class RendezvousMissionConfig:
    """Parsed, cross-checked LEO-RDV-001-family mission config. Times
    throughout the package are elapsed seconds from the shared epoch t0 at
    which ``target``/``chaser`` place the pair; no wall-clock mixing (the
    E12-01 single-shared-epoch rule)."""

    mission_id: str
    title: str
    dynamics: DynamicsCfg
    target: TargetCfg
    chaser: ChaserCfg
    hold_point_rel_lvlh: tuple[float, float, float]
    delta_v_budget_m_s: float
    max_position_error_m: float
    max_relative_speed_m_s: float
    windows: WindowsCfg
    phasing: PhasingCfg
    optimizer: OptimizerCfg
    safety: SafetyCfg | None = None

    @property
    def orbital_plane_deg(self) -> tuple[float, float]:
        """(inclination, RAAN) of the common plane: v1 is coplanar, the
        chaser inherits the target's orbit normal and is phased only along
        the track."""
        return (self.target.inclination_deg, self.target.raan_deg)

    @property
    def along_track_offset_m(self) -> float:
        """Chaser's (negative, i.e. aft) along-track LVLH offset at t0 that
        realises ``chaser.initial_separation_m``: the radial gap is fixed by
        the altitude difference, the rest of the separation is along-track."""
        radial = self.chaser.altitude_m - self.target.altitude_m
        s = self.chaser.initial_separation_m
        return -math.sqrt(s * s - radial * radial)


def _schema() -> dict:
    return json.loads(
        (resources.files("orbitopt") / "schemas" / _SCHEMA_FILENAME).read_text(encoding="utf-8")
    )


def validate_rendezvous_config(doc: dict, *, where: str = "rendezvous config") -> None:
    """Schema-validate a parsed config document; raises
    :class:`RendezvousConfigError` on the first violation."""
    try:
        jsonschema.validate(instance=doc, schema=_schema())
    except jsonschema.ValidationError as exc:
        raise RendezvousConfigError(f"{where}: {exc.message}") from exc


def _parse_safety(doc: dict, hold: tuple[float, float, float], where: str) -> SafetyCfg:
    """Build :class:`SafetyCfg` from the optional v1.0 ``safety`` block,
    applying the cross-checks the schema cannot express (knot monotonicity,
    keep-out must exclude the Hold Point)."""
    s = doc["safety"]
    knots = tuple(float(v) for v in s["approach_speed"]["range_knots_m"])
    caps = tuple(float(v) for v in s["approach_speed"]["cap_m_s"])
    if len(knots) != len(caps):
        raise RendezvousConfigError(
            f"{where}: safety.approach_speed needs one cap per range knot, "
            f"got {len(knots)} knots and {len(caps)} caps"
        )
    if any(b <= a for a, b in zip(knots, knots[1:])):
        raise RendezvousConfigError(f"{where}: safety.approach_speed.range_knots_m must strictly increase")
    if any(b < a for a, b in zip(caps, caps[1:])):
        raise RendezvousConfigError(
            f"{where}: safety.approach_speed.cap_m_s must be non-decreasing with range "
            f"(a slower approach closer in is the whole idea)"
        )
    hold_range = math.sqrt(sum(v * v for v in hold))
    if s["keep_out_radius_m"] >= hold_range:
        raise RendezvousConfigError(
            f"{where}: safety.keep_out_radius_m {s['keep_out_radius_m']:.0f} m swallows the Hold Point "
            f"at |hold| = {hold_range:.0f} m -- a mission that forbids its own arrival point is "
            f"self-contradictory"
        )
    return SafetyCfg(
        keep_out_radius_m=float(s["keep_out_radius_m"]),
        cross_track_max_m=float(s["cross_track_max_m"]),
        cone_half_angle_deg=float(s["cone_half_angle_deg"]),
        cone_range_m=float(s["cone_range_m"]),
        speed_range_knots_m=knots,
        speed_cap_m_s=caps,
        passive_contingency_s=float(s["passive_contingency_s"]),
        check_sample_dt_s=float(s["check_sample_dt_s"]),
    )


def parse_rendezvous_config(doc: dict, *, where: str = "rendezvous config") -> RendezvousMissionConfig:
    """Schema-validate and cross-check a document, then build the config.
    Cross-checks (schema can't express these):

    * ``dynamics.mu`` equals ``physics_io.KNOWN_MU["earth"]`` exactly --
      this family is Earth-orbit rendezvous, and the explicit value is what
      keeps ``solve_lambert_single`` off its solar default;
    * the gate ordering ``initial along-track < gate < hold-point y < 0``,
      so the V-bar drift actually reaches the departure gate before the
      hold point;
    * ``|radial gap| < initial_separation`` so the phasing geometry solves;
    * with the optional E12-04 ``safety`` block: speed knots strictly
      increase, speed caps are non-decreasing with range, and the keep-out
      sphere must exclude the Hold Point.
    """
    validate_rendezvous_config(doc, where=where)
    dyn, tgt, chs = doc["dynamics"], doc["target"], doc["chaser"]
    hold = tuple(float(v) for v in doc["hold_point"]["rel_lvlh"])
    win, ph, opt = doc["windows"], doc["phasing"], doc["optimizer"]
    if dyn["frame"] not in RELATIVE_PAIR.frames:
        raise RendezvousConfigError(
            f"{where}: dynamics.frame {dyn['frame']!r} is not a frame {RELATIVE_PAIR} accepts"
        )
    if float(dyn["mu"]) != KNOWN_MU["earth"]:
        raise RendezvousConfigError(
            f"{where}: dynamics.mu must equal the exact packaged Earth value "
            f"{KNOWN_MU['earth']!r} m^3/s^2, got {dyn['mu']!r} -- central-body/unit mixups "
            f"between the Lambert solver's solar default and Earth orbits are the whole "
            f"point of this check"
        )
    if chs["altitude_m"] >= tgt["altitude_m"]:
        raise RendezvousConfigError(
            f"{where}: chaser.altitude_m must be below target.altitude_m so the chaser "
            f"drifts FORWARD along V-bar toward the gate (lower orbit, higher mean motion)"
        )
    cfg = RendezvousMissionConfig(
        mission_id=doc["mission"]["id"],
        title=doc["mission"]["title"],
        dynamics=DynamicsCfg(frame=dyn["frame"], mu=float(dyn["mu"]), j2=bool(dyn["j2"])),
        target=TargetCfg(
            altitude_m=float(tgt["altitude_m"]),
            inclination_deg=float(tgt["inclination_deg"]),
            raan_deg=float(tgt["raan_deg"]),
            argument_of_latitude_deg=float(tgt["argument_of_latitude_deg"]),
        ),
        chaser=ChaserCfg(
            altitude_m=float(chs["altitude_m"]), initial_separation_m=float(chs["initial_separation_m"])
        ),
        hold_point_rel_lvlh=hold,
        delta_v_budget_m_s=float(doc["budget"]["delta_v_budget_m_s"]),
        max_position_error_m=float(doc["tolerances"]["max_position_error_m"]),
        max_relative_speed_m_s=float(doc["tolerances"]["max_relative_speed_m_s"]),
        windows=WindowsCfg(
            drift_max_s=float(win["drift_max_s"]),
            transfer_tof_min_s=float(win["transfer_tof_min_s"]),
            transfer_tof_max_s=float(win["transfer_tof_max_s"]),
        ),
        phasing=PhasingCfg(
            departure_gate_y_m=float(ph["departure_gate_y_m"]),
            gate_scan_step_s=float(ph["gate_scan_step_s"]),
        ),
        optimizer=OptimizerCfg(
            newton_tol_m=float(opt["newton_tol_m"]),
            newton_max_iter=int(opt["newton_max_iter"]),
            jacobian_eps_m_s=float(opt["jacobian_eps_m_s"]),
            golden_max_iter=int(opt["golden_max_iter"]),
        ),
        safety=_parse_safety(doc, hold, where) if "safety" in doc else None,
    )
    if cfg.windows.transfer_tof_max_s <= cfg.windows.transfer_tof_min_s:
        raise RendezvousConfigError(f"{where}: transfer_tof_max_s must exceed transfer_tof_min_s")
    if cfg.phasing.gate_scan_step_s >= cfg.windows.drift_max_s:
        raise RendezvousConfigError(f"{where}: phasing.gate_scan_step_s must be below windows.drift_max_s")
    y0 = cfg.along_track_offset_m
    gate, hold_y = cfg.phasing.departure_gate_y_m, cfg.hold_point_rel_lvlh[1]
    if not (y0 < gate < hold_y < 0.0):
        raise RendezvousConfigError(
            f"{where}: need initial along-track {y0:.0f} < gate {gate:.0f} < hold y {hold_y:.0f} < 0 "
            f"so the V-bar drift reaches the gate, then the transfer shortens to the hold point"
        )
    return cfg


def load_rendezvous_config(path: str | Path) -> RendezvousMissionConfig:
    """Load a rendezvous mission config from a YAML/JSON file."""
    text = Path(path).read_text(encoding="utf-8")
    doc = yaml.safe_load(text)
    if not isinstance(doc, dict):
        raise RendezvousConfigError(f"{path}: expected a mapping at the top level")
    return parse_rendezvous_config(doc, where=str(path))


def load_reference_config() -> RendezvousMissionConfig:
    """The packaged LEO-RDV-001 reference mission (``orbitopt/data/leo_rdv_001.yaml``)."""
    text = (resources.files("orbitopt") / "data" / _REFERENCE_DATA).read_text(encoding="utf-8")
    doc = yaml.safe_load(text)
    return parse_rendezvous_config(doc, where=_REFERENCE_DATA)
