"""Formal Flight Lab 6-DOF JSBSim backend (F-01, task #226): a config-driven
ascent adapter built on the E4 spike drivers (FL-401 #206, FL-402 #207,
FL-403 #208), selectable as ``backend="6DOF"`` next to the 3-DOF
OpenTsiolkovsky lineage.

What this refactor changes over the spike drivers: every input they took as
Python arguments -- vehicle (JSBSim aircraft model), initial conditions,
integration step, ignition, attitude program / TVC gains, staging split --
now comes from a **config document**, so switching vehicle or ICs is an
edit of YAML, never a code change::

    from pathlib import Path
    from orbitopt.flightlab.adapters.sixdof import run_ascent
    run = run_ascent(Path("sixdof_scenario.yaml"), Path("runs/jsb6-0001"))

or through the Gate-B pipeline: ``orbitopt flight simulate <target>
--vehicle <sixdof-config.yaml> --backend 6DOF -o runs/<id>`` (the packaged
spike scenarios resolve through :func:`packaged_config`).

Config document (SI units; required keys ``name``, ``model``; unknown
``ic`` keys or out-of-range values fail validation, naming the key)::

    name: fl401-vertical-hop          # run label
    provenance: illustrative          # contract FLIGHT_PROVENANCE_TAGS
    model: spike_stage                # vehicle: aircraft dir under the JSBSim root
    root: null                        # optional JSBSim root override
    dt_s: 0.01                        # integration step
    duration_s: 30.0                  # hold-mode flight duration (pid derives it)
    epoch_utc: '2026-01-01T00:00:00Z'  # mission epoch recorded in the manifest
    ic: {lat_geod_deg: 0.0, lon_deg: 0.0, alt_m: 300000.0,
         pitch_deg: 90.0, heading_deg: 90.0, roll_deg: 0.0,
         v_north_m_s: 0.0, v_east_m_s: 0.0, v_down_m_s: 0.0}
    ignition: {t_s: 0.0, throttle: 1.0}
    control: {mode: hold, gimbal_pitch_rad: 0.0, gimbal_yaw_rad: 0.0}
    # or closed-loop TVC on an attitude program (the FL-403 lineage):
    # control: {mode: pid, attitude_program: [{t_s, azimuth_deg, elevation_deg}, ...],
    #           meco_t_s: 100.0, coast_s: 5.0, gains: {kp: 3.0, ki: 0.01, kd: 4.0}}
    staging:                          # optional two-body handoff (FL-402 lineage)
      at_t_s: 10.0
      coast_s: 61.0
      separation_impulse_ns: 200000.0
      booster_dry_fraction: 0.4
      booster_propellant_fraction: 0.4
      booster_model: spike_booster
      upper_model: spike_upper
      step_order: ab
    reference_csv: tests/fixtures/.../trajectory.csv  # pid: 3-DOF reference (optional;
    #     relative paths resolve against the CWD; a missing file = MECO-deviation
    #     checks reported as not-assessed)
    tracking: {settled_from_s: 60.0, settled_max_deg: 2.0, settled_rms_deg: 0.6,
               full_max_deg: 2.0, gimbal_sat_fraction_max: 0.0,
               d_alt_m: 500.0, d_speed_m_s: 20.0, d_fpa_deg: 0.2}
    mass_model:                         # optional F-03 declared mass properties
      full: {mass_kg: 70000.0, cg_m: [18.2857143, 0.0, 0.0],
             inertia_kg_m2: {ixx: 111760.0, iyy: 4212000.0, izz: 4212000.0}}
      empty: {mass_kg: 10000.0, cg_m: [20.0, 0.0, 0.0],
              inertia_kg_m2: {ixx: 3.4e4, iyy: 1.7e6, izz: 1.7e6}}
      tolerances: {flow_lag_frames: 2.0, mass_abs_kg: 2.0, cg_abs_m: 0.02,
                   inertia_rel: 0.002, interpolation_cg_abs_m: 0.05,
                   interpolation_inertia_rel: 0.02, burnout_fraction: 0.001}
    output: {every: 1}
    assumptions: [ ... ]

Modes and what they fly:

* ``control.mode: hold`` + no ``staging`` -- one body on a fixed gimbal
  command (the FL-401 driver path).
* ``control.mode: pid`` -- PID TVC tracking the attitude program to MECO
  (throttle cut exactly at ``meco_t_s``), then coasting ``coast_s``. The
  controller, error metric and metric definitions are the FL-403 module's
  (:func:`orbitopt.flightlab.sixdof.tracking.chart_error_deg`,
  ``_metrics``); the GO bounds come from the config's ``tracking``
  section, and the ``reference_csv`` MECO deviations are checked when that
  file exists.
* ``staging`` (hold mode only) -- fly to ``at_t_s``, freeze the state, hand
  off to two independent FGFDMExec instances and coast them interleaved
  (the FL-402 path): bodies ``stack``/``booster``/``upper``, handoff
  bookkeeping as checks.

Conventions are the spike driver's (:mod:`orbitopt.flightlab.sixdof.
jsbsim_driver`): ECEF states (the manifest declares ``ECEF`` -- no frame
bridge is involved), scalar-first quaternion ``Tec2b``, SI everywhere; the
driver's extra columns ride along in the trajectory CSVs.
``physics_verified`` is the conjunction of this backend's own checks
(finite states, mass bookkeeping, handoff continuity, tracking bounds --
the spike's analytic-corroboration claims, evaluated per run); it is never
the integrator's word, and ``mission_constraints_passed`` stays None here
-- :func:`orbitopt.flightlab.simulate.simulate` settles it against the
target.

Determinism matches the OpenTsiolkovsky adapter: :func:`run_ascent` is a
pure function of the config (``created_utc`` = the config epoch, run id and
``vehicle_sha256`` from a digest of the config document), so repeated calls
write byte-identical run dirs. ``jsbsim==1.3.3`` is opt-in
(``pip install orbitopt[sixdof]``); importing this module works without it,
flying raises ``ImportError`` with the hint.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from orbitopt.flightlab.contract import (
    FLIGHT_PROVENANCE_TAGS,
    Check,
    Event,
    FlightRun,
    Verification,
    target_digest,
    write_run,
)
from orbitopt.flightlab.sixdof.jsbsim_driver import InitialConditions, SixDofSim, \
    mass_state_from_sample
from orbitopt.flightlab.sixdof.mass_properties import MassModelConfig, load_mass_model

#: Backend selector value accepted by ``simulate(backend=...)`` /
#: ``orbitopt flight simulate --backend``.
BACKEND = "6DOF"
BACKENDS = ("3DOF", "6DOF")

#: The jsbsim pip version this backend is written and tested against
#: (opt-in extra ``orbitopt[sixdof]``, the FL-401 spike environment).
JSBSIM_VERSION = "1.3.3"

_CONTROL_MODES = ("hold", "pid")
_STEP_ORDERS = ("ab", "ba")


class SixDofConfigError(ValueError):
    """A 6-DOF backend config document is malformed -- the offending key is
    named, so a scenario YAML typo fails here, not inside JSBSim."""


# --------------------------------------------------------------------------
# config document
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PidGainsConfig:
    kp: float = 3.0
    ki: float = 0.01
    kd: float = 4.0
    i_max_alpha: float = 0.05


@dataclass(frozen=True)
class PidConfig:
    """Closed-loop TVC: attitude program waypoints ``(t_s, az_deg, el_deg)``
    and the FL-403 controller settings. The flight duration is derived as
    ``meco_t_s + coast_s`` (throttle cuts exactly at ``meco_t_s``)."""

    program: tuple[tuple[float, float, float], ...]
    meco_t_s: float
    coast_s: float = 5.0
    gains: PidGainsConfig = field(default_factory=PidGainsConfig)


@dataclass(frozen=True)
class StagingConfig:
    """The FL-402 handoff scenario: freeze at ``at_t_s``, split dry mass /
    propellant by the declared fractions, coast both children ``coast_s``."""

    at_t_s: float
    coast_s: float = 61.0
    separation_impulse_ns: float = 0.0
    booster_dry_fraction: float = 0.4
    booster_propellant_fraction: float = 0.4
    booster_model: str = "spike_booster"
    upper_model: str = "spike_upper"
    step_order: str = "ab"


@dataclass(frozen=True)
class TrackingBounds:
    """Declared GO bounds for the closed-loop mode (the FL-403 ADR
    thresholds; tests assert against these config values)."""

    settled_from_s: float = 60.0
    settled_max_deg: float = 2.0
    settled_rms_deg: float = 0.6
    full_max_deg: float = 2.0
    gimbal_sat_fraction_max: float = 0.0
    d_alt_m: float = 500.0
    d_speed_m_s: float = 20.0
    d_fpa_deg: float = 0.2


@dataclass(frozen=True)
class SixDofConfig:
    """Parsed + validated 6-DOF backend scenario (module docstring for the
    document form). ``raw`` is the document as loaded -- its canonical JSON
    is the run-id digest, so the same file always produces the same run."""

    name: str
    provenance: str
    model: str
    dt_s: float
    duration_s: float | None
    ic: InitialConditions
    ignition_t_s: float
    ignition_throttle: float
    control_mode: str
    gimbal_pitch_rad: float
    gimbal_yaw_rad: float
    pid: PidConfig | None
    staging: StagingConfig | None
    mass_model: MassModelConfig | None
    tracking: TrackingBounds
    reference_csv: Path | None
    every: int
    epoch_utc: str
    body_id: str
    root: str | None
    assumptions: tuple[str, ...]
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False)

    @property
    def digest(self) -> str:
        text = json.dumps(dict(self.raw), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def duration_for_run_s(self) -> float:
        """Total stack flight duration: the closed-loop horizon (MECO +
        coast) in pid mode, else the declared ``duration_s``."""
        if self.control_mode == "pid":
            assert self.pid is not None
            return self.pid.meco_t_s + self.pid.coast_s
        assert self.duration_s is not None
        return self.duration_s


def _req_mapping(doc: Any, where: str) -> dict:
    if doc is None:
        return {}
    if not isinstance(doc, Mapping):
        raise SixDofConfigError(f"{where}: expected a mapping, got {type(doc).__name__}")
    return dict(doc)


def _num(doc: Mapping[str, Any], key: str, default: float | None, where: str,
         *, positive: bool = False) -> float:
    if key not in doc or doc[key] is None:
        if default is None:
            raise SixDofConfigError(f"{where}: required numeric key {key!r} is missing")
        return float(default)
    try:
        value = float(doc[key])
    except (TypeError, ValueError) as exc:
        raise SixDofConfigError(f"{where}: {key}={doc[key]!r} is not a number") from exc
    if positive and value <= 0.0:
        raise SixDofConfigError(f"{where}: {key} must be > 0, got {value}")
    return value


def _num_range(doc: Mapping[str, Any], key: str, default: float, where: str,
               low: float, high: float) -> float:
    value = _num(doc, key, default, where)
    if not low <= value <= high:
        raise SixDofConfigError(f"{where}: {key}={value} outside [{low}, {high}]")
    return value


def _resolve_reference(raw: str, config_dir: Path | None) -> Path:
    """Resolve a ``reference_csv`` path: absolute paths as-is; relative
    paths against the CWD first, then the config file's directory, then the
    repo root that owns this package (source checkouts). The reference is
    optional (FL-401/402 don't use one; FL-403 does), so a path that
    resolves to no file returns the *unresolved* path and the MECO-deviation
    checks are simply not assessed -- run_ascent checks ``.is_file()``."""
    p = Path(raw)
    if p.is_absolute():
        return p
    candidates = [Path.cwd() / p]
    if config_dir is not None:
        candidates.append(config_dir / p)
    candidates.append(Path(__file__).resolve().parents[4] / p)  # src/orbitopt/.../sixdof.py -> repo root
    for cand in candidates:
        if cand.is_file():
            return cand
    return p


def load_config(source) -> SixDofConfig:
    """Parse + validate a 6-DOF backend scenario: a path to a YAML/JSON
    file, or an already-loaded mapping. Raises :class:`SixDofConfigError`
    (a ValueError) naming the offending key."""
    config_dir: Path | None = None
    if isinstance(source, Mapping):
        doc = dict(source)
        where = "sixdof config"
    else:
        path = Path(source)
        config_dir = path.parent
        doc = _req_mapping(yaml.safe_load(path.read_text(encoding="utf-8")), str(path))
        where = str(path)
    for key in ("name", "model"):
        if not isinstance(doc.get(key), str) or not doc[key].strip():
            raise SixDofConfigError(f"{where}: required string key {key!r} is missing")
    provenance = str(doc.get("provenance", "illustrative"))
    if provenance not in FLIGHT_PROVENANCE_TAGS:
        raise SixDofConfigError(f"{where}: provenance {provenance!r} not in "
                                f"{FLIGHT_PROVENANCE_TAGS}")

    ic_doc = _req_mapping(doc.get("ic"), f"{where}: ic")
    _known = {"lat_geod_deg", "lon_deg", "alt_m", "pitch_deg", "heading_deg",
              "roll_deg", "v_north_m_s", "v_east_m_s", "v_down_m_s"}
    unknown_ic = sorted(set(ic_doc) - _known)
    if unknown_ic:
        raise SixDofConfigError(f"{where}: ic has unknown key(s): {', '.join(unknown_ic)}")
    try:
        ic = InitialConditions(**{k: float(v) for k, v in ic_doc.items()})
    except TypeError as exc:  # dataclass rejects an unexpected name
        raise SixDofConfigError(f"{where}: ic: {exc}") from exc

    ign_doc = _req_mapping(doc.get("ignition"), f"{where}: ignition")
    ignition_t_s = _num(ign_doc, "t_s", 0.0, f"{where}: ignition")
    ignition_throttle = _num_range(ign_doc, "throttle", 1.0, f"{where}: ignition", 0.0, 1.0)

    ctl_doc = _req_mapping(doc.get("control"), f"{where}: control")
    mode = str(ctl_doc.get("mode", "hold"))
    if mode not in _CONTROL_MODES:
        raise SixDofConfigError(f"{where}: control.mode {mode!r} not in {_CONTROL_MODES}")
    pid: PidConfig | None = None
    gimbal_pitch = _num(ctl_doc, "gimbal_pitch_rad", 0.0, f"{where}: control")
    gimbal_yaw = _num(ctl_doc, "gimbal_yaw_rad", 0.0, f"{where}: control")
    if mode == "pid":
        prog = ctl_doc.get("attitude_program")
        if not isinstance(prog, list) or not prog:
            raise SixDofConfigError(f"{where}: control.attitude_program must be a "
                                    "non-empty list of {t_s, azimuth_deg, elevation_deg}")
        points: list[tuple[float, float, float]] = []
        for i, p in enumerate(prog):
            p_doc = _req_mapping(p, f"{where}: control.attitude_program[{i}]")
            points.append((_num(p_doc, "t_s", None, f"{where}: attitude_program[{i}]"),
                           _num(p_doc, "azimuth_deg", 90.0, f"{where}: attitude_program[{i}]"),
                           _num(p_doc, "elevation_deg", 90.0,
                                f"{where}: attitude_program[{i}]")))
        points.sort(key=lambda p: p[0])
        gains_doc = _req_mapping(ctl_doc.get("gains"), f"{where}: control.gains")
        pid = PidConfig(
            program=tuple(points),
            meco_t_s=_num(ctl_doc, "meco_t_s", None, f"{where}: control", positive=True),
            coast_s=_num(ctl_doc, "coast_s", 5.0, f"{where}: control"),
            gains=PidGainsConfig(
                kp=_num(gains_doc, "kp", 3.0, f"{where}: control.gains"),
                ki=_num(gains_doc, "ki", 0.01, f"{where}: control.gains"),
                kd=_num(gains_doc, "kd", 4.0, f"{where}: control.gains"),
                i_max_alpha=_num(gains_doc, "i_max_alpha", 0.05, f"{where}: control.gains"),
            ),
        )

    staging: StagingConfig | None = None
    if doc.get("staging") is not None:
        st_doc = _req_mapping(doc["staging"], f"{where}: staging")
        if mode != "hold":
            raise SixDofConfigError(f"{where}: staging is supported with control.mode "
                                    "'hold' only (the FL-402 handoff flies open-loop)")
        order = str(st_doc.get("step_order", "ab"))
        if order not in _STEP_ORDERS:
            raise SixDofConfigError(f"{where}: staging.step_order {order!r} not in "
                                    f"{_STEP_ORDERS}")
        staging = StagingConfig(
            at_t_s=_num(st_doc, "at_t_s", None, f"{where}: staging", positive=True),
            coast_s=_num(st_doc, "coast_s", 61.0, f"{where}: staging"),
            separation_impulse_ns=_num(st_doc, "separation_impulse_ns", 0.0,
                                       f"{where}: staging"),
            booster_dry_fraction=_num_range(st_doc, "booster_dry_fraction", 0.4,
                                            f"{where}: staging", 0.0, 1.0),
            booster_propellant_fraction=_num_range(st_doc, "booster_propellant_fraction",
                                                   0.4, f"{where}: staging", 0.0, 1.0),
            booster_model=str(st_doc.get("booster_model", "spike_booster")),
            upper_model=str(st_doc.get("upper_model", "spike_upper")),
            step_order=order,
        )

    mass_model: MassModelConfig | None = None
    if doc.get("mass_model") is not None:
        mass_model = load_mass_model(doc["mass_model"], f"{where}: mass_model")

    trk_doc = _req_mapping(doc.get("tracking"), f"{where}: tracking")
    tracking = TrackingBounds(
        settled_from_s=_num(trk_doc, "settled_from_s", 60.0, f"{where}: tracking"),
        settled_max_deg=_num(trk_doc, "settled_max_deg", 2.0, f"{where}: tracking"),
        settled_rms_deg=_num(trk_doc, "settled_rms_deg", 0.6, f"{where}: tracking"),
        full_max_deg=_num(trk_doc, "full_max_deg", 2.0, f"{where}: tracking"),
        gimbal_sat_fraction_max=_num_range(trk_doc, "gimbal_sat_fraction_max", 0.0,
                                           f"{where}: tracking", 0.0, 1.0),
        d_alt_m=_num(trk_doc, "d_alt_m", 500.0, f"{where}: tracking"),
        d_speed_m_s=_num(trk_doc, "d_speed_m_s", 20.0, f"{where}: tracking"),
        d_fpa_deg=_num(trk_doc, "d_fpa_deg", 0.2, f"{where}: tracking"),
    )

    ref = doc.get("reference_csv")
    reference_csv = None if ref is None else _resolve_reference(str(ref), config_dir)

    out_doc = _req_mapping(doc.get("output"), f"{where}: output")
    every = int(_num_range(out_doc, "every", 1, f"{where}: output", 1, 1e6))
    duration_s = None if doc.get("duration_s") is None else _num(
        doc, "duration_s", None, where, positive=True)
    if mode == "hold" and duration_s is None:
        raise SixDofConfigError(f"{where}: duration_s is required in hold mode")
    if staging is not None and duration_s is not None and staging.at_t_s > duration_s:
        raise SixDofConfigError(f"{where}: staging.at_t_s={staging.at_t_s} exceeds "
                                f"duration_s={duration_s}")

    root = doc.get("root")
    return SixDofConfig(
        name=str(doc["name"]),
        provenance=provenance,
        model=str(doc["model"]),
        dt_s=_num(doc, "dt_s", 0.01, where, positive=True),
        duration_s=duration_s,
        ic=ic,
        ignition_t_s=ignition_t_s,
        ignition_throttle=ignition_throttle,
        control_mode=mode,
        gimbal_pitch_rad=gimbal_pitch,
        gimbal_yaw_rad=gimbal_yaw,
        pid=pid,
        staging=staging,
        mass_model=mass_model,
        tracking=tracking,
        reference_csv=reference_csv,
        every=every,
        epoch_utc=str(doc.get("epoch_utc", "2026-01-01T00:00:00Z")),
        body_id=str(doc.get("body_id", str(doc["model"]))),
        root=None if root is None else str(root),
        assumptions=tuple(str(a) for a in (doc.get("assumptions") or ())),
        raw=doc,
    )


def packaged_config(name: str) -> Path:
    """Path of a packaged 6-DOF scenario config (``orbitopt/data/flightlab/
    sixdof/<name>.yaml``): the FL-401/402/403 spike scenarios migrated to
    the formal backend; feed the path to :func:`load_config` or
    :func:`run_ascent`."""
    return Path(str(resources.files("orbitopt").joinpath(
        "data", "flightlab", "sixdof", f"{name}.yaml")))


# --------------------------------------------------------------------------
# flight: the three migrated scenarios, from one config
# --------------------------------------------------------------------------

def _finite_and_mass_checks(samples_per_body: Mapping[str, list],
                            checks: list[Check]) -> None:
    finite = all(math.isfinite(v) for samples in samples_per_body.values()
                 for s in samples for v in s.values())
    checks.append(Check(id="sixdof-states-finite", passed=finite,
                        reason_code="" if finite else "sixdof_non_finite_state"))
    min_mass = min(min(s["mass_kg"] for s in samples)
                   for samples in samples_per_body.values())
    checks.append(Check(id="sixdof-mass-non-negative", passed=min_mass >= 0.0,
                        value=min_mass, limit=0.0, unit="kg",
                        reason_code="" if min_mass >= 0.0 else "negative_mass"))


def _mass_state_deviation(state, ref) -> tuple[float, float]:
    """Worst (CG offset in m, relative inertia error) of a sampled
    :class:`MassState` against a reference one (F-03). Zero-valued tensor
    components compare against ``max(|ref|, 1 kg m^2)`` so that a numerically
    absent product-of-inertia cannot blow the relative error up."""
    cg = max(abs(a - b) for a, b in zip(state.cg_m, ref.cg_m))
    inertia = max(abs(a - b) / max(abs(b), 1.0)
                  for a, b in zip(state.inertia_kg_m2, ref.inertia_kg_m2))
    return cg, inertia


def _mass_model_checks(samples: list, cfg: SixDofConfig, checks: list[Check], *,
                       assess_burnout: bool = True) -> None:
    """F-03 (task #228): the variable mass / CG / inertia claims, judged
    against the scenario's declared mass model. Three acceptance claims:

    * ``sixdof-mass-conservation`` — the integrated engine flow (trapezoid,
      over the run's own sampled flow) must equal the mass the model reports
      having lost. JSBSim applies a step's drain to ``inertia/mass-slugs``
      one frame late, so the allowance is a declared number of integration
      frames of flow (``tolerances.flow_lag_frames``) plus an absolute floor.
    * ``sixdof-{cg,inertia}-full-load`` / ``sixdof-{cg,inertia}-burnout`` —
      the endpoints (first sample, and the last one once the tanks are dry)
      must match the declared ``full`` / ``empty`` states.
    * ``sixdof-{cg,inertia}-interpolation`` — in between, the flown state
      must match the model interpolated at each sample's own remaining
      propellant load, within looser tolerances (the one-frame drain lag is
      largest where the CG moves fastest, just before burnout).

    Only called when the config declares a ``mass_model``; ``assess_burnout``
    is False for staging runs, whose stack never reaches the endpoint."""
    props, tol = cfg.mass_model.properties, cfg.mass_model.tolerances

    def add(cid: str, value: float, limit: float, unit: str) -> None:
        ok = value <= limit
        checks.append(Check(id=cid, passed=ok, value=value, limit=limit, unit=unit,
                            reason_code="" if ok else "sixdof_mass_model_mismatch"))

    flow_peak = max(s["flow_kg_s"] for s in samples)
    dt_nom = (samples[-1]["t_s"] - samples[0]["t_s"]) / max(1, len(samples) - 1)
    m0, integral, residual = samples[0]["mass_kg"], 0.0, 0.0
    prev = samples[0]
    for s in samples[1:]:
        integral += 0.5 * (prev["flow_kg_s"] + s["flow_kg_s"]) * (s["t_s"] - prev["t_s"])
        residual = max(residual, abs(integral - (m0 - s["mass_kg"])))
        prev = s
    add("sixdof-mass-conservation", residual,
        tol.flow_lag_frames * flow_peak * dt_nom + tol.mass_abs_kg, "kg")

    cg_err, inertia_err = _mass_state_deviation(mass_state_from_sample(samples[0]),
                                                props.full)
    add("sixdof-cg-full-load", cg_err, tol.cg_abs_m, "m")
    add("sixdof-inertia-full-load", inertia_err, tol.inertia_rel, "fraction")

    worst_cg = worst_inertia = 0.0
    for s in samples:
        cg_err, inertia_err = _mass_state_deviation(
            mass_state_from_sample(s), props.at_propellant_mass_kg(s["propellant_kg"]))
        worst_cg = max(worst_cg, cg_err)
        worst_inertia = max(worst_inertia, inertia_err)
    add("sixdof-cg-interpolation", worst_cg, tol.interpolation_cg_abs_m, "m")
    add("sixdof-inertia-interpolation", worst_inertia,
        tol.interpolation_inertia_rel, "fraction")

    if not assess_burnout:
        return
    if samples[-1]["propellant_kg"] > tol.burnout_fraction * props.propellant_mass_kg:
        checks.append(Check(id="sixdof-mass-model-burnout-not-assessed", passed=False,
                            reason_code="sixdof_burnout_not_reached"))
        return
    cg_err, inertia_err = _mass_state_deviation(mass_state_from_sample(samples[-1]),
                                               props.empty)
    add("sixdof-cg-burnout", cg_err, tol.cg_abs_m, "m")
    add("sixdof-inertia-burnout", inertia_err, tol.inertia_rel, "fraction")


def _open_loop_flight(cfg: SixDofConfig, until_s: float | None = None) -> SixDofSim:
    """Fly one body open-loop (the FL-401 driver path): initialize, ignite
    at ``ignition.t_s``, hold the config's gimbal command until ``until_s``
    (default: the full ``duration_s``)."""
    sim = SixDofSim(cfg.dt_s, model=cfg.model, root=cfg.root)
    sim.initialize(cfg.ic)
    sim.run_for(cfg.ignition_t_s)
    sim.ignite(cfg.ignition_throttle)
    sim.set_gimbal(cfg.gimbal_pitch_rad, cfg.gimbal_yaw_rad)
    sim.run_for((cfg.duration_for_run_s() if until_s is None else until_s)
                - cfg.ignition_t_s)
    return sim


def _staging_flight(cfg: SixDofConfig, st: StagingConfig):
    """The FL-402 path, config-driven: fly open-loop to ``st.at_t_s``,
    freeze, hand off to two independent instances (FL-402 bookkeeping,
    impulse along body +x), coast them interleaved. Returns
    (stack, booster, upper, frozen, handoffs, bookkeeping)."""
    from orbitopt.flightlab.sixdof import staging as _staging

    if st.coast_s <= 0.0:
        raise SixDofConfigError("staging.coast_s must be > 0")
    stack = _open_loop_flight(cfg, until_s=st.at_t_s)
    frozen = _staging.capture_frozen_state(stack)
    alloc = _staging.StageSplit(st.booster_dry_fraction,
                                st.booster_propellant_fraction).allocate(frozen)
    # body +x at separation: first row of the body->ECEF DCM (staging.py's
    # quaternion helper; same package, FL-402's own convention)
    xb = _staging._mat3_row(_staging._quat_to_dcm(frozen.quaternion), 0)
    j = st.separation_impulse_ns
    masses = {side: sum(alloc[side][k] for k in ("dry_kg", "ox_kg", "fuel_kg"))
              for side in ("booster", "upper")}
    booster, hb = _staging.spawn_stage(
        st.booster_model, frozen, alloc["booster"], dt_s=cfg.dt_s,
        impulse_dv_ecef_m_s=tuple(-j * xb[i] / masses["booster"] for i in range(3)),
        root=cfg.root)
    upper, hu = _staging.spawn_stage(
        st.upper_model, frozen, alloc["upper"], dt_s=cfg.dt_s,
        impulse_dv_ecef_m_s=tuple(j * xb[i] / masses["upper"] for i in range(3)),
        root=cfg.root)

    b0, u0 = booster.samples[0], upper.samples[0]
    p0 = tuple(frozen.mass_kg * v for v in frozen.velocity_m_s)
    p_after = tuple(b0["mass_kg"] * b0[k] + u0["mass_kg"] * u0[k]
                    for k in ("vx_m_s", "vy_m_s", "vz_m_s"))
    bookkeeping = {
        "mass_before_kg": frozen.mass_kg,
        "mass_residual_kg": b0["mass_kg"] + u0["mass_kg"] - frozen.mass_kg,
        "momentum_before_kg_m_s": p0,
        "momentum_residual_kg_m_s": math.sqrt(sum((p_after[i] - p0[i]) ** 2
                                                  for i in range(3))),
        "dv_booster_m_s": j / masses["booster"],
        "dv_upper_m_s": j / masses["upper"],
        "position_continuity_err_m": max(hb["position_err_m"], hu["position_err_m"]),
        "attitude_continuity_err_rad": max(hb["attitude_err_rad"], hu["attitude_err_rad"]),
    }
    for _ in range(round(st.coast_s / cfg.dt_s)):
        if st.step_order == "ab":
            booster.step()
            upper.step()
        else:
            upper.step()
            booster.step()
    return stack, booster, upper, frozen, {"booster": hb, "upper": hu}, bookkeeping


def _staging_checks(bookkeeping: Mapping, handoff: Mapping,
                    checks: list[Check]) -> None:
    """Handoff bookkeeping as checks -- the claims the FL-402 spike made,
    with its declared tolerances, evaluated per run."""
    from orbitopt.flightlab.sixdof import staging as _staging

    p_norm = math.sqrt(sum(v * v for v in bookkeeping["momentum_before_kg_m_s"]))
    mom_tol = max(_staging.MOMENTUM_RESIDUAL_FLOOR_N_S,
                  _staging.MOMENTUM_RESIDUAL_TOL_REL * p_norm)
    vel_err = max(handoff["booster"]["velocity_err_m_s"], handoff["upper"]["velocity_err_m_s"])
    mass_tol = 1e-6 * max(1.0, bookkeeping["mass_before_kg"])
    specs = (
        ("sixdof-mass-closure", abs(bookkeeping["mass_residual_kg"]), mass_tol, "kg"),
        ("sixdof-momentum-residual", bookkeeping["momentum_residual_kg_m_s"], mom_tol, "N.s"),
        ("sixdof-position-continuity", bookkeeping["position_continuity_err_m"],
         _staging.HANDOFF_POSITION_TOL_M, "m"),
        ("sixdof-attitude-continuity", bookkeeping["attitude_continuity_err_rad"],
         _staging.HANDOFF_ATTITUDE_TOL_RAD, "rad"),
        ("sixdof-velocity-continuity", vel_err, _staging.HANDOFF_VELOCITY_TOL_M_S, "m/s"),
    )
    for cid, value, limit, unit in specs:
        ok = value <= limit
        checks.append(Check(id=cid, passed=ok, value=value, limit=limit, unit=unit,
                            reason_code="" if ok else "sixdof_handoff_continuity"))


def _pid_flight(cfg: SixDofConfig):
    """The FL-403 path, config-driven: closed-loop PID gimbal tracking of
    the config's attitude program, throttle cut exactly at ``meco_t_s``,
    coast ``coast_s``. The controller and the metric definitions are the
    tracking module's; the spike's fixed 5 s nominal coast and no-throttle-
    cut behaviour become the config's ``coast_s`` and a real MECO."""
    from orbitopt.flightlab.sixdof import tracking as _tracking

    assert cfg.pid is not None
    program = _tracking.AttitudeProgram(
        times=tuple(p[0] for p in cfg.pid.program),
        azimuth_deg=tuple(p[1] for p in cfg.pid.program),
        elevation_deg=tuple(p[2] for p in cfg.pid.program))
    reference = None
    if cfg.reference_csv is not None and Path(cfg.reference_csv).is_file():
        reference = _tracking.ReferenceTrajectory.from_csv(cfg.reference_csv)
    # _metrics needs *a* reference for its deviation numbers; with none
    # configured those numbers are meaningless and _pid_checks omits them.
    reference_for_metrics = reference if reference is not None else \
        _tracking.ReferenceTrajectory(
            t_s=np.array([0.0, 1.0]), altitude_m=np.zeros(2),
            speed_earth_m_s=np.zeros(2), fpa_deg=np.zeros(2),
            elevation_deg=np.zeros(2), mach=np.zeros(2), mass_kg=np.zeros(2))

    sim = SixDofSim(cfg.dt_s, model=cfg.model, root=cfg.root)
    sim.initialize(cfg.ic)
    r0 = math.sqrt(sim.samples[0]["x_m"] ** 2 + sim.samples[0]["y_m"] ** 2
                   + sim.samples[0]["z_m"] ** 2)
    sim.run_for(cfg.ignition_t_s)
    sim.ignite(cfg.ignition_throttle)
    ctrl = _tracking.PidAttitudeController(gains=_tracking.PidGains(
        kp=cfg.pid.gains.kp, ki=cfg.pid.gains.ki, kd=cfg.pid.gains.kd,
        i_max_alpha=cfg.pid.gains.i_max_alpha))
    meco_t_s = cfg.pid.meco_t_s
    meco_mach: list[float] = []
    cut: list[bool] = []

    def control(t: float, s: SixDofSim) -> None:
        cmd_p, cmd_y, _, _ = ctrl.command(s.sample(), program, s.dt_s)
        s.set_gimbal(cmd_p, cmd_y)
        if abs(t + s.dt_s - meco_t_s) <= 0.5 * s.dt_s and not cut:
            meco_mach.append(float(s["velocities/mach"]))
            s.set_throttle(0.0)  # MECO: the engine really stops here
            cut.append(True)

    sim.run_for(cfg.duration_for_run_s() - cfg.ignition_t_s, control)
    metrics = _tracking._metrics(sim, program, reference_for_metrics,
                                 meco_mach[0] if meco_mach else float("nan"),
                                 meco_t_s, meco_t_s, r0, cfg.dt_s)
    return _tracking.TrackingResult(sim, program, reference_for_metrics, metrics, r0), \
        reference


def _pid_checks(cfg: SixDofConfig, res, reference, checks: list[Check]) -> None:
    """FL-403 GO criteria as per-run checks, against the config's declared
    tracking bounds."""
    from orbitopt.flightlab.sixdof import tracking as _tracking

    b = cfg.tracking
    m = res.metrics

    def add(cid: str, value: float, limit: float, unit: str) -> None:
        ok = value <= limit
        checks.append(Check(id=cid, passed=ok, value=value, limit=limit, unit=unit,
                            reason_code="" if ok else "sixdof_tracking_out_of_bounds"))

    add("sixdof-att-tracking-full-max", m.att_err_max_deg, b.full_max_deg, "deg")
    settled = [max(_tracking.chart_error_deg(s, res.program), 0.0)
               for s in res.samples if s["t_s"] >= b.settled_from_s]
    err_max = max(settled) if settled else 0.0
    err_rms = math.sqrt(sum(e * e for e in settled) / len(settled)) if settled else 0.0
    add("sixdof-att-tracking-settled-max", err_max, b.settled_max_deg, "deg")
    add("sixdof-att-tracking-settled-rms", err_rms, b.settled_rms_deg, "deg")
    add("sixdof-gimbal-saturation", m.gimbal_sat_fraction_of_burn,
        b.gimbal_sat_fraction_max, "fraction")
    if reference is not None:
        add("sixdof-meco-altitude-deviation", abs(m.d_alt_m), b.d_alt_m, "m")
        add("sixdof-meco-speed-deviation", abs(m.d_speed_m_s), b.d_speed_m_s, "m/s")
        add("sixdof-meco-fpa-deviation", abs(m.d_fpa_deg), b.d_fpa_deg, "deg")


def run_ascent(config, out_dir, *, target=None) -> FlightRun:
    """Fly the 6-DOF scenario ``config`` (a path, a mapping accepted by
    :func:`load_config`, or a :class:`SixDofConfig`) into the FlightResult
    v1 run directory ``out_dir`` and return the :class:`FlightRun`. With a
    ``target``, its digest is recorded in the manifest; the constraint
    verdict itself is settled by :func:`orbitopt.flightlab.simulate.
    simulate`, never here.

    One body (hold/pid) or three (staging); frame ``ECEF`` throughout;
    ``physics_verified`` is the conjunction of this backend's checks
    (module docstring). Raises :class:`SixDofConfigError` on a malformed
    config, ``ImportError`` without the jsbsim extra, ``RuntimeError`` when
    JSBSim itself fails."""
    cfg = config if isinstance(config, SixDofConfig) else load_config(config)

    events: list[Event] = []
    checks: list[Check] = []
    assumptions_extra: tuple[str, ...] = ()
    if cfg.staging is not None:
        stack, booster, upper, frozen, handoff, bk = _staging_flight(cfg, cfg.staging)
        tracks = (
            stack.to_body_track(body_id="stack", epoch_utc=cfg.epoch_utc, every=cfg.every),
            booster.to_body_track(body_id="booster", epoch_utc=cfg.epoch_utc,
                                  every=cfg.every, parent_body_id="stack",
                                  t_offset_s=frozen.t_s),
            upper.to_body_track(body_id="upper", epoch_utc=cfg.epoch_utc,
                                every=cfg.every, parent_body_id="stack",
                                t_offset_s=frozen.t_s),
        )
        events.append(Event(cfg.ignition_t_s, "ignition", "stack",
                            {"throttle": cfg.ignition_throttle}))
        events.append(Event(frozen.t_s, "separation", "stack",
                            {"impulse_ns": cfg.staging.separation_impulse_ns,
                             "mass_kg": frozen.mass_kg}))
        events.append(Event(frozen.t_s, "separation", "booster",
                            {"dv_m_s": -bk["dv_booster_m_s"]}))
        events.append(Event(frozen.t_s, "separation", "upper",
                            {"dv_m_s": bk["dv_upper_m_s"]}))
        _finite_and_mass_checks({"stack": stack.samples, "booster": booster.samples,
                                 "upper": upper.samples}, checks)
        _staging_checks(bk, handoff, checks)
        if cfg.mass_model is not None:
            _mass_model_checks(stack.samples, cfg, checks, assess_burnout=False)
            assumptions_extra += (
                "Mass model checked on the stack only: the split children carry "
                "a fraction of the declared propellant, so their states are the "
                "parent model's neither, and the staging flight never burns out.",
            )
        vehicle_note = (f"{cfg.model} -> {cfg.staging.booster_model}/"
                        f"{cfg.staging.upper_model}")
        assumptions_extra = (
            f"Staging handoff at t={frozen.t_s:.3f} s: impulse "
            f"{cfg.staging.separation_impulse_ns} N.s along body +x, mass split "
            f"{cfg.staging.booster_dry_fraction}/{cfg.staging.booster_propellant_fraction} "
            f"(dry/propellant), children stepped interleaved "
            f"'{cfg.staging.step_order}' for {cfg.staging.coast_s} s.",
        )
    elif cfg.control_mode == "pid":
        res, reference = _pid_flight(cfg)
        tracks = (res.sim.to_body_track(body_id=cfg.body_id, epoch_utc=cfg.epoch_utc,
                                        every=cfg.every),)
        events.append(Event(cfg.ignition_t_s, "ignition", cfg.body_id,
                            {"throttle": cfg.ignition_throttle}))
        assert cfg.pid is not None
        events.append(Event(cfg.pid.meco_t_s, "meco", cfg.body_id, {}))
        _finite_and_mass_checks({cfg.body_id: res.samples}, checks)
        _pid_checks(cfg, res, reference, checks)
        if cfg.mass_model is not None:
            _mass_model_checks(res.samples, cfg, checks)
        if reference is None:
            checks.append(Check(id="sixdof-meco-deviation-not-assessed", passed=False,
                                reason_code="REFLY_BACKEND_MISSING"))
        vehicle_note = cfg.model
        assumptions_extra = (
            "Closed-loop TVC (acceleration-space PID with instrumented gain "
            "schedule, FL-403 controller) tracking the config's attitude "
            "program; MECO throttle cut at meco_t_s; GO bounds from the "
            "config's tracking section.",
        )
    else:
        sim = _open_loop_flight(cfg)
        tracks = (sim.to_body_track(body_id=cfg.body_id, epoch_utc=cfg.epoch_utc,
                                    every=cfg.every),)
        events.append(Event(cfg.ignition_t_s, "ignition", cfg.body_id,
                            {"throttle": cfg.ignition_throttle}))
        _finite_and_mass_checks({cfg.body_id: sim.samples}, checks)
        if cfg.mass_model is not None:
            _mass_model_checks(sim.samples, cfg, checks)
        vehicle_note = cfg.model

    if cfg.mass_model is not None:
        full, empty = cfg.mass_model.properties.full, cfg.mass_model.properties.empty
        assumptions_extra += (
            f"Declared mass model (F-03, task #228): {full.mass_kg} kg full / "
            f"{empty.mass_kg} kg burnout with the config's CG and inertia endpoints; "
            "the flown mass/CG/inertia trace is checked against it by the "
            "sixdof-mass-* / sixdof-cg-* / sixdof-inertia-* checks.",
        )

    physics = all(c.passed for c in checks)
    assumptions = [
        f"6-DOF backend (jsbsim {JSBSIM_VERSION}): vehicle {vehicle_note!r} from the "
        "config, ECEF states (frame ECEF, no frame bridge), zero-aero spike "
        "models: illustrative, not flight-certified.",
        "config-driven: vehicle, initial conditions and integration step come "
        f"from the scenario document (sha256 {cfg.digest[:16]}), not from code",
        "created_utc is the config epoch: run_ascent is a pure function of the "
        "config, so repeated calls write byte-identical runs",
        "physics_verified is the conjunction of this backend's checks ("
        + ", ".join(c.id for c in checks) + "); mission constraints are settled "
        "by the Gate-B pipeline, never here",
    ] + list(assumptions_extra) + [f"config: {a}" for a in cfg.assumptions]
    run = FlightRun(
        run_id=f"jsbsim-6dof-{cfg.digest[:12]}",
        target_sha256=(target_digest(target.to_dict()) if target is not None
                       else hashlib.sha256(f"no-target:{cfg.name}".encode()).hexdigest()),
        vehicle_sha256=cfg.digest,
        git_sha="0" * 40,
        created_utc=cfg.epoch_utc,
        engines={"jsbsim": JSBSIM_VERSION, "aircraft": vehicle_note},
        assumptions=tuple(assumptions),
        tracks=tracks,
        events=tuple(events),
        verification=Verification(
            optimizer_converged=None,
            physics_verified=physics,
            mission_constraints_passed=None,
            checks=tuple(checks),
        ),
    )
    write_run(Path(out_dir), run)
    return run
