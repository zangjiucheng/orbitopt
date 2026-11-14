"""Mission Designer / Vehicle Builder form engine (FL-502, #210).

A code API: Mission Control no longer has a designer page (Flight Lab only
renders runs), so a run is designed here or in YAML and simulated from code
or the CLI. Qt-free, so the whole form -> target/vehicle -> simulate -> cache
pipeline runs and tests on a base install without a display:

* :func:`target_from_form` turns Mission Designer fields into a schema-valid
  FlightTarget via :func:`orbitopt.flightlab.target_export.build_target`,
  mapping every rejection to a *field-level* message the form can show;
* :func:`vehicle_document` / :func:`vehicle_form_from_document` map Vehicle
  Builder fields to a flightlab vehicle file (OT stage configs in, UI units
  out) and back, so a preset can be loaded, edited and re-flown without the
  user ever touching YAML;
* :func:`validate_vehicle_document` reports OT-schema violations as field
  paths -> messages, the same contract as the target mapping;
* :func:`derived_metrics` computes the live preview numbers (liftoff T/W,
  per-stage ideal vacuum delta-v) from the document alone -- same formulas
  :mod:`orbitopt.flightlab.checks` applies post-run, no subprocess;
* :func:`inputs_hash` + :func:`cached_run_dir` implement the result cache:
  one run directory per distinct (mission, vehicle, options) input, keyed by
  a SHA-256 over the canonical JSON of exactly what would be simulated, so a
  second Simulate click with unchanged inputs reopens the existing run;
* :func:`run_form_job` is the picklable worker entry for the compute-pool
  run. It isolates the worker into its own POSIX process group with a SIGTERM
  handler that kills the whole group, which is what makes Cancel stop the
  OpenTsiolkovsky *grandchild* and not just the worker holding it.

Nothing here fabricates physics: fields the E1 3-DoF backend does not model
(min throttle) are recorded as declared assumptions, exactly as the rest of
the flightlab contract requires.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import signal
from pathlib import Path
from typing import Any, Mapping

import yaml

from orbitopt.flightlab.checks import MIN_LIFTOFF_THRUST_TO_WEIGHT, STANDARD_GRAVITY
from orbitopt.flightlab.contract import FLIGHT_PROVENANCE_TAGS, FlightTarget
from orbitopt.flightlab.target_export import build_target
from orbitopt.launch_sites import LaunchSite, default_launch_sites, get_launch_site

#: Sea-level standard atmosphere [Pa], used only for the derived *display*
#: value "thrust at sea level = vacuum thrust - exit area x p0" (OT's own
#: engine model, evaluated on the launch pad). Not a physics claim.
SEA_LEVEL_PRESSURE_PA = 101325.0

#: Fallback Cd/Ca-over-Mach table for a stage with no table of its own --
#: the generic tube-and-fins guess the packaged preset uses (illustrative).
DEFAULT_CA_TABLE: list[list[float]] = [
    [0, 0.35], [0.8, 0.38], [1.1, 0.55], [1.5, 0.5], [2.5, 0.4], [4, 0.32], [8, 0.26],
]

#: Files inside a cached run directory that mark a complete, openable run.
RUN_MANIFEST = "manifest.json"
#: In-flight marker written before a run starts (its text is the worker's
#: process-group id on POSIX, so Cancel can killpg even if the pool would
#: SIGKILL the worker); removed on both success and failure.
IN_PROGRESS = "_designer-in-progress"

#: Escape hatch for in-process tests of :func:`run_form_job` (the process
#: isolation it installs would otherwise reparent the test runner).
ENV_NO_PGRP = "ORBITOPT_FL_DESIGNER_NO_PGRP"

VEHICLE_PRESET_DIR = "data/flightlab/vehicles"


# --------------------------------------------------------------------------- mission form


def mission_sites() -> list[LaunchSite]:
    """The packaged launch-site database, for the form's site combo."""
    return list(default_launch_sites())


def _fnum(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)  # raises ValueError; the caller reports the field


def target_from_form(form: Mapping[str, Any]) -> tuple[FlightTarget | None, dict[str, str]]:
    """Build the FlightTarget a Mission Designer form describes, or collect
    field-level messages instead. ``form`` keys: ``site``, ``periapsis_km``,
    ``apoapsis_km``, ``inclination_deg``, ``raan_deg`` (None = unconstrained),
    ``date`` (ISO-8601), ``window_days``, ``recovery`` (``none``/``downrange``
    /``rtls``), ``recovery_reserve_kg``, ``max_q_kpa``, ``max_g``,
    ``min_periapsis_km``. Every rejection is attributed to the one field the
    user should fix; the returned dict is ``{field: message}``.
    """
    errors: dict[str, str] = {}
    for key in ("periapsis_km", "apoapsis_km", "inclination_deg", "window_days"):
        try:
            _fnum(form.get(key))
        except (TypeError, ValueError):
            errors[key] = "enter a number"
    if not str(form.get("date") or "").strip():
        errors["date"] = "a launch date is required -- an epoch is never invented here"
    if not str(form.get("site") or "").strip():
        errors["site"] = "pick a launch site"
    if errors:
        return None, errors

    constraints: dict[str, float] = {}
    for key, field in (("max_q_kpa", "max_dynamic_pressure_pa"),
                       ("max_g", "max_specific_acceleration_g"),
                       ("min_periapsis_km", "minimum_insertion_periapsis_m")):
        try:
            value = _fnum(form.get(key))
        except (TypeError, ValueError):
            errors["constraints"] = f"{key}: enter a number or leave it blank"
            continue
        if value is not None and value > 0:
            constraints[field] = value * (1000.0 if key != "max_g" else 1.0)
    if errors:
        return None, errors

    recovery_kind = str(form.get("recovery") or "none")
    recovery: dict[str, Any] = {"kind": recovery_kind}
    if recovery_kind in ("rtls", "downrange"):
        recovery["target"] = "launch_site"
        reserve = _fnum(form.get("recovery_reserve_kg"))
        if reserve is not None and reserve > 0:
            recovery["min_propellant_reserve_kg"] = reserve

    try:
        target = build_target(
            str(form["site"]),
            float(form["periapsis_km"]) * 1000.0,
            float(form["apoapsis_km"]) * 1000.0,
            float(form["inclination_deg"]),
            raan_deg=_fnum(form.get("raan_deg")),
            date_utc=str(form["date"]),
            constraints=constraints or None,
            recovery=recovery,
            window_days=float(form.get("window_days") or 2.0),
        )
    except KeyError as exc:
        errors["site"] = str(exc.args[0] if exc.args else exc)
    except ValueError as exc:
        errors[_classify_target_error(str(exc))] = str(exc)
    except Exception as exc:  # jsonschema.ValidationError etc.
        field = getattr(getattr(exc, "absolute_path", None), "popleft", lambda: None)()
        errors[str(field or "mission")] = str(exc).splitlines()[0]
        errors.setdefault("mission", type(exc).__name__)
    return (None, errors) if errors else (target, {})


def _classify_target_error(message: str) -> str:
    """Map build_target's readable rejection reasons onto form fields."""
    low = message.lower()
    if "apoapsis" in low and "below periapsis" in low:
        return "apoapsis_km"
    if "altitudes must be" in low:
        return "periapsis_km"
    if "date_utc" in low or "launch date" in low or "isoformat" in low:
        return "date"
    if "in-plane launch window" in low:
        return "date"
    if "window_days" in low:
        return "window_days"
    if "inclination" in low:
        return "inclination_deg"
    return "site"


def launch_azimuth(site: LaunchSite, inclination_deg: float) -> float | None:
    """A launch azimuth [deg] inside one of ``site``'s range-safety corridors
    that reaches ``inclination_deg`` (prograde preferred), or None. Used only
    to generate a default attitude program from the form -- the corridor and
    feasibility claim itself stays with build_target. Sampled + bisection
    refinement, so a corridor *containing* the exact solution is found even
    when the analytic solution sits on the other side of a wrap-around."""
    c = math.cos(math.radians(inclination_deg)) / math.cos(math.radians(site.latitude_deg))
    if abs(c) > 1.0:
        return None
    for az in (90.0 - math.degrees(math.acos(c)), 90.0 + math.degrees(math.acos(c))):
        az %= 360.0
        for lo, hi in site.azimuth_corridors_deg:
            a, b = sorted((lo % 360.0, hi % 360.0))
            if a <= az <= b and abs(site.inclination_for_azimuth(az) - inclination_deg) < 0.01:
                return az

    def err(az: float) -> float:
        return site.inclination_for_azimuth(az % 360.0) - inclination_deg

    for lo, hi in site.azimuth_corridors_deg:  # corridor-crossing refinement
        a = lo % 360.0
        for _ in range(721):
            b = min(a + 0.5, (hi % 360.0) if hi >= lo else 360.0)
            if err(a) * err(b) <= 0.0:
                for _ in range(40):  # bisect the sign change
                    mid = 0.5 * (a + b)
                    a, b = (mid, b) if err(a) * err(mid) <= 0.0 else (a, mid)
                return 0.5 * (a + b) % 360.0
            a = b
    return None


def default_attitude_program(azimuth_deg: float) -> list[dict]:
    """A gravity-turn-shaped default pitch program [deg] flying the given
    azimuth: vertical to t=30 s, then a smooth down-bend to horizontal by
    t=440 s -- the same shape the packaged preset hand-tuned, generated from
    the form's own azimuth so the program can never silently disagree with
    the chosen site/inclination."""
    bend = [(0.0, 90.0), (30.0, 87.0), (60.0, 72.0), (90.0, 57.0), (120.0, 44.0),
            (160.0, 30.0), (214.0, 26.0), (340.0, 16.0), (400.0, 4.0), (440.0, 0.0),
            (620.0, 0.0)]
    return [{"t_s": t, "azimuth_deg": round(azimuth_deg, 1), "elevation_deg": e}
            for t, e in bend]


# --------------------------------------------------------------------------- vehicle form


def vehicle_presets() -> dict[str, str]:
    """Packaged vehicle presets: ``{preset_name: YAML text}`` under
    ``orbitopt/data/flightlab/vehicles`` (the file stem is the preset name).
    Text, not paths: a wheel install has no stable on-disk path per file, and
    the simulate path writes the (edited) document to its own temp file
    anyway."""
    from importlib import resources

    base = resources.files("orbitopt").joinpath(*VEHICLE_PRESET_DIR.split("/"))
    out: dict[str, str] = {}
    for entry in sorted(base.iterdir(), key=lambda p: str(p.name)):
        if entry.name.endswith((".yaml", ".yml")):
            out[entry.name.rsplit(".", 1)[0]] = entry.read_text(encoding="utf-8")
    return out


def load_vehicle_preset(name: str) -> dict:
    return yaml.safe_load(vehicle_presets()[name])


def _curve_out(value: Any, scale: float = 1.0) -> Any:
    if isinstance(value, (int, float)):
        return float(value) * scale
    if isinstance(value, list):
        return [[float(x), float(y) * scale] for x, y in value]
    return value


def _curve_in(value: Any, scale: float = 1.0) -> Any:
    if isinstance(value, (int, float)):
        return float(value) / scale
    if isinstance(value, list):
        return [[float(x), float(y) / scale] for x, y in value]
    return value


def vehicle_form_from_document(doc: Mapping[str, Any]) -> dict:
    """Parse a flightlab vehicle document into Vehicle Builder form fields
    (UI units: masses in t, thrust in kN, payload separate). Inverse of
    :func:`vehicle_document` up to unit conversion."""
    stages = []
    for s in doc.get("stages") or []:
        eng = s.get("engine") or {}
        aero = s.get("aero") or {}
        stages.append({
            "initial_mass_t": float(s["mass_initial_kg"]) / 1000.0,
            "dry_mass_t": float(s.get("mass_dry_kg", 0.0)) / 1000.0,
            "thrust_kn": _curve_out(eng.get("thrust_vac_n"), 1e-3),
            "isp_vac_s": float(eng.get("isp_vac_s", 300.0)),
            "burn_start_s": float(eng.get("burn_start_s", 0.0)),
            "burn_end_s": float(eng.get("burn_end_s", 0.0)),
            "nozzle_area_m2": float(eng.get("nozzle_exit_area_m2", 0.0)),
            "diameter_m": float(aero.get("diameter_m", 3.7)),
            "ca": _curve_out(aero.get("ca", DEFAULT_CA_TABLE)),
            "cn_alpha": float(aero.get("cn_alpha_per_rad", 0.0)),
            "ballistic_coeff_kg_m2": float(aero.get("ballistic_coefficient_kg_m2", 100.0)),
            "free_flight": str(s.get("free_flight", "attitude_program")),
            "separation_time_s": (float(s["separation_time_s"])
                                  if "separation_time_s" in s else None),
            "jettisons": [
                {"name": str(j.get("name", "")), "time_s": float(j["time_s"]),
                 "mass_t": float(j["mass_kg"]) / 1000.0,
                 "ballistic_coeff_kg_m2": float(j.get("ballistic_coefficient_kg_m2", 100.0))}
                for j in s.get("jettisons") or []
            ],
        })
    sim = doc.get("simulation") or {}
    return {
        "name": str(doc.get("name", "designer-vehicle")),
        "provenance": str(doc.get("provenance", "illustrative")),
        "payload_t": float(doc.get("payload_kg", 0.0)) / 1000.0,
        "end_time_s": float(sim.get("end_time_s", 950.0)),
        "output_step_s": float(sim.get("output_step_s", 2.0)),
        "attitude_program": [dict(p) for p in doc.get("attitude_program") or []],
        "stages": stages,
        "preset": str(doc.get("_preset", "")),
    }


def vehicle_document(form: Mapping[str, Any]) -> dict:
    """The flightlab vehicle document a Vehicle Builder form describes -- the
    file :func:`orbitopt.flightlab.simulate.simulate` would fly. UI units are
    converted; the payload is folded into the final stage's wet and dry mass
    (it is not burnable), which is exactly the bookkeeping
    :func:`orbitopt.flightlab.checks.propellant_reserve_check` assumes, and
    recorded as ``payload_kg`` so the form can round-trip it. Stages carry
    nothing the OT schema would reject (``additionalProperties: false``)."""
    stages_in = list(form.get("stages") or [])
    payload_kg = (float(form.get("payload_t") or 0.0)) * 1000.0
    stages = []
    for i, s in enumerate(stages_in):
        payload = payload_kg if i == len(stages_in) - 1 else 0.0
        engine = {
            "thrust_vac_n": _curve_in(s.get("thrust_kn"), 1e-3),
            "isp_vac_s": float(s["isp_vac_s"]),
            "burn_start_s": float(s.get("burn_start_s", 0.0)),
            "burn_end_s": float(s["burn_end_s"]),
        }
        if float(s.get("nozzle_area_m2") or 0.0) > 0:
            engine["nozzle_exit_area_m2"] = float(s["nozzle_area_m2"])
        stage: dict[str, Any] = {
            "mass_initial_kg": float(s["initial_mass_t"]) * 1000.0 + payload,
            "mass_dry_kg": float(s["dry_mass_t"]) * 1000.0 + payload,
            "engine": engine,
            "aero": {
                "diameter_m": float(s.get("diameter_m") or 3.7),
                "ca": _curve_in(s.get("ca") or DEFAULT_CA_TABLE),
                "ballistic_coefficient_kg_m2": float(s.get("ballistic_coeff_kg_m2") or 100.0),
            },
            "free_flight": str(s.get("free_flight") or "attitude_program"),
        }
        if float(s.get("cn_alpha") or 0.0) > 0:
            stage["aero"]["cn_alpha_per_rad"] = float(s["cn_alpha"])
        if s.get("separation_time_s") is not None and str(s["separation_time_s"]) != "":
            stage["separation_time_s"] = float(s["separation_time_s"])
        if s.get("jettisons"):
            stage["jettisons"] = [
                {"time_s": float(j["time_s"]), "mass_kg": float(j["mass_t"]) * 1000.0,
                 "ballistic_coefficient_kg_m2": float(j.get("ballistic_coeff_kg_m2") or 100.0),
                 **({"name": str(j["name"])} if j.get("name") else {})}
                for j in s["jettisons"]
            ]
        stages.append(stage)
    doc = {
        "name": str(form.get("name") or "designer-vehicle"),
        "provenance": str(form.get("provenance") or "illustrative"),
        "payload_kg": payload_kg,
        "simulation": {"end_time_s": float(form.get("end_time_s") or 950.0),
                       "output_step_s": float(form.get("output_step_s") or 2.0),
                       "integrator": {"method": "rk45"}},
        "attitude_program": [dict(p) for p in form.get("attitude_program") or []],
        "stages": stages,
    }
    if any(s.get("min_throttle_pct") not in (None, 100.0) for s in stages_in):
        throttle = [float(s.get("min_throttle_pct") or 100.0) / 100.0 for s in stages_in]
        doc["min_throttle_fractions"] = throttle
        doc["assumptions"] = doc.get("assumptions", []) + [
            "min_throttle_fractions: recorded for the E2 guidance/landing model; the E1 "
            "OpenTsiolkovsky 3-DoF backend flies full throttle (no throttle law in v1.0.0)"]
    return doc


def validate_vehicle_document(doc: Mapping[str, Any]) -> dict[str, str]:
    """Field-level messages for a vehicle document: ``{}`` when it would be
    accepted by :func:`orbitopt.flightlab.simulate.simulate`, else
    ``"stages[0].engine.thrust_vac_n"``-style paths -> first message. The
    OT config schema's Stage rows are the authority (validated against the
    root ``stages`` property with the schema's own ``$defs``, i.e. the same
    rules ``build_config`` will enforce at simulate time); the top-level
    checks mirror :meth:`Vehicle.load`."""
    errors: dict[str, str] = {}
    if str(doc.get("provenance")) not in FLIGHT_PROVENANCE_TAGS:
        errors["provenance"] = (f"provenance must be one of {', '.join(FLIGHT_PROVENANCE_TAGS)}, "
                                f"got {doc.get('provenance')!r}")
    if not str(doc.get("name") or "").strip():
        errors["name"] = "give the vehicle a name"
    if not (float((doc.get("simulation") or {}).get("end_time_s", 0)) > 0):
        errors["end_time_s"] = "simulation.end_time_s must be > 0 (where to stop the ascent)"
    stages = doc.get("stages")
    if not isinstance(stages, list) or not stages:
        errors["stages"] = "at least one stage is required"
        return errors

    import jsonschema

    from orbitopt.flightlab.adapters.opentsiolkovsky import pinned_config_schema

    root = pinned_config_schema()
    stages_prop = dict(root["properties"]["stages"])
    stages_prop["minItems"] = 1
    subschema = {"$defs": root["$defs"], **stages_prop}
    validator = jsonschema.Draft202012Validator(subschema)
    for err in sorted(validator.iter_errors(list(stages)),
                      key=lambda e: str(list(e.absolute_path))):
        path = [str(p) for p in err.absolute_path]
        if path and path[0].isdigit():
            field = f"stages[{path[0]}]" + ("." + ".".join(path[1:]) if len(path) > 1 else "")
        else:
            field = "stages"
        errors.setdefault(field, str(err.message))

    # Semantic pass: the OT schema types every number but bounds none of them,
    # and OT fails at *run* time on these, so block them in the form instead.
    def _values(v: Any) -> list[float]:
        if isinstance(v, (int, float)):
            return [float(v)]
        return [float(r[1]) for r in v if isinstance(r, (list, tuple))] if isinstance(v, list) else []

    for i, s in enumerate(stages):
        p = f"stages[{i}]"
        if not isinstance(s, dict):
            continue
        m0 = s.get("mass_initial_kg")
        md = s.get("mass_dry_kg")
        if isinstance(m0, (int, float)) and m0 <= 0:
            errors.setdefault(p + ".mass_initial_kg", "initial mass must be positive")
        if isinstance(md, (int, float)) and isinstance(m0, (int, float)) and md >= m0:
            errors.setdefault(p + ".mass_dry_kg", "dry mass must be below initial mass (propellant is the difference)")
        eng = s.get("engine")
        if isinstance(eng, dict):
            if eng.get("thrust_vac_n") is not None and any(v <= 0 for v in _values(eng["thrust_vac_n"])):
                errors.setdefault(p + ".engine.thrust_vac_n", "thrust must be positive")
            if eng.get("isp_vac_s") is not None and any(v <= 0 for v in _values(eng["isp_vac_s"])):
                errors.setdefault(p + ".engine.isp_vac_s", "specific impulse must be positive")
            bs, be = eng.get("burn_start_s", 0.0), eng.get("burn_end_s")
            if isinstance(be, (int, float)) and isinstance(bs, (int, float)) and be <= bs:
                errors.setdefault(p + ".engine.burn_end_s", "burn ends before it starts")
        if i < len(stages) - 1 and s.get("separation_time_s") is None:
            errors.setdefault(p + ".separation_time_s", "every stage but the last needs a separation time")
        d = (s.get("aero") or {}).get("diameter_m")
        if isinstance(d, (int, float)) and d <= 0:
            errors.setdefault(p + ".aero.diameter_m", "diameter must be positive")
    return errors


def derived_metrics(doc: Mapping[str, Any]) -> dict:
    """Live preview numbers from the vehicle document alone (no subprocess):
    liftoff thrust-to-weight (the Gate-B formula from
    :mod:`orbitopt.flightlab.checks`: stage-1 thrust at t=0 over liftoff
    stack mass), sea-level thrust (OT's own engine model on the pad:
    ``vacuum thrust - exit area x 101325 Pa``), per-stage reference area and
    propellant mass, and per-stage *ideal vacuum* delta-v
    (``Isp g0 ln(m0/m_burnout)`` -- no gravity/drag losses, labelled as such
    wherever it is shown)."""
    out: dict[str, Any] = {"stages": []}
    stages = doc.get("stages") or []
    if not stages:
        return out
    first = stages[0]
    thrust0 = _liftoff_thrust0_n(first)
    mass0 = float(first.get("mass_initial_kg", 0.0))
    if thrust0 is not None and mass0 > 0:
        tw = thrust0 / (mass0 * STANDARD_GRAVITY)
        out["liftoff_thrust_to_weight"] = tw
        out["liftoff_tw_ok"] = tw >= MIN_LIFTOFF_THRUST_TO_WEIGHT
    for i, s in enumerate(stages):
        eng = s.get("engine") or {}
        aero = s.get("aero") or {}
        diameter = float(aero.get("diameter_m", 0.0))
        area = math.pi / 4.0 * diameter * diameter
        dry = float(s.get("mass_dry_kg", 0.0))
        initial = float(s.get("mass_initial_kg", 0.0))
        jett = sum(float(j.get("mass_kg", 0.0)) for j in s.get("jettisons") or [])
        propellant = max(initial - dry, 0.0)
        isp = _curve_at0(eng.get("isp_vac_s"))
        m_burnout = max(dry - jett, 1.0)
        dv = (isp * STANDARD_GRAVITY * math.log(initial / m_burnout)
              if isp and initial > m_burnout > 0 else None)
        sl = None
        if thrust0 is not None and i == 0:
            sl = max(thrust0 - float(eng.get("nozzle_exit_area_m2", 0.0)) * SEA_LEVEL_PRESSURE_PA, 0.0)
        out["stages"].append({
            "index": i,
            "propellant_kg": propellant,
            "reference_area_m2": area,
            "ideal_delta_v_m_s": dv,
            "sea_level_thrust_n": sl,
            "min_throttle_fraction": None,
        })
    if "min_throttle_fractions" in doc:
        for entry, frac in zip(out["stages"], doc["min_throttle_fractions"]):
            entry["min_throttle_fraction"] = float(frac)
    return out


def _liftoff_thrust0_n(stage: Mapping[str, Any]) -> float | None:
    """Same convention as ``checks._liftoff_thrust_n``: scalar thrust, or the
    first row of the throttle curve."""
    thrust = (stage.get("engine") or {}).get("thrust_vac_n")
    if isinstance(thrust, (int, float)):
        return float(thrust)
    if isinstance(thrust, list) and thrust:
        return float(thrust[0][1])
    return None


def _curve_at0(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list) and value:
        return float(value[0][1])
    return None


# --------------------------------------------------------------------------- cache + job


def cache_root() -> Path:
    """``$ORBITOPT_CACHE_DIR/flightlab`` (or ``~/.cache/orbitopt/flightlab``)
    -- outside the repo, same convention as :mod:`orbitopt.scene_cache`."""
    override = os.environ.get("ORBITOPT_CACHE_DIR")
    if override:
        return Path(override).expanduser() / "flightlab"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "orbitopt" / "flightlab"


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def inputs_hash(target_doc: Mapping[str, Any], vehicle_doc: Mapping[str, Any],
                options: Mapping[str, Any]) -> str:
    """SHA-256 of the exact inputs a run would consume: the target document
    *minus its generator provenance line* (the generator records which build
    and day produced it -- re-running the same form the next day is the same
    input), the vehicle document, and the simulate options. Same hash => the
    same subprocess would be launched on the same documents => a cache hit is
    always a valid answer."""
    target = copy.deepcopy(dict(target_doc))
    assumptions = target.get("assumptions")
    if isinstance(assumptions, list):
        # Generator provenance records the build *and the day* it ran -- two
        # identical form fills on different days are the same input.
        target["assumptions"] = [
            a for a in assumptions
            if not str(a).startswith("generated by ") and "generator:" not in str(a)]
    payload = _canonical({"target": target, "vehicle": vehicle_doc, "options": dict(options)})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cached_run_dir(digest: str) -> Path | None:
    """The run directory of a *complete* cached run for this input hash, or
    None (never cached, or still in flight, or a failed remnant)."""
    run = cache_root() / "runs" / digest
    if (run / IN_PROGRESS).exists():
        return None
    if (run / RUN_MANIFEST).exists():
        return run
    return None


def build_job(target: FlightTarget, vehicle_doc: Mapping[str, Any],
              options: Mapping[str, Any] | None = None,
              mission_form: Mapping[str, Any] | None = None,
              vehicle_form: Mapping[str, Any] | None = None) -> tuple[str, dict]:
    """``(digest, job)``: the cache key and the picklable payload for
    :func:`run_form_job`. Options: ``timeout_s`` (OpenTsiolkovsky bound),
    ``refly`` (tudatpy orbit re-fly; degrades to an honest
    REFLY_BACKEND_MISSING check without it). The raw form dicts ride along as
    audit material (``designer-form.json`` next to the run); they are *not*
    part of the cache key -- the documents are."""
    options = dict(options or {})
    options.setdefault("timeout_s", 300.0)
    options.setdefault("refly", True)
    digest = inputs_hash(target.to_dict(), vehicle_doc, options)
    run_dir = cache_root() / "runs" / digest
    job = {"run_dir": str(run_dir), "digest": digest,
           "target": target.to_dict(), "vehicle": dict(vehicle_doc), "options": options,
           "mission_form": dict(mission_form or {}), "vehicle_form": dict(vehicle_form or {})}
    return digest, job


def _isolate_worker_process_group() -> int | None:
    """Run the worker in its own process group and turn SIGTERM into
    kill-whole-group, so a Cancel (which reaches the *worker* as SIGTERM,
    possibly escalated to SIGKILL by the pool) also stops the OpenTsiolkovsky
    child. POSIX only; on Windows the pool's kill plus the subprocess timeout
    in ``run_ascent`` is the bound. Returns the new pgid (written to the
    in-flight marker so the UI can killpg directly regardless of how the pool
    escalates)."""
    if os.name != "posix" or os.environ.get(ENV_NO_PGRP, "") == "1":
        return None
    os.setpgrp()
    pgid = os.getpgrp()

    def _term(signum, frame):  # noqa: ARG001 -- signal handler signature
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGTERM, _term)
    return pgid


def run_form_job(job: Mapping[str, Any]) -> dict:
    """Worker-process entry (picklable module-level function, safe to
    ``ComputePool.run``): fly the job's target+vehicle into the job's run
    directory via :func:`orbitopt.flightlab.simulate.simulate`, leave the
    form audit file and the driver's ``result.json`` next to the run, and
    return a compact summary. Raises what ``simulate`` raises; the marker is
    removed either way so a failed run never reads as cached."""
    from orbitopt.flightlab.simulate import simulate

    run_dir = Path(job["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = run_dir / IN_PROGRESS
    pgid = _isolate_worker_process_group()
    marker.write_text(str(pgid if pgid is not None else os.getpid()), encoding="ascii")
    vehicle_path = run_dir / "vehicle.yaml"
    vehicle_path.write_text(yaml.safe_dump(job["vehicle"], sort_keys=False), encoding="utf-8")
    try:
        target = FlightTarget.from_dict(job["target"])
        run = simulate(target, vehicle_path, run_dir,
                       refly=bool(job["options"].get("refly", True)),
                       timeout_s=float(job["options"].get("timeout_s", 300.0)))
        (run_dir / "designer-form.json").write_text(_canonical({
            "digest": job.get("digest"), "mission_form": job.get("mission_form"),
            "vehicle_form": job.get("vehicle_form"), "options": job["options"],
        }), encoding="utf-8")
        return {
            "run_dir": str(run_dir),
            "run_id": run.run_id,
            "optimizer_converged": run.verification.optimizer_converged,
            "physics_verified": run.verification.physics_verified,
            "constraints_passed": run.verification.mission_constraints_passed,
            "failed_checks": sorted(c.id for c in run.verification.checks if not c.passed),
        }
    finally:
        marker.unlink(missing_ok=True)


def open_cached_run(digest: str) -> Path | None:
    """Convenience for the UI: cached dir or None (caller then builds and
    submits :func:`build_job`)."""
    return cached_run_dir(digest)


def optimize_available() -> bool:
    """Feature check for the E2 ascent optimizer (FL-502 wires Simulate to
    ``optimize`` behind this check once E2 is merged; until then the form
    says so honestly and only ``simulate`` runs)."""
    import importlib.util

    return importlib.util.find_spec("orbitopt.flightlab.optimize") is not None
