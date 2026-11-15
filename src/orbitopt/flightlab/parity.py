"""E1-01 (task #250): physics-parameter parity across backends.

Two backends that claim to fly "the 6DOF scenario" must disagree only about
fidelity, never about the scenario. This module holds the shared parameter
source (``data/flightlab/parity/*.yaml``, one document per aligned family --
packaged, shipped with the wheel), the native-side builder that consumes it
(:func:`native_stage_pair`, wired into
:func:`orbitopt.flightlab.sixdof.two_stage.f06_two_stage_scenario`), a
JSBSim-side reader that extracts the declared parameters straight out of the
packaged aircraft/engine XML (no ``jsbsim`` import -- the parity check runs
wherever the wheel is installed), and :func:`check_parity`, which asserts
the two sides agree on every aligned parameter.

Alignment is deliberately narrow and disclosed (see the header of
``spike_stage_parity.yaml`` and the fidelity table in docs/flightlab.md):

* masses / propellant / inertias / CG-and-mount geometry / Isp: exact match;
* stage-1 thrust: JSBSim derives ``isp * g0 * propflowmax``, the native
  plant states ``thrust_vacuum_n`` -- agreement within the scenario's
  documented relative band (actual gap ~0.15%);
* aero: shared geometry and reference quantities; the jsbsim *base* aircraft
  fly ZERO aero by design while the native plant flies the packaged
  coefficient file -- a fidelity boundary, asserted (and disclosed), not
  hidden.  The generated production aircraft (``jsbsim.production_aircraft``,
  E1-04) closes that boundary for the production path: its aerodynamics
  section is compiled from the same coefficient file, asserted by
  ``parity-jsbsim-production-aero`` and quantified against the native plant
  in ADR flightlab-0004.

Nothing here flies anything; it only reads, compares and reports.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace as _replace
from functools import lru_cache
from importlib import resources
from pathlib import Path

import yaml

from orbitopt.flightlab.contract import Check
from orbitopt.flightlab.sixdof.aero import load_packaged_aero

#: avoirdupois pound -> kg (exact); JSBSim propflow tables are in LBS/SEC.
_LB_TO_KG = 0.45359237

__all__ = ["ParityScenario", "load_parity_scenario", "native_stage_pair",
           "native_booster", "jsbsim_declared_params", "check_parity",
           "parity_report"]


@dataclass(frozen=True)
class ParityScenario:
    """The loaded, validated shared-parameter document for one aligned
    scenario family. ``native`` / ``jsbsim`` are the document's own
    parameter blocks; the typed helpers expose the values the checker and
    the native builder consume."""

    name: str
    document: dict
    stack: dict
    booster: dict
    upper: dict
    stage1_engine: dict
    stage2_engine: dict
    jsbsim: dict
    aero_model: str
    aero_reference: dict

    @property
    def thrust_rel_tol(self) -> float:
        return float(self.jsbsim.get("thrust_rel_tol", 0.0))


@lru_cache(maxsize=None)
def load_parity_scenario(name: str = "spike_stage_parity") -> ParityScenario:
    """Load a packaged parity document (``data/flightlab/parity/<name>.yaml``)
    and validate the keys the checker requires. Cached -- the file is
    packaged data."""
    path = resources.files("orbitopt").joinpath("data", "flightlab", "parity",
                                                f"{name}.yaml")
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    missing = [k for k in ("name", "stack", "booster", "upper", "stage1_engine",
                           "stage2_engine", "jsbsim", "aero_model", "aero_reference")
               if k not in doc]
    if missing:
        raise ValueError(f"parity scenario {name!r}: missing key(s) {missing}")
    for veh, keys in (("stack", ("name", "dry_mass_kg", "propellant_mass_kg",
                                 "inertia_kg_m2", "engine")),
                      ("booster", ("name", "dry_mass_kg", "propellant_mass_kg",
                                   "inertia_kg_m2", "engine")),
                      ("upper", ("name", "dry_mass_kg", "propellant_mass_kg",
                                 "inertia_kg_m2", "engine"))):
        miss = [k for k in keys if k not in doc[veh]]
        if miss:
            raise ValueError(f"parity scenario {name!r}: {veh}: missing {miss}")
    return ParityScenario(
        name=str(doc["name"]), document=doc, stack=doc["stack"],
        booster=doc["booster"], upper=doc["upper"],
        stage1_engine=doc["stage1_engine"], stage2_engine=doc["stage2_engine"],
        jsbsim=doc["jsbsim"], aero_model=str(doc["aero_model"]),
        aero_reference=doc["aero_reference"])


def native_stage_pair(scenario: ParityScenario | None = None, *,
                      stack_name: str | None = None):
    """Build the aligned stage pair (``StagePair`` from
    :mod:`orbitopt.flightlab.sixdof.two_stage`) exactly as the parity
    document states it: stack / booster share the stage-1 engine and the
    packaged aero model, the upper stage flies the stage-2 engine with no
    aero. This is the ONLY definition of the F-06/F-07 vehicle figures --
    ``f06_two_stage_scenario`` calls it, so "the scenario" and "the parity
    source" cannot drift apart."""
    # Late imports: two_stage imports this module at its top level; the
    # cycle is only ever closed by a call, never by an import.
    from orbitopt.flightlab.sixdof.two_stage import StagePair
    from orbitopt.flightlab.sixdof.rigidbody import RigidBodyEngine, RigidBodyVehicle

    p = scenario or load_parity_scenario()

    def engine(block: dict) -> RigidBodyEngine:
        return RigidBodyEngine(
            thrust_vacuum_n=float(block["thrust_vacuum_n"]),
            isp_vacuum_s=float(block["isp_vacuum_s"]),
            isp_sea_level_s=float(block["isp_sea_level_s"]),
            nozzle_exit_area_m2=float(block["nozzle_exit_area_m2"]),
            throttle_min=float(block["throttle_min"]),
            throttle_max=float(block["throttle_max"]),
            mount_x_cg_m=float(block["mount_x_cg_m"]),
            gimbal_limit_rad=float(block.get("gimbal_limit_rad", 0.0)))

    engines = {"stage1": engine(p.stage1_engine), "stage2": engine(p.stage2_engine)}

    def vehicle(block: dict, with_aero: bool) -> RigidBodyVehicle:
        eng = engines[str(block["engine"])]
        inertia = tuple(float(v) for v in block["inertia_kg_m2"])
        return RigidBodyVehicle(
            name=block["name"] if block is not p.stack or stack_name is None
            else stack_name,
            dry_mass_kg=float(block["dry_mass_kg"]),
            propellant_mass_kg=float(block["propellant_mass_kg"]),
            inertia_kg_m2=inertia, engine=eng,
            aero=load_packaged_aero(p.aero_model) if with_aero else None)

    stack = vehicle(p.stack, with_aero=True)
    booster = vehicle(p.booster, with_aero=True)
    upper = vehicle(p.upper, with_aero=False)
    return StagePair(stack=stack, booster=booster, upper=upper,
                     separation_impulse_ns=float(p.document["separation_impulse_ns"]))


def _jsbsim_data(*parts: str) -> Path:
    return Path(str(resources.files("orbitopt").joinpath("data", "flightlab",
                                                         "jsbsim", *parts)))


def _text(root: ET.Element, path: str) -> str:
    node = root.find(path)
    if node is None or node.text is None:
        raise ValueError(f"JSBSim XML: element {path!r} not found")
    return node.text.strip()


def _aerodynamics_declared(ac: ET.Element) -> bool:
    """Does this aircraft declare *non-zero* aerodynamics?

    An ``<aerodynamics>`` section full of zeros is a placeholder, not aero: the
    E4 spike declares one and flies ballistically. What counts as aero is a
    tabulated coefficient, a non-zero constant, or a coefficient wired to a
    flight-state property."""
    root = ac.find('aerodynamics')
    if root is None:
        return False
    if root.findall('.//table'):
        return True
    for node in root.iter('value'):
        if node.text and abs(float(node.text)) > 0.0:
            return True
    return bool(root.findall('.//property'))


def _kg(node: ET.Element) -> float:
    unit = (node.get("unit") or "LBS").upper()
    value = float(node.text)
    if unit == "KG":
        return value
    if unit == "LBS":
        return value * _LB_TO_KG
    raise ValueError(f"JSBSim XML: unsupported mass unit {unit!r}")


def native_booster(scenario: ParityScenario | None = None, *,
                   name: str | None = None):
    """The stage-1 booster alone: the parity document's booster vehicle
    (masses, inertias, stage-1 engine, packaged aero) as a
    ``RigidBodyVehicle`` -- what the F-05 single-stage studies fly. ``name``
    renames it without touching a single number."""
    p = scenario or load_parity_scenario()
    booster = native_stage_pair(p).booster
    return booster if name is None else _replace(booster, name=name)


def jsbsim_declared_params(aircraft: str = "spike_stage",
                           engine: str = "spike_engine") -> dict:
    """Read the parameters the packaged JSBSim aircraft/engine XMLs declare:
    dry mass, propellant contents, inertia, CG and thruster stations, Isp,
    mass flow, minimum throttle, and whether the model has ANY aerodynamics
    at all. Pure XML parsing -- runs without the optional ``jsbsim``
    package, so the parity gate is enforced in every environment."""
    ac_path = _jsbsim_data("aircraft", aircraft, f"{aircraft}.xml")
    en_path = _jsbsim_data("engine", f"{engine}.xml")
    ac, en = ET.parse(ac_path).getroot(), ET.parse(en_path).getroot()

    tanks = [t for t in ac.findall("./propulsion/tank")]
    propellant_kg = sum(_kg(t.find("capacity")) for t in tanks)
    engine_files = [e.get("file") for e in ac.findall("./propulsion/engine")]
    if engine not in engine_files:
        raise ValueError(f"{aircraft}.xml: engine {engine!r} not referenced "
                         f"(found {engine_files})")
    thruster = ac.find("./propulsion/engine/thruster/location")
    isp_node = en.find("isp")
    flow_node = en.find("propflowmax")
    flow_unit = (flow_node.get("unit") or "LBS/SEC").upper()
    flow_kg_s = float(flow_node.text) * _LB_TO_KG if "LBS" in flow_unit \
        else float(flow_node.text)
    act = ac.findall("./flight_control/channel/actuator/clipto/max")
    gimbals = [float(a.text) for a in act]
    return {
        "aircraft": aircraft,
        "engine": engine,
        "empty_mass_kg": _kg(ac.find("./mass_balance/emptywt")),
        "propellant_mass_kg": propellant_kg,
        "inertia_kg_m2": tuple(float(_text(ac, f"./mass_balance/{axis}"))
                               for axis in ("ixx", "iyy", "izz")),
        "cg_x_m": float(_text(ac, './mass_balance/location[@name="CG"]/x')),
        "thruster_x_m": float(thruster.find("x").text),
        "isp_s": float(isp_node.text),
        "propflowmax_kg_s": flow_kg_s,
        "min_throttle": float(_text(en, "minthrottle")),
        "gimbal_limit_rad": max(gimbals) if gimbals else 0.0,
        "aero_declared": _aerodynamics_declared(ac),
        "wingarea_m2": float(_text(ac, "./metrics/wingarea")),
        "wingspan_m": float(_text(ac, "./metrics/wingspan")),
        "chord_m": float(_text(ac, "./metrics/chord")),
    }


def _rel(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-30)
    return abs(a - b) / denom


def _close(a: float, b: float, tol: float = 0.0) -> bool:
    return _rel(a, b) <= tol


def check_parity(scenario: ParityScenario | None = None) -> tuple[Check, ...]:
    """Assert the aligned scenario agrees across the backends and return
    the checks (``Check`` from the Flight Lab contract, so a caller can
    publish them like any other verdict). Every check is honest: value is
    the measured discrepancy, limit the allowed one. ``check_parity`` never
    raises on a failing check -- callers gate on ``all(c.passed)``."""
    p = scenario or load_parity_scenario()
    pair = native_stage_pair(p)
    jb = jsbsim_declared_params(p.document.get("jsbsim_aircraft", "spike_stage"),
                                p.document.get("jsbsim_engine", "spike_engine"))
    g0 = float(p.jsbsim.get("g0_m_s2", 9.80665))
    native_engine = pair.stack.engine
    checks: list[Check] = []

    def add(check_id: str, passed: bool, value=None, limit=None, unit="", note=""):
        checks.append(Check(id=check_id, passed=bool(passed), value=value,
                            limit=limit, unit=unit, reason_code="" if passed else note))

    # -- mass bookkeeping (exact) ------------------------------------------
    dry_res = pair.stack.dry_mass_kg - (pair.booster.dry_mass_kg
                                        + pair.upper.dry_mass_kg
                                        + pair.upper.propellant_mass_kg)
    prop_res = pair.stack.propellant_mass_kg - pair.booster.propellant_mass_kg
    add("parity-stack-mass-closure", abs(dry_res) < 1e-9 and abs(prop_res) < 1e-9,
        value=max(abs(dry_res), abs(prop_res)), limit=1e-9, unit="kg",
        note="native stage split does not close on the stack from the parity source")
    m_ok = (_close(pair.booster.dry_mass_kg, jb["empty_mass_kg"])
            and _close(pair.booster.propellant_mass_kg, jb["propellant_mass_kg"]))
    add("parity-booster-mass-vs-jsbsim", m_ok,
        value=max(_rel(pair.booster.dry_mass_kg, jb["empty_mass_kg"]),
                  _rel(pair.booster.propellant_mass_kg, jb["propellant_mass_kg"])),
        limit=0.0, unit="",
        note="native booster dry/wet mass differs from spike_stage.xml")

    # -- inertia (exact per axis) ------------------------------------------
    i_res = max(_rel(n, x) for n, x in zip(pair.booster.inertia_kg_m2,
                                           jb["inertia_kg_m2"]))
    add("parity-booster-inertia-vs-jsbsim", i_res == 0.0, value=i_res, limit=0.0,
        unit="", note="native booster inertia differs from spike_stage.xml")

    # -- mount station: native CG-relative, JSBSim nose-relative -----------
    mount_jb = jb["thruster_x_m"] - jb["cg_x_m"]
    add("parity-engine-mount-vs-jsbsim",
        _close(native_engine.mount_x_cg_m, mount_jb),
        value=_rel(native_engine.mount_x_cg_m, mount_jb), limit=0.0, unit="",
        note="engine mount station differs (native mount_x_cg_m vs XML "
             "thruster-x minus CG-x)")

    # -- Isp exact; thrust within the documented derived-value band --------
    add("parity-isp-vs-jsbsim", native_engine.isp_vacuum_s == jb["isp_s"],
        value=_rel(native_engine.isp_vacuum_s, jb["isp_s"]), limit=0.0, unit="",
        note="vacuum Isp differs between the parity source and spike_engine.xml")
    thrust_derived = jb["isp_s"] * g0 * jb["propflowmax_kg_s"]
    thrust_gap = _rel(native_engine.thrust_vacuum_n, thrust_derived)
    add("parity-stage1-thrust-vs-jsbsim", thrust_gap <= p.thrust_rel_tol,
        value=thrust_gap, limit=p.thrust_rel_tol, unit="",
        note=f"native thrust_vacuum_n {native_engine.thrust_vacuum_n:.1f} N vs "
             f"JSBSim derived isp*g0*mdot {thrust_derived:.1f} N exceeds the "
             f"declared band {p.thrust_rel_tol:g}")
    flow_gap = _rel(jb["propflowmax_kg_s"], float(p.jsbsim["propflowmax_kg_s"]))
    add("parity-jsbsim-flow-vs-source", flow_gap <= 1e-5, value=flow_gap,
        limit=1e-5, unit="",
        note="spike_engine.xml propflowmax moved away from the parity source")
    add("parity-jsbsim-min-throttle-disclosed",
        jb["min_throttle"] <= native_engine.throttle_min,
        value=jb["min_throttle"], limit=native_engine.throttle_min, unit="",
        note="JSBSim minthrottle is ABOVE the native throttle floor: the "
             "control-law difference is no longer in the disclosed direction")

    # -- aero: shared geometry; JSBSim aero stays identically zero ---------
    aero = pair.booster.aero
    want = p.aero_reference
    geom_ok = (_close(aero.reference_span_m, float(want["span_m"]), 1e-9)
               and _close(aero.reference_length_m, float(want["length_m"]), 1e-9)
               and _close(aero.reference_area_m2, math.pi * float(want["diameter_m"]) ** 2 / 4,
                          1e-4))
    add("parity-aero-geometry", geom_ok, unit="",
        note="packaged aero reference quantities no longer match the parity "
             "source (diameter/span/length/frontal area)")
    zero_ok = (not jb["aero_declared"]) and bool(p.jsbsim.get("aero_zero", True))
    add("parity-jsbsim-aero-zero", zero_ok, unit="",
        note="spike_stage.xml now declares aerodynamics: the zero-aero "
             "fidelity boundary of the jsbsim packaged vehicles has changed "
             "-- update docs and this parity source")

    # -- the production (aero-bearing) JSBSim vehicle, E1-04 (#253) --------
    production = p.jsbsim.get("production_aircraft")
    if production:
        pr = jsbsim_declared_params(str(production), p.document.get("jsbsim_engine",
                                                                    "spike_engine"))
        shared = ("empty_mass_kg", "propellant_mass_kg", "cg_x_m",
                  "thruster_x_m", "isp_s", "propflowmax_kg_s", "min_throttle",
                  "gimbal_limit_rad")
        inherited = max([_rel(jb[k], pr[k]) for k in shared]
                        + [_rel(n, x) for n, x in zip(jb["inertia_kg_m2"],
                                                     pr["inertia_kg_m2"])])
        add("parity-jsbsim-production-inherited", inherited == 0.0, value=inherited,
            limit=0.0, unit="",
            note=f"{production}.xml no longer carries the same vehicle parameters "
                 "as the airframe it is generated from")
        add("parity-jsbsim-production-aero", pr["aero_declared"], unit="",
            note=f"{production}.xml declares no (or zero) aerodynamics: the "
                 "production JSBSim path is not aero-bearing")
        prod_geom = (_close(aero.reference_area_m2, pr["wingarea_m2"], 1e-4)
                     and _close(aero.reference_span_m, pr["wingspan_m"], 1e-9)
                     and _close(aero.reference_length_m, pr["chord_m"], 1e-9))
        add("parity-jsbsim-production-aero-geometry", prod_geom, unit="",
            note=f"{production}.xml reference geometry no longer matches the "
                 "aero document the native plant flies")
    return tuple(checks)


def parity_report(scenario: ParityScenario | None = None) -> dict:
    """The parity verdict as a small JSON-friendly document: the checks, the
    parameters each side declared, and the fidelity notes the document
    discloses. Used by the parity test and printable from a REPL."""
    p = scenario or load_parity_scenario()
    checks = check_parity(p)
    jb = jsbsim_declared_params(p.document.get("jsbsim_aircraft", "spike_stage"),
                                p.document.get("jsbsim_engine", "spike_engine"))
    pair = native_stage_pair(p)
    return {
        "scenario": p.name,
        "all_passed": all(c.passed for c in checks),
        "checks": [{"id": c.id, "passed": c.passed, "value": c.value,
                    "limit": c.limit, "unit": c.unit} for c in checks],
        "native": {
            "stack": {"dry_mass_kg": pair.stack.dry_mass_kg,
                      "propellant_mass_kg": pair.stack.propellant_mass_kg,
                      "inertia_kg_m2": list(pair.stack.inertia_kg_m2)},
            "booster": {"dry_mass_kg": pair.booster.dry_mass_kg,
                        "propellant_mass_kg": pair.booster.propellant_mass_kg,
                        "inertia_kg_m2": list(pair.booster.inertia_kg_m2)},
            "upper": {"dry_mass_kg": pair.upper.dry_mass_kg,
                      "propellant_mass_kg": pair.upper.propellant_mass_kg,
                      "inertia_kg_m2": list(pair.upper.inertia_kg_m2)},
            "stage1_engine": {"thrust_vacuum_n": pair.stack.engine.thrust_vacuum_n,
                              "isp_vacuum_s": pair.stack.engine.isp_vacuum_s,
                              "mount_x_cg_m": pair.stack.engine.mount_x_cg_m},
        },
        "jsbsim": {k: jb[k] for k in ("aircraft", "engine", "empty_mass_kg",
                                      "propellant_mass_kg", "inertia_kg_m2",
                                      "cg_x_m", "thruster_x_m", "isp_s",
                                      "propflowmax_kg_s", "min_throttle",
                                      "aero_declared")},
        "fidelity_notes": {
            "jsbsim": "zero aero by design; thrust derived from isp*g0*"
                      "propflowmax; minthrottle 0.0 vs the native 0.4 floor",
            "native": "coefficient-table aero (packaged spike_stage model)",
            "thrust_rel_tol": p.thrust_rel_tol,
        },
    }
