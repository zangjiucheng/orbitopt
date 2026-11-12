"""Compare two Flight Lab runs + recovery summaries (FL-503), Qt-free.

The Compare Runs view (:mod:`orbitopt.flightlab.ui.compare_page`) and the
replay page's recovery panel are thin views over this module, which only
reads what a run directory already contains -- it never flies anything:

* :func:`load_bundle` -- a FlightResult run directory plus the side files
  other Flight Lab steps leave next to it: ``vehicle.yaml`` (designer /
  optimizer), ``designer-form.json`` (FL-502), ``optimization.json`` (FL-202)
  and ``montecarlo.json`` (FL-304).
* :func:`run_inputs` / :func:`run_outputs` -- flat ``key -> Metric`` dicts of
  what went in (vehicle, guidance, mission) and what came out (payload,
  insertion errors, check margins, booster propellant at touchdown, landing
  errors, verdicts).
* :func:`compare_runs` -- the row-by-row diff of two bundles plus plain
  sentences for the trade-offs that matter (e.g. "recovery reserve cost X kg
  payload"), and both runs' failure reasons via the reason catalog.
* :func:`recovery_summary` -- touchdown checks + the Monte Carlo summary
  (success rate, dispersion ellipse, failure histogram) when present.
* :func:`trajectory_profiles` / :func:`overlay_channels` -- the overlay data
  for the trajectory and telemetry plots.

Nothing is guessed: a quantity the run does not record is ``None`` and the
views print it as missing.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from orbitopt.flightlab.contract import Check, FlightRun, read_run
from orbitopt.flightlab.reasons import ReasonEntry, explain, explain_all

__all__ = [
    "CompareResult",
    "DiffRow",
    "Metric",
    "RecoverySummary",
    "RunBundle",
    "check_margin",
    "compare_runs",
    "discover_runs",
    "ellipse_polygon",
    "failure_explanations",
    "find_montecarlo",
    "load_bundle",
    "load_side_files",
    "bundle_from_run",
    "overlay_channels",
    "recovery_kind",
    "recovery_summary",
    "run_inputs",
    "run_outputs",
    "trajectory_profiles",
]

EARTH_RADIUS_M = 6378137.0

#: Touchdown checks (FL-303) and the FL-304 recovery gate, in display order.
TOUCHDOWN_CHECK_IDS = (
    "landing-position",
    "landing-vertical-speed",
    "landing-lateral-speed",
    "landing-tilt",
    "landing-propellant-reserve",
    "landing-max-g",
)
RECOVERY_CHECK_IDS = ("recovery-landing", *TOUCHDOWN_CHECK_IDS, "recovery-propellant-reserve")

#: Reason codes of recovery checks that only say the recovery was declared
#: but never flown (checks.evaluate adds them for a declared recovery with no
#: booster result) -- they carry no landing outcome.
_NOT_FLOWN_CODES = frozenset({"RECOVERY_LANDING_NOT_ASSESSED",
                              "RECOVERY_PROPELLANT_RESERVE_DATA_MISSING"})

#: Pass direction per check id: "min" = value must be >= limit, "max" = value
#: must be <= limit, "target" = value should equal limit within a tolerance.
_CHECK_SENSE: dict[str, str] = {
    "liftoff-thrust-to-weight": "min",
    "propellant-reserve": "min",
    "recovery-propellant-reserve": "min",
    "insertion-above-atmosphere": "min",
    "insertion-periapsis-above-minimum": "min",
    "landing-propellant-reserve": "min",
    "max-q-within-limit": "max",
    "max-accel-within-limit": "max",
    "landing-position": "max",
    "landing-vertical-speed": "max",
    "landing-lateral-speed": "max",
    "landing-tilt": "max",
    "landing-max-g": "max",
    "insertion-periapsis-within-tolerance": "target",
    "insertion-apoapsis-within-tolerance": "target",
    "insertion-inclination-within-tolerance": "target",
    "insertion-raan-within-tolerance": "target",
}

_CHECK_LABELS: dict[str, str] = {
    "liftoff-thrust-to-weight": "Liftoff T/W",
    "propellant-reserve": "Upper-stage reserve",
    "recovery-propellant-reserve": "Booster recovery reserve",
    "insertion-above-atmosphere": "Periapsis above atmosphere",
    "insertion-periapsis-above-minimum": "Periapsis above minimum",
    "landing-propellant-reserve": "Landing propellant reserve",
    "max-q-within-limit": "Max-Q",
    "max-accel-within-limit": "Max g",
    "landing-position": "Landing position error",
    "landing-vertical-speed": "Touchdown vertical speed",
    "landing-lateral-speed": "Touchdown lateral speed",
    "landing-tilt": "Landing tilt",
    "landing-max-g": "Landing burn max g",
    "insertion-periapsis-within-tolerance": "Periapsis error",
    "insertion-apoapsis-within-tolerance": "Apoapsis error",
    "insertion-inclination-within-tolerance": "Inclination error",
    "insertion-raan-within-tolerance": "RAAN error",
    "recovery-landing": "Recovery landed",
}


# --------------------------------------------------------------------------- loading


@dataclass(frozen=True)
class RunBundle:
    """A run plus the optional side files found next to it."""

    run_dir: Path
    run: FlightRun
    vehicle: Mapping[str, Any] | None = None
    designer_form: Mapping[str, Any] | None = None
    optimization: Mapping[str, Any] | None = None
    montecarlo: Mapping[str, Any] | None = None
    montecarlo_path: Path | None = None

    @property
    def label(self) -> str:
        return self.run.run_id or self.run_dir.name


def _read_json(path: Path) -> Mapping[str, Any] | None:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, Mapping) else None


def _read_yaml(path: Path) -> Mapping[str, Any] | None:
    try:
        import yaml

        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 -- an unreadable side file is "absent", never a crash
        return None
    return doc if isinstance(doc, Mapping) else None


def find_montecarlo(run_dir: Path | str) -> Path | None:
    """``montecarlo.json`` for a run: inside the run directory, or -- the
    FL-304 CLI layout ``-o X.json`` + ascent run ``X.ascent/`` -- next to it."""
    run_dir = Path(run_dir)
    candidates = [run_dir / "montecarlo.json"]
    if run_dir.suffix == ".ascent":
        candidates.append(run_dir.with_suffix(".json"))
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_side_files(run_dir: Path | str) -> dict[str, Any]:
    """The optional side files of a run directory, as :class:`RunBundle`
    keyword arguments (absent / unreadable files are simply left out)."""
    run_dir = Path(run_dir).expanduser()
    side: dict[str, Any] = {}
    for cand in (run_dir / "vehicle.yaml", run_dir / "vehicle" / "vehicle.yaml"):
        if cand.is_file():
            vehicle = _read_yaml(cand)
            if vehicle is not None:
                side["vehicle"] = vehicle
                break
    for key, name in (("designer_form", "designer-form.json"), ("optimization", "optimization.json")):
        doc = _read_json(run_dir / name)
        if doc is not None:
            side[key] = doc
    mc_path = find_montecarlo(run_dir)
    mc = _read_json(mc_path) if mc_path else None
    if mc is not None:
        side["montecarlo"] = mc
        side["montecarlo_path"] = mc_path
    return side


def load_bundle(run_dir: Path | str) -> RunBundle:
    """Read a run directory (raises what :func:`read_run` raises) and pick
    up whichever side files exist."""
    run_dir = Path(run_dir).expanduser()
    return RunBundle(run_dir=run_dir, run=read_run(run_dir), **load_side_files(run_dir))


def bundle_from_run(run: FlightRun, run_dir: Path | str = ".", **side: Any) -> RunBundle:
    """A bundle for an in-memory run (tests, callers that already read it)."""
    return RunBundle(run_dir=Path(run_dir), run=run, **side)


# --------------------------------------------------------------------------- metrics


@dataclass(frozen=True)
class Metric:
    """One comparable quantity. ``value`` is a float, a string, or None
    (not recorded); ``better`` says which direction is an improvement for
    numeric deltas (``"higher"``, ``"lower"``, ``"closer"`` to zero, or "")."""

    section: str
    label: str
    value: float | str | None
    unit: str = ""
    better: str = ""


def check_margin(check: Check) -> float | None:
    """Signed margin to the limit, positive = inside the limit, for min/max
    checks; for target checks the signed error ``value - limit`` (RAAN
    wrapped to +-180 deg). None when value or limit is missing or the check
    id has no known direction."""
    if check.value is None or check.limit is None:
        return None
    sense = _check_sense(check.id)
    value, limit = float(check.value), float(check.limit)
    if sense == "min":
        return value - limit
    if sense == "max":
        return limit - value
    if sense == "target":
        err = value - limit
        if check.id == "insertion-raan-within-tolerance":
            err = (err + 180.0) % 360.0 - 180.0
        return err
    return None


def _check_sense(check_id: str) -> str:
    if check_id.startswith("orbit-refly-") and check_id != "orbit-refly-not-assessed":
        return "max"
    return _CHECK_SENSE.get(check_id, "")


def _check_label(check_id: str) -> str:
    return _CHECK_LABELS.get(check_id, check_id)


def recovery_kind(bundle: RunBundle) -> str:
    """``none`` / ``rtls`` / ``downrange`` from the designer form when it was
    recorded, else ``flown`` when the run carries a recovery outcome (a
    recovery check with a result, or a touchdown event), ``declared`` when
    it only carries the "not assessed" placeholders of a recovery that was
    never flown, else ``none``."""
    form = (bundle.designer_form or {}).get("mission_form") or {}
    kind = form.get("recovery")
    if kind:
        return str(kind)
    checks = [c for c in bundle.run.verification.checks if c.id in RECOVERY_CHECK_IDS]
    if any(c.passed or c.reason_code not in _NOT_FLOWN_CODES for c in checks):
        return "flown"
    if checks:
        return "declared"
    return "flown" if any(e.kind == "touchdown" for e in bundle.run.events) else "none"


def has_recovery(bundle: RunBundle) -> bool:
    return recovery_kind(bundle) != "none"


def _fnum(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _thrust_scalar(thrust: Any) -> float | None:
    if isinstance(thrust, (int, float)):
        return float(thrust)
    if isinstance(thrust, list) and thrust:
        try:
            return max(float(p[1]) for p in thrust)
        except (TypeError, ValueError, IndexError):
            return None
    return None


def run_inputs(bundle: RunBundle) -> dict[str, Metric]:
    """What went into the run: vehicle (per-stage masses / thrust / Isp /
    burn), guidance (attitude program, optimizer decision vector), mission
    (designer form fields)."""
    out: dict[str, Metric] = {}
    vehicle = bundle.vehicle or {}
    if vehicle:
        out["vehicle.name"] = Metric("Vehicle", "Vehicle", str(vehicle.get("name", "")))
        out["vehicle.payload_kg"] = Metric("Vehicle", "Payload (vehicle file)",
                                           _fnum(vehicle.get("payload_kg")), "kg", "higher")
        for i, stage in enumerate(vehicle.get("stages") or [], start=1):
            eng = stage.get("engine") or {}
            wet, dry = _fnum(stage.get("mass_initial_kg")), _fnum(stage.get("mass_dry_kg"))
            out[f"vehicle.s{i}.wet_kg"] = Metric("Vehicle", f"Stage {i} wet mass", wet, "kg")
            out[f"vehicle.s{i}.dry_kg"] = Metric("Vehicle", f"Stage {i} dry mass", dry, "kg", "lower")
            out[f"vehicle.s{i}.prop_kg"] = Metric(
                "Vehicle", f"Stage {i} propellant",
                wet - dry if wet is not None and dry is not None else None, "kg")
            thrust = _thrust_scalar(eng.get("thrust_vac_n"))
            out[f"vehicle.s{i}.thrust_kn"] = Metric("Vehicle", f"Stage {i} max vac thrust",
                                                    thrust / 1000.0 if thrust is not None else None, "kN")
            out[f"vehicle.s{i}.isp_s"] = Metric("Vehicle", f"Stage {i} Isp (vac)",
                                                _fnum(eng.get("isp_vac_s")), "s", "higher")
            out[f"vehicle.s{i}.burn_end_s"] = Metric("Vehicle", f"Stage {i} burn end",
                                                     _fnum(eng.get("burn_end_s")), "s")
            if "separation_time_s" in stage:
                out[f"vehicle.s{i}.sep_s"] = Metric("Vehicle", f"Stage {i} separation",
                                                    _fnum(stage.get("separation_time_s")), "s")
        program = vehicle.get("attitude_program") or []
        if program:
            out["guidance.program_points"] = Metric("Guidance", "Attitude program points",
                                                    float(len(program)))
            first = program[0]
            out["guidance.azimuth_deg"] = Metric("Guidance", "Launch azimuth",
                                                 _fnum(first.get("azimuth_deg")), "deg")
            kick = next((p for p in program if _fnum(p.get("elevation_deg")) not in (None, 90.0)), None)
            if kick is not None:
                out["guidance.pitch_kick_s"] = Metric("Guidance", "Pitch-over start",
                                                      _fnum(kick.get("t_s")), "s")
            for p in program:
                t = _fnum(p.get("t_s"))
                if t is None:
                    continue
                out[f"guidance.elev@{t:g}"] = Metric("Guidance", f"Elevation @ T+{t:g} s",
                                                     _fnum(p.get("elevation_deg")), "deg")
    opt = bundle.optimization or {}
    for key, value in sorted((opt.get("decision_vector") or {}).items()):
        out[f"guidance.x.{key}"] = Metric("Guidance", f"Optimizer {key}", _fnum(value))
    if opt:
        out["guidance.objective"] = Metric("Guidance", "Optimizer objective",
                                           str(opt.get("objective", "")))
    form = (bundle.designer_form or {}).get("mission_form") or {}
    for key, value in sorted(form.items()):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[f"mission.{key}"] = Metric("Mission", key.replace("_", " "), float(value))
        elif value is not None and not isinstance(value, (list, dict)):
            out[f"mission.{key}"] = Metric("Mission", key.replace("_", " "), str(value))
    out["mission.recovery"] = Metric("Mission", "Recovery", recovery_kind(bundle))
    return out


def _payload_kg(bundle: RunBundle) -> float | None:
    opt = bundle.optimization or {}
    if _fnum(opt.get("payload_kg")) is not None:
        return _fnum(opt.get("payload_kg"))
    vehicle = bundle.vehicle or {}
    return _fnum(vehicle.get("payload_kg"))


def _booster_track(run: FlightRun):
    for body_id in ("booster", "stage1_debris"):
        track = next((t for t in run.tracks if t.body_id == body_id), None)
        if track is not None:
            return track
    return next((t for t in run.tracks
                 if t.parent_body_id == "stage1" and t.body_id.endswith("_debris")), None)


def _verdict_text(value: bool | None) -> str:
    return "PASS" if value is True else "FAIL" if value is False else "N/A"


def run_outputs(bundle: RunBundle) -> dict[str, Metric]:
    """What came out: verdicts, payload, insertion orbit and its errors,
    every numeric check's margin, booster propellant at touchdown and the
    landing errors."""
    run = bundle.run
    ver = run.verification
    out: dict[str, Metric] = {
        "verdict.mission": Metric("Verdict", "Mission constraints", _verdict_text(ver.mission_constraints_passed)),
        "verdict.physics": Metric("Verdict", "Physics verified", _verdict_text(ver.physics_verified)),
        "verdict.optimizer": Metric("Verdict", "Optimizer converged", _verdict_text(ver.optimizer_converged)),
        "outcome.payload_kg": Metric("Outcome", "Payload", _payload_kg(bundle), "kg", "higher"),
    }
    try:
        from orbitopt.flightlab.checks import insertion_state

        ins = insertion_state(run)
        out["outcome.insertion_mass_kg"] = Metric("Outcome", "Upper-stage mass at insertion",
                                                  _fnum(ins["mass_kg"]), "kg", "higher")
        out["outcome.insertion_t_s"] = Metric("Outcome", "Insertion time", _fnum(ins["t_s"]), "s")
    except (ValueError, IndexError):
        pass
    by_id = {c.id: c for c in ver.checks}
    for key, cid, label in (
        ("insertion.periapsis_km", "insertion-periapsis-within-tolerance", "Periapsis altitude"),
        ("insertion.apoapsis_km", "insertion-apoapsis-within-tolerance", "Apoapsis altitude"),
    ):
        c = by_id.get(cid)
        out[key] = Metric("Insertion", label,
                          float(c.value) / 1000.0 if c and c.value is not None else None, "km")
    for key, cid, label in (
        ("insertion.inclination_deg", "insertion-inclination-within-tolerance", "Inclination"),
        ("insertion.raan_deg", "insertion-raan-within-tolerance", "RAAN"),
    ):
        c = by_id.get(cid)
        if c is not None:
            out[key] = Metric("Insertion", label, _fnum(c.value), "deg")
    for c in ver.checks:
        sense = _check_sense(c.id)
        margin = check_margin(c)
        if not sense:
            continue
        unit = c.unit
        scale = 1.0
        if unit == "m" and c.id.startswith("insertion-"):
            unit, scale = "km", 0.001
        elif unit == "Pa":
            unit, scale = "kPa", 0.001
        section = ("Landing" if c.id.startswith("landing-") or c.id.startswith("recovery-")
                   else "Insertion" if c.id.startswith("insertion-") else "Margins")
        if sense == "target":
            out[f"error.{c.id}"] = Metric(section, _check_label(c.id),
                                          margin * scale if margin is not None else None, unit, "closer")
        else:
            out[f"margin.{c.id}"] = Metric(section, f"{_check_label(c.id)} margin",
                                           margin * scale if margin is not None else None, unit, "higher")
            out[f"value.{c.id}"] = Metric(section, _check_label(c.id),
                                          _fnum(c.value) * scale if _fnum(c.value) is not None else None,
                                          unit)
    # Booster propellant at touchdown: landing check (FL-303) first, else the
    # recovery-reserve check, else nothing (never a guess from the debris arc).
    prop = None
    for cid in ("landing-propellant-reserve", "recovery-propellant-reserve"):
        c = by_id.get(cid)
        if c is not None and c.value is not None:
            prop = float(c.value)
            break
    out["recovery.booster_prop_kg"] = Metric("Landing", "Booster propellant at touchdown",
                                             prop, "kg", "higher")
    gate = by_id.get("recovery-landing")
    out["recovery.gate"] = Metric("Landing", "Recovery verdict",
                                  None if gate is None else ("LANDED" if gate.passed else "FAILED"))
    touchdown = next((e for e in run.events if e.kind == "touchdown"), None)
    if touchdown is not None:
        out["recovery.touchdown_t_s"] = Metric("Landing", "Touchdown time", float(touchdown.t_s), "s")
    mc = bundle.montecarlo or {}
    if mc:
        out["recovery.mc_success"] = Metric("Landing", "Monte Carlo success rate",
                                            _fnum(mc.get("success_rate")) * 100.0
                                            if _fnum(mc.get("success_rate")) is not None else None,
                                            "%", "higher")
    out["outcome.failed_reasons"] = Metric(
        "Verdict", "Failure reasons",
        ", ".join(e.title_en for e in failure_explanations(run)) or "none")
    return out


# --------------------------------------------------------------------------- diff


@dataclass(frozen=True)
class DiffRow:
    key: str
    section: str
    label: str
    unit: str
    a: float | str | None
    b: float | str | None
    delta: float | None
    changed: bool
    #: "better" / "worse" / "" for B relative to A (numeric rows with a direction).
    trend: str = ""


@dataclass(frozen=True)
class CompareResult:
    a: RunBundle
    b: RunBundle
    inputs: tuple[DiffRow, ...]
    outputs: tuple[DiffRow, ...]
    headlines: tuple[str, ...]
    reasons_a: tuple[ReasonEntry, ...] = field(default_factory=tuple)
    reasons_b: tuple[ReasonEntry, ...] = field(default_factory=tuple)

    @property
    def changed_inputs(self) -> tuple[DiffRow, ...]:
        return tuple(r for r in self.inputs if r.changed)


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
    return a == b


def _trend(better: str, a: Any, b: Any) -> str:
    if not isinstance(a, float) or not isinstance(b, float) or _same(a, b):
        return ""
    if better == "higher":
        return "better" if b > a else "worse"
    if better == "lower":
        return "better" if b < a else "worse"
    if better == "closer":
        return "better" if abs(b) < abs(a) else "worse"
    return ""


def _diff(ma: Mapping[str, Metric], mb: Mapping[str, Metric]) -> tuple[DiffRow, ...]:
    keys = list(ma)
    keys += [k for k in mb if k not in ma]
    rows: list[DiffRow] = []
    for key in keys:
        x, y = ma.get(key), mb.get(key)
        ref = x or y
        av = x.value if x else None
        bv = y.value if y else None
        delta = (bv - av) if isinstance(av, float) and isinstance(bv, float) else None
        rows.append(DiffRow(key, ref.section, ref.label, ref.unit, av, bv, delta,
                            not _same(av, bv), _trend(ref.better, av, bv)))
    return tuple(rows)


def failure_explanations(run: FlightRun) -> tuple[ReasonEntry, ...]:
    """Catalog entries for every failing check's reason code, in check order."""
    return explain_all(c.reason_code for c in run.verification.checks
                       if not c.passed and c.reason_code)


def _fmt_kg(value: float) -> str:
    return f"{abs(value):,.0f} kg"


def _headlines(a: RunBundle, b: RunBundle, oa: Mapping[str, Metric],
               ob: Mapping[str, Metric]) -> list[str]:
    lines: list[str] = []
    rec_a, rec_b = recovery_kind(a), recovery_kind(b)
    pa, pb = oa["outcome.payload_kg"].value, ob["outcome.payload_kg"].value
    ma = oa.get("outcome.insertion_mass_kg", Metric("", "", None)).value
    mb = ob.get("outcome.insertion_mass_kg", Metric("", "", None)).value
    if (rec_a == "none") != (rec_b == "none"):
        ascent, recov = (a, b) if rec_a == "none" else (b, a)
        p_asc, p_rec = (pa, pb) if rec_a == "none" else (pb, pa)
        m_asc, m_rec = (ma, mb) if rec_a == "none" else (mb, ma)
        tag = "A" if recov is a else "B"
        if isinstance(p_asc, float) and isinstance(p_rec, float):
            cost = p_asc - p_rec
            verb = "cost" if cost >= 0 else "gained"
            lines.append(f"Recovery reserve {verb} {_fmt_kg(cost)} payload "
                         f"(run {tag} recovers the booster: {p_rec:,.0f} kg vs {p_asc:,.0f} kg expendable).")
        elif isinstance(m_asc, float) and isinstance(m_rec, float):
            cost = m_asc - m_rec
            lines.append(f"Recovery reserve cost {_fmt_kg(cost)} of upper-stage mass at insertion "
                         f"(run {tag} recovers the booster; no payload figure recorded, "
                         "insertion mass used as the payload proxy).")
        else:
            lines.append(f"Run {tag} recovers the booster; the payload cost cannot be computed "
                         "(neither payload nor insertion mass recorded for both runs).")
    elif isinstance(pa, float) and isinstance(pb, float) and not _same(pa, pb):
        lines.append(f"Payload {'up' if pb > pa else 'down'} {_fmt_kg(pb - pa)} from A to B "
                     f"({pa:,.0f} → {pb:,.0f} kg).")
    elif isinstance(ma, float) and isinstance(mb, float) and not math.isclose(ma, mb, abs_tol=0.5):
        lines.append(f"Upper-stage mass at insertion {'up' if mb > ma else 'down'} "
                     f"{_fmt_kg(mb - ma)} from A to B.")
    prop_a = oa["recovery.booster_prop_kg"].value
    prop_b = ob["recovery.booster_prop_kg"].value
    # "at touchdown" only where the booster touched down: the value is the
    # reserve check's propellant left, which a booster that impacted (or
    # never flew home) also has.
    landed_a = oa["recovery.gate"].value == "LANDED"
    landed_b = ob["recovery.gate"].value == "LANDED"
    if isinstance(prop_a, float) and isinstance(prop_b, float):
        if landed_a and landed_b:
            lines.append(f"Booster propellant at touchdown: {prop_a:,.0f} kg (A) vs {prop_b:,.0f} kg (B).")
        else:
            lines.append(f"Booster propellant left: {prop_a:,.0f} kg (A{'' if landed_a else ', no touchdown'}) "
                         f"vs {prop_b:,.0f} kg (B{'' if landed_b else ', no touchdown'}).")
    elif isinstance(prop_a, float) or isinstance(prop_b, float):
        tag, val, landed = ("A", prop_a, landed_a) if isinstance(prop_a, float) else ("B", prop_b, landed_b)
        if landed:
            lines.append(f"Booster propellant at touchdown: {val:,.0f} kg (run {tag} only).")
        else:
            lines.append(f"Booster propellant left: {val:,.0f} kg (run {tag} only; no touchdown).")
    va = a.run.verification.mission_constraints_passed
    vb = b.run.verification.mission_constraints_passed
    if va != vb:
        lines.append(f"Mission verdict: A {_verdict_text(va)}, B {_verdict_text(vb)}.")
    return lines


def compare_runs(a: RunBundle, b: RunBundle) -> CompareResult:
    """Diff two bundles (inputs and outputs) and explain both verdicts."""
    ia, ib = run_inputs(a), run_inputs(b)
    oa, ob = run_outputs(a), run_outputs(b)
    return CompareResult(
        a=a, b=b,
        inputs=_diff(ia, ib),
        outputs=_diff(oa, ob),
        headlines=tuple(_headlines(a, b, oa, ob)),
        reasons_a=failure_explanations(a.run),
        reasons_b=failure_explanations(b.run),
    )


# --------------------------------------------------------------------------- recovery


@dataclass(frozen=True)
class RecoverySummary:
    """Recovery panel content. ``state``: ``none`` (no recovery in this
    run), ``flown`` (a recovery check with an outcome) or ``not_assessed``
    (recovery requested but never flown: no checks, or only the "not
    assessed" / "data missing" placeholders). ``montecarlo`` is None when no
    ``montecarlo.json`` was found."""

    state: str
    kind: str
    gate: Check | None
    checks: tuple[Check, ...]
    touchdown: Mapping[str, Any] | None
    explanations: tuple[ReasonEntry, ...]
    montecarlo: Mapping[str, Any] | None = None
    mc_reasons: tuple[tuple[str, int, ReasonEntry], ...] = ()
    touchdown_points: tuple[tuple[float, float, bool], ...] = ()

    @property
    def success_rate(self) -> float | None:
        return _fnum((self.montecarlo or {}).get("success_rate"))

    @property
    def ellipse(self) -> Mapping[str, Any]:
        return (self.montecarlo or {}).get("ellipse") or {}


def recovery_summary(bundle: RunBundle) -> RecoverySummary:
    run = bundle.run
    by_id = {c.id: c for c in run.verification.checks}
    checks = tuple(by_id[cid] for cid in RECOVERY_CHECK_IDS if cid in by_id)
    gate = by_id.get("recovery-landing")
    kind = recovery_kind(bundle)
    touchdown = next((dict(e.data, t_s=e.t_s, body_id=e.body_id)
                      for e in run.events if e.kind == "touchdown"), None)
    if any(c.passed or c.reason_code not in _NOT_FLOWN_CODES for c in checks):
        state = "flown"
    elif checks or kind != "none" or bundle.montecarlo:
        state = "not_assessed"
    else:
        state = "none"
    explanations = explain_all(c.reason_code for c in checks if not c.passed and c.reason_code)
    mc = bundle.montecarlo
    mc_reasons: list[tuple[str, int, ReasonEntry]] = []
    points: list[tuple[float, float, bool]] = []
    if mc:
        hist = mc.get("reason_histogram") or {}
        for code, count in sorted(hist.items(), key=lambda kv: (-int(kv[1]), kv[0])):
            mc_reasons.append((str(code), int(count), explain(str(code))))
        for s in mc.get("samples") or []:
            e, n = _fnum(s.get("touchdown_east_m")), _fnum(s.get("touchdown_north_m"))
            if e is not None and n is not None:
                points.append((e, n, s.get("outcome") == "landed"))
    return RecoverySummary(state=state, kind=kind, gate=gate, checks=checks, touchdown=touchdown,
                           explanations=explanations, montecarlo=mc, mc_reasons=tuple(mc_reasons),
                           touchdown_points=tuple(points))


def ellipse_polygon(ellipse: Mapping[str, Any], k: float = 1.0,
                    n: int = 72) -> tuple[tuple[float, float], ...]:
    """Points (east, north) [m] of the k-sigma ellipse FL-304 reports
    (``sigma_along_m`` along ``bearing_deg`` from north, ``sigma_cross_m``
    across it, centred on the mean). Empty unless ``state == assessed``."""
    if (ellipse or {}).get("state") != "assessed":
        return ()
    a = float(ellipse["sigma_along_m"]) * k
    c = float(ellipse["sigma_cross_m"]) * k
    brg = math.radians(float(ellipse.get("bearing_deg", 0.0)))
    me, mn = float(ellipse.get("mean_east_m", 0.0)), float(ellipse.get("mean_north_m", 0.0))
    ux, uy = math.sin(brg), math.cos(brg)        # along-axis unit (east, north)
    vx, vy = math.cos(brg), -math.sin(brg)       # cross-axis unit
    pts = []
    for i in range(n + 1):
        th = 2.0 * math.pi * i / n
        p, q = a * math.cos(th), c * math.sin(th)
        pts.append((me + p * ux + q * vx, mn + p * uy + q * vy))
    return tuple(pts)


# --------------------------------------------------------------------------- overlays


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in v))


def _central_angle(r0: Sequence[float], r1: Sequence[float]) -> float:
    n0, n1 = _norm(r0), _norm(r1)
    if n0 <= 0.0 or n1 <= 0.0:
        return 0.0
    cross = (r0[1] * r1[2] - r0[2] * r1[1], r0[2] * r1[0] - r0[0] * r1[2], r0[0] * r1[1] - r0[1] * r1[0])
    dot = sum(float(a) * float(b) for a, b in zip(r0, r1))
    return math.atan2(_norm(cross), dot)


def trajectory_profiles(run: FlightRun) -> tuple[tuple[str, tuple[tuple[float, float], ...]], ...]:
    """Per body: ``(downrange_km, altitude_km)`` points. Uses the track's own
    ``downrange_m`` / ``altitude_m`` columns when present (OpenTsiolkovsky),
    else the great-circle angle from the first sample of the earliest track
    times the WGS-84 equatorial radius (inertial downrange -- labelled as
    such in the view) and ``|r| - R``."""
    if not run.tracks:
        return ()
    root = min(run.tracks, key=lambda t: t.t_start_s)
    origin = root.position_m[0]
    out = []
    for track in run.tracks:
        extras = {k.lower(): v for k, v in track.extra_columns.items()}
        dr = extras.get("downrange_m")
        alt = extras.get("altitude_m")
        pts: list[tuple[float, float]] = []
        for i, pos in enumerate(track.position_m):
            d = (float(dr[i]) if dr is not None and i < len(dr)
                 else _central_angle(origin, pos) * EARTH_RADIUS_M)
            h = (float(alt[i]) if alt is not None and i < len(alt)
                 else _norm(pos) - EARTH_RADIUS_M)
            if math.isfinite(d) and math.isfinite(h):
                pts.append((d / 1000.0, h / 1000.0))
        out.append((track.body_id, tuple(pts)))
    return tuple(out)


def overlay_channels(channels_a: Iterable[Any], channels_b: Iterable[Any],
                     label_a: str = "A", label_b: str = "B") -> tuple[Any, ...]:
    """Merge two runs' telemetry channels (``TelemetryChannel`` tuples from
    :func:`orbitopt.flightlab.ui.telemetry.build_telemetry_channels`) into
    one channel per id: A's series solid, B's dashed, labels prefixed. Either
    side may be empty (only one run picked)."""
    by_a = {c.id: c for c in channels_a}
    by_b = {c.id: c for c in channels_b}
    merged = []
    for cid in list(by_a) + [k for k in by_b if k not in by_a]:
        ca, cb = by_a.get(cid), by_b.get(cid)
        series: tuple = ()
        if ca is not None:
            series += tuple(replace(s, label=f"{label_a} {s.label}", style="solid") for s in ca.series)
        if cb is not None:
            series += tuple(replace(s, label=f"{label_b} {s.label}", style="dash") for s in cb.series)
        base = ca or cb
        missing = "" if series else (base.missing_reason or "")
        merged.append(replace(base, series=series, missing_reason=missing))
    return tuple(merged)


def discover_runs(root: Path | str | None = None) -> list[tuple[str, Path]]:
    """Complete run directories under the designer result cache
    (``<cache>/runs/<digest>/``; in-progress ones are skipped), newest first,
    labelled ``run_id · recovery · digest[:8]``. ``root`` overrides the cache
    root (tests)."""
    if root is None:
        from orbitopt.flightlab.designer import cache_root

        root = cache_root()
    runs_dir = Path(root) / "runs"
    if not runs_dir.is_dir():
        return []
    found: list[tuple[float, str, Path]] = []
    for d in runs_dir.iterdir():
        manifest = d / "manifest.json"
        if not manifest.is_file() or (d / "_designer-in-progress").exists():
            continue
        doc = _read_json(manifest) or {}
        form = ((_read_json(d / "designer-form.json") or {}).get("mission_form") or {})
        recov = form.get("recovery") or "?"
        label = f"{doc.get('run_id', d.name)} · recovery={recov} · {d.name[:8]}"
        found.append((manifest.stat().st_mtime, label, d))
    found.sort(key=lambda item: -item[0])
    return [(label, path) for _t, label, path in found]
