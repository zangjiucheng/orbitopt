"""Plain-language reason-code catalog for Flight Lab verdicts (FL-503).

Every failing :class:`~orbitopt.flightlab.contract.Check` carries a
machine-readable ``reason_code`` (``PERIAPSIS_IN_ATMOSPHERE``,
``max_q_exceeded``, ``INSUFFICIENT_PROPELLANT_BOOSTBACK``, ...). This module
maps each code to a short title, a plain-language explanation and a suggested
fix, so the replay and compare views can say *why* a
run is red and *what to try next* instead of printing an identifier.

Lookup is case-insensitive (the OpenTsiolkovsky adapter emits lower-case
codes, the checks module upper-case ones): :func:`normalize_code` folds both
to upper snake case. ``SAMPLE_ERROR:<Exception>`` (FL-304 Monte Carlo) is
matched by its prefix. Unknown codes never raise: :func:`explain` returns a
generic entry flagged ``known=False`` so the UI can still show the raw code.

``tests/test_flightlab_reasons.py`` scans every module under
``orbitopt/flightlab`` and fails when a reason code is emitted that has no
entry here -- add the entry in the same change that adds the code.

Qt-free, stdlib-only: importable on a contract-only install.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping

__all__ = [
    "CATALOG",
    "ReasonEntry",
    "explain",
    "explain_all",
    "is_known",
    "lookup",
    "normalize_code",
]


@dataclass(frozen=True)
class ReasonEntry:
    """One catalogued reason code. ``category`` groups codes for display
    (``insertion``, ``loads``, ``propellant``, ``recovery``, ``landing``,
    ``physics``, ``data``, ``optimizer``, ``montecarlo``); ``severity`` is
    ``fail`` for a violated constraint, ``missing`` for a check that could not
    be assessed, ``info`` for a non-failure outcome."""

    code: str
    category: str
    title_en: str
    explanation_en: str
    fix_en: str
    severity: str = "fail"
    known: bool = True

    def text(self) -> str:
        """``title -- explanation  Fix: ...`` on one line."""
        return f"{self.title_en} — {self.explanation_en} Fix: {self.fix_en}"

    def block(self) -> str:
        """Multi-line title / explanation / fix (tooltips, the explanation panes)."""
        return f"{self.title_en}\n{self.explanation_en}\nFix: {self.fix_en}"


def _e(code, category, title_en, expl_en, fix_en, severity="fail"):
    return ReasonEntry(code, category, title_en, expl_en, fix_en, severity)


_ENTRIES: tuple[ReasonEntry, ...] = (
    # ------------------------------------------------------------ insertion orbit
    _e("PERIAPSIS_IN_ATMOSPHERE", "insertion",
       "Periapsis inside the atmosphere",
       "The orbit at engine cutoff dips below 120 km; drag would bring the upper stage and payload "
       "back down within hours. This is a failed mission regardless of the target's own floor.",
       "Raise the insertion velocity: loft the trajectory (shallower pitch-over), lengthen the upper-stage "
       "burn, or reduce payload / dry mass."),
    _e("INSERTION_PERIAPSIS_BELOW_MINIMUM", "insertion",
       "Periapsis below the mission minimum",
       "The orbit clears the atmosphere but its lowest point is under the minimum insertion periapsis "
       "this target declares.",
       "Add upper-stage delta-v (more propellant or Isp, less payload) or shape the ascent so the cutoff "
       "happens closer to apoapsis."),
    _e("INSERTION_PERIAPSIS_OUT_OF_TOLERANCE", "insertion",
       "Periapsis off target",
       "The achieved periapsis altitude is outside the allowed band around the target periapsis.",
       "Adjust the pitch program and upper-stage cutoff time, or run the ascent optimizer against this target."),
    _e("INSERTION_APOAPSIS_OUT_OF_TOLERANCE", "insertion",
       "Apoapsis off target",
       "The achieved apoapsis altitude is outside the allowed band around the target apoapsis — usually "
       "an over- or under-burn of the upper stage.",
       "Trim the upper-stage burn end time / propellant load, or flatten the late pitch program."),
    _e("INSERTION_INCLINATION_OUT_OF_TOLERANCE", "insertion",
       "Inclination off target",
       "The orbital plane is tilted differently from the target inclination by more than the tolerance.",
       "Correct the launch azimuth (attitude program azimuth) for the site latitude and Earth rotation."),
    _e("INSERTION_RAAN_OUT_OF_TOLERANCE", "insertion",
       "Orbital plane orientation (RAAN) off target",
       "The orbit's ascending node is in the wrong place: the vehicle launched at the wrong time for the "
       "requested plane.",
       "Move the liftoff epoch into the launch window for this RAAN (use the window search), or relax the "
       "RAAN requirement."),

    # ---------------------------------------------------------------- flight loads
    _e("MAX_Q_EXCEEDED", "loads",
       "Max-Q limit exceeded",
       "Peak dynamic pressure during ascent is above the structural limit — the vehicle flies too fast "
       "too low in the atmosphere.",
       "Throttle down through max-Q, pitch over later / more gently, or lower liftoff thrust-to-weight."),
    _e("MAX_ACCEL_EXCEEDED", "loads",
       "Acceleration (g-load) limit exceeded",
       "Thrust acceleration exceeds the g limit, typically near burnout when the stage is light.",
       "Add a throttle-down before MECO/SECO, or reduce thrust on the light end of the burn."),
    _e("MAX_Q_DATA_MISSING", "data",
       "Max-Q not measured",
       "The target declares a dynamic-pressure limit but the run has no max-Q value or dynamic-pressure "
       "column to check it against.",
       "Re-fly with a backend that reports dynamic pressure (OpenTsiolkovsky does), or drop the limit "
       "from the target.", "missing"),
    _e("MAX_ACCEL_DATA_MISSING", "data",
       "Acceleration not measured",
       "The target declares a g limit but the run reports no peak acceleration.",
       "Re-fly with a backend that reports acceleration, or drop the limit from the target.", "missing"),
    _e("LIFTOFF_THRUST_TO_WEIGHT_LOW", "loads",
       "Cannot lift off (T/W < 1)",
       "Stage-1 thrust at liftoff is below the vehicle's weight; the rocket would sit on the pad.",
       "Add stage-1 thrust (engines) or remove mass (propellant, upper stage, payload)."),

    # ------------------------------------------------------------------ propellant
    _e("PROPELLANT_RESERVE_LOW", "propellant",
       "Propellant reserve below requirement",
       "Less propellant is left than the target's reserve requires (upper stage at insertion, or the "
       "booster for recovery).",
       "Load more propellant, reduce payload, or improve the trajectory's efficiency (less gravity/drag loss)."),
    _e("PROPELLANT_RESERVE_DATA_MISSING", "data",
       "Propellant reserve not assessable",
       "The target declares an upper-stage reserve but the run/vehicle lacks the masses needed to measure it.",
       "Make sure the vehicle file states each stage's dry mass and the run has an insertion state.", "missing"),
    _e("RECOVERY_PROPELLANT_RESERVE_DATA_MISSING", "data",
       "Recovery reserve not assessable",
       "Recovery is planned but the run has no released first-stage track (or the vehicle has no stage-1 "
       "dry mass), so the propellant left for landing cannot be measured.",
       "Fly with a backend that writes the stage-1 track after separation and give stage 1 a dry mass.", "missing"),
    _e("NEGATIVE_MASS", "data",
       "Negative mass in the trajectory",
       "A body's mass went below zero: the backend burned more propellant than the stage carries.",
       "Check burn end times against propellant load (mass_initial - mass_dry) and engine flow rate."),
    _e("MASS_DISCONTINUITY", "data",
       "Unexplained mass jump",
       "Mass changed abruptly between samples outside a staging/jettison event — the mass bookkeeping is "
       "inconsistent.",
       "Declare the jettison (fairing, interstage) in the vehicle file or fix the stage masses."),

    # -------------------------------------------------------------- physics re-fly
    _e("REFLY_BACKEND_MISSING", "physics",
       "Physics re-fly not available",
       "The independent orbit re-fly needs tudatpy, which is not installed, so the physics verdict stays "
       "unassessed.",
       "Use the conda environment (envs/cpu.yml) that ships tudatpy, then re-run the simulation.", "missing"),
    _e("REFLY_NOT_ORBITAL", "physics",
       "Not in orbit — nothing to re-fly",
       "The insertion state is suborbital (it would hit the ground within one orbit), so the orbit re-fly "
       "cannot run.",
       "Fix the ascent first: see the insertion checks for what is short."),
    _e("REFLY_BACKEND_ERROR", "physics",
       "Physics re-fly crashed",
       "The tudatpy re-fly raised an error, so physics could not be verified.",
       "Open the run assumptions for the error text; re-run, and report it if it persists.", "missing"),
    _e("REFLY_ELEMENT_DRIFT", "physics",
       "Orbit not self-consistent",
       "Re-integrating the insertion state with an independent propagator drifted beyond the Gate-G5 "
       "limits: the reported orbit is not a consistent orbit.",
       "Check the trajectory frame/epoch (GCRS bridge) and the backend's output step; this is a data "
       "problem, not a tuning one."),

    # ------------------------------------------------- 6-DOF backend (F-01, #226)
    _e("SIXDOF_NON_FINITE_STATE", "physics",
       "6-DOF state went non-finite",
       "The JSBSim 6-DOF integration produced a NaN or infinite state — the flight diverged and no "
       "downstream check means anything.",
       "Shorten the integration step (dt_s), lower the throttle, or check the vehicle's mass/propellant "
       "figures in the scenario config."),
    _e("SIXDOF_HANDOFF_CONTINUITY", "physics",
       "Staging handoff not continuous",
       "At the 6-DOF staging split the child states do not match the frozen stack state within the "
       "declared tolerances (or the mass/momentum bookkeeping closed only loosely) — the handoff lost "
       "or invented state.",
       "Re-check the staging split fractions, the separation impulse and the interstage models; the "
       "handoff must be state-preserving by construction."),
    _e("SIXDOF_TRACKING_OUT_OF_BOUNDS", "physics",
       "Attitude tracking out of bounds",
       "The closed-loop 6-DOF attitude/trajectory deviation exceeded a bound declared in the scenario "
       "config (settled error, gimbal saturation, or MECO-state deviation vs the reference).",
       "Re-tune the PID gains or slow the attitude program down; verify the initial attitude matches the "
       "program's first waypoint."),
    _e("SIXDOF_MASS_MODEL_MISMATCH", "physics",
       "Mass properties off the declared model",
       "The integrated mass / centre-of-gravity / inertia trace disagreed with the mass model declared "
       "in the scenario config beyond its tolerances: mass is not conserved against the integrated "
       "flow, or the full/burnout endpoints, or the in-flight interpolation, are off.",
       "Check the scenario's mass_model endpoints against the aircraft XML (tank contents, locations, "
       "dry mass/inertia), the integration step dt_s (the drain shows up one step late), and the "
       "tolerances; a genuine mismatch means the model and the vehicle file disagree."),
    _e("SIXDOF_BURNOUT_NOT_REACHED", "physics",
       "Burnout endpoint not assessed",
       "The config declares a mass model with an empty (burnout) endpoint but the run never flew the "
       "propellant down to it, so the burnout CG/inertia endpoint could not be compared.",
       "Lengthen duration_s (or drop the ignition time) until the tanks run dry, or leave the burnout "
       "endpoint out of the config if the scenario is not meant to assess it."),

    # --------------------------------------------------------- optimizer (FL-201+)
    _e("INFEASIBLE_PERIAPSIS_OUT_OF_TOLERANCE", "optimizer",
       "Optimizer could not hit periapsis",
       "The best design the optimizer found still misses the target periapsis band.",
       "Give the optimizer more generations/population, widen decision bounds, or add vehicle performance."),
    _e("INFEASIBLE_APOAPSIS_OUT_OF_TOLERANCE", "optimizer",
       "Optimizer could not hit apoapsis",
       "The best design the optimizer found still misses the target apoapsis band.",
       "Give the optimizer more budget or widen the burn-time bounds."),
    _e("INFEASIBLE_INCLINATION_OUT_OF_TOLERANCE", "optimizer",
       "Optimizer could not hit inclination",
       "No design found reaches the target inclination — often the site latitude makes it unreachable "
       "without a plane change.",
       "Check that the inclination is at least the site latitude, or pick another site."),
    _e("INFEASIBLE_RAAN_OUT_OF_TOLERANCE", "optimizer",
       "Optimizer could not hit RAAN",
       "No design reaches the requested orbital plane from this liftoff epoch.",
       "Re-derive the liftoff epoch from the launch window for this RAAN."),
    _e("INFEASIBLE_MAX_Q_EXCEEDED", "optimizer",
       "Optimizer could not respect max-Q",
       "Every good design breaks the dynamic-pressure limit.",
       "Allow throttling in the decision vector, or relax the max-Q limit if the structure allows it."),
    _e("INFEASIBLE_MAX_ACCEL_EXCEEDED", "optimizer",
       "Optimizer could not respect the g limit",
       "Every good design breaks the acceleration limit.",
       "Allow end-of-burn throttle-down, or relax the g limit."),
    _e("INFEASIBLE_PROPELLANT_RESERVE", "optimizer",
       "Optimizer could not keep the reserve",
       "Reaching orbit uses up the propellant the target wants held in reserve.",
       "Lower payload or the reserve requirement, or add propellant."),
    _e("INFEASIBLE_SIMULATION", "optimizer",
       "No candidate flew",
       "Every candidate the optimizer tried failed to simulate; there is no flight to report.",
       "Check the base vehicle flies on its own (Simulate) before optimizing; narrow the decision bounds."),

    # --------------------------------------------------- recovery: boostback/entry
    _e("INSUFFICIENT_PROPELLANT_BOOSTBACK", "recovery",
       "Not enough propellant for boostback",
       "The booster runs dry before the boostback burn has turned it around toward the landing site.",
       "Hold back more stage-1 propellant (earlier MECO / larger recovery reserve), land downrange instead "
       "of RTLS, or lighten the upper stage."),
    _e("INSUFFICIENT_PROPELLANT_ENTRY_BURN", "recovery",
       "Not enough propellant for the entry burn",
       "After boostback the booster cannot afford the entry burn that slows it for re-entry.",
       "Increase the recovery reserve, shorten boostback (downrange landing), or accept a lighter payload."),
    _e("INSUFFICIENT_PROPELLANT_RESERVE", "recovery",
       "Landing reserve used up",
       "Boostback and entry burns leave less propellant than the landing burn needs.",
       "Raise the recovery reserve at MECO or trim the boostback/entry burns."),
    _e("PROPELLANT_EXHAUSTED", "recovery",
       "Propellant exhausted",
       "A booster burn ended because the tanks ran dry, not because guidance commanded cutoff.",
       "Increase the recovery reserve or shorten the burn that ran dry."),
    _e("COMMANDED_CUTOFF", "recovery",
       "Commanded cutoff",
       "A burn ended on the guidance command (normal end of burn).",
       "Nothing to fix.", "info"),
    _e("CORRIDOR_MISSED", "recovery",
       "Missed the entry corridor",
       "The booster reached the entry interface outside the corridor that leads to the landing site.",
       "Re-plan boostback (more burn time / better heading) or move the landing target."),
    _e("HANDOFF_STATE_DISCONTINUITY", "recovery",
       "Handoff state discontinuity",
       "A recovery segment restarted from a state that is not the one the previous "
       "segment handed over: position, velocity or mass jump across the phase boundary.",
       "Check the segment boundary against the HandoffState digest; the producer or "
       "the consumer re-derived the initial state instead of using the handed-over one."),
    _e("HANDOFF_MASS_NOT_CONSERVED", "recovery",
       "Mass not conserved across handoffs",
       "Mass bookkeeping broke in the recovery chain: mass rose on a flown track, "
       "changed during the ballistic coast between burns, or the chain ended below dry mass.",
       "Inspect the mass column and the propellant flow of the phase the failing check names."),
    _e("CORRIDOR_NOT_REACHED", "recovery",
       "Entry corridor not reached",
       "The booster's trajectory never reached the entry corridor altitude window.",
       "Check the separation state and boostback plan; a downrange landing may be the only feasible option."),
    _e("ENTRY_Q_LIMIT_EXCEEDED", "recovery",
       "Entry dynamic pressure too high",
       "During re-entry the booster sees more dynamic pressure than it is rated for.",
       "Lengthen the entry burn to slow down before the dense atmosphere."),
    _e("ENTRY_G_LIMIT_EXCEEDED", "recovery",
       "Entry deceleration too high",
       "Aerodynamic deceleration during re-entry exceeds the booster's g limit.",
       "Lengthen the entry burn or steepen/shallow the entry to spread the deceleration."),
    _e("RECOVERY_INFEASIBLE", "recovery",
       "Recovery infeasible",
       "The boostback/entry planner found no plan that brings the booster back.",
       "Increase the recovery reserve or choose a downrange landing."),
    _e("RECOVERY_LANDING_NOT_ASSESSED", "recovery",
       "Recovery not assessed",
       "The target asks for a recovery but the booster was never flown back (no recovery backend or an "
       "error), so the landing verdict is missing.",
       "Install the landing extras (envs/flightlab-landing.txt) and re-simulate, or set recovery to none.", "missing"),
    _e("RECOVERY_LANDING_NOT_ATTEMPTED", "recovery",
       "Landing burn not attempted",
       "Boostback/entry succeeded but no landing burn was flown.",
       "Check the landing solver dependencies and the landing-burn start conditions.", "missing"),

    # ---------------------------------------------------------- recovery: landing
    _e("LANDING_INFEASIBLE", "landing",
       "No feasible landing burn",
       "The powered-descent solver could not find any landing burn from the state the booster arrives in "
       "(too fast, too far off, or too little propellant).",
       "Arrive slower/closer (longer entry burn, better boostback) or keep more landing propellant."),
    _e("LANDING_FAILED", "landing",
       "Landing failed",
       "The booster flew a landing burn but did not touch down inside the limits.",
       "See the touchdown checks for which limit broke."),
    _e("LANDING_NO_TOUCHDOWN", "landing",
       "No touchdown",
       "The re-flown landing burn never reached the ground (it hovered out or fell past the horizon).",
       "Start the landing burn later / lower, or give it more thrust margin."),
    _e("LANDING_POSITION_OUT_OF_TOLERANCE", "landing",
       "Missed the landing pad",
       "The booster touched down farther from the target than the acceptance radius.",
       "Improve boostback targeting or allow more divert in the landing burn."),
    _e("LANDING_VERTICAL_SPEED_EXCEEDED", "landing",
       "Touchdown too hard (vertical)",
       "Vertical speed at touchdown is above the soft-landing limit.",
       "Keep more landing propellant or start the landing burn slightly higher."),
    _e("LANDING_LATERAL_SPEED_EXCEEDED", "landing",
       "Touchdown sliding sideways",
       "Horizontal speed at touchdown is above the limit; the legs could fail.",
       "Null horizontal velocity earlier in the landing burn (more divert time)."),
    _e("LANDING_TILT_EXCEEDED", "landing",
       "Booster tilted too far",
       "During the landing burn the thrust vector tilted further from vertical than allowed.",
       "Arrive closer to the pad so less divert is needed, or relax the tilt limit if the vehicle allows."),
    _e("LANDING_PROPELLANT_RESERVE_LOW", "landing",
       "Landed with too little propellant",
       "Propellant left at touchdown is below the vehicle's landing reserve.",
       "Raise the recovery reserve at MECO (costs payload) or trim boostback/entry burns."),
    _e("LANDING_MAX_G_EXCEEDED", "landing",
       "Landing burn g-load too high",
       "Peak proper acceleration during the landing burn exceeds the vehicle limit.",
       "Start the landing burn earlier at lower throttle (less aggressive suicide burn)."),
    _e("IMPACT_ELLIPSE_NOT_ASSESSED", "landing",
       "Impact ellipse not assessed",
       "There were not enough touchdowns/impacts to estimate a dispersion ellipse.",
       "Run the Monte Carlo with more samples.", "missing"),

    # ----------------------------------------------------------- Monte Carlo (FL-304)
    _e("LANDED", "montecarlo",
       "Landed in limits",
       "The sample touched down inside every landing limit.",
       "Nothing to fix.", "info"),
    _e("SAMPLE_ERROR", "montecarlo",
       "Monte Carlo sample crashed",
       "One dispersed sample raised an error and produced no verdict; it counts as a failure.",
       "Inspect the sample's error text in montecarlo.json; reduce extreme dispersions if they cause it."),
)

#: code (upper snake case) -> entry.
CATALOG: Mapping[str, ReasonEntry] = {entry.code: entry for entry in _ENTRIES}

#: Codes looked up by prefix (``SAMPLE_ERROR:ValueError`` -> ``SAMPLE_ERROR``).
_PREFIX_CODES = ("SAMPLE_ERROR",)

_NON_WORD = re.compile(r"[^A-Z0-9]+")


def normalize_code(code: str) -> str:
    """Upper snake case: ``max_q_exceeded`` / ``Max-Q exceeded`` ->
    ``MAX_Q_EXCEEDED``; a ``PREFIX:detail`` code keeps only its prefix."""
    text = str(code or "").strip()
    head = text.split(":", 1)[0] if ":" in text else text
    return _NON_WORD.sub("_", head.upper()).strip("_")


def lookup(code: str) -> ReasonEntry | None:
    """The catalog entry for ``code`` (case-insensitive), or None."""
    key = normalize_code(code)
    if not key:
        return None
    entry = CATALOG.get(key)
    if entry is None:
        entry = next((CATALOG[p] for p in _PREFIX_CODES if key.startswith(p)), None)
    return entry


def is_known(code: str) -> bool:
    return lookup(code) is not None


def explain(code: str) -> ReasonEntry:
    """Always an entry: the catalogued one, or a generic ``known=False`` entry
    that still shows the raw code (never raises on an unknown code)."""
    entry = lookup(code)
    if entry is not None:
        return entry
    raw = str(code or "").strip() or "(no reason code)"
    return ReasonEntry(
        code=raw, category="unknown",
        title_en=raw,
        explanation_en="This reason code has no plain-language entry yet.",
        fix_en="See the check's value and limit; add the code to orbitopt.flightlab.reasons.",
        known=False,
    )


def explain_all(codes: Iterable[str]) -> tuple[ReasonEntry, ...]:
    """Entries for ``codes`` in order, duplicates (after normalization) dropped."""
    seen: set[str] = set()
    out: list[ReasonEntry] = []
    for code in codes:
        if not code:
            continue
        entry = explain(code)
        key = entry.code if entry.known else normalize_code(code)
        if key in seen:
            continue
        seen.add(key)
        out.append(entry)
    return tuple(out)
