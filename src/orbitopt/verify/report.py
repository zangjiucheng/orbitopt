"""Machine-readable verification report (task #137).

Every high-fidelity verification (``verify.tudat_propagate`` and the mission
verifiers built on it) produces one of these: a deterministic JSON document
that quantifies how far the result can be trusted, rather than a single
"converged" flag. Each report carries

* **residuals** -- position / velocity residuals of the numerical solution
  against a named reference prediction (pykep Lambert arc, analytic screening
  orbit, targeter aim point, two-body solution of the same initial state...);
* **integrator** -- the integration method and tolerances actually used, the
  realised step-size statistics, and a *step-convergence* check: the same
  propagation re-run at tighter tolerances (adaptive) or half the step
  (fixed-step RK4), with the final-state difference as the integration-error
  estimate;
* **conservation** -- specific orbital energy and angular-momentum drift.
  Under the full perturbed model these quantities are not integrals of
  motion, so that block is flagged ``applicable: false`` and kept as a
  diagnostic; the *applicable* check re-integrates the same initial state
  under the central body's point mass alone (where both are conserved
  exactly), so any drift there is pure numerical error;
* **dynamics** -- the perturbation-model inventory: which body exerts which
  gravity model (point mass vs spherical harmonics with J2 / J22 named
  explicitly), which third bodies (Sun, Moon, planets) are included, and what
  is *not* modelled (SRP, drag, relativity, finite burns);
* **check categories** (task #243) -- every check in ``status`` carries one of
  three categories: ``numerical_consistency`` (the same model re-run by
  independent numerics: step convergence, conservation of quantities the model
  conserves exactly, solver convergence), ``model_agreement`` (the result
  against a *different* model or prediction: a reference propagator, the
  screening orbit, an aim point, a vehicle/site or corridor limit) and
  ``historical_validation`` (against published real-world data, e.g. a flown
  mission's timeline). ``status.check_categories`` maps each check name to its
  category and ``status.categories`` rolls the outcome up per category, so a
  reader sees which kind of evidence backs a pass -- and which kind is absent
  (a category with no check is ``not_checked``, never counted as a pass).
* **evidence qualification** (task #263, E4-03) -- a sharper, orthogonal read
  of the same checks: what the reference a verdict is graded against is *made
  of*, and how far a pass there reaches. Five types -- ``analytical`` (a closed
  form of the same/simplified model), ``independent_model`` (another model,
  formulated separately), ``historical_telemetry`` (published flown data),
  ``design_input`` (the reference is the aim point / target / envelope the
  solution was built to hit -- internal consistency, NOT validation) and
  ``numerical_consistency`` (integration/self-consistency error) -- each with a
  recorded scope, tolerance, applicable phases and what it *cannot* prove.
  ``status.check_evidence`` and ``status.evidence`` carry the per-check type and
  the per-type rollup; ``status.independent_validation`` says in one rollup
  whether any *validation-grade* (historical or independent-model) evidence
  backed the result, so a green ``passed`` on design-input and numerical checks
  alone is never mistaken for independent validation.

Reproducibility: the document contains no wall-clock time, keys are sorted,
floats are serialised with full ``repr`` precision, and ``digest`` is the
SHA-256 of the canonical JSON of everything else -- re-running the same
verification with the same inputs and library versions yields a byte-identical
file. Validate against ``orbitopt/schemas/verification-report-1.0.json``.

This module is pure numpy/stdlib (no tudatpy import) so it can be unit tested
and consumed anywhere; the tudat-specific glue lives in
:func:`orbitopt.verify.tudat_propagate.propagation_report_section` (a thin
wrapper around :func:`propagation_section`).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from importlib import metadata as _metadata
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

SCHEMA_VERSION = "1.0"
SCHEMA_FILE = "verification-report-1.0.json"

# Environment variable naming a directory: when set, every report built via
# ``finalize_report(..., write=True)`` (the default for the verify entry
# points) is also written there as ``<kind>-<label>.verification.json``.
REPORT_DIR_ENV = "ORBITOPT_VERIFICATION_REPORT_DIR"

# Default step-convergence acceptance: final-state difference between the
# nominal run and the tightened re-run. Absolute floor or relative-to-state
# magnitude, whichever is looser.
CONVERGENCE_POS_ABS_M = 10.0
CONVERGENCE_POS_REL = 1e-8
CONVERGENCE_VEL_ABS_M_S = 1e-3
CONVERGENCE_VEL_REL = 1e-8
# Two-body conservation acceptance (relative drift of energy / |h|).
CONSERVATION_REL_TOL = 1e-8

# Not modelled anywhere in the verification layer (tudat_propagate only sets
# up gravity accelerations); listed so a reader never has to assume.
_ALWAYS_UNMODELLED = (
    "solar_radiation_pressure",
    "atmospheric_drag",
    "relativistic_correction",
    "third_body_spherical_harmonics",
    "solid_and_ocean_tides",
    "finite_burn_thrust_arcs (burns are impulsive)",
)

# Check categories (task #243): what kind of evidence a check actually is.
# Three categories answer the three independent questions a verification can
# settle -- is the numerics trustworthy (the same model, independent
# numerics), do independent models agree (the result vs another model's
# prediction), and does the real world agree (vs published, observed data).
CHECK_NUMERICAL = "numerical_consistency"
CHECK_MODEL = "model_agreement"
CHECK_HISTORICAL = "historical_validation"
CHECK_CATEGORIES = (CHECK_NUMERICAL, CHECK_MODEL, CHECK_HISTORICAL)
CATEGORY_LABELS = {
    CHECK_NUMERICAL: "Numerical consistency",
    CHECK_MODEL: "Model agreement",
    CHECK_HISTORICAL: "Historical validation",
}
"""Human-readable titles for the Verification tab."""

# --------------------------------------------------------------------------
# Evidence qualification (task #263, E4-03)
# --------------------------------------------------------------------------
# The category answers *which question* a check settles. The evidence type
# answers a sharper one: *what is the reference actually made of*, and how far
# a pass there reaches. It refines the category, and the two are deliberately
# not in lockstep -- a model_agreement check can rest on a closed form (weak:
# it shares the model's assumptions), on a genuinely different simulator
# (strong), or on the design input itself (no independent force at all).
#
# The distinction this task is about: a comparison whose reference is the aim
# point / target / envelope the solution was *built to hit* is a design-input
# match -- internal consistency, not validation. Labelled ``design_input`` so a
# reader never mistakes "the solver landed where we aimed it" for independent
# (historical) proof.
EVIDENCE_ANALYTICAL = "analytical"
EVIDENCE_INDEPENDENT_MODEL = "independent_model"
EVIDENCE_HISTORICAL = "historical_telemetry"
EVIDENCE_DESIGN_INPUT = "design_input"
EVIDENCE_NUMERICAL = "numerical_consistency"
EVIDENCE_TYPES = (
    EVIDENCE_ANALYTICAL, EVIDENCE_INDEPENDENT_MODEL, EVIDENCE_HISTORICAL,
    EVIDENCE_DESIGN_INPUT, EVIDENCE_NUMERICAL,
)
# The evidence that reaches outside the model under test -- the only kinds that
# count as validation. A report that passes on everything else has NOT been
# independently validated.
INDEPENDENT_EVIDENCE = (EVIDENCE_HISTORICAL, EVIDENCE_INDEPENDENT_MODEL)

EVIDENCE_QUALIFICATION = {
    EVIDENCE_ANALYTICAL: {
        "label": "Analytic solution",
        "independent": False,
        "proves": "the numerics reproduce the closed-form solution of the model they integrate "
                  "(the equations and the integrator, not their agreement with reality).",
        "cannot_prove": "that the model represents reality -- the closed-form reference shares the "
                        "model's own assumptions (point mass, impulsive burns, idealised field).",
    },
    EVIDENCE_INDEPENDENT_MODEL: {
        "label": "Independent model",
        "independent": True,
        "proves": "two independently formulated models agree, so neither is trivially mis-implemented.",
        "cannot_prove": "that either model is right in the real world -- an assumption shared by both "
                        "stays invisible; the phase and force inventory of the comparison bound its reach.",
    },
    EVIDENCE_HISTORICAL: {
        "label": "Historical telemetry",
        "independent": True,
        "proves": "the model reproduces a published, observed (flown) value it was not tuned to.",
        "cannot_prove": "any other vehicle, phase or operating point -- one published event validates "
                        "that event, not the model as a whole, and the reference's own provenance is only "
                        "as good as its source.",
    },
    EVIDENCE_DESIGN_INPUT: {
        "label": "Matches design input",
        "independent": False,
        "proves": "the solver reached the aim point, or the plan respects the site/vehicle/corridor "
                  "envelope it was designed against -- internal consistency only.",
        "cannot_prove": "anything about fidelity to reality: the reference IS the design input, so this "
                        "is NOT independent validation and must never be read as historical proof.",
    },
    EVIDENCE_NUMERICAL: {
        "label": "Numerical consistency",
        "independent": False,
        "proves": "integration error is under tolerance and the model's own invariants / self-consistency "
                  "hold (step convergence, exact integrals, a same-model re-fly).",
        "cannot_prove": "agreement with any other model or with reality -- it bounds numerical error only.",
    },
}
"""Per evidence type: how much a pass there reaches. ``independent`` marks the
kinds that count as validation; ``cannot_prove`` is what the verdict must never
be taken to establish."""


def default_evidence(category: str) -> str:
    """The evidence type a check carries when its producer states only a
    category: numerical consistency stays numerical; a model-agreement check
    is *assumed* to lean on another model (the strongest reading a category
    alone supports) and a historical check on telemetry. Producers override
    this -- most importantly to demote a model-agreement comparison whose
    reference is a closed form (``analytical``) or the design input itself
    (``design_input``)."""
    if category == CHECK_HISTORICAL:
        return EVIDENCE_HISTORICAL
    if category == CHECK_MODEL:
        return EVIDENCE_INDEPENDENT_MODEL
    if category == CHECK_NUMERICAL:
        return EVIDENCE_NUMERICAL
    raise ValueError(f"unknown check category {category!r} (must be one of {CHECK_CATEGORIES})")


def evidence_qualification(evidence: str) -> dict:
    """The ``{label, independent, proves, cannot_prove}`` a verdict of this
    evidence type must be read against."""
    try:
        return dict(EVIDENCE_QUALIFICATION[evidence])
    except KeyError:
        raise ValueError(f"unknown evidence type {evidence!r} (must be one of {EVIDENCE_TYPES})") from None


# --------------------------------------------------------------------------
# JSON helpers
# --------------------------------------------------------------------------

def to_jsonable(obj):
    """Recursively convert numpy scalars/arrays (and tuples) to plain JSON
    types; non-finite floats become ``None`` (strict JSON has no NaN/inf)."""
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if obj is None or isinstance(obj, str):
        return obj
    raise TypeError(f"Not JSON-serialisable in a verification report: {type(obj).__name__}")


def canonical_json(obj) -> str:
    return json.dumps(to_jsonable(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)


def report_digest(report: Mapping) -> str:
    """SHA-256 over the canonical JSON of ``report`` without its ``digest``."""
    body = {k: v for k, v in report.items() if k != "digest"}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Perturbation-model inventory
# --------------------------------------------------------------------------

def spherical_harmonic_terms(degree: int, order: int) -> list:
    """Named coefficients of a (degree, order) field, beyond the point-mass
    C00 term: zonals ``J<n>`` and tesserals/sectorals ``C<n><m>/S<n><m>``,
    with the (2,2) sectoral also tagged ``J22``."""
    terms = []
    for n in range(2, int(degree) + 1):
        terms.append(f"J{n}")
        for m in range(1, min(n, int(order)) + 1):
            name = f"C{n}{m}/S{n}{m}"
            if (n, m) == (2, 2):
                name += " (J22)"
            terms.append(name)
    return terms


def perturbation_inventory(
    central_body: str,
    perturbing_bodies: Sequence[str],
    central_spherical_harmonic: Optional[Sequence[int]] = None,
    frame: str = "ECLIPJ2000",
    ephemeris: str = "SPICE standard kernels",
) -> dict:
    """Machine-readable list of every force in the propagated dynamics.

    Mirrors exactly what ``tudat_propagate`` sets up: one gravity
    acceleration per body in ``perturbing_bodies`` -- the central body's own
    gravity is included only if it appears there -- with the central body
    optionally upgraded to a spherical-harmonic field."""
    bodies = list(dict.fromkeys(perturbing_bodies))
    sh = tuple(int(x) for x in central_spherical_harmonic) if central_spherical_harmonic else None
    forces = []
    for body in bodies:
        role = "central" if body == central_body else "third_body"
        entry = {"body": body, "role": role, "type": "gravity"}
        if role == "central" and sh is not None:
            entry.update(model="spherical_harmonic", degree=sh[0], order=sh[1],
                         terms=spherical_harmonic_terms(*sh))
        else:
            entry.update(model="point_mass", degree=0, order=0, terms=[])
        if role == "third_body":
            entry["ephemeris"] = ephemeris
        forces.append(entry)

    central_terms = next((f["terms"] for f in forces if f["role"] == "central"), [])
    not_modelled = list(_ALWAYS_UNMODELLED)
    if sh is None:
        not_modelled.insert(0, f"{central_body}_spherical_harmonics (central body is a point mass)")
    if central_body not in bodies:
        not_modelled.insert(0, f"{central_body}_gravity (central body not in perturbing_bodies)")

    return {
        "central_body": central_body,
        "frame": frame,
        "forces": forces,
        "summary": {
            "central_point_mass_only": bodies == [central_body] and sh is None,
            "central_J2": central_body in bodies and "J2" in central_terms,
            "central_J22": central_body in bodies and any("J22" in t for t in central_terms),
            "earth_J2": central_body == "Earth" and "J2" in central_terms,
            "earth_J22": central_body == "Earth" and any("J22" in t for t in central_terms),
            "sun_gravity": "Sun" in bodies,
            "moon_gravity": "Moon" in bodies,
            "third_bodies": [b for b in bodies if b != central_body],
        },
        "not_modelled": not_modelled,
    }


# --------------------------------------------------------------------------
# Integrator: settings, step statistics, step convergence
# --------------------------------------------------------------------------

def _segment_slices(n: int, segments: Optional[Iterable[Sequence[int]]]):
    if not segments:
        return [(0, n)]
    return [(int(a), int(b)) for a, b in segments if int(b) - int(a) >= 1]


def step_statistics(epochs, segments=None) -> dict:
    """Realised integration-step statistics from a saved state history
    (tudat saves every accepted step). ``segments`` are ``(start, stop)``
    index ranges of independent coast arcs, so burn seams are not counted
    as zero-length steps."""
    epochs = np.asarray(epochs, dtype=float)
    steps = []
    for a, b in _segment_slices(len(epochs), segments):
        d = np.diff(epochs[a:b])
        steps.append(d[d > 0.0])
    steps = np.concatenate(steps) if steps else np.empty(0)
    if steps.size == 0:
        return {"n_steps": 0, "min_s": None, "max_s": None, "mean_s": None, "median_s": None,
                "n_segments": len(_segment_slices(len(epochs), segments))}
    return {
        "n_steps": int(steps.size),
        "min_s": float(steps.min()),
        "max_s": float(steps.max()),
        "mean_s": float(steps.mean()),
        "median_s": float(np.median(steps)),
        "n_segments": len(_segment_slices(len(epochs), segments)),
    }


def step_convergence(
    nominal_final_state,
    refined_final_state,
    method: str,
    refined_settings: Mapping,
    pos_abs_m: float = CONVERGENCE_POS_ABS_M,
    pos_rel: float = CONVERGENCE_POS_REL,
    vel_abs_m_s: float = CONVERGENCE_VEL_ABS_M_S,
    vel_rel: float = CONVERGENCE_VEL_REL,
) -> dict:
    """Integration-error estimate: difference between the nominal final state
    and a re-run with tightened integrator settings."""
    a = np.asarray(nominal_final_state, dtype=float)
    b = np.asarray(refined_final_state, dtype=float)
    dr = float(np.linalg.norm(a[:3] - b[:3]))
    dv = float(np.linalg.norm(a[3:] - b[3:]))
    pos_tol = max(pos_abs_m, pos_rel * float(np.linalg.norm(b[:3])))
    vel_tol = max(vel_abs_m_s, vel_rel * float(np.linalg.norm(b[3:])))
    return {
        "checked": True,
        "method": method,
        "refined_settings": dict(refined_settings),
        "delta_position_m": dr,
        "delta_velocity_m_s": dv,
        "position_tolerance_m": pos_tol,
        "velocity_tolerance_m_s": vel_tol,
        "converged": bool(dr <= pos_tol and dv <= vel_tol),
    }


def states_at_common_epoch(a, b):
    """``(epoch, state_a, state_b)`` at the latest epoch both histories cover.

    The run that ends later is interpolated (cubic Hermite dense output,
    impulsive-burn seams respected) down to the shared epoch; the run that
    ends there contributes its own final state unchanged."""
    from orbitopt.verify.events import HermiteDenseOutput

    ta, tb = float(a.epochs[-1]), float(b.epochs[-1])
    t = min(ta, tb)

    def at(res, t_end):
        if t_end == t:
            return np.asarray(res.final_state, dtype=float)
        return HermiteDenseOutput(res.epochs, res.states)(t)

    return t, at(a, ta), at(b, tb)


def convergence_not_checked(reason: str) -> dict:
    return {"checked": False, "reason": reason, "converged": None}


# --------------------------------------------------------------------------
# Energy / angular momentum
# --------------------------------------------------------------------------

def conservation_errors(states, mu: float, segments=None) -> dict:
    """Max relative drift of the two-body specific energy
    ``v^2/2 - mu/r`` and of the specific angular momentum vector ``r x v``
    within each coast segment (impulsive burns legitimately change both, so
    every segment is measured against its own first sample)."""
    states = np.asarray(states, dtype=float)
    e_drift = 0.0
    h_drift = 0.0
    per_segment = []
    for a, b in _segment_slices(len(states), segments):
        r = states[a:b, :3]
        v = states[a:b, 3:]
        rn = np.linalg.norm(r, axis=1)
        energy = 0.5 * np.einsum("ij,ij->i", v, v) - mu / rn
        h = np.cross(r, v)
        e0 = energy[0]
        h0 = h[0]
        de = float(np.max(np.abs(energy - e0)) / max(abs(e0), 1e-300))
        dh = float(np.max(np.linalg.norm(h - h0, axis=1)) / max(float(np.linalg.norm(h0)), 1e-300))
        per_segment.append({
            "start_index": a, "stop_index": b,
            "specific_energy_initial_J_kg": float(e0),
            "angular_momentum_initial_m2_s": float(np.linalg.norm(h0)),
            "energy_rel_drift_max": de,
            "angular_momentum_rel_drift_max": dh,
        })
        e_drift = max(e_drift, de)
        h_drift = max(h_drift, dh)
    return {
        "mu_m3_s2": float(mu),
        "energy_rel_drift_max": e_drift,
        "angular_momentum_rel_drift_max": h_drift,
        "segments": per_segment,
    }


def conservation_section(
    model_states,
    mu: float,
    segments,
    inventory: Mapping,
    two_body_states=None,
    two_body_segments=None,
    rel_tol: float = CONSERVATION_REL_TOL,
) -> dict:
    """Energy / angular-momentum block. The propagated-model numbers are only
    a numerical-error measure when the dynamics are a pure central point
    mass; otherwise they mix physical perturbation with integration error and
    are reported as a diagnostic (``applicable: false``)."""
    applicable = bool(inventory.get("summary", {}).get("central_point_mass_only"))
    model = conservation_errors(model_states, mu, segments)
    model["applicable"] = applicable
    model["reason"] = (
        "central point-mass dynamics: energy and angular momentum are integrals of motion"
        if applicable else
        "perturbed dynamics (third bodies / harmonics) do not conserve two-body energy or "
        "angular momentum; values include the physical perturbation and are a diagnostic only"
    )
    if applicable:
        model["within_tolerance"] = bool(
            model["energy_rel_drift_max"] <= rel_tol and model["angular_momentum_rel_drift_max"] <= rel_tol
        )
    out = {"tolerance_rel": rel_tol, "propagated_model": model}
    if two_body_states is not None:
        tb = conservation_errors(two_body_states, mu, two_body_segments)
        tb["applicable"] = True
        tb["reason"] = ("same initial state and integrator re-run under the central body's point "
                        "mass only; drift here is pure integration error")
        tb["within_tolerance"] = bool(
            tb["energy_rel_drift_max"] <= rel_tol and tb["angular_momentum_rel_drift_max"] <= rel_tol
        )
        out["two_body_reference"] = tb
    return out


# --------------------------------------------------------------------------
# One propagation
# --------------------------------------------------------------------------

# Tightening applied by the step-convergence re-run.
CONVERGENCE_TOL_FACTOR = 0.01  # adaptive: rtol and atol x 0.01
CONVERGENCE_STEP_FACTOR = 0.5  # fixed-step: step / 2


def propagation_section(
    result,
    label: str,
    mu: float,
    convergence_check: bool = True,
    conservation_check: bool = True,
) -> dict:
    """One ``propagations[]`` entry of a report.

    ``result`` is duck-typed on :class:`orbitopt.verify.tudat_propagate.
    PropagationResult`: ``epochs``, ``states``, ``final_state``,
    ``segments`` (coast-arc index ranges), ``model`` (dict with
    ``central_body``, ``perturbing_bodies``, ``earth_spherical_harmonic``,
    ``integrator`` -- see ``tudat_propagate.integrator_description`` -- and
    optionally ``function``/``initial_state``/``initial_epoch_ephemeris_s``/
    ``duration_s``/``arcs``) and ``rerun(**overrides)`` which re-propagates
    identically except for ``rtol``/``atol``/``step_size``/
    ``perturbing_bodies``/``earth_spherical_harmonic``.

    Costs up to two extra propagations: the step-convergence re-run
    (tolerances x0.01, or half the step for a fixed-step integrator) and,
    for perturbed dynamics, a central-point-mass-only re-run whose energy /
    angular-momentum drift is the applicable conservation check."""
    model = result.model
    central = model["central_body"]
    inventory = perturbation_inventory(central, model["perturbing_bodies"], model.get("earth_spherical_harmonic"))
    integ = dict(model["integrator"])
    integ["steps"] = step_statistics(result.epochs, result.segments)
    rerun = getattr(result, "rerun", None)

    if not convergence_check:
        integ["convergence"] = convergence_not_checked("disabled by caller")
    elif rerun is None:
        integ["convergence"] = convergence_not_checked("no re-run hook on this result")
    else:
        if integ["adaptive"]:
            refined = {"rtol": integ["rtol"] * CONVERGENCE_TOL_FACTOR, "atol": integ["atol"] * CONVERGENCE_TOL_FACTOR}
            method = f"re-run with rtol/atol x{CONVERGENCE_TOL_FACTOR:g}"
        else:
            refined = {"step_size": integ["step_s"] * CONVERGENCE_STEP_FACTOR}
            method = f"re-run with step x{CONVERGENCE_STEP_FACTOR:g}"
        tight = rerun(**refined)
        # A fixed-step run stops at the first step boundary at/after the
        # requested end, so the nominal and refined runs can end at different
        # epochs (1020 s vs 1005 s for a 1000 s request). Comparing their raw
        # final states would count ordinary orbital motion as integration
        # error, so both are evaluated at the latest epoch they share.
        t_cmp, nominal_state, tight_state = states_at_common_epoch(result, tight)
        integ["convergence"] = step_convergence(nominal_state, tight_state, method, refined)
        integ["convergence"]["compared_at_epoch_s"] = t_cmp

    two_body = None
    if conservation_check:
        if inventory["summary"]["central_point_mass_only"]:
            two_body = result
        elif rerun is not None:
            two_body = rerun(perturbing_bodies=(central,), earth_spherical_harmonic=None)
    conservation = conservation_section(
        result.states, mu, result.segments, inventory,
        two_body_states=None if two_body is None else two_body.states,
        two_body_segments=None if two_body is None else two_body.segments,
    )

    return {
        "label": label,
        "function": model.get("function"),
        "initial_epoch_ephemeris_s": model.get("initial_epoch_ephemeris_s"),
        "duration_s": model.get("duration_s"),
        "initial_state": model.get("initial_state"),
        "final_state": result.final_state,
        "arcs": model.get("arcs"),
        "dynamics": inventory,
        "integrator": integ,
        "conservation": conservation,
    }


# --------------------------------------------------------------------------
# Residuals
# --------------------------------------------------------------------------

def state_residual(name: str, reference: str, state=None, reference_state=None,
                   position=None, reference_position=None, tolerance: Optional[Mapping] = None,
                   category: str = CHECK_MODEL, evidence: Optional[str] = None,
                   scope: Optional[str] = None, phases: Optional[Sequence[str]] = None) -> dict:
    """Position / velocity residual (numerical minus reference). Pass full
    6-vectors (``state``/``reference_state``) or positions only.

    ``category`` classifies the check the residual is (task #243): the
    default, ``model_agreement``, fits a residual against another model's
    prediction (a reference propagator, the screening orbit, an aim point);
    pass ``numerical_consistency`` when the reference is the same model
    re-run by independent numerics, or ``historical_validation`` when it is
    published real-world data.

    ``evidence`` refines that with what the reference is *made of* (task #263,
    one of :data:`EVIDENCE_TYPES`); it defaults from ``category`` -- so pass it
    explicitly to say this model-agreement residual rests on an analytic closed
    form (``analytical``) or on a design aim point (``design_input``, i.e. NOT
    independent validation). ``scope`` and ``phases`` record the model range and
    mission phases the comparison covers, so the verdict states what it does and
    does not speak to; the residual carries an ``evidence`` block spelling out
    its type, independence, scope, phases and what it cannot prove."""
    if state is not None:
        s = np.asarray(state, dtype=float)
        ref = np.asarray(reference_state, dtype=float)
        dr = s[:3] - ref[:3]
        dv = s[3:] - ref[3:]
    else:
        dr = np.asarray(position, dtype=float) - np.asarray(reference_position, dtype=float)
        dv = None
    etype = evidence or default_evidence(category)
    qual = EVIDENCE_QUALIFICATION[etype]
    block = {"type": etype, "independent": bool(qual["independent"]),
             "cannot_prove": qual["cannot_prove"]}
    if scope is not None:
        block["scope"] = str(scope)
    if phases is not None:
        block["phases"] = [str(p) for p in phases]
    out = {
        "name": name,
        "reference": reference,
        "category": category,
        "evidence": block,
        "position_residual_m": float(np.linalg.norm(dr)),
        "position_residual_vector_m": dr,
        "velocity_residual_m_s": None if dv is None else float(np.linalg.norm(dv)),
        "velocity_residual_vector_m_s": dv,
    }
    if tolerance:
        out["tolerance"] = dict(tolerance)
        ok = True
        if "position_m" in tolerance:
            ok &= out["position_residual_m"] <= float(tolerance["position_m"])
        if "velocity_m_s" in tolerance and out["velocity_residual_m_s"] is not None:
            ok &= out["velocity_residual_m_s"] <= float(tolerance["velocity_m_s"])
        out["within_tolerance"] = bool(ok)
    return out


# --------------------------------------------------------------------------
# Assembly / IO
# --------------------------------------------------------------------------

def _version(dist: str):
    """The version of ``dist`` as recorded by :mod:`orbitopt.flightlab.toolchain`
    -- a conda build that ships no dist-info (tudatpy) still reports its real
    version from the environment's ``conda-meta``, and an importable-but-
    unrecorded package says so plainly. ``None`` therefore means genuinely not
    installed, which is what a null in this block has always claimed."""
    from orbitopt.flightlab.toolchain import ABSENT, installed_version

    version = installed_version(dist)
    return None if version == ABSENT else version


def provenance() -> dict:
    """Library versions that determine the numbers (no timestamps, so the
    report stays byte-reproducible)."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "orbitopt": _version("orbitopt"),
        "tudatpy": _version("tudatpy"),
        "pykep": _version("pykep"),
    }


def _residual_evidence_type(residual: Mapping):
    """The evidence type a residual carries, accepting either the full
    qualification block (from :func:`state_residual`) or a bare type string a
    hand-written residual dict sets next to its ``category``; ``None`` lets the
    status derive it from the residual's category."""
    ev = residual.get("evidence")
    if isinstance(ev, Mapping):
        return ev.get("type")
    return ev


def _status(propagations, residuals, targets) -> dict:
    checks, categories, evidence = {}, {}, {}
    def add(name: str, ok: bool, category: str) -> None:
        checks[name] = bool(ok)
        categories[name] = category
    for p in propagations:
        lab = p.get("label", "propagation")
        conv = p.get("integrator", {}).get("convergence", {})
        if conv.get("checked"):
            add(f"{lab}.step_convergence", bool(conv.get("converged")), CHECK_NUMERICAL)
        cons = p.get("conservation", {})
        tb = cons.get("two_body_reference")
        if tb is not None:
            add(f"{lab}.two_body_conservation", bool(tb.get("within_tolerance")), CHECK_NUMERICAL)
        pm = cons.get("propagated_model", {})
        if pm.get("applicable"):
            add(f"{lab}.model_conservation", bool(pm.get("within_tolerance")), CHECK_NUMERICAL)
    for r in residuals:
        if "within_tolerance" in r:
            category = r.get("category")
            if category not in CHECK_CATEGORIES:
                raise ValueError(
                    f"residual {r.get('name')!r} needs a check category (one of {CHECK_CATEGORIES}); "
                    "pass category= to state_residual or set 'category' on the residual dict")
            add(f"residual.{r['name']}", bool(r["within_tolerance"]), category)
            etype = _residual_evidence_type(r)
            if etype is not None:
                evidence[f"residual.{r['name']}"] = etype
    if isinstance(targets, Mapping) and "converged" in targets:
        add("targeting.converged", bool(targets["converged"]), CHECK_NUMERICAL)
    return status_from_checks(checks, categories, evidence)


# Tri-state verification outcome (task #162). "No checks ran" is *not* a
# pass: a report without evidence is ``not_checked``, and ``passed`` (kept
# as a boolean for existing consumers) is true only for RESULT_PASSED.
RESULT_PASSED = "passed"
RESULT_FAILED = "failed"
RESULT_NOT_CHECKED = "not_checked"
RESULTS = (RESULT_PASSED, RESULT_FAILED, RESULT_NOT_CHECKED)

def _category_rollup(checks: Mapping, categories: Mapping) -> dict:
    """Per-category outcome of a categorised check set: all three categories
    always appear; one with no check is ``not_checked`` (absent evidence is
    never counted as a pass)."""
    out = {}
    for cat in CHECK_CATEGORIES:
        names = [n for n, c in categories.items() if c == cat]
        n_ok = sum(bool(checks[n]) for n in names)
        if not names:
            result = RESULT_NOT_CHECKED
        elif n_ok == len(names):
            result = RESULT_PASSED
        else:
            result = RESULT_FAILED
        out[cat] = {"result": result, "checks_passed": int(n_ok), "checks_total": len(names)}
    return out


def _rollup_by(checks: Mapping, kinds: Mapping, keys: Sequence[str]) -> dict:
    """Result / checks_passed / checks_total per kind, every kind present, a
    kind with no check ``not_checked``."""
    out = {}
    for kind in keys:
        names = [n for n, k in kinds.items() if k == kind]
        n_ok = sum(bool(checks[n]) for n in names)
        if not names:
            result = RESULT_NOT_CHECKED
        elif n_ok == len(names):
            result = RESULT_PASSED
        else:
            result = RESULT_FAILED
        out[kind] = {"result": result, "checks_passed": int(n_ok), "checks_total": len(names)}
    return out


def _evidence_rollup(checks: Mapping, evidence: Mapping) -> dict:
    """Per-evidence-type rollup (task #263): every evidence type appears, the
    qualification that bounds how far a pass reaches rides with it, and an
    ``independent`` flag marks the types that count as validation."""
    roll = _rollup_by(checks, evidence, EVIDENCE_TYPES)
    for kind, entry in roll.items():
        entry.update(EVIDENCE_QUALIFICATION[kind])
    return roll


def _independent_validation(roll: Mapping) -> dict:
    """The one rollup a reader of a pass must check (task #263): evidence from
    OUTSIDE the model under test -- historical telemetry or an independent
    model. ``not_checked`` here means the report is internally consistent but
    never independently validated, however green its other categories."""
    passed = sum(roll[k]["checks_passed"] for k in INDEPENDENT_EVIDENCE)
    total = sum(roll[k]["checks_total"] for k in INDEPENDENT_EVIDENCE)
    failed = any(roll[k]["result"] == RESULT_FAILED for k in INDEPENDENT_EVIDENCE)
    if not total:
        result = RESULT_NOT_CHECKED
    elif failed:
        result = RESULT_FAILED
    elif passed == total:
        result = RESULT_PASSED
    else:
        result = RESULT_FAILED
    return {"result": result, "checks_passed": int(passed), "checks_total": int(total)}


def status_from_checks(checks: Mapping, categories: Optional[Mapping] = None,
                       evidence: Optional[Mapping] = None) -> dict:
    """Status block for a mapping of ``check name -> bool``, a mapping of
    ``check name -> category`` (one of :data:`CHECK_CATEGORIES`) and an
    optional mapping of ``check name -> evidence type`` (one of
    :data:`EVIDENCE_TYPES`): ``failed`` if any check failed, ``passed`` if
    there is at least one check and all passed, ``not_checked`` if there are no
    checks at all.

    Every check must be categorised (task #243). Its *evidence type* (task
    #263) defaults from its category (:func:`default_evidence`) when not given
    -- so pass ``evidence=`` to say a model-agreement check really rests on an
    analytic closed form (``analytical``) or on the design input itself
    (``design_input``) rather than an independent model. The status carries
    ``check_categories`` + ``categories`` (the three kinds), ``check_evidence``
    + ``evidence`` (the five evidence types, each with its qualification) and
    ``independent_validation`` (whether any validation-grade evidence backed
    the result)."""
    checks = {str(k): bool(v) for k, v in checks.items()}
    cats = {str(k): str(v) for k, v in (categories or {}).items()}
    missing = sorted(set(checks) - set(cats))
    if missing:
        raise ValueError(f"checks without a verification category: {missing} "
                         f"(every check must be one of {CHECK_CATEGORIES})")
    unknown = sorted(set(cats) - set(checks))
    if unknown:
        raise ValueError(f"categories given for unknown checks: {unknown}")
    bad = sorted({c for c in cats.values() if c not in CHECK_CATEGORIES})
    if bad:
        raise ValueError(f"unknown check categories: {bad} (must be one of {CHECK_CATEGORIES})")
    ev = {str(k): str(v) for k, v in (evidence or {}).items()}
    unknown_ev = sorted(set(ev) - set(checks))
    if unknown_ev:
        raise ValueError(f"evidence types given for unknown checks: {unknown_ev}")
    bad_ev = sorted({e for e in ev.values() if e not in EVIDENCE_TYPES})
    if bad_ev:
        raise ValueError(f"unknown evidence types: {bad_ev} (must be one of {EVIDENCE_TYPES})")
    ev = {name: ev.get(name) or default_evidence(cats[name]) for name in checks}
    if not checks:
        result = RESULT_NOT_CHECKED
    elif all(checks.values()):
        result = RESULT_PASSED
    else:
        result = RESULT_FAILED
    roll = _evidence_rollup(checks, ev)
    return {"result": result, "passed": result == RESULT_PASSED, "checks": checks,
            "check_categories": cats, "categories": _category_rollup(checks, cats),
            "check_evidence": ev, "evidence": roll,
            "independent_validation": _independent_validation(roll)}


def status_result(status: Mapping) -> str:
    """Tri-state result of a status block (falls back to recomputing it from
    ``checks`` for documents written before the ``result`` field existed)."""
    result = status.get("result")
    if result in RESULTS:
        return result
    checks = status.get("checks") or {}
    if not checks:
        return RESULT_NOT_CHECKED
    return RESULT_PASSED if all(bool(v) for v in checks.values()) else RESULT_FAILED


def format_status(status: Mapping) -> str:
    """Human-readable one-liner, e.g. ``PASSED (5/5 checks)``."""
    checks = status.get("checks") or {}
    n_ok = sum(bool(v) for v in checks.values())
    label = status_result(status).replace("_", " ").upper()
    return f"{label} ({n_ok}/{len(checks)} checks passed)"


def build_report(
    kind: str,
    label: str,
    propagations: Sequence[Mapping],
    residuals: Sequence[Mapping] = (),
    inputs: Optional[Mapping] = None,
    targets: Optional[Mapping] = None,
) -> dict:
    """Assemble a complete, digested report. ``propagations`` are sections
    from :func:`orbitopt.verify.tudat_propagate.propagation_report_section` (a thin
wrapper around :func:`propagation_section`)."""
    propagations = [to_jsonable(p) for p in propagations]
    residuals = [to_jsonable(r) for r in residuals]
    targets = to_jsonable(targets) if targets is not None else None
    report = {
        "schemaVersion": SCHEMA_VERSION,
        "kind": str(kind),
        "label": str(label),
        "status": _status(propagations, residuals, targets),
        "inputs": to_jsonable(inputs or {}),
        "propagations": propagations,
        "residuals": residuals,
        "targets": targets,
        "provenance": provenance(),
    }
    report["digest"] = report_digest(report)
    return report


def write_report(report: Mapping, path) -> Path:
    path = Path(path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(report), sort_keys=True, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    return path


def read_report(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _slug(text: str) -> str:
    keep = [c if c.isalnum() or c in "-_" else "-" for c in str(text).lower()]
    return "-".join(filter(None, "".join(keep).split("-"))) or "report"


def default_report_path(report: Mapping, directory) -> Path:
    return Path(directory) / f"{_slug(report['kind'])}-{_slug(report['label'])}.verification.json"


def finalize_report(report: dict, path=None) -> dict:
    """Write ``report`` to ``path`` if given, else into
    ``$ORBITOPT_VERIFICATION_REPORT_DIR`` if that is set. Returns it."""
    if path is not None:
        write_report(report, path)
    elif os.environ.get(REPORT_DIR_ENV):
        write_report(report, default_report_path(report, os.environ[REPORT_DIR_ENV]))
    return report


def schema_path() -> Path:
    return Path(__file__).resolve().parent.parent / "schemas" / SCHEMA_FILE


def validate_report(report: Mapping) -> None:
    """Raise ``jsonschema.ValidationError`` if ``report`` doesn't match the
    schema, or ``ValueError`` if its digest doesn't match its content."""
    import jsonschema

    schema = json.loads(schema_path().read_text(encoding="utf-8"))
    jsonschema.validate(to_jsonable(report), schema)
    if report.get("digest") != report_digest(report):
        raise ValueError("verification report digest does not match its content")


# Scene documents travel through orbitopt.mission_config adapters; reports
# ride along under this private key and are popped off (and written next to
# the scene) by ``orbitopt run`` before the scene is validated / written.
SCENE_REPORTS_KEY = "_verificationReports"


def attach_to_scene(scene: dict, report: Mapping) -> dict:
    scene.setdefault(SCENE_REPORTS_KEY, []).append(report)
    # Surface the tri-state outcome in the subtitle the GUI shows (and the
    # CLI echoes), so "not checked" is never mistaken for "verified".
    status = report.get("status") or {}
    line = (f"Verification report {report.get('kind', '?')}:{report.get('label', '?')}: "
            f"{format_status(status)}")
    scene["subtitle"] = (scene.get("subtitle", "") + "\n" + line) if scene.get("subtitle") else line
    return scene


def pop_from_scene(scene: dict) -> list:
    return list(scene.pop(SCENE_REPORTS_KEY, []) or [])


# The public, viewer-facing copy of a scene's verification (see the scene
# schema's ``verification``). Unlike SCENE_REPORTS_KEY it stays in the scene:
# without it the viewer only had the one-line summaries in the subtitle.
SCENE_VERIFICATION_KEY = "verification"


def publish_to_scene(scene: dict, reports: Sequence[Mapping]) -> dict:
    """Store ``reports`` in ``scene["verification"]`` as
    ``{"result": <tri-state over all reports>, "reports": [...]}`` -- the
    same documents ``orbitopt run`` writes next to the scene. A no-op for no
    reports: the viewer then says "not checked" rather than inventing one."""
    if not reports:
        return scene
    combined = bundle(reports, label=str(scene.get("id", "scene")))
    scene[SCENE_VERIFICATION_KEY] = {"result": combined["status"]["result"], "reports": combined["reports"]}
    return scene


def _merged_rollups(reports: Sequence[Mapping], field: str, keys: Sequence[str]) -> dict:
    """Combine the sub-reports' ``{kind -> {result, checks_passed, checks_total}}``
    rollups under ``field``: only a report that ran a check in a kind votes in
    it (failed beats passed), and a kind no report checked -- or a report
    written before checks were labelled that way -- leaves it ``not_checked``:
    absent evidence, never a pass."""
    out = {kind: {"result": RESULT_NOT_CHECKED, "checks_passed": 0, "checks_total": 0} for kind in keys}
    for r in reports:
        for kind, roll in ((r.get("status") or {}).get(field) or {}).items():
            if kind not in out or not roll.get("checks_total"):
                continue
            merged = out[kind]
            merged["checks_passed"] += int(roll.get("checks_passed") or 0)
            merged["checks_total"] += int(roll["checks_total"])
            if roll.get("result") == RESULT_FAILED:
                merged["result"] = RESULT_FAILED
            elif merged["result"] != RESULT_FAILED:
                merged["result"] = RESULT_PASSED
    return out


def _merged_category_rollups(reports: Sequence[Mapping]) -> dict:
    """Task #243: the per-category rollup over all sub-reports' checks."""
    return _merged_rollups(reports, "categories", CHECK_CATEGORIES)


def _merged_evidence_rollups(reports: Sequence[Mapping]) -> dict:
    """Task #263: the per-evidence-type rollup over all sub-reports' checks,
    with each type's qualification re-attached and the independent-validation
    rollup recomputed over the merged counts."""
    merged = _merged_rollups(reports, "evidence", EVIDENCE_TYPES)
    for kind, entry in merged.items():
        entry.update(EVIDENCE_QUALIFICATION[kind])
    return merged


def bundle(reports: Sequence[Mapping], label: str) -> dict:
    """Combine several reports (one per verification stage) into one file."""
    reports = [to_jsonable(r) for r in reports]
    results = [status_result(r["status"]) for r in reports]
    # A sub-report that was not checked contributes no check (it is neither
    # a pass nor a failure); it does keep the bundle from being "passed".
    checks = {f"{r['kind']}:{r['label']}": res == RESULT_PASSED
              for r, res in zip(reports, results) if res != RESULT_NOT_CHECKED}
    if RESULT_FAILED in results:
        result = RESULT_FAILED
    elif not results or RESULT_NOT_CHECKED in results:
        result = RESULT_NOT_CHECKED
    else:
        result = RESULT_PASSED
    evidence = _merged_evidence_rollups(reports)
    doc = {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "bundle",
        "label": str(label),
        # The bundle's own checks are per-report verdicts, so no per-check
        # category; the per-category and per-evidence rollups over all
        # sub-reports' checks are (tasks #243 and #263).
        "status": {"result": result, "passed": result == RESULT_PASSED, "checks": checks,
                   "categories": _merged_category_rollups(reports),
                   "evidence": evidence, "independent_validation": _independent_validation(evidence)},
        "reports": reports,
        "provenance": provenance(),
    }
    doc["digest"] = report_digest(doc)
    return doc
