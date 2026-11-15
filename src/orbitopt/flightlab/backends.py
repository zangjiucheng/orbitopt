"""E1-01 (task #250): the canonical identity of every Flight Lab ascent backend.

Before this module, ``6DOF`` named two different things: ``orbitopt flight
simulate --backend 6DOF`` flew the JSBSim adapter (zero-aero spike models;
since E1-04 the production path is a generated aircraft with the packaged
aero document compiled in — ADR flightlab-0007),
:mod:`orbitopt.flightlab.adapters.sixdof`) while ``orbitopt flight mission
--backend 6DOF`` flew the native RK4 rigid-body plant
(:mod:`orbitopt.flightlab.sixdof`). The same label with two meanings is a
verification hazard: two runs can look like they agree about their fidelity
while they disagree about their physics. This module fixes one unambiguous
identity per physics engine; the coarse ``3DOF``/``6DOF`` wording survives
only as a *deprecated input alias* inside each entry point that historically
took it, resolved here, never echoed into a run, report or scene.

The identities (lower-case, exactly as they are recorded in every output):

``jsbsim``             JSBSim 6-DOF adapter (F-01): config-driven scenarios
                       on the E4 spike models (zero-aero base or the
                       generated production aircraft with the aero document's
                       tables compiled in), ECEF
                       states, opt-in ``jsbsim`` extra.
``native-rigidbody``   the native RK4 rigid-body plant (F-02/F-04/F-06/F-07):
                       six states + coefficient-table aero + TVC actuator,
                       the reference plant of the 6-DOF studies.
``native-pointmass``    the native point-mass twin (F-05/F-07): the same
                       plant's kinematic-attitude 3-DOF model the 6-DOF
                       studies compare against -- not the OT adapter.
``opentsiolkovsky``     the OpenTsiolkovsky 3-DOF adapter (FL-004): the
                       pinned external binary behind the Gate-B pipeline and
                       the recovery chain.

Every produced :class:`~orbitopt.flightlab.contract.FlightRun` records its
identity in the run manifest (``manifest.json`` ``backend``), which the
verification report (``inputs.backend``) and the Mission Control scene
(``flightRun.backend``) inherit -- so any artefact says which physics engine
actually flew it. ``FIDELITY`` below is the machine-readable form of the
fidelity-boundary table in ``docs/flightlab.md``.
"""
from __future__ import annotations

BACKEND_JSBSIM = "jsbsim"
BACKEND_NATIVE_RIGIDBODY = "native-rigidbody"
BACKEND_NATIVE_POINTMASS = "native-pointmass"
BACKEND_OPENTSIOLOVSKY = "opentsiolkovsky"

#: Every canonical backend identity, in fidelity order (3-DOF lineages first).
BACKENDS: tuple[str, ...] = (
    BACKEND_OPENTSIOLOVSKY,
    BACKEND_NATIVE_POINTMASS,
    BACKEND_JSBSIM,
    BACKEND_NATIVE_RIGIDBODY,
)

#: What each backend physically models and where its word stops (the
#: fidelity-boundary table of docs/flightlab.md, same words).
FIDELITY: dict[str, dict[str, str]] = {
    BACKEND_OPENTSIOLOVSKY: {
        "label": "OpenTsiolkovsky 3-DOF point mass",
        "models": "point-mass ascent, prescribed attitude program, Cd(Mach) "
                  "drag; GCRS states from the pinned v1.0.0 external binary",
        "boundaries": "no attitude dynamics, no TVC, no staging forces; the "
                      "Gate-B pipeline and the FL-304 recovery chain build on "
                      "this lineage",
    },
    BACKEND_NATIVE_POINTMASS: {
        "label": "native point-mass twin",
        "models": "point-mass ascent on the native plant (US76 + J2 + "
                  "rotation), kinematic program attitude, q-bar S CA/CN aero",
        "boundaries": "no attitude states (the program IS the attitude) -- the "
                      "3-DOF side of the F-05/F-07 fidelity comparison, not "
                      "the OpenTsiolkovsky adapter",
    },
    BACKEND_JSBSIM: {
        "label": "JSBSim 6-DOF (E4 spike models)",
        "models": "full JSBSim rigid-body dynamics with gimballed TVC and "
                  "propellant-driven mass/CG/inertia on the E4 spike models, "
                  "ECEF states",
        "boundaries": "the packaged base aircraft fly with ZERO aero (a "
                      "TVC/mass paper test); the generated production "
                      "aircraft flies the packaged aero document "
                      "(ADR flightlab-0007); outside the optimization loop; "
                      "opt-in jsbsim==1.3.3 extra",
    },
    BACKEND_NATIVE_RIGIDBODY: {
        "label": "native RK4 rigid body",
        "models": "six-DOF rigid body with coefficient-table aero, the F-04 "
                  "tracker and a lag/rate-limited TVC actuator; US76 + J2 + "
                  "rotation; ECEF states",
        "boundaries": "the reference plant of the native studies (F-02/F-04/"
                      "F-06/F-07): illustrative coefficients, basic tracker "
                      "without roll control; not the JSBSim backend",
    },
}


def describe(backend: str | None) -> str:
    """One-line fidelity description of a canonical backend identity (the
    empty string for None, so callers can print unconditionally)."""
    if backend is None:
        return ""
    info = FIDELITY[canonical_backend(backend)]
    return (f"{backend} ({info['label']}): {info['models']} "
            f"-- boundary: {info['boundaries']}")


def canonical_backend(name: str, *, aliases: dict[str, str] | None = None,
                      allowed: tuple[str, ...] | None = None) -> str:
    """Resolve a user-supplied backend selector to its canonical identity.
    Matching is case-insensitive and whitespace-tolerant; ``aliases`` maps
    deprecated per-entry-point spellings (e.g. ``{"6DOF": "jsbsim"}`` for
    ``simulate``) onto canonical identities; ``allowed`` restricts the choice
    (and the error message) to the identities one entry point actually
    offers. Raises ``ValueError`` naming the accepted identities otherwise."""
    accepted = tuple(allowed) if allowed else BACKENDS
    key = str(name).strip().lower()
    if key in accepted:
        return key
    if aliases:
        hit = {k.lower(): v for k, v in aliases.items()}.get(key)
        if hit is not None and (allowed is None or hit in accepted):
            return hit
    hint = ""
    if aliases and key in ("3dof", "6dof") and allowed is not None:
        hint = (f" -- '{name}' is a deprecated ambiguous alias; "
                f"this entry point means one of: {', '.join(allowed)}")
    raise ValueError(f"backend {name!r} is not supported; choose "
                     f"{', '.join(accepted)}{hint}")
