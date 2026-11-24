# ADR relative-0001: LVLH frames, CW as a planning-only model, nonlinear pair acceptance (E12-02)

- **Status:** Accepted
- **Date:** 2026-11-24
- **Deciders:** E12 Orbital Rendezvous & Mission Operations (Epic #389), E12-02 (#392)
- **Code:** `src/orbitopt/physics_io.py`, `src/orbitopt/relative/`
- **Evidence:** `docs/benchmarks/e12_02_cw_validity.json` (+ `.md`),
  `tests/test_e12_02_relative_dynamics.py`, `tests/test_e12_02_cw_validity.py`
- **Builds on:** `physics_io` unit/frame contract (the geocentric band
  checks this ADR deliberately does not touch), `orbitopt.core` constant set

## Context and problem

`physics_io.py` rejected LVLH and VVLH as "a body-fixed orbital frame, not
an Earth-centred one" — correct as long as every interface was heliocentric
or geocentric. E12 rendezvous work needs the opposite: relative states
chief-centred in a frame that *rotates with the chief*. The risk is not
adding the frame, it is what an eager generalisation erodes: the geocentric
band checks are this project's unit-system tripwire, and CW — the obvious
fast model — is exactly the kind of model that gets silently used where it
does not apply.

## Decision

1. **LVLH/RTN join the frame registry as a new `"orbital"` family**, not as
   a mutation of the inertial/rotating dichotomy: x radial-out (R), y
   along-track (T), z orbit-normal (N); RTN ≡ LVLH; VVLH stays a rejected
   spelling with a hint. `check_frame_family` learns one more family label;
   the `EARTH_CENTRED_METRES` spec, its bands and every existing geocentric
   check are byte-identical. New specs `RELATIVE_STATE` (chief-centred,
   band 0 … lunar distance) and `RELATIVE_PAIR` (two absolute states, GCRS
   or EME2000 — J2 needs an equatorial frame) ride the same
   `PhysicsIOSpec` machinery as all others.
2. **`orbitopt.relative` is numpy-only and three modules**: `frames`
   (dcm-based ECI ↔ LVLH with the ω×ρ̇ velocity rule, round-trip exact to
   float64), `cw` (closed-form state transition, batched), `propagation`
   (`propagate_pair`: nonlinear two-spacecraft two-body + optional J2,
   fixed-step RK4 on the stacked 12-state, chief-centred LVLH output).
3. **CW is a planning model, and the boundary is measured, not asserted.**
   The companion benchmark sweeps separation 10 m … 100 km, duration up to
   5 orbits, chief eccentricity 0 … 0.2, with and without J2. Result: the
   linearisation error is linear in separation (~1 % of ρ per orbit, at any
   scale), grows ~t², and is dwarfed by any chief eccentricity ≥ 0.01
   (129 % of a 100 m separation at one orbit). The envelope gate
   (`test_e12_02_cw_validity.py`) pins the claim: circular chief, ≤ 1 km,
   ≤ 1 orbit, error < 25 % of separation — and acceptance runs on
   `propagate_pair`.

## Consequences

- A chief-centred state in an `"orbital"` frame validates like any other
  spec state, so every existing boundary (spec check → raise) holds; but
  the relative bands check *separations*, and a separation crossing the
  band means a unit-system error, not a far-apart pair — the pair spec
  still checks both absolute states against the full geocentric bands.
- CW output is deliberately **not** band-checked (a CW drift is already
  known-wrong at large t; policing it there hides the real error); the
  propagator's absolute states *are*.
- Anything beyond near-circular chiefs or horizons past the envelope (J2
  secular drift over many orbits, eccentric CW, Hill/LP) is out of scope
  for this layer and needs its own admission with the same benchmark
  discipline.
