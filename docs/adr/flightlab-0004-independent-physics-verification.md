# ADR flightlab-0004: Separate backend self-check from independent physics verification

- **Status:** Accepted
- **Date:** 2026-11-14
- **Task:** E4-01 Independent 6-DOF Verification (#261), parent epic E4 Verification Assurance (#249)
- **Contract:** Flight Lab v1 `verification.json` (`schema_version` stays `1.0`; `backend_self_check` is a new **optional** property)

## Context and problem

The Flight Lab run contract carries three independent Gate-B verdicts
(`optimizer_converged`, `physics_verified`, `mission_constraints_passed`). Two
backends fill `physics_verified` with **different meanings**:

- **3-DOF (OpenTsiolkovsky):** `physics_verified` is the outcome of the
  independent tudatpy insertion **re-fly** — a *different* propagator
  re-integrates the coast/insertion state and the Gate-G5 element-drift checks
  judge it (`orbitopt.flightlab.simulate.orbit_refly`). This is genuine
  cross-model verification.
- **6-DOF (JSBSim, formal backend, task #226):** `physics_verified` was simply
  the conjunction of JSBSim's *own* in-run checks (finite states, mass
  bookkeeping, handoff continuity, tracking bounds). That is the integrator
  **vouching for itself**, not an independent check — the same field name meant
  two different things, and a green 6-DOF run silently claimed a guarantee it
  never had.

Neither claim survived acceptance review: a verification flag must reflect
corroboration by something other than the producer, or it must not be set.

## Decision

Split the two ideas into two fields, and give the 6-DOF backend the same
*independent* physics check the 3-DOF lineage already had:

1. **New field `backend_self_check` (optional bool/null).** The conjunction of
   the producing backend's own checks. Written by every adapter
   (`opentsiolkovsky._verification`, `sixdof.run_ascent`) and by the spike
   drivers. Because it is optional and read with `.get(...)`, run directories
   written before this field existed read back as `None` and still validate.

2. **`physics_verified` now means one thing on both backends:** the result of an
   independent re-flight / cross-model check — never the producer's own word.
   `None` = no independent re-fly ran (not assessed), never a pass.

3. **6-DOF independent re-fly.** `simulate(backend="6DOF")` calls
   `simulate.orbit_refly_6dof`, which:
   - reads the run's coast/insertion state (ECEF, as the 6-DOF tracks it),
   - rotates Earth-fixed → GCRS through the frame bridge
     (`orbitopt.flightlab.frames.to_gcrs`, carrying the `ω × r` velocity term),
   - feeds that GCRS state into the **same** `_refly_gcrs_state` core (and the
     same tudatpy point-mass propagation and Gate-G5 drift checks) that
     `orbit_refly` uses for the 3-DOF state.
   A vertical hop / suborbital scenario whose coast state is not a closed orbit
   reads as **not assessed** (`physics_verified=None`), never a physics failure.
   If the frame bridge or tudatpy is unavailable the check is not assessed
   (`REFLY_BACKEND_ERROR` / `REFLY_BACKEND_MISSING`) — never a fake `True`.

4. `checks.evaluate` and the recovery merge **preserve** both fields verbatim;
   `simulate` threads `backend_self_check` through alongside the re-fly verdict.

## Consequences

- A green 6-DOF `physics_verified` now carries the same weight as a green 3-DOF
  one: an independent propagator agrees. Where it cannot be assessed, the field
  reads `None` and the run's assumptions say why.
- Backend self-consistency is still recorded and honest — it just moved to
  `backend_self_check`, so a self-vouch is never mistaken for corroboration.
- The field is additive/optional, so existing run directories and the CLI
  readers keep working; `orbitopt flight simulate`/`optimize` print both
  verdicts on separate lines.
- Accepted limitation: the 6-DOF re-fly is a point-mass re-integration of the
  coast state (matching the 3-DOF lineage's own re-fly), not a 6-DOF re-run; it
  verifies the orbital state the ascent produced, not the attitude dynamics.
  Attitude/actuator fidelity is the FL-401/402/403 story (ADR flightlab-0001),
  tracked there as its own checks.

## Evidence index

- Field + semantics: `src/orbitopt/flightlab/contract.py` (`Verification`),
  `src/orbitopt/schemas/flight-result-1.0.json`.
- 3-DOF re-fly core and 6-DOE→GCRS bridge entry:
  `src/orbitopt/flightlab/simulate.py` (`_refly_gcrs_state`, `orbit_refly`,
  `orbit_refly_6dof`, `_simulate_6dof`).
- Adapters record self-check: `adapters/opentsiolkovsky.py`,
  `adapters/sixdof.py`.
- Tests: `tests/test_flightlab_contract.py` (round-trip / legacy-default /
  independence), `tests/test_flightlab_simulate.py` (bridge ordering + failure
  path), `tests/test_flightlab_sixdof_backend.py` (adapter self-check, not
  physics).
