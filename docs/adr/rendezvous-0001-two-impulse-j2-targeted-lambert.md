# ADR rendezvous-0001: two-impulse J2-targeted Lambert with Earth-mu enforced at the boundary (E12-03)

- **Status:** Accepted
- **Date:** 2026-11-24
- **Deciders:** E12 Orbital Rendezvous & Mission Operations (Epic #389), E12-03 (#393)
- **Code:** `src/orbitopt/rendezvous/`, `src/orbitopt/schemas/rendezvous-mission-1.0.json`,
  `src/orbitopt/data/leo_rdv_001.yaml`
- **Evidence:** `tests/test_e12_03_rendezvous_planner.py`, `tests/test_e12_03_lambert_earth_mu.py`
- **Builds on:** ADR relative-0001 (LVLH frames, `propagate_pair`), `orbitopt.lambert.cpu`,
  `physics_io` band checks

## Context and problem

`lambert/cpu.py:solve_lambert_single` defaults to `mu=pk.MU_SUN`. That is right
for every interplanetary scene in the library and catastrophically wrong for a
500 km rendezvous — and *silent* wrong in the sense that pykep will happily
return a "trajectory": it is `physics_io`'s plausibility band that catches the
mixup, and only if the pair actually gets band-checked. E12-03 reuses this
solver for an Earth mission, so the default has to be structurally unreachable,
not merely avoided. Separately, the acceptance is *nonlinear*: the plan must
arrive within 50 m / 0.2 m/s of a 1 km Hold Point under J2 propagation, so any
model used to plan must be targeted against the model used to verify.

## Decision

1. **The mission is a config file, fixed before optimization.**
   `schemas/rendezvous-mission-1.0.json` + `data/leo_rdv_001.yaml` (LEO-RDV-001:
   500 km target, 490 km chaser, 20 km initial separation, 1 km V-bar-aft Hold
   Point, 25 m/s budget, 5400 s drift window, 1400–3600 s transfer window,
   departure gate y = −8000 m, tolerances 50 m / 0.2 m/s). The schema is strict
   (no unknown keys); the loader adds the cross-checks the schema cannot
   express — in particular `dynamics.mu` must **equal the exact packaged Earth
   value** `3.9860044150e14`, so this config family cannot silently name another
   central body.
2. **Plan model == verify model: full `propagate_pair` with configured μ and
   J2.** A two-body Lambert solution is the *initial guess only*. Measurement
   drove this: iterating the aim point (`aim += miss`) under J2 has gain ≈ 1 —
   the differential J2 short-period between transfer and hold orbits is
   ~6.5 km at 1200 s tof and does not shrink — so it never converges; Newton on
   the *departure velocity* against the full propagation converges in ≤ 4
   iterations (numerical Jacobian from one batched 4-pair `propagate_pair`).
   The arrival aim is the predicted chief state at t2 re-expressed as a hover
   at the hold point, so a converged Newton residual *is* the acceptance error.
3. **The optimizer searches the user's window, never outside it.** Golden-section
   over tof ∈ [tof_min, tof_max] minimizing total Δv; a plan that cannot cross
   the drift gate inside `drift_max_s` raises instead of stretching the window.
4. **μ defence is three independent nets.** (a) config equality check above;
   (b) every planner call site passes `mu=cfg.mu` explicitly as a keyword;
   (c) regression tests — a default-μ call on a LEO pair must raise
   `PhysicsIOError`, and a monkeypatch spy on `planner.solve_lambert_single`
   asserts every recorded μ equals the packaged Earth value exactly.

## Consequences

- A LEO pair handed to the solver's *default* μ fails loudly in the sun band
  check (`|r| outside [1e7, 1e13]`); the silent path is closed at both ends.
- Reference plan (deterministic, ~10 s): gate at t1 = 560.6 s, tof = 3389.0 s,
  total Δv = 10.34 m/s (margin 14.66 m/s), arrival position error 5·10⁻⁷ m and
  relative speed 5·10⁻¹⁰ m/s under the verifying propagation — the acceptance
  numbers are met with four orders of magnitude to spare because the planner is
  targeting the same propagator the acceptance runs on.
- The acceptance verdict lives only in `verify_plan` (a failed check must not
  raise — e.g. an over-tight budget flips the check, it does not abort);
  `plan_rendezvous` raises only when *no plan exists* under the config.
- Scope is the near-circular single-rev V-bar family the reference mission
  lives in. Multi-rev transfers, elliptic target orbits, and powered terminal
  guidance need their own admission; `plane_normal` pinning (equatorial-plane
  Lambert) and the `max_revs=0` choice are pinned in the planner, not the config.
