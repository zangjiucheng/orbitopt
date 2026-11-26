# ADR gnc-0001: five record contracts and a uniform clock for closed-loop GNC (E13-01)

- **Status:** Accepted
- **Date:** 2026-11-26
- **Deciders:** E13 Closed-Loop GNC & Relative Navigation (Epic #416), E13-01 (#418)
- **Code:** `src/orbitopt/gnc/` (`contracts.py`, `clock.py`, `feed.py`, `recording.py`, `sim.py`),
  `src/orbitopt/schemas/gnc-contracts-1.0.json`
- **Evidence:** `tests/test_e13_01_gnc_contracts.py`, `tests/test_e13_01_clock_contract.py`,
  `tests/test_e13_01_truth_isolation.py`, `tests/test_e13_01_replay.py`
- **Builds on:** ADR relative-0001 (`RelativeState`, pair propagation), ADR rendezvous-0004
  (sealed replay), E12-08 canonical-plan digest policy (sha256, 8-significant-digit quantization)

## Context

E12 qualified rendezvous plans open-loop: a plan was verified, then propagated. Closed-loop
guidance and navigation (#416) introduces a plant whose true state must never be an input to the
law that steers it, and a sensor/estimator/control loop whose timing decides what each module was
allowed to know. Two failure modes have to be closed architecturally, not by care in a notebook:

1. **Truth leakage.** If plant truth reaches the guidance law — or the record claimed as an
   observation carries truth instead of a measurement — every covariance and robustness number the
   loop produces is fiction.
2. **Clock drift.** If a dropped frame shifts the next sample, or latency is retro-applied, the
   run's timestamps no longer state what the estimator consumed and when; replay stops being proof.
   E12-10 also measured actuator execution error as the dominant failure source, so what the
   thruster *actually* did needs its own record, separate from what was commanded.

## Decision

1. **Five versioned record kinds** — `TruthSample`, `Observation`, `Estimate`, `Command`,
   `ControlEvent` — schema-bundled in `gnc-contracts-1.0.json`, each with provenance fields
   (source sensor, source observation/command, source command). `Observation` and `Estimate`
   embed the E12-01 `relative-state-1.0` document verbatim; no parallel relative-state type is
   introduced.
2. **One `ClockSpec` owns every instant.** Timescale is uniform `TT` or `TDB` (UTC inside a control
   loop is rejected). Every sample, update and tick time is a pure function of its tick index
   (`phase + n·period`); the set of dropped sensor ticks is declared in the clock document, so a
   drop deletes records and never retimes later ones. Per-sensor latency is declared and enforced;
   control separates command-fire time from actuation time.
3. **Guidance sees a one-record view, not the plant.** `GuidanceInput` is a frozen capability
   object with `__slots__ = ("_latest",)` carrying one `Estimate`; laws are free functions of
   `(GuidanceInput, time)`. `feed.py` is forbidden from mentioning truth: an AST test fails on any
   `truth`/`plant` identifier in the module. The plant writes truth straight to the run log.
4. **A run log closes and seals the run.** `GncRunLog` validates every causal gate on append
   (measurement after sample, estimate after arrival, actuation at the requested time, one event
   per command, mission roster epoch equals clock `t0`), then seals a `run_digest` over the
   canonical document using the E12-08 digest policy; `read_run_log` re-validates and re-checks the
   digest, so a sealed run is replayable evidence, not a transcript.
5. **Optional `plan_digest`** (64-hex) binds a run to an E12-08 canonical plan without forcing
   closed-loop studies to be plan-bound.

## Consequences

- Closed-loop results become replayable and tamper-evident under the same digest discipline E12
  established; #416 follow-ons inherit the clock and cannot redefine timing.
- Isolation is machine-checked (push-side `TypeError`, `__slots__`, AST scan), so a future
  convenience "just pass truth to the law" breaks tests, not silently the experiment.
- Costs: one more schema bundle to version; a run cannot mix wall-clock UTC (intentional); the
  reference sim in `sim.py` is a J2/CW-class pair propagator — independent-physics re-flight stays
  the job of the TudatPy tooling (ADR rendezvous-0003/0004), not of this contract.

## Alternatives rejected

- **A message bus / pub-sub simulation framework:** adds a runtime dependency and still needs a
  sealing story for replay; the five record kinds plus a push-only feed give the same isolation
  with no new machinery.
- **Per-sensor timescales:** bookkeeping for conversions that E12 deliberately avoided; a uniform
  clock with declared phase, period and latency already covers asynchronous sampling.
- **Passing truth to guidance in fast tests behind a flag:** a flag is a leak with a TODO; the
  acceptance criteria for #418 require structural isolation.
