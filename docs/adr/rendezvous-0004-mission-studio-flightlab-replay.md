# ADR rendezvous-0004: mission studio replay as a Flight Lab run with a sidecar (E12-06)

- **Status:** Accepted
- **Date:** 2026-11-25
- **Deciders:** E12 Orbital Rendezvous & Mission Operations (Epic #389), E12-06 (#396)
- **Code:** `src/orbitopt/rendezvous/studio.py`, `src/orbitopt/flightlab/ui/rendezvous_panel.py`,
  `src/orbitopt/flightlab/{compare,scene_export}.py`, `src/orbitopt/flightlab/ui/{telemetry,flight_panel}.py`,
  `src/orbitopt/cli.py` (`orbitopt rendezvous studio`)
- **Evidence:** `tests/test_e12_06_mission_studio.py`
- **Builds on:** ADR rendezvous-0001 (plan), ADR rendezvous-0003 (independent verification), ADR relative-0001
  (`propagate_pair`, LVLH frames), the frozen FlightResult/scene/mission/replay v1 contracts

## Context and problem

LEO-RDV-001 was planned and verified (E12-03) but had never been *seen*: the task
requires the mission to replay in Flight Lab with both trajectories, relative
range/speed curves, the maneuver timeline and the safety verdicts — reusing the
existing panels, not a parallel UI. Three contract facts shape the design:

1. `flight-result-1.0` is frozen and its event-kind enum is ascent-only
   (`ignition`…`MECO`…), so rendezvous maneuvers cannot enter the run manifest
   without breaking every consumer of that schema.
2. The contract run has one Earth scene and no vehicle: `vehicles=()` is legal,
   but there is then nowhere to carry the plan, the keep-out sphere or the
   hold point — and the verification report (`RendezvousVerification`) is not a
   `Verification` block, so its five gate checks do not appear in the Checks tab
   unless the producer restates them.
3. The dense display tracks that `propagate_pair` produces do not end at the
   verified arrival state to the bit: sub-stepping to a grid endpoint differs
   from the verified two-point leg by ~4e-5 m, and re-deriving the arrival from
   `arrival_epoch_s - drift_end_s` instead of the plan's `tof_s` costs another
   ulp. A replay whose last sample disagrees with the report is exactly the
   drift the verification story forbids.

## Decision

1. **The studio run is a `FlightResult` v1 run directory plus one new sidecar.**
   `build_studio_run` writes the contract run (`orbitopt rendezvous studio -o DIR`),
   the `MultiSpacecraftMission` document (`mission.json`) and the unified replay
   (`replay.json`) — all schema-valid as shipped — and adds `rendezvous.json`
   (`rendezvous-studio-1.0`): plan, verification, events, keep-out result, hold
   point and static marker positions. Maneuver/hold events and zone data live
   **only** in the sidecar; the run's `events` stay `()`. `RunBundle.rendezvous`
   carries it to the UI; anything that does not know the sidecar ignores the run
   unchanged.
2. **The last timeline sample is the verified arrival state, bit-exact.** The
   chaser's final sample is re-derived from `verify_plan`'s exact two-point legs
   run from the same t0 states (`tof_s`, not a re-subtracted difference), not
   from the dense grid; the arrival row also carries the *post*-Δv2 velocity. A
   guard in `build_studio_run` (tolerance 1e-9 m) compares the re-derived
   relative state with the report and refuses the run on disagreement — the
   first version shipped with the dense endpoint and the guard failed at
   4.13e-5 m, which is how the discrepancy was caught instead of shipped.
   Consequence: `rel_speed`/`rel_*_lvlh` curves end on exactly the numbers the
   Rendezvous tab shows (test asserts `==`, not `approx`).
3. **Display extensions without renderer surgery.** The keep-out centre and the
   hold point enter the scene as site-kind marker bodies (the launch-site
   mechanism, `hideTrail` static trails); the two sidecar events join
   `timeline["events"]` like any contract event. The five relative channels reuse
   the extra-column telemetry pipeline and are emitted **only when present**, so
   a launch run's channel set is untouched. The Rendezvous tab renders the
   sidecar; the key-event table shows the sidecar maneuvers alongside contract
   events.
4. **Deterministic provenance markers.** `git_sha`/`created_utc` and the omitted
   `backend` field carry fixed documented markers (all-zero sha, mission epoch)
   so the run stays byte-reproducible; `target_sha256` is the canonical-JSON
   digest of the mission config itself.

## Consequences

- Generate LEO-RDV-001 with `orbitopt rendezvous studio -o runs/leo-rdv-001-studio`,
  then select `leo-rdv-001-studio` in Mission Control's **Runs** workspace.
- The sidecar is display metadata: `scene_export` skips malformed markers rather
  than failing a render, and no UI path reads it into physics.
- A future event-bearing contract version can absorb the sidecar's `events`
  verbatim; nothing else in the pipeline needs to change.
