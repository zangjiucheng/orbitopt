---
title: Examples and validation
parent: Missions and examples
nav_order: 1
---

# Examples and validation

The built-in missions demonstrate models at different fidelity levels. A
comparison with a historical flight is different from solving for a published
target; a passing verification report only covers its stated checks and model.

| Mission | What is modeled | Interpretation |
|---|---|---|
| Mars 2020 | Launch-window search, injection, cruise and Mars orbit insertion | Historical launch timing comparison; the injection report records 9/9 checks for the documented replay. See [`14_mars2020_replay.py`](../examples/14_mars2020_replay.py) and its [config](../examples/mars2020_replay.yaml). |
| Artemis II-like | Lunar flyby and unpowered Earth return | A 6,545 km perilune is an input target, not an independent reconstruction. The chosen departure is 2026-08-01; the real flight launched on 2026-04-01. The entry-angle check is vacuum geometry, not an assessment of heating or survival. See [re-entry constraints](reentry_safety_constraints.md). |
| GOES | GTO-to-GEO raising with an Atlas V/GOES-16 reference profile | A modeled campaign; the built-in scene starts **not checked** until high-fidelity verification is run. See the [mission plan](goes_gto_geo_mission_plan.md). |
| Starship Flight 15 | Return-pass prediction using assumed orbit and vehicle properties | A pre-flight prediction, not a reconstruction. The ascent is illustrative. The verification report fails under a 10% lift reduction; see the assumptions in [`starship_flight.py`](../src/orbitopt/viz/starship_flight.py). |
| Two-stage LEO (reference scenario) | Launch site to a 400 × 400 km, 51.6° parking orbit: ascent, staging, insertion and propagation | Graded as one chain with handoff gates and expected parking-orbit elements; the synthetic vehicle is illustrative, not a real rocket. Run `orbitopt run src/orbitopt/data/mission_two_stage_leo.yaml`; see [Flight Lab](flightlab.md#mission-pipeline). |
| Solar System | Planetary ephemeris visualization | No trajectory optimization or verification. |

## Historical launch-window comparisons

| Comparison | orbitopt | Historical reference |
|---|---|---|
| Mars 2020, KSC, 2020-07-30 | 11:51:28 UTC optimum | 11:50 UTC lift-off |
| Mars 2020, lowest C3 in the window | July 18, 13.2 km²/s² | Launch period opened July 17 |
| Sentinel-2A/2B, Kourou SSO window | Within 6–9 minutes | Actual lift-off times |
| ISS, KSC | 44.9° launch azimuth; about 23 minutes earlier per day | Approximate published values |

For the Mars launch-time comparison, the flown departure date and flight time
are inputs; the time of day is compared afterward. Launch-site and vehicle
figures are planning-level data: see [sources and limitations](launch-data-sources.md).

`orbitopt benchmarks` turns two more references into regression gates: the
analytic two-body and J2 drift solutions, and the Starship Flight 15 booster
catch against its published timeline. See [verification](verification.md).

## Run an example

```bash
python examples/14_mars2020_replay.py
orbitopt run examples/mars2020_replay.yaml
orbitopt view artemis2
```

The [examples directory](../examples/) also includes Lambert, multi-gravity
assist, porkchop, free-return and Flight Lab examples. Computed results depend
on the installed numerical stack and the selected configuration.
