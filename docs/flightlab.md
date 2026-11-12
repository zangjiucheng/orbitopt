# Flight Lab — user guide

Flight Lab is orbitopt's rocket-ascent playground: you design a launch
(site, target orbit, booster recovery), simulate it, replay the result in
Mission Control's 3D view, and get an **honest verdict** — the numbers and
verification checks behind "did it reach orbit / did the booster land", not
a score. It lives in `src/orbitopt/flightlab/` and in Mission Control's
*Missions | Flight Lab* sidebar switch.

This page is the user guide. Design history lives in the epics (OpenProject
#187–#198) and the ADRs ([flightlab-0001](adr/flightlab-0001-jsbsim-6dof.md),
[flightlab-0002](adr/flightlab-0002-closed-loop-landing-guidance.md));
the E4 spike numbers are in
[flightlab-jsbsim-spike-findings.md](flightlab-jsbsim-spike-findings.md).

## Quick start

**GUI.** Launch Mission Control (`python -m orbitopt.viz.app`), flip the
sidebar to **Flight Lab**:

- **Designer** — fill in the launch site, target orbit and recovery
  settings, pick or edit a vehicle, press *Simulate*. The vehicle panel
  carries a **provenance tag** (FL-502/FL-602): a file whose numbers are
  illustrative or reconstructed says so right there.
- **Replay** — any run in 3D with telemetry, verification checks,
  plain-language failure reasons and the booster recovery track.
- **Compare** — two runs side by side, including where their verdicts
  diverge.
- **Recent runs / Open run…** — every run with its verdict; any run
  directory, such as one written by `orbitopt flight simulate`, loads.

**CLI** (everything the GUI decides is reproducible without the GUI —
hard gate G4):

```bash
orbitopt flight validate  target.yaml          # schema-check a FlightTarget or a run dir
orbitopt flight simulate  target.yaml -o run/  # Gate B: ascend, check, re-fly, honest verdict
orbitopt flight optimize  target.yaml -o run/  # FL-202 pygmo ascent optimisation
orbitopt flight montecarlo run/ --n 32 --seed 7 # FL-304 recovery Monte Carlo
orbitopt flight to-scene  run/ -o scene.json   # run dir -> viewer scene
```

## Kerbal → engineering concepts

Flight Lab is deliberately a *KSP-ish* surface over engineering models. The
mapping it actually implements:

| You know it as (KSP) | Flight Lab calls it | Where |
|---|---|---|
| VAB / vehicle editor | **Designer** vehicle form + vehicle YAML file | `flightlab/ui/designer_page.py`, `data/flightlab/vehicles/` |
| `F9` config.cfg numbers | vehicle **data with provenance tags**: every number is `official`, `public_estimate` or `illustrative`, with source, date, uncertainty | `data/flightlab/schemas/flight-vehicle-1.0.json`, `flightlab/vehicle_catalog.py` |
| Science mode readouts | `FlightRun` contract: frame, epoch, units, per-field provenance | `flightlab/contract.py` |
| Onboard computer / SAS | closed-loop **guidance re-planning** (booster landing: plan-level retarget each step) | `flightlab/recovery/` (FL-302/FL-403; [ADR 0002](adr/flightlab-0002-closed-loop-landing-guidance.md)) |
| "Recoverable booster" toggle | recovery target + gates: touched down inside limits, with reason codes | `flightlab/recovery/pipeline.py`, `checks.py` |
| Contract reward / funds | verification checks against declared tolerances | `flightlab/verification` via `contract.Verification` |
| MaxQ, apoapsis, crash counter | trajectory columns + events in the ECEF rotating frame; apoapsis alone is **not** orbit (G2) | `flightlab/recovery/propagator.py`, `frames.py` |
| Time warp, replay | Mission Control timeline replay of a run (`flight to-scene`, Replay tab) | `flightlab/scene_export.py` |
| "It exploded" (ragdoll) | **reason codes**: every failure names its governing constraint | `flightlab/recovery/checks.py` |

## Fidelity levels

What "simulate" means, honestly, from cheapest to most faithful:

1. **3-DOF point-mass ascent** (default; Gate B, FL-006/FL-202). The
   flight article is OpenTsiolkovsky: a patched-conic/point-mass launcher
   design tool. Gravity turn or declared attitude program, piecewise Cd(M)
   aero. Attitude is *displayed* kinematically — a pointing hint, not a
   6-DOF state. Fast enough to optimise (pygmo loop, `flight optimize`).
2. **3-DOF booster recovery in the rotating frame** (FL-301, Gate C,
   FL-302 landing guidance). ECEF point mass with J2, Coriolis/centrifugal,
   US76/Vallado atmosphere, relightable engine group, Cd(Mach) table —
   RK4 with per-step event bisection (`recovery/propagator.py`).
3. **Monte Carlo at plan level** (FL-304, `flight montecarlo`). Thrust, Isp,
   Cd, density-scale and separation-state dispersions; guidance re-plans on
   each sample's dispersed vehicle. *Plan-level*: the guidance sees the
   dispersed vehicle, the sample is still 3-DOF. Default dispersion bands now
   come from the vehicle file's uncertainty tags (FL-602, below).
4. **6-DOF tracking spike** (E4, opt-in `pip install orbitopt[sixdof]`,
   `flightlab/sixdof/`). JSBSim 1.3.3 flying the 3-DOF program with PID
   gimbal tracking through MECO, and a staging handoff. **GO**
   ([ADR 0001](adr/flightlab-0001-jsbsim-6dof.md)) — the deviation budget is
   documented; it is a verification cross-check, *not* yet the default
   propagator and *not* inside optimisation loops.

## Hard gates (epic #187 — never negotiable)

- **G1 — provenance rides along.** Every trajectory records frame, epoch,
  units and the provenance of the data it flew (`contract.py`); a run you
  can't trace is a run you can't trust.
- **G2 — apoapsis is not orbit.** "Orbited" means the FL-006 velocity /
  perigee / inclination checks pass at the declared epoch — never a peak
  altitude alone.
- **G3 — touching down is not landing.** "Landed" means touchdown inside
  every declared landing limit (position, vertical and lateral speed,
  load); a soft-ish stick still reads red with its reason code.
- **G4 — CLI reproducibility.** Every physical verdict the UI shows is
  reproducible via `orbitopt flight ...` with no GUI, same verdict.
- **G5 — thresholds are declared before the run.** Gate limits live in the
  target YAML / vehicle file; nothing is loosened after a failure to make a
  run pass. A red verdict stays red in git history.

## What is *not* modelled

Flight Lab answers "is this trajectory consistent", not "will this rocket
fly". Not modelled, at any fidelity level:

- **6-DOF flight dynamics by default** — attitude is kinematic (levels 1–3
  above); JSBSim exists as an opt-in cross-check (level 4, E4 GO) and stays
  outside optimisation loops.
- **CFD-grade aerodynamics** — piecewise-linear Cd(Mach) tables only, no
  fins/body interactions, no base drag modelling, grid fins treated as Cd
  numbers.
- **Structures and loads** — no finite-element stress, no MaxQ airframe
  limit, no loads analysis; "max Q" is a readout, not a gate.
- **Propellant sloshing, geysering, ullage** — tanks are a scalar mass.
- **Stage dynamics** — separation is an impulse + jettison event, no
  collision physics.
- **Wind, weather, range safety** — co-rotating still air; the density
  *scale* is dispersed but there is no atmosphere model variation.
- **Guidance autonomy** — recovery re-planning is plan-level (FL-302), not
  a simulated onboard computer.

## Vehicle data, provenance and uncertainty (FL-602)

Catalog: `src/orbitopt/data/flightlab/vehicles/`, validated against
`data/flightlab/schemas/flight-vehicle-1.0.json`
(`orbitopt.flightlab.vehicle_catalog`). Every packaged vehicle file carries:

- `synthetic_two_stage.yaml` — **illustrative**. A generic two-stage to
  orbit + recoverable booster. Every number, and its declared uncertainty
  band, is a made-up class-typical value.
- `f9_class_public_estimate.yaml` — **public_estimate**. A Falcon-9-class
  vehicle built from the MIT-licensed OpenTsiolkovsky
  `examples/falcon9.json` (PACE mission config, itself a community
  reconstruction) cross-checked against public sources (SpaceX Falcon 9
  User Guide 2020 for published engine figures; telemetry reconstructions
  for landing mass). **It is not official SpaceX performance data and is
  never labelled as such** — the file says so in its `disclaimer`, and the
  Designer badge echoes the tag.

How it works:

- Each numeric field has a resolved provenance entry: `provenance` tag
  (`official` / `public_estimate` / `illustrative`), `source`, `as_of`,
  optional `accuracy` and `uncertainty_pct` (symmetric ± percent
  half-width). Tags live in a top-level `data_provenance:` sidecar keyed by
  document path (longest prefix wins; nearest entry overrides a broader
  cover), plus inline `{value, provenance, source, ...}` nodes allowed only
  under `booster:` — the OT `stages:` block stays a plain OT config so the
  file remains flyable (the adapter passes it through untouched).
- Tests (`tests/test_flightlab_vehicle_catalog.py`) enforce the acceptance
  criteria: every numeric field of every catalog file has a resolved tag
  with source and as_of; schema violations and untagged numbers fail.
- **Uncertainty feeds FL-304 by default**: `orbitopt flight montecarlo`
  (or `run_monte_carlo(dispersions=None)`) takes the thrust / Isp / Cd
  dispersion half-widths from the vehicle file's tags — the Monte Carlo
  inherits the data's stated confidence bands unless you override them.
  The result's `assumptions` record which bands came from the vehicle.
- The Designer badge (FL-502) shows the loaded file's tag; `illustrative`
  and `public_estimate` render as warnings, not muted facts.
