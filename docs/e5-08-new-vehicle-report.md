---
title: E5-08 design variants
parent: Flight Lab
nav_order: 0.5
description: Recorded Kourou and Starbase RTLS designs, nominal outcomes and remaining qualification gaps.
---

# E5-08 design variants

E5-08 adds powered-entry RTLS to the E5-07 public-estimate stack. E5-08b
changes staging (DC7); E5-08c moves the design to Starbase and retunes its
ascent. These are modeled designs, not historical flight validation.

## Recorded results

| Record | Original E5-08 | E5-08b: DC7 | E5-08c: Starbase |
|---|---|---|---|
| Catalog vehicle | `starship_class_rtls_derived` | `starship_class_rtls_dc7` | `starship_class_rtls_starbase` |
| Launch site / target inclination | Kourou / 51.6° | Kourou / 51.6° | Starbase / 30.5° |
| Delivered orbit, km | 232.5 × 292.1 | 232.0 × 234.8 | 232.2 × 264.5 |
| Peak ascent load | 2.888 g | 2.976 g | 2.928 g |
| Nominal recovery | Landing failed | Landed; 113.1 t remaining | Landed; 107.1 t remaining |
| Cross-environment joint gate, recorded summary | Failed: landing | Passed | Passed |
| Independent-model checks | 27/27 passed | 27/27 passed | 26/27 passed |
| Hold-out recovery, pinned script | Not recorded | 11/36 | 4/36 |
| Hold-out recovery, per-sample refinement | 0/30 | 3/36 | 0/36 |

DC7 cuts off stage 1 earlier, leaving the booster heavier and slower while
spending more ship propellant. The recorded ship retains 101.2 t at Kourou
and 102.5 t at Starbase, versus 118.8 t in E5-07. Starbase also changes the
launch geometry and target plane; its result is not an isolated site comparison.

The variants retain the 55 kPa ascent Max-Q cap, 3.0 g ascent limit,
230 km periapsis floor, 250 ± 60 km insertion band, 3/1 m/s vertical/lateral
touchdown limits and 100 t reserve. Starbase changes the inclination target
to 30.5°. Read each scenario rather than the inherited 51.6° prose in some
recording metadata.

## What remains open

- **Original design:** the powered-entry script reaches the corridor, but
  the landing is refused or fails. Its projected post-landing reserve is
  96.1 t, about 3.9 t below the 100 t requirement. A reserve check on an
  aborted rollout does not evidence a successful landing.
- **DC7 independent dynamics:** closed by E10-04, band unchanged (see below).
- **Starbase independent dynamics:** debris impact-speed discrepancy is
  12.80% against a 12% band. E10-04 evidenced it as a genuine drag-model
  difference, not an implementation artifact (see below). The band was not
  changed and the check is recorded as failing.
- **Reliability:** both nominally closing designs have low hold-out success
  under the declared FL-304 dispersions. Pinned and per-sample-refined
  experiments use different planning modes; do not pool their rates.

The DC7 and Starbase experiments each use seeds 8, 9 and 10, with 12 flights
per seed. Original-design results use seeds 5, 6 and 7, with 10 per seed.
See [E7-05](benchmarks/e7_05_out_of_sample_reliability.md) for cluster-aware
statistics. The DC7 fixed-versus-replanning gap is diagnosed and a robust
single-script arm is compared on the same hold-out in
[E10-05](benchmarks/e10_05_recovery_planner_robust.md). The original
independent landing check starts from a declared design corridor and is
recorded as a `design_input`, not full-path
verification: it does not re-fly the actual recovery, so it cannot be read
as independent physical verification of the complete RTLS chain. The E10-07
full recovery re-flight ([E10-07 report](e10-07-full-recovery-reflight.md))
supersedes it for that purpose: it re-flies boostback, unpowered entry and
the landing phase from the actual DC8 hot-staging separation handoff under
an independently written twin model.

## E10-04: closing the two open discrepancies (task #301)

Neither tolerance was changed. The two open checks closed differently, and the
difference between the two outcomes is the point of the exercise.

**DC7 SECO semi-major axis — implementation artifact, fixed (27.7 km -> 5.85 km
against the same 15 km band).** The independent-dynamics twin compares itself
against the raw OpenTsiolkovsky state history interpolated at each declared
event epoch. That interpolation was a straight chord across the bracketing 2 s
segment. On this design MECO (123.0 s) and SECO (514.663 s) fall mid-segment,
across the engine-cutoff discontinuity: the chord averaged the powered and
coasting slopes, ~12.4 m/s too low at MECO and ~13 m/s too low at SECO - about
1.75 km of semi-major axis per m/s, which was ~22 km of the 27.7 km. It also
over-read the recorded coast mass at MECO by ~8.4 t. The fix reconstructs the
state inside such a segment from the LEFT raw row and its own recorded
accelerations (position quadratic, velocity linear, mass from the left row's
thrust/isp flow), which is exact up to the mid-segment switch; event times that
land on a grid row or a smooth segment behave exactly as before. After the fix
all 28 DC7 checks pass, with the measured budget lines (E8-02 discipline)
accounted in the report: MECO mass discrepancy −6,436 kg with the debris-mass
line −6,436 kg (one engine-Isp line, ~1.2% of the stage-1 burn), SECO velocity
+5.2 m/s, SECO SMA +5.85 km. Regression tests:
`tests/test_e805_independent_dynamics.py` (kink reconstruction, ignition mass
hold, recorded-DC7 SMA invariance).

**Starbase debris impact speed — genuine model difference, kept over-tolerance
(12.80% against the same 12% band).** Replaying the twin's own power-off debris
integrator from the RECORDED separation state and RECORDED debris mass
(identical initial conditions, same US76 atmosphere, no shared code path)
reproduces the gap: +12.7% at the declared CA polar versus the report's 12.8%,
so timing, frame convention, interpolation and engine lines are ruled out. The
twin reproduces the recorded descent to within 0.5 % at every shared altitude down to
22 km (dynamic pressure ~98 kPa) and from there to impact the recorded body sheds
393 m/s against the twin's 190 (~2x the deceleration); scaling the
declared blunt-body power-off polar by x1.1 closes the gap (+10.8%). The two
independent models disagree on the power-off drag coefficient of a tumbling
upper stage in dense transonic flow - an illustrative design-table input, not a
measurement - so this is recorded as a genuine model difference per the E8-05
evidence rule, the check stays red, and the band stays at 12%. Evidence and
sweep: `debris-model-difference.json` in the Starbase reports directory.

**E10-02 hot staging (DC8) re-evaluation.** The same fix was applied to the
DC8 pseudo-reference comparison (`docs/benchmarks/e10_02_hot_staging.json`,
regenerated). It moves every affected band the right way - SECO velocity
discrepancy 45.2 -> 37.0 m/s, SECO SMA 50.7 -> 34.5 km, SECO mass −2.9 ->
−2.3 t, separation position back inside its band - because DC8's MECO (125.5 s)
and SECO (513.663 s) also sit mid-segment. What still fails is the declared
hot-timeline representation split itself (OT ends all stack power at
separation; the twin burns the booster's post-separation floor burn), which is
larger on this timeline because the stack flies powered through the events, and
is disclosed in the benchmark, not absorbed.

## Replay and view

Use the [landing environment](../envs/README.md), including gfold 0.3.2,
from the repository root. Select `dc7` or `starbase`; unset the variable for
the original gap record. Unlike the E5-05 script, `run` takes a positional
output directory:

```bash
export ORBITOPT_E508_VARIANT=starbase
python scripts/e508_new_vehicle.py run runs/e508-starbase
python scripts/e508_new_vehicle.py compare runs/e508-starbase
orbitopt flight to-scene runs/e508-starbase/recovery/run -o runs/e508-starbase/scene.json
orbitopt view runs/e508-starbase/scene.json
```

For telemetry, add `runs/e508-starbase/recovery/run` in **Runs**. Select the
matching catalog vehicle in **Vehicles** for its declared-versus-flown overlay.
Replay uses the recorded ascent; it does not re-optimize the design.

## Add parking propagation

Install the checkout in both interpreters, then run the matching plan:

```bash
orbitopt crossenv run --plan src/orbitopt/data/crossenv_e508_starbase_plan.yaml \
  --env tudat=/path/to/tudatpy/bin/python \
  --env gfold=/path/to/landing/bin/python -o runs/crossenv-e508-starbase
ORBITOPT_E508_VARIANT=starbase python scripts/e508_new_vehicle.py \
  crossenv-check runs/crossenv-e508-starbase --no-run
```

For DC7, use `crossenv_e508_dc7_plan.yaml` and `ORBITOPT_E508_VARIANT=dc7`.
A single-environment replay omits parking propagation and its physical gate
remains `not_checked`, even when its ordinary mission report passes. The
joint checker must evidence all five physical checks; a nominal joint pass
does not clear the independent-model failures or the dispersion study.

## Evidence files

Each `tests/data/mission/reference/e5-08-new-vehicle[-dc7|-starbase]/` tree
contains the sealed inputs, `PROVENANCE.json`, `comparison.json` and mission
reports. The sibling `*-reports/` directory contains independent dynamics
and Monte Carlo JSON. The
[DC7 notes](../tests/data/mission/reference/e5-08-new-vehicle-dc7-reports/README.md)
and [Starbase notes](../tests/data/mission/reference/e5-08-new-vehicle-starbase-reports/README.md)
record the experiments and joint-gate outcomes. These summaries describe
committed results; the commands above establish results for a fresh run.
