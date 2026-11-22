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
| Independent-model checks | 27/27 passed | 26/27 passed | 26/27 passed |
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
- **DC7 independent dynamics:** SECO semi-major-axis discrepancy is 27.7 km
  against a 15 km band. The band was not changed.
- **Starbase independent dynamics:** debris impact-speed discrepancy is
  12.80% against a 12% band. The band was not changed.
- **Reliability:** both nominally closing designs have low hold-out success
  under the declared FL-304 dispersions. Pinned and per-sample-refined
  experiments use different planning modes; do not pool their rates.

The DC7 and Starbase experiments each use seeds 8, 9 and 10, with 12 flights
per seed. Original-design results use seeds 5, 6 and 7, with 10 per seed.
See [E7-05](benchmarks/e7_05_out_of_sample_reliability.md) for cluster-aware
statistics. The DC7 fixed-versus-replanning gap is diagnosed and a robust
single-script arm is compared on the same hold-out in
[E10-05](benchmarks/e10_05_recovery_planner_robust.md). The original independent
landing check uses a declared design corridor; it does not validate the failed
mission's actual arrival state.

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
