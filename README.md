# orbitopt

[![CI](https://github.com/zangjiucheng/orbitopt/actions/workflows/ci.yml/badge.svg)](https://github.com/zangjiucheng/orbitopt/actions/workflows/ci.yml)
[![Pages](https://github.com/zangjiucheng/orbitopt/actions/workflows/pages.yml/badge.svg)](https://github.com/zangjiucheng/orbitopt/actions/workflows/pages.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](pyproject.toml)
[![License: GPL v3+](https://img.shields.io/badge/license-GPL--3.0--or--later-blue)](LICENSE)
[![Site](https://img.shields.io/badge/site-zangjiucheng.github.io%2Forbitopt-e8a23e)](https://zangjiucheng.github.io/orbitopt/)

**Screen fast, verify exactly.** orbitopt is an orbital trajectory
optimization framework. It searches huge design spaces with cheap
patched-conic models (**pykep + pygmo**, batched on the GPU with **CuPy**),
then re-flies the winners under high-fidelity numerical propagation
(**tudatpy**) and writes a machine-readable verification report for every
result. Mission Control, a desktop 3D app, plays the missions back and shows
why a trajectory was chosen and how well it was verified.

![Mission Control on the Earth-to-Mars transfer: the optimizer's candidate trajectories on the left, the heliocentric transfer in 3D, the launch-window porkchop with the candidates on it below, and the selected candidate's numbers on the right](docs/assets/mission-control-mars.png)

## What it does

| Stage | What you get | Where |
|---|---|---|
| **Search** | Batched Lambert solver, porkchop grids, pygmo optimizers with GPU batch fitness, multi-basin multistart | `lambert/`, `optimize/`, `problems/` |
| **Launch** | Daily launch windows from a site database (range-safety azimuth corridors), vehicle C3/payload curves as optimizer constraints, LEO parking orbit → TLI/TMI injection | `launch_windows.py`, `launch_constraints.py`, `injection.py` |
| **Verify** | tudatpy N-body propagation (adaptive RKDP8(7)), dense-output event location, Newton/B-plane targeters (TCM, MOI, lunar flyby, GEO insertion), JSON verification reports | `verify/` |
| **Visualize** | Versioned scene file format, single-scene viewer, Mission Control: planet-centered views, candidate comparison, verification reports, linked porkchop / distance charts | `scene_format.py`, `viz/` |

Built-in missions: **Artemis II** free return, **GOES** GTO→GEO raising,
**Earth → Mars** orbit insertion, and the solar system.

## Checked against real missions

Two different kinds of check, and they should not be read the same way.

**Historical replays.** orbitopt is given only the physics and the real
constraints of the time (launch site, vehicle, departure window) and
predicts what was actually flown. The flown values are *not* inputs, so
these are independent comparisons.

| Mission | orbitopt | Real |
|---|---|---|
| Mars 2020: KSC launch-window optimum, Jul 30 | 11:51:28 UTC | lift-off 11:50 UTC |
| Mars 2020: lowest-C3 departure in the 2020 window | Jul 18 (C3 13.2 km²/s²) | launch period opened Jul 17 |
| Sentinel-2A/2B: Kourou SSO launch time | within 6–9 min | actual lift-off |
| ISS: KSC launch azimuth / daily window drift | 44.9°, ~23 min earlier per day | same |

**Targeting checks.** A published mission figure is fed in as a *target*
and orbitopt checks that its solvers hit it under high-fidelity dynamics.
They show the targeters work. They do not show that orbitopt reproduces the
real trajectory.

| Mission | Input (target) | orbitopt result |
|---|---|---|
| Artemis II-like lunar free return | 6,545 km perilune altitude (Artemis II's published figure); entry flight-path angle −6° | perilune within the 50 km targeting tolerance, and the unpowered return reaches the entry interface inside the [−7°, −5°] corridor (TudatPy Earth+Moon+Sun) |

The Artemis II mission (`orbitopt view artemis2`,
[`examples/artemis2_free_return.yaml`](examples/artemis2_free_return.yaml))
departs on a date we chose, **2026-08-01**. The real Artemis II launched on
2026-04-01. Its parking orbit (185 km circular) and departure geometry are
modelling choices too. The run solves a free return of the same class as
Artemis II, with the same perilune and a survivable entry angle. It is not
an independent reconstruction of the flown trajectory: NASA has not
published navigation-grade state vectors to compare against. See
[`docs/engineering-notes.md`](docs/engineering-notes.md#cislunar--artemis-ii-free-return-pipeline-in-full).

The Mars 2020 numbers come from
[`examples/14_mars2020_replay.py`](examples/14_mars2020_replay.py), which runs
the whole chain: porkchop → multistart → launch constraints → launch window →
parking orbit and TMI, numerically verified (9/9 checks pass). The same search
shows a Falcon 9 could not have flown that 4,037 kg payload: its C3 ceiling is
8.6 km²/s², below the 13.2 minimum of the whole window. The tudatpy cruise
(TCM targeted in the B-plane, then Mars orbit insertion at 505 km periapsis)
runs from a config:
`orbitopt run examples/mars2020_replay.yaml`.

## Quick start

pykep, pygmo and tudatpy are only on conda-forge / tudat-team, so the compute
stack comes from a locked conda environment:

```bash
micromamba create -n orbitopt -f envs/cpu.conda-lock.yml   # or envs/gpu.conda-lock.yml (CuPy, CUDA 13)
micromamba activate orbitopt
pip install -e . --no-deps
pytest tests/
```

On Windows, `.\scripts\setup_env.ps1` (add `-Gpu` for CUDA) does the same.
See [`envs/README.md`](envs/README.md) for regenerating the lockfiles and the
GPU CI job.

```bash
orbitopt                                   # Mission Control
orbitopt view mars                         # one built-in mission in the plain viewer
orbitopt run examples/mars2020_replay.yaml # optimize, verify in tudatpy, write scene + report
orbitopt launch-window ksc --rla 9.6 --dla 23.4 --epoch 2020-07-30
python examples/14_mars2020_replay.py      # the end-to-end walkthrough above
```

## CLI

| Command | Does |
|---|---|
| `orbitopt` / `orbitopt app` | Mission Control |
| `orbitopt view <mission or scene.json>` | single-scene 3D viewer |
| `orbitopt export <mission> -o scene.json` | cache a built-in mission's scene |
| `orbitopt missions` | list registered missions (built-in and plugins) |
| `orbitopt validate config.yaml` | schema-check a mission config, no compute |
| `orbitopt run config.yaml` | run a config: optimize, verify, write scene + `.verification.json` |
| `orbitopt view config.yaml` | run a config (cached) and open the 3D viewer |
| `orbitopt launch-window <site> ...` | daily launch windows into an orbit plane (`--inclination`, `--raan`/`--ltan`, `--dogleg`) or along a departure asymptote (`--rla/--dla`) |

## Mission configs

Missions can be described in YAML instead of Python. Every section except
`mission` and `output` is optional, and omitted fields fall back to the
built-in mission's values:

```yaml
mission: { id: my-geo-mission, title: My GEO raising campaign, kind: geo-raising }
target: { geostationary_longitude_deg: -101.0 }
campaign_search: { n_burns: { min: 2, max: 6 } }
output: { scene_file: scenes/my-geo-mission.json }
```

Supported kinds: `geo-raising`, `mars-transfer` and `lunar-free-return`
(an Artemis II-like free return; see `examples/artemis2_free_return.yaml`). Schemas are in
`src/orbitopt/schemas/`. `orbitopt view config.yaml` runs a config and opens
the result; the scene is cached like the built-in missions' scenes and is
recomputed when the config or orbitopt's source changes.

The Solar System scene is not a config kind. It is a near-instant ephemeris
export with no optimizer, targeting or verification stage, so there is
nothing for a config to tune or for `run` to verify. It stays a built-in
(`orbitopt view solar-system`).

## Verification reports

Each verification stage emits a JSON report validated against
`schemas/verification-report-1.0.json`. A report lists the dynamics model and
what it leaves out, integrator settings, a step-convergence re-run,
conservation checks against a two-body reference, targeting residuals and
pass/fail checks. `orbitopt run` writes the reports next to the scene and
warns when any check fails.

Launch-site and launch-vehicle data cite a source and are tagged `official`
or `public_estimate`, with accuracy and date. The values are planning-level;
see [`docs/launch-data-sources.md`](docs/launch-data-sources.md).

## Layout

```
src/orbitopt/
  cli.py, missions.py          console command; mission registry (+ entry-point plugins)
  scene_format.py, schemas/    scene / config / report JSON Schemas (stdlib + jsonschema only)
  mission_config.py            YAML mission configs
  core/                        numpy/CuPy switch; OrbitOptProblem (pygmo UDP + batch_fitness)
  lambert/                     batched 0-rev Lambert (GPU); pykep Izzo wrapper (multi-rev)
  dynamics/                    GPU-batched Earth-Moon-Sun screening propagator
  problems/                    transfer, MGA-1DSM, free return, GEO raising
  optimize/                    pygmo runner + GPU batch evaluator, multistart
  launch_*.py, injection.py    sites, vehicles, constraints, windows, parking orbit → TLI/TMI
  verify/                      tudatpy propagation, events, B-plane, targeters, reports
  viz/                         porkchop, mission exporters, renderer, viewer, Mission Control
                               (app.py + state, panels, charts; see its module docstring)
```

To add a problem type, subclass `orbitopt.core.problem.OrbitOptProblem`. If it
implements a vectorized `batch_fitness()`, `optimize.runner.run_optimization`
evaluates it on the GPU. To add a mission without touching orbitopt, register
it from another package:

```toml
[project.entry-points."orbitopt.missions"]
my-mission = "your_package.missions:my_mission"   # returns Mission(id, title, loader)
```

## Installing without conda

The package has three layers, and only the last needs conda:

1. **Scene format**: `pip install orbitopt` gives you `read_scene` /
   `write_scene` / `validate_scene` against the versioned
   `scene-1.0.json`. Adding fields is non-breaking; breaking changes get a
   new schema version.
2. **Viewer**: `pip install orbitopt[viewer]` adds PyVista, PySide6 and
   pyvistaqt, which is enough to open any scene file in the viewer or Mission
   Control.
3. **Compute**: problems, verification and mission exporters need the conda
   environment above. There is deliberately no `[compute]` extra, because
   pip cannot install those packages.

## Mission Control

A desktop app (PySide6 + pyvistaqt) for working through a mission, not just
playing it. Every panel follows the same selection and mission time:

- **Missions and trajectories** (left). Switch missions without relaunching;
  `File > Open scene file...` opens any scene JSON. For an optimized mission
  the TRAJECTORIES list shows every distinct candidate the optimizer found,
  with what happened to it at the verification gate. Picking one highlights
  its two-body Lambert screening arc in 3D.
- **3D view, centered where it matters** (middle). The CENTER strip
  re-centers the view on the Sun, Earth, Mars or the Moon, so the Earth
  departure and the Mars capture are visible at their own scale inside a
  heliocentric transfer. It also says whether body sizes are to scale.
- **Linked charts** (below). *Launch window*: the porkchop grid the search
  screened, with the candidates on it; click one to select it. *Distance*:
  the spacecraft's distance to each body over the mission; click or drag to
  move the mission time.
- **Inspector and Verification** (right). Body readouts, maneuvers and the
  selected candidate's numbers; and the high-fidelity verification reports:
  every check, residuals against their tolerances, force model, integrator
  step convergence, conservation. The title bar always shows VERIFIED,
  VERIFY FAILED or NOT CHECKED.
  **GOES loads unverified by default** (the TudatPy re-fly adds about a
  minute of compute): its badge reads NOT CHECKED, and a smooth animation
  does not mean it passed high-fidelity verification. The Verification tab
  says so and has a **Run high-fidelity verification (TudatPy)** button that
  re-runs GOES with verification on and shows the resulting report (cached
  after the first run). From the CLI, `orbitopt run` a `geo-raising` config
  with `verification.enabled: true`.
- **Timeline** (bottom). Time warp, events, and the mission's phases
  (parking orbit, cruise, lunar flyby, ...); click a phase to jump to it.

![Artemis II centered on the Moon at closest approach: the lunar-flyby phase lit on the timeline, the distance chart dipping at the Moon, and the Verification tab listing the checks the free return passed](docs/assets/mission-control-artemis.png)

<img src="docs/assets/mission-control-shortcuts.png" alt="Mission Control keyboard shortcuts" width="320" align="right">

Shortcuts: Space play/pause · ←/→ step · Shift+←/→ previous/next event ·
Home restart · `[` `]` time warp · T track body · M focus body ·
I toggle the right panel · F1 shortcut reference.

The screenshots are regenerated with `scripts/capture_screenshots.py`.

<br clear="right">

## Performance

Measured on an RTX A3000 Laptop GPU (6 GB):

| Workload | CPU (numpy) | GPU (CuPy) | Speedup |
|---|---|---|---|
| 2,000,000 Lambert solves | 82.8 s | 17.7 s | 4.7× |
| 90,000-cell Earth–Mars porkchop | — | 1.08 s | — |

Speedup grows with batch size: it is ~2.5× at 200k solves. The GPU is not
always faster: the cislunar screening propagator is slower on the GPU below
~20,000 candidates. Fused-kernel benchmarks are in
[`docs/benchmarks/`](docs/benchmarks/fused_kernel_evaluation.md).

## Lessons learned

Root causes are written up in
[`docs/engineering-notes.md`](docs/engineering-notes.md) and
[`docs/mars_mission_plan.md`](docs/mars_mission_plan.md):

- If a pygmo `bfe` is attached after wrapping the UDA in `pg.algorithm`,
  pygmo silently evaluates one candidate at a time, ~15× slower.
- SPICE `J2000` and `ECLIPJ2000` are 23.4° apart; mixing them puts errors in
  the hundreds of thousands of km.
- Fixed-step RK4 at 600 s put a lunar flyby at 81,733 km closest approach; the
  converged value is 4,379 km.
- Aiming a Mars TCM at a fixed-time point flew the 2020 trajectory *below the
  surface* because gravity focusing bends the approach inward. The TCM now
  targets periapsis in the B-plane.
- A conservation check that used SPICE's Mars GM instead of the GM tudat
  integrates with (1.9e-8 apart) reported 1.2e-7 of energy drift that was not
  there.

## Limitations

- The batched Lambert solver is 0-revolution only. Multi-rev falls back to
  pykep's Izzo solver.
- Verification models point-mass gravity plus optional central-body
  spherical harmonics. It has no SRP, drag or relativity.
- No low-thrust trajectories (pykep's Sims-Flanagan modules would plug in like `mga.py`).
- The GPU screening ephemeris for the Moon and Sun is a two-body osculating
  orbit, good for spans of up to ~10 days from its SPICE seed.
- Missions end at orbit insertion, with no EDL. The TCM/MOI targeters are
  not guaranteed to converge for every geometry. When the TCM doesn't
  close, or a flown candidate's verification is failed / not checked, the
  Mars pipeline moves on to the next candidate. It never exports a trajectory
  that hits Mars, and an export where no candidate passed verification is
  flagged best-effort (`success: false` plus a warning in the pipeline result,
  scene and CLI output).

## License

GPL-3.0-or-later. See [`LICENSE`](LICENSE). Copyright (C) 2026 Jiucheng Zang.
