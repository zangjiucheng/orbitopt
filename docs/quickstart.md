---
title: Quickstart
nav_order: 2
description: Install orbitopt, open a mission and run a configuration.
---

# Quickstart

## Installation

Clone the repository and create the locked environment for the full compute
stack. It includes the viewer, pykep, pygmo and tudatpy.

```bash
git clone https://github.com/zangjiucheng/orbitopt.git
cd orbitopt
micromamba create -n orbitopt -f envs/cpu.conda-lock.yml
micromamba activate orbitopt
pip install -e . --no-deps
```

Use `envs/gpu.conda-lock.yml` for CUDA on Linux or Windows. On Windows,
`.\scripts\setup_env.ps1` also creates the environment; add `-Gpu` for CUDA.
macOS uses the CPU lock. See [environment details](../envs/README.md) for
platforms and optional dependencies.

For scene files without the compute stack, install from the checkout with
`pip install -e .`; use `pip install -e '.[viewer]'` to add the viewer.
Python 3.10–3.12 is supported; the locked compute environment uses Python 3.10.

## Open a mission

```bash
orbitopt                   # Mission Control
orbitopt missions          # list mission IDs
orbitopt view mars         # single-scene viewer
orbitopt export mars -o mars.json
```

Built-in scenes include `solar-system`, `artemis2`, `goes`, `mars` and
`starship-flight-15`. See the [Mission Control guide](guide.md) or
[Flight Lab guide](flightlab.md).

## Mission configs

Validate a YAML mission before running it:

```bash
orbitopt validate examples/mars2020_replay.yaml
orbitopt run examples/mars2020_replay.yaml
orbitopt view examples/mars2020_replay.yaml
```

`validate` checks the schema without running a simulation. `run` computes and
exports the scene and reports; `view` opens a cached or newly computed result.
The supported kinds are `geo-raising`, `mars-transfer`,
`lunar-free-return`, `flight-lab` (a Flight Lab ascent run through the
pipeline; see `src/orbitopt/data/flightlab/mission_leo_two_stage.yaml`) and
`two-stage-leo` (the reference scenario below).
Each has a schema in
[`src/orbitopt/schemas/`](../src/orbitopt/schemas/) and example configurations
in [`examples/`](../examples/). Omitted optional fields use the corresponding
built-in defaults.

To calculate a daily launch window:

```bash
orbitopt launch-window ksc --rla 9.6 --dla 23.4 --epoch 2020-07-30
```

Use `orbitopt --help` or `orbitopt <command> --help` for all options.

## Launch to orbit

`orbitopt launch` runs a launch end to end: the launch window from a site into
an orbit plane, an optimized ascent flown by Flight Lab, and orbit injection
graded against the target, ending in one verification report. It needs the
pinned OpenTsiolkovsky binary (`python scripts/fetch_opentsiolkovsky.py`).

```bash
orbitopt launch --site cape-canaveral --vehicle synthetic_two_stage \
  --orbit 400x400@51.6 --date 2026-11-14 -o launch/
```

The output directory holds the target, the ascent run and
`launch.verification.json`. Add `--raan` to constrain the orbit plane (the
in-plane pass then sets the liftoff time). A short search
(`--generations`, `--population`) may not converge; the report says so rather
than passing.

The two-stage LEO reference scenario grades the whole chain, from the launch
site through staging and orbit insertion to the propagated parking orbit, into
one report:

```bash
orbitopt run src/orbitopt/data/mission_two_stage_leo.yaml
```

See [Flight Lab](flightlab.md#mission-pipeline) for the segments and their
handoff gates.

## Verification

A verification report records the dynamics, integration settings, checks and
residuals. A passing result is conditional on the model and tolerances; it does
not establish flight safety. GOES is **not checked** on first load: run its
verification from the app's Verification tab, or enable it in a config. A
Mars run may export a **best-effort** result when no candidate passes.

Each check is labelled numerical consistency, model agreement or historical
validation; `orbitopt benchmarks` checks the physics against analytic
solutions and published data. See [verification](verification.md).

See [examples and validation](examples.md) for the distinction between
historical comparisons and targeting checks, and
[engineering notes](engineering-notes.md) for report and pipeline details.
