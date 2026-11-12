# orbitopt

[![CI](https://github.com/zangjiucheng/orbitopt/actions/workflows/ci.yml/badge.svg)](https://github.com/zangjiucheng/orbitopt/actions/workflows/ci.yml)
[![Pages](https://github.com/zangjiucheng/orbitopt/actions/workflows/pages.yml/badge.svg)](https://github.com/zangjiucheng/orbitopt/actions/workflows/pages.yml)
[![Docs](https://img.shields.io/badge/docs-orbitopt-e8a23e)](https://zangjiucheng.github.io/orbitopt/docs/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![License: GPL v3+](https://img.shields.io/badge/license-GPL--3.0--or--later-blue)](LICENSE)

orbitopt searches spacecraft trajectories with pykep and pygmo, then checks
selected candidates with tudatpy. CuPy can accelerate batched screening.
**Mission Control** displays trajectories, timelines and verification reports;
**Flight Lab** simulates launch and booster recovery.

![Mission Control showing an Earth-to-Mars trajectory](docs/assets/mission-control-mars.png)

[Documentation](https://zangjiucheng.github.io/orbitopt/docs/) ·
[Quickstart](docs/quickstart.md) · [Mission Control guide](docs/guide.md) ·
[Flight Lab](docs/flightlab.md) · [Examples and validation](docs/examples.md) ·
[API reference](docs/api.md)

## Quick start

From a clone of this repository, create the locked CPU environment and install
the project:

```bash
micromamba create -n orbitopt -f envs/cpu.conda-lock.yml
micromamba activate orbitopt
pip install -e . --no-deps

orbitopt                     # Mission Control
orbitopt missions            # list built-in missions
orbitopt view mars           # open the single-scene viewer
```

For GPU, Windows, macOS and optional Flight Lab dependencies, see
[installation](docs/quickstart.md#installation) and
[environment details](envs/README.md).

## CLI

```bash
orbitopt validate examples/mars2020_replay.yaml  # check a config without computing
orbitopt run examples/mars2020_replay.yaml       # compute and export a scene
orbitopt export mars -o mars.json                # export a built-in mission
orbitopt flight --help                          # launch and recovery tools
```

Run `orbitopt --help` for all commands. The [quickstart](docs/quickstart.md)
explains configs, outputs and launch windows.

## Mission configs

YAML configs support `geo-raising`, `mars-transfer` and `lunar-free-return`.
See [mission configuration](docs/quickstart.md#mission-configs) and the
[examples](examples/). The Solar System scene is an ephemeris visualization,
not an optimized mission.

## Verification reports

Reports record the model, numerical settings, residuals and checks. A
verification verdict applies to those checks and assumptions; it is not proof
of real-world flight safety. A Mars export may be marked **best effort** if no
candidate passes, and GOES loads **not checked** by default. See
[verification](docs/quickstart.md#verification) and
[mission examples](docs/examples.md).

## Mission Control

The desktop app links a 3D view, timeline, charts and verification inspector.
Flight Lab adds a launch designer, replay and comparison. See the
[Mission Control guide](docs/guide.md) and [Flight Lab guide](docs/flightlab.md).

## Checked against real missions

Historical launch-window comparisons and target-based checks answer different
questions. The [validation examples](docs/examples.md) identify which values
are inputs, what was compared, and the limits of each model.

## Performance

On an RTX A3000 Laptop GPU, 2 million Lambert solves took 17.7 s with CuPy
versus 82.8 s with NumPy (4.7×). Performance depends on batch size and hardware;
see the [benchmark](docs/benchmarks/fused_kernel_evaluation.md).

## Layout

| Path | Purpose |
|---|---|
| `src/orbitopt/lambert`, `problems`, `optimize` | Screening and optimization |
| `src/orbitopt/verify` | Propagation, targeting and verification reports |
| `src/orbitopt/flightlab` | Launch and recovery simulation |
| `src/orbitopt/viz` | Scene export, viewer and Mission Control |
| `examples`, `docs`, `tests` | Examples, guides and regression tests |

The documentation site is built from `docs/` with Jekyll. To preview it, run
`cd docs && bundle install && bundle exec jekyll serve`.

## Installing without conda

The scene format and viewer can be installed without the compute stack; see
[installation options](docs/quickstart.md#installation). Full trajectory
optimization and verification require the locked conda environment.

## Lessons learned

See [engineering notes](docs/engineering-notes.md) for numerical, frame and
UI pitfalls, and the [Mars mission plan](docs/mars_mission_plan.md) for its
phase-boundary assumptions.

## Limitations

- The batched Lambert solver supports zero revolutions; multi-revolution cases
  use pykep's Izzo solver. Low-thrust optimization is not implemented.
- Verification results depend on the specified force model. The orbital
  verification layer omits solar radiation pressure, drag and relativity.
- The cislunar GPU screening ephemeris is a two-body approximation intended for
  spans of about ten days from its SPICE seed.
- Mars mission targeters may fail to converge; a failed or unchecked candidate
  is not a verified result. The Mars scene ends at orbit insertion, not landing.
- Flight Lab uses simplified dynamics and public or illustrative vehicle data;
  its [fidelity and limitations](docs/flightlab.md#fidelity-levels) are explicit.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY.md).
Copyright © 2026 Jiucheng Zang.
