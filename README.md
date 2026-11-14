# orbitopt

[![CI](https://github.com/zangjiucheng/orbitopt/actions/workflows/ci.yml/badge.svg)](https://github.com/zangjiucheng/orbitopt/actions/workflows/ci.yml)
[![Pages](https://github.com/zangjiucheng/orbitopt/actions/workflows/pages.yml/badge.svg)](https://github.com/zangjiucheng/orbitopt/actions/workflows/pages.yml)
[![Docs](https://img.shields.io/badge/docs-orbitopt-e8a23e)](https://zangjiucheng.github.io/orbitopt/docs/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![License: GPL v3+](https://img.shields.io/badge/license-GPL--3.0--or--later-blue)](LICENSE)

orbitopt searches trajectories with pykep and pygmo, verifies selected
candidates with tudatpy, and visualizes them in **Mission Control**. **Flight
Lab** simulates launch and booster recovery. CuPy can accelerate batched
screening.

![Mission Control showing an Earth-to-Mars trajectory](docs/assets/mission-control-mars.png)

[Documentation](https://zangjiucheng.github.io/orbitopt/docs/) ·
[Quickstart](docs/quickstart.md) · [Mission Control](docs/guide.md) ·
[Flight Lab](docs/flightlab.md) · [Verification](docs/verification.md) ·
[API reference](docs/api.md)

## Quick start

From a repository checkout:

```bash
micromamba create -n orbitopt -f envs/cpu.conda-lock.yml
micromamba activate orbitopt
pip install -e . --no-deps

orbitopt             # Mission Control
orbitopt missions    # list missions
orbitopt view mars   # single-scene viewer
```

See [installation](docs/quickstart.md#installation) for GPU, Windows, macOS
and viewer-only options.

## CLI

```bash
orbitopt validate examples/mars2020_replay.yaml
orbitopt run examples/mars2020_replay.yaml
orbitopt launch --site cape-canaveral --vehicle synthetic_two_stage \
  --orbit 400x400@51.6 --date 2026-11-14 -o launch/
orbitopt benchmarks
```

`orbitopt --help` lists commands; the [quickstart](docs/quickstart.md) covers
outputs and required dependencies.

## Mission configs

YAML kinds include `geo-raising`, `mars-transfer`, `lunar-free-return`,
`flight-lab` and `two-stage-leo`. See [mission configs](docs/quickstart.md#mission-configs)
and [examples](docs/examples.md).

## Verification reports

Reports record models, numerical settings, residuals and evidence categories.
A pass applies only to the stated checks and assumptions; GOES loads **not
checked** by default, and an unverified Mars result may be **best effort**.
See [verification](docs/verification.md).

## Mission Control

The desktop app links a 3D view, timeline, charts and verification inspector.
Its Flight Lab mode adds launch design, replay and comparison. See the
[guide](docs/guide.md) and [Flight Lab](docs/flightlab.md).

## Checked against real missions

[Validation examples](docs/examples.md) distinguish historical comparisons
from target-based checks.

## Performance

On an RTX A3000 Laptop GPU, 2 million Lambert solves took 17.7 s with CuPy
versus 82.8 s with NumPy (4.7×). Results depend on hardware and batch size;
see the [performance overview](https://zangjiucheng.github.io/orbitopt/#results).

## Layout

`src/orbitopt/` contains optimization, verification, Flight Lab, mission
pipelines and visualization; `examples/`, `docs/` and `tests/` contain their
examples, guides and regression tests. Preview the site with
`cd docs && bundle install && bundle exec jekyll serve`.
See the [translation workflow](translations/README.md) for automatic Chinese review copies.

## Installing without conda

The scene format and viewer can be installed separately; full trajectory
optimization and verification need the [conda environment](envs/README.md).

## Lessons learned

See [engineering notes](docs/engineering-notes.md) for numerical and UI
pitfalls and the [Mars plan](docs/mars_mission_plan.md) for model assumptions.

## Limitations

- The batched Lambert solver supports zero revolutions; multi-revolution cases
  use pykep. Low-thrust optimization is not implemented.
- Orbital verification omits solar radiation pressure, drag and relativity.
- Cislunar GPU screening uses an approximate ephemeris intended for spans of
  about ten days from its SPICE seed.
- Mars targeters may fail to converge; the modeled mission ends at orbit
  insertion, not landing. Flight Lab uses simplified dynamics and public or
  illustrative vehicle data; its 6-DOF backend is opt-in and outside the
  ascent optimizer. See [fidelity and limitations](docs/flightlab.md#fidelity-levels).

## License

GPL-3.0-or-later. See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY.md).
Copyright © 2026 Jiucheng Zang.
