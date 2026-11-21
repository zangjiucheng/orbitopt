<p align="center">
  <img src="docs/assets/orbitopt-icon.png" alt="orbitopt icon: a tilted orbit with a spacecraft leaving a gap in it, around a central body" width="112" height="112">
</p>

<h1 align="center">orbitopt</h1>

<div align="center">

[![CI](https://github.com/zangjiucheng/orbitopt/actions/workflows/ci.yml/badge.svg)](https://github.com/zangjiucheng/orbitopt/actions/workflows/ci.yml)
[![Pages](https://github.com/zangjiucheng/orbitopt/actions/workflows/pages.yml/badge.svg)](https://github.com/zangjiucheng/orbitopt/actions/workflows/pages.yml)
[![Docs](https://img.shields.io/badge/docs-orbitopt-111111)](https://zangjiucheng.github.io/orbitopt/docs/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![License: GPL v3+](https://img.shields.io/badge/license-GPL--3.0--or--later-blue)](LICENSE)

</div>

orbitopt optimizes trajectories with pykep and pygmo, verifies candidates
with TudatPy, and explores results in **Mission Control**. **Flight Lab**
simulates launch and booster recovery, compares saved runs, and displays
vehicle profiles with declared-versus-flown data. CuPy accelerates batched
screening.

![Mission Control replaying a Starship Flight 15 launch: Super Heavy on its boostback and the ship climbing, each with its flight-data label](docs/assets/mission-control-starship-ascent.png)

## Quick start

From a repository checkout:

```bash
micromamba create -n orbitopt -f envs/cpu.conda-lock.yml
micromamba activate orbitopt
pip install -e . --no-deps
orbitopt
```

See [installation](docs/quickstart.md#installation) for GPU, macOS, Windows
and viewer-only options. Use `orbitopt --help` for commands.

## Explore

- [Quickstart](docs/quickstart.md): run YAML mission configs and inspect outputs.
- [Mission Control](docs/guide.md): 3D view, timeline, telemetry and verification.
- [Flight Lab](docs/flightlab.md): ascent, recovery and run comparisons;
  [Vehicle view](docs/flightlab.md#vehicle-view) shows the catalog and flight overlays.
- [Computed mission](docs/computed-mission.md): replay ascent and recovery,
  then propagate across TudatPy and G-FOLD environments;
  [Kourou/Starbase designs](docs/e5-08-new-vehicle-report.md) report remaining qualification gaps.
- [API reference](docs/api.md): Python interfaces, CLI commands and schemas.

Verification applies to declared models and tolerances. See
[verification](docs/verification.md) and
[Flight Lab fidelity](docs/flightlab.md#backend-identities-and-fidelity-levels)
for evidence boundaries and limitations.

## License

GPL-3.0-or-later · [License](LICENSE) · [Third-party notices](THIRD_PARTY.md).
Copyright © 2026 Jiucheng Zang.
