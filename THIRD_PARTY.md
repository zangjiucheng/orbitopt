# Third-party components and licensing

orbitopt is distributed under the **GPL-3.0-or-later** (see `LICENSE`;
`NOTICE` carries the OpenTsiolkovsky MIT text). This file lists every
third-party component the project depends on or drives, how it is used, and
why its license is compatible with GPL-3.0-or-later. The exact versions are
pinned and recorded per run (FL-601): conda packages in
`envs/{cpu,gpu}.conda-lock.yml`, the pip add-on layer in
`envs/flightlab-landing.txt` and `envs/sixdof-requirements.txt`, the
external OpenTsiolkovsky binary by version + SHA-256 in
`scripts/fetch_opentsiolkovsky.py` -- and all of it is written into every
Flight Lab run manifest under `toolchain`
(`orbitopt.flightlab.toolchain.toolchain_record`).

## OpenTsiolkovsky -- MIT; external binary, NOT distributed

- Upstream: <https://github.com/istellartech/OpenTsiolkovsky> (Rust), Interstellar Technologies Inc.
- Version: pinned release **v1.0.0** (`scripts/fetch_opentsiolkovsky.py`,
  per-platform SHA-256 verified at fetch; `ORBITOPT_OPENTSIOLKOVSKY`
  overrides the binary path).
- Usage: `orbitopt.flightlab.adapters.opentsiolkovsky` generates a
  run-config JSON, invokes `openTsiolkovsky-cli` as a **separate process**
  and parses its outputs. It is never linked, vendored, embedded or
  distributed with orbitopt: the fetch script downloads it at setup time
  into `~/.cache/orbitopt/bin/`. Recorded test fixtures under
  `tests/fixtures/flightlab/opentsiolkovsky/` are data files it produced.
- GPL compatibility: MIT is a permissive license compatible with GPL-3.0-or-later.
  Because the binary is only invoked as a separate program and is not part
  of orbitopt's distribution, no combined-work licensing question arises at
  all; even treated as a work that interacts with orbitopt at arm's length,
  MIT permits the use and imposes nothing on orbitopt's GPL grant.

## gfold -- MIT; pip add-on, exact pin

- Upstream: PyPI `gfold` (Rust core, Clarabel SOCP) -- the FL-303 G-FOLD
  landing-burn solver used by `orbitopt.flightlab.recovery.landing`.
- Version: **0.3.2**, pinned exactly in `envs/flightlab-landing.txt`
  (solver discretization/bound semantics are version-sensitive; the module
  refuses to run against a drifted install).
- Usage: imported as a Python library at run time; it is an optional add-on
  layer installed on top of the locked conda env, not a build dependency,
  and tests skip cleanly when it is absent.
- GPL compatibility: MIT is compatible with GPL-3.0-or-later. If gfold's SOCP
  output were a derivative-work question at all (it is a plain library call,
  so it is not), MIT's terms permit combining under GPL-3.0-or-later.

## JSBSim -- LGPL-2.1-or-later; dynamic Python import

- Upstream: <https://jsbsim-team.github.io/jsbsim-team/> (C++ flight
  dynamics model), Python bindings wheel on PyPI.
- Version: **1.3.3**, hash-pinned in `envs/sixdof-requirements.txt`
  (opt-in; the FL-401 E4 spike, `orbitopt.flightlab.sixdof`).
- Usage: `orbitopt.flightlab.sixdof.jsbsim_driver` imports the `jsbsim`
  Python module dynamically at run time (the base package imports nothing of
  it) and drives it through its public Python API with XML aircraft
  configurations under `src/orbitopt/data/flightlab/jsbsim/`.
- GPL compatibility: LGPL-2.1+ permits use of the library from GPL-3.0 code
  via **dynamic linking** -- the Python `import` of the separately installed
  wheel is exactly that; no JSBSim source is combined into orbitopt, users
  can swap the wheel (the mechanism LGPL-2.1 §6 requires), and orbitopt's
  GPL grant is unaffected. The data/XML files we ship are our own
  configurations.

## astropy -- BSD-3-Clause; conda-locked

- Upstream: <https://www.astropy.org>.
- Version: **6.1.7**, pinned in `envs/{cpu,gpu}.yml` and locked in
  `envs/*.conda-lock.yml`.
- Usage: `orbitopt.flightlab.frames` (FL-002) uses astropy's ITRS↔GCRS frame
  transform at the launch epoch; the import is lazy, so orbitopt runs without
  astropy installed (at the cost of the frame bridge).
- GPL compatibility: the BSD-3-Clause license is compatible with
  GPL-3.0-or-later; astropy is used as a normal library dependency.

## tudatpy -- BSD-3-Clause; conda-locked

- Upstream: <https://github.com/tudat-team/tudatpy>, TU Delft.
- Version: **1.0.x** (`tudatpy=1.0` in `envs/{cpu,gpu}.yml`), from the
  tudat-team conda channel; the exact build is in the lockfiles.
- Usage: the independent physics verification behind Gate B and Gate C --
  SPICE frame services and `orbitopt.verify.tudat_propagate` re-flights of
  recorded insertion states (see `orbitopt.flightlab.simulate.orbit_refly`).
  Optional at run time: without it the physics gate reports "not assessed".
- GPL compatibility: BSD-3-Clause is compatible with GPL-3.0-or-later; used
  as a normal library dependency.

## Other pinned pieces (for completeness)

- **cvxpy** (Apache-2.0, in `envs/flightlab-landing.txt`): reference SOCP
  implementation for cross-checking gfold in tests only. Apache-2.0 is
  compatible with GPL-3.0-or-later (one-way; see GPLv3 §11).
- **pykep / pygmo** (GPL-3.0-or-later): the mission-design search stack --
  same license as orbitopt, no issue.
- **CuPy** (MIT), **PyVista / VTK** (BSD-3-Clause), **PySide6** (LGPL-3.0
  with the Qt `LGPL` option, used as a dynamic import; the optional
  commercial alternative is simply not exercised), **conda-forge runtime
  libs** per the lockfiles -- all license-compatible with GPL-3.0-or-later.

## Process rules (FL-601 / G5)

- A Flight Lab run manifest records the whole toolchain
  (`manifest.toolchain`, asserted in `tests/test_flightlab_contract.py`);
  a new run is therefore attributable to an exact environment.
- The Gate B regression reference under `tests/data/flightlab/reference/`
  is regenerated only by `scripts/fl601_reference.py update`, and a commit
  that changes it must carry the tag `[fl601-reference-update]` in its
  message (enforced in CI; tamper-checked by `fl601_reference.py verify`).
- Adding a new third-party runtime dependency requires a row in this file
  (license, pin location, usage) as part of the change.
