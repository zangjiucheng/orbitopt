# Conda environments

| File | Edited by | Purpose |
|---|---|---|
| `cpu.yml` | humans | CPU-only spec: tudatpy/pykep/pygmo compute stack + PyVista/PySide6 viewer |
| `gpu.yml` | humans | `cpu.yml` + `cupy` on `cuda-version=13` |
| `cpu.conda-lock.yml` | conda-lock | exact pins for `cpu.yml` (linux-64, win-64) |
| `gpu.conda-lock.yml` | conda-lock | exact pins for `gpu.yml` (linux-64, win-64) |
| `virtual-packages-gpu.yml` | humans | `__cuda=13.0` for the GPU solve (conda-lock defaults to 11.4) |

Everything comes from conda-forge / tudat-team (CuPy and the viewer packages
included), so the lockfiles cover the whole environment -- there is no
unpinned `pip:` section.

## Install

```bash
micromamba create -n orbitopt     -f envs/cpu.conda-lock.yml
micromamba create -n orbitopt-gpu -f envs/gpu.conda-lock.yml
# or, with conda:  conda-lock install -n orbitopt envs/cpu.conda-lock.yml
pip install -e . --no-deps
```

## Version constraints

* `python=3.10` -- tudatpy's tudat-team builds.
* `tudatpy=1.0` -- 1.0.0 declares its boost (`boost-cpp 1.84`) and numpy
  pins. An unpinned solve picked tudatpy 0.9.0, whose binary links
  libboost 1.85 without declaring it; a later `tudat` rebuild moved to 1.84,
  so that env failed with `libboost_unit_test_framework.so.1.85.0` not found.
* `numpy>=1.26.4,<2` -- tudatpy 1.0.0's real conda requirement. The old
  `environment.yml` said `numpy<2.0.0` yet installs ended up on numpy 2.x:
  its unlocked `pip:` section (cupy-cuda13x, pyvista, pyside6, pyvistaqt)
  pulled numpy 2 wheels over the conda numpy. Now that every package
  (CuPy included) comes from conda and is locked, the declared constraint
  and the installed version agree; the lock pins the exact version.
* `scipy>=1.11,<1.14` -- pykep.trajopt still calls
  `scipy.interpolate.interp2d`, removed in scipy 1.14.
* `cuda-version=13` (GPU only) -- selects CUDA 13 CuPy/cudart builds. To
  target another CUDA major, change it in `gpu.yml` and re-lock. The host
  only needs a compatible NVIDIA driver.

`tests/test_env_lockfiles.py` checks the locks honour these constraints and
that CuPy never appears in the CPU env.

## Regenerating the locks

After editing `cpu.yml` / `gpu.yml`:

```bash
conda-lock lock --micromamba -f envs/cpu.yml -p linux-64 -p win-64 --lockfile envs/cpu.conda-lock.yml
conda-lock lock --micromamba -f envs/gpu.yml -p linux-64 -p win-64 \
    --virtual-package-spec envs/virtual-packages-gpu.yml --lockfile envs/gpu.conda-lock.yml
```

To bump pins without editing the spec, delete the lockfile first (or use
`conda-lock lock --update <pkg>`). CI's `lockfile-drift` job fails if a spec
changes without its lockfile being regenerated.

## CI

* `.github/workflows/ci.yml` -- every push/PR: CPU lockfile on ubuntu,
  full test suite; `gpu`-marked tests are skipped.
* `.github/workflows/gpu.yml` -- the CUDA path: GPU lockfile on a
  self-hosted runner labelled `[self-hosted, linux, x64, gpu]`. Runs on
  manual dispatch or when a PR carries the `gpu-ci` label. It sets
  `ORBITOPT_REQUIRE_GPU=1`, which makes `tests/conftest.py` error out
  rather than skip if no CUDA device is visible, then runs `pytest -m gpu`.

Locally on a CUDA machine: `ORBITOPT_REQUIRE_GPU=1 pytest -m gpu -v`.
