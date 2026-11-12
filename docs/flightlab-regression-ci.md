# Flight Lab reproducibility & regression CI (FL-601)

Every Flight Lab run is attributable to an exact toolchain, and the
OpenTsiolkovsky-driven Gate B pipeline has a numeric regression baseline.
This page is the operational summary; the licenses are in
[`THIRD_PARTY.md`](../THIRD_PARTY.md).

## The pinned toolchain

| Tool | Pin | Where |
|---|---|---|
| python / conda stack | python 3.10, tudatpy 1.0, numpy 1.26.4, … | `envs/{cpu,gpu}.yml` + `envs/*.conda-lock.yml` (CI job `lockfile-drift` fails spec/lock skew) |
| astropy (frame bridge) | `astropy=6.1.7`, exact | `envs/{cpu,gpu}.yml` |
| IERS Earth-orientation data | `astropy-iers-data` from the lockfile, **no download** in the Gate B regression | `scripts/fl601_reference.py` (`_pin_earth_orientation`) |
| gfold (FL-303 landing solver) | `gfold==0.3.2`, exact | `envs/flightlab-landing.txt` |
| jsbsim (E4 6-DOF spike) | `jsbsim==1.3.3`, hash-locked | `envs/sixdof-requirements.txt` |
| OpenTsiolkovsky (external binary) | v1.0.0, per-platform SHA-256 | `scripts/fetch_opentsiolkovsky.py` (fetch refuses any other bytes; `--check` verifies what is installed) |

Every FlightResult manifest written by `orbitopt.flightlab.contract.write_run`
carries all of this under `manifest.toolchain` (interpreter, orbitopt, every
tool distribution's installed version or `"absent"`, OpenTsiolkovsky's pinned
version and the SHA-256 of the binary actually found on disk).
`tests/test_flightlab_contract.py` asserts the record is complete.

### Why the regression never downloads IERS data

The frame bridge rotates OpenTsiolkovsky's Earth-fixed states into GCRS
with astropy, which needs Earth-orientation data (UT1-UTC, polar motion).
astropy downloads the latest IERS-A table by default, and that table changes
every day as predictions become measurements: the same flight's GCRS
positions moved by tenths of a metre between days, far beyond the
micrometre tolerances (the regression's first CI run failed on 7,670
samples). `fl601_reference.py run` / `update` therefore use only the table
bundled with the pinned `astropy-iers-data`. It ends before the reference
epoch, so astropy takes UT1 = UTC and zero polar motion there: a frame a few
tens of metres off the true one, but the same on every run with the same
lockfile. Bumping `astropy-iers-data` in the lockfile changes the reference;
re-record it with `update` and the `[fl601-reference-update]` tag. Record
it in the locked environment (CI's), not a local one.

## CI layout (.github/workflows/ci.yml)

* **`test` (linux-64, locked env)** — Flight Lab unit tests run on the
  recorded fixtures only: the OT adapter tests replay recorded v1.0.0
  outputs through a stub CLI, so no OT binary is needed (nor present) here.
  This job also runs `tests/test_flightlab_reference.py`, the provenance
  tamper check of the committed regression reference.
* **`flightlab-gateb` (slow, linux-64, locked env)** — fetches the pinned
  OpenTsiolkovsky binary (SHA-256 verified at download, version-probed),
  flies the reference Gate B mission
  (`leo_two_stage` target + `synthetic_two_stage` vehicle, ascent-only,
  `refly=False`, `run_recovery=False`) with the real binary and compares
  every trajectory sample, event and check against
  `tests/data/flightlab/reference/` within the tolerances declared in
  `tests/data/flightlab/reference/tolerances.json`. It also runs the two
  G5 guards: `verify` (reference files match their provenance digests) and
  `tag-check` (any diff touching `tests/data/flightlab/reference/` needs
  `[fl601-reference-update]` in a commit message of the push/PR range).
* **`test-macos`** — the full suite on the locked env, arm64 + x86_64 (no
  Gate B job on macOS in CI; see the manual run below).

## The reference dataset

```
tests/data/flightlab/reference/
  gateb-leo-two-stage/     the recorded Gate B run dir (manifest, trajectories,
                           events, verification + raw opentsiolkovsky/ outputs)
  tolerances.json          the declared per-quantity tolerances
  PROVENANCE.json          sha256 of every file above, written by the script
```

Updating (e.g. when re-pinning OpenTsiolkovsky, or when an intended physics
change lands):

```bash
python scripts/fetch_opentsiolkovsky.py --force   # new pin already in the script
python scripts/fl601_reference.py update          # refly the run, rewrite PROVENANCE.json
git add tests/data/flightlab/reference
git commit -m "test(flightlab): re-record Gate B reference for OT vX.Y.Z [fl601-reference-update]"
```

Anything else is rejected by CI: hand-edited reference files fail `verify`
(in the fast job), un-tagged reference changes fail `tag-check`, and an
unintended physics change fails the numerical `compare`.

## Manual macOS-arm64 run

CI's Gate B job is Linux-only (the OT macOS binary is pinned and fetched the
same way, but we keep one canonical CI platform). Do this on an Apple
Silicon Mac after OT-pin or physics-sensitive changes:

```bash
# 1. locked env (osx-arm64 section) -- see envs/README.md
micromamba create -n orbitopt -f envs/cpu.conda-lock.yml
micromamba activate orbitopt
pip install -e . --no-deps

# 2. the pinned OT binary (macos-arm64 asset, SHA-256 verified)
python scripts/fetch_opentsiolkovsky.py          # installs + --check

# 3. the regression, exactly as CI runs it
python scripts/fl601_reference.py verify
python scripts/fl601_reference.py run --out /tmp/gateb
python scripts/fl601_reference.py compare /tmp/gateb
```

Expect `compare` to pass; on a genuine platform delta it may come apart
within the declared tolerances only if the tolerances are too tight --
that is a `tolerances.json` update, which goes through the same
update-script + tagged-commit route as the reference itself.
