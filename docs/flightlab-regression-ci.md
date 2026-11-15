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
| gfold (FL-303 landing solver) | `gfold==0.3.2`, exact (wheels need Python >= 3.11) | `envs/flightlab-landing.txt`; CI job `flightlab-landing` runs it on Python 3.12 via `envs/flightlab-landing-ci.txt` |
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

## E2-04: the frozen dual-branch reference mission (task #257)

FL-601 freezes the ascent-only Gate B regression. **E2-04** freezes the
*complete dual-branch mission* end to end as the E2 closure baseline: the
same `leo_two_stage` / `synthetic_two_stage` campaign, but an ascent whose
separation hands the booster 35 t of propellant (frozen run directory,
pinned constants in `scripts/e204_reference.py`), the upper branch parked at
400 km @ 51.6 deg, and the booster branch flown back and landed through the
full recovery chain (FL-302 plan recipe, gfold 0.3.2 landing stack) — the
E2-03 full-mission verdict (`status: OK`, `mass.separation_fork`, station
chain, landing gates) as one comparable artifact.

* **Frozen inputs:** the mission is declared in the packaged scenario
  `src/orbitopt/data/mission_two_stage_leo_recovery.yaml` (recovery plan
  recipe pinned: the proven-green nominal 7-vector guess, `max_fevals 18`;
  `expected.propagation_result: not_checked` — the tudatpy leg belongs to
  M-07, the gfold-pinned stack cannot run it). The ascent input is rebuilt
  from the literal `ASCENT` constants (pad, epoch `2026-11-10T12:00:00Z`,
  textbook separation state, masses) — a past epoch plus the no-download
  IERS pin (above) makes the frame-bridge numbers identical across IERS
  table versions.
* **Frozen result + backend versions:** `tests/data/mission/reference/`
  `e2-04-frozen-mission/` holds the reference report + merged recovery run
  and `tolerances.json`; `PROVENANCE.json` records the sha256 of every file
  plus the toolchain fingerprint (gfold pin, numpy/scipy/astropy/
  astropy-iers-data versions, Python, platform) that produced it.
* **CI gate:** the `flightlab-landing` job (gfold 0.3.2, Python 3.12) runs
  `tests/test_e204_reference.py` — fast provenance/pin/tolerance guards
  (which also run, skipping the numerical test, in every other job) and the
  slow regression: re-fly the dual-branch mission and compare every
  trajectory sample, event, check value and verdict within the declared
  tolerances. Out of band = red. The G5 pair (`verify` + `tag-check`,
  `[e204-reference-update]`) guards the committed reference the same way.

```bash
python scripts/e204_reference.py run --out /tmp/e204        # re-fly
python scripts/e204_reference.py compare /tmp/e204          # vs reference
python scripts/e204_reference.py update                     # intended change:
git add tests/data/mission/reference/e2-04-frozen-mission \
        src/orbitopt/data/mission_two_stage_leo_recovery.yaml
git commit -m "test(mission): re-record E2-04 frozen reference [e204-reference-update]"
```
