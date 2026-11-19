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
| IERS Earth-orientation data | `astropy-iers-data` from the lockfile (E8-04 snapshot-pinned), **no download** in any reference regression | `orbitopt.reference_eop` via `scripts/{fl601,e204}_reference.py` (`_pin_earth_orientation`) |
| gfold (FL-303 landing solver) | `gfold==0.3.2`, exact (wheels need Python >= 3.11) | `envs/flightlab-landing.txt`; CI job `flightlab-landing` runs it on Python 3.12 via `envs/flightlab-landing-ci.txt` |
| jsbsim (E4 6-DOF spike) | `jsbsim==1.3.3`, hash-locked | `envs/sixdof-requirements.txt` |
| OpenTsiolkovsky (external binary) | v1.0.0, per-platform SHA-256 | `scripts/fetch_opentsiolkovsky.py` (fetch refuses any other bytes; `--check` verifies what is installed) |

Every FlightResult manifest written by `orbitopt.flightlab.contract.write_run`
carries all of this under `manifest.toolchain` (interpreter, orbitopt, every
tool distribution's installed version or `"absent"`, OpenTsiolkovsky's pinned
version and the SHA-256 of the binary actually found on disk).
`tests/test_flightlab_contract.py` asserts the record is complete.

### How an installed version is looked up (task #295)

The lookup (`orbitopt.flightlab.toolchain.installed_version`, also used by the
verification-report provenance in `orbitopt.verify.report`) never imports the
tool. It asks `importlib.metadata` first; a package that answers is reported at
its recorded version. A conda build can ship no `dist-info` — `tudatpy` from the
`tudat-team` channel does not — and then `importlib.metadata` raises
`PackageNotFoundError` for a package the run demonstrably used: CI run
44140956089 sealed `tudatpy: absent` into the cross-environment envelope of the
very environment that flew the ascent. So the lookup falls back to the
environment's own `conda-meta/<name>-<version>-<build>.json`, and a tool that is
importable but recorded by neither source is reported as
`installed (version unknown)` — never `absent`, which would misdescribe the
environment.

The fallback is keyed on nothing but the distribution name, because a missing
`dist-info` is a property of how a channel builds packages, not of one package.
The per-tool conclusion (checked in CI run 44401859923 and locally):

| Tool | Install path | `dist-info`? | Consequence |
|---|---|---|---|
| `tudatpy` | conda, `tudat-team` | **no** | conda-meta gives the version the lockfile installed (`1.0.0` in `cpu.conda-lock.yml`); the CI envelope and the manifest carry it |
| `gfold` | pip wheel (`envs/flightlab-landing.txt`) | yes | first source answers; `tests/test_flightlab_toolchain.py` pins that it reports a version, not `absent` |
| `cvxpy` | pip (`envs/flightlab-landing.txt`) | yes | as `gfold` (skipped where not installed) |
| `jsbsim` | pip wheel, hash-locked (`envs/sixdof-requirements.txt`) | yes | as `gfold` |
| `pygmo` / `pykep` | conda-forge | yes (conda-forge builds install through pip) | not in `TOOL_DISTRIBUTIONS`; recorded by the E2-04/E5-05 scripts, whose committed `PROVENANCE.json` carries `pygmo: 2.20.0`, so no false `absent` there |

`pyyaml` is imported as `yaml`, so the importable-but-unrecorded fallback maps
distribution names to import names before calling `find_spec`.

### Why the regression never downloads IERS data

The frame bridge rotates OpenTsiolkovsky's Earth-fixed states into GCRS
with astropy, which needs Earth-orientation data (UT1-UTC, polar motion).
astropy downloads the latest IERS-A table by default, and that table changes
every day as predictions become measurements: the same flight's GCRS
positions moved by tenths of a metre between days, far beyond the
micrometre tolerances (the regression's first CI run failed on 7,670
samples). `fl601_reference.py run` / `update` therefore use only the table
bundled with the pinned `astropy-iers-data`. E8-04 hardened this into the
shared mechanism in `orbitopt.reference_eop`: the reference directory
carries `earth_orientation.json` (mode `pinned-bundled`), which records
the recording stack's exact versions, the default offline table it used
(IERS-B) and the effective UT1-UTC / polar motion at the reference epoch;
every run verifies the installed stack reproduces all of it and refuses
with `ToolchainMismatch` instead of silently flying a different Earth
orientation (astropy 6 vs 8 disagree offline by ~28 m of track). E2-04
keeps its four-row snapshot in `override` mode -- one mechanism, per-mode
policy. Bumping `astropy-iers-data` in the lockfile invalidates the
snapshot; re-record with `update` (audited) and the
`[fl601-reference-update]` tag. Record it in the locked environment
(CI's), not a local one. Tolerance policy, bands and the change history
live in [reference-tolerance-audit.md](reference-tolerance-audit.md).

## CI layout (.github/workflows/ci.yml)

Documentation-only changes skip compute CI; benchmark JSON used as test input
remains gated. The job split is:


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
* **`flightlab-landing`** — Python 3.12 with gfold 0.3.2: recovery tests,
  E2-04 and E5-05 replay/provenance guards.
* **`cross-env`** — locked TudatPy Python 3.10 plus a Python 3.12 landing
  environment; verifies sealed exchanges and the E5-05 joined mission.
* **`test-macos`** — compute tests on the locked arm64/x86_64 environments.
  Flight Lab navigation tests use scratch folders/settings rather than the
  checkout runs or user preferences. No Gate B flight runs on macOS CI.

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

RAAN sensitivity and fixed-epoch optimizer tests compare against an
unoptimized insertion under the same Earth-orientation table. They exercise
both offline IERS-A and IERS-B; an absolute RAAN recorded with another table
is not their oracle. The production RAAN tolerance remains unchanged.

## E2-04: the frozen dual-branch reference mission (task #257)

E2-04 tests mission integration using an **illustrative ascent fixture** and
an actual simulated booster return (kept by E5-01 as a fast-test exception --
see the E5-01 section below). `scripts/e204_reference.py` constructs
its two-sample ascent arcs from constants: separation has 35 t of booster
propellant, and insertion is prescribed at 400 km / 51.6°. Neither ascent
optimization nor continuous ascent propagation runs in this regression.

The scenario is `src/orbitopt/data/mission_two_stage_leo_recovery.yaml`.
It replays the frozen ascent, checks the separation fork and mass bookkeeping,
and flies the pinned recovery plan with gfold 0.3.2. Parking propagation is
explicitly `not_checked` and non-applicable to its full-mission gate; the
separate M-07 regression covers that leg. The shipped CI environments separate
landing and TudatPy, rather than proving these packages can never coexist.

`tests/data/mission/reference/e2-04-frozen-mission/` contains the reference,
tolerances and `PROVENANCE.json` (input/file hashes and recording toolchain).
The landing CI job checks provenance, reference-update tags, rebuilt inputs
and the numerical rerun:

```bash
python scripts/e204_reference.py verify
python scripts/e204_reference.py run --out /tmp/e204
python scripts/e204_reference.py compare /tmp/e204
```

The reference runner scopes a four-row `earth_orientation.json` snapshot to
its frame conversions and restores the caller's table afterwards. This avoids
Astropy 6/8 selecting different IERS tables offline. Nested numeric event
arrays use declared tolerances; booleans, keys and check outcomes remain exact.

| Quantity | Absolute comparison budget |
|---|---|
| Rebuilt ascent position / velocity | `1e-6 m` / `1e-6 m/s` |
| Flown position / velocity | `0.01 m` / `1e-4 m/s` (unchanged) |
| Landing-start planned fuel / duration | `0.001 kg` / `0.01 s` |

The planning budgets apply only to advisory `landing_start` fields: 1 g is
50 times tighter than the solver's 0.05 kg stopping budget, and 10 ms is a
tenth of the closed-loop integrator step. See the [acceptance results](acceptance-e1-e4.md).
The portability fix preserves every recorded flight output. Its metadata-only
update used `python scripts/e204_reference.py update --tolerances-only`, which
verifies existing hashes before adding the frame snapshot and comparison
policy, while retaining the original recording toolchain.

For an independently justified reference change, use
`python scripts/e204_reference.py update`, review the resulting provenance and
numeric diff, and commit with `[e204-reference-update]`. CI's `verify` and
`tag-check` guards remain required.

## E5-01: the full mission chain flies a computed ascent (task #268)

E5-01 closes the E2-04 evidence gap for the ascent leg: the packaged M-07
scenario (`src/orbitopt/data/mission_two_stage_leo.yaml`, `ascent.mode:
optimize`) is flown now, and `tests/test_mission_e5_real_ascent.py` grades
WHERE the chain's evidence came from -- station times equal the engine's
event log (including solver-determined `max_q`/`impact` times the chain never
asks for), the parking handoff equals the computed track sample at the
engine's `seco` event, staging states equal flown track samples, and the raw
OpenTsiolkovsky outputs sit beside the run directory. Run it like the M-07
step (same `flightlab-gateb` job, same stack):

```bash
python scripts/fetch_opentsiolkovsky.py
pytest tests/test_mission_e5_real_ascent.py -m slow -v
```

Without tudatpy the M-05 propagation leg is recorded `not_checked` (never
passed), exactly as the E2-04 scenario declares it; the fast guards run
anywhere and the recorded-flight contract check needs no compute stack.

The E5-01 exception, kept deliberately: the E2-04 frozen mission retains its
synthetic two-sample ascent with preset separation/SECO constants as a FAST
integration test. The exception is annotated in the scenario, the builder,
its test module and `PROVENANCE.json` (`opentsiolkovsky: not used ...`), and
`tests/test_mission_e5_real_ascent.py` trips if any other consumer appears or
if the scenario surface grows a knob for injecting an ascent/SECO state.

## E5-05: the reproducible full-physical reference mission (task #272)

**Recorded:** commit `64f64a6`, from code `74f30c8`, seals the computed-ascent
RTLS mission under `tests/data/mission/reference/e5-05-reference-mission/`.
The vehicle decision is resolved: `synthetic_two_stage_rtls` preserves
stage-1 propellant and restarts its upper stage. The committed trajectory
reaches **265.5 × 278.1 km at 51.0°** from Kourou. It supersedes the blocked
400 km/CMA-ES proposal; recording now flies a fixed OT program, not a new
pygmo search. See the [vehicle decision](adr/flightlab-0008-rtls-capable-vehicle-redesign.md).

| Reference artifact | Meaning |
|---|---|
| `mission-optimized/` | Historical directory name for the recording: computed OT ascent, raw backend output, target, recovery and joined report. It is not evidence that this recording ran an optimizer. |
| `mission/` | Replay baseline: the recorded ascent plus freshly simulated recovery. |
| `earth_orientation.json` | Four IERS-A rows from the recording data package, scoped during frame conversion. |
| `PROVENANCE.json` | 30 file hashes, recording code/toolchain, scenario/vehicle digests and canonical separation handoff identity. |
| `tolerances.json` | Comparison budgets; changing them is an audited reference update. |

The recovery uses the E5-05c fixed recipe (`search: keep`): IIP boostback
cutoff, throttled speed-cutoff entry, closed-loop landing forecast, 0.1 s
final integration and a 3.6 g landing-law cap. Touchdown is graded against
its 10 t reserve and speed limits. The committed single-environment report
passes applicable phases but its **physical gate remains `not_checked`**:
parking propagation was not run. No missing propagation is counted as a pass.

### Replaying it

Use the landing stack with gfold 0.3.2; no OT executable or pygmo is needed
for this replay:

```bash
python scripts/e505_reference.py verify
python scripts/e505_reference.py run --out runs/e505
python scripts/e505_reference.py compare runs/e505
```

`verify` checks the committed hashes. `run` consumes the recorded ascent by
digest and re-flies recovery from its canonical separation state; `compare`
grades that output. The [computed mission guide](computed-mission.md) adds
scene export and local viewing.

For an intended recording change, `python scripts/e505_reference.py update`
re-flies the fixed-program ascent with OT v1.0.0 and the pinned recovery
recipe, refuses a red mission and writes fresh provenance. Use
`[e505-reference-update]` in the commit message. `tag-check --base SHA`
only guards this reference directory; the Gate B and E2-04 references have
their own tags. Do not regenerate results to conceal unexplained drift.

### Tolerance provenance

Ascent replay is digest-identical. Recovery bands are `0.01 m` / `1e-4 m/s`
for flown states and `1e-6` for clock/mass bookkeeping. Advisory landing-start
fuel/time diagnostics use `0.001 kg` / `0.01 s`, justified against solver
convergence and integrator resolution. Check identities and verdicts remain
exact. The script and reference tolerance file document each comparison
budget; the [tolerance audit](reference-tolerance-audit.md) covers the wider
reference policy.

### Two environments (cross-env job)

`crossenv_e505_plan.yaml` replays the computed ascent and **runs parking
propagation** in the locked TudatPy environment. A separate Python 3.12
environment re-flies gfold recovery from the sealed handoff. Both stages
set `options.earth_orientation_snapshot` to the reference snapshot so their
GCRS/ECEF conversions use the recording table, rather than interpreter-specific
IERS defaults; each stage report records the snapshot digest. The
[one-command example](computed-mission.md#add-parking-propagation-across-environments)
supplies both interpreters; then run:

```bash
python scripts/e505_reference.py crossenv-check runs/crossenv-e505
```

The checker verifies both exchange envelopes, handoff identity, joined
physical-gate evidence and recovery numerics against the reference. Since
E5-06 it re-grades the joint full-mission gate itself from the sealed
envelopes: all five physical checks must pass, each pinned to the envelope
digest of the environment that measured it, and the joint job also runs the
negative tests (`tests/test_crossenv_qualification.py`: deleted propagation
evidence, a broken shared separation state and a fabricated landing failure
must each grade `failed` or `not_checked`). The existence of a plan or a green
single-stack recording is insufficient. E2-04 remains the fast synthetic
fixture.
