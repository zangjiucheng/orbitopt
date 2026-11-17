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

E5-01 proved the chain *can* fly a computed ascent; E5-05 freezes the result
as the mission of record: a two-stage LEO mission with a booster RTLS,
recorded end to end from a **computed** ascent -- no preset states anywhere.
`src/orbitopt/data/mission_two_stage_leo_rtls_reference.yaml` (Kourou,
2026-11-10, 400 km / 51.6° parking, RTLS recovery) is optimized-and-flown
once and sealed under
`tests/data/mission/reference/e5-05-reference-mission/`:

> **Recording status (task #272):** the recording is blocked on a vehicle
> decision, and every consumer above (script subcommands, the
> `tests/test_e505_reference.py` guards, the CI steps) detects the missing
> reference directory, prints the reason and passes -- so CI stays green and
> everything arms itself the moment a recording lands. The blocker is
> physical, not software: the pinned OpenTsiolkovsky binary burns stage-1 to
> depletion on every run (debris separates at exactly its 30 t dry mass --
> verified on the recorded Gate-B run too), the recovery chain plans from
> that actual mass, and RTLS from a real MECO (~150 km downrange, ~3.5 km/s
> suborbital) would need tens of tons of stage-1 reserve the demonstrator
> vehicle cannot leave without failing its own insertion (upper-stage slack
> buys ~400 m/s; `downrange` landing needs the same propellant). FL-305's
> accepted landing was flown off the textbook separation state, whose
> provenance is `illustrative` -- which is why E2-04 stays a fast integration
> test rather than computed-landing evidence. A recording needs either an
> RTLS-capable vehicle/ascent redesign or a revised acceptance (ContextDB
> record `e5-05-rtls-infeasibility-evidence`).

```
tests/data/mission/reference/e5-05-reference-mission/
├── mission-optimized/   the recording: full optimize run (computed ascent,
│                        raw OpenTsiolkovsky outputs, RTLS recovery, joined
│                        mission.verification.json)
├── mission/             the replay regression baseline (run_dir mode of the
│                        recorded ascent + re-flown recovery)
├── earth_orientation.json  the four-row IERS-A snapshot the run was scoped to
├── tolerances.json      the declared comparison budgets (below)
└── PROVENANCE.json      file digests, git SHA, full toolchain matrix,
                         EOP snapshot id, separation-handoff identity
```

`PROVENANCE.json` records the code version (`git_sha_recording`), the
dependency matrix (`toolchain`: python, platform, gfold, pygmo, numpy,
scipy, astropy, astropy-iers-data, opentsiolkovsky), the IERS-A snapshot
(the same table the reference epoch falls inside of -- finalized values, no
predictions), and the **separation-handoff identity**: the recording's
recovery leg is *driven* from the ascent's canonical coupled-branch separation
state (`drive_from_separation: true`), and its `handoff_id` is recorded so
every later replay -- including across the environment boundary -- can prove
it branched from that one state.

Unlike the E2-04 builder, nothing here is constructed from constants: the
ascent is the M-07 recipe (CMA-ES 40x40, seed 11, `refly=True`) optimizing the
pinned target with the real OpenTsiolkovsky v1.0.0 binary, and the recovery
is the FL-304/FL-305 plan flown by gfold 0.3.2 from the computed separation
state. The recovery warm start (`initial_guess`, `max_fevals: 18`) is the
pinned FL-305 acceptance recipe for this launch geometry, recorded in the
scenario, the script's `frozen_inputs` and the provenance.

### Replaying it

`src/orbitopt/data/mission_two_stage_leo_rtls_replay.yaml` consumes the
recorded ascent in `run_dir` mode (the same digests, re-graded) and re-flies
the recovery from the canonical handoff. The regression pair mirrors E2-04
and FL-601:

```bash
python scripts/e505_reference.py verify            # digests vs PROVENANCE.json (fast, every job)
python scripts/e505_reference.py run --out /tmp/e505   # replay (needs the landing stack)
python scripts/e505_reference.py compare /tmp/e505     # vs reference within tolerances.json
python scripts/e505_reference.py tag-check --base "$BASE"   # G5: reference diffs need the tag
python scripts/e505_reference.py update            # intended reference change (re-record)
```

`update` refuses a red run, regenerates the EOP snapshot from the pinned
`astropy-iers-data` table, and records the *current* toolchain; the
`[e505-reference-update]` commit tag is required (same G5 rule as FL-601 and
E2-04). The slow replay is the `flightlab-landing` job; the fast guards run
everywhere (`tests/test_e505_reference.py`).

### Tolerance provenance

The replay consumes the recorded ascent by digest, so no tolerance applies to
the ascent itself -- only to the re-flown recovery numerics and the report
fields. Bands: `1e-6` (clock/mass bookkeeping), `1e-2 m` / `1e-4 m/s` for
flown states (unchanged from E2-04 -- cross-build drift of the same pinned
gfold/OT stack, observed 1.4e-7 m at most on reviewed builds),
`1e-4` on check values and report values, and advisory SOCP-plan fields at
50x inside the solver's own stopping budget. Every band's source is restated
in `scripts/e505_reference.py`'s docstring and in `tolerances.json`
provenance; physical check verdicts and check identities are compared
exactly, never within a band.

### Two environments (cross-env job)

`src/orbitopt/data/crossenv_e505_plan.yaml` flies the same reference mission
across the tudatpy-3.10 / gfold-3.12 boundary: the ascent leg replays the
recorded ascent and *evidences* the M-05 propagation in the tudatpy
environment (`options.propagate: true` -- the propagation the single-
interpreter stack cannot run), seals the canonical separation handoff
(`options.coupled_separation: true`), and the recovery leg is driven from the
reconstructed handoff across the boundary (`options.drive_from_separation:
true`). `python scripts/e505_reference.py crossenv-check <out>` then grades
the joined run: both envelopes re-verify by digest, the recovery leg's
reported handoff id equals the recorded one, and the sealed recovery run
matches the reference within `tolerances.json`. This is the E5-05 acceptance
that the reference is replayable in the E5-03 environments, and the E5-04
physical gate's one missing piece -- propagation -- is evidenced there, so a
cross-env run reads `physical: passed` on all five required checks.

E2-04 remains in place as the fast dual-branch integration test: E5-05
supplements it, and `tests/test_e505_reference.py` guards that its scenario,
script, test and frozen reference files stay present.
