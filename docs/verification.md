# Verification and benchmarks

Reports record checks, tolerances, force models and numerical settings. A
pass applies to those assumptions; it does not establish flight safety.
Mission Control displays **VERIFIED**, **VERIFY FAILED** or **NOT CHECKED**.
For the reviewed E1–E4 integration and its fixes, see the
[2026-11-15 acceptance report](acceptance-e1-e4.md).

## What a report contains

The `verification-report-1.0` document contains residuals, omitted effects,
integrator settings, provenance and a canonical SHA-256 digest. Reports are
written beside mission scenes and by Flight Lab, `orbitopt launch` and the
reference-scenario chain. Inspect both the overall result and its evidence.

## Three kinds of evidence

The check category describes its purpose:

| Category | Purpose |
|---|---|
| `numerical_consistency` | Convergence or conservation within a model. |
| `model_agreement` | Agreement with another prediction, model or constraint. |
| `historical_validation` | Agreement with published real-world data. |

A category without checks is **not checked**.

## Five evidence types (task #263)

Evidence type identifies the reference used by the check:

| Type | Reference | Scope of a pass |
|---|---|---|
| `analytical` | Closed form of the same/simplified model. | Agreement with that solution. |
| `independent_model` | Separately formulated propagator or model. | Cross-model agreement over the checked phase. |
| `historical_telemetry` | Published flight observations, not model inputs. | Agreement with those observations. |
| `design_input` | Target orbit, aim point or declared envelope. | The solution satisfies its design constraints. |
| `numerical_consistency` | Same model under independent/tighter numerics. | Numerical error or conservation checks pass. |

`status.check_evidence` maps checks to types; `status.evidence` rolls them up.
`status.independent_validation` includes only `independent_model` and
`historical_telemetry`. Its `not_checked` result means no independent evidence
was included, even if the report's overall result is green. Independent-model
agreement still does not establish real-world validity.

### Design-input comparisons and why they must be labelled

Hitting the orbit or corridor the optimizer targeted is `design_input`, not
historical validation. Producers must supply that evidence override rather
than inheriting the `model_agreement` default of `independent_model`.
Missing references or skipped checks must remain visibly unassessed.

### The three reference benchmarks

| Benchmark | Evidence |
|---|---|
| Kepler two-body | `analytical`; energy drift is `numerical_consistency`. |
| J2 secular drift | `analytical`. |
| Starship Flight 15 booster catch | `historical_telemetry` against its published timeline. |

The catch contributes two independent-validation checks; the other five
benchmark checks are analytic or numerical. This is scoped evidence, not a
claim that every mission phase was validated.

## Full mission verification gate (E2-03, task #256)

`inputs.full_mission` grades three phases separately:

| Phase | Checks / applicability |
|---|---|
| `orbit_injection` | Ascent verdict, station order, staging, mass and expected orbit; always applicable. |
| `parking_propagation` | M-05 propagation; out of scope only when `expected.propagation_result: not_checked` is declared (refused when `full_mission` is declared). |
| `recovery_touchdown` | Recovery/handoff/touchdown limits; applicable when `recovery:` is declared (always applicable when `full_mission` is declared). |

Among applicable phases, **failed** takes precedence over **not_checked**,
which takes precedence over **passed**. An applicable phase with no checks
is `not_checked`. On a run that does not declare `scenario.full_mission`, a
declared out-of-scope phase does not block the verdict. The chain's
diagnostic `status` and the pipeline's `full_mission_verdict` are separate
fields; inspect the phase document when a run is only partially successful.

### Verification coverage gate (E8-03, task #283)

A scenario may declare `scenario.full_mission: true`: the run *is* the
complete physical mission, so all three phases above become **required
physical phases** and cannot be taken out of scope by declaration:

- `expected.propagation_result: not_checked` is **refused** — the phase
  stays applicable, records no evidence and reads `not_checked`;
- a missing `recovery:` block likewise leaves `recovery_touchdown`
  `not_checked`;
- an applicable-but-ungraded required phase caps the gate verdict at
  `not_checked`, so the out-of-scope route can no longer produce an
  overall pass for a mission that never propagated to its parking orbit
  or never brought the booster home.

Every gate document carries the coverage of required phases —
`inputs.full_mission.coverage` lists a per-phase status
(`checked`, `failed` counts as checked, `not_checked`,
`declared_out_of_scope`, `not_declared`) and a `summary` like `2/3`, which
the pipeline verify stage records (`full_mission_coverage`) and echoes in
its note — and a
`inputs.full_mission.qualification` block. The report's overall description
may claim **independently verified** only when `qualification.independently_verified`
is `true`: the run declared `full_mission`, the gate verdict is `passed`
with complete `3/3` coverage, and the run's independent-evidence rollup
(`independent_validation`) passed. A partial run (e.g. an expendable
mission) never qualifies, however green its applicable phases — its
coverage summary says exactly how far the run reaches.

The chain preserves `ascent.physics_verified` as independent-model evidence:
`true` passes the check; `false` fails orbit injection and the full-mission
gate. `null` remains explicitly unassessed and contributes no independent
validation. A green design-constraint rollup with a null flag does not prove
independent verification. See the [acceptance report](acceptance-e1-e4.md).

### Full physical mission gate (E5-04, task #271)

That out-of-scope exemption is exactly what a *computed* mission must not
hide behind (E2-04 declared the parking propagation out of scope and still
passed). `inputs.full_mission.physical` re-grades the same chain checks as
**five required physical checks** — no applicability escape hatch:

| Required check | Evidence (the chain's existing checks) |
|---|---|
| `orbit_insertion` | station chain, ascent verdicts, expected-orbit agreement |
| `parking_propagation` | M-05 propagation (a `not_checked` *declaration* is not evidence) |
| `landing` | recovery touchdown/reserve/handoff — an unflown first stage caps at `not_checked` |
| `mass_conservation` | every mass gate wherever it is measured (`mass.*`, boundary `*.mass_conservation`) |
| `epoch_frame_continuity` | boundary continuity gates (`handoff.*`, `parking.*`: epoch, frame identity, coupled shared source) |

Each check carries its own tri-state verdict and its own reason codes (a
missing check reports `<check>_not_evidenced` with a reason read from the
scenario/mission notes). The overall verdict is the same conjunction as the
phase gate — failed beats not_checked beats passed — so **any required check
that is `failed` or merely `not_checked` keeps the physical gate from
passing**; only all-five-passed yields `passed`. The pipeline records it as
`physical_mission_verdict`; the E2-03 phase verdict is reported alongside and
unchanged, so a frozen mission like E2-04 still reads `passed` per phase while
the physical gate honestly says `not_checked`.

### The reproducible reference mission (E5-05, task #272)

E5-04 says what a physical mission must show; E5-05 commits one. The mission
of record (`src/orbitopt/data/mission_two_stage_leo_rtls_reference.yaml`) is
computed end to end — OpenTsiolkovsky ascent, station chain, staging, M-05
propagation evidence, and the RTLS recovery driven from the one canonical
separation state — with full provenance (code SHA, dependency matrix, IERS
snapshot, backend pins, handoff identity). Regression replay and cross-
environment replay (both E5-03 environments, with the propagation evidenced
in the tudatpy leg so the physical gate reads `passed`, not `not_checked`)
compare within `tests/data/mission/reference/e5-05-reference-mission/
tolerances.json`, whose every band carries a source; see the
[regression CI page](flightlab-regression-ci.md#e5-05-the-reproducible-full-physical-reference-mission-task-272).
E2-04 is kept unchanged as the fast integration test. **Recording status:**
the sealing run is blocked on a vehicle decision — RTLS from a computed
OpenTsiolkovsky ascent is physically infeasible for the current catalog
vehicle (stage-1 separates dry; the demonstrator's insertion margin cannot
carry the recovery propellant), so the reference is not recorded and every
consumer of it skips with the reason until a recording lands. See the
regression CI page's E5-05 status note and ContextDB record
`e5-05-rtls-infeasibility-evidence`.

## Benchmarks against public references

```bash
orbitopt benchmarks
orbitopt benchmarks --json
```

The command exits 1 if any benchmark fails. References and assumptions are
in the [engineering notes](engineering-notes.md#benchmark-scenarios-with-public-references--analytic-solutions-task-245).

## Cross-model comparison

Compare a common ascent/coast across the available 3-DOF, JSBSim, TudatPy
and analytic backends:

```bash
python -m orbitopt.verify.cross_model --backends three_dof,six_dof,tudatpy
python -m orbitopt.verify.cross_model --threshold ascent.position_m=1000 -o cross_model.json
```

The `three_dof_vs_two_body_analytic` pair is `analytical`; other backend pairs
are `independent_model`. These harness names are distinct from Flight Lab's
[canonical run backend identities](flightlab.md#backend-identities-and-fidelity-levels).

## Integrator convergence

LEO/GTO coast, re-entry and powered-ascent studies sweep tolerances and,
where supported, step sizes. Their checks are `numerical_consistency`:

```bash
python -m orbitopt.verify.integrator_convergence --plot
```

## Cross-backend acceptance matrix

`src/orbitopt/data/acceptance_matrix.yaml` declares limits per backend pair,
phase and metric, with their justification and failure cause classes.
Results use the standard report/gate machinery. An unavailable backend is
`not_checked` with a cause, not silently passed. Ill-conditioned or explicitly
ungated rows are recorded; `--threshold key=none` disables that row's gate.

```bash
python -m orbitopt.verify.acceptance_matrix
python -m orbitopt.verify.acceptance_matrix --matrix my.yaml \
  --threshold three_dof_vs_two_body_analytic.coast.position_m=0.005
```

A pass on one available pair does not cover the remaining matrix. The current
acceptance ran the real numerical/analytic pair, not every backend pair.

## Model discrepancy budget (E8-02, task #282)

A gated total says *whether*, not *what*. `orbitopt.verify.discrepancy_budget`
reuses the acceptance matrix (same harness run, same resolved tolerances) and
splits every measured pair difference into five source lines — frame/time,
numerical integration, engine, aero, controller — recording the residual
between the component sum and the total. The residual is the unattributed
plant-formulation remainder (rigid-body vs point-mass, FCS/actuator dynamics),
stated, not zero-padded. Lines carry how they were obtained:

- **frame/time** — one shared bridge at one epoch: 0 applied, with the
  IERS 0.9 s desync sensitivity recorded as the budget;
- **numerical integration** — a tolerance-relaxation probe (3-DOF reference
  at rtol/atol ×1000); the fixed-step side is the published E6-04 bound;
- **engine** — the live E6-01 parity residual applied as a thrust-scale
  probe (a failed alignment is refused, not budgeted); the mass channel
  carries the disclosed `mdot·dt` one-frame bookkeeping bound;
- **aero** — 0 by declaration (`parity-jsbsim-aero-zero`), with the
  production-airframe *potential* recorded, not summed;
- **controller** — a loop-stiffening probe share (2× PID gains).

Pairs that did not fly yield `not_measured` lines with the matrix's own
cause code — never a fake zero. `--no-probes` records the structure with
the full totals in the residual. Only matrix rows vote: the report status
mirrors the matrix verdict.

```bash
python -m orbitopt.verify.discrepancy_budget            # all installed backends
python -m orbitopt.verify.discrepancy_budget --no-probes --json
```

## Model coverage and declared conventions

Mission results record each phase's model and limitations.
`orbitopt.physics_io` checks frame, epoch, timescale and unit conventions at
physics boundaries. Boundary continuity establishes consistent handoffs; it
does not prove a sparse recorded ascent was dynamically propagated.

## Further reading

[Flight Lab](flightlab.md) · [Regression CI](flightlab-regression-ci.md) ·
[Examples and validation](examples.md) · [Acceptance findings](acceptance-e1-e4.md)
