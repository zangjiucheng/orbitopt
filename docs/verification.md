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
| `parking_propagation` | M-05 propagation; out of scope only when `expected.propagation_result: not_checked` is declared. |
| `recovery_touchdown` | Recovery/handoff/touchdown limits; applicable when `recovery:` is declared. |

Among applicable phases, **failed** takes precedence over **not_checked**,
which takes precedence over **passed**. An applicable phase with no checks
is `not_checked`. A declared out-of-scope phase does not block the verdict.
The chain's diagnostic `status` and the pipeline's `full_mission_verdict`
are separate fields; inspect the phase document when a run is only partially
successful.

The chain preserves `ascent.physics_verified` as independent-model evidence:
`true` passes the check; `false` fails orbit injection and the full-mission
gate. `null` remains explicitly unassessed and contributes no independent
validation. A green design-constraint rollup with a null flag does not prove
independent verification. See the [acceptance report](acceptance-e1-e4.md).

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

## Model coverage and declared conventions

Mission results record each phase's model and limitations.
`orbitopt.physics_io` checks frame, epoch, timescale and unit conventions at
physics boundaries. Boundary continuity establishes consistent handoffs; it
does not prove a sparse recorded ascent was dynamically propagated.

## Further reading

[Flight Lab](flightlab.md) · [Regression CI](flightlab-regression-ci.md) ·
[Examples and validation](examples.md) · [Acceptance findings](acceptance-e1-e4.md)
