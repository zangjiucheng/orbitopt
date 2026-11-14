# Verification and benchmarks

A verification report records checks, tolerances, numerical settings and the
physical model. A pass applies only to those assumptions; it does not establish
real-world flight safety. Mission Control displays **VERIFIED**, **VERIFY
FAILED** or **NOT CHECKED**.

## What a report contains

Reports use the `verification-report-1.0` schema and include residuals, force
models and omitted effects, integrator settings and a canonical SHA-256
digest. They are written beside mission scenes and by Flight Lab,
`orbitopt launch` and the reference-scenario chain.

## Three kinds of evidence

Every check is first classified by **what it is testing** (task #243):

| Category | What it checks |
|---|---|
| `numerical_consistency` | Numerical convergence or conservation within a model. |
| `model_agreement` | Agreement with another model, prediction or constraint. |
| `historical_validation` | Agreement with published real-world data. |

A category without checks is **not checked**, not passed.

## Five evidence types (task #263)

The category tells you *what shape* the check has; the **evidence type**
(task #263) tells you what the *reference on the other side of the tolerance*
is actually made of, and therefore how far a pass there reaches. Two checks
can share a category and still mean very different things: a `model_agreement`
residual that compares against the closed-form solution of the same two-body
problem is not evidence of the model's contact with reality, and a
`model_agreement` residual that compares against the *design aim point* is not
a comparison against anything independent. Every check in a report therefore
carries both a category and one of:

| Evidence type | Independent? | Reference | A pass proves | A pass cannot prove |
|---|---|---|---|---|
| `analytical` | no | a closed form of the same (or a simplified) model | the numerics agree with the closed form over the covered phase | the model describes reality |
| `independent_model` | **yes** | a model formulated separately (JSBSim, tudatpy, a hand-written propagator) | the model agrees with another independent formulation over the covered phase | real-world validity — no independent model is the truth |
| `historical_telemetry` | **yes** | published data from a real flight (never an input to the model) | the model matches an actual observed flight over the covered phase | other mission phases, other vehicles, or the vehicle's full envelope |
| `design_input` | no | the aim point / target / envelope the solution was built against | the solver hit its aim or respected its envelope — internal consistency | **anything about fidelity to reality**; a design-input match is *not* validation and must never be read as historical proof |
| `numerical_consistency` | no | the same model re-run by independent numerics | the solver converged (energy, angular momentum, integration error) | that the model is the right one to have converged |

A report's `status.evidence` field rolls up every type with its qualification,
`status.check_evidence` maps every check to its type, and
`status.independent_validation` rolls up the **validation-grade** types only
(`historical_telemetry` + `independent_model`). A green report whose
`independent_validation.result` is `not_checked` was never checked against
anything outside the model — its `passed` verdict is internal consistency
plus matching design input, nothing more. **Read `independent_validation`
before you trust `passed`.**

### Design-input comparisons and why they must be labelled

A targeting loop drives a trajectory to an aim point; the residual it converges
is between the propagated state and the aim. That residual is a *fact* — the
solver did converge — but comparing the answer to the target the solver used
to produce the answer cannot confirm the model against reality. `design_input`
labels these checks so a report's headline status cannot silently upgrade
"the solver converged" to "the mission is validated". The same applies to
vehicle-limit envelopes, site DLA limits, launch-azimuth corridors, expected
orbital-element bands and safe-entry corridors: those are constraints the
plan is *built to satisfy*, not external observations.

Where a mission verifier's `model_agreement` check is really one of these, its
producer passes `evidence=...` on the residual / status call to override the
default; where the reference is a genuine independent model or published
flown data (the Gate-B re-fly, cross-backend comparisons, the Starship F15
timeline benchmark), the default is already the correct reading and no
override is needed.

### The three reference benchmarks

The three public benchmarks in `orbitopt benchmarks` are individually
classified in `BENCHMARK_EVIDENCE`:

| Benchmark | Evidence | Reference |
|---|---|---|
| `kepler_two_body` | `analytical` (energy check: `numerical_consistency`) | a closed-form Kepler solution of the same two-body model |
| `j2_secular_drift` | `analytical` | first-order averaged secular-J2 formulas + the sun-synchronous definition |
| `starship_booster_catch` | `historical_telemetry` | SpaceX's published Flight 15 timeline, never an input to the solve |

Only the third carries validation-grade evidence. The benchmark suite's
`independent_validation` rollup is therefore 2/2 (the F15 catch has two
checks); the other five checks are analytic or numerical and reach only as
far as their closed-form or their integration error.

## Benchmarks against public references

`orbitopt benchmarks` checks a Kepler two-body orbit, J2 secular drift and the
Starship Flight 15 booster catch against analytic or published references. It
returns exit status 1 if any check fails; `--json` outputs a verification
report. See [engineering notes](engineering-notes.md#benchmark-scenarios-with-public-references--analytic-solutions-task-245)
for the references.

```bash
orbitopt benchmarks
orbitopt benchmarks --json
```

## Cross-model comparison

Compare a single-stage ascent and coast across available 3-DOF, JSBSim,
TudatPy and analytic backends. Results are resampled onto a common time grid;
thresholds can be overridden by phase and metric. The agreed acceptance line
for these pairs is the [acceptance matrix](#cross-backend-acceptance-matrix).
Every backend-pair check carries `evidence=independent_model` **except** the
`three_dof_vs_two_body_analytic` pair, which is graded against a closed-form
two-body reference and is labelled `analytical` — a pass there is agreement
with an analytic coast, not validation of the 6-DOF sim.

```bash
python -m orbitopt.verify.cross_model --backends three_dof,six_dof,tudatpy
python -m orbitopt.verify.cross_model --threshold ascent.position_m=1000 -o cross_model.json
```

## Integrator convergence

The convergence study sweeps tolerances and, where applicable, step sizes for
LEO and GTO coasts, re-entry and powered ascent. It estimates numerical error
and assesses the production setting for each scenario. Every check is
`numerical_consistency`: the same model, run tighter.

```bash
python -m orbitopt.verify.integrator_convergence --plot
```

## Cross-backend acceptance matrix

The acceptance matrix grades the cross-model comparison above against a
configurable line per **backend pair × phase × metric**: which differences
are accepted, what evidence justifies each tolerance (a convergence budget,
a measured deviation, a physical argument, or ill-conditioning — recorded
but never gated), and which cause class a violation carries
(`numerical_integration`, `model_formulation`, `state_handoff`, …). The
default lives in `src/orbitopt/data/acceptance_matrix.yaml` (schema
`acceptance-matrix-1.0`); edit or replace it — the numbers are the
acceptance line, not code constants.

The result is an `acceptance-matrix` verification report, so it flows
through the same pipeline gate as every other check. A row whose backend is
not installed is `not_checked` with its cause (`backend_environment`, …),
never a silent pass; `--threshold key=none` records a row without gating.

```bash
python -m orbitopt.verify.acceptance_matrix                      # default matrix
python -m orbitopt.verify.acceptance_matrix --matrix my.yaml \
    --threshold three_dof_vs_two_body_analytic.coast.position_m=0.005
```

## Model coverage and declared conventions

Mission results record each phase's model and limitations. `orbitopt.physics_io`
checks declared frame, time and unit conventions at physics boundaries.

## Further reading

[Flight Lab](flightlab.md) · [Regression CI](flightlab-regression-ci.md) ·
[Examples and validation](examples.md) · [Engineering notes](engineering-notes.md)
