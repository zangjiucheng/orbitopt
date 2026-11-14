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

| Category | What it checks |
|---|---|
| `numerical_consistency` | Numerical convergence or conservation within a model. |
| `model_agreement` | Agreement with another model, prediction or constraint. |
| `historical_validation` | Agreement with published real-world data. |

A category without checks is **not checked**, not passed.

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

```bash
python -m orbitopt.verify.cross_model --backends three_dof,six_dof,tudatpy
python -m orbitopt.verify.cross_model --threshold ascent.position_m=1000 -o cross_model.json
```

## Integrator convergence

The convergence study sweeps tolerances and, where applicable, step sizes for
LEO and GTO coasts, re-entry and powered ascent. It estimates numerical error
and assesses the production setting for each scenario.

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
