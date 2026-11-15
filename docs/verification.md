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

## Full mission verification gate (E2-03, task #256)

The two-stage LEO mission report carries, under `inputs.full_mission`, a gate
that grades the mission as three independent phases — `orbit_injection`
(station chain, ascent run verdict, staging and parking-orbit handoffs, mass
bookkeeping, expected-orbit agreement), `parking_propagation` (the M-05
propagation link) and `recovery_touchdown` (the FL-302/303/304 booster
return) — each with its own tri-state verdict and failure reason codes,
followed by one overall verdict. `orbitopt.mission.phases` owns the rule; the
chain's own `status` (which link broke) is unchanged beside it.

The overall verdict is the conjunction of the **applicable** phases with the
unified evidence semantics above (absent evidence is never a pass):

| orbit_injection | parking_propagation | recovery_touchdown | full mission |
|---|---|---|---|
| any `failed` | — | — | **failed** |
| otherwise, any applicable `not_checked` | | | **not_checked** |
| `passed` | `passed` | `passed` or not applicable | **passed** |

A phase is applicable when the mission flies it: orbit injection always,
propagation unless the scenario declares `expected.propagation_result:
not_checked`, recovery when a `recovery:` stage is declared. A phase the
mission declares but that produced no evidence (the ascent never flew, the
caller skipped propagation) is `not_checked` and caps the overall verdict —
it never silently passes. This is what makes a partial success readable:
injection green with a lost booster is overall **failed** with a green
`orbit_injection` phase, and an expendable mission's `recovery_touchdown` is
not applicable, not failed. The `orbitopt run` pipeline gate records the
verdict as `full_mission_verdict` on its verify stage.

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
thresholds can be overridden by phase and metric.

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

## Model coverage and declared conventions

Mission results record each phase's model and limitations. `orbitopt.physics_io`
checks declared frame, time and unit conventions at physics boundaries.

## Further reading

[Flight Lab](flightlab.md) · [Regression CI](flightlab-regression-ci.md) ·
[Examples and validation](examples.md) · [Engineering notes](engineering-notes.md)
