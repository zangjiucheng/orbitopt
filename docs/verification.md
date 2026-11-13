# Verification and benchmarks

orbitopt never reports a bare "passed". Every check is written into a
machine-readable verification report (schema `verification-report-1.0`)
together with the model it was computed under, its tolerance and the kind of
evidence it is. A pass is conditional on that model and those tolerances; it
is not a statement of real-world flight safety.

## What a report contains

- **Residuals** of the numerical solution against a reference, each with its
  tolerance block and whether it is within tolerance.
- **Dynamics**: which gravity model each body exerts (point mass, or
  spherical harmonics with J2 / J22 named), which third bodies are included,
  and what is *not* modelled (solar radiation pressure, drag, relativity,
  finite burns).
- **Numerical settings**: integrator, tolerances, step limits.
- **Reproducibility**: no wall-clock time, sorted keys, full-precision floats
  and a SHA-256 `digest` of the canonical document, so the same run gives the
  same report.

Reports are produced by `orbitopt run` (the `*.verification.json` beside the
scene), by Flight Lab runs, by `orbitopt launch` (`launch.verification.json`)
and by the reference-scenario chain (`mission.verification.json`). Mission
Control's **Verification** tab shows them; its badge reads **VERIFIED**,
**VERIFY FAILED** or **NOT CHECKED**.

## Three kinds of evidence

Every check carries one category, and the report rolls the outcome up per
category, so a reader sees which kind of evidence backs a pass, and which
kind is missing:

| Category | Meaning | Examples |
|---|---|---|
| `numerical_consistency` | The same model re-run by independent numerics | Step-size convergence, conservation of quantities the model conserves exactly, solver convergence |
| `model_agreement` | The result against a *different* model or prediction | A reference propagator, the screening orbit, an aim point, a vehicle, site or corridor limit |
| `historical_validation` | The result against published real-world data | A flown mission's timeline |

A category with no check is **not checked**, never counted as a pass: absent
evidence is not evidence.

## Benchmarks against public references

`orbitopt benchmarks` runs three scenarios, each with a frozen expected
value (a published number or a closed-form solution) and a declared
tolerance. The measured value is always the physics engine's own answer,
never the reference. The command exits with status 1 if any benchmark fails,
and the fast CI suite runs all three.

| Benchmark | Reference |
|---|---|
| Kepler two-body | The analytic two-body solution after 7,500 s against a numerically propagated orbit, plus the numerical run's energy drift (Vallado, *Fundamentals of Astrodynamics*, ch. 2) |
| J2 secular drift | Node drift of a 700 km sun-synchronous orbit and the frozen perigee at the critical inclination (Vallado, 4th ed., §9.7) |
| Starship Flight 15 booster catch | Catch time and offset from the boostback, ballistic fall and landing-burn physics against the published timeline (about T+6:35, caught over the tower) |

```bash
orbitopt benchmarks          # human-readable results, exit 1 on any failure
orbitopt benchmarks --json   # the verification-report-1.0 document
```

## Cross-model comparison

`orbitopt.verify.cross_model` flies one scenario (a single-stage ascent to
MECO, then a coast) on every physics backend available in the environment:
the 3-DOF point mass, the JSBSim 6-DOF stage, TudatPy and an analytic two-body
coast. It resamples the trajectories onto a common time grid and reports
pairwise differences in position, velocity, mass and orbital elements, each
against a threshold for its phase (ascent or coast). Override a threshold with
`--threshold phase.metric=value`; the value `none` records a metric without
checking it.

```bash
python -m orbitopt.verify.cross_model --backends three_dof,six_dof,tudatpy
python -m orbitopt.verify.cross_model --threshold ascent.position_m=1000 -o cross_model.json
```

## Integrator convergence

`orbitopt.verify.integrator_convergence` asks whether the numerical error is
under control in the four integrations the published numbers rest on: a LEO
coast with J2, one revolution of a GTO coast, a lifting re-entry through peak
deceleration, and the powered ascent to MECO. For each it sweeps the
integrator tolerance (and, where a fixed-step path exists, the step size),
fits the convergence curve and order, bounds the error at the production
setting, and says whether that setting is justified, conservative or not
justified.

```bash
python -m orbitopt.verify.integrator_convergence --plot
```

## Model coverage and declared conventions

Every mission run lists, per phase, the physical model it was computed with,
its assumptions and limitations, and the effects it left unmodelled; the
pipeline summary prints the union (`orbitopt.model_fidelity`). Physics entry
points declare their frame, timescale, epoch format and units, and
`orbitopt.physics_io` rejects mismatches at the boundary: a kilometre handed
to a metre interface, a UTC label handed to an ephemeris-time interface, or an
Earth-fixed track handed to an inertial one fails loudly instead of producing
a plausible wrong trajectory.

## Further reading

- [Engineering notes](engineering-notes.md): the verification layer, the
  pipeline and its Verification Gate.
- [Flight Lab](flightlab.md): Gate B, the handoff gates and the recovery
  Monte Carlo statistics.
- [Flight Lab regression CI](flightlab-regression-ci.md): the pinned
  toolchain and the Gate B reference flight.
- [Examples and validation](examples.md): historical launch-window
  comparisons.
