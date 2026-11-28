# ADR rendezvous-0006: independent per-sample robustness re-flight (E12-10)

- **Status:** Accepted
- **Date:** 2026-11-26
- **Deciders:** E12C Rendezvous Qualification Closure (Epic #408), E12-10 (#412)
- **Code:** `src/orbitopt/rendezvous/robustness.py` (`passive_contingency_in_success`,
  `select_reflight_samples`, `build_reflight_spec`), `src/orbitopt/verify/rendezvous_refly.py`
  (`run_sample_reflight_in_current_environment`, `sample_reflight_comparison`,
  `verify_sample_reflight`), `src/orbitopt/verify/tudat_propagate.py` (drag on
  `ReusableCoastPropagator`), `src/orbitopt/crossenv/stages.py` (`rendezvous_refly_samples`)
- **Evidence:** `docs/benchmarks/e12_10_independent_reflight.json`,
  `benchmarks/bench_e12_10_independent_reflight.py`, `tests/test_e12_10_independent_reflight.py`
- **Builds on:** ADR rendezvous-0002 (TudatPy re-flight, ECLIPJ2000 rotation, crossenv
  exchange), ADR rendezvous-0003 (the E12-07 campaign, amended by this task),
  E12-08 (canonical sealed plan)

## Context and problem

E12-07 measured LEO-RDV-001 under dispersion: 14 of 264 hold-out samples passed.
Every number came from orbitopt grading itself. The TudatPy line was
`NOT_CHECKED`, and its one TudatPy entry point flew the nominal E12-05 plan, not
any perturbed sample. The success test also left out the passive contingency
releases that the nominal verifier checks. A sample could pass while a coast
after a missed gate drifted through the keep-out sphere.

## Decision

1. **Passive contingencies are part of success under E12-10 grading.** The
   campaign flag `passive_contingency_in_success` flies the three E12-04 releases
   (no burn at t0, gate abort at t1, coast from the Hold Point) on each sample's
   own perturbed physics. A release that enters the keep-out sphere within
   `passive_contingency_s` is a failed check named `passive_<kind>`, classified
   `safety_violation`. The packaged campaign keeps the flag off, so the E12-07
   artifact still reproduces. The E12-10 benchmark turns it on.
2. **The re-flight selection is a declared rule, not a hand pick.** Every
   hold-out success (cap 16, above the 13 the reference campaign yields), up to 8
   passive-contingency failures with passive-only failures first, the boundary
   sample of each failure class (closest to its threshold), and one
   `planning_infeasible` row recorded as skipped. Total cap 32, ties broken on
   `(cluster, index)`. The rule, the counts and whether every success was
   re-flown go into the artifact.
3. **Each sample is re-flown with its own perturbation.** The spec carries the
   sample's re-planned gate epoch, its executed burns (with its execution error),
   its truth J2 and its drag factor. The spec is digested. The worker refuses a
   spec whose mission, config sha or digest does not match. The TudatPy verdict
   comes from `grade_trajectory_checks`, the same grader the main model uses,
   applied to the independently flown trajectory.
4. **The bands are absolute and come from measurement.** Both models fly the
   same per-sample physics on the same 5 s grid, so a residual measures the
   implementation only. On the 23 re-flown samples the largest residuals were
   4.5e-6 m in arrival position, 5.8e-9 m/s in arrival speed, 9.2e-7 m in the
   grid minimum of the range, and 2.3e-6 m over all 69 passive keep-out margins. The bands
   are 0.01 m, 1e-5 m/s, 0.01 m and 0.01 m. That is at least 1000 times the
   measured residual and at least 1000 times below the scales the contract
   grades (50 m, 0.2 m/s, a 500 m keep-out radius).
5. **Wording.** The E12-07 conclusion is a **Robustness Characterization**.
   The E12-10 artifact may say **Independently Verified Robustness
   Characterization** only when the re-flight is `checked`, every hold-out
   success was re-flown, there are no verdict flips and there are no band
   violations. The benchmark derives the label from those four facts. Its
   `outcome` field says what was found. The label certifies the *evidence*, not
   the design: the first wording, "Robust Safety Qualification", read as a
   result for a design that succeeds on 4.9 % of the hold-out samples, so it was
   dropped. No success-rate requirement is declared, so the artifact records the
   rate and does not claim the design is robust.

## Two harness defects the first runs exposed

The first full runs reported 6 verdict flips (5 of 8 successes failing the
hold-point coast in TudatPy), 10 m arrival residuals and about 150 m of
min-range disagreement. A drag-free, J2-free re-flight of one sample still
disagreed by 156 m. Two point-mass integrations of the same pulses should agree
to millimetres, so these residuals were harness bugs, not physics:

- **Fixed-step RK4 legs overshot their end epoch.** tudat's RK4 stops at the
  first grid point past the requested end. A 560.64 s gate on a 5 s grid ended
  at 565 s, and burn 1, the transfer leg and the hold-point coast all started
  from that state. The sample coast helper now drops samples past the end and
  takes one partial step onto it, which matches the main model's inclusive
  `_grid`.
- **tudat measured drag altitude from the mean-radius sphere.** tudat's default
  Earth shape is a 6371.0 km sphere. The main model measures altitude from
  `EARTH_RE` (6378.137 km). The 7.1 km offset is a 13 % density difference at
  H = 50 km, about 11 m at arrival. The drag model now carries `body_radius_m`
  and pins the central body's shape to it.

With both fixed, every Keplerian, J2 and J2+drag variant agrees to the
micrometre, and the full selection reports 0 flips.

## Results (reference campaign, hold-out arm, E12-10 grading)

| | E12-07 baseline | E12-10 re-graded |
|---|---|---|
| Successes | 14 / 264 (5.3 %) | 13 / 264 (4.9 %) |
| Wilson 95 % | 3.2 – 8.7 % | 2.9 – 8.2 % |
| Cluster bootstrap 95 % | 2.7 – 8.0 % | 2.7 – 7.6 % |
| `safety_violation` | 69 | 87 |
| `arrival_accuracy` | 181 | 164 |

Including the passive checks moves 18 samples into `safety_violation`: 1
former success and 17 former arrival failures. Safety takes precedence over
arrival in the taxonomy. Every one of those passive failures is the hold-point
coast. 74 safety violations involve it, and 18 fail on it alone. The
independent re-flight covers 23 samples (13 successes, 8 passive failures,
2 boundary samples) with 0 verdict flips and 0 band violations.

## Consequences

- The robustness result now has an independent cross-check, but the design
  it describes is not robust. 95 % of dispersed samples fail, and the
  hold-point coast is the dominant safety mode once it is graded.
- The selection is a sample of the failures, not all of them. A failure that
  was not re-flown still has a main-model verdict only, and the artifact names
  the samples that were re-flown.
- Bands at the micrometre scale will trip on a tudat upgrade that changes an
  integrator or field convention. That is the point: such a change must be
  re-measured, not absorbed.
