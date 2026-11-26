# ADR rendezvous-0005: a dispersion campaign, not a safety factor (E12-07)

- **Status:** Accepted
- **Date:** 2026-11-25
- **Deciders:** E12 Orbital Rendezvous & Mission Operations (Epic #389), E12-07 (#397)
- **Code:** `src/orbitopt/rendezvous/robustness.py`, `src/orbitopt/data/leo_rdv_001_robustness.yaml`,
  `src/orbitopt/verify/tolerances.py`
- **Evidence:** `docs/benchmarks/e12_07_rendezvous_robustness.json`,
  `tests/test_e12_07_rendezvous_robustness.py`, `benchmarks/bench_e12_07_rendezvous_robustness.py`
- **Builds on:** ADR rendezvous-0001 (LEO-RDV-001, `plan_rendezvous`), ADR rendezvous-0002
  (whole-trajectory safety constraints), ADR rendezvous-0003
  (the TudatPy re-flight whose backend this campaign
  records as missing), ADR relative-0001 (`propagate_pair`, `_accel`), E3-03/E7-05
  (sample-size planning, hold-out discipline), E10-03/R-06 (clustered intervals, source sensitivity)

## Context and problem

E12-03 planned LEO-RDV-001, E12-04 proved the plan satisfies its constraints *on
nominal*, E12-05 confirmed the nominal trajectory in an independent propagator.
None of that prices the mission against the things that are not nominal: a
navigation solution that is a few metres off, a burn a degree off its commanded
direction, a gravity field that is not exactly the declared J2, an atmosphere
the plan does not model at all. The task asks for a hold-out success rate with
credible intervals, a failure taxonomy, a sensitivity reading, and an
independent re-check of the results that matter. Four decisions were forced by
that wording.

The campaign must not be able to tune itself into its own answer. A single
sample pool that people look at while choosing disturbance levels reports a
rate that is partly a measurement of the design and partly a measurement of the
patience of whoever ran it.

The intervals must not lie about independence. Samples drawn under one gravity
field and one space-weather condition are not independent draws of that
environment: a bad field is bad for every one of them, and a binomial interval
over 264 correlated samples reports a precision it does not have.

The truth model must differ from the planning model *only* by the declared
disturbances. A harness with a private re-implementation of the shared physics
measures the difference between two implementations, and reports it as robustness.

And the independent re-check has a backend that does not exist in this
container. The honest state of an unrunnable cross-check is unrecorded-or-said-so,
never a quiet pass.

## Decision

**Train and hold-out arms on disjoint seed streams.** `parse_campaign` refuses a
declaration whose two seed bases are less than `max(n)+1` apart, so a sample
identified by `[seed_base, cluster, index]` cannot appear in both arms. The
hold-out arm is the headline; the train arm is printed beside it and explains
nothing. Both arms fly the *same* ground plan (`plan_rendezvous_safe`'s
constrained optimum) at *its* time-of-flight: what is priced is a locked design
under dispersion, not a per-sample re-optimisation. The re-plan a sample does
get is the guidance the spacecraft would actually run — re-open the gate on its
own estimated state, re-target burn 1 at the fixed time-of-flight, re-screen
against the fixed constraints — and a sample that no longer admits a screened
plan is a `planning_infeasible` outcome, a class of its own, not a flight with
fabricated margins.

**Cluster, then sample.** Each cluster draws the environment once (a J2 factor
and a log-normal density factor, median 1) from `default_rng([seed_base,
cluster])`; each sample inside the cluster then draws its own navigation and
execution errors from `default_rng([seed_base, cluster, index])`, in a fixed
order where a disabled or zero-width component consumes no randomness. So the
cluster is the exchangeable unit, and the interval on the rate is quoted twice:
the Wilson interval on the pooled count (correct if clusters were independent
draws of everything), and a whole-cluster percentile bootstrap, which prices
what the clusters actually share. The sample count is not chosen: it is derived
as the smallest `n` whose Wilson interval is within the declared
`target_ci_width` at the worst-case planning rate, then rounded up to whole
clusters.

**Truth is the planning physics plus the declared disturbances.** The two-body
and J2 terms are `orbitopt.relative.propagation._accel`, imported and not
restated; drag is the repo's `ExponentialAtmosphere` in a co-rotating frame; the
integrator is the same fixed-step RK4 discipline as `propagate_pair`. Drag is
present whenever the source is enabled — the campaign's premise is that the plan
ignores the atmosphere, so the truth carries the atmosphere and a log-normal
factor prices what is unknown about it. The navigation error moves the *estimate*
and not the truth: the vehicles start at the declared config states, and the
planner sees the chaser's state plus the declared sigma. The control test asserts
what this buys: with every family off, the flown arrival matches the planned
arrival to 1.6e-4 m, so every metre the campaign measures belongs to a
disturbance.

**Grade with the mission's own checker, at the campaign grid.** Every success is
an E12-04 pass (`evaluate_trajectory_safety`) on the trajectory actually flown,
plus the config's own budget and arrival tolerances, and a sample that violates
two gates is attributed once, by declared precedence
(`planning_infeasible > safety_violation > fuel_over_budget > arrival_accuracy`).
The truth grid is `sample_dt_s` = 5 s, coarser than the config's
`check_sample_dt_s` by necessity — a Monte Carlo cannot afford the plan-check
grid on 264 samples twice over — and `grid_convergence` re-flies the marginal
samples at 1 s to measure whether the interval can clip a violation, reporting
`grid_dependent` if a verdict ever moves.

**Three independent lines, and only one of them is an independent model.** The
spot check re-flies the samples whose verdict a small model difference could
move (tightest of each constraint, worst-arrival successes, one per failure
class), never a random cut. The *independent integration* line re-flies those
samples adaptively (DOP853 at `orbitopt.verify.tolerances.VERIFY_RTOL/ATOL`)
against the production RK4 — same force law, so it bounds integration error and
is measurable anywhere; that is why the tolerance pair now lives in a
module that does not import `tudatpy`. The *independent formulation* line
replays the plan on the linear CW model against E12-05's declared validity
bands — CW is not a candidate truth model for a mission with J2 and drag, so
this line bounds what an approximate model could explain, and is not
confirmation. The *independent model* line is the E12-05 TudatPy re-flight; with
no backend it is recorded `not_checked` with `REFLY_BACKEND_MISSING`, never
passed.

**Report the association and the marginal, and let them disagree.** Under joint
dispersion the sensitivity reading ranks each source by the rank correlation of
its own sigma-equivalents against failure, and labels a source it cannot assess
`uninformative` or `inactive_or_bias_only` rather than reporting a zero
correlation that would read as "measured, found nothing". `solo_campaigns` runs
the same layout and the same seeds with one family at a time, which is the only
reading that separates a source that hurts on its own from one that merely
co-occurs with failure.

## Consequences

The headline number is a rate with two intervals and a per-cluster spread, not a
percentage. The campaign's declared levels are stress levels — the edge of what
mission analysis can bound per source (documented line by line in the campaign
YAML), not an RSS — so the failure taxonomy is populated and the sensitivity
ranking has a spread to rank. The measured answer for LEO-RDV-001, at the
packaged campaign (22 clusters × 12 samples per arm), is a hold-out success rate
of **0.053** (14/264): Wilson [0.032, 0.087], whole-cluster bootstrap
[0.027, 0.080], train arm 0.053. Failures: 181 arrival-accuracy, 69 safety
(keep-out and terminal cone), zero planning-infeasible, zero over-budget. The
two-impulse, open-loop design is fragile — a velocity error of a few mm/s is
carried through the drift leg into tens of metres the burns cannot see, and a
mis-pointed 7 m/s burn is a position error that grows with the drift leg. The
solo lineup is where the sources separate: burn execution alone leaves 0.064,
navigation error alone 0.508, drag alone 0.591, J2 alone 1.000 — pointing is the
one source that is mission-critical by itself. The joint rank-correlation
reading ranks `drag, execution, j2, initial_state`, and the two readings
disagreeing is expected: with a 95 % base failure rate the correlation against
failure saturates, which is exactly why the solo lineups run. The reading for
the design is recorded with the number — a mid-course correction, or a pointing
budget priced against the 50 m gate rather than against the launcher's
convenience.

Costs: a sample re-plan and truth flight costs about a second and a half, so a
full campaign with the solo lineup is ~1,800 trajectories and runs on the
compute host with `--jobs`; the re-plan assumes the onboard guidance re-solves
the transfer, which is a guidance assumption the mission's GNC would have to
confirm; and the 1 s convergence check is only as convincing as the samples it
picks, which is why the pick is the marginal ones and the reported number is the
largest observed change rather than a mean.
