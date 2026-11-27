# GNC simulation contract (E13)

Closed-loop guidance, navigation and control runs in orbitopt exchange exactly five kinds of
record, on one clock, and every run seals to a digest. This page states what each record says
and what the harness guarantees; `src/orbitopt/schemas/gnc-contracts-1.0.json` is the normative
schema. Design rationale is in ADR gnc-0001.

## The five records

| Kind (`kind` field) | Says | Carries |
| --- | --- | --- |
| `orbitopt.gnc.truth-1.0` | What the plant actually was | embedded `relative-state-1.0` document, dynamics tick |
| `orbitopt.gnc.observation-1.0` | What a sensor measured | state estimate + diagonal position/velocity covariance, sensor id, sample tick, arrival time, source tick |
| `orbitopt.gnc.estimate-1.0` | What the filter believed | state estimate + full 6x6 covariance, source observation ids |
| `orbitopt.gnc.command-1.0` | What guidance asked for | commanded delta-v, actuator id, fire time, requested actuation time, source estimate id |
| `orbitopt.gnc.control_event-1.0` | What actually executed | realized delta-v (zero when `executed` is false), source command id |

An unexecuted `ControlEvent` must carry a zero realized delta-v: a misfire is recorded as a
non-execution, never as a silent burn. Truth flows only into the run log — see the isolation
guarantee below.

## One clock

`ClockSpec` (in `orbitopt.gnc`) declares the run epoch (a `TT`/`TDB` `Epoch`), the dynamics step,
one `SensorSpec` per sensor (period, phase, latency, `dropped_ticks`), the estimator cadence and
the controller cadence. Every instant is a pure function of the tick index: `t0 + phase + n*period`.
A dropped frame deletes that measurement and never shifts the next one; declared latencies are
enforced by the run log on append.

## Running and replaying

    from orbitopt.gnc import ClockSpec, SensorSpec, EstimatorSpec, ControllerSpec, \
        SensorNoise, ProportionalLaw, simulate, write_run_log, read_run_log

    log = simulate(run_id="rdv-01", clock=clk, frame="GCRS", target_id="tgt", chaser_id="chaser",
                   chief_r_m=..., chief_v_m_s=..., deputy_r_m=..., deputy_v_m_s=...,
                   laws={"chaser": ProportionalLaw(1e-2, 0.2)},
                   noise={"navcam": SensorNoise(1.0, 0.02, 7)}, t_end_s=80.0)
    write_run_log(log, "rdv-01.json")     # seals run_digest
    log = read_run_log("rdv-01.json")     # re-validates every record and the digest

`read_run_log` refuses a log whose records were edited (digest mismatch), whose measurements claim
to predate their sample or arrival, whose estimates predate arrival, whose actuations miss the
requested time, or whose mission roster epoch disagrees with the clock — the acceptance walls are
tested negative-first in `tests/test_e13_01_*`.

## Isolation guarantee

`run_guidance_step(law, feed, ...)` is the only way guidance sees the loop; the law receives a
`GuidanceInput` exposing exactly one `Estimate` (`__slots__ = ("_latest",)`). Feeding a
`TruthSample` (or any non-estimate) raises `TypeError`, and `orbitopt.gnc.feed` contains no truth or
plant identifier — an AST test in `tests/test_e13_01_truth_isolation.py` fails the suite if one is
ever added, so the guidance side of the package cannot even name plant state.

## Sensor models (E13-03)

`orbitopt.gnc.sensors` declares *where observations come from and how they lie*. A
`RelativeSensorModel` is built from a task-configuration document
(`orbitopt/schemas/gnc-sensor-config-1.0.json`; `reference_task_config()` ships a
documentation-quality pair) and declares per channel: white-noise sigma (strictly positive — a
zero-variance claim is refused), constant bias, and a mandatory `source` provenance string on
every error parameter (a citation or `DESIGN ASSUMPTION: …`). Implemented profiles:
`polar_range_angle` (range + LOS azimuth/elevation, optional Doppler range-rate) and
`cartesian_relnav` (differenced relative position, optional velocity); new channels or profiles
are new rows in `CHANNELS` / `SENSOR_KINDS`, not new classes.

The four error mechanisms and who owns them:

| Error | Model | Wall |
| --- | --- | --- |
| White noise | Gaussian, seeded on `(seed, tick)` like every orbitopt draw | sigma must be > 0; moments tested against config |
| Bias | constant per run, added before noise | `bias_covariance_mode`: `none` (filter models it; NIS inflates by `Σ bias²/σ²` — demonstrated) or `bounded` (declared variance + = bias²) |
| Latency | the clock's constant, never random | `t_available == t_sample + latency` enforced E13-01; `measure()` refuses a disagreeing tick |
| Data loss | per-tick Bernoulli + geometric bursts (mean ≥ 1) + explicit scenario ticks | generated as `dropout_schedule(n_ticks)` from `(seed, n_ticks)`, merged into `sensor_spec()` → the clock stays the single timing authority |

A dropped frame is the *absence* of an Observation (`observe()` returns `None`; the
`SensorFix` layer carries the explicit `valid` flag and reason for plant-side bookkeeping).
Observation output maps the diagonal measurement covariance through the channel Jacobians
(polar position: `(Jᵀ W J)⁻¹`; range-rate-only velocity: rank-1 `σ² r̂ r̂ᵀ`, honestly along-LOS);
a sensor with no velocity channel ships no velocity covariance rather than an invented one.
The polar reconstruction refuses the ±Z boresight singularity (`1e-6·|r|` of azimuth).
`measurement_nis(fix, truth)` is the shared measurement-consistency statistic (E13-02 consumes it).

Consistency evidence: `tests/test_e13_03_*` (per-error moments, NIS gates) and
`benchmarks/bench_e13_03_sensor_consistency.py` (sealed at
`docs/benchmarks/e13_03_sensor_consistency.json`: chi-square moments z-scored < 4, polar
covariance coverage 0.9985, bias inflation predicted). Filter-level NEES was deferred to the
merge of E13-02 (#419) — no Estimate stream existed on that branch — and the reason was asserted
in `tests/test_e13_03_nis_consistency.py` so the deferral failed loudly when #419 landed. #419
has landed: the deferral is discharged there by #419's sealed 6-DoF NEES gate
(`docs/benchmarks/e13_02_relative_nav_ekf.json`).

## Independent closed-loop verification (E13-07)

`orbitopt.verify.gnc_closed_loop` grades a sealed run log with four SEPARATE error classes, never
blended into one flag; each line carries its own evidence type and its own `cannot_prove` limits:

| Class | Question asked | Evidence type |
| --- | --- | --- |
| `numerical_error` | Does re-integrating the recorded commands converge to the recorded truth (step-refinement sweep + tolerance gate)? | `numerical_consistency` |
| `model_error` | Does an INDEPENDENT propagator (TudatPy, multi-arc over the recorded events) reproduce the truth? Absent backend: `not_checked`, never imputed. | `independent_model` |
| `navigation_inconsistency` | Are observations and estimates statistically consistent with their DECLARED covariances (NIS / NEES against a truth reference refined through the run's own declared dynamics)? | `analytical` |
| `optimizer_control_nonconvergence` | Did every command execute within its declared bounds, and does the producer's convergence assertion (if any) agree with the record? | `analytical` |

    from orbitopt.gnc import read_run_log
    from orbitopt.verify.gnc_closed_loop import verify_gnc_closed_loop, verify_evidence_binding

    report = verify_gnc_closed_loop(read_run_log("rdv-01.json"), out_path="rdv-01-gnc.json")
    verify_evidence_binding(report, "rdv-01.json")   # digests re-match, or ValueError

The replay splits segments at actuation instants and carries the PRE-impulse truth sample at each
(evidenced against the reference sim's event priority); a truth stream shorter than two samples is
recorded `not_checked`, not scored; and the verification digest binds this verdict to this exact run
— flipping any check, or pointing the same report at another run, raises. Acceptance tests:
`tests/test_e13_07_independent_closed_loop_verification.py`.
