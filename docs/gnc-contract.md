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
