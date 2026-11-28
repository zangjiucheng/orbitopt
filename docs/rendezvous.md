# Rendezvous

orbitopt plans and verifies two-impulse orbital rendezvous. In the
**LEO-RDV-001** reference mission, a chaser starts at 490 km altitude and
transfers to a hold point 1 km behind a target in a 500 km orbit.

| Page | |
|---|---|
| [Quickstart](rendezvous-quickstart.md) | Open the precomputed run, plan and verify a mission, re-run the independent re-flight. |
| [Theory](rendezvous-theory.md) | Relative motion, the CW planning envelope, pair propagation and safety constraints. |
| [Mission configuration](rendezvous-mission-config.md) | The rendezvous config v1.0 schema, field by field. |
| [Verification report](rendezvous-verification-report.md) | Nominal, independent and robust evidence — labelled separately, with digests. |

The decisions behind each layer are ADRs:
[relative-0001](adr/relative-0001-lvlh-relative-frames-cw-and-pair-propagation.md)
(LVLH frames, CW envelope, pair propagation),
[rendezvous-0001](adr/rendezvous-0001-two-impulse-j2-targeted-lambert.md)
(planner), [rendezvous-0002](adr/rendezvous-0002-safety-constraints-whole-trajectory.md)
(safety), [rendezvous-0003](adr/rendezvous-0003-independent-refly-tudatpy.md)
(independent re-flight),
[rendezvous-0004](adr/rendezvous-0004-mission-studio-flightlab-replay.md)
(studio replay), [rendezvous-0005](adr/rendezvous-0005-dispersion-campaign-not-a-safety-factor.md)
(dispersion campaign). The [E12-10 re-flight ADR](adr/rendezvous-0006-independent-robustness-reflight.md)
describes the independent checks on selected dispersed cases.

A precomputed Mission Studio run of LEO-RDV-001 is committed under
`examples/runs/leo-rdv-001`; the [quickstart](rendezvous-quickstart.md)
shows how to open it with no compute stack.

## Closed-loop guidance (E13)

E13 adds relative navigation with an extended Kalman filter, feedback guidance,
constrained control and a safety supervisor. The [GNC run contract](gnc-contract.md)
defines the shared clock and the records for truth, observations, estimates,
commands and executed control events.

| Recorded study | Result under its declared conditions |
|---|---|
| [Feedback guidance](benchmarks/e13_04_closed_loop_guidance.md) | **258/264** hold-out cases passed with feedback and **13/264** passed open loop on the same draws. |
| [Constrained control](benchmarks/e13_05_constrained_control.md) | The baseline, LQR and MPC passed **258**, **260** and **255** of 264 cases. The paired comparisons found no statistically significant difference. |
| [Safety supervisor](benchmarks/e13_06_safety_supervisor.md) | The injected navigation-fault arm missed none of the 247 truth violations. **155/264** post-abort checks passed. |

These are simulation results for the declared stress cases. The E12 sealed
reference and its independent evidence remain described in the
[verification report](rendezvous-verification-report.md).
