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
(dispersion campaign). The [E12-10 re-flight ADR](adr/rendezvous-0004-independent-robustness-reflight.md)
describes the independent checks on selected dispersed cases.

A precomputed Mission Studio run of LEO-RDV-001 is committed under
`examples/runs/leo-rdv-001`; the [quickstart](rendezvous-quickstart.md)
shows how to open it with no compute stack.
