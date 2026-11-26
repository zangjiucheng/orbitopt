# Rendezvous

orbitopt plans and verifies two-impulse orbital rendezvous: a chaser
transfers from a phasing drift to a hold point near a target, on a model
that runs from planning through independent re-flight to a stress-tested
dispersion campaign. The reference mission is **LEO-RDV-001** — a chaser
500 km minus 10 km below a Sentinel-class target, transferred to a 1 km
V-bar hold point.

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
(dispersion campaign).

A precomputed Mission Studio run of LEO-RDV-001 is committed under
`examples/runs/leo-rdv-001`; the [quickstart](rendezvous-quickstart.md)
shows how to open it with no compute stack.
