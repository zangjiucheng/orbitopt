# Rendezvous verification report

LEO-RDV-001 carries four lines of evidence. They are **labelled separately**
— a nominal pass is neither an independent pass nor a robust success — and
each names the digests it grades.

The sealed plan: canonical planner `plan_rendezvous_safe` on
`leo_rdv_001.yaml` (config sha256 `7cf9a0a2…158b03`),
**plan_digest `5dd0a6cd084b7a6782eb9f4700ee16d56160f119f700413248f68f8daefae6f2`**
(maneuvers sha256 `b2699e8d…62aba2`), full text in the file
`examples/runs/leo-rdv-001-sealed-plan.json`.
It spends **10.345 m/s** of a 25 m/s budget (Δv margin 14.65 m/s), time of
flight 3416.7 s, arrival 3977.3 s after the shared epoch.

## 1. Nominal verification (same model — not independent)

The acceptance propagator re-runs the sealed plan and grades the config's
checks:

| Check | Result |
|---|---|
| Arrival position error | 4.7×10⁻⁷ m (tolerance 50 m) |
| Arrival relative speed | 5.2×10⁻¹⁰ m/s (tolerance 0.2 m/s) |
| Total Δv | 10.3454 m/s (budget 25 m/s) |
| Gate / time-of-flight windows | passed |
| Overall `verify_plan` | **passed** |

Whole-trajectory safety on the dense grid also passes with margins: keep-out
min range 1000.0 m against a 500 m sphere (margin 413.0 m to surface),
cross-track margin 79.9 m, terminal-cone margin 20.0°, approach-speed margin
2.54 m/s, and 3 passive-contingency releases checked clear. Raw records:
`examples/runs/leo-rdv-001/rendezvous.json` and `verification.json`.

## 2. Mission Studio replay — local, not independent

`orbitopt rendezvous studio` re-propagates the sealed plan with the same
`propagate_pair` model inside the container
([ADR rendezvous-0004](adr/rendezvous-0004-mission-studio-flightlab-replay.md)).
It exists to inspect and share a flight, not to verify physics. Its sidecar
records the consumed `plan_digest`; the precomputed run at
`examples/runs/leo-rdv-001` matches the sealed plan digest exactly.

## 3. Independent re-flight (TudatPy — different physics stack)

A TudatPy two-spacecraft nonlinear re-flight in a separate environment is
the independent line
([ADR rendezvous-0003](adr/rendezvous-0003-independent-refly-tudatpy.md),
[report](benchmarks/e12_05_independent_refly.json)). The recorded run
**passed** its declared checks, closest approach ≈ 909 m.

Digest discipline (fail-closed): the re-flight is graded against the plan
whose maneuver digests it matches. The recorded evidence carries maneuvers
sha256 `1d2b1926…c04612` — the E12-03 planning reference (10.3437 m/s) —
not the sealed safety plan's `b2699e8d…62aba2`. So the studio run reports
`physics_verified: null` (**not assessed**, not failure) for the sealed
plan, and a re-flight of the sealed plan stays open before the epic closes.
Re-run with
`orbitopt crossenv run --plan src/orbitopt/data/crossenv_e12_05_plan.yaml`.

## 4. Robust dispersion (Monte Carlo — a stress price, not a nominal result)

The E12-07 campaign
([ADR rendezvous-0005](adr/rendezvous-0005-dispersion-campaign-not-a-safety-factor.md),
[data](benchmarks/e12_07_rendezvous_robustness.json)) prices the **sealed**
design (ground plan = this digest, tof 3416.67 s) under declared stress,
with clustered sampling and a train/hold-out split:

| Quantity | Value |
|---|---|
| Hold-out success | **14 / 264 = 5.3 %** |
| Wilson 95 % CI | 3.2 – 8.7 % |
| Cluster bootstrap CI | 2.7 – 8.0 % |
| Failure classes | 181 arrival accuracy, 69 safety, 0 planning/fuel |

This number is **robust**, stated against stress arms; the nominal pass in §1
is **nominal**, stated at zero stress. Report one as the other and the
document is wrong either way.

## Where the numbers live

| Evidence | Artifact |
|---|---|
| Nominal + safety | `examples/runs/leo-rdv-001/{rendezvous,verification}.json` |
| CW envelope | `docs/benchmarks/e12_02_cw_validity.{md,json}` |
| Independent re-flight | `docs/benchmarks/e12-05-refly-run/` + `e12_05_independent_refly.json` |
| Robust campaign | `docs/benchmarks/e12_07_rendezvous_robustness.json` |
