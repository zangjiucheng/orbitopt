# Rendezvous verification report

LEO-RDV-001 carries five lines of evidence. They are **labelled separately**
— a nominal pass is neither an independent pass nor a robust success — and
each names the digests it grades.

The sealed plan: canonical planner `plan_rendezvous_safe` on
`leo_rdv_001.yaml` (normalized config SHA-256 `7915deab…bb0549`),
**plan_digest `11cb8cd24bd8c96391576703000cf14b42cde2de5978d4b0d8631ea4a34e9bfb`**
(maneuvers sha256 `d10771e1…8b9e5d`), full text in the file
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
**passed** its declared checks, closest approach ≈ 913 m.

Digest discipline (fail-closed): the re-flight is graded against the plan whose maneuver digest it matches. The
recorded evidence carries maneuvers sha256 `d10771e1…8b9e5d` — the sealed plan's (10.3454 m/s) — so the studio
run built with `--refly-report` reports `physics_verified: true`, derived from that match. Evidence whose digest
does not match, or no report at all, gives `physics_verified: null` (**not assessed**, not failure). The
re-flight covers the *nominal* sealed plan; §5 re-flies perturbed samples of it.
Re-run with
`orbitopt crossenv run --plan src/orbitopt/data/crossenv_e12_05_plan.yaml`.
To reproduce the packaged Studio verdict, use the
[full report](benchmarks/e12-05-refly-run/rendezvous-refly-report.json) as
`--refly-report`. The benchmark summary lists the comparison and tolerances.

## 4. Robust dispersion (Monte Carlo — a stress price, not a nominal result)

The E12-07 campaign
([ADR rendezvous-0005](adr/rendezvous-0005-dispersion-campaign-not-a-safety-factor.md),
[data](benchmarks/e12_07_rendezvous_robustness.json)) evaluates the reference
transfer under declared stress, with clustered sampling and a train/hold-out
split. The archived E12-07 JSON records the transfer timing and Δv. E12-10
records the sealed plan digest explicitly.

| Quantity | Value |
|---|---|
| Hold-out success | **14 / 264 = 5.3 %** |
| Wilson 95 % CI | 3.2 – 8.7 % |
| Cluster bootstrap CI | 2.7 – 8.0 % |
| Failure classes | 181 arrival accuracy, 69 safety, 0 planning/fuel |

This number is **robust**, stated against stress arms; the nominal pass in §1
is **nominal**, stated at zero stress. Report one as the other and the
document is wrong either way.

## 5. Independent robustness re-flight (E12-10 — the same sealed plan, perturbed)

The E12-10 benchmark
([ADR rendezvous-0004](adr/rendezvous-0004-independent-robustness-reflight.md),
[data](benchmarks/e12_10_independent_reflight.json)) re-grades the E12-07 hold-out with the passive contingency
releases (no-burn at t0, gate abort at t1, hold-point coast) counted in success, and re-flies a declared selection
of 23 samples in TudatPy, each with its own perturbation. Its ground plan is the sealed plan above (same
`plan_digest`).

| Quantity | Value |
|---|---|
| Hold-out success, passive contingencies counted | **13 / 264 = 4.9 %** |
| Cluster bootstrap 95 % CI | 2.7 – 7.6 % |
| Failure classes | 164 arrival accuracy, 87 safety (the hold-point coast dominates) |
| TudatPy per-sample agreement | 23 of 23 verdicts agree, 0 flips, every residual inside its band |

The classification is **Independently Verified Robustness Characterization**: it says the *evidence* is
independent, not that the design is robust. The design succeeds on about one hold-out sample in twenty.

## Digest stability

A plan solved on two machines differs in the last bits of its burns (about 3e-12 relative here), so the digests
hash floats rounded to 8 significant digits (`DIGEST_SIGNIFICANT_DIGITS`). Before that, one config gave a
different `plan_digest` on every machine tried. Rounding can still split two values that straddle a boundary (about
3e-4 per number), so equality across machines is stable in practice, not guaranteed; a mismatch should be read
as "recompute and compare the maneuvers", not as proof of a different plan.
`tests/test_e12_08_canonical_plan.py` checks the committed sealed plan, run and benchmark against the digest
this code computes.

## Where the numbers live

| Evidence | Artifact |
|---|---|
| Nominal + safety | `examples/runs/leo-rdv-001/{rendezvous,verification}.json` |
| CW envelope | `docs/benchmarks/e12_02_cw_validity.{md,json}` |
| Independent re-flight | `docs/benchmarks/e12-05-refly-run/` + `e12_05_independent_refly.json` |
| Robust campaign | `docs/benchmarks/e12_07_rendezvous_robustness.json` |
| Independent robustness re-flight | `docs/benchmarks/e12_10_independent_reflight.json` |
