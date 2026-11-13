# ADR flightlab-0003: Landing-aware Boostback objective (divert budget + fuel margin)

- **Status:** Accepted
- **Date:** 2026-11-12
- **Deciders:** Recovery Guidance & Robustness (Epic #217), R-01 (#233),
  follow-up to FL-305 (#214, ADR flightlab-0002)
- **Code:** `src/orbitopt/flightlab/recovery/boostback.py`
  (`_evaluate`, `plan_boostback_entry`, `DEFAULT_LANDING_DIVERT_BUDGET_M`,
  `DEFAULT_LANDING_RESERVE_MARGIN_KG`, `CorridorReport`),
  `src/orbitopt/flightlab/recovery/montecarlo.py` (sample records)
- **Evidence:** `docs/benchmarks/r01_boostback_objective.json`
  (`benchmarks/bench_r01_boostback_objective.py`),
  `docs/benchmarks/fl305_recovery_montecarlo.json`,
  `tests/test_flightlab_recovery_boostback.py`

## Context and problem

ADR flightlab-0002 closed the loop on the landing burn and re-pinned the
FL-302 warm start, but its own outlook named the remaining coupling:
*"FL-302's corridor penalty is still nearly blind to misses under the
corridor radius. Coupling the corridor to the landing burn's divert budget
is the natural next step if the dispersed success rate matters."*

The FL-305-baseline Monte Carlo (n = 32, seed 1, default dispersions) says
where that blindness costs: 14/32 land, and 13 of the 18 failures are
propellant-margin failures (`INSUFFICIENT_PROPELLANT_RESERVE` 8,
`LANDING_PROPELLANT_RESERVE_LOW` 5), not corridor failures. Meanwhile the
objective graded the corridor miss against the 8 km radius -- at 1.76 km
that penalty is 0.048, invisible -- and asked for nothing beyond
"landing need + reserve" until a sample violated it outright, while the
closed-loop landing burn (ADR 0002) had measured what it can actually close:
0.35 km diverts land inside every limit, 1.76 km does not.

The corridor handoff decides landing feasibility -- the divert the landing
burn must close and the propellant it inherits. R-01 moves both into the
Boostback objective instead of discovering them downstream as reason codes.

## Options considered

1. **Hard feasibility constraints in FL-302** (reject a candidate whose
   divert or margin fails). Nelder-Mead is unconstrained; a hard-rejection
   penalty cliff makes every simplex vertex inf as soon as the warm start
   neighbourhood is infeasible, and the accepted plan stops being a rollout
   comparison. Rejected.
2. **Raise the search budget** (30 -> 120 evaluations per sample). Cost
   scales ~4x and the objective stays indifferent to what it buys. Rejected
   as the primary lever; it composes with this decision later.
3. **Soft objective terms at the crossed corridor, defaults as design
   assumptions, disabled state reproducing FL-305 exactly** (adopted).
   No new reason codes, no change to the frozen feasibility checks -- the
   terms steer, the checks still govern.

## Decision

Two additive, dimensionless squared terms in the crossed-branch penalty of
`_evaluate`, both normalized by their own budget (same convention as the
existing violation terms):

- landing feasibility: `(miss_m / landing_divert_budget_m) ** 2` -- the
  same measured miss, now also against what the landing burn can close;
- fuel reserve: `short = need + reserve + landing_reserve_margin_kg - prop`
  (clamped at 0), where FL-305 used margin 0.

Defaults: 1000 m divert budget and 1000 kg margin -- DESIGN ASSUMPTIONs
anchored on the measurements above (the accepted handoff is 0.35 km; the
landing need's momentum balance sees no dispersion allowance, so one tonne
of headroom on a ~15 t recovery burn under 5 % thrust / 3 % Isp / 10 %
density bands). `landing_divert_budget_m=None` and
`landing_reserve_margin_kg=0.0` reproduce the FL-305 objective bit-for-bit
(both terms become exact-0 additions); that switch is what lets the same
conditions be compared before/after. `CorridorReport` now carries the bar
(`landing_divert_budget_m`, `min_landing_reserve_margin_kg`) so a consumer
can check the accepted plan against the bar it was steered to, and the
Monte Carlo records gained `corridor_miss_m` / `corridor_fuel_margin_kg`.

## Results

Nominal (both FL-305 and R-01 warm starts, search included): the accepted
handoff is unchanged -- 351 m off, +1662 kg above need + reserve. FL-305's
re-pinned script already sits inside the R-01 bar; the green E2E is
unmoved. The R-01 terms act where the nominal assumptions break, i.e. in
the dispersions.

Monte Carlo, identical conditions (seed 1, n = 32, default dispersions,
FL-305 warm start, closed loop; baseline column reproduces
`fl305_recovery_montecarlo.json` exactly):

| reason | FL-305 objective | R-01 objective |
|---|---|---|
| `LANDED` | 14 | **17** |
| `INSUFFICIENT_PROPELLANT_RESERVE` (FL-302) | 8 | **2** |
| `LANDING_PROPELLANT_RESERVE_LOW` | 5 | 5 |
| `LANDING_POSITION_OUT_OF_TOLERANCE` | 4 | 7 |
| `CORRIDOR_MISSED` (FL-302) | 1 | 1 |
| **success rate** | 0.438 | **0.531** |

Corridor handoff over all 32 samples (percentiles):

| handoff metric | FL-305 | R-01 |
|---|---|---|
| divert min / p50 | 203 / 925 m | 285 / 911 m |
| divert p90 / max | 1724 / 2692 m | 1879 / 3717 m |
| fuel margin min | **−1461 kg** | **−448 kg** |
| fuel margin p50 | 1079 kg | **1349 kg** |

The reserve class -- FL-305's dominant failure mode -- collapses from 13 to
7 and the worst handoff shortfall from −1461 kg to −448 kg: the margin term
does exactly its job. The cost is honest and recorded: the 30-evaluation
budget trades divert against propellant where both cannot be had, so the
divert tail grows and 3 samples move into the position-tolerance class
(nominal landed-subset pad-distance p95 nonetheless tightens, 2.18 -> 1.72 m).
Net +3 landings, and every sample is where the reason codes say it is.

## Consequences

- The FL-302 default objective is R-01's; FL-305's remains the documented
  disabled state, and the FL-305 before/after benchmark pins it (its
  numbers stay the historical record).
- The FL-305-era wiring-only Monte Carlo test pins the same switch (it
  tests statistics plumbing, not the objective).
- Remaining `LANDING_POSITION_OUT_OF_TOLERANCE` samples hand off beyond
  what the guidance cone closes at 3 km; and 2 samples still hand off
  short of the reserve. Sharper weights, a larger budget, or vehicle
  sizing are the levers for those -- not reverting the coupling.
