# ADR flightlab-0005: Joint chance-constrained recovery planning (E3-02)

- **Status:** Accepted
- **Date:** 2026-11-14
- **Deciders:** Recovery Guidance & Robustness (Epic #217), E3-02 (#259),
  follow-up to R-01 (ADR flightlab-0003) and R-07 (epic #217 stack audit)
- **Code:** `src/orbitopt/flightlab/recovery/chance.py` (`JointChanceSpec`,
  `evaluate_joint_chance`, `joint_penalty`),
  `src/orbitopt/flightlab/recovery/boostback.py` (`plan_boostback_entry(joint_chance=...)`,
  `_evaluate`, `CorridorReport`), `src/orbitopt/flightlab/recovery/montecarlo.py`
  (sample records `landing_probability` / `joint_violation`)
- **Evidence:** `docs/benchmarks/e3_02_joint_recovery.json`
  (`benchmarks/bench_e3_02_joint_recovery.py`),
  `tests/test_flightlab_recovery_chance.py`,
  `docs/benchmarks/r07_nominal_dispersed.json` (the regression this ADR fixes)

## Context and problem

R-01 (ADR flightlab-0003) moved the corridor handoff's two landing-relevant
consequences — the divert the landing burn must close and the propellant
margin it inherits — into the Boostback objective as *soft weighted terms*
(`landing_divert_penalty = (miss/budget)^2`,
`reserve_shortfall^2`). R-07 then stacked R-01 onto the declared-actuator
landing stack and measured two things (epic #217 audit):

* the stacked column is *item-for-item identical* to R-01 alone — the extra
  robustness layers bought nothing on top of the soft terms; and
* the landing scatter **regressed**: max touchdown radius 338 m (FL-305
  baseline) → 833 m (the R-01 stack).

Both symptoms have the same root cause: a weighted sum trades. A penalty
weight is a price, not a requirement — the optimiser happily spends divert
penalty to buy margin (or the reverse) whenever the weighted sum says the
exchange is profitable, and the *population* scatter is the visible bill.
The task (E3-02) asks for the three recovery quantities — fuel margin,
divert feasibility, landing probability — to enter the plan **as
constraints, or chance constraints, not as penalty weights to tune**.

## Options considered

1. **Re-tune the R-01 weights** (larger divert exponent, bigger reserve
   weight). Rejected: keeps the trade structure R-07 just measured as the
   cause; no principled way to pick the weights; R-07's zero-gain stack
   says the family is exhausted.
2. **Hard feasibility constraints inside FL-302** (reject non-qualifying
   rollouts, e.g. `minimize` under `NonlinearConstraint`). Rejected: from an
   infeasible warm start the Nelder–Mead simplex sees a plateau of rejected
   vertices with zero gradient information, and the corridor-crossing
   condition itself is discontinuous in the plan vector; the accepted plan
   stops being a rollout-to-rollout comparison, i.e. `CorridorReport` loses
   its meaning. (Same argument FL-302 already used against hard terrain/
   corridor gates.)
3. **Joint chance constraints with a deterministic equivalent and an exact
   penalty** (adopted, below): each requirement becomes a probability
   statement evaluated from the *recorded* dispersion bands, the
   probabilities are aggregated *without trading* (worst normalized
   violation, an l∞ structure), and the violation enters as an exact
   penalty with a scale outside every tradeable term — no weight to tune.

## Decision

`orbitopt.flightlab.recovery.chance` defines the joint requirement set
`JointChanceSpec` and `plan_boostback_entry(..., joint_chance=...)` selects
the mode (it *replaces* the two R-01 soft terms — the planner refuses to
build a mode where both would double-count, and the report keeps the R-01
fields `None` in joint mode):

* **Fuel margin, chance form.** The handoff margin must cover the p-quantile
  of the dispersion loss on the propellant still at risk at handoff:
  `margin ≥ Φ⁻¹(p_fuel) · σ_loss(exposure)`, with σ_loss = exposure-weighted
  1σ of the recorded FL-304 dispersion bands (thrust 5 %, Isp 3 %, density
  10 %, C_d 8 %; Gaussian-composed) applied to the propellant *at risk*
  (already burned + landing need). Default `p_fuel = 0.9`, σ floor 40 m.
  At the nominal 20 t exposure this requires ≈ +1.03 t — R-01's hand-picked
  1000 kg design point, now derived instead of asserted.
* **Divert feasibility, scatter-derived.** The corridor divert bar is not a
  free knob but derived from the acceptance metric itself: requiring the
  q-quantile of a Rayleigh touchdown scatter (σ from the closed-loop
  measurements, σ = σ₀ + gain·divert, σ₀ floor 40 m, gain 0.5 — the ADR
  flightlab-0003 table, 0.35 km divert landing at ~0.4 m and the 1σ ellipse
  ≈ 1 m) caps the *predicted* divert at
  `(budget/√(−2·ln(1−q)) − σ₀)/gain` ≈ **852 m** for budget 1000 m at
  q = 0.9, and the handoff's Rayleigh coverage `P(radius ≤ budget)` is
  recorded as `p_divert`.
* **Landing probability floor.** `p_land = p_fuel · p_divert ≥ 0.8`
  (default), the joint statement the acceptance asks for.
* **Aggregation without trading.** The three normalized violations enter the
  objective as `JOINT_PENALTY_SCALE · max_i g_i` with an l∞ (worst-violation)
  structure and an exact-penalty scale of 25 — above the full
  corridor-miss/impact band (≤ 25) and any plausible base-term difference —
  so the optimiser cannot buy margin with divert or the reverse: it must fix
  the worst violation first. `JointChanceSpec` records both verdicts in the
  `CorridorReport` (`landing_probability`, `joint_violation`), which the
  Monte Carlo records per sample, so tail rollouts carry their own
  probability statement for E3-03.

**Defaults are design assumptions pinned to recorded evidence** (documented
as `DESIGN ASSUMPTION` in the spec docstring): p_fuel 0.9 ≈ R-01's
validated 1 t design point; the scatter budget 1000 m is the E3 acceptance
scale (well inside the 2.5 km landing tolerance); the Rayleigh parameters
are fitted on ADR flightlab-0003's landing table. Nothing is fitted from the
evidence set below, so the comparison stays honest.

## Consequences (n = 32, seed 1, paired, 20 deg/s TVC pinned in every arm)

`bench_e3_02_joint_recovery.py` flies the three objective modes over the
identical ascent/dispersions (per-sample rows in the JSON; paired by
construction):

| arm | success (Wilson 95 %) | touchdown radius p50 / p90 / max | 1σ ellipse |
|---|---|---|---|
| fl305_baseline | 0.406 [0.255, 0.577] | 0.9 / 295 / 436 m | 1.2 × 0.7 m |
| r01_landing_aware | 0.500 [0.336, 0.664] | 1.0 / 182 / 638 m | 1.1 × 0.7 m |
| **e3_02_joint_chance** | **0.531 [0.364, 0.691]** | **0.9 / 118 / 628 m** | **1.1 × 0.6 m** |

* **Do-no-harm, exactly:** the 13 samples both FL-305 and the joint arm land
  have **identical** touchdown points (paired Δ = 0.0 m) — the joint mode
  differs from the baseline only where a constraint binds.
* **Paired flips:** +4 landings (samples 1, 19, 24, 28), −0 (McNemar 4–0,
  p = 0.125 at n = 32 — E3-03's larger benchmark settles significance); the
  worst corridor margin over the run improves −1461 kg → −448 kg.
* **Max-radius regression explained, R-07's bill paid:** like R-01 the joint
  arm's max radius (628 m) exceeds the baseline's (436 m), and the whole
  gap is a *selection effect*: the largest-radius sample (idx 18) is one the
  baseline never landed at all (`INSUFFICIENT_PROPELLANT_RESERVE`; it
  entered the joint arm's landed set at 628 m, off tolerance, exactly as
  R-01's arm did — and the joint arm's max is below R-01's 638 m).
  Conditional on landing, the joint arm's scatter is the best of the three
  (p90 118 m vs 295/182 m; ellipse 1.1×0.6 m). The constraint set converts
  propellant-reserve losses into marginal landings — a better failure mode
  to have to certify, and the trade E3-03 must now confirm statistically.
* **Reason-code signature:** `INSUFFICIENT_PROPELLANT_RESERVE` 8 → 1 while
  `CORRIDOR_MISSED` 1 → 3: the quantile constraint rejects handoffs whose
  *predicted* margin is thin *before* the burn spends the reserve.
* Actuator limits are held constant across arms (declared 20 deg/s TVC pin,
  E3-01 consistency) — the planner switch is the only difference.

## Outlook

E3-03 re-runs this benchmark on its statistical sample (n ≥ 64) and consumes
the `paired_rows` for per-sample tests; the paired-flip counts and the
landed-conditional scatter, not the raw max radius, are the numbers to
quote. If E3-03 confirms, R-01's soft terms become the fallback mode and
`joint_chance` can become the default objective.
