# E13-05 — constrained control research: LQR / MPC vs the baseline, paired on the holdout (task #422)

Epic #416. This task asks what a *different mid-course feedback law* buys on
the closed-loop seam E13-04 built (`orbitopt.rendezvous.gnc_campaign`): an
unconstrained CW-LQR arm and a constraint-in-the-QP MPC arm
(`orbitopt.gnc.lqr`, `orbitopt.gnc.mpc`, flown by
`orbitopt.rendezvous.constrained_campaign`), compared against the E13-04
baseline on **identical** perturbation and execution draws. The machine
reading is [`e13_05_constrained_control.json`](e13_05_constrained_control.json);
reproduce with `PYTHONPATH=src python
benchmarks/bench_e13_05_constrained_control.py --jobs 16 -o …` (≈20 min wall
at 24 jobs on a 64-core host, CPU only). The baseline arm is re-flown through
a pass-through timing wrapper and the run **aborts** unless it reproduces the
frozen E13-04 artifact sample-by-sample (it did: exact on all 264).

## The experiment

The same LEO-RDV-001 holdout as E13-04: 22 clusters × 12 samples, seed base
20 000 000, identical perturbations, ground plan (tof 3416.67 s, planned
10.345 m/s), tolerances (50 m / 0.2 m/s) and the untouched 25 m/s budget
(`guards.budget_unchanged: true`). Each sample flies three arms that share
the departure burn, the EKF/sensing stack, the plant and the grading:

* **baseline** — the E13-04 schedule (mid-course retarget at t₁ + 0.5·tof,
  terminal trim, arrival);
* **lqr** — frozen CW-LQR gain on the compensated estimate, regulating the
  sample's planned arc, saturating at the 1 m/s pulse cap;
* **mpc** — 12-step (120 s) receding-horizon condensed QP over the same CW
  impulse model with the mission's constraint families *in* the problem
  (keep-out 500 m, cross-track 100 m, cone, range-indexed speed ceiling),
  per-pulse thrust cap, and the remaining Δv budget as a smooth `sum|u|`
  surrogate; terminal cost = LQR cost-to-go from the shared Riccati
  sequence. Solve failure falls back to the LQR gain, counted.

Both new arms fire the mission's own terminal pair (`ManeuverPlan`s at t_f −
40 s and t_f − 20 s, caps 3 / 8 m/s) as *scheduled* burns and reserve those
caps inside their budget gate — so the arms differ only in the mid-course
phase (ADR gnc-0003, decision 1).

## What the numbers say

| arm | success | Wilson 95 % | terminal pos (med/max) m | Δv (med/max) m/s | realized family violations | decision ms (mean/p95) |
|---|---|---|---|---|---|---|
| baseline | 258/264 | [0.951, 0.990] | 42.1 / 82.0 | 11.38 / 12.82 | 0 | 0.011 / 0.006 |
| lqr | 260/264 | [0.962, 0.994] | 45.8 / 50.2 | 11.50 / 12.16 | 0 | 0.36 / 0.43 |
| mpc | 255/264 | [0.936, 0.982] | 45.4 / 176.2 | 15.97 / 24.92 | 0 | 23.4 / 48.7 |

* **Paired tests**: none of the four McNemar comparisons is significant
  (lqr success 3↔1 discordant, p = 0.625; mpc 3↔6, p = 0.508; violation
  endpoints 0↔0, p = 1.0; Holm-adjusted p = 1.0 throughout). On this mission
  the *law shape* does not move the success rate; the terminal pair and the
  budget policy dominate.
* **Failure classes move, though.** Baseline's 6 losses are
  `safety_violation` (campaign trajectory checks); LQR's 4 and MPC's 9 are
  3 `safety_violation` + 1/6 `arrival_accuracy`. MPC's worst arrivals
  (176 m, 1.0 m/s) share one mechanism: its mid-course appetite
  (feedback 6.05 m/s mean vs LQR's 1.07) spends into the terminal-pair
  reserve, and the capped 8 m/s arrival burn can no longer finish the job.
  Constraint-aware, Δv-hungry, terminal-risky — with 14 036 fallback ticks
  at a 15.7 % rate across the campaign's ~89 000 MPC decisions.
* **The four realized constraint families (keep-out, cone, cross-track,
  speed) held in every sample of every arm.** The families' worst flown
  margins rank as expected: MPC tightest to the walls it can see (speed
  +0.07 m/s, cone +9.8 m), LQR looser (+2.5 m/s, +20.7 m), and this is the
  one dimension where the arms genuinely differ in kind, not degree.
* **Compute** spans three orders of magnitude: closed-form schedule
  0.011 ms/decision, frozen-gain LQR 0.36 ms, SLSQP-MPC 23 ms mean / 49 ms
  p95 — fine offline or at a 10 s cadence, but MPC buys nothing *on this
  mission's numbers* except the certified in-horizon constraint handling.
* **Tracking**: RMSE ≈ 79 m in all arms — dominated by the long approach-leg
  offset prescribed by the plan itself, not by the law; Δv totals stay
  inside budget everywhere (max 24.92 < 25, no `fuel_over_budget`).

## Limits of the conclusion

Single mission, single scenario family, linear-CW prediction inside the QP,
SLSQP (not a real-time solver), and a holdout whose plan keeps ≥ 2 m/s
margin to the speed ceiling. A mission that must *hold* margins near a
boundary (or a stress set that erodes them) is where MPC's design case
lives; the thin MPC margins measured here show its constraint active set
does engage. The applicability/limits conclusion is ADR gnc-0003.
