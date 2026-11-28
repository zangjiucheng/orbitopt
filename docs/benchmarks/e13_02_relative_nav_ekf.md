# E13-02 — relative-navigation EKF: RMSE, consistency, observability, mismatch (task #419)

Epic #416. This task adds the relative-state Extended Kalman Filter
(`orbitopt.gnc.ekf`, see [ADR gnc-0002](../adr/gnc-0002-relative-state-ekf-on-the-contract-seam.md)):
a 6-vector LVLH state `(ρ, ρ̇)`, a pluggable process model that reuses the
E12-02 CW and nonlinear pair propagation verbatim, measurement models for the
sensor families E13-03 will provide, and a publishing path onto the E13-01
`Estimate` contract. Acceptance is statistical, so the machine reading is
[`e13_02_relative_nav_ekf.json`](e13_02_relative_nav_ekf.json) — 70 cells ×
5 seeds; reproduce with `PYTHONPATH=src python
benchmarks/bench_e13_02_relative_nav_ekf.py -o …` (≈1 min, CPU only).

## The experiment

Chief at 700 km / 51.6°; deputy at 300 m, either pure radial or pure
along-track, at rest in LVLH. Truth is always the nonlinear two-spacecraft
`propagate_pair` (two-body, J2, or J2 + impulsive differential drag,
`BC_dep = 0.1` m²/kg at ρ₀ = 1.3·10⁻¹¹ kg/m³, H = 50 km); the CW truth family
is `propagate_cw` stepped along the same chief. The filter is CW or
nonlinear(+J2), started at the true state + N(0, diag(30, 30, 30, 0.1, 0.1,
0.1)²) with Q = 10⁻⁹ m²/s³, updated every 60 s for 1 200 s. Five sensor
families: Cartesian position; position+velocity; range/range-rate;
range/azimuth/elevation (0.5 m, 0.5°, 50 µrad); Cartesian+angle stack.
Metrics after a 600 s settle: position/velocity RMSE, per-epoch **NEES**
against the declared χ²(6) 90 % band [1.64, 12.59] and **NIS** per sensor dof,
3σ per-axis coverage (nominal 0.997), and a 20-epoch R-weighted stacked
observability sweep of `W·H_k·Φ(t_k,t₀)`. Clusters report mean/median/q05/q95
over the 5 seeds; 62 declared gates all pass.

## What the numbers say

* **Consistency holds where the model is the truth.** Matched cells with a
  Cartesian update: position RMSE 0.6–0.9 m against a 30 m initial error,
  velocity RMSE 1.2–1.9 mm/s, NEES pass 0.79–0.92 of the χ²(6) band, NIS pass
  0.87–0.93, 3σ coverage 1.000 — and dropping below the declared 0.70/0.98
  gate fails the suite, so this is a tested claim, not a vibe.
* **Consistency is exactly as honest as the model.** CW filter on J2 truth at
  300 m is barely distinguishable (RMSE 0.82 m, NEES 0.83) — the differential
  J₂ there is O(10⁻⁶) m/s² and 60 s updates absorb it. Differential drag is
  the sharp case: RMSE 5.1 m (6×), NEES pass **0.14**, NIS pass 0.43, 3σ
  coverage 0.51/0.36 — the filter reports centimetre covariance while the
  truth leaves the model, which is precisely the failure NEES/NIS exist to
  catch. The nonlinear(+J2) process model does **not** rescue it (0.12): the
  mismatch is the unmodelled force, not the dynamics approximation.
* **Native sensors trade RMSE for linearisation bias.** Range/angle pins
  position to 0.52 m RMSE, but its NEES pass is 0.33 vs 0.92 for the same
  Cartesian cell — with a 30 m prior error the angle Jacobian lies, and the
  filter inherits the lie even though the truth is matched. The iterated/UK
  update is the way back (E13-04); the numbers here are the motivation.
* **The observability sweep explains the weak cells.** Range/rate over a
  20-minute arc is **rank 4/6 with an exact null space** (σ_min = 0): the
  along-track/cross-track pair stays unobserved, and the cell pays 56 m RMSE.
  Range+angle is full rank only through arc curvature (σ_min/σ_max 1.5·10⁻⁴,
  along-track geometry 3.7·10⁻⁵) — it pins position and misses velocity, so
  NEES dies quietly. Cartesian or Cartesian+angle stacks are the robust sets.
* **Geometry is symmetric here, seeds are tight.** Radial and along-track
  cells agree to <5 % in every matched row (no J2 asymmetry at 20 min), and
  per-seed RMSE spread is q05/q95 = 0.51/0.87 m — the estimator, not the
  seed, sets the scale.

## Caveats

Chief ephemeris is exact at every epoch (E13-03's sensor errors, latency and
drops are that task's scope, not this harness's); u = 0 every interval; NEES
gates are declared only for cells whose update fully observes the state
(Cartesian) — the angle cells' numbers are reported and explained via the
sweep, not gated, because failing them is the finding, not a regression.
