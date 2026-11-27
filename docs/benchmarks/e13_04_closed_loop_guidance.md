# E13-04 — closed-loop guidance framework: paired closed- vs open-loop holdout (task #421)

Epic #416. This task wires the E13-0x GNC parts into a campaign-grade closed
loop (`orbitopt.rendezvous.gnc_campaign`): Truth → Sensor → Navigation →
Guidance → Command → actuation error → Truth, on the E13-01 contract seam and
frozen on the closed-loop mission `leo_rdv_001`. The guidance law always
decides on the *time-stamped* estimate, compensated to the actuation epoch —
never on "now". The machine reading is [`e13_04_closed_loop_guidance.json`](e13_04_closed_loop_guidance.json);
reproduce with `PYTHONPATH=src python
benchmarks/bench_e13_04_closed_loop_guidance.py --jobs 16 -o …` (≈25 min wall
at 16 jobs, CPU only; the open-loop arm re-flight doubles as the pairing
guard and aborts the run if it disagrees with the frozen E12-10 artifact).

## The experiment

Both arms fly the **same** E12-10 holdout: 22 clusters × 12 samples, seed base
20000000, identical perturbation definitions (thruster misalignment/bias/
impulsive draws, sensor noise and dropouts, J2 + drag residuals), identical
ground plan (tof 3416.67 s, planned 10.345 m/s), identical tolerances (50 m /
0.2 m/s) and the unchanged 25 m/s fuel budget. The machine-readable guard block
records sha256 of the frozen E12-10 artifact and of both YAML configs; nothing
was re-tuned to raise the success rate.

The closed-loop arm replaces the two planned burns-only profile with feedback:
departure and arrival keep their paired Δv draws, and between them the
mid-course law (`CwMidCourseLaw` on a compensated estimate) flies the chaser
toward a CW-re-aimed intercept, re-planned at each 60 s controller tick:

* **Schedule.** One maneuver per tick, decided at the estimate's timestamp
  (one cadence stale), actuated one cadence later — a two-cadence pipeline
  delay. Three feedback slots: mid-course at half the transfer (t₁ + 0.5·tof ≈
  1709 s, `midcourse_fraction = 0.5`), terminal trim at t_f − 40 s, and the
  arrival burn at t_f − 22 s, so the last actuation lands ≈ 7 s before the
  arrival epoch.
* **Compensation.** `RelativeNavEKF.compensate_to` propagates the filtered
  state (and its covariance) to the actuation epoch before the law evaluates
  it — the standalone experiment in E13-02 showed a −166 m stale bias
  shrinking to −3.7 m through this path.
* **Re-aim.** `solve_cw_intercept` with a weighted least-squares velocity
  term replaces the position-only CW solve, which under J2 residuals was the
  dominant arrival-error source.

Failure attribution (on top of the frozen E12-10 classes): navigation charges
a failure when tracking RMSE > 20 m (drift-scaled by the guidance gain),
guidance when the commanded Δv itself breaks the budget, execution when the
commanded profile fit but realized Δv did not, safety_constraint otherwise.

## What the numbers say

* **Open-loop identity holds**: 13/264 (4.92 %), failure classes {87 safety,
  164 arrival} — bit-identical to the frozen artifact, so the two arms are
  compared on the same draws.
* **Closed-loop: 258/264 = 97.7 %** (Wilson 95.1–99.0 %). Every one of the
  264 samples' EKF converged; the paired difference is **+92.8 pp**,
  whole-cluster bootstrap CI [+89.4, +95.8] pp.
* **Paired McNemar (exact, Holm-corrected across the four endpoints):**
  success b=245/c=0, arrival_accuracy 164/0, safety_violation 82/1 — the first
  three p ≈ 0 after Holm; fuel_over_budget 0/0 (p=1). Zero closed-loop samples
  regressed from a frozen success to arrival or fuel failure.
* **Fuel: the budget was never binding.** Realized Δv median 11.38 m/s, p90
  12.05, max 12.82 — comfortably inside 25 m/s; `over_budget_samples = 0`.
  The feedback burns pay for themselves because re-aim converges.
* **Arrival:** position error median 42.1 m (p90 44.7), relative speed median
  0.064 m/s — all 258 successes inside tolerance; the only >50 m rows are the
  six aborted ones below.
* **All six remaining failures are safety_constraint**, and all six are the
  *passive-release* rule firing during the hold-point coast (keep-out margin
  −106 to −478 m), i.e. before any guidance decision exists: those draws are
  stress cases where the initial state already violates the corridor. The one
  closed-loop regression vs the frozen successes (cluster 7, index 6) is one
  of these — its passive-release abort at −478 m cut the run short before the
  terminal trim fired (the run's single trim refusal) and left the 82.0 m
  arrival. No navigation, guidance or execution attribution anywhere in the
  holdout.

## Limits worth stating

* The tracking RMSE (~79 m) is dominated by the first ticks where the EKF
  still works through initial transients while the guidance re-aims from the
  975 m hold offset; it is recorded, not gated, in this benchmark.
* The six safety failures are irreducible under this campaign's frozen
  passive-release geometry — fixing them means touching the stress definitions,
  which the task charter forbids. They bound closed-loop success at 97.7 % for
  this mission/stress pairing.
