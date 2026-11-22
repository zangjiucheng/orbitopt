# E10-07 — Full recovery independent re-flight (task #304)

E8-05's independent landing evidence starts from a **declared design
corridor** ([E5-08 report](e5-08-new-vehicle-report.md)), so it never
re-flew an actual recovery trajectory. That check is a `design_input`: it
cannot be read as independent physical verification of the complete RTLS
chain. E10-07 re-flies the **whole chain from the actual separation
handoff** — boostback, the E10-01 unpowered aerodynamic entry, and the
landing phase — under an independently written twin model, and grades the
discrepancies against the main FL-301 model inside measured bands with an
error-budget split.

## Start condition: the actual handoff, not a corridor

The start state is the DC8 hot-staging separation state (E10-02 timeline)
flown fresh through the OpenTsiolkovsky OT plant (run `ot-ec11389b-da42dd33`,
sealed in the committed report): separation at **t = 125.0 s**, mass
**900 172.42 kg**, speed **1 569.85 m/s**, ECEF state pinned by SHA-256
handoff digest
`60b3130118befaba86d8806d8f539adb6de4e1099ef129104934d52f768ab99c`.
The fast gate recomputes the digest from the recorded state and cross-checks
the OT event times against `benchmarks/e10_02_hot_staging.json`. No
declared-corridor state enters the re-flight.

## The twin

`orbitopt.verify.full_recovery_reflight` re-flies the *identical authored
`_Mission` script* (the sealed E10-01 keep vector `[21.2, -2.0, 1.0, 0, 0]`,
IIP cutoff, unpowered entry, closed-loop landing-need model) through an
independently written point-mass twin:

* own RK4 integrator (nominal 0.1 s, halved for the numerical budget line);
* own WGS-84 geodesy (fixed-point latitude iteration);
* own gravity: point mass + J2 in ECEF with Coriolis and centrifugal terms;
* textbook US76 layers on geopotential altitude with an isothermal
  stratopause continuation (independent of the main model's fitted `entry`
  exponential profile);
* physical-thrust replay from the vehicle file (count · level · vacuum-minus-
  ambient thrust, ambient-corrected Isp), Cd from the vehicle polar.

The controller line is declared zero (same script); the frame line is
declared zero (same ECEF frame and epoch).

## Results (report kind `full-recovery-reflight`, all checks PASS)

| Quantity | Main (FL-301) | Twin | Δ | Band |
| --- | --- | --- | --- | --- |
| Cutoff time (s) | 182.25 | 182.50 | 0.25 | 0.6 s |
| Cutoff position | — | — | 194.2 m | 600 m |
| Cutoff speed (m/s) | — | — | 3.50 | 6 |
| Entry peak-q (kPa) | 459.8 | 437.7 | 4.81 % | 12 % |
| Entry peak proper-g | 6.31 | 6.04 | 0.28 g | 1.0 g |
| Impact time (s) | 400.225 | 400.985 | 0.76 | 1.0 s |
| Impact position | — | — | 543.9 m | 1500 m |
| Impact speed (m/s) | 892.2 | 814.3 | 8.73 % | 12 % |
| Propellant at cutoff (kg) | 300 652 | 301 105 | 453 | 1200 |

The impact-speed check is relative (%), matching the E8-05 ascent
debris-impact precedent; the terminal dive speed is density-profile sensitive.

### Error-budget split

* **frame** — declared zero (same frame, epoch, rotation rate).
* **controller** — declared zero (identical authored `_Mission` script).
* **numerical** — twin at dt/2: ≈0 on q, 0.08 m/s on impact speed, 116 m on
  impact position.
* **gravity** — twin without J2: 0.31 s impact time, 111 m position.
* **atmosphere (form)** — twin re-run on the main model's `orbitopt.entry`
  density profile (a labelled perturbation experiment): **5.1 %** of the
  4.8 % peak-q delta and **8.7 %** of the 8.7 % impact-speed delta — i.e.
  the two density profiles account for the *entire* discrepancy in those
  checks; the gated residual is **0.0** for both.

## Honest verdict

The main model itself is **not closed** at this handoff: `plan_boostback_entry`
returns `CORRIDOR_MISSED` + `ENTRY_Q_LIMIT_EXCEEDED` (corridor reached with a
2 246 m miss at 1 067.5 m/s), consistent with the E10-01 VEHICLE_CAPABILITY
finding. The landing burn is **unplannable** from the real crossed handoff
(`fly_landing_burn` probe: `LANDING_INFEASIBLE`; closed-loop touchdown would
be 740 m/s vertical), so the terminal comparison is the ballistic impact —
and the report says so. The independent re-flight certifies the **chain
physics**, not a closed design; the corridor-start landing check remains
`design_input` and is superseded for full-chain purposes by this report.
All nine checks carry `independent_model` evidence.

## Reproduce

```bash
python benchmarks/bench_e10_07_full_recovery_reflight.py \
    --out docs/benchmarks/e10_07_full_recovery_reflight.json   # ~6 min
PYTHONPATH=src python -m pytest tests/test_e10_07_full_recovery_reflight.py -q
```
