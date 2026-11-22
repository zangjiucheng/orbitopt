# E5-08b (DC7) -- what was recorded, and what was not

Flight target: **Kourou**, 250 km / 51.6 deg, every constraint identical to the E5-07 /
E5-08 flight target. Vehicle: `starship_class_rtls_dc7` = the E5-08 design with the stage-1
cutoff 12 s earlier (DC7). Regenerate with `ORBITOPT_E508_VARIANT=dc7` and
`scripts/e508_new_vehicle.py` (`record`, `crossenv-check`, `e805`, `mc [--pinned]`).

| layer | result |
|---|---|
| recovery leg (both graded checks) | **pass**: landed, 1.5 m/s vertical, 0.003 m/s lateral, 4.59 g, 113.1 t remaining (gate 100 t) |
| joint full-mission gate (cross-env) | **passed**, all five physical checks |
| ascent limits (3.0 g, max-Q 55 kPa, insertion band) | pass; peak 2.976 g, periapsis 232.0 km -- thin margins |
| ship propellant kept for deorbit/landing | 101.2 t (the E5-07 reference keeps 118.8 t): the price of DC7 |
| E8-05 independent dynamics | **28 of 28**; E10-04 closed the former `ascent.seco.orbit_semi_major_axis` fail (27.7 km) at ~5.9 km, band unchanged |
| Monte Carlo, pinned script, seeds 8/9/10 | **11 / 36 (31 %)** |
| Monte Carlo, per-sample refine, seeds 8/9/10 | **3 / 36 (8 %)** |

## Honest reading

* The nominal mission closes. It is **not robust**: under the FL-304 standard dispersions most
  samples miss a corridor condition (CORRIDOR_MISSED). 31 % is the unbiased pinned figure; the
  first pick on seeds 5/6/7 read 42 %, the difference being selection on the exam seeds.
* Optimising the script on an 8-sample dispersed ensemble (seed 101) *overfitted*: 1-4 of 36 on
  the hold-out seeds, worse than the nominal pick. Not used.
* E8-05 (E10-04 closure): this design once failed the SMA band at 27.7 km against 15 km. The
  discrepancy lived in the comparison, not the design: the raw 2-s history was chorded linearly
  across the MECO (123.0 s) and SECO (514.663 s) mid-segment engine cutoffs, a ~13 m/s / ~22 km
  artifact at both events. Reconstructed from the left-segment recorded accelerations the SECO
  discrepancy is ~5.9 km (the velocity discrepancy ~5.2 m/s, inside its 20 m/s band) and every
  check passes. The band was NOT loosened; the fix is in `orbitopt.verify.independent_dynamics`
  with regression tests in `tests/test_e805_independent_dynamics.py`.
* The first DC7 search produced a post-MECO pitch program with a 9 deg bump (11.8 deg between
  5.8 and 2.5 deg). Two independent models disagreed about it (E8-05 failed 2 more checks) and
  it was rejected: the final program never pitches back up after MECO.
* The recovery solution was searched against the canonical separation handoff (t = 124 s), the
  state the full mission and the cross-environment recovery leg fly from, not the plain read-out
  (t = 126 s): a script that closes from the latter did not close in the mission.
* Nothing graded was relaxed: 100 t reserve, 3 / 1 m/s touchdown, 5.0 g landing limit, the
  250 kPa / 12 g entry limits the vehicle file declares.
