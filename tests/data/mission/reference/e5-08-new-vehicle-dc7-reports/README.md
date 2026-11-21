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
| E8-05 independent dynamics | **27 of 28**; fails `ascent.seco.orbit_semi_major_axis` (27.7 km vs a 15 km band) |
| Monte Carlo, pinned script, seeds 8/9/10 | **11 / 36 (31 %)** |
| Monte Carlo, per-sample refine, seeds 8/9/10 | **3 / 36 (8 %)** |

## Honest reading

* The nominal mission closes. It is **not robust**: under the FL-304 standard dispersions most
  samples miss a corridor condition (CORRIDOR_MISSED). 31 % is the unbiased pinned figure; the
  first pick on seeds 5/6/7 read 42 %, the difference being selection on the exam seeds.
* Optimising the script on an 8-sample dispersed ensemble (seed 101) *overfitted*: 1-4 of 36 on
  the hold-out seeds, worse than the nominal pick. Not used.
* E8-05: the SMA band (15 km) is stated as "the SECO velocity band seen through vis-viva"
  (~1.75 km per m/s), which for the 20 m/s velocity band would be ~35 km; this design's SECO
  velocity discrepancy is 15.1 m/s (inside its band). The band was NOT loosened.
* The first DC7 search produced a post-MECO pitch program with a 9 deg bump (11.8 deg between
  5.8 and 2.5 deg). Two independent models disagreed about it (E8-05 failed 2 more checks) and
  it was rejected: the final program never pitches back up after MECO.
* The recovery solution was searched against the canonical separation handoff (t = 124 s), the
  state the full mission and the cross-environment recovery leg fly from, not the plain read-out
  (t = 126 s): a script that closes from the latter did not close in the mission.
* Nothing graded was relaxed: 100 t reserve, 3 / 1 m/s touchdown, 5.0 g landing limit, the
  250 kPa / 12 g entry limits the vehicle file declares.
