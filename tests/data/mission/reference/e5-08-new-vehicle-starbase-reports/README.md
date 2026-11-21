# E5-08c (Starbase) -- what was recorded, and what was not

Flight target: **Starbase** (25.997 N, 97.157 W, `launch_sites.yaml`), 250 km / **30.5 deg**,
launch azimuth 110.8 deg (inside the public 40-115 deg corridor). Every graded constraint is
identical to the Kourou flight targets; only the launch site and the target inclination
differ. Vehicle: `starship_class_rtls_starbase` = the E5-08b (DC7) design with MECO at 120 s
(separation 122 s, on the 2 s trajectory grid), a re-searched ship burn (387.9 s) and post-MECO
pitch, and the launch azimuth. Regenerate with `ORBITOPT_E508_VARIANT=starbase` and
`scripts/e508_new_vehicle.py` (`record`, `crossenv-check`, `e805`, `mc [--pinned]`).

| layer | result |
|---|---|
| recovery leg (both graded checks) | **pass**: landed, 1.70 m/s vertical, 0.007 m/s lateral, 4.56 g, 107.1 t remaining (gate 100 t) |
| joint full-mission gate (cross-env) | **passed**, all five physical checks (a fresh run: the orchestrator is not re-graded from a stale report) |
| ascent limits | pass; peak 2.928 g (3.0), periapsis 232.2 km (230), inclination 30.79 deg |
| ship propellant kept for deorbit/landing | 102.5 t (the E5-07 reference keeps 118.8 t) |
| E8-05 independent dynamics | **27 of 28**; fails `ascent.debris_impact.speed` (12.8 % against a 12 % band) |
| Monte Carlo, pinned script, seeds 8/9/10 | **4 / 36 (11 %)** |
| Monte Carlo, per-sample refine, seeds 8/9/10 | **0 / 36** |

## Honest reading

* The nominal mission closes from Starbase with more ascent margin than the Kourou DC7 design
  (2.93 g against 2.98 g; Starbase's Earth-rotation gain is ~130 m/s larger). It is **not
  robust**: less than the Kourou DC7 design under the same FL-304 dispersions.
* The recovery solution was searched against the canonical separation handoff (the state the
  full mission and the cross-environment recovery leg fly from), under the E5-07 Earth-
  orientation snapshot the cross-environment plan freezes.
* A separation time off the 2 s trajectory grid fails the mission's handoff-continuity checks
  (5.7 m / 11 m/s / 4.1 t across the fork): the stage-1 cutoff had to move with it.
* Found on the way, and fixed for both sites: the record step only filled a `RECIPE` placeholder,
  so a re-record left an earlier script in the mission document and the cross-environment gate
  graded a script other than the one sealed; and `crossenv-check` re-grades an existing report
  rather than re-running the plan. Both are guarded by tests now.
* Nothing graded was relaxed: 100 t reserve, 3 / 1 m/s touchdown, 5.0 g landing limit, the
  250 kPa / 12 g entry limits the vehicle file declares.
* Not a Kourou-to-Starbase isolated comparison: the comparison to the E5-07 reference crosses
  sites (launch geometry, target inclination).
