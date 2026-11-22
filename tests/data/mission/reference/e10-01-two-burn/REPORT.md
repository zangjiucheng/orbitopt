# E10-01 — two-burn RTLS recovery: boostback → unpowered entry → landing

Record for task #298. Toolchain @ dd24361 + the `entry_burns = 0` script family
(`src/orbitopt/flightlab/recovery/boostback.py`). Machine-readable results:
`two-burn-outcome.json`.

## What changed in the chain

`entry_burns = 0` is now a first-class script: **5 parameters** (boostback
start delay, iip aim, throttle, pitch, yaw trim), **no entry-burn vector**,
`ReturnPlan.entry_burn is None`, `ignitions_used == 1` on the return track and
`ignitions_needed == 1` at the planner gate (the landing burn is budgeted on
top, exactly as for 1/2-burn scripts). The FL-301 propagator coasts the whole
entry phase; the entry q (250 kPa) / g (12) window is graded from the boostback
cutoff to the corridor crossing just as with a powered entry — unpowered entry
is **checked, not skipped**. `fly_recovery`, the R-02 handoff report and the
verification gates speak it unchanged (`plan_options={"entry_burns": 0, ...}`).
`entry_throttle=True` with 0 burns is a `ValueError`; the speed-cutoff entry
trigger still requires exactly 1 burn.

The E5-07 gap statement (`RECOVERY_GAP.md`: "no planner representation" for the
reference vehicle's boostback + aerodynamic + landing architecture) is closed:
the reference vehicle's profile now plans on its own separation state
(`tests/test_e1001_two_burn_entry.py::test_e507_reference_vehicle_profile_now_plans`).
The gap file itself is a historical record and is kept verbatim.

## Closure verdict: NOT CLOSED — classified VEHICLE_CAPABILITY (guidance secondary)

On the sealed E5-08 separations, with the sealed grading recipe unchanged
(iip cutoff, closed-loop need model, 250 kPa / 12 g, 100 t reserve + 2.5 t
margin, 3/1 m/s touchdown, 4.0 g landing law), no two-burn script closes:

| vehicle | script | verdict | arrival v | entry q_pk | entry g_pk | prop. at corridor | pred. touchdown |
|---|---|---|---|---|---|---|---|
| DC7 | keep `[21.2, -2.0 km, 1.0, 0, 0]` | CORRIDOR_MISSED + ENTRY_Q_LIMIT_EXCEEDED | 1065 m/s | 454 kPa | 5.9 | 332.7 t | 745 m/s |
| DC7 | Nelder–Mead from the sealed seed | same | 1053 m/s | 454 kPa | 7.8 | 184.0 t | — |
| DC7 | global DE, box delay 0–150 s / aim −20…+80 km / throttle 0.4–1 / pitch 0–30° / yaw ±45° | best penalty 58.99, CORRIDOR_MISSED + INSUFFICIENT_PROPELLANT_RESERVE | 1125 m/s (miss 139 km) | — | — | 99.5 t | — |
| Starbase | keep `[12, +2 km, 0.9, 7°, -4°]` | CORRIDOR_MISSED + ENTRY_Q_LIMIT_EXCEEDED | 1123 m/s | 494 kPa | 5.8 | 389.7 t | 833 m/s |

**Category: vehicle capability.** The modelled booster aero is an
attitude-independent Cd(M) table (blunt cylinder, "grid fins stowed", 0.8–1.3)
flown in a point-mass propagator: the unpowered phase has no lift, flap or
bank authority, so entry deceleration comes only from the ballistic path.
With the graded 100 t reserve + landing need pinning corridor mass ≥ ~400 t,
terminal velocity near 3 km is ~330–490 m/s and a ballistic arrival overshoots
it at ~1100 m/s while exceeding 250 kPa at ~10 km — the corridor is missed and
the 4.0 g closed-loop landing forecast rejects the handover. The lightest state
legal under the reserve gate (dry + 100 t) still arrives above the closed-loop
landing envelope, so a lighter derived variant does not close the gap either.

**Category: guidance (secondary).** A lifting entry with bank steering could
spend the 12 g / 250 kPa budget over a long glide; the model gives the planner
no unpowered control to plan with.

## Comparison with the sealed three-burn DC7 record

The unpowered profile keeps **+172.6 t** in the tanks at the corridor
(332.7 t vs 160.1 t sealed) and a reserve margin of +202.7 t vs +10.1 t —
i.e. the entry-burn propellant is entirely "saved" — yet the mission fails:
the saved mass stays as ballast (arrival 1065 vs 257.2 m/s, q_pk 454 vs
117.8 kPa, miss 2439 vs 55.2 m, g_pk 5.9 vs 7.31). The powered entry pays
propellant precisely where the atmosphere cannot help (Mach > 2 above ~40 km),
which the "fins-stowed" aero cannot do unpowered. No two-burn vehicle variant
is recorded; the three-burn sealed records stand.

## Reproduce

```
PYTHONPATH=src python -m pytest tests/test_e1001_two_burn_entry.py -q
```

(9 tests: layout, validation, single-burn coast flight, q/g grading of the
unpowered window, `mission_command` re-fly, restart-2 clamping, E5-07
representability, `fly_recovery` chain e2e, DC7 classification.)
Cross-environment E5-06 joint physics on the two-burn handover is moot while
the corridor verdict rejects the handover; it applies to the sealed three-burn
records, which are unchanged and still pass
(`scripts/e508_new_vehicle.py crossenv-check`, tudatpy host).
