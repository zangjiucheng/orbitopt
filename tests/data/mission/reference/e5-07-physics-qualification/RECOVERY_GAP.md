# E5-07 recovery-leg gap report (booster RTLS, Starship-class public estimate)

The flight target declares an RTLS recovery with soft-touchdown assumptions.
The vehicle's recovery architecture is two ignitions after separation
(boostback, landing burn) and a purely AERODYNAMIC entry -- `restarts_max: 2`,
no entry burn (the Block 1 booster decelerates with flaps/grid fins, not a
retropropulsion entry burn).

The FL-302 planner `plan_boostback_entry` accepts only entry-burn counts 1 or
2, and requires one ignition for boostback plus one per entry burn (the
landing burn is budgeted separately). Asked about THIS recording's separation
state, it answers:
    entry_burns=0 -> ValueError: entry_burns must be 1 or 2, got 0
    entry_burns=2 -> RecoveryError: boostback + entry burns need 3 ignitions, vehicle allows 2

There is therefore no planner representation of THIS vehicle's RTLS profile:
`entry_burns=0` (the vehicle's actual architecture) is refused by the script
family, and every accepted script family needs more ignitions than the
vehicle has. Running the closest accepted profile (`entry_burns=1`) would fly
an engine architecture the vehicle does not have -- that is a constraint
relaxation dressed as a result, forbidden by the E5-07 acceptance rules.

Consequence, reported not hidden: the mission declares NO recovery stage, the
recovery-phase checks grade `not applicable`, the E5-04 physical gate caps at
`not_checked`, and `full_mission` is not declared anywhere in the E5-07
chain. The booster's un-flown recovery budget is intact and a planner that
represents a restart-2 aerodynamic-entry RTLS can close this leg without
touching any recorded number.

The recorded ascent leaves the booster (dry 275 t plus 466.1 t of unburned propellant -- the whole boostback and landing budget, unspent) on a ballistic arc that ends at t=440.1 s at 1958 m/s, lat 8.563 deg, lon -50.371 deg, 453 km downrange.
