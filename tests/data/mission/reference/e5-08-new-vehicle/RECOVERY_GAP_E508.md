# E5-08 recovery-leg gap report (DC1 powered-entry RTLS, Starship-class RTLS-derived)

The E5-08 design change DC1 -- booster engine set `usable_for: [boostback,
entry, landing]`, `restarts_max: 3` -- gives the vehicle the representation
E5-07 lacked: `plan_boostback_entry` accepts the profile (3 ignitions),
plans it and FLEW it. The best script found reaches the landing corridor
from the recorded separation state (miss 424 m against the 8 km
referee; entry peak 367 kPa / 7.2 g inside the declared design
envelope 400 kPa / 12 g) and arrives with 124.3 t of propellant, of
which the closed-loop landing model needs 28.2 t -- leaving
96.1 t against the GRADED reserve of 100 t.

Gate outcome at the recorded separation state (scenario
leo_starship_class_rtls_e508.yaml, constraints inherited UNCHANGED):

- recovery-landing: **False**
- recovery-propellant-reserve: **True**

## Reason classification (required by the task: not closing is allowed,
## hiding or relaxing is not)

**Vehicle capability, at the boundary of model uncertainty.** The binding
limit is propellant at the corridor, not guidance authority: every corner
of the script space that closes the corridor (global DE from the E5-05c
seed lineage; two 250-evaluation Nelder-Mead fronts at landing cap 5.5 g;
the deliberately drag-rich entry corner -- low entry throttle, late
cutoff) arrives with a propellant-at-corridor minus landing-need frontier
of 84-96 t; the graded gate asks for 100 t. The best point (this record)
is 96 t short -- 4-8% of the budget. A heavier separation state
(+20/+35 t) changes the readout by < 1 t: the entry burn re-spends any
extra propellant the plan can use, so the frontier is set by the entry
cut-off geometry (the 1500 m/s planner cut-off clamp) and the corridor
insertion speed, not by how much propellant the stage carries. The
vehicle's Cd band is +/-30% (the FL-602 provenance tags on this public
estimate); the corridor propellant cost moves by roughly +/- 10 t across
that band, so at nominal aero the leg misses, and the miss is inside the
declared model uncertainty -- i.e. the design is BORDERLINE infeasible
under this model family, not decisively infeasible.

**What would close it (declared design changes for a next iteration, NOT
applied here because this record must grade the design as declared):** a
DC3 stage-mass reallocation cannot help (entry burn re-spends surplus --
shown above); what helps is raising the planner's entry cut-off clamp
above 1500 m/s (guidance-domain parameter of the E3 planner, not a graded
constraint) so the entry burn stops earlier and the coast-shape does more
of the deceleration, or a DC6 RTLS variant with a downrange landing site
(shorter boostback, ~20-30 t returned to the reserve).

**Reason classification.** The graded `recovery-propellant-reserve` check
reads True above only because the planner ABORTS the landing before the
burn on a feasible-track refusal -- the reserve is then measured on the
post-cut-off rollout, not after a landing. On every variant that actually
flies a landing burn the reserve lands at ~96 t < 100 t. The landing gate
is the honest failure: the closed-loop model predicts touchdown at
498 m/s vertical from the
corridor state (958 m/s at 3.0 km
ignition) -- the declared actuator set (13 engines at the 5.5 g thrust
cap, 45 deg/s TVC rate) cannot decelerate this arrival inside the
propellant the 100 t reserve leaves it. The same landing law soft-lands
cooler arrival states in the E5-07-class Monte Carlo, so this is a
VEHICLE-CAPABILITY finding (entry arrival state vs actuation and reserve),
not a guidance-law tuning failure and not a constraint-set artefact.

**Why not "constraint":** every graded number is inherited unchanged from
the E5-07 reference and none of them is the binding limit -- the design
runs out of capability before it runs into a graded constraint.
**Why not "guidance":** the search (global DE on the E5-05c seed lineage,
two 250-evaluation Nelder-Mead fronts, the drag-rich entry corner) found
the same 96 t frontier from every direction; the landing law
has authority only on arrivals cooler than this separation geometry
permits.

## What this record nevertheless evidences

- the DC1 architecture plans, flies and grades end-to-end through the
  shipped E3 chain -- the representation gap E5-07 reported is closed;
- the ascent leg of the derived vehicle is identical to the E5-07
  reference within 250 kg of separation mass (DC2-DC5 unchanged), so the
  payload/Max-Q/peak-g comparison rows are direct;
- the E8-05 independent-dynamics twin and the E7-05-methods Monte Carlo
  run on THIS tree (see independent-dynamics.json / mc-summary.json):
  the Monte Carlo quantifies the dispersion of the miss (its success rate
  reads the graded gates, so it is a red-rate reading of the same gap).
