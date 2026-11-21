# E5-08 new-vehicle design & qualification report

Recorded commit `cd6572eead11be5ad2f1764518e30b01ef5a7209`; toolchain python 3.13.18, numpy 2.5.5, scipy 1.18.1, gfold 0.3.2.

The vehicle: `starship_class_rtls_derived` -- the E5-07 reference stack with the DC1 powered-entry RTLS recovery scheme (three ignitions: boostback, one throttled entry burn, landing burn). Constraints inherited from the E5-07 target UNCHANGED (max-q 55 kPa, 3.0 g, 230 km floor, 250 km insertion band +/- 60 km, RTLS touchdown 3/1 m/s, reserve >= 100 t). Design changes and their reasons are declared in the vehicle file; every non-design number carries its source there.

## Comparison against the E5-07 reference

```json
{
  "ascent": {
    "max_q_pa": {
      "reference": 37479.27754272647,
      "derived": 37479.27754272647,
      "delta": "0 by construction (DC2-DC5 unchanged)"
    },
    "peak_g": {
      "reference": 2.8875404152803634,
      "derived": 2.8875404152803634,
      "delta": "0 by construction (DC2-DC5 unchanged)"
    }
  },
  "payload_capability": {
    "delivered_orbit": "identical insertion (250 km class, same band, same epoch): the ascent is byte-for-byte the reference's (DC2/DC3 unchanged), so payload capability is UNCHANGED BY DESIGN",
    "ship_propellant_margin": "unchanged (~119 t kept, stage-2 cutoff 378 s)"
  },
  "propellant_margin": {
    "booster_sep_propellant_kg": null,
    "touchdown_propellant_kg": 96128.53753693955,
    "graded_reserve_kg": 100000,
    "attribution": "DC1 buys a FLOWN-to-the-corridor recovery leg at the price of the entry-burn propellant: propellant at the corridor minus the landing model's need is 96.1 t against the inherited 100 t reserve gate -- the gate is NOT met on any landing-flying variant of this design (see RECOVERY_GAP_E508.md for the classification)"
  },
  "recovery": {
    "reference": "NOT FLOWN (RECOVERY_GAP.md: restart-2 aerodynamic entry has no planner representation)",
    "derived": "FLOWN (DC1) to the landing corridor -- the corridor IS reached and the leg is graded (not the E5-07 no-representation outcome); soft landing NOT achieved, see RECOVERY_GAP_E508.md",
    "touchdown": {
      "propellant_remaining_kg": 96128.53753693955,
      "max_proper_g": 7.2433054567986845
    }
  },
  "model_uncertainty": "aero tables +-30 % (nose_aero band, inherited): recorded max-q 37.5 kPa -> 48.75 kPa at +30 %, still inside the 55 kPa ceiling; engine/Isp bands enter the recovery leg as the FL-304 MC dispersion (see mc-summary.json)"
}
```

## E8-05 independent-dynamics validation (new vehicle)

Verdict **passed** (27/27 independent; 28/28 total), independent native point-mass ascent twin + two-model powered-landing refly on the DECLARED design corridor in the 3-engine landing configuration (the vehicle file records the 13-engine set cannot throttle below weight; inherited limitation, DC1 unchanged). The mission's ACTUAL corridor state (recorded engines=13, landing burn aborted by the planner) is independently UNPLANNABLE for a soft landing from here -- the two models agree with the E3 planner, corroborating the landing gap below.

## E7-05-methods recovery Monte Carlo

Success rate **0.000** (0/30), whole-seed cluster bootstrap 95% CI [0.0, 0.0]. Seeds [5, 6, 7] are hold-out (disjoint from the search seed 287); per-seed rates: seed 5: 0.00, seed 6: 0.00, seed 7: 0.00

## Reason classification of the unclosed fraction

Failure classes per seed are in `tests/data/mission/reference/e5-08-new-vehicle-reports/mc-summary.json`; classified as vehicle capability / guidance / constraint in the per-sample reason codes.

## Cross-environment joint full-mission gate

Two-leg coupled run across the tudatpy (ascent + parking propagation) and gfold (recovery) environments, graded as ONE joint gate. Verdict **failed**, full-mission declared **True**, physical **failed**; per-phase physical: orbit_insertion=passed, parking_propagation=passed, landing=failed, mass_conservation=passed, epoch_frame_continuity=passed. The recovery phase carries the DC1 powered-entry scheme's honest result (see the gap classification below): the ascent and parking phases pass the inherited physical checks in the tudatpy environment, and the recovery leg is flown and graded across the boundary rather than assumed.

Reason codes: CORRIDOR_MISSED, INSUFFICIENT_PROPELLANT_RESERVE, LANDING_LATERAL_SPEED_EXCEEDED, LANDING_POSITION_OUT_OF_TOLERANCE, LANDING_VERTICAL_SPEED_EXCEEDED, PROPELLANT_RESERVE_LOW

## Recovery gap: classification (E5-07 pattern -- reported, not relaxed)

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

