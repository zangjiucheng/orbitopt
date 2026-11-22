# ADR flightlab-0011: The upper-stage guidance envelope (E6-03)

- **Status:** Accepted
- **Date:** 2026-11-16
- **Deciders:** E6 6-DOF Production Maturity (Epic #265), E6-03 (#275)
- **Code:** `src/orbitopt/flightlab/sixdof/guidance_matrix.py`,
  `benchmarks/bench_e6_03_guidance_matrix.py`
- **Evidence:** `docs/benchmarks/e6_03_guidance_matrix.json`,
  `tests/test_flightlab_e6_03_guidance_matrix.py`
- **Builds on:** ADR flightlab-0002 (closed-loop two-stage mission), ADR
  flightlab-0006 (rate-limited landing guidance -- the servo model this matrix
  reuses), E1-03 / F-07 (the shared upper-stage law and its single disclosed
  scenario point)

## Context and problem

E1-03 cut the 6-DOF/3-DOF upper-stage altitude difference from 235 km to
1.65 km, and disclosed that the result was obtained on a **non-parity variant**
(1 t upper dry mass, 7 t propellant, 3 deg gimbal, one 220 km target, the F-06
pad on heading 090). One scenario point says the law is *shared*; it does not
say where the law works, and three questions were left open:

* **Is the result target-dependent?** 220 km was the only declared orbit.
* **Is it inclination-dependent?** No inclination was ever declared -- the
  28.3 deg the F-06 pad inherits was simply what the ascent realised.
* **Is it the law, the vehicle or the servo?** When a cell misses, "the guidance
  is not accurate" is not a finding; a 6.5 t tank that runs dry, a 0.02 rad/s
  servo that cannot track, and a law that cannot terminate all look the same in
  a single missed-tolerance boolean.

E6-03's acceptance is a matrix over all three axes with the cause of every miss
named, including the ORIGINAL F-06 vehicle that has no TVC axis at all.

## Decision

Ship the matrix as a **declared artefact with a rule-based attribution**, not
as a pile of runs:

* **One law, one runner.** Every guided cell flies the *identical*
  `UpperStageGuidanceSpec` through the packaged `run_mission`; only the declared
  `ParkingOrbitTarget` (altitude, inclination, tolerances) and the cell's
  vehicle/servo change. No per-cell gain tuning: tuning per point would measure
  the tuning, not the law.
* **Guided cells hold the stack wet mass at the parity 78 t and the upper-stage
  wet mass at the parity 8 t**, so the tank *split* is the payload axis. The
  stage-1 vehicle, the attitude program and the separation bookkeeping are the
  same flight in every guided cell.
* **Inclination is realised by launch latitude on a due-east heading.** The
  native plant steers with one TVC axis and no bank -- a probe at 45 deg with a
  dog-leg heading put the rigid body in a 179 x 7 567 km orbit, so dog-leg is
  not available to this plant and would not be an honest test of it. The graded
  residual against the declared inclination is the geodetic-to-geocentric
  latitude conversion of the pad (~0.19 deg, well inside the declared 1 deg
  band), which is recorded per cell rather than hidden.
* **The actuator axis is applied twice on purpose**: the cell's travel limit
  sets BOTH `RigidBodyEngine.gimbal_limit_rad` (the plant) and
  `TvcActuator.limit_rad` (the servo). One axis, not two knobs that can disagree.
* **A miss is attributed by rule over recorded observables**, in this order:
  no TVC travel at all -> *vehicle capability*; exactly one backend inserted ->
  *actuator constraint* (the actuator-free point-mass twin is the control);
  every backend ran dry with the terminal orbit **tighter** than the target
  circle -> *vehicle capability*; anything else -> *guidance*. The discriminator
  between the last two is the terminal orbit's **energy**
  (`a_terminal - r_target`), not the propellant boolean: a throttle trace that
  closes at the end of a dry tank is indistinguishable from a commanded cutoff,
  so each record also carries `burn_ended_by` in
  `{no-law, still-burning, depletion, law-cutoff}`.

## Measured (17 cells, dt = 0.05 s, both native backends each)

6 cells insert, 11 miss (7 guidance, 3 vehicle capability, 1 actuator
constraint); see `docs/benchmarks/e6_03_guidance_matrix.json`.

* **Inclination axis -- no dependence found.** At 220 km the stage inserts at
  28.5, 45.0 and 51.6 deg with periapsis errors of +5.3/+0.5, +2.6/+2.4 and
  +0.8/+0.9 km (3-DOF/6-DOF) and a cross-backend periapsis difference of
  0.09-4.84 km. The E1-03 result is not a pad-latitude accident.
* **Altitude axis -- bounded at BOTH ends, by the criterion, not the tank.** At
  180 km the cutoff fires with 478/484 kg left and periapsis inside the band
  while apoapsis sits +52.8/+50.8 km outside it: the criterion grades
  periapsis and apoapsis is whatever follows. From 260 km up the osculating
  periapsis never closes on target, the stage burns to depletion and
  over-shoots (+5 647 to +7 096 km apoapsis). The tank is exonerated by the
  stretched-tank cell: 200 kg MORE propellant at 300 km makes the miss worse
  (+6 862 -> +17 763 km apoapsis), which only a termination failure does.
* **Payload axis -- the heavy stage is a vehicle limit.** 1.5 t dry / 6.5 t
  propellant runs dry with its terminal orbit 23-88 km *tighter* than the target
  circle at 180 and 220 km (5 581 m/s ideal budget against the disclosed
  variant's 6 933 m/s), while the 0.8 t / 7.2 t variant inserts at 220 km
  bit-identically to the disclosed one.
* **Actuator axis -- one constraint failure, cleanly separated.** The derated
  (2 deg / 0.05 rad/s / 0.15 s) and wide (5 deg / 0.15 rad/s / 0.08 s) sets
  insert. At 2 deg / **0.02 rad/s** / 0.25 s the point-mass twin still inserts
  (+5.3/+8.5 km) while the rigid body lands with periapsis 6 217 km under the
  target orbit and inclination 41.6 deg -- the servo, not the law, and not
  enough propellant to blame.
* **The original F-06 stage** (2 t / 6 t, zero gimbal travel) cannot have the
  law instantiated at all; open loop on a 4 620 m/s budget its periapsis lands
  3 082 km (3-DOF) / 2 718 km (6-DOF) low, and the two backends disagree by
  365 km -- the pre-E1-03 baseline, recorded so the matrix is not all good news.

## Consequences

* The E1-03 claim is restated with its envelope: the shared law inserts this
  stage at 220 km (and at every declared inclination) inside +/-25 km / 1 deg;
  it does not insert at 180 km, and does not terminate above ~240 km. Both
  bounds are the **cutoff criterion's**, not the vehicle's -- the next guidance
  task should add an energy/apoapsis termination term, not a bigger tank.
* `docs/flightlab.md` (E6-03) reports the matrix and links the study; the F-07
  block's "shared law does not mean general accuracy" is now measured rather
  than asserted.
* The matrix is a benchmark, not a unit test: 17 cells x 2 backends is ~15 min
  of CPU. The test module locks the committed study (schema, coverage,
  declared verdicts, attribution consistency, the E1-03 point's band) and
  re-flies exactly one cell under `slow`.
* `burn_ended_by` is now part of the record vocabulary: any future miss analysis
  must distinguish a commanded cutoff from a dry tank before naming a cause.

## Alternatives considered

* **Tune the law per scenario point** (gravity-term gain, command-alpha limit)
  until every cell inserts: tried at 300 km -- `gravity_term_gain=1.0` and
  `command_alpha_limit_deg=15` both made the over-shoot worse (+18 266 and
  +36 423 km apoapsis). Rejected: the acceptance asks what the shipped law does.
* **Declare inclination by launch azimuth** with a dog-leg: rejected as above
  (the plant cannot fly it; the result would be a steering-program artefact).
* **Grade only periapsis, as the cutoff does**: rejected -- it is exactly the
  blindness that makes the 180 km cell miss. Periapsis, apoapsis and inclination
  are all graded against declared tolerances.
* **Attribute a dry tank to the vehicle**: rejected, because a law that cannot
  terminate also empties the tank. The terminal orbit's energy separates them.

## Follow-through: E6-05 (task #292)

The consequences asked the next guidance task for "an energy/apoapsis
termination term". It exists now, and the envelope did not move:

* The cutoff is a two-sided insertion box (periapsis *and* apoapsis inside the
  declared band) instead of the apoapsis-margin proxy, with a transfer-apogee
  planning guard of one declared tolerance (25 km) that exists only to keep the
  guidance target finite when the box sits under the current orbit, and an
  apogee-ceiling hold that trims tangential command once the box is armed.
  The hold is off-by-default (`apogee_ceiling_gain`); the matrix flies 8e-4.
* The matrix was re-flown on the generalised law: the same six insert, the same
  eleven miss, zero unexpected. The law change is verdict-neutral on every cell.
* The seven guidance-class misses split into two mechanisms, both measured:
  **six** deplete with the terminal orbit *wider* than target and periapsis
  still short of the band floor while the tank empties — the box never arms,
  and no cutoff, cap, feed or hold can arm it; their binding limit is the
  stage's delta-v budget, and re-labelling them "vehicle-capability" would
  contradict the energy rule this ADR's last alternative already rejected.
  **One** (a180-i28.5-V1-A1) is the overshoot this ADR diagnosed: the box
  *would* arm, but only while the apoapsis is already past the ceiling; the
  apogee hold's retro correction is physically unrealisable inside the
  ±30° velocity-frame command clamp (any command inside 30° of the velocity
  vector is ≥ 87% prograde), so the perigee passage still prints 30–70 km
  above the ceiling.
* Widening that clamp to 45° converts the a180 cell on both backends
  (-21/+17 km, 517 kg to spare) — and flips a220-i45 and a220-i51.6 from
  insert to depletion 3–6 km *under* the periapsis floor, with the a220
  reference cell's own periapsis margin down to ~1 km. One shared law cannot
  buy one cell with two others; 30° stays, and the rejection is recorded as
  the envelope's boundary. Conversion of the overshoot class needs a law that
  commands attitude beyond the airframe's realisable AoA window (a
  pitch-reversal / braking-phasing regime), not a wider limiter.
* The same cell set runs on the JSBSim channel (`sixdof/jsbsim_matrix.py`,
  record `benchmarks/e6_05_jsbsim_matrix.json`): each leg restarts JSBSim at
  the native run's frozen separation state, and 16 of 17 verdicts agree. The
  exception is a220-i28.5-V1-A3 — the cell this matrix already attributes to
  the servo realisation — which cuts off on JSBSim with 480 kg of propellant
  left where the native rigid body depletes (6.85 % propellant history
  divergence, recorded and expected, not gated away). On the six inserted
  cells the engine histories agree to ≤0.05 %, the cutoff times to <1 s, and
  the powered-window attitude error grows by 0.00° over the drift handed at
  ignition; the predicted-miss cells burn out chasing an unreachable target
  and are judged on their verdict, which stays a miss on both channels.

## Follow-through: E10-08 (task #305) -- the guidance class converted

E10-08 asked for a broader envelope on the guidance-class misses without
buying it with a relaxed band. Reading the six "depletion" misses on the
trace, not just the terminal elements, overturns E6-05's last finding. On a260
the stage peaks at 227 km with `v_r` passing through zero, then burns the
rest of the tank horizontally. The osculating periapsis can never exceed the
current radius, so the box could not arm. The law never flew to the target
*radius*. Its `Lambda` targets velocity only, so the climb is whatever the
handoff leaves: 2 km/s of vertical speed and a 326 km ballistic apogee at
ignition. The energy rule was right (terminal orbits thousands of km wider
than the target), and the binding limit was the law, not the delta-v budget.

**Decision.** The shared law gains a terminal-radius radial channel
(`upper_stage` module docstring, E10-08), on for every guided matrix cell
and off by default (the F-07 point is untouched):

* **Radial plan.** The classic PEG radial channel: a vertical thrust
  fraction `s(t) = A + B t` that lands `r(T) = r_target` *and* `v_r(T) = 0`,
  using the closed-form thrust integrals `L, J, S, Q`. `T` is solved so the
  plan's horizontal thrust integral buys exactly the missing horizontal
  speed. Below 12 s to go the law latches velocity-to-be-gained steering.
* **Settle cutoff.** Inside the declared band the burn continues while the
  next control sample is predicted to land closer to the circle, on the same
  side. At ~20 g the band is crossed in a handful of 0.05 s samples.
* **q-alpha limiter.** The 30 deg pitch-bias bound becomes
  `clip(30 000 Pa deg / q, 30, 90)`. It is unchanged wherever q >= 1 kPa and
  opens only near vacuum (q ~ 10 Pa at the ~90 km ignition). Inside the fixed
  30 deg clamp the plan cannot point below the steep handoff velocity vector,
  and that is exactly the authority it needs.
* **Shared command history.** The command is held prograde until ignition
  and the slew limiter is seeded there. The command slew is 6 deg/s, not the
  E1-03 3 deg/s. At 3 deg/s the end-game swing from the plan's pitch-down
  onto velocity-to-be-gained cannot be made, and the rigid body sweeps through
  the band on a300. At 10 deg/s every cell inserts, but the JSBSim PID lags
  the commanded slew by 5-8 deg, over the E6-05 5 deg arrest allowance. At
  6 deg/s that growth is <= 0.4 deg.

**Measured** (`docs/benchmarks/e6_03_guidance_matrix.json`, 17 cells, both
native backends): **12 insert, up from 6, with zero unexpected verdicts**.

* **Converted.** Five of the six depletion-class guidance misses (a260,
  a300, a300-V2, a260-i45, a300-i51.6) and the overshoot cell a180 now
  insert on both backends. Each ends on the law's own cutoff with
  65-684 kg of propellant left; the worst periapsis error is -22.1 km
  (a220-i51.6, point mass). The six E6-03 insert cells still insert, and native
  point-mass / rigid-body propellant agrees to <= 10 kg on every inserted
  cell.
* **Still misses, and why.**
  * **a400: vehicle-capability.** No continuous full-thrust burn from this
    handoff arrives level at 400 km. The radial plan needs `|s| > 1` (more
    than all of the thrust vertical) over the whole burn, and the stage has
    no coast or relight. The unchanged attribution rule now names it a
    capability limit: dry tank, orbit 1.7-1.9 Mm *tighter* than the target.
    That label comes from the data, not from a relabelling.
  * **V3 heavy stage at 180 and 220 km: vehicle-capability.** Unchanged:
    dry tank, orbit 59-211 km tighter than the target.
  * **A3 rate-limited servo: actuator-constraint.** The point mass inserts
    on the same command and the rigid body does not.
  * **F-06 original stage: vehicle-capability.** No TVC axis, so no law.
  * None of these passes, and none of the tolerances moved.
* **Alternatives measured and rejected.**
  * Plan with level arrival `s(T) = 0` (quadratic `s`): infeasible from this
    handoff above 220 km, because it needs `s(0) > 1`.
  * Adaptive level-arrival weighting: discontinuous plans that chatter, and
    a300 regressed to -133 km.
  * An altitude-hold terminal on a lead horizon: the plan's dive then
    outruns the 3 deg/s slew.
  * A 5 s terminal handover: the rigid body sweeps through the band on a300.
  * The prior command semantics (law consulted from separation on the rigid
    body and JSBSim, only from ignition on the point mass): a ~70 deg
    command step at ignition for the frozen trackers. The result was a 35 kg
    native/JSBSim propellant split on a180, over the 0.5 % parity gate.

**JSBSim parity** (`docs/benchmarks/e6_05_jsbsim_matrix.json`, re-flown on
the new law from each cell's native frozen separation state). All 130 E6-05
checks pass, with no gate re-declared.

* **Verdicts.** 16 of 17 agree. The single difference is again
  a220-i28.5-V1-A3, the cell attributed to the servo realisation
  (`verdict_difference_expected`): JSBSim inserts on the 1.15 deg/s servo
  where the native rigid body depletes. So 17/17 match expectation.
* **Co-inserted cells.** Propellant histories agree to <= 0.11 %. Cutoff
  times agree to <= 1.0 s, on the native 1 s sample grid. The powered-window
  attitude growth over the drift handed at ignition is <= 2.3 deg.
* **Terminal elements.** The median |d_periapsis| / |d_apoapsis| between
  JSBSim and native is 12.0 / 3.1 km. JSBSim's 0.01 s control step lets the
  settle cutoff stop within ~0.4 km of the target periapsis, where the native
  0.05 s step stops up to ~22 km short. Both are inside the declared band.

**Consequences.**

* The E1-03 envelope statement is restated again: the shared law, with the
  E10-08 channel, inserts this stage at 180-300 km and at every declared
  inclination inside +/-25 km / 1 deg, on all three plants.
* The next envelope boundary is 400 km. It is a mission-profile limit (no
  coast or relight), not a law limit, so extending it needs a coast-to-apogee
  burn, which is a vehicle/sequence change.
* The settle cutoff's accuracy is the control step at ~20 g. A matrix that
  wants tighter apsides than ~20 km on the native plants needs a finer
  upper-stage control interval, not a different law.
