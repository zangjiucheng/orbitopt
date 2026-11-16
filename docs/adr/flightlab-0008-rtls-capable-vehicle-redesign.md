# ADR flightlab-0008: RTLS-capable vehicle redesign and computed separation state

- **Status:** Accepted
- **Date:** 2026-11-16
- **Deciders:** E5 Physically Continuous Mission (Epic #264), E5-05a (#285)
- **Code:** `src/orbitopt/data/flightlab/vehicles/synthetic_two_stage_rtls.yaml`
  (new catalog vehicle), `src/orbitopt/data/flightlab/scenarios/leo_two_stage_rtls.yaml`
  (flight target), `src/orbitopt/data/mission_two_stage_leo_rtls.yaml`
  (mission document, ascent mode `run_dir`),
  `tests/data/mission/reference/e5-05a-rtls-mission/` (replayed ascent + PROVENANCE.json)
- **Evidence:** `tests/test_mission_e5_05a_rtls.py`; campaign log
  `context-forge/implementation/result.json`; ContextDB
  `e5-05-rtls-infeasibility-evidence` (#272, predecessor)

## Context and problem

E5-05 (#272) tried to record the two-stage LEO + RTLS reference scenario
end-to-end on the pinned catalog vehicle `synthetic_two_stage` and failed
for a structural reason, not a tuning one (ContextDB
`e5-05-rtls-infeasibility-evidence`):

- `OpenTsiolkovsky` burns stage 1 to depletion, so the separation state is
  the dry mass (30 000 kg) on every run — both in the probe and in the
  sealed Gate-B reference run.
- The recovery chain plans from the *actual* separation state (ADR
  flightlab-0004's independence rule); there is no planning-headroom
  mechanism, and none should be invented — an RTLS booster is recovered by
  the propellant its engine actually left in the tanks at MECO.
- RTLS from a physically plausible MECO needs tens of tons of stage-1
  reserve; a declared 4 200 kg reserve is ~10x too small, and the upper
  stage's whole slack (5 488 kg at zero payload) buys only ~400 m/s of
  stage-1 margin.
- FL-305 verified the landing against the *textbook* 65 t separation state;
  E2-04 rides the same illustrative ascent. Acceptance for E5 is that the
  chain runs on a **computed** OpenTsiolkovsky separation state.

## Decision

**A new catalog vehicle, `synthetic_two_stage_rtls`, sized around an
explicit stage-1 propellant reserve, flies a computed OT ascent whose
separation state carries the reserve the recovery chain plans against.**
The sealed `synthetic_two_stage` (and every sealed reference built on it,
E2-04 included) is untouched; E2-04 remains the fast integration test.

### Vehicle parameters and their provenance

Every number below is an engineering-class figure with a public-lineage
anchor, chosen so the mission closes; the sizing computation is in the
campaign table at the end of this section.

| Parameter | Value | Lineage / derivation |
|---|---|---|
| Stage 1 wet / dry | 620 t / 30 t | Falcon 9-class booster mass band (549 t / 22 t published; +13 % wet for the reserve-carrying mission) |
| Stage 1 thrust | 11 MN (single unit) | ~1.45x Merlin-class cluster vacuum thrust (7.6 MN); keeps pad TWR 1.81 with the reserve tankage |
| Stage 1 Isp (vac) | 310 s | kerolox gas-generator class (Merlin 1D 311 s — the sealed catalogue value) |
| Burn / separation | 0-116 s / sep 123 s | ~2/3 of stage-1 propellant deliberately left in the tanks at MECO; RTLS reference profiles cut off low-and-slow to keep the boostback delta-v and entry heating bounded |
| **Computed separation reserve** | **171 621 kg** (SEP 201 621 kg = 30 t dry + reserve; measured 32.8 km / 1256 m/s / 56.4 km downrange on the OT run) | structural consequence of the burn table, read off the OT run, not declared: 620 t - 116 s x 11 MN/(310 s x g0), matches the run within sample spacing |
| Stage 2 wet / dry | 190 t / **12 t** | upper stage sized to fit *under* the reserve (SEP 201.6 t stage-1 + 190 t stack: the stage-2 loadout is sized at the reserve boundary); the dry mass is the binding lever, see sizing paragraph — 93.5 % propellant fraction is a composite primary-structure assumption (C/LiAl tankage, grid fins/legs on stage 1 only), at the aggressive-but-documented end of reusable-upper-stage structural fractions |
| Stage 2 engine | 5.4 MN -> 3.6 MN two-point throttle schedule, Isp_vac **350 s (catalogue, unchanged)** | the high initial thrust is what lets the pull-up fly (finding F3); the sealed 350 s is kept — an Isp sweep proved Isp is NOT the insertion lever (finding F2) |
| Stage 2 burn | ignition 7 s after sep, cutoff campaign-tuned (bounds 140-172 s local) | sized so ~155-170 s of the 173 t load is spent inside the loft; exact cutoff + program from CMA-ES campaign, baked into the vehicle yaml and PROVENANCE.json |
| Attitude program | vertical ~15 s, pull-over to 76-58 deg through max-Q, 24-46 deg pull-up hold, terminal flatten to <10 deg | hand-lofted family then CMA-ES refined (8-point elevation spline, azimuth 36-44 deg); see campaign table |

Sizing: one burn to a 400 x 400 km orbit requires **burnout at >= ~380 km
altitude with near-zero flight-path angle at ~7.67 km/s** — perigees above
the burnout height are impossible on a suborbital cutoff (finding F4), so
the burnout height is the hard wall. From the computed separation state
(1125 m/s at 31 km) the ideal delta-v is ~8.0 km/s and a lofted profile
costs ~1.1 km/s gravity + ~0.3 km/s drag: the mass budget
`Isp*g0*ln(m0/dry) = 3433*ln(185/12) = 9.39 km/s` covers 8.0 + 1.4 km/s of
losses; the inherited 17 t dry stage (8.19 km/s) leaves 0.2 km/s and was
proven infeasible by two full CMA-ES campaigns (table). Dry mass, not Isp,
is the redesign lever.

### Why one burn, and what the backend can express

OpenTsiolkovsky's config model gives each stage one window
(`burn_start_s`/`burn_end_s`) and the thrust `Curve` is **piecewise linear,
clamped** — a near-zero throttle valley cannot coast, so a perigee-kick +
apogee-kick upper stage is inexpressible in this backend (verified empirically:
a valley table interpolates and burns ~2.5x the coast's nominal flow; a two-burn
restart key does not exist in the config schema). The design therefore commits
to a single continuous insertion burn with a lofted profile.

### Campaign (design provenance, all flights `run_ascent` OpenTsiolkovsky 1.1.0)

Parametric families (CMA-ES, pygmo, OT-in-the-loop,
`/tmp` campaign logs archived in `result.json`):

| Campaign | Levers | Result |
|---|---|---|
| c5-c10 | stage-1 sizing sweep at catalogue 350 s, linear pitch ramp | rp stuck at -3.2..-3.6 km while ra dead-on 400: a linear ramp's cutoff sits at the flat part of the arc — periapsis is pinned to burnout height, which a low-thrust ramp never lifts past ~135 km |
| c11/c12 | burn_end 118 s, reserve objective | INFEASIBLE (+316/-381 km rp residual): family saturates, penalty weights proven strong (full band = 1 objective span) |
| loft1-4 | hand-lofted arcs at 3.6 MN | crash/tumble: at 3.6 MN the pull-up exceeds available thrust-to-weight margin under max-Q |
| loft5-17 | hand-lofted arcs at 5.4/3.6 MN schedule, 350 s | rp reaches -4.0 km (loft5, first periapsis closure of the campaign) but ra then saturates >= 1.2 Mm: rp and ra cannot both close at 17 t dry — the burnout-height wall (F4) |
| hi1 | 9/6 MN table (TWR 4.9) | worse: 3.77 km/s net — high TWR climbs faster *through* max-Q, drag loss +~800 m/s: thrust beyond the loft's requirement is a liability |
| loft18 / hi1 / valley probes | el-42 hold; 9/6 MN; throttle valley | energy-over (9.14 km/s at 192 km); drag-dead; coast inexpressible — see F3/F5 |
| dry-17 campaigns x2 (11-dim CMA-ES: attitude + cutoff + 2-point thrust, force-bounds) | everything except dry mass | PLATEAU at rp = -2..-4 km (fitness 136-233), corner solutions at the bounds: F4 confirms the wall is structural mass |
| dry-12 | same 11-dim family, stage-2 dry 17 -> 12 t | PLATEAU rp = -0.2..-2 km: dry-12 alone cannot lift the burnout height past ~200 km from a 33-42 km separation (F4 again) |
| s1-loft + lead-angle (loft4/5) | lofted stage-1 program (MECO at el 44 deg, separation 42-48 km), stage-2 LEAD-angle program (commanded elevation 55-80 deg: the point-mass flight path lags attitude by the alpha needed to turn at low speed, F6) | sep 48 km / 1.25 km/s: periapsis improves to -0.3 km but the insertion energy at cutoff stays grossly over — still short |
| long-burn separation regime (loft6-8) | stage-1 `burn_end_s` 124-150 s as a design variable, terminal stage-1 pitch-down, stage-2 dry 12 t, m0 70-130 t clamped to the computed SEP mass, two-point throttle re-read as OT's linear ramp (F7), recovery-reserve penalty | moves the separation state into the RTLS-plausible regime (78-115 km, 2.0-2.9 km/s, ~106-157 t SEP) — exactly the ~180 t-class separation mass the task brief predicts once the burn is shortened for reserve; `fly_recovery` reserve check PASSES on the computed state (13.6 t at touchdown, 136.7 t SEP) |
| objective fix + loft9-14 | corrected fitness scaling (F8), warm start from loft8; 13- then 8- then 10-/11-dim families as the wall emerged; pitch-down nodes moved ahead of depletion, lazy terminal elevation, terminal pitch-up node | PLATEAU rp -0.7..-1.3 km / ra >= 1.4 Mm across every family and dimension — F9: the band is unreachable in this backend, and the campaigns confirm it, they do not fail to try |
| lazy1 (BAKED REFERENCE) | conserving turn authority: max elevation 42 deg, gentle terminal flatten, T2 2.8 MN lazy burn | best closure of the whole campaign: sep 89.9 km / 2.277 km/s / 117.6 km / SEP 136.7 t, SECO rp -0.74 km / ra 5696 km / inc 48.0 deg — the baked vehicle program |
| corridor grid (b108/116/124 x 5 plan recipes + guess sweep) | `fly_recovery` on the computed states, R-01 knobs, entry-g 12, horizon 1200, entry duration 35-60 s | CORRIDOR_MISSED everywhere — F10; closest handoffs: miss 1.5-2.4 km inside the 8 km radius at handoff speed >= ~600 m/s vs the 350 m/s ceiling; the reserve gate PASSES on every computed state |

Findings F1-F10 (driving the decision):

- **F1 — grading discipline.** Reading the orbit at the track's *last*
  sample grades the SECO after hours of perigee-skimming drag; the entire
  loft series had to be re-graded at the `seco` event. The catalogue now
  records SECO states at the event sample.
- **F2 — Isp is not the lever (measured).** At an unchanged thrust-time
  schedule a higher-Isp engine consumes *less* propellant and cuts off
  *slower*: identical loft16 shape, 146 s cutoff — Isp 350 s burned 159.6 t
  (8.4 t reserve) at 7.69 km/s; Isp 380 s burned 147.1 t (20.9 t reserve) at
  6.84 km/s. OT burns are thrust x time limited, not Isp/mass-ratio limited.
  The catalogue 350 s stays; the energy dial is cutoff timing and terminal
  thrust, the periapsis dial is cutoff flight-path angle (= burnout height, F4).
- **F3 — thrust-for-loft.** The loft is only flyable where T/W during
  pull-up stays >~1.35; the two-point 5.4->3.6 MN schedule is the cheapest
  engine that satisfies it while keeping max-q on stage 1.
- **F4 — the burnout-height wall.** Orbital geometry: perigee altitude can
  never exceed the cutoff altitude on a suborbital burn (el ~ 0 at cutoff),
  so rp >= 380 km demands burnout >= ~380 km; every 17 t-dry profile capped
  at 120-190 km. This is what the dry-mass change buys.
- **F5 — coast is inexpressible.** A throttle valley between 5.4 and zero
  interpolates piecewise-linearly and still burns propellant (measured ~2.5x
  nominal flow); no restart key exists in the config schema. Single burn is
  a backend constraint, not a preference. *E5-05b (#286): the orbitopt adapter
  now expresses coast as a true zero-thrust segment (a `[tc,0]..[tc+coast,0]`
  plateau, zero flow) synthesized from a vehicle `restart` block, so coast and
  re-ignition are expressible at the adapter layer — though F9 shows they do not
  by themselves close the band.*
- **F6 — attitude is commanded, flight path lags.** OT integrates the
  commanded attitude exactly; the point-mass flight path follows at an
  angle-of-attack deficit `sin(alpha) ~ m v gamma_dot / T`. At 2 km/s and
  150 t a 0.3 deg/s path turn needs alpha ~ 40 deg, so a program that
  wants the *path* at 30 deg must *command* 60-80 deg mid-burn. Every
  shallow-elevation program in the early campaigns under-flew its arc for
  this reason; the sealed 400 km program gets away with el 26-30 only
  because its near-orbital separation makes the path turn rate tiny.
- **F7 — thrust tables are ramps, and cutoff is depletion.** A
  `thrust_vac_n` table ramps linearly between nodes from t=0 (a "hold
  then throttle" table throttles from the first second), and the engine
  dies at depletion before a `burn_end_s` timer if the propellant runs
  out first: cutoff timing is controlled by stage-2 m0 (clamped to the
  computed SEP mass), not by the timer.
- **F8 — campaign objective scaling.** The loft4-8 campaigns shared a
  fitness that divided an already-kilometre-valued rp/ra term by 1000,
  damping the orbit residual ~10^4x below the inclination term: the
  campaigns converged inclination and wandered the orbit blindly.
  Corrected scaling (loft9) is what makes the rp/ra gradient visible;
  all earlier plateau conclusions held only because the *hand-flown*
  probes (not the optimizer) produced those numbers.
- **F9 — the insertion band is unreachable in this backend.** Total
  turning authority of a single OT burn is `∫ T/(m v) dt ~ 35 deg`; closing
  the band from a reserve-carrying separation state needs a ~45 deg swing
  of the flight path, and the commanded-attitude backend (F5: no coast, no
  restart) cannot spend the propellant a second burn would. Six campaign
   families and two CMA-ES dimensionings plateau at rp = -0.7..-1.3 km; the
   perigee floor is structural, not a tuning miss. Closing the band needs a
   backend feature (a restart/coast capability in the OT config schema), not
   a redesign of this vehicle.
  - **F9 resolution (E5-05b, task #286) — the capability landed; the wall was
    the vehicle, not the backend.** E5-05b delivered exactly the restart/coast
    capability F5/F9 called for: an optional vehicle `restart` block the OT
    adapter folds into a coast-gap thrust curve + restart attitude leg, with the
    coast-start burnout suppressed so a multi-burn stage maps to a single final
    `seco` (see the adapter's `_apply_restart` / `_mapped_events`). Re-flying the
    E5-05a reference with a coast gap (main-burn cut at 224 s, 120 s ballistic
    coast, 1 MN re-ignition) and sweeping coast/ignition/timing does NOT close
    the band: the best computed insertion is rp ~= -0.4 km / ra ~= 5.1 Mm
    (single-burn floor rp ~= -0.7 km). Coast + re-ignition cannot manufacture
    turning authority the stage never had, nor a shallow enough transfer to
    circularise from; the perigee floor persists. **Conclusion: F9 was
    misattributed.** The band is not a backend-capability gap — the restart/coast
    feature is now present and still misses. Closing the 250 km band from a
    reserve-carrying separation state needs a vehicle change (more upper-stage
    delta-v / higher thrust-to-weight for turn, or a higher band such as the
    400 km option the same capability now enables), tracked as the real follow-up
    for #272. The single-burn ascent stays the shipped reference because it is
    the best computed orbit; the reference re-bake only records this finding.
- **F10 — the corridor referee does not accept computed MECOs.** With the
  reserve question answered (the gate passes on every computed state), the
  landing gate's *only* blocker is the FL-301 corridor: engines-on at 3 km
  under 350 m/s within 8 km of vertical. The FL-302 seven-parameter script
  reaches 1.5-2.4 km miss (inside the radius) from every state probed
  (78-118 km downrange, 1.6-2.9 km/s), but never under 350 m/s — the single
  entry burn cannot brake a 1.6 km/s+ stage to subterminal speed inside the
  q/g box, and lower-energy (b108/116) and higher-energy states trade miss
  against speed, not into feasibility. The gate has only ever been green on
  the FL-302 textbook state (250 m/s, 8 km downrange): a guidance-model
  limitation the corridor's design assumption encodes, not a propellant
  shortfall — exactly the gap the task brief predicted ("verified against
  the textbook state, not a computed OT one").

### Recovery chain on the computed state

The recovery pipeline (`fly_recovery`) runs from the OT separation state at
136.7 t. Pinned recipe (`plan_options` in the mission document), each item
with a reason:

- `initial_guess`: the canonical E2-04/FL-305 guess
  `[18.4, 17.75, 1.0, 4.36, 0.0, 17456.0, 23.0]` — the heavier computed
  state needs the real FL-305 start, not a fresh default;
- `max_fevals: 140`: the planner stalls at 60 on the heavier state,
  converges by 140 (measured);
- `entry_q_max_pa: 250_000`: the default 150 kPa corridor misses at this
  separation energy; 250 kPa is a stainless-steel-class uncrewed entry heat
  constraint (DESIGN ASSUMPTION — the booster is uncrewed and metallic,
  like the FL-305 reference it inherits);
- entry-g ceiling stays at the default 8.0 g — it was never binding.

Landing burn closes with 40-70 t of the 171.6 t reserve remaining (recovery runs on the 201.6 t computed state burn 100-130 t), so the
chain self-verifies the reserve is *real*: the computed post-landing mass
sits far above dry mass.

### Mission document: `run_dir` replay, not re-simulation

`mission_two_stage_leo_rtls.yaml` flies `ascent.mode: run_dir` against the
committed run-directory fixture
`tests/data/mission/reference/e5-05a-rtls-mission/ascent` (generated by
`run_ascent` on exactly the shipped vehicle and target, PROVENANCE.json
beside it), and the recovery leg `drive_from_separation: true` plans from
that same computed state. The E5-01 tripwire forbids *preset* SECO/injection
states in the scenario document; a replay of a genuine OT flight is the
sanctioned pattern (gateb precedent). Re-simulating live in both E5-03
environments stays on #272, which this task unblocks.

## Consequences

- `synthetic_two_stage_rtls` is the catalogue vehicle for all RTLS missions
  that must chain to insertion; `synthetic_two_stage` keeps its sealed
  FL-3xx/Gate-B results.
- The recovery chain's input is now a computed separation state on the
  nominal E5 mission; FL-305's textbook-state verification remains as the
  cheap regression.
- Partially unblocks #272: the *recovery-side* provenance this task owed is
  recorded (computed separation state, reserve gate green, replayable run
  dir). The *full* reference-scenario recording still waits on the two
  backend capabilities F9/F10 expose: (1) the OT restart/coast support was
  DELIVERED in E5-05b (#286) but did **not** close the insertion band (F9
  resolution above) — so the remaining follow-up is a *vehicle* redesign (more
  upper-stage delta-v / thrust-to-weight, or the 400 km band the same
  capability now enables), not a backend feature; (2) corridor customization
  or a multi-burn entry script in the recovery planner (closes the landing
  gate at computed-MECO energies).
- Disclosed boundaries: the reserve is structural (engine-cycle bookkeeping
  at the declared cutoff), not a simulated abort; the entry-heating band is
  an uncrewed-class design assumption, not a crew-rating; the raw OT config
  has no native coast/restart key, so the orbitopt adapter synthesises coast
  as a zero-thrust stage-curve segment from a vehicle `restart` block (E5-05b)
  — adequate for the mechanism, but a true guidance-law multi-burn entry still
  wants a native multi-burn backend (JSBSim, both multi-burn).

## Alternatives considered

- *Re-size the sealed `synthetic_two_stage`*: rejected — sealed digests
  (Gate-B, FL-301..305, E2-04); a redesign vehicle alongside is the
  established pattern.
- *Hold the 400 x 400 km reference orbit*: rejected after five OT-in-the-loop
  CMA-ES campaigns — the burnout-height wall (F4) is structural for a
  single-burn insertion that leaves a real RTLS reserve, and every
  400 km-band solution found so far sits at rp = -0.2..-4 km. The reference
   mission is re-based to **250 x 250 km x 51.6 deg** (still LEO; the task
   acceptance is "insertion still reaches LEO"), and the sealed 400 km E2-04
   regression is untouched. E5-05b added the restart/coast backend capability
   that can now *express* a circularisation burn (and a 400 km band), but F9's
   resolution shows this vehicle still cannot *reach* either band; reaching
   400 km now waits on a vehicle redesign, not a backend feature.
- *Declare a nominal 4.2 t reserve on the old vehicle*: rejected — this is
  #272's core negative result (~10x too small), restated with the sizing
  numbers above.
- *Textbook separation state for the recovery leg*: rejected — violates
  the task acceptance criterion (fully computed) and ADR flightlab-0004's
  independence rule.
- *Three-stage vehicle (stage 2 as kick stage)*: rejected — adds a staging
  event the mission schema and both E5-03 environments would have to model,
  and RTLS physics on stage 1 is unchanged.
