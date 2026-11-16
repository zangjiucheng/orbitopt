# ADR flightlab-0007: Production JSBSim fidelity (aero-bearing ascent path)

- **Status:** Accepted
- **Date:** 2026-11-14
- **Deciders:** E1 6-DOF Fidelity Integration (Epic #246), E1-04 (#253)
- **Code:** `src/orbitopt/flightlab/sixdof/jsbsim_aero.py`
  (generator: `aerodynamics_block`, `_cg_cancel`, `promotion_fidelity`,
  `regenerate`, `aircraft_aero_note`),
  `src/orbitopt/data/flightlab/jsbsim/aircraft/spike_stage_aero/`
  (generated artefact),
  `src/orbitopt/data/flightlab/aero/spike_stage.yaml` (source of truth),
  `src/orbitopt/flightlab/sixdof/production_compare.py`
  (`compare_production_paths`),
  `src/orbitopt/data/flightlab/sixdof/e104_aero_ascent.yaml`
- **Evidence:** `tests/test_flightlab_sixdof_jsbsim_aero.py`,
  `tests/test_flightlab_backend_identity_parity.py` (production checks)

## Context and problem

Since ADR flightlab-0001 the JSBSim backend is the only Flight Lab engine
with gimballed TVC actuators and propellant-driven mass/CG/inertia -- but
E1-01 deliberately left it flying **zero aero** (ADR 0001's disclosed
boundary): the packaged `spike_stage` aircraft carries an all-zero
`<aerodynamics>` section, so no JSBSim scenario could ever exercise an
aero-bearing ascent, and the aero-fidelity studies could only be done on the
native rigid-body plant with its illustrative coefficient file.

E1-04 asks for a production JSBSim path that flies the *same* aero the
native plant flies -- the packaged aero document
`data/flightlab/aero/spike_stage.yaml` (axisymmetric: `ca`/`cn`/`cy` tables,
5 % static margin in `cm`/`cn_yaw`, rot-damping) -- and for the difference
against the native rigid-body plant to be quantified and explained, or one
path declared production with reasons.

## Decision

**The production JSBSim path is the generated aircraft `spike_stage_aero`,
and it is the production path for aero-bearing 6-DOF ascent.** The aero
document stays the source of truth; the aircraft is compiled from it and
regenerated, never hand-edited (`regenerate(check=True)` in the test suite).

The native rigid-body plant remains the reference plant for the native
studies (F-02/F-04/F-06/F-07) and keeps its engine-fidelity advantages
(ambient-corrected thrust and mass flow); the JSBSim production path brings
aerodynamics, the live mass bookkeeping and the TVC actuator model that the
native plant does not have. The quantified deltas between them are the table
below, not a hand-wave.

### How the document is compiled

- `ca`/`cn` are rotated into stability axes on the document's
  `(alpha, Mach)` grid and written as JSBSim 2-D tables
  (`CD0`, `CL0`); `cy` becomes `CYbeta`, the static-margin tables become
  `CMa`/`CNb`, the rot-damping terms become `CMq`/`CNr`, and the beta-drag
  increment becomes `CDbeta` (bound to the document's own `cn`-based values,
  never fitted).
- Reference geometry comes from the document: `Sw = area_m2`,
  `bw = span_m`, `cbarw = length_m`, AERORP at the *dry* CG.
- `_check_expressible` refuses documents whose tables vary along an axis the
  generated terms cannot express, instead of silently dropping a coefficient.
- **CG-coupling cancellation (`_cg_cancel`).** The document applies its
  forces at the *live* CG and folds the static margin into the couple terms;
  JSBSim applies its forces at one fixed point. Without a correction, the
  applied normal/side force carries an extra `arm x force` couple that
  depends on propellant load -- measured: the naive compilation made the
  production airframe's restoring stiffness differ from the native model's
  by up to ~200x at full tanks (alpha 2.9 deg vs 0.000 deg through max-Q).
  The generated tables therefore subtract exactly that couple,
  `(cg - aerorp) / cbar` times the very force table it belongs to, as a
  function of `inertia/cg-x-in`. The net moment is the document's own
  CG-referenced couple at every propellant load; with the cancellation the
  residual promotion error in `cm`/`cn_yaw` is at machine precision
  (`promotion_fidelity.cm < 1e-9`).
- Promotion error on the *forces* is honest interpolation residue:
  worst 0.0035 absolute, at the 15 deg grid edge the E1-04 ascent never
  reaches (test band 0.005).
- Parity gate: `check_parity` now reads `jsbsim.production_aircraft` from the
  parity source and asserts the production aircraft inherits the same
  masses/propellant/inertias/mount/Isp, *declares* aerodynamics, and declares
  the document's reference geometry (`parity-jsbsim-production-*`).

## Quantified: the same ascent, both plants

`production_compare.compare_production_paths()` flies the packaged E1-04
ascent (sea-level pad, held-vertical hold, full throttle to burnout,
`dt = 0.02 s`) on both plants from the same initial state and publishes
`Check` verdicts. The native side runs `j2=False` so the comparison isolates
thrust and aero from gravity models (JSBSim carries central gravity).
Numbers below are the measured run of 2026-11-14 (Python 3.13,
jsbsim 1.3.3) -- the table stands as published, i.e. **before** the E6-01
engine alignment; ADR flightlab-0009 carries the same table after it
(pad thrust 0.999995, Max-Q 0.14 %).

| Quantity | native rigid-body | JSBSim production | rel. diff | band | Explanation |
|---|---|---|---|---|---|
| Thrust at pad | 1.677 MN (ambient debit over 2 m^2 exit) | 1.883 MN (token `0.001 ft^2` exit, vacuum-like) | 1.08 % vs the predicted `Isp_vac/I_sl` ratio 1.135 | 2 % | The spike engine XML keeps the E4 token nozzle; native debits back pressure. Exactly the disclosed engine-model difference, confirmed at the predicted ratio. |
| Max-Q | 71.5 kPa | 76.1 kPa | 6.1 % | 20 % | The 12 % pad-thrust debit difference: JSBSim climbs faster through denser air. |
| Burnout (t=104 s) altitude | 150.4 km | 165.4 km | 10.0 % | 15 % | Same: vacuum-like thrust flies higher. |
| Burnout speed | 4493 m/s | 4878 m/s | 7.8 % | 10 % | Same. |
| Propellant bookkeeping | mdot from ambient-corrected thrust/Isp | fixed `propflowmax` 600 kg/s | 164 kg max, 0.27 % of load | 0.6 % | Native `T(p)/Isp(p)` runs 0–1 % hotter at low altitude. End dry mass identical. |
| Pitch rate, max abs diff | — | — | 0.0189 rad/s | 0.02 | Actuated TVC vs native gimbal: the same commanded profile. |
| Alpha, pad..60 s (aero authority) | to −2.9 deg | to −0.000 deg | 2.76 deg | 3.0 deg | Native alpha grows as the uncontrolled native trajectory sags off the held body axis under its own engine/gravity model; JSBSim's aero holds the nose on the velocity. Not an aero-term difference: the CG-corrected plants share the document's restoring slope (`cm` exact). |
| Alpha at MECO | −27 deg | −0.0 deg | 27 deg | **not checked** | Past MECO the stage is an unguided ballistic body: with q → 0 the restoring moment vanishes on both plants and the two frame/gravity integrations diverge without bound. Disclosed as `alpha_coast_deg`, not band-checked. |
| JSBSim alpha, whole flight | — | ≤ 0.0 deg | — | 1.0 deg | The production airframe flies the commanded ascent on its own aerodynamics. |

All checks pass (`e104-*` in `production_compare.py`; the bands live in that
module and are asserted by
`tests/test_flightlab_sixdof_jsbsim_aero.py::test_backend_comparison_stays_inside_its_disclosed_bands`).

## Consequences

- `jsbsim` backend: packaged *base* aircraft stay zero-aero (they remain the
  TVC/mass paper test of E4/FL-401); the production path flies the compiled
  tables. The fidelity table in `docs/flightlab.md` and the backend notes in
  `backends.py` say so.
- New scenario `e104_aero_ascent` is the reference aero-bearing 6-DOF ascent;
  the F-03 mass-model checks run on it through the formal adapter
  (`backend_self_check`, all mass/CG/inertia checks pass; `physics_verified` is
  left to the independent re-fly, ADR flightlab-0004).
- Regeneration is deterministic: the artefact, its ADR and the parity source
  move together or the test gate fails (`stale artefact` and the three
  production parity checks).
- Not solved here, and unchanged from E1-01: the engine XML's token nozzle
  exit area (JSBSim thrust is vacuum-like at any altitude) and the fixed
  `propflowmax`. Both are visible as the pad-thrust/burnout-state deltas in
  the table; closing them would change every FL-401/FL-403 result and is
  out of scope for an aero task.
  *Closed by ADR flightlab-0009 (E6-01, #273)*, which re-baselines the FL-401
  /FL-403 results and the table above: the pad-thrust row is now 0.999995 and
  the Max-Q row 0.14 %, and what survives of the burnout-state delta is the
  trajectory/formulation residual, no longer an engine gap.
- The 27 deg coast-phase alpha divergence means the production path is *not*
  a guidance-validation plant for the post-MECO coast of an *unguided*
  stage; with the F-04-style tracker closing the loop (E2) the body follows
  its commands and the divergence is a controller question, not an aero one.

## Alternatives considered

- *Hand-authored `spike_stage_aero.xml`*: rejected — the whole point is that
  the two engines read the same document; a hand-authored file rots.
- *AERORP at a fixed compromise station instead of the dry CG*: rejected —
  it only trades one propellant-load-dependent error for another; the
  `_cg_cancel` correction makes the net couple exact at every load.
- *Tuning the JSBSim engine to match native burnout states* (bigger nozzle
   exit area, `thrustsl`): rejected for this task — it breaks the E1-01
   parity gate's thrust derivation (`isp·g0·ṁ`) and the FL-401/403 results
   built on it; the delta is disclosed instead.
   *Superseded by ADR flightlab-0009*, which closes the gap without a tuning
   knob: the physical 2 m^2 exit plus a choked `propflowmax` derived from
   `T_vac/(Isp_vac*g0)` leave the `isp·g0·ṁ` derivation exact (with
   `isp_sea_level_s` derived rather than assumed) and the FL-401/403 baselines
   re-recorded.
- *Native rigid-body as "the" production path*: rejected — it has no live
  propellant bookkeeping, no TVC actuator and illustrative coefficients; the
  task is about the JSBSim path's fidelity, and the table above is the
  price list of choosing it.
