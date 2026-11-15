# ADR flightlab-0009: Engine and mass-flow parity (one propulsion law, both plants)

- **Status:** Accepted — supersedes the "not solved here" engine bullet of
  ADR flightlab-0007
- **Date:** 2026-11-15
- **Numbering note:** 0008 is taken by E6-04 (#276)'s sampled-data control
  record; this one is the next free number.
- **Deciders:** E6 6-DOF Production Maturity (Epic #265), E6-01 (#273)
- **Code:** `src/orbitopt/data/flightlab/jsbsim/engine/spike_engine.xml`,
  `spike_nozzle.xml`, `src/orbitopt/data/flightlab/parity/spike_stage_parity.yaml`,
  `src/orbitopt/flightlab/parity.py` (`_check_nozzle_area`, `_check_pad_thrust`),
  `src/orbitopt/flightlab/sixdof/production_compare.py`,
  `src/orbitopt/verify/cross_model.py`,
  `src/orbitopt/flightlab/sixdof/tracking.py` (`run_tracking_spike(..., cutoff_t_s=)`)
- **Evidence:** `tests/test_flightlab_backend_identity_parity.py`,
  `tests/test_flightlab_sixdof_jsbsim_aero.py`,
  `tests/test_flightlab_sixdof_tracking_jsbsim.py`,
  `tests/test_verify_cross_model.py`,
  `tests/fixtures/flightlab/opentsiolkovsky/fl403_spike/` (re-recorded)

## Context and problem

ADR flightlab-0007 flew the same E1-04 ascent on both plants and published the
engine-model gap it had to leave open:

- JSBSim read **1.883 MN** on the pad where the native plant read **1.677 MN**.
  The spike engine XML carries the E4 token `0.001 ft^2` exit, so `FNozzle`
  debits essentially no back pressure and the thrust is vacuum-like at every
  altitude; the native plant debits `A_exit * p_ambient` over the 2 m^2 exit
  its parity source declares.
- JSBSim burns a rounded, fixed `propflowmax` of 600 kg/s, while native derives
  its flow from `T(p)/Isp(p)`.

ADR-0007 declined to close them: *"it breaks the E1-01 parity gate's thrust
derivation (`isp·g0·ṁ`) and the FL-401/403 results built on it"*. E6-01 is that
closure. The acceptance target is not "make the numbers equal by tuning" — it is
that both plants implement **one** propulsion law, so any residual left in the
comparison is a claim about the *plant formulation*, not about a knob.

## Decision

One propulsion law, written three ways (parity YAML, JSBSim XML, native plant):

| Quantity | Value | Where |
|---|---|---|
| Vacuum thrust `T_vac` | 1.8800 MN (unchanged) | parity doc, `spike_engine.xml`, native stage |
| Vacuum Isp | 320 s (unchanged) | same |
| Mass flow (choked) | `T_vac/(Isp_vac*g0)` = **599.083275 kg/s** = 1320.7525 lb/s | `propflowmax` |
| Nozzle exit area | **2.0 m^2** = 21.527821 ft^2 | `spike_nozzle.xml`, parity doc |
| Sea-level Isp | **285.506383 s** = `T_pad/(mdot*g0)` | parity doc |
| Pad thrust | `T_vac - A*p_sl` = **1.67735 MN** | derived, checked both sides |

Three consequences worth stating explicitly:

1. **The E1-01 derivation survives.** The parity gate's thrust is still
   `isp * g0 * mdot`; the flow stopped being a rounded 600 kg/s and became the
   choked value, and `isp_sea_level_s` is no longer an assumption but the value
   that makes the two-point native Isp law reproduce that choked flow at the
   pad. Because the native law is linear in the pressure relief and the derived
   `T_sl/Isp_sl` equals `T_vac/Isp_vac`, native `T(p)/Isp(p)` is *identically*
   599.083275 kg/s at every altitude — the two mdot histories are the same
   function, not two close ones.
2. **A physical exit area replaces a numerical token.** `0.001 ft^2` was never
   a nozzle, it was a way of writing "no debit". The area is now a real
   2 m^2, which is what the parity document always claimed and what the native
   debit always used.
3. **MECO becomes a command, not a tank-capacity accident.** With 599.083 kg/s
   the 60 t load lasts 100.153 s, so the old coincidence "`burn_end_s: 100` ==
   tanks dry" is gone. `run_tracking_spike` gained `cutoff_t_s`, and the V-02
   6-DOF adapter now passes the scenario's `burn_end_s` to it: every backend
   shuts down on the *same commanded cutoff*, and a 60 t tank capacity is a
   capacity again rather than a timing parameter.

## Quantified: the same E1-04 ascent, both plants

`compare_production_paths()`, measured 2026-11-15 (Python 3.13, jsbsim 1.3.3,
OT CLI 1.0.0). "Before" is the ADR-0007 table; "after" is this change.

| Quantity | before | after |
|---|---|---|
| Thrust at pad | 1.677 vs 1.883 MN (12.6 % apart, disclosed) | 1.6774 vs 1.6774 MN, ratio **0.999995** |
| Max-Q | 71.5 vs 76.1 kPa, 6.1 % | 71.230 vs 71.329 kPa, **0.14 %** |
| Endpoint (T+103.6 s) altitude | 150.4 vs 165.4 km, 10.0 % | 149.7 vs 157.4 km, **5.9 %** |
| Endpoint speed | 4493 vs 4878 m/s, 7.8 % | 4501 vs 4786 m/s, **5.9 %** |
| Propellant bookkeeping | 164 kg, 0.27 % of load | **12.0 kg**, 0.02 % |
| Pitch rate, max abs diff | 0.0189 rad/s | 0.01885 rad/s |
| Alpha, pad..60 s | 2.76 deg | 2.739 deg |
| Alpha at MECO (recorded, not gated) | 27 deg | 26.8 deg |

Bands moved with the physics in `production_compare.py`: pad-thrust ratio
`0.02 -> 0.005`, max-Q `0.20 -> 0.05`, propellant `0.006 -> 0.002`. The
endpoint altitude/speed bands stay at 0.15 / 0.10, for the reason below. All
eight `e104-*` checks pass.

### Where the remaining residual lives

- **Closed (implementation, was the engine law).** Pad thrust, Max-Q and the
  propellant book. The 4.7e-6 pad-thrust residual is JSBSim's lb/ft^2 rounding
  plus its atmosphere's sea-level pressure against `pressure_pa(0)`; the 12.0 kg
  is the disclosed one-to-two-frame lag of JSBSim's tank drain (the F-03
  scenario allows 2 frames * 599.083 kg/s * 0.02 s = 24 kg), not a flow
  difference.
- **Open (implementation, trajectory formulation, pre-existing).** The 5.9 %
  endpoint delta is *not* thrust level: `thrust_n` agrees to 3e-5 through the
  burn and the mass book to 12 kg, yet the along-track difference goes
  -0.9 m/s at T+10 s, -2.7 m/s at T+60 s, +27.0 m/s at T+80 s, +285 m/s at
  burnout. It grows exactly where the alpha-window row lives (the native body
  sags ~2.7 deg off its own velocity vector while the actuated TVC and compiled
  aero hold the JSBSim nose at alpha ~ 0), and a near-vertical coast amplifies
  ~2.5 m/s of burnout velocity into kilometres of altitude. That is the
  native-plant-vs-FEM difference ADR-0007 bought with the aero path, now visible
  without an engine gap sitting on top of it — which is why the endpoint bands
  were *not* tightened.
- **Open (formulation, disclosed as before).** Post-MECO alpha 26.8 deg: past
  burnout the stage is an unguided ballistic body with q -> 0, so the restoring
  moment vanishes on both plants. Recorded as `alpha_coast_deg`, never gated.
- **New and expected.** FL-403's dt-halving spread grew from 30 m to 58 m.
  The pad thrust is 11 % lower than the old vacuum-like JSBSim figure, so TVC
  authority (`thrust * arm`) is 11 % lower and the loop runs closer to its
  actuator lag (peak deflection 0.42 -> 0.47 deg). The MECO altitude deviation
  is -110 / -52 / -22 m at dt = 0.02 / 0.01 / 0.005 s, a Richardson
  extrapolation of **+8 m**: the FL-403 deviation is step-size-limited loop lag,
  and the whole number sits ~10x under its 500 m line either way.

## Re-baselined artefacts (audit record)

- **FL-403 3-DOF reference** `tests/fixtures/.../fl403_spike/` re-recorded with
  OT CLI 1.0.0 from the aligned vehicle file (1.8800 MN, choked flow noted,
  `nozzle_exit_area_m2: 2.0`, commanded `burn_end_s: 100`). Recorded MECO
  (T+100 s): 73.94 km, 5233.9 m/s Earth-relative, Mach 18.00, max-Q 95760 Pa;
  dry mass at MECO 10091.7 kg (91.7 kg of commanded-MECO residue).
- **FL-403 GO bounds are unchanged** (2 deg / 500 m / 20 m/s). Measured now:
  0.51 deg whole-burn, 0.052 deg settled, -52 m MECO altitude, -2.03 m/s,
  +0.015 deg FPA, 0.47 deg peak gimbal, 0.0 % saturation. `acceptance_matrix.yaml`
  `control.measured.meco_position_m` moved 51 -> 52 with this note; the findings
  doc carries the dt sweep.
- **V-02 cross-model** (`verify.cross_model`): the 3-DOF ascent RHS now debits
  `A_exit * p_ambient` through `atmosphere.pressure_pa`, `CrossModelScenario`
  derives its defaults from the aligned engine, and the 6-DOF side is commanded
  to cut at `burn_end_s`. The 3-DOF-vs-OT MECO pair tightened from ~1.3 km to
  **410 m** of semi-major axis (0.009 %), 0.7 m/s of Earth-relative speed, and
  identical dry mass, so `test_default_pair_passes_and_matches_the_fl403_reference`
  is re-pinned to a = 4367061 m, e = 0.544891, v_e = 5219 m/s.
- **ADR flightlab-0001 keeps its numbers.** That ADR records the E4 spike
  as flown against the token nozzle; rewriting it would destroy the record.
  This ADR is the change record instead.

## Consequences

- The JSBSim spike stage, its aero-bearing production airframe, the booster and
  the upper stage all inherit the choked flow and the 2 m^2 exit from the two
  shared engine XMLs — one change, four consumers. The generated airframe is
  regenerated by `python -m orbitopt.flightlab.sixdof.jsbsim_aero` and
  `--check` gates staleness, as before.
- The parity gate grew two checks (`nozzle exit area`, `pad thrust` on both
  engines) so the law cannot silently drift again: the area, the flow, the
  derived sea-level Isp and the pad thrust now have to move together or
  `tests/test_flightlab_backend_identity_parity.py` fails.
- `run_tracking_spike(..., cutoff_t_s=)` is the only API change. FL-403 does
  not use it (it lets the plant run to tank depletion, which is what FL-401
  measures); the V-02 harness does.
- Recovery and mission-side engine models are untouched: they carry their own
  `T(p)/Isp(p)` tables and are not consumers of these XMLs.

## Alternatives considered

- *Leave the token `0.001 ft^2` exit and keep disclosing the gap*: rejected —
  that is where ADR-0007 ended, and the 12 % pad-thrust delta it forced into
  the table is what E6-01 exists to remove.
- *`thrustsl` / a JSBSim thrust scale factor to match native*: rejected — a
  free knob that reproduces agreement at one altitude only. The physical exit
  area derives agreement at every altitude from one number.
- *Give the native plant a pressure-varying mass flow*: rejected — unnecessary.
  With the derived `isp_sea_level_s` the two-point linear native law is
  identically constant-flow, so making `mdot(p)` pressure-dependent would
  *add* a difference rather than remove one.
- *Set the OT fixture's `burn_end_s` to the 100.153 s depletion instant so the
  3-DOF reference burns the load out too*: tried and rejected. The reference
  CSV is sampled on a 1 s grid, so a non-grid cutoff cannot be read back
  (linear interpolation across it costs 21 m/s of pure artefact), and it made
  JSBSim, OT and the V-02 harness disagree about what "MECO" means. A commanded
  100 s cutoff on every backend is the version that states its own convention.
