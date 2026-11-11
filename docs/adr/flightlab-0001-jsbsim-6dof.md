# ADR flightlab-0001: JSBSim as the 6-DOF engine for Flight Lab ascent

- **Status:** Accepted — **GO**
- **Date:** 2026-11-11
- **Deciders:** Flight Lab E4 spike (Epic #196; FL-401 #206, FL-402 #207, FL-403 #208)
- **ContextDB:** `epic187/jsbsim-go-no-go`

## Context and problem

Flight Lab's ascent product today is 3-DOF (OpenTsiolkovsky via the FL-004
adapter, Epic #187): it cannot show attitude dynamics, TVC behaviour or
staging artefacts. Epic #196 was a time-boxed spike to answer one question:
**can JSBSim (jsbsim==1.3.3) serve as the 6-DOF engine behind the Flight Lab
contracts** — a liquid stage with gimballed TVC flying the 3-DOF ascent
program, and two vehicles side by side after separation? A NO-GO would mean
falling back to kinematic attitude display over 3-DOF, or re-evaluating
RocketPy / an in-house 6-DOF model.

Three sub-spikes produced evidence:

| Spike | Question | Code |
|---|---|---|
| FL-401 (#206) | Can JSBSim fly our stage model faithfully (thrust, mass, TVC moments, actuator)? | `src/orbitopt/flightlab/sixdof/jsbsim_driver.py`, `src/orbitopt/data/flightlab/jsbsim/aircraft/spike_stage.xml` |
| FL-402 (#207) | Can two independent `FGFDMExec` instances coexist and hand off state (staging)? | `src/orbitopt/flightlab/sixdof/staging.py` |
| FL-403 (#208) | Can a simple PID on the gimbals track the 3-DOF attitude program to MECO (~76–81 km, Mach ~18) and stay close to the 3-DOF trajectory? | `src/orbitopt/flightlab/sixdof/tracking.py` |

## Decision drivers — the GO criteria, declared before the FL-403 run

Declared up front in `tests/test_flightlab_sixdof_tracking_jsbsim.py` (and
the FL-401/402 test files):

1. **Tracking within declared bounds.** Settled phase (t ≥ 60 s, after the
   program-shaping window): attitude error max ≤ 2.0°, RMS ≤ 0.6°.
2. **Gimbal saturation ≤ X % of the burn** with X = 0: the *actuator output*
   (not the command) must never touch the ±5° travel limits while thrusting,
   and commands must stay within travel.
3. **Stable integration.** No non-finite states over the whole flight to
   MECO+5 s (~76 km, Mach 18); dt-halving (0.01 → 0.005 s) agreement:
   MECO altitude < 300 m, speed < 20 m/s, FPA < 0.5°.
4. **Trajectory stays near the 3-DOF reference.** At MECO: |Δalt| ≤ 6 km,
   |Δv| ≤ 200 m/s, |ΔFPA| ≤ 1.0° (generous, transient-inclusive bounds —
   the 6-DOF vehicle must fly its own attitude schedule, not the 3-DOF one).
5. **Separation works** (FL-402 criteria): momentum residual < 1e-6 rel,
   position continuity < 1 mm, determinism across reruns and step orders.

## Results (FL-403 run, `docs/benchmarks/fl403_tracking_spike.json`,
regenerate with `PYTHONPATH=src python scripts/fl403_run_spike.py`)

Vehicle: the FL-401 spike stage (10 t dry, 60 t prop, Isp 320 s, 1.883 MN,
100 s burn), zero aerodynamics on **both** sides so deviations isolate
attitude/actuator effects; 3-DOF reference: mirrored OpenTsiolkovsky run
(`tests/fixtures/flightlab/opentsiolkovsky/fl403_spike/`,
`scripts/fl403_record_reference.py`). Controller: PID in angular-acceleration
space on the SO(3) geodesic error + pitch rate feed-forward, converted to
gimbal angle through the **live** plant gain `T·arm/I` (gain-scheduled by
instrumented thrust/inertia/CG); actuator lag/slew/travel sit between
command and physics.

| Criterion | Bound | dt = 0.01 s | dt = 0.005 s | Verdict |
|---|---|---|---|---|
| Settled attitude error max (t ≥ 60 s) | ≤ 2.0° | **1.12°** | **1.12°** | PASS |
| Settled attitude error RMS | ≤ 0.6° | **0.52°** | **0.52°** | PASS |
| Gimbal saturation (actuator at ±5° limit) | 0 % of burn | **0.0 %** | **0.0 %** | PASS |
| Peak gimbal deflection | < 5° travel | 0.86° | 0.84° | PASS |
| Body rates (max p / q / r) | report only | ~0 / 15.8 / 14.0 °/s | ~0 / 15.5 / 13.8 °/s | report |
| MECO altitude deviation (ECEF climb vs OT alt) | ±6 km | **−4.71 km** (76.55 vs 81.25 km) | −4.67 km | PASS |
| MECO speed deviation | ±200 m/s | **−89.8 m/s** (5224.0 vs 5313.8) | −89.2 m/s | PASS |
| MECO FPA deviation | ±1.0° | **−0.25°** (19.45° vs 19.71°) | −0.25° | PASS |
| MECO Mach / altitude (stress point) | finite states | Mach 18.2 @ ~76.5 km | Mach 18.2 | PASS |
| dt-halving: MECO altitude / speed / FPA | <300 m / <20 m/s / <0.5° | **40 m / 0.58 m/s / 0.006°** | — | PASS |
| Non-finite states | none | none | none | PASS |
| Separation (FL-402) | momentum 1e-6 rel, pos 1 mm | 1.4e-8 rel / 2.1e-7 m, bit-identical reruns & step orders | — | PASS |

![FL-403 attitude tracking and MECO deviations](../assets/fl403_tracking.png)
![FL-403 gimbal commands vs actuator outputs](../assets/fl403_gimbal.png)
![FL-403 6-DOF vs 3-DOF trajectory](../assets/fl403_meco_deviation.png)

### The one caveat, by design: the early program transient

The full-profile (t ∈ [0.5, 100] s) error peaks at **24.0°** — *not* a
control failure. The 3-DOF OT program ramps elevation from 90° toward 18°
with a slew the chart law `d(el)/dt = q·cos(el) + …` makes **kinematically
unachievable for a TVC vehicle near vertical at low speed** (cos(el) → 0
kills the pitch authority to change elevation). The 6-DOF vehicle lags, the
loop recovers as the program flattens, and from t = 60 s tracking is
sub-degree and stays that way through MECO at Mach 18. FL-401's open-loop
run shows the same lag with *no* controller — the discrepancy is a property
of the program, not of the simulator or the loop. **Consequence:** any
production 6-DOF program generator must rate-limit attitude commands in body
rates, not just in chart angles.

## Decision: **GO**

All five declared criteria pass with margin. JSBSim 1.3.3, driven through
the thin `SixDofSim` driver, is viable as the E4+ 6-DOF engine:

- physics bookkeeping is exact where it is checkable (FL-401 table:
  Tsiolkovsky to 2e-4 rel, TVC moment exact to 1 %, actuator lag 0.03 %);
- a naïve gain-scheduled PID saturates nothing and holds sub-degree tracking
  through max-q-free (no aero), high Mach, 76 km — headroom for a real
  controller and real aerodynamics;
- two instances + a proven state-handoff recipe make staging a first-class
  contract (`to_flight_run` already emits a valid three-body FlightRun v1);
- costs are known and bounded: first-order-in-dt convergence (one-frame
  ignition/mass latency), opt-in pip-only dependency (hash-pinned
  `envs/sixdof-requirements.txt`; conda locks untouched), ECEF/geoid altitude
  bookkeeping done in our driver, ~5 k steps/s.

## Consequences

Positive: attitude/actuator effects become simulable inside the Flight Lab
contract types; staging runs are already contract-valid; no new mandatory
dependency.

Negative / accepted: dt-halving convergence is first order (fine for a
display/fidelity layer, would need care in an accuracy budget); jsbsim
wheels exist only for CPython 3.10+/x86_64 macOS+Linux (fine per current
targets); program feasibility (rate-limited attitude) becomes **our** job,
not the simulator's; the 24° early transient is real and must be designed
around, not tested away.

## Follow-up epic draft: "6-DOF ascent & recovery tracking" (E5)

1. Promote the spike driver into `orbitopt.flightlab.sixdof` proper:
   contract-typed `SixDofRun`, config-driven models, no spike fixtures.
2. **Rate-feasible attitude program generator** (body-rate limits from plant
   authority `T·arm/I`) so the 6-DOF vehicle can fly its own commands;
   regression = FL-403 test with the early-window bound tightened.
3. Real aerodynamics: drag/ST tables on the stage, max-q regression, then
   gravity-turn ascent end-to-end vs 3-DOF with aero on both sides.
4. Staging through the contract: E4 recipe (FL-402) + booster reuse arc and
   upper-stage relight; two-body FlightRun v1 emission in the pipeline.
5. Recovery leg: reentry attitude authority check at declared heat-flux /
   q constraints (`docs/reentry_safety_constraints.md`), landing burn TVC.
6. Perf pass: sample decimation + streaming for the mission-control worker
   (docs/mission-control-worker-lifecycle.md), dt policy study
   (0.02/0.01/0.005) against the FL-403 convergence numbers.
7. Controller upgrade (LQR or gain-scheduled PID+FF on quaternion error)
   only if a driven use-case demands better than the FL-403 sub-degree RMS.

## If this GO is ever overturned, the fallbacks are

1. 3-DOF flight + kinematic attitude display (zero 6-DOF fidelity, zero cost).
2. RocketPy re-evaluation conditions: only if Python-native actuator model
   + aero tables maturity is demonstrated past JSBSim's, incl. the same
   two-instance staging story (RocketPy has no property bus; handoff would
   be a full state re-init each time).
3. In-house 6-DOF re-evaluation conditions: only if a hard requirement
   appears for differentiable or GPU-batched 6-DOF propagation (JSBSim is
   neither), i.e. 6-DOF enters optimisation loops, not just verification.

## Evidence index

- FL-401 findings + gotchas: `docs/flightlab-jsbsim-spike-findings.md` (§FL-401)
- FL-402 proven handoff recipe: same file (§FL-402); ContextDB `e4/staging-handoff/proven-recipe`
- FL-403 spike code: `src/orbitopt/flightlab/sixdof/tracking.py`; tests
  `tests/test_flightlab_sixdof_tracking_jsbsim.py` (thresholds declared at top)
- FL-403 artifacts: `docs/benchmarks/fl403_tracking_spike.json`,
  `docs/assets/fl403_{tracking,gimbal,meco_deviation}.png`, regenerated by
  `scripts/fl403_run_spike.py`; 3-DOF reference by `scripts/fl403_record_reference.py`
- Conventions: ContextDB `e4/jsbsim-spike/conventions`
