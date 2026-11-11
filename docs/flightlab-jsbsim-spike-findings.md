# Flight Lab E4 — JSBSim 6-DOF spike findings (FL-401, #206)

Input for the E4 go/no-go ADR (Epic #196). Code: `src/orbitopt/flightlab/sixdof/`,
model: `src/orbitopt/data/flightlab/jsbsim/`, checks:
`tests/test_flightlab_sixdof_jsbsim.py` (thresholds declared at the top).

## What was built

- `spike_stage`: one liquid stage written from scratch (J246 read for structure
  only). 10 t dry + 60 t LOX/fuel (MR 2.5), Isp 320 s, 600 kg/s → 1.883 MN
  vacuum thrust, 100 s burn. Rocket engine + nozzle thruster with `<orient>`,
  two FCS actuator channels (lag τ = 0.2 s, 0.35 rad/s slew limit, ±5° travel)
  driving `propulsion/engine[0]/{pitch,yaw}-angle-rad`. Zero aerodynamics,
  no ground reactions, token nozzle exit area (vacuum thrust at any altitude).
- `SixDofSim` driver (jsbsim 1.3.3 Python API): geodetic/attitude ICs, ignite,
  throttle, gimbal commands, per-step sampling in SI, export to a contract
  `BodyTrack` (frame `ECEF`) with extra columns: quaternion (from `Tec2b`),
  body rates, gimbal cmd/actual (pitch, yaw), thrust, throttle, inertia,
  CG, altitude, TVC moments.

## What worked (measured, dt = 0.01 s)

| Check | Result | Declared tolerance |
|---|---|---|
| Vertical hop Δv vs. Tsiolkovsky − ∫(g − ω²r) dt, 30 s, throttle 1.0 / 0.5 | 2e-4 rel | 1e-3 |
| Thrust vs. Isp·g0·ṁ | exact (1e-9) | 1e-4 |
| Mass vs. m0 − ṁt | within one frame of flow | 1 frame + 1 kg |
| Mass, Ixx/Iyy/Izz strictly decrease; CG moves aft (LOX forward) | yes | — |
| Engine starves at depletion, mass → dry | yes | 1 kg |
| TVC moment vs. T·sin δ·(x_thr − x_cg) | exact | 1 % |
| q(t) vs. ∫ M/Iyy dt (1 s step) | 0.6 % | 2 % |
| Actuator vs. first-order lag (Tustin, half-step delay) | 0.03 % of cmd | 1 % |
| Rate limit / travel clip honoured and active | yes | 1e-6 |
| dt halving (0.01 → 0.005 s, 20 s gimballed burn): Δpos/flown, Δv/speed, attitude | 1.1e-3, 6e-4, 0.027°; body rates 1.4e-5 rel | 2e-3, 2e-3, 0.05°; 1e-4 |

Runtime: ~5 k steps/s single-threaded (whole test file ≈ 5 s).

## What didn't / gotchas and workarounds

1. **`set_dt()` must come before `load_model()`.** FCS components (actuator
   lag/rate filters) cache the executive's dt at construction; calling
   `set_dt` afterwards silently leaves the FCS at the default 1/120 s while
   the dynamics use the new step (lag response came out dt-independent *per
   frame*). Workaround: driver calls `set_dt` first.
2. **Locations need `<x>/<y>/<z>` child elements.** `<location unit="M"> 1 2 3
   </location>` parses without error as (0, 0, 0) — CG, tanks and thruster all
   collapsed onto the origin, so TVC produced zero moment. No warning.
3. **Aero axis names** are `DRAG/SIDE/LIFT` (+ `ROLL/PITCH/YAW`); `MOMENT` etc.
   are rejected. All three force axes are needed even when zero, and a
   `<ground_reactions>` element must exist (empty is fine).
4. **Driver-written properties must be declared** (`<property>` in
   `<flight_control>`) or `run_ic()` fails.
5. **Gimbal sign:** positive `pitch-angle-rad`/`yaw-angle-rad` on an aft
   thruster gives nose-down / nose-left moments; the FCS applies gain −1 so
   positive command == nose-up (q > 0) / nose-right (r > 0).
6. **One-frame latencies:** the engine lights one frame after the ignition
   command (thrust 0 on frame 1) and mass reports one frame stale. Both are
   O(dt) and make the integration converge only **first order** in dt
   (errors halve with dt) — acceptable for the spike, worth noting for any
   accuracy budget.
7. **Body rates are relative to the rotating ECEF frame** (`velocities/p,q,r`);
   inertial rates are `pi/qi/ri`. Expect ~1e-8 rad/s cross-axis rates from
   Earth rotation.
8. **No quaternion property**; use `fdm.get_propagate().get_Tec2b()` (returns a
   deprecated `numpy.matrix`; driver suppresses the PendingDeprecationWarning).
   Euler angles are gimbal-locked for a vertical vehicle.
9. **Packaging:** `jsbsim==1.3.3` is pip-only for our targets; kept out of the
   conda locks (lock tests forbid unlocked `pip:` sections) and shipped as the
   hash-pinned `envs/sixdof-requirements.txt` + `orbitopt[sixdof]` extra.
   Inline `<engine>`/`<thruster>` definitions were not used — engine and
   nozzle live in `engine/*.xml` referenced by `file=`.
