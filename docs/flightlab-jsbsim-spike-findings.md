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

---

# Flight Lab E4 — staging handoff spike (FL-402, #207)

## What was built

`orbitopt/flightlab/sixdof/staging.py` + two child aircraft models
(`spike_booster`, `spike_upper` — the aft 25 m / forward 20 m segments of
`spike_stage`). `run_staging_scenario()` flies the stack 20 s (10 s in tests),
freezes the state (`capture_frozen_state`: ECEF position/velocity, Tec2b
quaternion, body rates, empty+tank mass), splits dry mass and remaining
propellant (`StageSplit`), spawns each child as a **fresh top-level
`FGFDMExec`** initialized purely through `ic/...` properties (+ an optional
momentum-conserving separation impulse along body +x), then steps both
instances interleaved in one process for >= 60 s. `to_flight_run()` emits a
valid three-body FlightResult v1 run (`stack` + two children parented to it,
children re-based onto the stack clock).

## Verdict: works — two independent instances side by side is viable

With a 2e5 N.s impulse, dt = 0.01 s, 61 s coast (all measured):

* **Mass closure:** children's total mass = frozen stack mass to 8.8e-4 kg
  (1.4e-11 relative; the residual is JSBSim's internal lb rounding).
* **Momentum residual:** 0.16 kg·m/s against |p| = 1.17e7 (1.4e-8 relative;
  declared tolerance 1e-6 relative + 1 N.s floor). The impulse itself is
  applied exactly bookkeeping-wise (+J upper / -J booster).
* **Handoff continuity:** position to 2.1e-7 m, attitude bit-identical
  (0 rad), velocity to <1e-6 m/s.
* **Determinism:** full sample logs bit-identical across reruns.
* **Step order:** `booster→upper` vs `upper→booster` sample logs bit-identical
  — the instances share nothing but the process.
* **No leakage:** a property written on one instance raises on the other;
  throttle/gimbal commands on one leave the other's samples untouched.

## Gotchas and workarounds (jsbsim 1.3.3 Python bindings)

1. **No state-injection API.** `FGPropagate` exposes only `dump_state` and
   DCM getters; everything must go through `ic/` properties + `run_ic()`.
2. **`ic/h-sl-ft` is geoid-referenced** while the frozen state is ECEF, and
   `ic/h-agl-ell-ft` is not recognized. Workaround: fixed-point iteration on
   (latitude, h-sl) against the achieved ECEF position (Bowring gives the
   start point; converges to ~1e-7 m in <= 3 `run_ic()` calls).
3. **ECEF velocity injection** goes through `ic/vn|ve|vd-fps` (NED at the
   freeze point); the rotating-frame ECEF match is exact (<1e-6 m/s), since
   `velocities/ecef-*-fps` are themselves rotating-frame.
4. **Euler IC from the quaternion** (Euler is gimbal-locked vertical): rotate
   `Tec2b` into NED (`Tbn = Tec2b · R_n2e`) and read the 3-2-1 angles;
   round-trips to ~1e-10 rad. Tank contents written *before* `run_ic()`
   survive and are used — that is the mass-split injection point.
5. **`inertia/mass-slugs` runs one frame (mdot·dt) high while the engine
   burns** (already seen in FL-401): the frozen stack mass must be taken as
   `inertia/empty-weight-lbs + Σ tank contents`, else the mass/momentum
   bookkeeping shows a spurious mdot·dt (6 kg here) mismatch.
6. **`--` inside XML comments is fatal**: JSBSim's parser raises
   "not well-formed (invalid token)" on `load_model` (verified) — the child
   model comments avoid double hyphens (ASCII prose with em-dashes is fine).

---

# Flight Lab E4 — attitude tracking spike + go/no-go (FL-403, #208)

## What was built

`orbitopt/flightlab/sixdof/tracking.py`: an SO(3) geodesic attitude-error
PID in angular-acceleration space with pitch-rate feed-forward, converted
to gimbal angle through the *instrumented live* plant gain `T·arm/I`
(thrust, Iyy/Izz, CG read back from the model each step — a gain schedule
that cancels the burn's mass/CG/inertia sweep); plus the 3-DOF reference
plumbing (OT `trajectory.csv` replay, ECEF-climb altitude comparison that
cancels the geoid offset). Reference: `fl403_spike` fixture, a run of the
OT vehicle mirrored from the spike stage (same masses/Isp/ṁ/burn, zero
aero); regenerable with `scripts/fl403_record_reference.py`. Plots + all
metrics regenerate via `scripts/fl403_run_spike.py`.

## Numbers (dt = 0.01 s; full table in the ADR)

Settled (t ≥ 60 s) attitude error **1.12° max / 0.52° RMS**; gimbal
saturation **0.0 %** of burn, peak deflection 0.86° of ±5° travel; MECO
deviation vs 3-DOF: **−4.7 km alt / −90 m/s / −0.25° FPA**; dt-halving
agreement 40 m / 0.58 m/s / 0.006°; no non-finite states to MECO+5 s at
Mach 18. Full-profile error peaks at 24° in the first 60 s — the 3-DOF
program's near-vertical elevation slew is kinematically unachievable by TVC
(cos(el) → 0), an open-loop FL-401 run shows the same lag with no controller:
a property of the program, not of the loop. Production programs must be
rate-limited in body rates.

## Verdict

**GO** — see `docs/adr/flightlab-0001-jsbsim-6dof.md` for the declared
criteria, the evidence table and the E5 draft scope.
