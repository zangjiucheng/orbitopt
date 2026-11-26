# Rendezvous theory

Four layers, each graded by the layer above it: a linear planning model, a
nonlinear pair propagator, a safety constraint set, and an independent
re-flight in a different physics stack.

## Relative motion in LVLH

All relative states are expressed in the target's LVLH (local vertical,
local horizontal) frame centred on the chief: x along the radial, y along
-track velocity (V-bar), z completing the right-handed set. A separation ρ
is the norm of the relative state. The frame convention and its
consequences are in
[ADR relative-0001](adr/relative-0001-lvlh-relative-frames-cw-and-pair-propagation.md).

## The CW planning model and its measured envelope

Clohessy–Wiltshire linearised relative motion is used for **planning only**
— never for acceptance. The companion benchmark
([study write-up](benchmarks/e12_02_cw_validity.md),
[raw data](benchmarks/e12_02_cw_validity.json)) sweeps separations
10 m … 100 km against the full pair propagator. The error:

* grows **quadratically with separation** — absolute error ≈ 10⁻⁴·ρ² per
  orbit on a circular chief (0.01 m at 10 m, 1.05 m at 100 m, 105 m at
  1 km) — so the *relative* error is ~1 % per orbit only near ρ ≈ 100 m
  and scales with ρ; there is no "1 % at any scale";
* grows ~t²;
* is dwarfed by any chief eccentricity ≥ 0.01 (129 % of a 100 m
  separation at one orbit).

The gate `tests/test_e12_02_cw_validity.py` pins the planning envelope:
circular chief, ρ ≤ 1 km, ≤ 1 orbit, error < 25 % of separation. Any
verdict runs on `propagate_pair`, not on CW.

## Acceptance propagation: `propagate_pair`

The acceptance model propagates chief and deputy together — two-body plus
J2, fixed-step RK4, chief-centred LVLH output — so J2 differential effects
appear without a linearisation assumption. The plan and its verification
share one model, one frame (GCRS) and the exact packaged Earth μ
(3.9860044150×10¹⁴ m³/s²), enforced at the config boundary
([ADR rendezvous-0001](adr/rendezvous-0001-two-impulse-j2-targeted-lambert.md)).

## Mission geometry

A coplanar chaser sits 10 km below the target so the orbit-rate drift closes
the initial 20 km separation until the chaser crosses the departure gate
(8 km aft, on V-bar). There the planner solves a two-pulse Lambert transfer,
J2-targeted with a Newton correction on burn 1, that arrives **hovering** at
the hold point — 1 km V-bar-aft — with zero relative velocity. Δv budget and
both time windows are fixed in the config before any optimisation runs
([mission configuration](rendezvous-mission-config.md)).

## Whole-trajectory safety

Safety constraints gate the **entire** trajectory, not just the terminal
approach
([ADR rendezvous-0002](adr/rendezvous-0002-safety-constraints-whole-trajectory.md)):

* a 500 m keep-out sphere around the target at every sample;
* a corridor bounding cross-track excursion (±100 m) and forbidding forward
  excursion inside a terminal cone;
* a range-dependent relative-speed ceiling (piecewise-linear in range);
* passive-contingency checks: a no-thrust coast from t0, the gate and the
  hold point must still stay clear.

Checks run on a dense grid (2 s) with burn epochs always sampled, and the
verdict reports per-check margins.

## Independent verification and robustness

A TudatPy two-spacecraft re-flight
([ADR rendezvous-0003](adr/rendezvous-0003-independent-refly-tudatpy.md))
re-flies the pair in a different physics stack, graded only when maneuver
digests match the sealed plan. A dispersion campaign
([ADR rendezvous-0005](adr/rendezvous-0005-dispersion-campaign-not-a-safety-factor.md))
prices the locked design under stress. The Mission Studio replay
([ADR rendezvous-0004](adr/rendezvous-0004-mission-studio-flightlab-replay.md))
re-propagates the sealed plan locally for inspection — it is a replay, not
an independent check. Recorded numbers, with nominal, independent and robust
results kept separate: [verification report](rendezvous-verification-report.md).
