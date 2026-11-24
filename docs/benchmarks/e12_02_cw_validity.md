# E12-02 — CW validity envelope (task #392)

Epic #389. [`physics_io.py`](../../src/orbitopt/physics_io.py) previously
rejected LVLH/VVLH as "a body-fixed orbital frame, not an Earth-centred
one". This task adds relative-frame support — LVLH/RTN states, ECI ↔ LVLH
transforms, the Clohessy–Wiltshire model and a nonlinear two-spacecraft
propagator (`orbitopt.relative`, see
[relative ADR-0001](../adr/relative-0001-lvlh-relative-frames-cw-and-pair-propagation.md)) —
and acceptance needs the number this benchmark exists to produce: **CW is
only a fast planning model — where exactly does it stop being one?** The
machine reading is [`e12_02_cw_validity.json`](e12_02_cw_validity.json);
reproduce with
`PYTHONPATH=src python benchmarks/bench_e12_02_cw_validity.py -o …`
(≈2 min, CPU only, numpy only).

## The experiment

A 700 km, 51.6° chief; a deputy at rest in LVLH at a radial offset (the
classic worst case: a free hover point, station-keeping off). The nonlinear
baseline is `orbitopt.relative.propagation.propagate_pair` — RK4, two-body
with and without J2, 2000 steps per chief period — and CW uses the only
mean motion it legally has, `n = √(μ/a³)` of the reference chief. Grid:
separations 10 m … 100 km, durations 0.25 … 5 orbits, chief eccentricity
0 … 0.2 at fixed `a`, both J2 variants. Every error is the max
relative-position difference CW vs baseline over the window. The baseline
carries its own accuracy check: the reference case re-integrated at half
the step size agrees with it to 8.9·10⁻⁵ m — four orders below every CW
error reported, so the data measures the model, not the integrator.

## What the numbers say

* **Separation: linear, as the Taylor remainder promises.** J2 off,
  e = 0, 1 orbit: 0.01 m at 10 m, 1.05 m at 100 m, 105 m at 1 km,
  10 512 m at 10 km — the *normalised* error is one percent of the
  separation per orbit at any scale, so small formations do not buy
  accuracy, they buy proportionally less error.
* **Duration: ~ t².** Same cell, 1 orbit → 5 orbits: 1.1 m → 25 m (24×).
  The along-track linearisation error accumulates as drift × time; a CW
  plan must be short, not merely small.
* **Eccentricity dominates everything else.** At 100 m, 1 orbit, J2 on:
  e = 0 gives 6 m, e = 0.01 gives 129 m (129 % of the separation),
  e = 0.05 gives 724 m, e = 0.2 gives 4 141 m. CW has no eccentric chief —
  the J2-off variant shows this is not oblateness, it is the model.
* **J2 is the second-order tax.** J2 on vs off at e = 0, 100 m, 1 orbit:
  6.01 m vs 1.05 m. Both matter at km separations; neither is inside a
  plan's budget at 5 %.

## The envelope (the acceptance artefact)

Longest duration whose CW error stays under 25 % of the initial separation,
J2 on, circular chief: 2 orbits at ≤ 100 m, 1 orbit at 1 km, half an orbit
at 10 km, a quarter orbit at 100 km. Under the stricter 5 %: 0.25 orbits
even at 10 m (J2 noise). **Any chief eccentricity ≥ 0.01 leaves the CW
envelope at a quarter orbit or less at every separation tested.** The
acceptance rule encoded in
[`tests/test_e12_02_cw_validity.py`](../../tests/test_e12_02_cw_validity.py):
CW may plan a circular-chief, ≤ 1 km formation over ≤ 1 orbit with error
bounded by 25 % of the separation, and even then only when the grid itself
says so — e = 0.01 already breaks 5 % of the separation at a quarter orbit,
so no measurable eccentricity gets a pass. Any acceptance-grade trajectory
comes from `propagate_pair`, not `propagate_cw`.
