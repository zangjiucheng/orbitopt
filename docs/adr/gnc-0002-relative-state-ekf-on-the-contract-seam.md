# ADR gnc-0002: a relative-state EKF on the contract seam (E13-02)

- **Status:** Accepted
- **Date:** 2026-11-27
- **Deciders:** E13 Closed-Loop GNC & Relative Navigation (Epic #416), E13-02 (#419)
- **Code:** `src/orbitopt/gnc/ekf.py`, `src/orbitopt/gnc/nav_metrics.py`,
  `benchmarks/bench_e13_02_relative_nav_ekf.py`
- **Evidence:** `tests/test_e13_02_relative_nav_ekf.py`,
  `docs/benchmarks/e13_02_relative_nav_ekf.{json,md}`
- **Builds on:** ADR relative-0001 (CW + `propagate_pair` + LVLH frames), ADR gnc-0001
  (`Observation`/`Estimate` contracts, uniform clock, provenance rule)

## Context

Closed-loop guidance (E13-04) needs a navigation state with a covariance, not just a propagated
one: the acceptance for this task is statistical (RMSE, NEES/NIS inside declared chi-square bands,
coverage, observability, behaviour under model mismatch, multi-seed clusters). Three questions had
to be settled before any number could mean anything:

1. **In which frame does the filter live?** The planning model the whole epic reuses (E12-02 CW)
   is defined in LVLH; process-noise intuition (white acceleration) is frame-fixed; the contract
   carries inertial vectors.
2. **Where does the process model come from?** Duplicating CW or the pair propagator inside the
   filter would put a second source of truth next to E12-02 — the exact drift this repo's ADRs
   keep killing.
3. **What is a sensor measurement?** The `Observation` contract carries a Cartesian relative state.
   A range/azimuth sensor does *not* measure that; re-deriving angles from the Cartesian field and
   then "correcting" with the filter would double-count the noise the record already contains.

## Decision

1. **Filter state = 6-vector LVLH `(ρ, ρ̇)`**; control `u` is a LVLH Δ-v impulse entering through
   `B = [0; I]`. Publishing through `make_estimate` converts state and covariance to the contract's
   inertial frame with the chief's state at the publishing instant; the covariance is mapped by the
   linear chief-fixed change-of-frame Jacobian (`frame_covariance`, finite-difference-verified
   against `lvlh_to_eci`).
2. **The process model is a pluggable pair `(f, F)` that delegates.** `CwProcess` calls E12-02's
   `propagate_cw`; `NonlinearRelativeProcess` calls `propagate_pair` with the reference chief it
   advances itself (the chief's motion is the nominal row of the Jacobian batch — no extra
   propagation), `F` by central differences around that batch. Tests assert both against the E12-02
   callables directly: the filter reuses E12-02's dynamics, it does not restate them.
   Q is the piecewise-constant white-acceleration integral (`white_noise_acceleration_q`).
3. **Measurement models are components with a `dof` property** (`RelativePosition`,
   `RelativeVelocity`, `RangeRate`, `RangeAngle`, `Stacked`). Cartesian extraction converts the
   contract's inertial reading to LVLH through the caller-supplied chief state (the model refuses
   without it). Native sensors are read from `Observation.metadata` under the declared keys
   `native_measurement` / `native_measurement_covariance` — the extract **refuses** to re-derive
   angles or ranges from the Cartesian field (message says why: double-counting the noise).
4. **Standard numerics, one report type.** Joseph-form update, symmetrised covariances; every
   update returns an `UpdateReport` (innovation, innovation covariance, NIS, ids consumed) because
   the acceptance metrics consume the filter's own outputs, not a reimplementation.
5. **Acceptance is a gated Monte-Carlo JSON.** 70 cells × 5 seeds
   (`docs/benchmarks/e13_02_relative_nav_ekf.json`): matched-matrix, mismatch (CW/TwoBody/J2/drag
   truth vs CW/nonlinear filter), five sensor families, two geometries; declared χ²(6) 90 % NEES
   band and per-dof NIS bands; 3σ coverage; R-weighted 20-epoch stacked observability sweep.
   Tests gate consistency only on Cartesian cells — the angle-cell NEES loss is a *reported
   finding* (linearisation bias at 30 m prior error), and gating it would encode the bug.

## Consequences

- Matched Cartesian cells are consistent to the declared gates (NEES pass ≥ 0.79, coverage 1.000);
  the mismatch study shows NEES dropping to 0.12–0.14 under unmodelled differential drag — the
  filter *detects itself* to be wrong, which is the property the closed loop will key off.
- Native angle sensors carry a linearisation bias the plain EKF cannot fix (NEES 0.22–0.51 on
  matched truth): E13-04 adopts an iterated/UK update or accepts it; the JSON is the baseline.
- The drag mismatch is fixable only in the model: a drag state or Q inflation sized to the
  force — both are `f`/`Q` plug-ins behind this interface, no filter change.
- `NonlinearRelativeProcess` costs 13 batched pair propagations per update (f + 12 FD columns):
  exactness first, an analytic-variational `F` can replace the batch later without API change.

## Alternatives rejected

- **Inertial-frame EKF:** the CW planning model (and its Q) are LVLH objects; an inertial filter
  either loses the cheap model or linearises CW into inertial frames — second-order bookkeeping
  for no observed gain at 300 m.
- **Error-state (KF in the error) formulation:** the absolute-vs-relative errors here are small
  (tens of metres at 300 m separation); a full-state EKF is simpler and the mismatch study already
  shows where linearisation actually bites (angle updates, not state magnitude).
- **Re-deriving native observables from the Cartesian field:** would make every native-sensor cell
  look as good as Cartesian while lying about the noise model; refused at the extraction boundary.
