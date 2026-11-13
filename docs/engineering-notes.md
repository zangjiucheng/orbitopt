# Engineering notes

Implementation decisions, numerical pitfalls and evidence for orbitopt's
mission pipeline. Measurements below describe the named examples and hardware;
they are not accuracy guarantees for other inputs.

## pygmo: `bfe` must be attached before wrapping

Attach a batch fitness evaluator to the raw UDA before wrapping it in
`pg.algorithm`. Not every pygmo algorithm supports `set_bfe`;
`optimize.runner` handles the supported cases. A missed attachment can fall
back to serial evaluation.

## SPICE: `J2000` and `ECLIPJ2000` are not interchangeable

The frames differ by Earth's approximately 23.4° obliquity. Mixing their
states can cause errors of hundreds of thousands of kilometers. Check frame
and origin at every handoff; the orbital ephemeris paths use `ECLIPJ2000`.

## RK4 step size that's fine for a coast is not fine for a flyby

Near a close flyby, a coast-sized step can miss the encounter. One documented
case produced 81,733 km closest approach with a 600 s RK4 step versus 4,379 km
at steps of 15 s or less. Repeat convergence checks for the geometry being
studied; a fixed step is not a universal accuracy guarantee.

## Verification layer: adaptive integrator + event location (task #136)

The verification propagator defaults to tudat's adaptive RKDP 8(7)
(`integrator="dop853"`); `rk4` is available for comparison. Events use
root-finding on dense output, with fine re-propagation where supported. In the
Keplerian example in `examples/11_event_location_comparison.py`, a 600 s
sample interval missed a 1,837.400 km perilune by 48.42 km; event location
recovered it to within 1 mm. The result is example-specific. An encounter
whose range rate does not change sign between samples can fall back to a
sample-wise minimum.

## Verification report: every verification emits machine-readable JSON (task #137)

Reports include residuals, integrator settings and convergence, conservation
checks, the force model and omitted effects. Energy and angular momentum are
not generally conserved in a perturbed trajectory; the applicable comparison
uses a central-body-only re-run. Reports use the
`verification-report-1.0.json` schema. `orbitopt run` writes reports next to
the scene; `ORBITOPT_VERIFICATION_REPORT_DIR` also enables library-call
report output.

Every check is filed under one of three categories (task #243), so a pass
never blurs what was actually demonstrated. *Numerical consistency* — the
integration and targeting did what the equations say (step convergence,
conservation, converged targeting). *Model agreement* — this model's result
matches a reference model, aim point or corridor (the default for residuals).
*Historical validation* — the simulation matches a published, flown reality
(Starship's announced timeline; a flown mission's telemetry once that exists).
Producers must classify every check (`status_from_checks` refuses otherwise);
the status carries a per-check map and a per-category roll-up, and the
Verification tab lists checks under their category header. Bundles roll the
categories up across reports. The addition is additive in the 1.0 schema:
documents without categories still validate and render as before.

## GPU is not always faster -- measured, not assumed

GPU performance depends on batch size and kernel-launch overhead. Precomputing
Moon/Sun ephemerides removed repeated solves from the screening loop; an
optional fused Numba CPU backend accelerates closest-approach screening. See
the [backend benchmark](benchmarks/fused_kernel_evaluation.md) for the host,
timings, accuracy and unmeasured CUDA estimates. Measure the target workload
before changing backend defaults.

## Undamped Newton diverges on real (imprecise) inputs, not just in theory

A coarse lunar-flyby seed once sent an undamped Newton solve into divergence.
The targeter checks refined coast guesses and caps and backtracks Newton steps.
`tests/test_cislunar.py` exercises the previously failing seed; convergence is
still input-dependent.

## Qt/VTK: which screenshot API actually captures the 3D view

`QWidget.grab()` may return a black embedded OpenGL view. Use PyVista's
`plotter.screenshot()` for the viewport and composite it with the window
capture. `scripts/capture_screenshots.py` handles this, including Retina
logical-coordinate placement. VTK tests need a display on platforms where
Qt's offscreen plugin cannot create the OpenGL context.

## VTK point markers: constant apparent size, and a bug this sidesteps

VTK's `render_points_as_spheres` renders markers at a constant screen-pixel
size, avoiding world-space scaling errors when the camera zooms.

## PyVista slider: `interaction_event` defaults to release-only

`add_slider_widget` defaults to callbacks on release. Use
`interaction_event="always"` when dragging must update continuously; see
`tests/test_pv_viewer.py`.

## Cislunar / Artemis II free-return pipeline, in full

The built-in mission uses an Earth-only Lambert seed followed by joint lunar
B-plane and free-return targeting in Earth/Moon/Sun dynamics. An independent
re-propagation checks the entry-interface angle against [−7°, −5°]. This is
a **vacuum geometric check**, not an assessment of atmospheric heating, loads
or survival; see [re-entry constraints](reentry_safety_constraints.md).

The 6,545 km perilune is a target input. The 2026-08-01 departure and 185 km
parking orbit are modeling choices, so the result is not a reconstruction of
the real Artemis II flight. The older `examples/05_artemis2_free_return.py`
targets flyby distance only and does not guarantee a return.

## Window search: one porkchop minimum is not one basin

`optimize.multistart` identifies separated local minima, optimizes each and
also searches the full window. It records the single-basin baseline and merges
duplicate optima. MGA window screening likewise keeps diverse first-leg
basins and re-ranks them using full MGA runs. `tests/test_multistart.py`
contains a two-valley regression case.

## Launch constraints in the fitness: measured effect on the optimum (task #153)

`LaunchConstraints` applies vehicle C3 and site declination limits in scalar
and batch fitness. It rejects payloads outside the tabulated C3 curve rather
than extrapolating. Invalid candidates receive graded violations; penalty mode
requires a weight large enough to discourage violations.

In the documented 2028 Earth–Mars example (Falcon 9, 4,000 kg), the
unconstrained optimum exceeded the C3 limit: 9.09 versus 9.00 km²/s². The
feasible solution departed about 2.4 days later and cost 26.8 m/s more. See
`examples/12_launch_constrained_transfer.py` and
[launch-data sources](launch-data-sources.md); these inputs are planning-level
estimates.

## Epochs: one leap-second-aware UTC -> TDB conversion (task #159)

`orbitopt.timescales` treats MJD2000 as a UTC label and converts to TDB seconds
past J2000 noon using leap seconds and the TDB−TT correction. It is checked
against SPICE and ERFA in `tests/test_timescales.py`. Earlier paths disagreed
by roughly 69 s for modern epochs, so frame and epoch conversions must use
the shared implementation.

## Unified mission pipeline (task #170)

`pipeline.run_pipeline` runs the configured lifecycle:

```text
config → constraints → search → optimize → select → refine → verify → export
```

Its `PipelineResult` records per-stage status, candidates, physical models,
numerical checks and timing. Unused stages are `skipped`; stages after a
failure are `not_run`. The CLI also writes a pipeline JSON report. Stage
recording is implemented for `geo-raising` and `mars-transfer`.

## Candidate Selection and the Verification Gate (task #171)

`candidate_selection.py` removes infeasible and near-duplicate candidates,
ranks the remainder and favors diversity when applying the candidate limit.
The verification gate records whether each candidate was accepted, rejected,
degraded or not evaluated. A failed or unchecked candidate is not an
accepted success.

For Mars, if no selected candidate passes, the pipeline can export a degraded
or fallback trajectory as **best effort**, with a warning and `success: false`.
The gate, scene and CLI carry the warning. A completed computation alone does
not imply a passed verification.
