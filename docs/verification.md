# Verification and benchmarks

Reports record checks, tolerances, force models and numerical settings. A
pass applies to those assumptions; it does not establish flight safety.
Mission Control displays **VERIFIED**, **VERIFY FAILED** or **NOT CHECKED**.
For the reviewed E1–E4 integration and its fixes, see the
[2026-11-15 acceptance report](acceptance-e1-e4.md).

## What a report contains

The `verification-report-1.0` document contains residuals, omitted effects,
integrator settings, provenance and a canonical SHA-256 digest. Reports are
written beside mission scenes and by Flight Lab, `orbitopt launch` and the
reference-scenario chain. Inspect both the overall result and its evidence.

## Three kinds of evidence

The check category describes its purpose:

| Category | Purpose |
|---|---|
| `numerical_consistency` | Convergence or conservation within a model. |
| `model_agreement` | Agreement with another prediction, model or constraint. |
| `historical_validation` | Agreement with published real-world data. |

A category without checks is **not checked**.

## Five evidence types (task #263)

Evidence type identifies the reference used by the check:

| Type | Reference | Scope of a pass |
|---|---|---|
| `analytical` | Closed form of the same/simplified model. | Agreement with that solution. |
| `independent_model` | Separately formulated propagator or model. | Cross-model agreement over the checked phase. |
| `historical_telemetry` | Published flight observations, not model inputs. | Agreement with those observations. |
| `design_input` | Target orbit, aim point or declared envelope. | The solution satisfies its design constraints. |
| `numerical_consistency` | Same model under independent/tighter numerics. | Numerical error or conservation checks pass. |

`status.check_evidence` maps checks to types; `status.evidence` rolls them up.
`status.independent_validation` includes only `independent_model` and
`historical_telemetry`. Its `not_checked` result means no independent evidence
was included, even if the report's overall result is green. Independent-model
agreement still does not establish real-world validity.

### Design-input comparisons and why they must be labelled

Hitting the orbit or corridor the optimizer targeted is `design_input`, not
historical validation. Producers must supply that evidence override rather
than inheriting the `model_agreement` default of `independent_model`.
Missing references or skipped checks must remain visibly unassessed.

### The three reference benchmarks

| Benchmark | Evidence |
|---|---|
| Kepler two-body | `analytical`; energy drift is `numerical_consistency`. |
| J2 secular drift | `analytical`. |
| Starship Flight 15 booster catch | `historical_telemetry` against its published timeline. |

The catch contributes two independent-validation checks; the other five
benchmark checks are analytic or numerical. This is scoped evidence, not a
claim that every mission phase was validated.

## Full mission verification gate (E2-03, task #256)

`inputs.full_mission` grades three phases separately:

| Phase | Checks / applicability |
|---|---|
| `orbit_injection` | Ascent verdict, station order, staging, mass and expected orbit; always applicable. |
| `parking_propagation` | M-05 propagation; out of scope only when `expected.propagation_result: not_checked` is declared (refused when `full_mission` is declared). |
| `recovery_touchdown` | Recovery/handoff/touchdown limits; applicable when `recovery:` is declared (always applicable when `full_mission` is declared). |

Among applicable phases, **failed** takes precedence over **not_checked**,
which takes precedence over **passed**. An applicable phase with no checks
is `not_checked`. On a run that does not declare `scenario.full_mission`, a
declared out-of-scope phase does not block the verdict. The chain's
diagnostic `status` and the pipeline's `full_mission_verdict` are separate
fields; inspect the phase document when a run is only partially successful.

### Verification coverage gate (E8-03, task #283)

A scenario may declare `scenario.full_mission: true`: the run *is* the
complete physical mission, so all three phases above become **required
physical phases** and cannot be taken out of scope by declaration:

- `expected.propagation_result: not_checked` is **refused** — the phase
  stays applicable, records no evidence and reads `not_checked`;
- a missing `recovery:` block likewise leaves `recovery_touchdown`
  `not_checked`;
- an applicable-but-ungraded required phase caps the gate verdict at
  `not_checked`, so the out-of-scope route can no longer produce an
  overall pass for a mission that never propagated to its parking orbit
  or never brought the booster home.

Newly generated gate documents carry the coverage of required phases —
`inputs.full_mission.coverage` lists a per-phase status
(`checked`, `failed` counts as checked, `not_checked`,
`declared_out_of_scope`, `not_declared`) and a `summary` like `2/3`, which
the pipeline verify stage records (`full_mission_coverage`) and echoes in
its note — and a
`inputs.full_mission.qualification` block. The report's overall description
may claim **independently verified** only when `qualification.independently_verified`
is `true`: the run declared `full_mission`, the gate verdict is `passed`
with complete `3/3` coverage, and the run's independent-evidence rollup
(`independent_validation`) passed. A partial run (e.g. an expendable
mission) never qualifies, however green its applicable phases — its
coverage summary says exactly how far the run reaches.

The chain preserves `ascent.physics_verified` as independent-model evidence:
`true` passes the check; `false` fails orbit injection and the full-mission
gate. `null` remains explicitly unassessed and contributes no independent
validation. A green design-constraint rollup with a null flag does not prove
independent verification. See the [acceptance report](acceptance-e1-e4.md).

### Full physical mission gate (E5-04, task #271)

That out-of-scope exemption is exactly what a *computed* mission must not
hide behind (E2-04 declared the parking propagation out of scope and still
passed). `inputs.full_mission.physical` re-grades the same chain checks as
**five required physical checks** — no applicability escape hatch:

| Required check | Evidence (the chain's existing checks) |
|---|---|
| `orbit_insertion` | station chain, ascent verdicts, expected-orbit agreement |
| `parking_propagation` | M-05 propagation (a `not_checked` *declaration* is not evidence) |
| `landing` | recovery touchdown/reserve/handoff — an unflown first stage caps at `not_checked` |
| `mass_conservation` | every mass gate wherever it is measured (`mass.*`, boundary `*.mass_conservation`) |
| `epoch_frame_continuity` | boundary continuity gates (`handoff.*`, `parking.*`: epoch, frame identity, coupled shared source) |

Each check carries its own tri-state verdict and its own reason codes (a
missing check reports `<check>_not_evidenced` with a reason read from the
scenario/mission notes). The overall verdict is the same conjunction as the
phase gate — failed beats not_checked beats passed — so **any required check
that is `failed` or merely `not_checked` keeps the physical gate from
passing**; only all-five-passed yields `passed`. The pipeline records it as
`physical_mission_verdict`; the E2-03 phase verdict is reported alongside and
unchanged, so a frozen mission like E2-04 still reads `passed` per phase while
the physical gate honestly says `not_checked`.

### The reproducible reference mission (E5-05, task #272)

The reference is now committed under
`tests/data/mission/reference/e5-05-reference-mission/` (recording commit
`64f64a6`). It uses the RTLS vehicle redesign, a fixed-program computed OT
ascent and recovery driven from the canonical separation handoff. The
recorded orbit is **265.5 × 278.1 km at 51.0°**; the earlier blocked 400 km
recording proposal is superseded.

The sealed single-environment report passes its applicable phase checks but
has **physical verdict `not_checked`**, with parking propagation not
assessed. It predates the E8-03 coverage/qualification fields. A current
cross-environment run must supply the missing TudatPy propagation evidence
and satisfy the physical gate; a stored green report alone does not do so.
See [running and viewing the reference](computed-mission.md) and the
[regression procedure](flightlab-regression-ci.md#e5-05-the-reproducible-full-physical-reference-mission-task-272).
E2-04 remains the synthetic integration fixture.

## Full-chain reference validation (E8-01, task #281)

E5-05 commits the full-physical reference mission; E8-01 grades that
*recorded chain* — MECO, separation, fairing jettison, SECO/insertion, the
RTLS boostback/entry coast and the landing — against the independent
references committed with it, in one `full-chain-reference` report
(`python -m orbitopt.verify.full_chain`, gated by
`tests/test_e801_full_chain.py`). Every band and its derivation are listed
in the [reference tolerance audit](reference-tolerance-audit.md#full-chain-reference-bands-e8-01-task-281);
the PROVENANCE digest check re-runs on every build, so the report can never
be graded off edited data. No simulation runs: all compared numbers are the
committed reference files.

The E4-03 evidence split is honest about what is what:

* `independent_model` — the shared unpowered ballistic window: the raw
  OpenTsiolkovsky stage-1 debris track against the gfold recovery leg between
  separation and boostback ignition — two independently formulated models of
  the same physical arc. Since E8-05 (task #294) this is the only
  independent-model check in this report: the ascent event times and the
  insertion elements compared against OpenTsiolkovsky's *own raw output*
  compare the chain against the program that computed the ascent, so they are
  labelled `numerical_consistency` (event extraction from the raw state
  history) and `analytical` (Kepler identities over the recorded state), and
  are not counted as independent-model evidence.
* `analytical` — vis-viva/Kepler identities of the recorded SECO state,
  the Tsiolkovsky ideal-Δv bound on the stage-2 burn, and exact two-body
  propagation of the boostback coast to the entry interface.
* `numerical_consistency` — SECO-state serialisation/bridge round-trip,
  landing-ignition and touchdown events against the sealed booster state
  history.
* `design_input` — the touchdown gates, graded against the flight target's
  own limits. The landing of a synthetic vehicle has no independent or
  telemetry reference, and no check claims one
  (`historical_telemetry: not_checked`).

Because the ascent was computed *by* OpenTsiolkovsky, the event checks
validate the chain's extraction, framing and reporting of the external
recording rather than two independent ascent simulators; the recovery leg
is where two independently formulated models (the 3-DOF recovery model vs the pinned
external binary) genuinely cross. The report's
`inputs.honesty_notes` repeat this so a reader of the JSON alone cannot
miss it.

## Independent dynamics validation (E8-05, task #294)

The E5-07 physics-qualification reference is a Starship-class mission whose
ascent was computed by OpenTsiolkovsky, so — per the acceptance rule that an
OpenTsiolkovsky-versus-OpenTsiolkovsky comparison is not independent-model
evidence — E8-05 recomputes it with models that are independent *of that
program* and grades the discrepancies inside declared bands
(`python -m orbitopt.verify.independent_dynamics`, gated by
`tests/test_e805_independent_dynamics.py`, report kind `independent-dynamics`):

* **Ascent and stage-1 debris: a native point-mass twin.** The repo's own
  3-DOF plant (`propagate_point_mass`), driven *open-loop* by the authored
  attitude program and throttle schedules (no event-driven feedback), from
  pad to MECO, coast to separation, ship to SECO; the spent stage re-entrys
  through a separately written RK4 integrator with the power-off CA polar.
  MECO/separation/SECO event states, mass, max-Q, the SECO orbit and the
  debris impact point are compared with the sealed raw state history and the
  recorded impact event. Every discrepancy is split E8-02-style into a
  measured engine-Isp line (re-run at the recorded pad Isp 329.26 s), a
  measured aero/atmosphere line (density x1.03), a measured numerical line
  (dt/2), a declared-zero frame/time line (identical epoch and ECI
  convention, pad match < 1 m), a declared-zero controller line (identical
  authored program), and a *reported, gated residual* that absorbs the
  interaction and tracking-implementation terms — never zeroed.
* **Powered landing: two independent re-flights of one plan.** No landing is
  flown in E5-07 (`RECOVERY_GAP.md`), so the burn is validated on a declared
  design corridor at the pad (insertion state and the 3-engine landing
  configuration are `design_input`; the vehicle file itself records that the
  13-engine set cannot throttle below weight): the gfold SOCP arc is re-flown
  by the FL-301 propagator and by a newly written independent integrator
  (own geodesy, IGF-80 gravity, US76 tables, physical-thrust command replay).
  Touchdown position, vertical/lateral speed, end-of-burn state and remaining
  propellant are compared between the re-flights; the FL-301 design gates are
  graded separately as `design_input`.

The bands and their derivations are in the
[reference tolerance audit](reference-tolerance-audit.md#independent-dynamics-bands-e8-05-task-294).
No check in this report uses `historical_telemetry` evidence: the reference
vehicle is a public estimate, not flown hardware.

The [E5-08 variants](e5-08-new-vehicle-report.md) apply these checks to new
designs. DC7 fails the 15 km SECO semi-major-axis band (27.7 km measured);
Starbase fails the 12% debris impact-speed band (12.80% measured). Their
nominal joint physical passes do not clear these independent-model failures.

## Benchmarks against public references

```bash
orbitopt benchmarks
orbitopt benchmarks --json
```

The command exits 1 if any benchmark fails. References and assumptions are
in the [engineering notes](engineering-notes.md#benchmark-scenarios-with-public-references--analytic-solutions-task-245).

## Cross-model comparison

Compare a common ascent/coast across the available 3-DOF, JSBSim, TudatPy
and analytic backends:

```bash
python -m orbitopt.verify.cross_model --backends three_dof,six_dof,tudatpy
python -m orbitopt.verify.cross_model --threshold ascent.position_m=1000 -o cross_model.json
```

The `three_dof_vs_two_body_analytic` pair is `analytical`; other backend pairs
are `independent_model`. These harness names are distinct from Flight Lab's
[canonical run backend identities](flightlab.md#backend-identities-and-fidelity-levels).

## Integrator convergence

LEO/GTO coast, re-entry and powered-ascent studies sweep tolerances and,
where supported, step sizes. Their checks are `numerical_consistency`:

```bash
python -m orbitopt.verify.integrator_convergence --plot
```

## Cross-backend acceptance matrix

`src/orbitopt/data/acceptance_matrix.yaml` declares limits per backend pair,
phase and metric, with their justification and failure cause classes.
Results use the standard report/gate machinery. An unavailable backend is
`not_checked` with a cause, not silently passed. Ill-conditioned or explicitly
ungated rows are recorded; `--threshold key=none` disables that row's gate.

```bash
python -m orbitopt.verify.acceptance_matrix
python -m orbitopt.verify.acceptance_matrix --matrix my.yaml \
  --threshold three_dof_vs_two_body_analytic.coast.position_m=0.005
```

A pass on one available pair does not cover the remaining matrix. The current
acceptance ran the real numerical/analytic pair, not every backend pair.

## Model discrepancy budget (E8-02, task #282)

A gated total says *whether*, not *what*. `orbitopt.verify.discrepancy_budget`
reuses the acceptance matrix (same harness run, same resolved tolerances) and
splits every measured pair difference into five source lines — frame/time,
numerical integration, engine, aero, controller — recording the residual
between the component sum and the total. The residual is the unattributed
plant-formulation remainder (rigid-body vs point-mass, FCS/actuator dynamics),
stated, not zero-padded. Lines carry how they were obtained:

- **frame/time** — one shared bridge at one epoch: 0 applied, with the
  IERS 0.9 s desync sensitivity recorded as the budget;
- **numerical integration** — a tolerance-relaxation probe (3-DOF reference
  at rtol/atol ×1000); the fixed-step side is the published E6-04 bound;
- **engine** — the live E6-01 parity residual applied as a thrust-scale
  probe (a failed alignment is refused, not budgeted); the mass channel
  carries the disclosed `mdot·dt` one-frame bookkeeping bound;
- **aero** — 0 by declaration (`parity-jsbsim-aero-zero`), with the
  production-airframe *potential* recorded, not summed;
- **controller** — a loop-stiffening probe share (2× PID gains).

Pairs that did not fly yield `not_measured` lines with the matrix's own
cause code — never a fake zero. `--no-probes` records the structure with
the full totals in the residual. Only matrix rows vote: the report status
mirrors the matrix verdict.

```bash
python -m orbitopt.verify.discrepancy_budget            # all installed backends
python -m orbitopt.verify.discrepancy_budget --no-probes --json
```

## Model coverage and declared conventions

Mission results record each phase's model and limitations.
`orbitopt.physics_io` checks frame, epoch, timescale and unit conventions at
physics boundaries. Boundary continuity establishes consistent handoffs; it
does not prove a sparse recorded ascent was dynamically propagated.

## Further reading

[Flight Lab](flightlab.md) · [Regression CI](flightlab-regression-ci.md) ·
[Examples and validation](examples.md) · [Acceptance findings](acceptance-e1-e4.md)
