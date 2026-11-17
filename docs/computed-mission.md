---
title: Computed mission workflow
parent: Flight Lab
nav_order: 0
description: Replay the recorded LEO and RTLS mission, add propagation across environments, and inspect evidence.
---

# Computed mission workflow

E5–E8 add a computed-ascent RTLS reference, coupled separation handoffs,
cross-environment execution, guided JSBSim studies and stricter evidence
gates. Choose the run that answers your question:

| Workflow | What runs | Evidence boundary |
|---|---|---|
| E2-04 | Synthetic ascent endpoints + simulated recovery. | Fast integration fixture; no computed ascent or parking propagation. |
| E5-05 replay | Recorded computed ascent + freshly simulated recovery. | One canonical separation state; parking propagation remains `not_checked`. |
| E5-05 cross-environment | Recorded ascent + TudatPy parking propagation + gfold recovery. | The joined run must evidence all five physical checks. |
| E8-01 reference check | Compares sealed records with raw backend output and closed forms. | No new flight; validates extraction, continuity and stated reference comparisons. |

## Replay and view the RTLS mission

Use the [landing environment](../envs/README.md), with gfold **0.3.2**, from
the repository root:

```bash
python scripts/e505_reference.py verify
python scripts/e505_reference.py run --out runs/e505
python scripts/e505_reference.py compare runs/e505
orbitopt flight to-scene runs/e505/recovery/run -o runs/e505/scene.json
orbitopt view runs/e505/scene.json
```

The last command needs the local viewer. Computation may run on another
machine: copy the output directory back, then open its scene locally. For
telemetry, add `runs/e505/recovery/run` in Mission Control's **Flight Lab**.
The scene is the combined flight run; the unified mission report is
`runs/e505/mission.verification.json`.

The reference was recorded in `64f64a6`, using code `74f30c8`. Its
`synthetic_two_stage_rtls` vehicle flew a fixed-program OpenTsiolkovsky
ascent from Kourou and reached **265.5 × 278.1 km at 51.0°**. This replaces
the old blocked 400 km/CMA-ES recording proposal. `update` re-flies that
computed ascent and the fixed recovery recipe; routine replay does not
re-optimize or re-fly ascent. See the
[vehicle decision](adr/flightlab-0008-rtls-capable-vehicle-redesign.md) and
[recording procedure](flightlab-regression-ci.md#e5-05-the-reproducible-full-physical-reference-mission-task-272).

The committed single-environment report passes its applicable phase checks,
but its physical gate is **`not_checked`** because parking propagation was
not run. Read that gate rather than treating the overall green report as
proof of a complete physical mission.

## Add parking propagation across environments

Install the checkout in the locked TudatPy Python 3.10 environment and a
separate Python 3.12 landing environment. Supply their interpreter paths:

```bash
orbitopt crossenv run --plan src/orbitopt/data/crossenv_e505_plan.yaml \
  --env tudat=/path/to/orbitopt/bin/python \
  --env gfold=/path/to/landing/bin/python -o runs/crossenv-e505
python scripts/e505_reference.py crossenv-check runs/crossenv-e505
```

Each stage seals its artifacts in a versioned exchange with SHA-256 digests
and interpreter/toolchain provenance. The consumer verifies it before use;
the recovery stage must report the same separation handoff identity. The
reference plan also selects the recorded Earth-orientation snapshot in both
interpreters; stage reports record its digest, which `crossenv-check` verifies. The
joined report must actually pass propagation, landing, insertion, mass and
continuity checks; configuring this plan alone is not evidence of a pass.

## Check the recorded chain without re-flying

```bash
python -m orbitopt.verify.full_chain --out runs/e505/full-chain.verification.json
```

This checks 22 reference comparisons. Its independent-model evidence covers
the chain's extraction of raw OT ascent data and the booster's shared
unpowered window. Touchdown limits are **design input**; no historical
telemetry or independently validated landing is claimed. See
[verification](verification.md#full-chain-reference-validation-e8-01-task-281).

## Other merged studies

| Study | Read or reproduce |
|---|---|
| E6: engine parity, guided JSBSim staging, guidance envelope and control-step convergence | [6-DOF studies](flightlab.md#6-dof-studies) and ADRs 0008–0011. |
| E7: enforced joint constraints, probability calibration, coupled risk and paired failure analysis | [Recovery studies](flightlab.md#joint-chance-constrained-planning-e3-02). |
| E8: required-phase qualification, discrepancy attribution and reference portability | [Verification](verification.md) and [tolerance audit](reference-tolerance-audit.md). |

Study scripts export measurement JSON; they do not necessarily produce a
Flight Lab run directory. Recorded results describe their declared vehicles,
controllers and dispersions, not a general flight qualification.
