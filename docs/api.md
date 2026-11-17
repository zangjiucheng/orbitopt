---
title: API reference
nav_order: 6
permalink: /api/
description: Python modules, command-line options and data schemas.
---

# API reference

| Interface | Reference |
|---|---|
| Python | [Modules, classes, functions and methods]({{ site.baseurl }}/api/python/) |
| Command line | [Commands and options]({{ site.baseurl }}/api/cli/) |
| Data | [Mission, scene, report and Flight Lab schemas]({{ site.baseurl }}/api/schemas/) |
| Cross-environment data | [Plan]({{ site.baseurl }}/api/schemas/crossenv-plan-1.0/) and [sealed exchange]({{ site.baseurl }}/api/schemas/crossenv-exchange-1.0/) |

The reference is generated from the source and packaged schemas at site-build
time. Public names are included by naming convention; inclusion does not imply
that an internal implementation is a stable compatibility promise. Private
names and inherited third-party methods are omitted. Where source docstrings
are absent, the reference says so and links to the implementation.

The [quickstart](quickstart.md) introduces common workflows. The
[engineering notes](engineering-notes.md) explain numerical assumptions.

## Mission and qualification interfaces

| Task | Python reference |
|---|---|
| Load a plan, run stages and verify exchanges | [Cross-environment API]({{ site.baseurl }}/api/python/orbitopt.crossenv/) |
| Construct and grade the shared separation handoff | [Coupled branches]({{ site.baseurl }}/api/python/orbitopt.mission.coupled_branches/) |
| Run guided JSBSim/native comparison studies | [Guided ascent]({{ site.baseurl }}/api/python/orbitopt.flightlab.sixdof.guided_ascent/) |
| Evaluate calibration, joint risk and paired failure counts | [Calibration]({{ site.baseurl }}/api/python/orbitopt.flightlab.recovery.calibration/), [joint risk]({{ site.baseurl }}/api/python/orbitopt.flightlab.recovery.joint_risk/), [failure analysis]({{ site.baseurl }}/api/python/orbitopt.flightlab.recovery.failure_driven/) |
| Build a recorded-chain report or attribute model discrepancies | [Full-chain reference]({{ site.baseurl }}/api/python/orbitopt.verify.full_chain/), [discrepancy budget]({{ site.baseurl }}/api/python/orbitopt.verify.discrepancy_budget/) |

For executable examples and the distinction between replay and fresh
propagation, start with the [computed mission workflow](computed-mission.md).
