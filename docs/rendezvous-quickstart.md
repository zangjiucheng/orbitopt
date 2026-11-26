# Rendezvous quickstart

Open the precomputed LEO-RDV-001 run, then compute your own.

## Open the precomputed run — no compute stack

`examples/runs/leo-rdv-001` is a committed Mission Studio replay of
LEO-RDV-001, built from the canonical sealed plan
(`examples/runs/leo-rdv-001-sealed-plan.json`).
The viewer extra is all you need:

```bash
pip install -e '.[viewer]'
orbitopt view examples/runs/leo-rdv-001-scene.json
```

The scene carries the Earth, target and chaser tracks, the keep-out centre
and the hold point as markers, and the maneuver events on the timeline.

To load the run inside Mission Control (`orbitopt-app`), add the directory
`examples/runs/leo-rdv-001` under **Runs**. The Rendezvous panel appears
because the run directory holds a `rendezvous.json` sidecar; it reads the
same sealed plan the scene was exported from, and its `plan_digest` matches
`leo-rdv-001-sealed-plan.json`.

## Compute a plan and verify it

From Python, with the conda compute stack (see the
[quickstart](quickstart.md)):

```python
from orbitopt.rendezvous import load_reference_config, canonical_plan, verify_plan
cfg = load_reference_config()
plan = canonical_plan(cfg)          # sealed two-impulse plan, J2-targeted
result = verify_plan(cfg, plan)     # re-propagates; result.passed == all checks
print(plan.total_delta_v_m_s, result.position_error_m)
```

## Replay it in Flight Lab

```bash
orbitopt rendezvous studio --config src/orbitopt/data/leo_rdv_001.yaml \
    --sealed-plan examples/runs/leo-rdv-001-sealed-plan.json \
    --refly-report docs/benchmarks/e12-05-refly-run/rendezvous-refly-report.json \
    -o runs/leo-rdv-001-studio
```

This writes a new run without replacing the committed example. The replay
validates the sealed plan against the mission configuration and checks that
the independent report matches the plan before displaying its physics verdict.
Open the run in Mission Control or export it with
`orbitopt flight to-scene runs/leo-rdv-001-studio -o scene.json`.

For a new configuration, omit `--sealed-plan` and use
`--write-sealed-plan PATH` to save the plan that Studio uses. Supply a
matching `--refly-report` once independent evidence is available.

## Re-run the independent re-flight

The TudatPy re-flight needs the cross-environment stack, not this one:

```bash
orbitopt crossenv run --plan src/orbitopt/data/crossenv_e12_05_plan.yaml
```

See the [verification report](rendezvous-verification-report.md) for which
digests each line of evidence grades, and
[ADR rendezvous-0003](adr/rendezvous-0003-independent-refly-tudatpy.md) for
the re-flight design.
