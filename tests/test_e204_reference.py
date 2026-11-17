"""E2-04: the committed frozen dual-branch reference mission (task #257).

Two guards, in the FL-601 pattern:

* FAST (every CI job): the files under
  ``tests/data/mission/reference/e2-04-frozen-mission/`` must match their
  recorded digests, the recorded toolchain must still name the pinned
  backend stack, and the committed frozen ascent must equal what the pinned
  constants in ``scripts/e204_reference.py`` build -- hand-edited reference
  numbers, a stale target pin or a drifted builder all fail here.
* SLOW (the ``flightlab-landing`` job, gfold 0.3.2): ``run`` the complete
  dual-branch mission through the real chain and ``compare`` every recovery
  number against the reference within the declared tolerances. Out of band =
  red, with FL-601's advice: an INTENDED change goes in via
  ``scripts/e204_reference.py update`` + an ``[e204-reference-update]``
  commit tag (G5), which CI's verify + tag-check pair enforces.

E5-01 (task #268) exception: the ascent this mission replays is a SYNTHETIC
two-sample handoff built from the constants in ``scripts/e204_reference.py``
(preset separation/SECO states). It is kept as the annotated fast
integration-test exception; ``tests/test_mission_e5_real_ascent.py`` guards
that no other full-mission path consumes a preset SECO state.
"""
from __future__ import annotations

import importlib.util
import dataclasses
import json
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REF_ROOT = Path(__file__).parent / "data" / "mission" / "reference" / "e2-04-frozen-mission"


def _script():
    spec = importlib.util.spec_from_file_location(
        "e204_reference", REPO / "scripts" / "e204_reference.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_committed_reference_matches_its_provenance():
    assert _script().cmd_verify() == 0, (
        "reference files changed without scripts/e204_reference.py update "
        "-- see tests/test_e204_reference.py docstring")


def test_provenance_records_the_backend_versions():
    prov = json.loads((REF_ROOT / "PROVENANCE.json").read_text(encoding="utf-8"))
    script = _script()
    assert prov["requires_gfold"] == script.GOLDEN_GFOLD
    assert prov["requires_commit_tag"] == script.COMMIT_TAG
    tool = prov["toolchain"]
    for pkg in ("python", "gfold", "numpy", "scipy", "astropy", "astropy_iers_data"):
        assert tool.get(pkg), f"toolchain record missing {pkg!r}"
    assert prov["files"], "no file digests recorded"
    assert prov["earth_orientation"].startswith("bundled IERS only")
    recipe = prov["frozen_inputs"]["recovery_plan_options"]
    assert recipe == {"initial_guess": [18.4, 17.75, 1.0, 4.36, 0.0, 17456.0, 23.0],
                      "max_fevals": 18}


def test_frozen_inputs_match_the_packed_target_and_scenario():
    pytest.importorskip("jsonschema")
    from orbitopt.flightlab.contract import load_target

    script = _script()
    target = load_target(script.TARGET)
    a = json.loads((REF_ROOT / "PROVENANCE.json").read_text(encoding="utf-8"))["frozen_inputs"]["ascent_constants"]

    scenario = script.load_scenario()
    doc = scenario.to_dict()
    assert doc["ascent"]["mode"] == "run_dir"
    assert doc["expected"]["propagation_result"] == "not_checked"
    assert doc["recovery"]["plan_options"] == json.loads(
        (REF_ROOT / "PROVENANCE.json").read_text(encoding="utf-8")
    )["frozen_inputs"]["recovery_plan_options"]
    assert Path(doc["recovery"]["target_file"]).name == script.TARGET.name

    site = target.launch_site
    assert (site.latitude_deg, site.longitude_deg, site.altitude_wgs84_m) == tuple(a["pad"])
    manifest = json.loads((REF_ROOT / "ascent" / "manifest.json").read_text(encoding="utf-8"))
    assert {b["epoch_utc"] for b in manifest["bodies"]} == {a["epoch_utc"]}


def test_declared_tolerances_are_complete():
    script = _script()
    declared = json.loads((REF_ROOT / "tolerances.json").read_text(encoding="utf-8"))
    assert set(declared) == set(script.DEFAULT_TOLERANCES)
    assert all(isinstance(v, (int, float)) and v >= 0 for v in declared.values()
               if not isinstance(v, dict))


@pytest.fixture()
def built_ascent(tmp_path):
    script = _script()
    script._pin_earth_orientation()
    from orbitopt.flightlab.contract import write_run

    out = tmp_path / "ascent"
    write_run(out, script._build_ascent_run())
    return out


def test_committed_ascent_tracks_match_the_builder(built_ascent):
    pytest.importorskip("jsonschema")
    from orbitopt.flightlab.contract import read_run

    fresh, committed = read_run(built_ascent), read_run(REF_ROOT / "ascent")
    tol = json.loads((REF_ROOT / "tolerances.json").read_text())
    assert fresh.run_id == committed.run_id
    assert fresh.target_sha256 == committed.target_sha256
    assert fresh.vehicle_sha256 == committed.vehicle_sha256
    ft = {t.body_id: t for t in fresh.tracks}
    ct = {t.body_id: t for t in committed.tracks}
    assert set(ft) == set(ct)
    for body in ct:
        f, c = ft[body], ct[body]
        assert f.frame == c.frame and f.provenance == c.provenance
        assert f.t_s == c.t_s and f.mass_kg == c.mass_kg
        for (fp, cp), (fv, cv) in zip(zip(f.position_m, c.position_m),
                                      zip(f.velocity_m_s, c.velocity_m_s)):
            assert all(abs(x - y) <= tol["ascent_build_position_m_abs"]
                       for x, y in zip(fp, cp)), body
            assert all(abs(x - y) <= tol["ascent_build_velocity_m_s_abs"]
                       for x, y in zip(fv, cv)), body
    assert [(e.kind, e.body_id) for e in fresh.events] == \
           [(e.kind, e.body_id) for e in committed.events]
    for fe, ce in zip(fresh.events, committed.events):
        assert abs(fe.t_s - ce.t_s) <= 1e-9


def test_reference_mission_is_green_with_a_landed_booster():
    rep = json.loads((REF_ROOT / "mission" / "mission.verification.json")
                     .read_text(encoding="utf-8"))
    assert rep["status"]["result"] == "passed"
    assert rep["status"]["checks"]["recovery.landed"] is True
    assert rep["status"]["checks"]["mass.separation_fork"] is True
    ver = json.loads((REF_ROOT / "mission" / "recovery" / "run" / "verification.json")
                     .read_text(encoding="utf-8"))
    landing = next(c for c in ver["checks"] if c["id"] == "landing-position")
    assert landing["passed"] and landing["value"] < landing["limit"]


@pytest.mark.slow
def test_frozen_mission_rerun_matches_the_reference(tmp_path):
    """The regression gate: fly the dual-branch mission again on the pinned
    landing stack and compare every recovery number + verdict within the
    declared tolerances."""
    script = _script()
    pytest.importorskip("gfold")
    from importlib.metadata import version

    if version("gfold") != script.GOLDEN_GFOLD:
        pytest.skip(f"reference recorded under gfold {script.GOLDEN_GFOLD}, "
                    f"env has {version('gfold')}")
    out = tmp_path / "mission"
    assert script.cmd_run(out) == 0, "frozen mission went red on rerun"
    assert script.cmd_compare(out) == 0, (
        "recovery numbers drifted outside the declared tolerances; if this "
        f"is intended: scripts/e204_reference.py update + {script.COMMIT_TAG}")


def test_event_arrays_use_numeric_tolerances_recursively():
    script = _script()
    fails = []
    script._compare_dicts('event', {'v': [1.0, {'r': [2.0, 3.0]}]},
                          {'v': [1.0 + 1e-12, {'r': [2.0 + 1e-12, 3.0]}]},
                          1e-6, 0.0, fails)
    assert fails == []
    script._compare_dicts('event', {'v': [1.01, 2.0]}, {'v': [1.0, 2.0]},
                          1e-6, 0.0, fails)
    assert any('event[v][0]' in fail for fail in fails)


@pytest.mark.parametrize('got,ref', [
    ([1.0], [1.0, 2.0]), ([float('nan')], [1.0]),
    ([float('inf')], [float('inf')]), (True, 1),
    ({'extra': 1}, {}), ('Solved', 'Failed'),
])
def test_comparison_rejects_nonfinite_shape_and_verdict_changes(got, ref):
    fails = []
    _script()._compare_values('probe', got, ref, 1e-6, 1e-9, fails)
    assert fails


@pytest.mark.parametrize('field,delta,passes', [
    ('fuel_planned_kg', 5e-4, True), ('fuel_planned_kg', 2e-3, False),
    ('tof_flying_s', 5e-3, True), ('tof_flying_s', 2e-2, False),
    ('gravity_m_s2', 5e-4, False),
])
def test_only_declared_landing_plan_diagnostics_get_their_own_budget(tmp_path, field, delta, passes):
    script = _script()
    candidate = tmp_path / 'candidate'
    shutil.copytree(REF_ROOT / 'mission/recovery/run', candidate)
    events_file = candidate / 'events.json'
    events = json.loads(events_file.read_text())
    event = next(e for e in events if e['kind'] == 'landing_start')
    event['data'][field] += delta
    events_file.write_text(json.dumps(events))
    tol = json.loads((REF_ROOT / 'tolerances.json').read_text())
    fails = script.compare_runs(candidate, tol)
    assert (not fails) is passes, fails


@pytest.mark.parametrize('channel,delta', [('position_m', .02), ('velocity_m_s', .0002)])
def test_flown_states_keep_the_original_physical_budgets(tmp_path, channel, delta):
    from orbitopt.flightlab.contract import read_run, write_run

    run = read_run(REF_ROOT / 'mission/recovery/run')
    booster = next(t for t in run.tracks if t.body_id == 'booster')
    values = list(getattr(booster, channel))
    values[0] = (values[0][0] + delta, *values[0][1:])
    changed = dataclasses.replace(booster, **{channel: tuple(values)})
    candidate = tmp_path / 'candidate'
    write_run(candidate, dataclasses.replace(run, tracks=tuple(
        changed if t.body_id == 'booster' else t for t in run.tracks)))
    tol = json.loads((REF_ROOT / 'tolerances.json').read_text())
    fails = _script().compare_runs(candidate, tol)
    assert any('booster[0]' in fail for fail in fails), fails


def test_tolerances_only_update_preserves_flight_outputs_and_recording_toolchain(tmp_path, monkeypatch):
    script = _script()
    root = tmp_path / 'reference'
    shutil.copytree(REF_ROOT, root)
    for name, value in [('REF_ROOT', root), ('PROVENANCE', root / 'PROVENANCE.json'),
                        ('TOLERANCES', root / 'tolerances.json')]:
        monkeypatch.setattr(script, name, value)
    monkeypatch.setattr(script, '_toolchain', lambda: {'python': 'test'})
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert script.cmd_update(tolerances_only=True, reason="test: audited policy round-trip") == 0
    assert script.cmd_verify() == 0
    for relative, content in before.items():
        if str(relative) not in ('PROVENANCE.json', 'tolerances.json'):
            assert (root / relative).read_bytes() == content
    recorded = json.loads(before[Path('PROVENANCE.json')])
    updated = json.loads((root / 'PROVENANCE.json').read_text())
    assert updated['toolchain'] == recorded['toolchain']
    assert updated['frozen_inputs'] == recorded['frozen_inputs']


def test_tolerances_only_update_refuses_an_already_tampered_reference(tmp_path, monkeypatch):
    script = _script()
    root = tmp_path / 'reference'
    shutil.copytree(REF_ROOT, root)
    monkeypatch.setattr(script, 'REF_ROOT', root)
    monkeypatch.setattr(script, 'PROVENANCE', root / 'PROVENANCE.json')
    monkeypatch.setattr(script, 'TOLERANCES', root / 'tolerances.json')
    provenance = script.PROVENANCE.read_bytes()
    (root / 'ascent/events.json').write_text('[]')
    assert script.cmd_update(tolerances_only=True, reason="test: tampered reference") == 1
    assert script.PROVENANCE.read_bytes() == provenance


def test_reference_earth_orientation_is_scoped_and_covers_the_frozen_epoch():
    from astropy.utils import iers

    script = _script()
    original = iers.earth_orientation_table.get()
    with script.frozen_earth_orientation():
        frozen = iers.earth_orientation_table.get()
        assert frozen is not original
        assert len(frozen) == 4
        assert float(frozen['MJD'][0].value) <= 61354.5 <= float(frozen['MJD'][-1].value)
    assert iers.earth_orientation_table.get() is original
