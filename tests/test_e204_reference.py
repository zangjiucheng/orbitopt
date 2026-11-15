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
"""
from __future__ import annotations

import importlib.util
import json
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
    from orbitopt.flightlab.contract import write_run

    out = tmp_path / "ascent"
    write_run(out, script._build_ascent_run())
    return out


def test_committed_ascent_tracks_match_the_builder(built_ascent):
    pytest.importorskip("jsonschema")
    from orbitopt.flightlab.contract import read_run

    fresh, committed = read_run(built_ascent), read_run(REF_ROOT / "ascent")
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
            assert all(abs(x - y) <= 1e-9 for x, y in zip(fp, cp)), body
            assert all(abs(x - y) <= 1e-9 for x, y in zip(fv, cv)), body
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
