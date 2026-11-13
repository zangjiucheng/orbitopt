"""FL-602: vehicle data catalog provenance tests (fast, no propagation).

Every packaged vehicle file under ``orbitopt/data/flightlab/vehicles/`` must
(a) validate against ``flight-vehicle-1.0.json``, (b) leave NO numeric field
without a resolved provenance tag (``uncovered_numeric_paths`` empty), (c)
carry source + as_of on every tagged number, (d) expose uncertainty bands on
the FL-304 Monte Carlo knobs that drive ``vehicle_dispersions`` by default,
and (e) never present public-estimate vehicles as official performance. Also
covers the schema's rejections (missing sidecar, untagged number, inline node
in the OT stages, dangling sidecar key) and the montecarlo default wiring.
"""
from __future__ import annotations

import copy

import pytest

from orbitopt.flightlab import vehicle_catalog as vc

F9 = "f9_class_public_estimate"
SYNTH = "synthetic_two_stage"


def _f9_doc() -> dict:
    return vc.load_validated(vc.catalog_paths()[F9])


# ---------------------------------------------------------------------------
# packaged catalog: schema + complete provenance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(vc.catalog_paths()))
def test_catalog_file_validates_against_flight_vehicle_schema(name):
    doc = vc.load_validated(vc.catalog_paths()[name])
    assert doc["schema_version"] == vc.SCHEMA_VERSION
    assert doc["provenance"] in ("official", "public_estimate", "illustrative")


@pytest.mark.parametrize("name", sorted(vc.catalog_paths()))
def test_every_numeric_field_has_a_provenance_tag(name):
    doc = vc.load_validated(vc.catalog_paths()[name])
    assert vc.uncovered_numeric_paths(doc) == []
    annotations = vc.provenance_annotations(doc)
    # every covered leaf carries source AND as_of (FL-602 acceptance):
    missing = [p for p, a in annotations.items() if not a.source or not a.as_of]
    assert missing == []
    # ... and the walk is exhaustive: annotated == numeric leaves on disk
    from orbitopt.flightlab.vehicle_catalog import _numeric_paths
    assert set(annotations) == set(_numeric_paths(doc))


@pytest.mark.parametrize("name", sorted(vc.catalog_paths()))
def test_no_dangling_sidecar_keys(name):
    doc = vc.load_vehicle_file(vc.catalog_paths()[name])
    assert vc.dangling_sidecar_keys(doc) == []


@pytest.mark.parametrize("name", sorted(vc.catalog_paths()))
def test_stages_stay_plain_ot_numbers(name):
    """The OT adapter passes stages through untouched: inline provenance
    nodes are only legal under ``booster:``. The flight article must stay a
    plain OT config, so the file remains flyable."""
    from orbitopt.flightlab.adapters.opentsiolkovsky import Vehicle

    doc = vc.load_vehicle_file(vc.catalog_paths()[name])
    vehicle = Vehicle.load(vc.catalog_paths()[name])
    assert vehicle.stages == tuple(doc["stages"])
    assert vehicle.provenance == doc["provenance"]
    # BoosterModel.load must keep reading the booster section, tags included.
    from orbitopt.flightlab.recovery.propagator import BoosterModel

    model = BoosterModel.load(vc.catalog_paths()[name])
    assert model.provenance in ("official", "public_estimate", "illustrative")


# ---------------------------------------------------------------------------
# the F9-class file specifically: public estimate, never official SpaceX
# ---------------------------------------------------------------------------

def test_f9_file_is_public_estimate_with_disclaimer_and_citations():
    doc = _f9_doc()
    assert doc["provenance"] == "public_estimate"
    disclaimer = doc["disclaimer"].lower()
    assert "not official spacex performance data" in disclaimer
    ann = vc.provenance_annotations(doc)
    tags = {a.provenance for a in ann.values()}
    assert {"public_estimate", "illustrative"} <= tags
    for path, a in ann.items():
        if a.provenance != "illustrative":
            # every non-illustrative citation names an external source
            assert len(a.source) > 15, path
    # the officially-published figures are individually cited (Merlin Isp),
    # while no number is claimed to be SpaceX *performance* data:
    assert vc.provenance_of(doc, "stages[0].engine.isp_vac_s").provenance == "official"
    assert "user guide" in vc.provenance_of(
        doc, "stages[0].engine.isp_vac_s").source.lower()
    assert "not official spacex performance" in doc["disclaimer"].lower()


def test_f9_file_built_from_the_opentsiolkovsky_example():
    """The flight article is the MIT example config -- PACE liftoff mass,
    Merlin-1D vacuum Isp, 3.66 m diameter -- not invented numbers."""
    doc = _f9_doc()
    assert doc["stages"][0]["mass_initial_kg"] == 433100.0
    assert doc["stages"][0]["engine"]["isp_vac_s"] == 311.0
    assert doc["stages"][1]["engine"]["thrust_vac_n"] == 981000.0
    prov = vc.provenance_of(doc, "stages[0].mass_initial_kg")
    assert "falcon9.json" in prov.source.lower() or "opentsiolkovsky" in prov.source.lower()


def test_nearest_sidecar_entry_wins_and_inheritance_works():
    doc = _f9_doc()
    # the per-field entry (uncertainty 3%) overrides the stages[0] cover (none)
    thrust = vc.provenance_of(doc, "stages[0].engine.thrust_vac_n")
    assert thrust.uncertainty_pct == 3.0 and thrust.provenance == "public_estimate"
    # jettison mass: per-field public_estimate entry under the stages[1] cover
    jm = vc.provenance_of(doc, "stages[1].jettisons[0].mass_kg")
    assert jm.provenance == "public_estimate" and jm.uncertainty_pct == 15.0
    # bare booster.engines numbers inherit the sidecar 'booster.engines' entry
    eng = vc.provenance_of(doc, "booster.engines.thrust_vacuum_n")
    assert eng.provenance == "official"
    # inline node tags are read too (dry_mass_kg is public_estimate inline)
    dry = vc.provenance_of(doc, "booster.dry_mass_kg")
    assert dry.provenance == "public_estimate" and dry.uncertainty_pct == 8.0


# ---------------------------------------------------------------------------
# uncertainties -> FL-304 dispersions (FL-602 default feed)
# ---------------------------------------------------------------------------

def test_vehicle_dispersions_take_the_files_bands():
    from orbitopt.flightlab.recovery.montecarlo import Dispersions

    f9, notes = vc.vehicle_dispersions(vc.catalog_paths()[F9])
    assert (f9.thrust_pct, f9.isp_pct, f9.cd_pct) == (3.0, 2.0, 15.0)
    # scenario-side knobs keep the class defaults:
    defaults = Dispersions()
    assert f9.density_pct == defaults.density_pct
    assert f9.sep_altitude_m == defaults.sep_altitude_m
    assert any("FL-602" in n for n in notes)

    # the packaged synthetic file states bands equal to the class defaults,
    # so the packaged scenario's default Monte Carlo is unchanged (E3
    # benchmark continuity) -- while still being vehicle-file driven.
    syn, syn_notes = vc.vehicle_dispersions(vc.catalog_paths()[SYNTH])
    assert (syn.thrust_pct, syn.isp_pct, syn.cd_pct) == (5.0, 3.0, 8.0)
    assert syn == defaults
    assert any("FL-602" in n for n in syn_notes)


def test_run_monte_carlo_default_dispersions_follow_the_vehicle_file(monkeypatch,
                                                                      tmp_path):
    """Wiring test (no physics): with ``dispersions=None`` the result must
    carry the vehicle file's bands and an assumption line saying so."""
    from orbitopt.flightlab.recovery import montecarlo as mc

    def fake_sample(job):  # echo the uncertainty declaration the dispatcher computed
        unc = job["uncertainty"]
        return {"reason": "stub", "outcome": "stub",
                "dispersions": {k: v.get("half_width_pct", 0.0) or 0.0
                                for k, v in unc.items()
                                if k in ("thrust", "isp", "cd")}}

    monkeypatch.setattr(mc, "_run_sample", fake_sample)
    monkeypatch.setattr(mc, "write_run", lambda *a, **k: None)
    result = mc.run_monte_carlo(None, str(vc.catalog_paths()[F9]), None,
                                n=1, seed=3, workers=1)
    assert result.dispersions["thrust_pct"] == 3.0
    assert result.dispersions["cd_pct"] == 15.0
    assert any("FL-602" in a for a in result.assumptions)
    # explicit dispersions still win, unchanged, no FL-602 note:
    explicit = mc.Dispersions(thrust_pct=9.0, isp_pct=1.0)
    out = mc.run_monte_carlo(None, str(vc.catalog_paths()[F9]), None,
                             dispersions=explicit, n=1, seed=3, workers=1)
    assert out.dispersions["thrust_pct"] == 9.0
    assert not any("FL-602" in a for a in out.assumptions)


# ---------------------------------------------------------------------------
# schema rejections
# ---------------------------------------------------------------------------

def _strip(doc, key):
    out = copy.deepcopy(doc)
    del out[key]
    return out


def test_missing_provenance_section_is_rejected():
    with pytest.raises(vc.VehicleCatalogError, match="data_provenance"):
        vc.validate_vehicle_document(_strip(_f9_doc(), "data_provenance"))


def test_provenance_entry_without_source_or_as_of_is_rejected():
    doc = _f9_doc()
    doc["data_provenance"]["stages[1]"] = {"provenance": "public_estimate"}
    with pytest.raises(vc.VehicleCatalogError, match="as_of"):
        vc.validate_vehicle_document(doc)


def test_inline_provenance_node_in_the_ot_stages_is_rejected():
    """Provenance tags annotate the document, never the OT flight article:
    an inline node inside stages would break the adapter passthrough."""
    doc = _f9_doc()
    doc["stages"][0]["mass_dry_kg"] = {"value": 22200.0,
                                       "provenance": "public_estimate",
                                       "source": "x", "as_of": "2026-11"}
    with pytest.raises(vc.VehicleCatalogError):
        vc.validate_vehicle_document(doc)


def test_public_estimate_document_without_disclaimer_is_rejected():
    with pytest.raises(vc.VehicleCatalogError, match="disclaimer"):
        vc.validate_vehicle_document(_strip(_f9_doc(), "disclaimer"))


def test_dangling_sidecar_key_is_rejected():
    doc = _f9_doc()
    doc["data_provenance"]["stages[7].mass_dry_kg"] = {
        "provenance": "illustrative", "source": "typo", "as_of": "2026-11"}
    with pytest.raises(vc.VehicleCatalogError, match="no document node"):
        vc.load_validated(doc)


def test_unknown_top_level_key_is_rejected():
    doc = _f9_doc()
    doc["thrust_to_weight_42"] = 3.5
    with pytest.raises(vc.VehicleCatalogError):
        vc.validate_vehicle_document(doc)


def test_schema_itself_is_valid_draft2020_12():
    import jsonschema

    jsonschema.Draft202012Validator.check_schema(vc.vehicle_schema())
