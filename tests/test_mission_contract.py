"""Unit tests for the M-01 mission segment / state handoff contract
(orbitopt.mission.contract): schema round-trips, and continuity/conservation
gates on both the continuous/conserved and the discontinuous/non-conserved
side of every tolerance."""
import dataclasses
import math

import jsonschema
import pytest

from orbitopt.mission.contract import (
    EPOCH_TIMESCALES,
    MISSION_FRAMES,
    Epoch,
    HandoffState,
    HandoffTolerances,
    MissionSegment,
    SegmentResult,
    check_handoff,
    check_handoff_chain,
    check_mass_conservation,
    is_valid_state_document,
    run_checked,
    state_from_dict,
    state_to_dict,
    validate_state,
)
from orbitopt.timescales import SECONDS_PER_DAY, tt_minus_utc


def state(**overrides):
    """A nominal GCRS parking-orbit-ish handoff state; keyword overrides for
    the field under test."""
    base = dict(
        epoch=Epoch(0.0, "UTC"),
        frame="GCRS",
        r_m=(7_000_000.0, 0.0, 0.0),
        v_m_s=(0.0, 7_500.0, 0.0),
        mass_kg=12_000.0,
        stage_config={"vehicle": "ascent-2", "stage": "core"},
    )
    base.update(overrides)
    return HandoffState(**base)


def check(report, check_id):
    return next(c for c in report.checks if c.id == check_id)


class TestHandoffState:
    def test_construction_coerces_vectors_to_float_tuples(self):
        s = state(r_m=[7e6, 1, 2], v_m_s=(0, 7.5e3, 0))
        assert s.r_m == (7_000_000.0, 1.0, 2.0)
        assert all(isinstance(c, float) for c in s.r_m)

    def test_rejects_nonfinite_vectors_and_mass(self):
        with pytest.raises(ValueError, match="not finite"):
            state(r_m=(float("nan"), 0, 0))
        with pytest.raises(ValueError, match="not finite"):
            state(v_m_s=(0, float("inf"), 0))
        with pytest.raises(ValueError, match="not finite"):
            state(mass_kg=float("nan"))

    def test_rejects_nonpositive_mass(self):
        with pytest.raises(ValueError, match="> 0"):
            state(mass_kg=0.0)
        with pytest.raises(ValueError, match="> 0"):
            state(mass_kg=-5.0)

    def test_rejects_bad_frame_and_timescale(self):
        with pytest.raises(ValueError, match="GCRS"):
            state(frame="J2000")
        with pytest.raises(ValueError, match="timescale"):
            Epoch(0.0, "UTC+0")

    def test_rejects_wrong_vector_arity_and_non_mapping_config(self):
        with pytest.raises(ValueError, match="3 components"):
            state(r_m=(1.0, 2.0))
        with pytest.raises(ValueError, match="stage_config"):
            state(stage_config=["not", "a", "mapping"])
        with pytest.raises(ValueError, match="keys must be strings"):
            state(metadata={7: "seven"})

    def test_replace_still_gates_through_construction(self):
        # dataclasses.replace re-runs __post_init__, so an invalid mutation
        # cannot bypass the construction checks.
        with pytest.raises(ValueError, match="> 0"):
            dataclasses.replace(state(), mass_kg=0.0)
        validate_state(state())  # a valid state passes the explicit gate too

    def test_epoch_timescales_are_the_three_orbitopt_converts(self):
        assert EPOCH_TIMESCALES == ("UTC", "TT", "TDB")
        assert MISSION_FRAMES == ("GCRS", "ECEF", "LAUNCH_ECI")


class TestEpochConversions:
    def test_same_instant_different_timescales_compare_to_zero(self):
        utc = Epoch(0.0, "UTC")
        tt = Epoch(0.0 + tt_minus_utc(0.0) / SECONDS_PER_DAY, "TT")
        assert utc.delta_seconds(tt) == pytest.approx(0.0, abs=1e-9)
        assert tt.delta_seconds(utc) == pytest.approx(0.0, abs=1e-9)
        assert utc.to_utc_mjd2000() == pytest.approx(tt.to_utc_mjd2000(), abs=1e-9)

    def test_tdb_label_maps_directly_to_tdb_seconds(self):
        # J2000.0 epoch = MJD2000 label 0.5 on any label scale.
        assert Epoch(0.5, "TDB").to_tdb_seconds() == pytest.approx(0.0)
        # 2000-01-01 12:00 TDB is 11:58:55 UTC -> MJD2000 label just below 0.5
        assert Epoch(0.5, "TDB").to_utc_mjd2000() == pytest.approx(0.4993, abs=2e-3)

    def test_utc_before_1972_is_rejected_when_compared(self):
        with pytest.raises(ValueError, match="1972"):
            Epoch(-20_000.0, "UTC").to_tdb_seconds()


class TestStateDocument:
    def test_roundtrip_equals_original(self):
        s = state(metadata={"producer": "test", "run_id": "demo-1"})
        doc = state_to_dict(s)
        assert doc["schema_version"] == "1.0"
        assert is_valid_state_document(doc)
        assert state_from_dict(doc) == s

    @pytest.mark.parametrize(
        "field", ["schema_version", "epoch", "frame", "position_m", "velocity_m_s", "mass_kg", "stage_config"]
    )
    def test_schema_requires_every_required_field(self, field):
        doc = state_to_dict(state())
        del doc[field]
        assert not is_valid_state_document(doc)

    def test_schema_rejects_unknown_fields_and_bad_enums(self):
        doc = state_to_dict(state())
        doc["surprise"] = 1
        assert not is_valid_state_document(doc)
        doc = state_to_dict(state())
        doc["frame"] = "J2000"
        assert not is_valid_state_document(doc)
        doc = state_to_dict(state())
        doc["epoch"]["timescale"] = "TAI"
        assert not is_valid_state_document(doc)

    def test_missing_or_unknown_schema_version_raises_valueerror(self):
        doc = state_to_dict(state())
        del doc["schema_version"]
        with pytest.raises(ValueError, match="schema_version"):
            state_from_dict(doc)
        with pytest.raises(ValueError, match="2.0"):
            state_from_dict({**doc, "schema_version": "2.0"})

    def test_finite_number_cross_check(self):
        doc = state_to_dict(state())
        doc["position_m"][1] = float("nan")  # JSON Schema says "number"; code rejects NaN
        with pytest.raises(ValueError, match="not finite"):
            state_from_dict(doc)
        doc = state_to_dict(state())
        doc["mass_kg"] = 0.0  # the schema's exclusiveMinimum gates it before the code does
        with pytest.raises((ValueError, jsonschema.ValidationError)):
            state_from_dict(doc)
        doc = state_to_dict(state())
        doc["epoch"]["mjd2000"] = -20_000.0  # UTC label before 1972
        with pytest.raises(ValueError, match="1972"):
            state_from_dict(doc)

    def test_raising_and_non_raising_forms_agree(self):
        doc = state_to_dict(state())
        assert is_valid_state_document(doc)
        state_from_dict(doc)
        doc["mass_kg"] = -1.0
        assert not is_valid_state_document(doc)
        with pytest.raises((ValueError, jsonschema.ValidationError)):
            state_from_dict(doc)


class TestContinuityChecks:
    def test_identical_states_pass_every_gate(self):
        report = check_handoff(state(), state())
        assert report.passed and not report.failures
        assert {c.id for c in report.checks} == {
            "handoff.position_continuity",
            "handoff.velocity_continuity",
            "handoff.epoch_monotonic",
            "handoff.mass_conservation",
        }

    def test_continuous_within_default_tolerance_passes(self):
        after = state(r_m=(7_000_000.5, 0.0, 0.0), v_m_s=(0.0, 7_500.005, 0.0),
                      epoch=Epoch(0.0, "UTC"))
        assert check_handoff(state(), after).passed

    def test_position_jump_fails_and_velocity_still_passes(self):
        after = state(r_m=(7_010_000.0, 0.0, 0.0))
        report = check_handoff(state(), after)
        assert not report.passed
        pos = check(report, "handoff.position_continuity")
        assert not pos.passed and pos.reason_code == "position_discontinuity"
        assert pos.value == pytest.approx(10_000.0) and pos.limit == 1.0
        assert check(report, "handoff.velocity_continuity").passed

    def test_velocity_jump_fails(self):
        after = state(v_m_s=(0.0, 7_501.0, 0.0))
        report = check_handoff(state(), after)
        vel = check(report, "handoff.velocity_continuity")
        assert not vel.passed and vel.reason_code == "velocity_discontinuity"
        assert vel.value == pytest.approx(1.0)

    def test_tolerances_are_configurable(self):
        after = state(r_m=(7_010_000.0, 0.0, 0.0), v_m_s=(0.0, 7_501.0, 0.0))
        loose = HandoffTolerances(position_m=20_000.0, velocity_m_s=5.0)
        assert check_handoff(state(), after, loose).passed
        tight = HandoffTolerances(position_m=0.25)
        assert not check_handoff(state(), state(r_m=(7_000_000.5, 0.0, 0.0)), tight).passed

    def test_frame_mismatch_blocks_measurement_without_converting(self):
        after = state(frame="ECEF")
        report = check_handoff(state(), after)
        for cid in ("handoff.position_continuity", "handoff.velocity_continuity"):
            c = check(report, cid)
            assert not c.passed and c.reason_code.startswith("frame_mismatch") and c.value is None

    def test_epoch_cannot_go_backwards_beyond_skew_tolerance(self):
        after = state(epoch=Epoch(-10.0 / 86400.0, "UTC"))  # 10 s back
        report = check_handoff(state(), after)
        c = check(report, "handoff.epoch_monotonic")
        assert not c.passed and c.reason_code == "epoch_backwards"
        assert c.value == pytest.approx(-10.0, abs=1e-6)
        # within the configured skew it passes
        assert check_handoff(state(), state(epoch=Epoch(-0.5 / 86400.0, "UTC")),
                             HandoffTolerances(epoch_s=1.0)).passed

    def test_epoch_gap_limit_is_opt_in(self):
        one_day_later = state(epoch=Epoch(1.0, "UTC"))
        assert check_handoff(state(), one_day_later).passed  # default: unbounded gap
        capped = check_handoff(state(), one_day_later, HandoffTolerances(max_epoch_gap_s=3600.0))
        c = check(capped, "handoff.epoch_monotonic")
        assert not c.passed and c.reason_code == "epoch_gap" and c.limit == 3600.0

    def test_epoch_across_timescales_compared_on_tdb(self):
        utc = state(epoch=Epoch(0.0, "UTC"))
        tt = state(epoch=Epoch(0.0 + tt_minus_utc(0.0) / SECONDS_PER_DAY, "TT"))
        report = check_handoff(utc, tt)
        assert report.passed
        assert check(report, "handoff.epoch_monotonic").value == pytest.approx(0.0, abs=1e-9)


class TestMassConservation:
    def test_conserved_within_tolerance_passes(self):
        assert check_mass_conservation(state(), state(), HandoffTolerances()).passed

    def test_mass_gain_fails(self):
        after = state(mass_kg=12_100.0)  # mass appeared from nowhere
        c = check_mass_conservation(state(), after, HandoffTolerances())
        assert not c.passed and c.reason_code == "mass_gain"
        assert c.value == pytest.approx(100.0)

    def test_small_gain_within_tolerance_passes(self):
        after = state(mass_kg=12_000.0005)
        assert check_mass_conservation(state(), after, HandoffTolerances(mass_kg=1e-3)).passed

    def test_unexplained_mass_loss_fails(self):
        after = state(mass_kg=9_000.0)  # 3 t gone, stage_config unchanged
        c = check_mass_conservation(state(), after, HandoffTolerances())
        assert not c.passed and c.reason_code == "mass_loss_without_staging"

    def test_staging_mass_drop_is_legal(self):
        after = state(mass_kg=9_000.0, stage_config={"vehicle": "ascent-2", "stage": "upper"})
        c = check_mass_conservation(state(), after, HandoffTolerances())
        assert c.passed and c.reason_code == "staging_mass_drop"
        assert c.value == pytest.approx(-3_000.0)

    def test_tolerance_scales_the_verdict(self):
        after = state(mass_kg=11_990.0)  # 10 kg loss
        assert not check_mass_conservation(state(), after, HandoffTolerances(mass_kg=1e-3)).passed
        assert check_mass_conservation(state(), after, HandoffTolerances(mass_kg=15.0)).passed

    def test_negative_tolerance_rejected(self):
        with pytest.raises(ValueError, match=">= 0"):
            HandoffTolerances(position_m=-1.0)
        with pytest.raises(ValueError, match=">= 0"):
            HandoffTolerances(max_epoch_gap_s=-1.0)


class TestChain:
    def test_chain_reports_each_boundary_separately(self):
        pad = state(stage_config={"stage": "ground"}, mass_kg=12_000.0)
        seco = state(epoch=Epoch(0.001, "UTC"), stage_config={"stage": "upper"}, mass_kg=9_000.0)
        jump = state(epoch=Epoch(0.002, "UTC"), r_m=(8e6, 0.0, 0.0),
                     stage_config={"stage": "upper"}, mass_kg=9_000.0)
        reports = check_handoff_chain([pad, seco, jump])
        assert len(reports) == 2
        assert reports[0].passed  # staging drop is documented by the stage_config change
        assert not reports[1].passed
        assert check(reports[1], "handoff.position_continuity").reason_code == "position_discontinuity"

    def test_chain_input_is_validated(self):
        with pytest.raises(ValueError, match="at least 2"):
            check_handoff_chain([state()])
        with pytest.raises(ValueError, match="list/tuple"):
            check_handoff_chain("not a chain")


class FakeAscentSegment:
    """A conforming MissionSegment: burns to a parking orbit, sheds mass via
    a documented stage_config change, hands off in GCRS."""

    name = "fake-ascent"

    def __init__(self, *, jump_m=0.0, create_mass=False):
        self._jump_m = jump_m
        self._create_mass = create_mass

    def run(self, initial_state):
        after = HandoffState(
            epoch=Epoch(initial_state.epoch.to_utc_mjd2000() + 0.001, "UTC"),
            frame="GCRS",
            r_m=(7_000_000.0 + self._jump_m, 0.0, 0.0),
            v_m_s=(0.0, 7_500.0, 0.0),
            mass_kg=initial_state.mass_kg * (1.1 if self._create_mass else 0.75),
            stage_config={"vehicle": "ascent-2", "stage": "upper"},
        )
        return SegmentResult(
            segment=self.name,
            final_state=after,
            events=({"kind": "meco", "t_s": 160.0}, {"kind": "separation", "t_s": 165.0},
                    {"kind": "seco", "t_s": 540.0}),
            diagnostics={"solver": "fake", "steps": 1},
        )


class TestSegmentContract:
    def test_fake_segment_satisfies_the_runtime_protocol(self):
        assert isinstance(FakeAscentSegment(), MissionSegment)

    def test_non_conforming_object_is_not_a_segment(self):
        class NoRun:
            name = "nope"

        assert not isinstance(NoRun(), MissionSegment)

    def test_run_checked_gates_a_good_segment(self):
        result, report = run_checked(FakeAscentSegment(), state())
        assert result.segment == "fake-ascent"
        assert [e["kind"] for e in result.events] == ["meco", "separation", "seco"]
        assert result.diagnostics["solver"] == "fake"
        assert report.passed

    def test_run_checked_reports_a_discontinuous_segment(self):
        _, report = run_checked(FakeAscentSegment(jump_m=50_000.0), state())
        assert not report.passed
        assert check(report, "handoff.position_continuity").reason_code == "position_discontinuity"

    def test_run_checked_reports_unexplained_mass_gain(self):
        _, report = run_checked(FakeAscentSegment(create_mass=True), state())
        assert not report.passed
        assert check(report, "handoff.mass_conservation").reason_code == "mass_gain"

    def test_run_checked_rejects_result_naming_another_segment(self):
        class Liar(FakeAscentSegment):
            def run(self, initial_state):
                result = super().run(initial_state)
                return dataclasses.replace(result, segment="someone-else")

        with pytest.raises(ValueError, match="someone-else"):
            run_checked(Liar(), state())

    def test_segment_result_shape_is_validated(self):
        with pytest.raises(ValueError, match="non-empty"):
            SegmentResult(segment="", final_state=state())
        with pytest.raises(ValueError, match="HandoffState"):
            SegmentResult(segment="x", final_state="not a state")
        with pytest.raises(ValueError, match="kind"):
            SegmentResult(segment="x", final_state=state(), events=({"t_s": 1.0},))


def test_infinity_velocity_caught_by_code_not_schema():
    # handoff-state-1.0.json says "number" (JSON has no finite number), so
    # the finiteness cross-check in state_from_dict is the real gate.
    doc = state_to_dict(state())
    doc["velocity_m_s"][2] = math.inf
    assert not is_valid_state_document(doc)
    with pytest.raises(ValueError, match="not finite"):
        state_from_dict(doc)
