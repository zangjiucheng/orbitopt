"""Unit tests for the E12-01 multi-spacecraft mission contract
(orbitopt.mission.multispacecraft): document round-trips and the version
gate, RelativeState formation rules, the multi-body event vocabulary, the
validation gates on both sides of every tolerance (negative tests for the
acceptance criteria: epoch inconsistency, frame mixing, unit errors,
missing roles), and the unified-timeline replay of two trajectories."""
import dataclasses
import json
import math
from datetime import datetime, timezone

import jsonschema
import pytest

from orbitopt.flightlab.contract import BodyTrack
from orbitopt.mission.contract import Epoch, HandoffState
from orbitopt.mission.multispacecraft import (
    MISSION_EVENT_KINDS,
    ROLES,
    MissionEvent,
    MissionTolerances,
    MultiSpacecraftMission,
    RelativeState,
    SpacecraftSpec,
    Zone,
    build_unified_replay,
    is_valid_mission_document,
    is_valid_relative_state_document,
    is_valid_replay_document,
    mission_from_dict,
    mission_to_dict,
    relative_state_from_dict,
    relative_state_from_tracks,
    relative_state_to_dict,
    replay_from_dict,
    validate_mission,
)
from orbitopt.timescales import (
    SECONDS_PER_DAY,
    mjd2000_to_utc_datetime,
    utc_datetime_to_mjd2000,
)

MISSION_MJD2000 = 9300.0  # 2025-06-01T00:00:00Z-ish label, fixed in tests
MISSION_TDB = Epoch(MISSION_MJD2000, "UTC").to_tdb_seconds()
EPOCH_ISO = mjd2000_to_utc_datetime(MISSION_MJD2000).strftime("%Y-%m-%dT%H:%M:%SZ")


def spec(spacecraft_id, role="CHASER"):
    return SpacecraftSpec(spacecraft_id=spacecraft_id, role=role)


def mission(**overrides):
    base = dict(
        mission_id="e12-01-rdv",
        title="two-craft rendezvous contract",
        frame="GCRS",
        epoch=Epoch(MISSION_MJD2000, "UTC"),
        spacecraft=(spec("tgt", "TARGET"), spec("chaser", "CHASER")),
    )
    base.update(overrides)
    return MultiSpacecraftMission(**base)


def track(body_id="tgt", frame="GCRS", epoch_iso=EPOCH_ISO, r0=(7_000_000.0, 0.0, 0.0),
          v=(0.0, 7_500.0, 0.0), t0=0.0, t1=300.0, dt=60.0):
    """Straight-line constant-velocity BodyTrack: coherence error exactly 0,
    so every check that fails here fails for the reason under test."""
    n = int(round((t1 - t0) / dt)) + 1
    t_s = tuple(t0 + i * dt for i in range(n))
    position_m = tuple((r0[0] + v[0] * t, r0[1] + v[1] * t, r0[2] + v[2] * t) for t in t_s)
    velocity_m_s = tuple(v for _ in t_s)
    return BodyTrack(
        body_id=body_id, parent_body_id=None, frame=frame, epoch_utc=epoch_iso,
        provenance="illustrative", t_s=t_s, mass_kg=tuple(500.0 for _ in t_s),
        position_m=position_m, velocity_m_s=velocity_m_s,
    )


def nominal_tracks():
    return (track("tgt"), track("chaser", r0=(7_010_000.0, 0.0, 0.0), v=(0.0, 7_489.0, 0.0)))


def check(report, check_id):
    return next(c for c in report.checks if c.id == check_id)


def iso_at(mission_tdb_offset_s):
    """epoch_utc string of the mission epoch shifted by dt (keeps every
    track on one shared instant except for the shift under test)."""
    mjd = MISSION_MJD2000 + mission_tdb_offset_s / SECONDS_PER_DAY
    return mjd2000_to_utc_datetime(mjd).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class TestRelativeState:
    def test_from_states_subtracts_and_reports_range(self):
        tgt = HandoffState(epoch=Epoch(MISSION_MJD2000, "UTC"), frame="GCRS",
                           r_m=(7_000_000.0, 0.0, 0.0), v_m_s=(0.0, 7_500.0, 0.0),
                           mass_kg=5_000.0, metadata={"spacecraft_id": "tgt"})
        chs = HandoffState(epoch=Epoch(MISSION_MJD2000, "UTC"), frame="GCRS",
                           r_m=(7_001_000.0, 0.0, 0.0), v_m_s=(0.0, 7_495.0, 0.0),
                           mass_kg=2_000.0, metadata={"spacecraft_id": "chaser"})
        rel = RelativeState.from_states(tgt, chs)
        assert (rel.target_id, rel.chaser_id) == ("tgt", "chaser")
        assert rel.r_rel_m == pytest.approx((1_000.0, 0.0, 0.0))
        assert rel.range_m == pytest.approx(1_000.0)
        assert rel.range_rate_m_s == pytest.approx(0.0)

    def test_frame_mixing_is_a_construction_failure(self):
        tgt = HandoffState(epoch=Epoch(MISSION_MJD2000, "UTC"), frame="GCRS",
                           r_m=(7_000_000.0, 0.0, 0.0), v_m_s=(0.0, 7_500.0, 0.0),
                           mass_kg=5_000.0, metadata={"spacecraft_id": "tgt"})
        chs = dataclasses.replace(tgt, frame="ECEF", metadata={"spacecraft_id": "chaser"})
        with pytest.raises(ValueError, match="frame_mismatch"):
            RelativeState.from_states(tgt, chs)

    def test_epoch_skew_beyond_tolerance_is_a_construction_failure(self):
        tgt = HandoffState(epoch=Epoch(MISSION_MJD2000, "UTC"), frame="GCRS",
                           r_m=(7_000_000.0, 0.0, 0.0), v_m_s=(0.0, 7_500.0, 0.0),
                           mass_kg=5_000.0, metadata={"spacecraft_id": "tgt"})
        chs = dataclasses.replace(tgt, epoch=Epoch(MISSION_MJD2000 + 2.0 / SECONDS_PER_DAY, "UTC"),
                                  metadata={"spacecraft_id": "chaser"})
        with pytest.raises(ValueError, match="epoch_mismatch"):
            RelativeState.from_states(tgt, chs, epoch_alignment_s=1.0)
        rel = RelativeState.from_states(tgt, chs, epoch_alignment_s=3.0)  # inside tolerance: passes
        assert rel.r_rel_m == (0.0, 0.0, 0.0)  # same state, just 2 s apart in the report

    def test_the_same_instant_on_different_timescales_still_agrees(self):
        """A TT-labelled epoch and its UTC equivalent are the same instant
        (leap seconds included) -- the shared timeline compares TDB seconds,
        not day labels."""
        from orbitopt.timescales import tt_minus_utc
        tgt = HandoffState(epoch=Epoch(MISSION_MJD2000, "UTC"), frame="GCRS",
                           r_m=(7_000_000.0, 0.0, 0.0), v_m_s=(0.0, 7_500.0, 0.0),
                           mass_kg=5_000.0, metadata={"spacecraft_id": "tgt"})
        chs = dataclasses.replace(tgt, epoch=Epoch(MISSION_MJD2000 + tt_minus_utc(MISSION_MJD2000) / SECONDS_PER_DAY, "TT"),
                                  metadata={"spacecraft_id": "chaser"})
        rel = RelativeState.from_states(tgt, chs, epoch_alignment_s=1.0)
        assert abs(tgt.epoch.delta_seconds(chs.epoch)) < 1e-3

    def test_pair_must_be_named(self):
        tgt = HandoffState(epoch=Epoch(MISSION_MJD2000, "UTC"), frame="GCRS",
                           r_m=(7_000_000.0, 0.0, 0.0), v_m_s=(0.0, 7_500.0, 0.0), mass_kg=5_000.0)
        with pytest.raises(ValueError, match="spacecraft_id"):
            RelativeState.from_states(tgt, tgt)

    def test_relative_state_document_round_trip_and_negatives(self):
        rel = RelativeState(target_id="tgt", chaser_id="chaser", epoch=Epoch(MISSION_MJD2000, "UTC"),
                            frame="GCRS", r_rel_m=(1_000.0, 0.0, 0.0), v_rel_m_s=(0.0, -5.0, 0.0))
        doc = relative_state_to_dict(rel)
        assert relative_state_from_dict(doc) == rel
        assert is_valid_relative_state_document(doc)
        wrong_field = dict(doc, position_km=[1, 0, 0])
        wrong_field.pop("position_m")
        assert not is_valid_relative_state_document(wrong_field)  # unit lives in the field name
        no_version = {k: v for k, v in doc.items() if k != "schema_version"}
        with pytest.raises(ValueError, match="schema_version"):
            relative_state_from_dict(no_version)
        with pytest.raises(ValueError, match="2.0"):
            relative_state_from_dict(dict(doc, schema_version="2.0"))

    def test_a_spacecraft_is_never_relative_to_itself(self):
        with pytest.raises(ValueError, match="relative to itself"):
            RelativeState(target_id="tgt", chaser_id="tgt", epoch=Epoch(MISSION_MJD2000, "UTC"),
                          frame="GCRS", r_rel_m=(0.0, 0.0, 0.0), v_rel_m_s=(0.0, 0.0, 0.0))


class TestMissionDocument:
    def test_round_trip_through_the_versioned_schema(self):
        m = mission(zones=(Zone("keepout", (7_000_000.0, 0.0, 0.0), 100.0),),
                    events=(MissionEvent("chaser", "maneuver", Epoch(MISSION_MJD2000, "UTC"),
                                         data={"delta_v_m_s": 0.4}),))
        doc = mission_to_dict(m)
        assert mission_from_dict(json.loads(json.dumps(doc))) == m
        assert is_valid_mission_document(doc)

    def test_missing_role_is_a_schema_rejection(self):
        doc = mission_to_dict(mission())
        del doc["spacecraft"][1]["role"]
        with pytest.raises(jsonschema.ValidationError):
            mission_from_dict(doc)

    def test_unknown_role_is_a_schema_rejection(self):
        doc = mission_to_dict(mission())
        doc["spacecraft"][1]["role"] = "PROBE"
        with pytest.raises(jsonschema.ValidationError):
            mission_from_dict(doc)

    def test_role_is_a_mandatory_constructor_argument(self):
        with pytest.raises(TypeError):
            SpacecraftSpec(spacecraft_id="x")
        with pytest.raises(ValueError, match="not in"):
            SpacecraftSpec(spacecraft_id="x", role="OBSERVER")

    def test_shared_timescale_is_declared_and_bounded(self):
        with pytest.raises(ValueError, match="timescale"):
            mission(epoch=Epoch(MISSION_MJD2000, "TAI"))
        with pytest.raises(ValueError, match="frame"):
            mission(frame="J2000")
        with pytest.raises(ValueError, match="at least 2"):
            mission(spacecraft=(spec("solo", "TARGET"),))

    def test_version_gate(self):
        doc = mission_to_dict(mission())
        with pytest.raises(ValueError, match="schema_version"):
            mission_from_dict({k: v for k, v in doc.items() if k != "schema_version"})
        with pytest.raises(ValueError, match="no multi-spacecraft mission schema"):
            mission_from_dict(dict(doc, schema_version="9.9"))


class TestEventVocabulary:
    def test_vocabulary_is_the_multi_body_set(self):
        assert MISSION_EVENT_KINDS == ("maneuver", "hold_point_arrival", "zone_entry", "zone_exit")
        assert ROLES == ("TARGET", "CHASER")

    def test_kind_requires_its_reference(self):
        e = Epoch(MISSION_MJD2000, "UTC")
        with pytest.raises(ValueError, match="zone"):
            MissionEvent("chaser", "zone_entry", e)
        with pytest.raises(ValueError, match="hold_point_id"):
            MissionEvent("chaser", "hold_point_arrival", e)
        with pytest.raises(ValueError, match="only meaningful"):
            MissionEvent("chaser", "maneuver", e, zone="keepout")
        MissionEvent("chaser", "zone_entry", e, zone="keepout")  # contract-satisfying: fine
        MissionEvent("chaser", "hold_point_arrival", e, hold_point_id="hp-250m")

    def test_unknown_kind_and_self_counterpart(self):
        e = Epoch(MISSION_MJD2000, "UTC")
        with pytest.raises(ValueError, match="not in"):
            MissionEvent("chaser", "separation", e)  # a single-craft ascent kind is not multi-body
        with pytest.raises(ValueError, match="counterpart"):
            MissionEvent("chaser", "maneuver", e, counterpart_id="chaser")


class TestValidateMission:
    def test_nominal_mission_passes(self):
        report = validate_mission(mission())
        assert report.passed
        assert report.failures == ()

    def test_missing_and_multiple_roles(self):
        two_targets = validate_mission(mission(spacecraft=(spec("tgt", "TARGET"), spec("tow", "TARGET"))))
        reasons = ";".join(c.reason_code for c in two_targets.failures)
        assert "multiple_targets" in reasons and "no_chaser" in reasons
        no_target = validate_mission(mission(spacecraft=(spec("a"), spec("b"))))
        assert "no_target" in check(no_target, "msm.spacecraft_roles").reason_code

    def test_duplicate_ids_and_event_references(self):
        dup = validate_mission(mission(spacecraft=(spec("dup", "TARGET"), spec("dup", "CHASER"))))
        assert "duplicate_spacecraft_id" in check(dup, "msm.spacecraft_roles").reason_code
        m = mission(events=(
            MissionEvent("ghost", "maneuver", Epoch(MISSION_MJD2000, "UTC")),
            MissionEvent("chaser", "zone_entry", Epoch(MISSION_MJD2000, "UTC"), zone="undeclared"),
        ))
        refs = check(validate_mission(m), "msm.event_references")
        assert not refs.passed
        assert "unknown_spacecraft" in refs.reason_code and "unknown_zone" in refs.reason_code

    def test_track_frame_mixing_fails(self):
        report = validate_mission(mission(), (track("tgt"), track("chaser", frame="ECEF")))
        frames = check(report, "msm.frame_consistency")
        assert not frames.passed
        assert "frame_mismatch:chaser:ECEF" in frames.reason_code

    def test_track_epoch_mismatch_fails(self):
        skewed = track("chaser", r0=(7_010_000.0, 0.0, 0.0), v=(0.0, 7_489.0, 0.0), epoch_iso=iso_at(10.0))
        report = validate_mission(mission(), (track("tgt"), skewed))
        epoch = check(report, "msm.epoch_consistency")
        assert not epoch.passed and "epoch_mismatch:chaser" in epoch.reason_code
        # inside the tolerance the same pair passes
        aligned = validate_mission(mission(), (track("tgt"), skewed), MissionTolerances(epoch_alignment_s=30.0))
        assert check(aligned, "msm.epoch_consistency").passed

    def test_unit_error_fails_the_coherence_gate(self):
        """Velocity authored in km/s while position is in m: the positions
        say 7.5 m/s, the velocity column says 7.5 -- chord vs midpoint
        velocity disagrees by ~7490 m/s."""
        broken = track("chaser", r0=(7_010_000.0, 0.0, 0.0), v=(0.0, 7_489.0, 0.0))
        broken = dataclasses.replace(broken, velocity_m_s=tuple(
            (vx / 1000.0, vy / 1000.0, vz / 1000.0) for vx, vy, vz in broken.velocity_m_s))
        report = validate_mission(mission(), (track("tgt"), broken))
        coh = check(report, "msm.unit_coherence")
        assert not coh.passed and coh.reason_code == "unit_incoherence:chaser"
        assert coh.value > 1000.0

    def test_missing_duplicate_and_uncovered_tracks(self):
        missing = validate_mission(mission(), (track("tgt"),))
        assert "missing_track:chaser" in check(missing, "msm.track_coverage").reason_code
        orphan = validate_mission(mission(), nominal_tracks() + (track("probe"),))
        assert "unknown_track:probe" in check(orphan, "msm.track_coverage").reason_code
        dup = validate_mission(mission(), nominal_tracks() + (track("tgt"),))
        assert "duplicate_track:tgt" in check(dup, "msm.track_coverage").reason_code

    def test_empty_common_window_fails(self):
        early = track("tgt", t1=100.0)
        late = track("chaser", r0=(7_010_000.0, 0, 0), v=(0.0, 7_489.0, 0.0), t0=500.0, t1=900.0)
        report = validate_mission(mission(), (early, late))
        assert check(report, "msm.common_window").reason_code == "empty_common_window"

    def test_event_outside_track_coverage_fails(self):
        late_event = MissionEvent("chaser", "maneuver", Epoch(MISSION_MJD2000 + 10_000.0 / SECONDS_PER_DAY, "UTC"))
        report = validate_mission(mission(events=(late_event,)), nominal_tracks())
        assert "event_outside_track" in check(report, "msm.event_timeline_coverage").reason_code

    def test_event_inside_coverage_passes(self):
        ev = MissionEvent("chaser", "maneuver", Epoch(MISSION_MJD2000 + 120.0 / SECONDS_PER_DAY, "UTC"),
                          data={"delta_v_m_s": 0.4})
        assert validate_mission(mission(events=(ev,)), nominal_tracks()).passed


class TestUnifiedReplay:
    def test_two_tracks_replay_on_one_axis(self):
        m = mission(events=(MissionEvent("chaser", "maneuver",
                                         Epoch(MISSION_MJD2000 + 120.0 / SECONDS_PER_DAY, "UTC")),))
        tgt, chs = track("tgt"), track("chaser", r0=(7_010_000.0, 0.0, 0.0), v=(0.0, 7_489.0, 0.0),
                                      t0=60.0, t1=360.0, dt=45.0)  # offset sample grid on purpose
        replay = build_unified_replay(m, (tgt, chs))
        assert is_valid_replay_document(replay)
        assert replay_from_dict(replay)["mission_id"] == m.mission_id
        axis = replay["time_axis_tdb_s"]
        assert all(b > a for a, b in zip(axis, axis[1:]))
        assert len(axis) > 2
        # the union axis carries times from BOTH grids inside the overlap
        assert any(abs(t - (MISSION_TDB + 60.0)) < 1e-6 for t in axis)
        assert any(abs(t - (MISSION_TDB + 105.0)) < 1e-6 for t in axis)  # the chaser's 45 s grid, from tgt's 60 s grid absent
        by_id = {s["spacecraft_id"]: s for s in replay["spacecraft"]}
        assert {s["role"] for s in replay["spacecraft"]} == {"TARGET", "CHASER"}
        for craft in replay["spacecraft"]:
            assert [s["t_tdb_s"] for s in craft["samples"]] == axis  # one clock for everyone
        # straight-line constant-velocity tracks keep the analytic relative state
        for t, s_t, s_c in zip(axis, by_id["tgt"]["samples"], by_id["chaser"]["samples"]):
            r_rel = tuple(c - t_ for c, t_ in zip(s_c["position_m"], s_t["position_m"]))
            assert r_rel == pytest.approx((10_000.0, -11.0 * (t - MISSION_TDB), 0.0), abs=1e-6)
        assert len(replay["events"]) == 1
        assert replay["events"][0]["t_tdb_s"] == pytest.approx(MISSION_TDB + 120.0, abs=1e-3)
        assert replay["events"][0]["spacecraft_id"] == "chaser"
        assert replay["events"][0]["kind"] == "maneuver"

    def test_relative_states_and_track_query_agree(self):
        tgt, chs = nominal_tracks()
        m = mission()
        replay = build_unified_replay(m, (tgt, chs))
        rels = replay["relative_states"]
        assert len(rels) == len(replay["time_axis_tdb_s"])
        mid = replay["time_axis_tdb_s"][len(replay["time_axis_tdb_s"]) // 2]
        live = relative_state_from_tracks(m, tgt, chs, mid)
        at_mid = next(r for r in rels if r["t_tdb_s"] == mid)
        assert at_mid["position_m"] == pytest.approx(list(live.r_rel_m), abs=1e-9)
        assert at_mid["velocity_m_s"] == pytest.approx(list(live.v_rel_m_s), abs=1e-9)
        assert live.range_m > 0.0

    def test_invalid_input_never_reaches_the_replay_reader(self):
        m = mission()
        with pytest.raises(ValueError, match="failed validation"):
            build_unified_replay(m, (track("tgt"), track("chaser", frame="ECEF")))
        with pytest.raises(ValueError, match="failed validation"):
            build_unified_replay(m, (track("tgt"), track("chaser", epoch_iso=iso_at(60.0),
                                                         r0=(7_010_000.0, 0, 0), v=(0.0, 7_489.0, 0.0))))

    def test_replay_reader_rejects_a_hand_mangled_axis(self):
        replay = build_unified_replay(mission(), nominal_tracks())
        broken = json.loads(json.dumps(replay))
        broken["spacecraft"][0]["samples"].append(broken["spacecraft"][0]["samples"][0])
        with pytest.raises(ValueError):
            replay_from_dict(broken)

    def test_timescale_of_the_mission_epoch_does_not_move_the_axis(self):
        """The same physical mission epoch declared TT instead of UTC must
        put the shared axis on the same TDB-seconds numbers."""
        from orbitopt.timescales import tt_minus_utc
        tt_epoch = Epoch(MISSION_MJD2000 + tt_minus_utc(MISSION_MJD2000) / SECONDS_PER_DAY, "TT")
        utc_replay = build_unified_replay(mission(), nominal_tracks())
        tt_replay = build_unified_replay(mission(epoch=tt_epoch), nominal_tracks())
        for a, b in zip(utc_replay["time_axis_tdb_s"], tt_replay["time_axis_tdb_s"]):
            assert math.isclose(a, b, rel_tol=0, abs_tol=1e-3)

class TestReviewHardening:
    """Gates added after the fresh-eyes review: uniform-scale unit swap,
    pair queries on a >2 roster, an unparseable mission epoch, empty tracks."""

    def test_uniform_km_track_fails_the_plausibility_bound(self):
        # position AND velocity both written in km/km-s: chord-coherent, but
        # the orbit "radius" reads 7 km -- no body track can be that small
        km = track("tgt", r0=(7_000.0, 0.0, 0.0), v=(0.0, 7.5, 0.0))
        report = validate_mission(mission(), (km,))
        coh = check(report, "msm.unit_coherence")
        assert "implausible_radius:tgt" in coh.reason_code

    def test_relative_state_pair_query_works_on_a_three_craft_roster(self):
        three = mission(spacecraft=(spec("tgt", "TARGET"), spec("chaser"), spec("ch2")))
        tgt, chs = nominal_tracks()
        rel = relative_state_from_tracks(three, tgt, chs, MISSION_TDB + 60.0)
        assert rel.range_m > 0.0
        wrong_order = dataclasses.replace(three, spacecraft=(
            spec("tgt"), spec("chaser", "TARGET"), spec("ch2", "CHASER")))
        with pytest.raises(ValueError, match="TARGET track must come first"):
            relative_state_from_tracks(wrong_order, tgt, chs, MISSION_TDB + 60.0)

    def test_unparseable_mission_epoch_reports_instead_of_raising(self):
        report = validate_mission(mission(epoch=Epoch(-20_000.0, "UTC")), nominal_tracks())
        assert not report.passed
        assert "mission_epoch_unparseable" in check(report, "msm.epoch_consistency").reason_code

    def test_build_with_no_tracks_says_so(self):
        with pytest.raises(ValueError, match="needs a track per spacecraft"):
            build_unified_replay(mission(), ())
