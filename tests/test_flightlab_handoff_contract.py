"""M-01 handoff contract (R-02, task #234): the stdlib-only ``HandoffState``
document. Round-trip, the schema-version policy, and the validation rules
that make a malformed handoff impossible instead of merely discouraged: a
foreign phase boundary, a non-contract frame, a mass below the dry mass or
a timezone-less epoch must be rejected at construction, not discovered by
the continuity checker downstream.
"""
from __future__ import annotations

import pytest

from orbitopt.flightlab.handoff import (
    HANDOFF_KIND,
    HANDOFF_SCHEMA_VERSION,
    PHASE_BOOSTBACK,
    PHASE_ENTRY,
    PHASE_LANDING,
    PHASE_SEPARATION,
    PHASE_TOUCHDOWN,
    RECOVERY_HANDOFF_BOUNDARIES,
    RECOVERY_PHASES,
    HandoffState,
    handoff_digest,
)

BOUNDARY = (PHASE_BOOSTBACK, PHASE_ENTRY)


def _state(**over):
    kw = dict(from_phase=PHASE_BOOSTBACK, to_phase=PHASE_ENTRY, t_s=312.5,
              epoch_utc="2026-11-10T12:05:12Z", frame="ECEF",
              position_m=(-2_345_678.9, 1_234_567.8, 4_567_890.1),
              velocity_m_s=(-120.5, 34.25, -301.0),
              mass_kg=52_345.0, dry_mass_kg=30_000.0, ignitions_used=1)
    kw.update(over)
    return HandoffState(**kw)


def test_vocabulary_is_the_recovery_chain():
    assert RECOVERY_PHASES == ("boostback", "entry", "landing")
    assert RECOVERY_HANDOFF_BOUNDARIES == (
        ("separation", "boostback"), ("boostback", "entry"),
        ("entry", "landing"), ("landing", "touchdown"))


def test_dict_round_trip_is_identity_and_digest_is_stable():
    st = _state()
    assert HandoffState.from_dict(st.to_dict()) == st
    doc = st.to_dict()
    assert doc["kind"] == HANDOFF_KIND
    assert doc["schema_version"] == HANDOFF_SCHEMA_VERSION
    assert handoff_digest(st) == handoff_digest(_state())
    assert handoff_digest(st) != handoff_digest(_state(mass_kg=52_344.0))


def test_from_dict_ignores_unknown_optional_fields():
    doc = _state().to_dict()
    doc["some_future_field"] = 42
    assert HandoffState.from_dict(doc) == _state()


def test_from_dict_rejects_foreign_kind_and_wrong_version():
    doc = _state().to_dict()
    doc["kind"] = "orbitopt.something-else"
    with pytest.raises(ValueError, match="kind"):
        HandoffState.from_dict(doc)
    doc = _state().to_dict()
    doc["schema_version"] = "9.9"
    with pytest.raises(ValueError, match="version"):
        HandoffState.from_dict(doc)


def test_construction_rejects_malformed_handoffs():
    with pytest.raises(ValueError, match="boundary"):
        _state(from_phase=PHASE_SEPARATION)          # separation -> entry
    with pytest.raises(ValueError, match="boundary"):
        _state(to_phase=PHASE_TOUCHDOWN)             # boostback -> touchdown
    with pytest.raises(ValueError, match="frame"):
        _state(frame="ENU")
    with pytest.raises(ValueError, match="dry mass"):
        _state(mass_kg=29_999.0)                     # below dry mass
    with pytest.raises(ValueError, match="epoch_utc"):
        _state(epoch_utc="2026-11-10T12:05:12")      # no zone -> not a UTC stamp
    with pytest.raises(ValueError, match="epoch_utc"):
        _state(epoch_utc="yesterday")


def test_propellant_and_boundary_view():
    st = _state()
    assert st.propellant_kg == pytest.approx(22_345.0)
    assert st.boundary == BOUNDARY


def test_of_accepts_duck_typed_states_and_datetimes():
    from datetime import datetime

    class Producer:  # carries more than the contract needs -- fine
        t_s = 312.5
        epoch_utc = datetime(2026, 11, 10, 12, 5, 12)  # naive = UTC
        frame = "GCRS"
        position_m = (1.0, 2.0, 3.0)
        velocity_m_s = (4.0, 5.0, 6.0)
        mass_kg = 40_000.0
        extra = "ignored"

    st = HandoffState.of(Producer(), from_phase=PHASE_SEPARATION,
                         to_phase=PHASE_BOOSTBACK, dry_mass_kg=30_000.0)
    assert st.epoch_utc == "2026-11-10T12:05:12Z"
    assert st.propellant_kg == pytest.approx(10_000.0)
    # a HandoffState is itself a legal producer (passing one onward)
    again = HandoffState.of(_state(), from_phase=PHASE_ENTRY,
                            to_phase=PHASE_LANDING, dry_mass_kg=30_000.0)
    assert again.position_m == _state().position_m
