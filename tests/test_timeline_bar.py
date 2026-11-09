"""TimelineBar readouts follow the loaded scene."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


def _scene(ref_et=None):
    timeline = {"min": 0.0, "max": 4.0, "unitLabel": "days", "events": []}
    if ref_et is not None:
        timeline["referenceEpochEt"] = ref_et
    return {"id": "t", "title": "T", "distanceUnit": "km", "bodies": [], "timeline": timeline}


def test_date_readout_is_cleared_for_a_timeline_without_an_epoch():
    """GOES has no calendar epoch: switching to it from a mission that has one
    used to leave the previous mission's date on screen."""
    from PySide6.QtWidgets import QApplication

    from orbitopt.viz.state import MissionState
    from orbitopt.viz.timeline_bar import TimelineBar

    QApplication.instance() or QApplication([])
    state = MissionState()
    bar = TimelineBar(state)
    state.set_scene(_scene(ref_et=838814469.0))
    assert bar.date_label.text().endswith("UTC")
    state.set_scene(_scene())
    assert bar.date_label.text() == ""
    state.set_time(2.0)
    assert bar.date_label.text() == "" and bar.met_label.text() == "T+2.00 days"
