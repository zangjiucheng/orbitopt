"""FL-502 designer page + workbench GUI tests (slow: needs PySide6/VTK).

Covers what the Qt-free core tests cannot: form -> field-level red-border
errors on real widgets, Explorer/Engineer visibility, the simulate worker
thread on the compute pool, the cache-hit shortcut, Cancel killing the
worker's process group, and the designer -> replay handoff. Same env stub as
the FL-501 replay tests."""
from __future__ import annotations

import datetime as dt
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
OT_FIX = Path(__file__).parent / "fixtures" / "flightlab" / "opentsiolkovsky"

pytestmark = pytest.mark.slow


def _wait(predicate, timeout=60.0, app=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if app is not None:
            app.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _live_members(pgid: int) -> list[int]:
    """PIDs in process group ``pgid`` that are actually running. Zombies are
    excluded: this dev container's PID 1 never reaps orphaned grandchildren,
    so "killed" here means "no live member", not "no entry in /proc".
    No /proc (macOS, BSD): ``ps``, which reports the same pid/pgid/state."""
    if not Path("/proc").is_dir():
        out = subprocess.run(["ps", "-A", "-o", "pid=,pgid=,stat="],
                             capture_output=True, text=True, check=True).stdout
        return [int(pid) for pid, group, stat in (line.split()[:3] for line in out.splitlines()
                                                  if len(line.split()) >= 3)
                if int(group) == pgid and not stat.startswith("Z")]
    live = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            after = (entry / "stat").read_text().split(") ", 1)[1].split()
        except OSError:
            continue
        if int(after[2]) == pgid and after[0] != "Z":
            live.append(int(entry.name))
    return live


@pytest.fixture()
def qapp():
    pytest.importorskip("PySide6")
    pytest.importorskip("pyvista")
    pytest.importorskip("pyvistaqt")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture()
def stub_env(tmp_path, monkeypatch):
    """Real CLI stub, optionally wrapped so the run stalls long enough for
    the Cancel test (sleep then exec the stub, so Gate-B output still lands
    if the run is allowed to finish)."""
    cli = tmp_path / "otcli"
    shutil.copy(OT_FIX / "stub_cli.py", cli)
    cli.chmod(0o755)
    wrapper = tmp_path / "slow_otcli"
    wrapper.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys, time\n"
        "if len(sys.argv) > 1 and sys.argv[1] == 'run':\n"
        "    time.sleep(float(os.environ.get('OT_STUB_DELAY_S', '30')))\n"
        f"os.execv(sys.executable, [sys.executable, {str(cli)!r}] + sys.argv[1:])\n",
        encoding="utf-8")
    wrapper.chmod(0o755)
    monkeypatch.setenv("ORBITOPT_OPENTSIOLKOVSKY", str(cli))
    monkeypatch.setenv("OT_STUB_FIXTURE", str(OT_FIX / "synthetic_out"))
    monkeypatch.setenv("OT_STUB_SRC", str(REPO / "src" / "orbitopt"))
    monkeypatch.setenv("ORBITOPT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("ORBITOPT_INPROCESS_COMPUTE", raising=False)
    monkeypatch.setenv("OT_STUB_DELAY_S", "0")
    return {"cli": cli, "wrapper": wrapper}


def test_workbench_shows_designer_and_replay_and_delegates_playback(qapp, stub_env, tmp_path):
    from orbitopt.flightlab.ui import FlightLabWorkbench

    wb = FlightLabWorkbench()
    wb.show()
    qapp.processEvents()
    try:
        assert wb.tabs.tabText(0) == "Designer"
        assert wb.tabs.tabText(1) == "Replay"
        page = wb.designer_page
        assert page._validate() and page.simulate_button.isEnabled()
        assert "T/W" in page.derived.text()

        page.apoapsis.setValue(100.0)
        qapp.processEvents()
        assert not page._validate()
        assert page.apoapsis.property("flErr") == "1"
        assert "below periapsis" in page.apoapsis.toolTip()
        assert not page.simulate_button.isEnabled()
        page.apoapsis.setValue(400.0)
        assert page._validate()

        page.mode_button.setChecked(True)  # engineer mode reveals the deep fields
        qapp.processEvents()
        assert page._cards[0].burn_start.isVisibleTo(page) and page._cards[0].cd_button.isVisibleTo(page)
        page.mode_button.setChecked(False)
        qapp.processEvents()
        assert not page._cards[0].burn_start.isVisibleTo(page)

        for verb in ("stop_playback", "toggle_playback", "restart"):
            getattr(wb, verb)()  # delegates, no raise
        wb.step_time(+1)
        wb.seek_relative_event(+1)
        wb.terminate_compute()
    finally:
        wb.deleteLater()


def test_simulate_flies_caches_and_opens_replay(qapp, stub_env, tmp_path):
    from orbitopt.flightlab.ui import FlightLabWorkbench

    wb = FlightLabWorkbench()
    wb.show()
    qapp.processEvents()
    page = wb.designer_page
    opened: list[str] = []
    wb.designer_page.run_ready.connect(lambda path: opened.append(path))
    try:
        page.simulate()
        assert _wait(lambda: opened, 90.0, qapp), "worker never reported a run"
        run_dir = Path(opened[0])
        assert (run_dir / "manifest.json").exists()
        assert wb.tabs.currentWidget() is wb.replay_page
        assert wb.replay_page._run is not None  # replay opened the finished run

        # same inputs -> instant cache hit, no second worker
        assert page._validate()
        assert "cached" in page.simulate_button.text()
        before = len(opened)
        page.simulate()
        assert _wait(lambda: len(opened) > before, 5.0, qapp)
        assert opened[-1] == str(run_dir)
        assert page._thread is None
    finally:
        page.terminate_compute()
        wb.deleteLater()


def test_cancel_kills_the_worker_process_group(qapp, stub_env, tmp_path, monkeypatch):
    import orbitopt.flightlab.designer as D
    from orbitopt.flightlab.ui import FlightLabDesignerPage

    monkeypatch.setenv("ORBITOPT_OPENTSIOLKOVSKY", str(stub_env["wrapper"]))
    monkeypatch.setenv("OT_STUB_DELAY_S", "30")
    page = FlightLabDesignerPage()
    page.show()
    qapp.processEvents()
    try:
        page.simulate()
        # watch the *page's* run dir -- build_job lives inside simulate(),
        # and capture the path eagerly: the page clears _job on teardown
        def marker_path():
            return Path(page._job["run_dir"]) / D.IN_PROGRESS if page._job else None
        assert _wait(lambda: (marker_path() is not None and marker_path().exists()
                              and bool(page._pool and page._pool.worker_pids())),
                     30.0, qapp), "worker never started the ascent"
        marker = marker_path()
        pgid = int(marker.read_text(encoding="ascii"))
        assert pgid > 1 and pgid not in {os.getpid(), os.getpgrp()}
        if os.name == "posix":
            assert _live_members(pgid), "worker group was never actually running"

        page.cancel_simulate()
        assert _wait(lambda: page._thread is None, 30.0, qapp), "cancel did not retire the thread"
        assert not marker.exists()  # the page cleaned the in-flight marker
        assert "cancelled" in page.status.text()
        assert page.badge._state != "failed"
        assert page.simulate_button.isEnabled()  # ready to run again immediately
        if os.name == "posix":
            # both the worker and its OpenTsiolkovsky grandchild share the
            # recorded group -- nothing live may remain in it
            assert _wait(lambda: not _live_members(pgid), 15.0, qapp), \
                f"live members left in group {pgid}"
    finally:
        page.terminate_compute()
        page.deleteLater()


def test_explorer_mode_hides_every_engineer_field_and_caption(qapp):
    """Explorer mode left the stage cards' engineer fields up (the cards are
    built after the mode is applied) and the RAAN caption without its field
    (a second, unregistered label); "remove stage" sat on the Coast
    behaviour combo's grid cell; and removing a stage left deleted widgets
    for the next mode switch to trip over."""
    from PySide6.QtWidgets import QLabel

    from orbitopt.flightlab.ui import FlightLabDesignerPage

    page = FlightLabDesignerPage()
    try:
        assert not page._engineer and page._cards
        card = page._cards[0]
        for field in (card.burn_start, card.throttle, card.nozzle, card.free_flight):
            assert not field.isVisibleTo(page)
        assert card.mass_initial.isVisibleTo(page)
        raan_captions = [w for w in page.findChildren(QLabel) if w.text() == "RAAN"]
        assert raan_captions and not any(w.isVisibleTo(page) for w in raan_captions)

        page._toggle_mode(True)
        assert card.free_flight.isVisibleTo(page) and card.nozzle.isVisibleTo(page)
        grid = card.layout()
        at = grid.itemAtPosition(6, 3)
        assert at is not None and at.widget() is card.free_flight  # not covered by "remove stage"

        page._add_stage()
        qapp.processEvents()
        assert page._cards[-1].nozzle.isVisibleTo(page)  # a new card follows the current mode
        page._remove_stage(page._cards[-1])
        qapp.processEvents()
        page._toggle_mode(False)  # no RuntimeError on the removed card's widgets
        page._toggle_mode(True)
    finally:
        page.terminate_compute()
        page.deleteLater()
