"""Mission Control window <-> compute worker process lifecycle (task #167).

Like test_app.py these open a real window (VTK needs a real GL context), so
they need a display -- CI runs them under xvfb. The pool-level behaviour is
covered headless in test_compute_pool.py; these check the wiring in
MissionControlWindow: quit mid-compute, a crashing worker, and repeated
mission switches. Loaders are stdlib partials (picklable, so they take the
worker-process path exactly like the registered missions) to keep each test
to a few seconds and independent of the pykep/tudatpy compute stack.
"""
from __future__ import annotations

import copy
import functools
import multiprocessing
import os
import time

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyvistaqt")

from test_app import _pump_until
from test_compute_pool import _pid_alive, _zombie_children

# closeEvent must return promptly even with a minutes-long compute in flight.
QUIT_BUDGET_S = 10.0


@pytest.fixture
def window():
    from PySide6.QtWidgets import QApplication

    from orbitopt.viz.app import MissionControlWindow

    app = QApplication.instance() or QApplication([])
    win = MissionControlWindow()
    win.show()
    _pump_until(app, lambda: not win._loading)
    assert win.state.scene is not None
    win.app = app
    yield win
    if not win._closing:
        win.plotter.close()
        win.close()


def _close(win) -> float:
    start = time.monotonic()
    win.plotter.close()  # release the VTK/OpenGL context first, as test_app does
    win.close()
    return time.monotonic() - start


def test_quit_during_compute_kills_worker_without_hanging(window):
    window._loaders["slow"] = functools.partial(time.sleep, 600)
    window._start_loading("slow")
    assert window._loading
    _pump_until(window.app, lambda: window._compute_pool.worker_pids(), timeout=60.0)
    (pid,) = window._compute_pool.worker_pids()
    time.sleep(0.3)  # the task is now running in the worker

    elapsed = _close(window)
    for _ in range(5):
        window.app.processEvents()  # deliver any queued finished/failed: must be a no-op

    assert elapsed < QUIT_BUDGET_S, f"closeEvent took {elapsed:.1f}s"
    assert not _pid_alive(pid)
    assert _zombie_children() == []
    assert multiprocessing.active_children() == []


def test_quit_while_idle_kills_worker(window):
    pids = window._compute_pool.worker_pids()  # the startup load ran in the worker
    elapsed = _close(window)
    assert elapsed < QUIT_BUDGET_S
    assert all(not _pid_alive(p) for p in pids)
    assert multiprocessing.active_children() == []


def test_worker_crash_is_reported_and_next_load_works(window, monkeypatch):
    import orbitopt.viz.app as app_module

    errors = []
    monkeypatch.setattr(app_module.QMessageBox, "critical",
                        staticmethod(lambda parent, title, text: errors.append(text)))
    first_pids = window._compute_pool.worker_pids()

    window._loaders["crash"] = functools.partial(os._exit, 1)  # native crash stand-in
    window._start_loading("crash")
    _pump_until(window.app, lambda: not window._loading, timeout=60.0)

    assert len(errors) == 1 and "exited unexpectedly" in errors[0]
    assert window.load_status_badge._label.text() == "FAILED"
    assert window.sidebar.mission_list.isEnabled()  # UI not stuck in "loading"
    assert all(not _pid_alive(p) for p in first_pids)
    assert _zombie_children() == []

    window._loaders["again"] = functools.partial(copy.deepcopy, window.state.scene)
    window._start_loading("again")
    _pump_until(window.app, lambda: not window._loading, timeout=60.0)
    assert window._scene_cache.get("again") is not None
    assert window.load_status_badge._label.text() == "READY"
    new_pids = window._compute_pool.worker_pids()
    assert len(new_pids) == 1 and new_pids[0] not in first_pids


def test_repeated_switching_reuses_one_worker(window):
    scene = window.state.scene
    for key in ("a", "b"):
        window._loaders[key] = functools.partial(copy.deepcopy, scene)

    pids = set()
    for i in range(6):
        key = "ab"[i % 2]
        window._scene_cache.pop(key, None)  # force the worker path every time
        window._on_mission_activated(key, key)
        thread = window._load_thread
        # A second activation while loading is ignored, not a second thread.
        window._on_mission_activated("ab"[(i + 1) % 2], "other")
        assert window._load_thread is thread
        _pump_until(window.app, lambda: not window._loading, timeout=60.0)
        assert window.load_status_badge._label.text() == "READY"
        pids.update(window._compute_pool.worker_pids())

    assert len(pids) == 1
    assert len(multiprocessing.active_children()) == 1
    assert _zombie_children() == []
