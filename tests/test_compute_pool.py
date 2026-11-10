"""Lifecycle tests for Mission Control's compute worker process
(orbitopt.viz.compute_pool.ComputePool and orbitopt._compute_worker).

Headless and Qt-free except the SceneLoader/responsiveness tests at the end,
which only need QtCore (no display, no VTK). The pool always uses the
``spawn`` start method, so these exercise the same worker start-up path on
Linux CI as on macOS and Windows.

Scenarios (task #167): worker dies mid-compute / while idle, quit while a
compute is running (no zombie, no hang), repeated mission switches reuse one
worker, the app dying without cleanup, and the UI thread staying responsive
while the worker computes.
"""
from __future__ import annotations

import functools
import multiprocessing
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time

import pytest

from orbitopt.viz.compute_pool import ComputePool

# Generous: a spawned worker re-imports Python from scratch, which is slow on
# a loaded 2-vCPU CI runner. Only turns a hang into a failure.
SPAWN_TIMEOUT_S = 60.0

# Set in the parent only; a spawned (not forked) worker sees the default.
_PARENT_MARK = None


def _read_parent_mark():
    return _PARENT_MARK


def _busy(seconds: float) -> int:
    """Pure-Python CPU loop that holds the GIL the whole time -- the shape of
    the real mission loaders that starved the UI under a QThread."""
    deadline = time.perf_counter() + seconds
    n = 0
    while time.perf_counter() < deadline:
        n += 1
    return n


def _raise_value_error():
    raise ValueError("bad mission config")


# ----------------------------------------------------------------- helpers
def _pid_alive(pid: int) -> bool:
    """True while ``pid`` is a running (not zombie) process, on any OS."""
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    if sys.platform.startswith("linux"):
        try:
            with open(f"/proc/{pid}/stat") as f:
                state = f.read().rsplit(")", 1)[1].split()[0]
        except OSError:
            return False
        return state not in ("Z", "X")
    return True


def _zombie_children() -> list[int]:
    """Linux only: direct children of this process left unreaped."""
    if not sys.platform.startswith("linux"):
        return []
    me, zombies = os.getpid(), []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as f:
                fields = f.read().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if fields[0] == "Z" and int(fields[1]) == me:
            zombies.append(int(entry))
    return zombies


def _wait_until(predicate, timeout=SPAWN_TIMEOUT_S, interval=0.05):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out after {timeout}s waiting for {predicate}")
        time.sleep(interval)


def _assert_no_leftover_workers():
    # active_children() also reaps any finished child it finds, so check
    # zombies first.
    assert _zombie_children() == []
    assert multiprocessing.active_children() == []


@pytest.fixture
def pool():
    p = ComputePool()
    yield p
    p.terminate()


# ------------------------------------------------------------------- tests
def test_repeated_runs_reuse_one_spawned_worker(pool, monkeypatch):
    """Switching missions back and forth must not spawn a process per load,
    and the worker must be a fresh interpreter (spawn) on every platform --
    Linux's default fork would hide pickling/import bugs that only show up
    on macOS/Windows."""
    monkeypatch.setattr(sys.modules[__name__], "_PARENT_MARK", "parent")
    assert pool.worker_pids() == []

    pids = {pool.run(os.getpid) for _ in range(5)}
    assert len(pids) == 1
    (pid,) = pids
    assert pid != os.getpid()
    assert pool.worker_pids() == [pid]
    assert pool.run(_read_parent_mark) is None  # module state not inherited


def test_exception_in_loader_propagates_and_keeps_worker(pool):
    pid = pool.run(os.getpid)
    with pytest.raises(ValueError, match="bad mission config"):
        pool.run(_raise_value_error)
    assert pool.run(os.getpid) == pid  # an ordinary error is not a crash


def test_worker_crash_mid_compute_fails_call_and_recovers(pool):
    old_pid = pool.run(os.getpid)
    with pytest.raises(RuntimeError, match="exited unexpectedly"):
        pool.run(functools.partial(os._exit, 3))  # native-code crash stand-in
    assert not _pid_alive(old_pid)
    assert _zombie_children() == []

    new_pid = pool.run(os.getpid)
    assert new_pid != old_pid
    assert pool.worker_pids() == [new_pid]


def test_idle_worker_death_is_replaced_transparently(pool):
    """A worker killed between loads (OOM killer, Activity Monitor/Task
    Manager) must not make the *next* load fail."""
    old_pid = pool.run(os.getpid)
    os.kill(old_pid, signal.SIGTERM)  # TerminateProcess on Windows
    _wait_until(lambda: not _pid_alive(old_pid))

    new_pid = pool.run(os.getpid)
    assert new_pid != old_pid
    assert _zombie_children() == []


def test_terminate_during_compute_unblocks_caller_and_reaps_worker():
    """Quitting mid-compute: the loader thread blocked in run() returns
    promptly (closeEvent waits on it) and the worker is reaped, not left as
    a zombie or an orphan still computing."""
    pool = ComputePool()
    outcome = {}

    def call():
        try:
            outcome["result"] = pool.run(functools.partial(time.sleep, 120))
        except Exception as exc:  # noqa: BLE001
            outcome["error"] = exc

    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    _wait_until(lambda: pool.worker_pids())
    (pid,) = pool.worker_pids()
    time.sleep(0.5)  # let the task reach the worker

    start = time.monotonic()
    pool.terminate()
    thread.join(10.0)
    elapsed = time.monotonic() - start

    assert not thread.is_alive(), "run() still blocked after terminate()"
    assert elapsed < 10.0
    assert isinstance(outcome.get("error"), RuntimeError)
    assert not _pid_alive(pid)
    _assert_no_leftover_workers()


def test_terminate_closes_pool_and_is_idempotent():
    pool = ComputePool()
    pool.terminate()  # before any worker exists
    pool.terminate()
    assert pool.closed
    with pytest.raises(RuntimeError, match="shut down"):
        pool.run(os.getpid)
    assert pool.worker_pids() == []
    _assert_no_leftover_workers()


def test_terminate_racing_a_starting_run_leaves_no_worker():
    """closeEvent can call terminate() while the loader thread is between
    'no executor yet' and 'submit'. The pool must not spawn a worker after
    it was terminated."""
    for _ in range(5):
        pool = ComputePool()
        errors = []

        def call(pool=pool, errors=errors):
            try:
                pool.run(functools.partial(time.sleep, 120))
            except RuntimeError as exc:
                errors.append(exc)

        thread = threading.Thread(target=call, daemon=True)
        thread.start()
        pool.terminate()
        thread.join(SPAWN_TIMEOUT_S)
        assert not thread.is_alive()
        assert len(errors) == 1
        assert pool.worker_pids() == []
    _assert_no_leftover_workers()


_ORPHAN_SCRIPT = textwrap.dedent("""
    import functools, sys, threading, time
    from orbitopt.viz.compute_pool import ComputePool

    if __name__ == "__main__":
        pool = ComputePool()
        threading.Thread(
            target=lambda: pool.run(functools.partial(time.sleep, 300)),
            daemon=True).start()
        while not pool.worker_pids():
            time.sleep(0.05)
        print(pool.worker_pids()[0], flush=True)
        time.sleep(300)
""")


def test_worker_exits_when_app_is_killed(tmp_path):
    """The app crashing or being force-quit (no closeEvent) must not leave
    the worker computing as an orphan. On Windows an orphan is never
    re-parented, so this specifically covers the parent-sentinel watchdog."""
    script = tmp_path / "fake_app.py"
    script.write_text(_ORPHAN_SCRIPT)
    env = dict(os.environ)
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [src, env.get("PYTHONPATH")]))
    app = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, env=env, text=True)
    try:
        line = app.stdout.readline()
        assert line.strip().isdigit(), f"fake app did not report a worker pid: {line!r}"
        worker_pid = int(line)
        assert _pid_alive(worker_pid)
        app.kill()  # SIGKILL / TerminateProcess: no cleanup code runs
        app.wait(10)
        _wait_until(lambda: not _pid_alive(worker_pid), timeout=10.0)
    finally:
        if app.poll() is None:
            app.kill()
        app.stdout.close()


# ---------------------------------------------- SceneLoader (needs QtCore)
@pytest.fixture
def qt_core():
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QCoreApplication

    app = QCoreApplication.instance()
    if app is not None:
        return app
    # Qt allows one application object per process and it cannot be upgraded.
    # A bare QCoreApplication created here would be returned by the widget
    # tests' ``QApplication.instance() or QApplication([])`` and the first
    # QWidget would SIGABRT the whole pytest run, so create the full
    # QApplication whenever QtWidgets is importable.
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:  # headless box without the GUI libs: core is enough
        return QCoreApplication([])
    return QApplication([])


def _run_loader(loader):
    got = {}
    loader.finished.connect(lambda name, scene: got.update(finished=(name, scene)))
    loader.failed.connect(lambda name, msg: got.update(failed=(name, msg)))
    loader.run()
    return got


def test_scene_loader_reports_worker_crash_as_failure(qt_core, pool):
    from orbitopt.viz.compute import SceneLoader

    got = _run_loader(SceneLoader("crash", functools.partial(os._exit, 1), pool))
    assert got["failed"][0] == "crash"
    assert "exited unexpectedly" in got["failed"][1]
    # ...and the next mission loads on a fresh worker.
    got = _run_loader(SceneLoader("ok", functools.partial(dict, id="x"), pool))
    assert got["finished"] == ("ok", {"id": "x"})


def test_scene_loader_after_quit_fails_without_spawning(qt_core):
    from orbitopt.viz.compute import SceneLoader

    pool = ComputePool()
    pool.terminate()
    got = _run_loader(SceneLoader("late", functools.partial(dict, id="x"), pool))
    assert "failed" in got and "finished" not in got
    _assert_no_leftover_workers()


def test_scene_loader_runs_unpicklable_loader_in_thread(qt_core, pool):
    from orbitopt.viz.compute import SceneLoader

    got = _run_loader(SceneLoader("file", lambda: {"pid": os.getpid()}, pool))
    assert got["finished"] == ("file", {"pid": os.getpid()})
    assert pool.worker_pids() == []  # File > Open never starts the worker


def test_ui_timer_stays_responsive_during_worker_compute(qt_core, pool):
    """The point of the worker process: a GIL-holding compute must not starve
    the main thread. Same measurement as commit 4ed2d5e (a 16 ms timer on the
    main thread), with loose bounds so a busy CI runner doesn't flake."""
    from PySide6.QtCore import QThread, QTimer

    from orbitopt.viz.compute import SceneLoader

    pool.run(os.getpid)  # pay the spawn start-up outside the measurement
    compute_s = 2.0
    ticks: list[float] = []
    done = {}

    timer = QTimer()
    timer.setInterval(16)
    timer.timeout.connect(lambda: ticks.append(time.perf_counter()))

    thread = QThread()
    loader = SceneLoader("busy", functools.partial(_busy, compute_s), pool)
    loader.moveToThread(thread)
    thread.started.connect(loader.run)
    loader.finished.connect(lambda name, scene: done.update(ok=scene))
    loader.failed.connect(lambda name, msg: done.update(err=msg))
    loader.finished.connect(thread.quit)
    loader.failed.connect(thread.quit)

    timer.start()
    thread.start()
    deadline = time.monotonic() + SPAWN_TIMEOUT_S
    while thread.isRunning() and time.monotonic() < deadline:
        qt_core.processEvents()
        time.sleep(0.001)
    timer.stop()
    assert thread.wait(5000)
    qt_core.processEvents()

    assert "ok" in done, done
    gaps = [b - a for a, b in zip(ticks, ticks[1:])]
    expected = compute_s / 0.016
    assert len(ticks) > 0.3 * expected, f"only {len(ticks)} of ~{expected:.0f} ticks"
    assert max(gaps) < 0.5, f"main thread stalled for {max(gaps):.2f}s"
