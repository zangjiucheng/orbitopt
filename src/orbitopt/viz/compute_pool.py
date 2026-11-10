"""Mission Control's compute worker process (Qt-free, so its lifecycle can be
tested headless -- see tests/test_compute_pool.py)."""
from __future__ import annotations

import os
import threading

#: How long terminate() waits for a terminated worker to exit before it
#: escalates to kill(), and again after kill(). Termination is near-instant on
#: every platform; the bound only keeps a wedged process from blocking quit.
JOIN_TIMEOUT_S = 2.0

#: How long an idle worker gets to answer the pre-submit health check before
#: it is treated as wedged and replaced.
PING_TIMEOUT_S = 5.0


class ComputePool:
    """A single persistent worker *process* for mission compute.

    A QThread alone does not keep the UI responsive here: the loaders
    (pykep/pygmo Python callbacks, small-array numpy, tudatpy bindings) hold
    the GIL nearly the whole time, so the main thread is starved. Measured
    while computing Artemis II with no scene cache: a 16 ms main-thread timer
    fired 63 times in 20 s instead of ~1,260, with gaps up to 1.08 s. A
    separate process has its own GIL. SceneLoader still runs on a QThread,
    but that thread only blocks in ``future.result()``, which releases the GIL.

    ``spawn`` (not fork) on every platform: forking a process that already
    holds Qt, VTK and an OpenGL context is unsafe on macOS, Windows has no
    fork, and using one start method everywhere means Linux CI exercises the
    same pickling/import path as macOS and Windows users. The worker is started
    lazily on the first submit and reused, so heavy imports are paid once per
    session. ``ORBITOPT_INPROCESS_COMPUTE=1`` disables it (debugging, profilers).

    Lifecycle guarantees (covered by tests/test_compute_pool.py):

    * a worker that dies mid-compute fails that one call with RuntimeError and
      is reaped; the next call starts a fresh worker;
    * a worker that died while idle is replaced transparently on the next call;
    * ``terminate()`` (on quit) kills and reaps the worker, unblocks a caller
      waiting in ``run()``, and permanently closes the pool, so a load racing
      the quit cannot spawn a new worker after it;
    * the worker exits on its own within ~1 s if the app dies without calling
      ``terminate()`` (orbitopt._compute_worker.watch_parent).

    ``run()`` is called from the loader QThread and ``terminate()`` from the
    main thread, so all executor bookkeeping happens under one lock.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._executor = None
        self._closed = False

    @staticmethod
    def enabled() -> bool:
        return os.environ.get("ORBITOPT_INPROCESS_COMPUTE", "") != "1"

    @property
    def closed(self) -> bool:
        return self._closed

    def worker_pids(self) -> list[int]:
        """PIDs of the live worker process(es) -- empty before the first run()
        and after terminate()."""
        with self._lock:
            return [p.pid for p in _processes(self._executor) if _alive(p)]

    def run(self, fn):
        """Call ``fn()`` in the worker process and return its result (blocking)."""
        from concurrent.futures.process import BrokenProcessPool

        with self._lock:
            if self._closed:
                raise RuntimeError("the compute process has been shut down")
            executor = self._live_executor()
            try:
                future = executor.submit(fn)
            except BrokenProcessPool:
                # The pool noticed an idle-worker death before we did: nothing
                # of ours ran yet, so retrying on a fresh worker is safe.
                self._discard(executor)
                executor = self._live_executor()
                future = executor.submit(fn)
        try:
            # Outside the lock: terminate() must be able to get in and kill
            # the worker while we wait here.
            return future.result()
        except BrokenProcessPool as exc:
            # The worker died (crash in native code, OOM kill, or terminate()
            # on quit). Drop the broken executor so the next load starts a
            # fresh one -- unless terminate() already swapped it out.
            with self._lock:
                if self._executor is executor:
                    self._discard(executor)
            raise RuntimeError("the compute process exited unexpectedly") from exc

    def terminate(self):
        """Kill the worker immediately and close the pool (on quit) -- a
        mission compute can run for minutes and must not hold the app open.
        Safe to call more than once and from any thread."""
        with self._lock:
            self._closed = True
            executor, self._executor = self._executor, None
        if executor is not None:
            _kill_and_reap(executor)

    def restart(self):
        """Kill the worker but keep the pool open: the next run() starts a
        fresh process, which imports the code as it is on disk now (reload
        after a source edit). Call it only while no run() is in flight --
        that run would fail with RuntimeError."""
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            _kill_and_reap(executor)

    # ------------------------------------------------------------ internals
    def _live_executor(self):
        """The current executor, replacing it first if its worker has died
        while idle (caller holds the lock).

        An existing worker is pinged with a trivial task first. Checking its
        process state is not enough: right after a worker dies, the executor's
        manager thread may have reaped it without having marked the pool
        broken yet, and the next submit would then fail with the *user's*
        load reported as the crash. The ping costs one round-trip (~1 ms) per
        load, which is nothing next to a mission compute."""
        if self._executor is not None and not self._ping(self._executor):
            self._discard(self._executor)
        if self._executor is None:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            from orbitopt._compute_worker import watch_parent

            self._executor = ProcessPoolExecutor(
                max_workers=1, mp_context=mp.get_context("spawn"),
                initializer=watch_parent, initargs=(os.getpid(),),
            )
        return self._executor

    @staticmethod
    def _ping(executor) -> bool:
        if getattr(executor, "_broken", False):
            return False
        try:
            executor.submit(os.getpid).result(timeout=PING_TIMEOUT_S)
        except Exception:  # noqa: BLE001 -- broken pool, timeout (wedged worker), shutdown
            return False
        return True

    def _discard(self, executor):
        if self._executor is executor:
            self._executor = None
        _kill_and_reap(executor)


def _processes(executor) -> list:
    if executor is None:
        return []
    # ProcessPoolExecutor._processes is private but stable since 3.x and the
    # only handle on the worker; it is set to None after shutdown.
    return list((getattr(executor, "_processes", None) or {}).values())


def _alive(proc) -> bool:
    """Process liveness that doesn't race the executor's manager thread.
    ``Process.is_alive()`` polls with waitpid(); if the manager thread reaped
    the dead worker first, that poll fails with ECHILD and reports the dead
    worker as *alive*. The sentinel (a pipe on POSIX, a process handle on
    Windows) becomes ready when the process exits, whoever reaps it."""
    from multiprocessing.connection import wait

    try:
        if proc.exitcode is not None:
            return False
        return not wait([proc.sentinel], timeout=0)
    except (OSError, ValueError):  # sentinel closed: the process is gone
        return False


def _kill_and_reap(executor) -> None:
    """Terminate the executor's workers and join them so none is left as a
    zombie, then release the executor without waiting on its queue."""
    procs = _processes(executor)
    for proc in procs:
        try:
            if _alive(proc):
                proc.terminate()
        except Exception:  # noqa: BLE001 -- best-effort teardown
            pass
    for proc in procs:
        try:
            proc.join(JOIN_TIMEOUT_S)
            if _alive(proc):
                proc.kill()
                proc.join(JOIN_TIMEOUT_S)
        except Exception:  # noqa: BLE001 -- best-effort teardown
            pass
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception:  # noqa: BLE001 -- best-effort teardown
        pass
