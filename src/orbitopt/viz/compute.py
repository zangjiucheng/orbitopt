"""Background compute plumbing for Mission Control: the worker process that
runs mission loaders, the QThread object that waits on it, and the one-shot
GPU-backend probe. None of this touches Qt widgets or VTK -- results travel
back to the main thread through queued signals (see SceneLoader)."""
from __future__ import annotations

import os

from PySide6.QtCore import QObject, Signal


def _is_picklable(fn) -> bool:
    import pickle

    try:
        pickle.dumps(fn)
    except Exception:  # noqa: BLE001 -- lambdas/closures (e.g. File > Open) just run in-thread
        return False
    return True


class ComputePool:
    """A single persistent worker *process* for mission compute.

    A QThread alone does not keep the UI responsive here: the loaders
    (pykep/pygmo Python callbacks, small-array numpy, tudatpy bindings) hold
    the GIL nearly the whole time, so the main thread is starved. Measured
    while computing Artemis II with no scene cache: a 16 ms main-thread timer
    fired 63 times in 20 s instead of ~1,260, with gaps up to 1.08 s. A
    separate process has its own GIL. SceneLoader still runs on a QThread,
    but that thread only blocks in ``future.result()``, which releases the GIL.

    ``spawn`` (not fork): forking a process that already holds Qt, VTK and an
    OpenGL context is unsafe on macOS. The worker is started lazily on the
    first submit and reused, so heavy imports are paid once per session.
    ``ORBITOPT_INPROCESS_COMPUTE=1`` disables it (debugging, profilers).
    """

    def __init__(self):
        self._executor = None

    @staticmethod
    def enabled() -> bool:
        return os.environ.get("ORBITOPT_INPROCESS_COMPUTE", "") != "1"

    def run(self, fn):
        """Call ``fn()`` in the worker process and return its result (blocking)."""
        from concurrent.futures.process import BrokenProcessPool

        if self._executor is None:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            from orbitopt._compute_worker import watch_parent

            self._executor = ProcessPoolExecutor(
                max_workers=1, mp_context=mp.get_context("spawn"),
                initializer=watch_parent, initargs=(os.getpid(),),
            )
        try:
            return self._executor.submit(fn).result()
        except BrokenProcessPool as exc:
            # The worker died (crash in native code, or terminate() on quit).
            # Drop the broken executor so the next load starts a fresh one.
            self._executor = None
            raise RuntimeError("the compute process exited unexpectedly") from exc

    def terminate(self):
        """Kill the worker immediately (on quit) -- a mission compute can run for
        minutes and must not hold the app open."""
        executor, self._executor = self._executor, None
        if executor is None:
            return
        for proc in list(getattr(executor, "_processes", {}).values()):
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001 -- best-effort teardown
                pass
        executor.shutdown(wait=False, cancel_futures=True)


class SceneLoader(QObject):
    """Runs a scene loader (e.g. compute_and_export_mission, ~10s of pure
    Python/numpy/tudatpy work) off the UI thread: picklable loaders (every
    registered mission) in the ComputePool worker process, anything else
    (e.g. File > Open's lambda, a fast JSON read) directly on this QThread.
    moveToThread + signals is the standard PySide
    pattern for this, not a raw threading.Thread, because Qt needs the
    result handed back via a queued signal/slot connection to land safely
    on the main thread rather than touching any Qt/VTK objects directly
    from the worker thread (which isn't thread-safe). The loader itself
    must stay pure-Python -- it must never touch self.plotter/self.renderer,
    both of which live on the main thread.
    """

    # (name, scene) rather than just (scene): the name has to travel with the
    # signal itself, not via a Python closure. A `lambda scene: cb(name, scene)`
    # connected to a cross-thread signal is invoked *directly on the emitting
    # (worker) thread* -- PySide can only infer "this needs to be queued to
    # the receiver's thread" when the receiver is a bound method of a QObject;
    # a plain lambda has no thread affinity of its own, so AutoConnection
    # silently falls back to a direct call. That let _on_scene_loaded (which
    # touches self.renderer/self.plotter, i.e. VTK's OpenGL context) run on
    # the background thread, fighting the main thread for the same Win32 GL
    # context -- observed as `wglMakeCurrent failed ... resource in use` and
    # the whole app hanging. Emitting (name, scene) and connecting straight to
    # the bound method (no lambda) lets Qt detect the correct thread and queue
    # the call properly; the connections below also pass QueuedConnection
    # explicitly so this doesn't regress silently if a lambda creeps back in.
    finished = Signal(str, object)
    failed = Signal(str, str)

    def __init__(self, name, loader, pool: ComputePool | None = None):
        super().__init__()
        self._name = name
        self._loader = loader
        self._pool = pool

    def run(self):
        try:
            if self._pool is not None and ComputePool.enabled() and _is_picklable(self._loader):
                scene = self._pool.run(self._loader)
            else:
                scene = self._loader()
        except Exception as exc:  # noqa: BLE001 -- reported to the user via the failed signal
            self.failed.emit(self._name, str(exc))
            return
        self.finished.emit(self._name, scene)


class BackendProbeWorker(QObject):
    """One-shot check of orbitopt.core.gpu.GPU_AVAILABLE, run on a background
    QThread the same way SceneLoader is: importing cupy and querying the CUDA
    runtime is cheap when cupy isn't installed at all (the common case for a
    pip-only ``orbitopt[viewer]`` install) but can take a couple hundred ms
    the first time in a full conda/CUDA environment -- not worth risking a
    startup hitch for, given the same cross-thread-signal machinery already
    exists for scene loading."""

    finished = Signal(bool)

    def run(self):
        try:
            from orbitopt.core.gpu import GPU_AVAILABLE
        except Exception:  # noqa: BLE001 -- absence of the compute stack just means "CPU"
            GPU_AVAILABLE = False
        self.finished.emit(GPU_AVAILABLE)
