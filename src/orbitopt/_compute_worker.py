"""Initializer for Mission Control's compute worker process (see
orbitopt.viz.compute_pool.ComputePool). Kept in its own stdlib-only module so
the spawned worker can import it without pulling in Qt or VTK."""
from __future__ import annotations

import os
import threading
import time


def watch_parent(parent_pid: int, interval_s: float = 1.0) -> None:
    """Exit this worker as soon as its parent process is gone.

    ProcessPoolExecutor workers block on their task queue and never notice
    the parent dying. A normal quit terminates them (MissionControlWindow.
    closeEvent), but a crashed or killed app would leave the worker -- and a
    possibly minutes-long mission compute -- running as an orphan.

    Two signals, because neither works everywhere on its own:

    * ``multiprocessing.parent_process().sentinel`` -- a pipe that hits EOF
      on POSIX, a process handle that becomes signalled on Windows, when the
      parent exits. This is the only one that works on Windows, where an
      orphan is never re-parented, so ``os.getppid()`` keeps returning the
      dead (and possibly reused) parent PID forever.
    * ``os.getppid()`` changing (POSIX only) -- the fallback if no sentinel
      is available, e.g. a worker not started by multiprocessing.
    """
    import multiprocessing
    from multiprocessing.connection import wait

    parent = multiprocessing.parent_process()
    sentinel = None
    if parent is not None and parent.pid == parent_pid:
        sentinel = getattr(parent, "sentinel", None)

    def parent_gone() -> bool:
        if sentinel is not None:
            try:
                if wait([sentinel], timeout=interval_s):
                    return True
            except OSError:  # sentinel already closed: the parent is gone
                return True
        else:
            time.sleep(interval_s)
        return os.name != "nt" and os.getppid() != parent_pid

    def run():
        while not parent_gone():
            pass
        os._exit(0)

    threading.Thread(target=run, name="orbitopt-parent-watch", daemon=True).start()
