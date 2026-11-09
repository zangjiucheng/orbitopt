"""Initializer for Mission Control's compute worker process (see
orbitopt.viz.compute.ComputePool). Kept in its own stdlib-only module so the
spawned worker can import it without pulling in Qt or VTK."""
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
    """

    def run():
        while True:
            time.sleep(interval_s)
            if os.getppid() != parent_pid:
                os._exit(0)

    threading.Thread(target=run, name="orbitopt-parent-watch", daemon=True).start()
