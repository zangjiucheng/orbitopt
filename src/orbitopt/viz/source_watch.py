"""Notices edits to orbitopt's own source while Mission Control runs.

Missions are computed in a worker process that imported the code once
(viz.compute_pool), so an edit only takes effect in a fresh worker. The app
polls :meth:`SourceWatch.changed` and, when the code on disk differs from
the code the worker runs, offers a reload (restart the worker, recompute).
Polling file stats, not a QFileSystemWatcher: editors save by replacing the
file, which drops the watch on several platforms. Qt-free, so it is tested
headless.
"""
from __future__ import annotations

from pathlib import Path


def package_root() -> Path:
    import orbitopt

    return Path(orbitopt.__file__).resolve().parent


def source_signature(root: Path | None = None) -> dict[str, tuple[int, int]]:
    """``{relative path: (mtime_ns, size)}`` for every ``.py`` under ``root``
    (the orbitopt package by default). About a millisecond for this package."""
    root = root if root is not None else package_root()
    out = {}
    for path in root.rglob("*.py"):
        try:
            st = path.stat()
        except OSError:  # deleted between the listing and the stat
            continue
        out[path.relative_to(root).as_posix()] = (st.st_mtime_ns, st.st_size)
    return out


class SourceWatch:
    def __init__(self, root: Path | None = None):
        self.root = root if root is not None else package_root()
        self._baseline = source_signature(self.root)

    def changed_files(self) -> list[str]:
        """Source files added, removed or modified since the last accept()."""
        now = source_signature(self.root)
        return sorted(p for p in self._baseline.keys() | now.keys() if self._baseline.get(p) != now.get(p))

    def accept(self) -> None:
        """The current code is what will run from now on (worker restarted)."""
        self._baseline = source_signature(self.root)
