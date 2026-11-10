"""Hot reload's source watch (Qt-free)."""
from __future__ import annotations

import os

from orbitopt.viz.source_watch import SourceWatch, package_root, source_signature


def test_signature_covers_the_package():
    sig = source_signature()
    assert "viz/app.py" in sig and "viz/source_watch.py" in sig
    assert package_root().name == "orbitopt"


def test_watch_reports_edits_additions_and_removals(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("y = 2\n")
    (tmp_path / "notes.txt").write_text("not code")
    watch = SourceWatch(tmp_path)
    assert watch.changed_files() == []

    (tmp_path / "a.py").write_text("x = 10\n")  # size changes
    st = (tmp_path / "sub" / "b.py").stat()
    os.utime(tmp_path / "sub" / "b.py", ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))  # touched only
    (tmp_path / "c.py").write_text("")
    (tmp_path / "notes.txt").write_text("still not code")
    assert watch.changed_files() == ["a.py", "c.py", "sub/b.py"]

    watch.accept()
    assert watch.changed_files() == []
    (tmp_path / "c.py").unlink()
    assert watch.changed_files() == ["c.py"]
