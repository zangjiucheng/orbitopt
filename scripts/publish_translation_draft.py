#!/usr/bin/env python3
"""Publish generated Chinese drafts without PR-creation/approval permissions."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

BRANCH = "automation/translate-zh"
OUTPUT = "translations/zh/"


def _git(root: Path, *args: str, check: bool = True):
    return subprocess.run(["git", *args], cwd=root, text=True,
                          capture_output=True, check=check)


def publish(root: Path) -> bool:
    """Publish only generated files; lease-protect the dedicated draft branch.

    Returns whether there are unmerged generated changes to review. The
    workflow supplies checkout credentials; this script does not call the
    pull-request API or change repository permissions.
    """
    staged = _git(root, "diff", "--cached", "--name-only").stdout.splitlines()
    if any(not name.startswith(OUTPUT) for name in staged):
        raise ValueError("refusing to commit staged files outside translations/zh/")
    _git(root, "add", "--", OUTPUT)
    diff = _git(root, "diff", "--cached", "--quiet", check=False)
    if diff.returncode == 0:
        return False
    if diff.returncode != 1:
        diff.check_returncode()

    ref = f"refs/heads/{BRANCH}"
    remote = _git(root, "ls-remote", "--exit-code", "--heads", "origin", ref, check=False)
    if remote.returncode not in (0, 2):
        remote.check_returncode()
    expected = remote.stdout.split()[0] if remote.returncode == 0 else ""
    if expected:
        _git(root, "fetch", "--no-tags", "origin", ref)
        same = _git(root, "diff", "--cached", "--quiet", "FETCH_HEAD", "--", OUTPUT, check=False)
        if same.returncode == 0:
            return True
        if same.returncode != 1:
            same.check_returncode()
    _git(root, "-c", "user.name=github-actions[bot]",
         "-c", "user.email=41898282+github-actions[bot]@users.noreply.github.com",
         "commit", "-m", "docs: update Simplified Chinese translation drafts")
    _git(root, "push", f"--force-with-lease={ref}:{expected}", "origin", f"HEAD:{ref}")
    return True


def main():
    changed = publish(Path(__file__).resolve().parents[1])
    print("Translation draft branch ready for review" if changed else "No translation changes")
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a") as stream:
            stream.write(f"has_changes={str(changed).lower()}\n")


if __name__ == "__main__":
    main()
