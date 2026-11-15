"""Exercise draft publication against a real local Git remote."""
from pathlib import Path
import runpy
import subprocess

import pytest

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                           "scripts/publish_translation_draft.py"))
publish = MODULE["publish"]
BRANCH = MODULE["BRANCH"]


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True,
                          text=True, check=True).stdout.strip()


@pytest.fixture
def repository(tmp_path):
    remote = tmp_path / "origin.git"
    root = tmp_path / "checkout"
    git(tmp_path, "init", "--bare", str(remote))
    git(tmp_path, "init", "-b", "main", str(root))
    git(root, "config", "commit.gpgsign", "false")
    git(root, "config", "user.name", "Test")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "remote", "add", "origin", str(remote))
    (root / "translations/zh").mkdir(parents=True)
    (root / "translations/zh/README.md").write_text("Original draft\n")
    (root / "README.md").write_text("English source\n")
    git(root, "add", ".")
    git(root, "commit", "-m", "initial")
    git(root, "push", "origin", "main")
    return root, remote


def generate(root, text="Updated draft\n"):
    (root / "translations/zh/README.md").write_text(text)
    (root / "translations/zh/.sources.json").write_text('{"README.md": "hash"}\n')


def test_publish_scopes_the_commit_and_leaves_remote_main_unchanged(repository):
    root, remote = repository
    main = git(remote, "rev-parse", "refs/heads/main")
    generate(root)
    (root / "README.md").write_text("Unrelated local edit\n")
    assert publish(root) is True
    assert git(remote, "rev-parse", "refs/heads/main") == main
    assert git(remote, "show", f"{BRANCH}:README.md") == "English source"
    assert git(remote, "show", f"{BRANCH}:translations/zh/README.md") == "Updated draft"
    assert git(remote, "show", f"{BRANCH}:translations/zh/.sources.json") == '{"README.md": "hash"}'
    assert (root / "README.md").read_text() == "Unrelated local edit\n"


def test_identical_regeneration_keeps_the_existing_branch_commit(repository):
    root, remote = repository
    generate(root)
    assert publish(root) is True
    draft = git(remote, "rev-parse", f"refs/heads/{BRANCH}")
    git(root, "reset", "--hard", "origin/main")
    generate(root)
    assert publish(root) is True
    assert git(remote, "rev-parse", f"refs/heads/{BRANCH}") == draft
    assert git(root, "rev-parse", "HEAD") == git(remote, "rev-parse", "main")


def test_no_changes_does_not_create_a_draft_branch(repository):
    root, remote = repository
    assert publish(root) is False
    assert git(remote, "for-each-ref", "--format=%(refname)") == "refs/heads/main"


def test_unrelated_staged_files_are_rejected(repository):
    root, remote = repository
    generate(root)
    (root / "README.md").write_text("Staged unrelated edit\n")
    git(root, "add", "README.md")
    with pytest.raises(ValueError, match="outside translations/zh"):
        publish(root)
    assert git(remote, "for-each-ref", "--format=%(refname)") == "refs/heads/main"


def test_concurrent_branch_change_is_not_overwritten(repository, monkeypatch):
    root, remote = repository
    generate(root)
    publish(root)
    git(root, "reset", "--hard", "origin/main")
    generate(root, "New generated revision\n")
    original_git = publish.__globals__["_git"]
    concurrent = git(remote, "rev-parse", "main")

    def race(checkout, *args, **kwargs):
        if args[0] == "push":
            git(remote, "update-ref", f"refs/heads/{BRANCH}", concurrent)
        return original_git(checkout, *args, **kwargs)

    monkeypatch.setitem(publish.__globals__, "_git", race)
    with pytest.raises(subprocess.CalledProcessError):
        publish(root)
    assert git(remote, "rev-parse", f"refs/heads/{BRANCH}") == concurrent
