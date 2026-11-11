"""Fetch the pinned OpenTsiolkovsky release binary into the orbitopt cache.

OpenTsiolkovsky (https://github.com/istellartech/OpenTsiolkovsky, Rust, MIT)
is an EXTERNAL tool: its binary is deliberately NOT vendored into this (GPL)
tree -- see NOTICE. This script downloads one pinned release asset for the
current platform over HTTPS, verifies its SHA-256 against the table below,
extracts the single executable and installs it (mode 0755) into
``~/.cache/orbitopt/bin/`` -- the location ``orbitopt.flightlab.adapters.
opentsiolkovsky`` looks in. The env var ``ORBITOPT_OPENTSIOLKOVSKY``
overrides the binary path for both this script and the adapter.

    python scripts/fetch_opentsiolkovsky.py            # install for this host
    python scripts/fetch_opentsiolkovsky.py --check     # verify what is installed
    python scripts/fetch_opentsiolkovsky.py --force     # re-download over an existing install

Re-pinning a new OT release: download the new assets, update VERSION, the
asset names and their SHA-256 digests here, regenerate the adapter's cached
config schema (``--write-schema`` below), and re-record the test fixtures.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import platform
import stat
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

VERSION = "1.0.0"
BASE_URL = "https://github.com/istellartech/OpenTsiolkovsky/releases/download"

#: Pinned assets: platform tag -> (archive filename, sha256 hex digest).
#: Digests from the v1.0.0 release (checked 2026-11-11); the fetch REFUSES
#: to install anything whose hash differs, so a tampered release page cannot
#: put an unverified binary into the cache.
ASSETS = {
    "linux-x86_64": (
        "openTsiolkovsky-v1.0.0-linux-x86_64.tar.gz",
        "e39da128a084a8782e0159f26e777048dac5c49b9855d50030ba5a2e5dca7e06",
    ),
    "macos-arm64": (
        "openTsiolkovsky-v1.0.0-macos-arm64.tar.gz",
        "8b08ac464dd435bf0e814b03d9cd3cfcb88133851e5d150e819de63e419233f4",
    ),
    "windows-x86_64": (
        "openTsiolkovsky-v1.0.0-windows-x86_64.zip",
        "fe29fe6d09275a77c9e29427416e1956b712981d9793ac6f594cef2087063aa4",
    ),
}

BINARY_NAME = "openTsiolkovsky-cli"


def binary_name() -> str:
    return f"{BINARY_NAME}.exe" if os.name == "nt" else BINARY_NAME


def detect_platform() -> str:
    """Map the running host onto a pinned asset tag, or raise. Only the
    three platforms OT ships binaries for are supported; anything else is a
    loud error rather than a silently wrong (or absent) download."""
    system = {"Linux": "linux", "Darwin": "macos", "Windows": "windows"}.get(platform.system())
    machine = {"x86_64": "x86_64", "AMD64": "x86_64", "aarch64": "arm64", "arm64": "arm64"}.get(
        platform.machine()
    )
    tag = f"{system}-{machine}" if system and machine else None
    if tag not in ASSETS:
        raise SystemExit(
            f"no pinned OpenTsiolkovsky v{VERSION} asset for {platform.system()}/"
            f"{platform.machine()} (pinned: {', '.join(sorted(ASSETS))}); build it from "
            "source (cargo build --release) and point ORBITOPT_OPENTSIOLKOVSKY at it"
        )
    return tag


def binary_path() -> Path:
    """Where the adapter looks for the binary: the ORBITOPT_OPENTSIOLKOVSKY
    override if set, else ~/.cache/orbitopt/bin/openTsiolkovsky-cli."""
    override = os.environ.get("ORBITOPT_OPENTSIOLKOVSKY", "").strip()
    if override:
        return Path(override)
    return Path.home() / ".cache" / "orbitopt" / "bin" / binary_name()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> None:
    print(f"downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": f"orbitopt-fetch-{BINARY_NAME}/{VERSION}"})
    with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as out:
        total = resp.headers.get("Content-Length")
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done}/{total} bytes", end="", flush=True)
        print()


def _extract_binary(archive: Path, tag: str) -> Path:
    """Pull the single executable out of the verified archive into a temp
    dir. Members are rejected unless they are plain files whose name is the
    binary -- no archive-entry path traversal, no extra files installed."""
    dest_dir = Path(tempfile.mkdtemp(prefix="otsk-"))
    if archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as tf:
            members = [m for m in tf.getmembers() if m.isfile() and Path(m.name).name == binary_name()]
            if len(members) != 1:
                raise SystemExit(f"{archive.name}: expected exactly one {binary_name()!r} member, found {len(members)}")
            member = members[0]
            with tf.extractfile(member) as src, (dest_dir / binary_name()).open("wb") as dst:
                dst.write(src.read())
    else:
        with zipfile.ZipFile(archive) as zf:
            names = [n for n in zf.namelist() if Path(n).name == binary_name() and not n.endswith("/")]
            if len(names) != 1:
                raise SystemExit(f"{archive.name}: expected exactly one {binary_name()!r} member, found {len(names)}")
            with zf.open(names[0]) as src, (dest_dir / binary_name()).open("wb") as dst:
                dst.write(src.read())
    out = dest_dir / binary_name()
    out.chmod(out.stat().st_mode | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP)
    return out


def install(*, force: bool = False) -> Path:
    tag = detect_platform()
    asset, digest = ASSETS[tag]
    target = binary_path()
    if target.is_file() and not force:
        print(f"already installed: {target} (use --force to re-download)")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="otsk-fetch-") as tmp:
        archive = Path(tmp) / asset
        _download(f"{BASE_URL}/v{VERSION}/{asset}", archive)
        got = _sha256(archive)
        if got != digest:
            raise SystemExit(
                f"SHA-256 mismatch for {asset}:\n  expected {digest}\n  got      {got}\n"
                "refusing to install; the pinned digest may be stale (release re-tagged?)"
            )
        extracted = _extract_binary(archive, tag)
        # stage through a sibling name so a concurrent process never half-sees it
        staged = target.with_name(target.name + ".new")
        staged.write_bytes(extracted.read_bytes())
        staged.chmod(0o755)
        os.replace(staged, target)
    print(f"installed OpenTsiolkovsky v{VERSION} ({tag}) -> {target}")
    return target


def check() -> int:
    path = binary_path()
    if not path.is_file():
        print(f"not installed: {path}\nrun:  python scripts/fetch_opentsiolkovsky.py")
        return 1
    import subprocess

    proc = subprocess.run([str(path), "--version"], capture_output=True, text=True, timeout=15)
    out = (proc.stdout or proc.stderr).strip()
    expected = f"OpenTsiolkovsky {VERSION}"
    if proc.returncode != 0 or VERSION not in out:
        print(f"{path}: version probe failed or reports {out!r} (pinned: {expected})")
        print("re-run with --force to reinstall the pinned release")
        return 1
    print(f"{path}: {out} (pinned v{VERSION}) OK")
    return 0


def write_schema(out_dir: Path) -> int:
    """Ask the installed binary for its config JSON Schema and write it next
    to the adapter's data files, so CI validates generated configs without
    the binary. The adapter loads the packaged copy; this regenerates it."""
    path = binary_path()
    if not path.is_file():
        install()
    import subprocess

    proc = subprocess.run([str(path), "schema"], capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        print(f"'{path}' schema failed: {proc.stderr.strip()}", file=sys.stderr)
        return 1
    import json

    schema = json.loads(proc.stdout)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"opentsiolkovsky-config-v{VERSION}.json"
    dest.write_text(json.dumps(schema, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {dest}")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify the installed binary (version probe)")
    parser.add_argument("--force", action="store_true", help="re-download even if already installed")
    parser.add_argument(
        "--write-schema",
        action="store_true",
        help="regenerate the adapter's cached config schema from the binary (maintainers only)",
    )
    args = parser.parse_args(argv)
    if args.write_schema:
        repo_schema_dir = Path(__file__).resolve().parents[1] / "src" / "orbitopt" / "data" / "flightlab" / "schemas"
        return write_schema(repo_schema_dir)
    if args.check:
        return check()
    install(force=args.force)
    return check()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
