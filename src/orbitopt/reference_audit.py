"""E8-04 tolerance-change audit for the reference regressions.

Reference tolerances are a comparison *policy*: loosening one can hide a real
behaviour change, tightening one can turn unrelated machine noise red. Every
change therefore goes through the owning script's
``update --tolerances-only --reason "..."`` command, which records an audit
entry inside the reference's ``PROVENANCE.json`` under ``tolerance_audit``:
who-changed-what (old -> new per tolerance), why, and the hash of the policy
file before and after. ``docs/reference-tolerance-audit.md`` is the human
ledger; a test cross-checks that the ledger mirrors the committed
``tolerances.json`` files exactly.
"""
from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def flatten_tolerances(tol: dict, prefix: str = "") -> dict[str, float]:
    """Dotted-key view of a (possibly nested) tolerance mapping."""
    flat: dict[str, float] = {}
    for key, value in tol.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_tolerances(value, prefix=dotted + "."))
        else:
            flat[dotted] = value
    return flat


def diff_tolerances(old: dict, new: dict) -> dict[str, dict]:
    """Old -> new for every tolerance key that differs or (dis)appears."""
    changes: dict[str, dict] = {}
    old_flat, new_flat = flatten_tolerances(old), flatten_tolerances(new)
    for key in sorted(set(old_flat) | set(new_flat)):
        before, after = old_flat.get(key), new_flat.get(key)
        if before != after:
            changes[key] = {"old": before, "new": after}
    return changes


def committed_file(repo: Path, relpath: str) -> tuple[dict, str]:
    """The committed JSON file at HEAD: (parsed, sha256-of-committed-bytes).

    The honest "old" value for an audit: working-tree edits are what is being
    audited *against* the record, so the diff must not read them."""
    proc = subprocess.run(["git", "-C", str(repo), "show", f"HEAD:{relpath}"],
                          capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return {}, ""
    import json

    return json.loads(proc.stdout), hashlib.sha256(proc.stdout.encode("utf-8")).hexdigest()


def make_entry(*, reason: str, changes: dict[str, dict],
               previous_tolerances_sha256: str,
               tolerances_sha256: str) -> dict:
    if not reason or not reason.strip():
        raise ValueError("a tolerance change requires a non-empty --reason")
    return {"date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "reason": reason.strip(),
            "changes": changes,
            "previous_tolerances_sha256": previous_tolerances_sha256,
            "tolerances_sha256": tolerances_sha256}


def append_audit(prov: dict, entry: dict) -> dict:
    """Append an audit entry to a provenance mapping (returned, not written)."""
    prov["tolerance_audit"] = list(prov.get("tolerance_audit", [])) + [entry]
    return prov
