"""The versioned exchange envelope every cross-environment stage writes.

One envelope per stage lives at ``<stage-dir>/exchange/exchange.json`` and
implements the E5-03 contract:

* **Schema version** -- ``kind`` ``orbitopt.crossenv.exchange`` plus
  ``schema_version`` ``1.0``, validated against
  ``orbitopt/schemas/crossenv-exchange-1.0.json`` on every read; a foreign
  kind or an unknown version fails closed (``CrossEnvError``), exactly like
  the FlightRun and HandoffState readers.
* **Hashes** -- every artifact carries ``sha256:<hex>``; single files are
  hashed over their bytes, a ``FlightRun`` directory over a deterministic
  sorted listing of its per-file digests. :func:`read_envelope` re-computes
  every digest, so a tampered or half-written artifact is rejected before a
  consumer reads it.
* **Provenance** -- the producing environment is recorded: label,
  interpreter path, Python version, platform and the
  :func:`orbitopt.flightlab.toolchain.toolchain_record` package matrix (so a
  tudatpy-3.10 envelope reads ``tudatpy: <version>, gfold: absent`` and a
  landing-3.12 envelope reads the reverse).

Payloads are the mission's own versioned formats, never a crossenv-private
serialization: a ``flight-run`` payload points at a ``FlightRun`` directory
readable by :func:`orbitopt.flightlab.contract.read_run`, the
``*-handoff-state`` payloads at M-01 ``HandoffState`` documents readable by
``orbitopt.mission.contract.state_from_dict`` /
``orbitopt.flightlab.handoff.HandoffState.from_dict``, and a ``report``
payload at a report document whose embedded ``kind``/``schema_version`` the
reader checks against the envelope entry.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

ENVELOPE_KIND = "orbitopt.crossenv.exchange"
ENVELOPE_SCHEMA_VERSION = "1.0"
EXCHANGE_FILE = "exchange.json"
SCHEMA_FILE = "crossenv-exchange-1.0.json"

#: Payload types the reader knows how to load and verify (E5-03 v1).
PAYLOAD_TYPES = ("flight-run", "mission-handoff-state", "flightlab-handoff-state", "report")

DIGEST_PREFIX = "sha256:"
_RUN_DIR_DOMAIN = b"orbitopt.crossenv.run-dir.v1\n"


class CrossEnvError(ValueError):
    """A violated cross-environment invariant: wrong schema, bad digest,
    missing artifact or unparseable payload. Subclasses ``ValueError`` so
    callers already guarding the contract readers catch one exception family."""


def _schema() -> dict:
    try:
        text = resources.files("orbitopt").joinpath("schemas", SCHEMA_FILE).read_text(
            encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise CrossEnvError(f"no packaged schema orbitopt/schemas/{SCHEMA_FILE}") from exc
    return json.loads(text)


def _validate_schema(doc: Mapping[str, Any]) -> None:
    import jsonschema

    try:
        jsonschema.validate(instance=dict(doc), schema=_schema())
    except jsonschema.ValidationError as exc:
        raise CrossEnvError(f"exchange envelope violates {SCHEMA_FILE}: {exc.message}") from exc


def file_digest(path) -> str:
    """``sha256:<hex>`` over the file's bytes (streamed; run CSVs are large)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return DIGEST_PREFIX + h.hexdigest()


def run_dir_digest(dir) -> str:
    """``sha256:<hex>`` over a directory's whole content: the digest of the
    sorted ``<posix-relative-path>\\t<file-digest>`` listing, domain-prefixed
    so a directory digest can never collide with a file digest. Two
    directories digest identically iff every file matches byte for byte and
    no file is added or removed -- the exact property a consumed ``FlightRun``
    needs before a later stage trusts it."""
    root = Path(dir)
    if not root.is_dir():
        raise CrossEnvError(f"run-dir payload {str(root)!r} is not a directory")
    lines = []
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        if path.is_file():
            lines.append(f"{path.relative_to(root).as_posix()}\t{file_digest(path)}\n")
    h = hashlib.sha256(_RUN_DIR_DOMAIN)
    for line in lines:
        h.update(line.encode("utf-8"))
    return DIGEST_PREFIX + h.hexdigest()


def payload_digest(path, digest_kind: str) -> str:
    """Digest with the kind recorded in the envelope: ``file`` or ``run-dir``."""
    if digest_kind == "file":
        return file_digest(path)
    if digest_kind == "run-dir":
        return run_dir_digest(path)
    raise CrossEnvError(f"unknown digest kind {digest_kind!r}")


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def environment_record(env_label: str) -> dict:
    """Provenance of the environment executing right now: the interpreter
    that will be stamped into an envelope written here. ``toolchain`` comes
    from :func:`orbitopt.flightlab.toolchain.toolchain_record`, so the two
    environments differ exactly where they must (tudatpy present in one,
    gfold in the other)."""
    from orbitopt.flightlab.toolchain import toolchain_record

    return {
        "env": str(env_label),
        "python_executable": str(sys.executable),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "toolchain": dict(toolchain_record()),
        "argv": [str(a) for a in sys.argv],
    }


def write_envelope(exchange_dir, *, stage: str, runner: str, env_label: str,
                   payloads: list[dict], created_utc: str | None = None) -> Path:
    """Seal ``exchange_dir`` (``<stage-dir>/exchange``) with the manifest
    listing ``payloads`` and return the manifest path.

    Each payload entry: ``name`` (unique), ``type`` (:data:`PAYLOAD_TYPES`),
    ``schema_version``, ``path`` (POSIX path *inside* the exchange dir),
    ``digest_kind`` (``file``/``run-dir``) and, for ``report`` payloads,
    ``report_kind``. Digests are computed here, over the bytes as written --
    sealing after the artifacts exist is what makes the read-side re-hash
    meaningful.
    """
    ex = Path(exchange_dir)
    seen: set[str] = set()
    entries = []
    for p in payloads:
        name, ptype = str(p["name"]), str(p["type"])
        if not name or name in seen:
            raise CrossEnvError(f"payload name {name!r} empty or duplicated in envelope")
        seen.add(name)
        if ptype not in PAYLOAD_TYPES:
            raise CrossEnvError(f"payload {name!r}: unknown type {ptype!r} "
                                f"(known: {', '.join(PAYLOAD_TYPES)})")
        rel = Path(str(p["path"]))
        if rel.is_absolute() or ".." in rel.parts:
            raise CrossEnvError(f"payload {name!r}: path must stay inside the exchange dir, "
                                f"got {str(p['path'])!r}")
        abs_path = ex / rel
        if not abs_path.exists():
            raise CrossEnvError(f"payload {name!r}: artifact {str(abs_path)!r} does not exist "
                                "-- seal only artifacts already written")
        entry = {
            "name": name,
            "type": ptype,
            "schema_version": str(p["schema_version"]),
            "path": rel.as_posix(),
            "digest_kind": str(p["digest_kind"]),
            "digest": payload_digest(abs_path, str(p["digest_kind"])),
        }
        if "report_kind" in p:
            entry["report_kind"] = str(p["report_kind"])
        entries.append(entry)
    if not entries:
        raise CrossEnvError("an exchange envelope with no payloads carries nothing across the "
                            "environment boundary -- refusing to seal an empty envelope")

    doc = {
        "kind": ENVELOPE_KIND,
        "schema_version": ENVELOPE_SCHEMA_VERSION,
        "stage": str(stage),
        "runner": str(runner),
        "created_utc": created_utc or _now_utc(),
        "environment": environment_record(env_label),
        "payloads": entries,
    }
    _validate_schema(doc)
    ex.mkdir(parents=True, exist_ok=True)
    (ex / EXCHANGE_FILE).write_text(
        json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return ex / EXCHANGE_FILE


def read_envelope(exchange_dir, *, validate: bool = True) -> dict:
    """Read ``exchange.json`` and (unless ``validate=False`` -- advisory only)
    enforce the whole read contract: packaged JSON Schema, exact kind and
    schema version, and a re-computed digest for every artifact. Raises
    :class:`CrossEnvError` on any mismatch; returns the manifest dict."""
    ex = Path(exchange_dir)
    manifest_path = ex / EXCHANGE_FILE
    if not manifest_path.is_file():
        raise CrossEnvError(f"no {EXCHANGE_FILE} in exchange dir {str(ex)!r}")
    try:
        doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CrossEnvError(f"{str(manifest_path)!r} is not valid JSON: {exc}") from exc
    if doc.get("kind") != ENVELOPE_KIND:
        raise CrossEnvError(f"not a {ENVELOPE_KIND!r} envelope (kind: {doc.get('kind')!r})")
    version = doc.get("schema_version")
    if version != ENVELOPE_SCHEMA_VERSION:
        raise CrossEnvError(
            f"no exchange schema for version {version!r} (this orbitopt implements "
            f"{ENVELOPE_SCHEMA_VERSION!r})")
    if not validate:
        return doc
    _validate_schema(doc)
    for entry in doc["payloads"]:
        rel = Path(entry["path"])
        abs_path = ex / rel
        if not abs_path.exists():
            raise CrossEnvError(f"payload {entry['name']!r}: artifact {str(abs_path)!r} is "
                                "missing from the exchange")
        got = payload_digest(abs_path, entry["digest_kind"])
        if got != entry["digest"]:
            raise CrossEnvError(
                f"payload {entry['name']!r} digest mismatch: envelope records {entry['digest']},"
                f" artifact hashes {got} -- the artifact changed after sealing (corruption or "
                "tampering); refusing to hand it across the environment boundary")
    return doc


def _load_payload(ex: Path, entry: Mapping[str, Any]) -> Any:
    """Dispatch one verified payload entry to its native versioned reader."""
    abs_path = ex / Path(entry["path"])
    ptype = entry["type"]
    if ptype == "flight-run":
        from orbitopt.flightlab.contract import read_run

        return read_run(abs_path)
    doc = json.loads(abs_path.read_text(encoding="utf-8"))
    if ptype == "mission-handoff-state":
        from orbitopt.mission.contract import state_from_dict

        return state_from_dict(doc)
    if ptype == "flightlab-handoff-state":
        from orbitopt.flightlab.handoff import HandoffState

        return HandoffState.from_dict(doc)
    if ptype == "report":
        if doc.get("kind") != entry.get("report_kind"):
            raise CrossEnvError(f"report payload {entry['name']!r}: document kind "
                                f"{doc.get('kind')!r} is not the sealed {entry['report_kind']!r}")
        if str(doc.get("schema_version")) != entry["schema_version"]:
            raise CrossEnvError(
                f"report payload {entry['name']!r}: document schema_version "
                f"{doc.get('schema_version')!r} != sealed {entry['schema_version']!r}")
        return doc
    raise CrossEnvError(f"payload {entry['name']!r}: unknown type {ptype!r}")


def read_payloads(exchange_dir) -> dict[str, Any]:
    """Every payload of a validated exchange, by name, loaded through its
    native reader (``FlightRun`` objects, ``HandoffState`` objects, report
    documents). This is what a consuming stage calls: validation first,
    native typed objects second -- never a half-read artifact."""
    doc = read_envelope(exchange_dir)
    ex = Path(exchange_dir)
    return {entry["name"]: _load_payload(ex, entry) for entry in doc["payloads"]}


def envelope_digest(exchange_dir) -> str:
    """``file_digest`` of the envelope document itself -- what the mission
    report records to pin exactly which sealed exchange it graded."""
    return file_digest(Path(exchange_dir) / EXCHANGE_FILE)


def _rel_posix(ex: Path, path: Path) -> str:
    return os.path.relpath(path, ex).replace(os.sep, "/")
