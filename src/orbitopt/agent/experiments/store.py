"""A content-addressed store of JSON artifacts: each blob lives under the SHA-256 of its exact bytes.

``put`` writes to a temporary file in the target directory and renames it
into place, so a reader never sees a partial blob and two writers of the
same bytes converge on one file. ``get`` recomputes the digest of what it
reads and raises ``ArtifactError`` when the bytes changed. The store parses
JSON with ``json.loads`` and nothing else: no pickle, no eval, no import.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from .model import ArtifactError, InvalidIdError

DIGEST = re.compile(r"^sha256:([0-9a-f]{64})$")


def _strict_json(data: bytes) -> Any:
    def reject(token: str) -> Any:
        raise ValueError(f"non-finite number {token}")

    return json.loads(data.decode("utf-8"), parse_constant=reject)


class ArtifactStore:
    """JSON blobs under ``root/sha256/<2 hex>/<64 hex>.json``, addressed by ``sha256:<hex>``."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, digest: str) -> Path:
        match = DIGEST.fullmatch(digest) if isinstance(digest, str) else None
        if match is None:
            raise InvalidIdError(f"not an artifact digest (sha256:<64 hex digits>): {digest!r:.80}")
        hexdigest = match.group(1)
        return self.root / "sha256" / hexdigest[:2] / f"{hexdigest}.json"

    def put(self, data: bytes) -> str:
        """Store ``data`` (UTF-8 JSON, no NaN or infinity) and return its ``sha256:`` digest. Idempotent."""
        try:
            _strict_json(data)
        except (UnicodeDecodeError, ValueError) as exc:
            raise ArtifactError(f"not a JSON artifact: {exc}") from None
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        path = self._path(digest)
        if path.exists() and self._intact(path, digest):
            return digest
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return digest

    def get(self, digest: str) -> bytes:
        """The exact bytes stored under ``digest``; ``ArtifactError`` when missing or changed."""
        path = self._path(digest)
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ArtifactError(f"artifact {digest} is missing: {exc.strerror or exc}") from None
        actual = "sha256:" + hashlib.sha256(data).hexdigest()
        if actual != digest:
            raise ArtifactError(f"artifact {digest} is corrupt: its bytes hash to {actual}")
        return data

    def get_json(self, digest: str) -> Any:
        """``get`` parsed as JSON."""
        try:
            return _strict_json(self.get(digest))
        except (UnicodeDecodeError, ValueError) as exc:  # unreachable for a blob ``put`` wrote
            raise ArtifactError(f"artifact {digest} is not JSON: {exc}") from None

    def has(self, digest: str) -> bool:
        """Whether ``digest`` is stored and intact."""
        path = self._path(digest)
        return path.exists() and self._intact(path, digest)

    @staticmethod
    def _intact(path: Path, digest: str) -> bool:
        try:
            return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() == digest
        except OSError:
            return False
