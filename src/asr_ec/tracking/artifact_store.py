"""Content-addressed storage for immutable binary or JSONL research artifacts."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path


class ArtifactStoreError(RuntimeError):
    """Raised when an immutable artifact cannot be written or verified."""


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Stable identity and location of bytes in a content-addressed store."""

    sha256: str
    byte_count: int
    path: Path


class ArtifactStore:
    """Store bytes once under their SHA-256 digest without mutable overwrite paths."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def put_bytes(self, payload: bytes, *, suffix: str = "") -> ArtifactRef:
        """Persist bytes atomically and return the existing object for repeated content."""

        if not isinstance(payload, bytes):
            raise ArtifactStoreError("payload must be bytes")
        if suffix and not re.fullmatch(r"\.[a-z0-9]+", suffix):
            raise ArtifactStoreError("suffix must be empty or a simple lowercase extension")

        digest = hashlib.sha256(payload).hexdigest()
        artifact_dir = self._root / digest[:2]
        destination = artifact_dir / f"{digest}{suffix}"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        if destination.exists():
            self._verify_existing(destination, payload, digest)
            return ArtifactRef(digest, len(payload), destination)

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=artifact_dir, delete=False) as temporary_file:
                temporary_file.write(payload)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
                temporary_path = Path(temporary_file.name)
            try:
                os.link(temporary_path, destination)
            except FileExistsError:
                self._verify_existing(destination, payload, digest)
            finally:
                temporary_path.unlink(missing_ok=True)
        except OSError as error:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise ArtifactStoreError(f"could not persist artifact {digest}") from error

        return ArtifactRef(digest, len(payload), destination)

    @staticmethod
    def _verify_existing(destination: Path, expected_payload: bytes, expected_digest: str) -> None:
        existing_payload = destination.read_bytes()
        if existing_payload != expected_payload:
            raise ArtifactStoreError(
                f"content-addressed path collision or mutation detected for {expected_digest}"
            )
