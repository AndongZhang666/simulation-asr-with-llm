"""Minimal immutable manifest created before expensive pipeline stages."""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from .config_hash import canonical_json, config_sha256, run_id


class RunManifestError(RuntimeError):
    """Raised when a run directory or manifest would overwrite prior evidence."""


@dataclass(frozen=True, slots=True)
class RunManifest:
    run_id: str
    config_sha256: str
    command: tuple[str, ...]
    created_utc: str
    python_version: str
    platform: str
    git_commit: str | None
    git_dirty: bool | None
    dependency_lock_sha256: str | None
    input_artifact_hashes: Mapping[str, str] = field(default_factory=dict)
    output_artifact_hashes: Mapping[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def create_run_manifest(
    runs_root: Path,
    *,
    prefix: str,
    resolved_config: Mapping[str, Any],
    command: tuple[str, ...],
    input_artifact_hashes: Mapping[str, str] | None = None,
    retry_on_collision: bool = False,
) -> tuple[Path, RunManifest]:
    """Create an empty append-only run directory and its resolved provenance files."""

    if not command:
        raise RunManifestError("command must not be empty")
    identifier = run_id(prefix, resolved_config)
    run_directory = runs_root / identifier
    if run_directory.exists() and retry_on_collision:
        attempt = 1
        while run_directory.exists():
            run_directory = runs_root / f"{identifier}-attempt-{attempt:02d}"
            attempt += 1
    try:
        run_directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise RunManifestError(f"run directory already exists: {run_directory}") from error

    repository_root = Path.cwd()
    git_commit, git_dirty = _git_state(repository_root)
    manifest = RunManifest(
        run_id=run_directory.name,
        config_sha256=config_sha256(resolved_config),
        command=command,
        created_utc=datetime.now(timezone.utc).isoformat(),
        python_version=platform.python_version(),
        platform=platform.platform(),
        git_commit=git_commit,
        git_dirty=git_dirty,
        dependency_lock_sha256=_file_sha256_if_exists(repository_root / "uv.lock"),
        input_artifact_hashes=input_artifact_hashes or {},
    )
    canonical_config = json.loads(canonical_json(resolved_config))
    (run_directory / "resolved_config.yaml").write_text(
        yaml.safe_dump(canonical_config, allow_unicode=False, sort_keys=True), encoding="utf-8"
    )
    (run_directory / "run_manifest.json").write_text(manifest.to_json(), encoding="utf-8")
    return run_directory, manifest


def _git_state(repository_root: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, dirty


def _file_sha256_if_exists(path: Path) -> str | None:
    if not path.is_file():
        return None
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as lock_file:
        for chunk in iter(lambda: lock_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
