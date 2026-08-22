"""Public dataset manifest preparation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from asr_ec.data.librispeech import prepare_librispeech, validate_expected_count
from asr_ec.data.manifests import ManifestArtifact, validate_no_split_overlap, write_manifest_once
from asr_ec.data.normalization import (
    ConservativeEnglishNormalizer,
    IdentityNormalizer,
    TextNormalizer,
)
from asr_ec.tracking.run_manifest import create_run_manifest


class DataPreparationError(ValueError):
    """Raised when a data-preparation configuration is incomplete or unsafe."""


@dataclass(frozen=True, slots=True)
class DataPreparationResult:
    dataset: str
    artifacts: tuple[ManifestArtifact, ...]
    dry_run: bool
    run_directory: Path | None

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "dry_run": self.dry_run,
            "run_directory": self.run_directory.as_posix() if self.run_directory else None,
            "artifacts": [
                {
                    "sha256": artifact.sha256,
                    "path": artifact.path.as_posix(),
                    "record_count": artifact.record_count,
                }
                for artifact in self.artifacts
            ],
        }


def load_data_config(config_path: Path) -> Mapping[str, Any]:
    """Load one YAML mapping and reject implicit defaults that change dataset identity."""

    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise DataPreparationError(f"could not read configuration: {config_path}") from error
    if not isinstance(loaded, dict):
        raise DataPreparationError("data configuration must be a YAML mapping")
    return loaded


def run_prepare_data(config_path: Path, *, dry_run: bool) -> DataPreparationResult:
    """Validate and build immutable public-data manifests for the configured dataset."""

    config = load_data_config(config_path)
    if config.get("dataset") != "librispeech":
        raise DataPreparationError("only dataset: librispeech is implemented in this phase")
    source_root = _required_path(config, "source_root")
    output_root = _required_path(config, "output_root")
    splits = config.get("splits")
    if (
        not isinstance(splits, list)
        or not splits
        or not all(isinstance(split, str) for split in splits)
    ):
        raise DataPreparationError("splits must be a non-empty list of strings")
    normalizer = _normalizer_from_id(config.get("normalizer"))
    expected_counts = config.get("expected_counts", {})
    if not isinstance(expected_counts, dict):
        raise DataPreparationError("expected_counts must be a mapping when provided")

    manifests = []
    for split in splits:
        records = prepare_librispeech(source_root, split=split, normalizer=normalizer)
        expected_count = expected_counts.get(split)
        if expected_count is not None and not isinstance(expected_count, int):
            raise DataPreparationError("expected split counts must be integers")
        validate_expected_count(records, expected_count)
        manifests.append(records)
    validate_no_split_overlap(manifests)

    if dry_run:
        return DataPreparationResult(
            dataset="librispeech",
            artifacts=tuple(
                ManifestArtifact(
                    sha256="dry-run", path=Path(records[0].split), record_count=len(records)
                )
                for records in manifests
            ),
            dry_run=True,
            run_directory=None,
        )

    runs_root = _required_path(config, "runs_root")
    run_directory, _ = create_run_manifest(
        runs_root,
        prefix="prepare-data",
        resolved_config=config,
        command=("asr-ec", "prepare-data", "--config", str(config_path)),
    )
    artifacts = tuple(
        write_manifest_once(records, output_root=output_root) for records in manifests
    )
    return DataPreparationResult(
        dataset="librispeech",
        artifacts=artifacts,
        dry_run=False,
        run_directory=run_directory,
    )


def _required_path(config: Mapping[str, Any], key: str) -> Path:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DataPreparationError(f"{key} must be a non-empty path string")
    return Path(value)


def _normalizer_from_id(normalizer_id: object) -> TextNormalizer:
    if normalizer_id == ConservativeEnglishNormalizer.normalizer_id:
        return ConservativeEnglishNormalizer()
    if normalizer_id == IdentityNormalizer.normalizer_id:
        return IdentityNormalizer()
    raise DataPreparationError(f"unsupported normalizer: {normalizer_id}")
