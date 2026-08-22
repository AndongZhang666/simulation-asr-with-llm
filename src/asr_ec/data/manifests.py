"""Immutable, inspectable dataset manifests built before ASR decoding."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .normalization import TextNormalizer


class ManifestValidationError(ValueError):
    """Raised when input data cannot produce a trustworthy deterministic manifest."""


@dataclass(frozen=True, slots=True)
class DataManifestRecord:
    """Audio/reference metadata independent of an ASR backend."""

    utt_id: str
    dataset: str
    split: str
    audio_path: str
    audio_sha256: str
    reference_raw: str
    reference_normalized: str

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.utt_id, "utt_id"),
            (self.dataset, "dataset"),
            (self.split, "split"),
            (self.audio_path, "audio_path"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ManifestValidationError(f"{field_name} must be a non-empty string")
        if len(self.audio_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.audio_sha256
        ):
            raise ManifestValidationError("audio_sha256 must be a lowercase SHA-256 digest")
        if not self.reference_normalized:
            raise ManifestValidationError("reference_normalized must not be empty")

    def to_dict(self) -> dict[str, str]:
        return {
            "utt_id": self.utt_id,
            "dataset": self.dataset,
            "split": self.split,
            "audio_path": self.audio_path,
            "audio_sha256": self.audio_sha256,
            "reference_raw": self.reference_raw,
            "reference_normalized": self.reference_normalized,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DataManifestRecord:
        return cls(
            utt_id=data["utt_id"],
            dataset=data["dataset"],
            split=data["split"],
            audio_path=data["audio_path"],
            audio_sha256=data["audio_sha256"],
            reference_raw=data["reference_raw"],
            reference_normalized=data["reference_normalized"],
        )


@dataclass(frozen=True, slots=True)
class ManifestArtifact:
    """Content-addressed result of one deterministic manifest build."""

    sha256: str
    path: Path
    record_count: int


def file_sha256(path: Path) -> str:
    """Hash an audio file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest_records(records: Sequence[DataManifestRecord]) -> None:
    """Reject duplicate IDs or paths before an immutable manifest is published."""

    utt_ids = [record.utt_id for record in records]
    if len(utt_ids) != len(set(utt_ids)):
        raise ManifestValidationError("manifest contains duplicate utterance IDs")
    audio_paths = [record.audio_path for record in records]
    if len(audio_paths) != len(set(audio_paths)):
        raise ManifestValidationError("manifest contains duplicate audio paths")


def validate_no_split_overlap(manifests: Iterable[Sequence[DataManifestRecord]]) -> None:
    """Reject utterance IDs appearing in more than one requested split."""

    seen: set[str] = set()
    for manifest in manifests:
        for record in manifest:
            if record.utt_id in seen:
                raise ManifestValidationError("utterance ID overlaps between requested splits")
            seen.add(record.utt_id)


def canonical_manifest_bytes(records: Sequence[DataManifestRecord]) -> bytes:
    """Serialize sorted records to stable JSONL bytes for a content-addressed path."""

    validate_manifest_records(records)
    lines = [
        json.dumps(record.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        for record in sorted(records, key=lambda item: item.utt_id)
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def write_manifest_once(
    records: Sequence[DataManifestRecord], *, output_root: Path
) -> ManifestArtifact:
    """Write a split manifest below its content hash, reusing identical bytes safely."""

    if not records:
        raise ManifestValidationError("refusing to publish an empty manifest")
    split = records[0].split
    dataset = records[0].dataset
    if any(record.split != split or record.dataset != dataset for record in records):
        raise ManifestValidationError("a manifest artifact must contain one dataset and split")
    payload = canonical_manifest_bytes(records)
    digest = hashlib.sha256(payload).hexdigest()
    destination = output_root / dataset / split / digest / "records.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != payload:
            raise ManifestValidationError("existing manifest path has unexpected content")
    else:
        destination.write_bytes(payload)
    return ManifestArtifact(sha256=digest, path=destination, record_count=len(records))


def normalize_reference(raw_reference: str, normalizer: TextNormalizer) -> str:
    normalized_reference = normalizer.normalize(raw_reference)
    if not normalized_reference:
        raise ManifestValidationError("reference is empty after named normalization")
    return normalized_reference
