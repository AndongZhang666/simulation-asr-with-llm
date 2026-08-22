"""Deterministic LibriSpeech transcript parsing and structural audio validation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .manifests import (
    DataManifestRecord,
    ManifestValidationError,
    file_sha256,
    normalize_reference,
    validate_manifest_records,
)
from .normalization import TextNormalizer


def prepare_librispeech(
    source_root: Path,
    *,
    split: str,
    normalizer: TextNormalizer,
) -> tuple[DataManifestRecord, ...]:
    """Read one LibriSpeech split without changing source audio or transcripts."""

    split_directory = source_root / split
    if not split_directory.is_dir():
        raise ManifestValidationError(
            f"LibriSpeech split directory does not exist: {split_directory}"
        )
    transcript_paths = sorted(split_directory.glob("*/*/*.trans.txt"))
    if not transcript_paths:
        raise ManifestValidationError(f"no LibriSpeech transcript files found for split: {split}")

    records: list[DataManifestRecord] = []
    for transcript_path in transcript_paths:
        for line_number, line in enumerate(
            transcript_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            utt_id, raw_reference = _parse_transcript_line(line, transcript_path, line_number)
            audio_path = transcript_path.parent / f"{utt_id}.flac"
            _validate_flac(audio_path)
            records.append(
                DataManifestRecord(
                    utt_id=utt_id,
                    dataset="librispeech",
                    split=split,
                    audio_path=audio_path.relative_to(source_root).as_posix(),
                    audio_sha256=file_sha256(audio_path),
                    reference_raw=raw_reference,
                    reference_normalized=normalize_reference(raw_reference, normalizer),
                )
            )
    validate_manifest_records(records)
    return tuple(sorted(records, key=lambda item: item.utt_id))


def validate_expected_count(
    records: Sequence[DataManifestRecord], expected_count: int | None
) -> None:
    """Fail explicitly when a requested public split differs from its configured count."""

    if expected_count is not None and len(records) != expected_count:
        raise ManifestValidationError(
            f"manifest count mismatch: expected {expected_count}, found {len(records)}"
        )


def _parse_transcript_line(line: str, transcript_path: Path, line_number: int) -> tuple[str, str]:
    parts = line.strip().split(maxsplit=1)
    if len(parts) != 2:
        raise ManifestValidationError(
            f"invalid transcript line {line_number} in {transcript_path}: expected ID and reference"
        )
    return parts[0], parts[1]


def _validate_flac(audio_path: Path) -> None:
    if not audio_path.is_file():
        raise ManifestValidationError(f"missing LibriSpeech audio file: {audio_path}")
    if audio_path.stat().st_size < 4:
        raise ManifestValidationError(f"invalid or empty FLAC file: {audio_path}")
    with audio_path.open("rb") as audio_file:
        if audio_file.read(4) != b"fLaC":
            raise ManifestValidationError(f"invalid FLAC header: {audio_path}")
