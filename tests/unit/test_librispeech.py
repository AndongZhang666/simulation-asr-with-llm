from pathlib import Path

import pytest

from asr_ec.data.librispeech import prepare_librispeech, validate_expected_count
from asr_ec.data.manifests import (
    ManifestValidationError,
    validate_no_split_overlap,
    write_manifest_once,
)
from asr_ec.data.normalization import ConservativeEnglishNormalizer


def create_librispeech_fixture(root: Path, *, split: str = "test-clean") -> Path:
    chapter = root / split / "1" / "2"
    chapter.mkdir(parents=True)
    (chapter / "1-2-0000.flac").write_bytes(b"fLaCfixture-one")
    (chapter / "1-2-0001.flac").write_bytes(b"fLaCfixture-two")
    (chapter / "1-2.trans.txt").write_text(
        "1-2-0000 HELLO, WORLD!\n1-2-0001 I'VE GOT 2 APPLES\n", encoding="utf-8"
    )
    return root


def test_prepare_librispeech_normalizes_and_sorts_records(tmp_path: Path) -> None:
    root = create_librispeech_fixture(tmp_path / "LibriSpeech")

    records = prepare_librispeech(
        root, split="test-clean", normalizer=ConservativeEnglishNormalizer()
    )

    assert [record.utt_id for record in records] == ["1-2-0000", "1-2-0001"]
    assert records[0].reference_raw == "HELLO, WORLD!"
    assert records[0].reference_normalized == "hello world"
    assert records[0].audio_path == "test-clean/1/2/1-2-0000.flac"


def test_manifest_write_is_content_addressed_and_idempotent(tmp_path: Path) -> None:
    root = create_librispeech_fixture(tmp_path / "LibriSpeech")
    records = prepare_librispeech(
        root, split="test-clean", normalizer=ConservativeEnglishNormalizer()
    )

    first = write_manifest_once(records, output_root=tmp_path / "manifests")
    second = write_manifest_once(records, output_root=tmp_path / "manifests")

    assert first == second
    assert first.path.read_text(encoding="utf-8").count("\n") == 2


def test_prepare_librispeech_rejects_missing_or_invalid_audio(tmp_path: Path) -> None:
    root = create_librispeech_fixture(tmp_path / "LibriSpeech")
    missing_audio = root / "test-clean" / "1" / "2" / "1-2-0001.flac"
    missing_audio.unlink()

    with pytest.raises(ManifestValidationError, match="missing LibriSpeech audio"):
        prepare_librispeech(root, split="test-clean", normalizer=ConservativeEnglishNormalizer())

    missing_audio.write_bytes(b"invalid")
    with pytest.raises(ManifestValidationError, match="invalid FLAC header"):
        prepare_librispeech(root, split="test-clean", normalizer=ConservativeEnglishNormalizer())


def test_manifest_validation_detects_duplicate_ids_and_cross_split_overlap(tmp_path: Path) -> None:
    root = create_librispeech_fixture(tmp_path / "LibriSpeech")
    chapter = root / "dev-clean" / "1" / "2"
    chapter.mkdir(parents=True)
    (chapter / "1-2-0000.flac").write_bytes(b"fLaCfixture")
    (chapter / "1-2.trans.txt").write_text("1-2-0000 NEW REFERENCE\n", encoding="utf-8")
    normalizer = ConservativeEnglishNormalizer()
    test_records = prepare_librispeech(root, split="test-clean", normalizer=normalizer)
    dev_records = prepare_librispeech(root, split="dev-clean", normalizer=normalizer)

    with pytest.raises(ManifestValidationError, match="overlaps"):
        validate_no_split_overlap((test_records, dev_records))
    with pytest.raises(ManifestValidationError, match="count mismatch"):
        validate_expected_count(test_records, expected_count=3)
