from pathlib import Path

import pytest

from asr_ec.asr.whisper_nbest import NBestCandidate, NBestDecodingResult
from asr_ec.asr.whisper_records import (
    NBestArtifactError,
    build_utterance_record,
    write_nbest_artifact_once,
)
from asr_ec.data.manifests import DataManifestRecord
from asr_ec.data.normalization import IdentityNormalizer


def make_manifest_record() -> DataManifestRecord:
    return DataManifestRecord(
        utt_id="fixture-1",
        dataset="librispeech",
        split="test-clean",
        audio_path="test-clean/fixture.flac",
        audio_sha256="a" * 64,
        reference_raw="Reference",
        reference_normalized="reference",
    )


def make_decoded_result() -> NBestDecodingResult:
    return NBestDecodingResult(
        language="en",
        no_speech_prob=0.01,
        candidates=(
            NBestCandidate(1, "Raw Candidate", (1, 2), -1.0, -0.5),
            NBestCandidate(2, "Raw Candidate", (3,), -1.1, -1.1),
        ),
    )


def test_build_utterance_record_preserves_raw_duplicates_and_score_semantics() -> None:
    record = build_utterance_record(
        make_manifest_record(),
        make_decoded_result(),
        normalizer=IdentityNormalizer(),
        checkpoint="small.en@sha256:fixture",
        code_revision="openai-whisper@fixture",
        decode_config_sha256="b" * 64,
        length_penalty=None,
        software_lock_sha256="c" * 64,
    )

    assert [hypothesis.text.raw for hypothesis in record.asr.hypotheses] == [
        "Raw Candidate",
        "Raw Candidate",
    ]
    assert record.asr.hypotheses[0].sequence_logscore == -1.0
    assert record.asr.hypotheses[0].metadata["ranking_score"] == -0.5
    assert record.asr.hypotheses[0].metadata["token_logprobs_available"] is False


def test_nbest_artifact_is_content_addressed_and_rejects_duplicate_utterances(
    tmp_path: Path,
) -> None:
    record = build_utterance_record(
        make_manifest_record(),
        make_decoded_result(),
        normalizer=IdentityNormalizer(),
        checkpoint="small.en@sha256:fixture",
        code_revision="openai-whisper@fixture",
        decode_config_sha256="b" * 64,
        length_penalty=None,
        software_lock_sha256=None,
    )

    first = write_nbest_artifact_once((record,), output_root=tmp_path)
    second = write_nbest_artifact_once((record,), output_root=tmp_path)

    assert first == second
    with pytest.raises(NBestArtifactError, match="duplicate"):
        write_nbest_artifact_once((record, record), output_root=tmp_path)
