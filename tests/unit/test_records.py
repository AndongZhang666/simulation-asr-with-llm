import math

import pytest

from asr_ec.contracts.records import (
    ASRMetadata,
    Hypothesis,
    RecordProvenance,
    RecordValidationError,
    ScoreType,
    TextFields,
    UtteranceRecord,
)


def make_hypothesis(rank: int = 1, *, score: float = -0.3) -> Hypothesis:
    return Hypothesis(
        rank=rank,
        text=TextFields(raw=f"candidate {rank}", normalized=f"candidate {rank}"),
        source_system_id="whisper-small.en",
        source_original_rank=rank,
        token_ids=(1, 2),
        token_logprobs=(-0.1, -0.2),
        sequence_logscore=score,
        score_type=ScoreType.SUM_LOGPROB,
    )


def make_record(*, hypotheses: tuple[Hypothesis, ...] | None = None) -> UtteranceRecord:
    return UtteranceRecord(
        utt_id="test-clean-0001",
        dataset="librispeech",
        split="test_clean",
        audio_path="LibriSpeech/test-clean/0001.flac",
        reference=TextFields(raw="A reference", normalized="a reference"),
        asr=ASRMetadata(
            system_id="whisper-small.en",
            checkpoint="small.en@revision",
            code_revision="openai-whisper@revision",
            hypotheses=hypotheses or (make_hypothesis(),),
            decode_config_sha256="a" * 64,
        ),
        provenance=RecordProvenance(
            created_utc="2026-08-21T00:00:00Z",
            host="test-host",
            software_lock_sha256="b" * 64,
        ),
    )


def test_record_json_round_trip_is_exact() -> None:
    record = make_record(
        hypotheses=(make_hypothesis(1, score=-0.3), make_hypothesis(2, score=-0.7))
    )

    restored = UtteranceRecord.from_json(record.to_json())

    assert restored == record
    assert restored.to_json() == record.to_json()


def test_hypothesis_rejects_mismatched_token_scores() -> None:
    with pytest.raises(RecordValidationError, match="matching lengths"):
        Hypothesis(
            rank=1,
            text=TextFields(raw="raw", normalized="normalized"),
            source_system_id="whisper-small.en",
            source_original_rank=1,
            token_ids=(1, 2),
            token_logprobs=(-0.1,),
        )


@pytest.mark.parametrize("invalid_score", [math.nan, math.inf, -math.inf])
def test_hypothesis_rejects_nonfinite_sequence_score(invalid_score: float) -> None:
    with pytest.raises(RecordValidationError, match="sequence_logscore"):
        make_hypothesis(score=invalid_score)


def test_record_rejects_duplicate_or_out_of_order_ranks() -> None:
    with pytest.raises(RecordValidationError, match="contiguous and ordered"):
        make_record(hypotheses=(make_hypothesis(1), make_hypothesis(1)))


def test_record_rejects_hypothesis_from_another_system() -> None:
    other_system_hypothesis = Hypothesis(
        rank=1,
        text=TextFields(raw="candidate", normalized="candidate"),
        source_system_id="other-system",
        source_original_rank=1,
    )

    with pytest.raises(RecordValidationError, match="source_system_id"):
        make_record(hypotheses=(other_system_hypothesis,))
