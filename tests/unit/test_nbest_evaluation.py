from asr_ec.contracts.records import (
    ASRMetadata,
    Hypothesis,
    RecordProvenance,
    TextFields,
    UtteranceRecord,
)
from asr_ec.data.normalization import IdentityNormalizer
from asr_ec.evaluation.nbest import evaluate_nbest


def make_hypothesis(rank: int, text: str) -> Hypothesis:
    return Hypothesis(
        rank=rank,
        text=TextFields(raw=text, normalized=text),
        source_system_id="fixture-asr",
        source_original_rank=rank,
    )


def make_record(utt_id: str, reference: str, hypotheses: tuple[Hypothesis, ...]) -> UtteranceRecord:
    return UtteranceRecord(
        utt_id=utt_id,
        dataset="fixture",
        split="test",
        audio_path=f"{utt_id}.flac",
        reference=TextFields(raw=reference, normalized=reference),
        asr=ASRMetadata(
            system_id="fixture-asr",
            checkpoint="fixture",
            code_revision="fixture",
            hypotheses=hypotheses,
        ),
        provenance=RecordProvenance(created_utc="2026-08-21T00:00:00Z", host="fixture"),
    )


def test_nbest_evaluation_aggregates_baseline_oracle_and_diversity() -> None:
    records = (
        make_record(
            "one",
            "a b",
            (make_hypothesis(1, "a x"), make_hypothesis(2, "a b")),
        ),
        make_record(
            "two",
            "c",
            (make_hypothesis(1, "c"), make_hypothesis(2, "c")),
        ),
    )

    result = evaluate_nbest(records, normalizer=IdentityNormalizer(), requested_n=2)

    assert result.baseline.error_rate == 1 / 3
    assert result.oracle_by_n[1].error_rate == 1 / 3
    assert result.oracle_by_n[2].error_rate == 0
    assert result.diversity.average_raw_unique == 1.5
    assert result.diversity.average_normalized_unique == 1.5
    assert result.diversity.cross_wer.counts.substitutions == 1
