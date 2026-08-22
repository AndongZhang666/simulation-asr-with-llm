from asr_ec.contracts.records import (
    ASRMetadata,
    Hypothesis,
    RecordProvenance,
    TextFields,
    UtteranceRecord,
)
from asr_ec.data.normalization import IdentityNormalizer
from asr_ec.decoding.closest import project_to_closest
from asr_ec.decoding.nbest_constrained import constrained_decode, tune_lambda
from asr_ec.decoding.unconstrained import unconstrained_output


def make_record() -> UtteranceRecord:
    hypotheses = (
        Hypothesis(
            rank=1,
            text=TextFields(raw="a wrong answer", normalized="a wrong answer"),
            source_system_id="fixture-asr",
            source_original_rank=1,
            sequence_logscore=-1.0,
        ),
        Hypothesis(
            rank=2,
            text=TextFields(raw="a correct answer", normalized="a correct answer"),
            source_system_id="fixture-asr",
            source_original_rank=2,
            sequence_logscore=-2.0,
        ),
    )
    return UtteranceRecord(
        utt_id="fixture-1",
        dataset="fixture",
        split="validation",
        audio_path="fixture.flac",
        reference=TextFields(raw="a correct answer", normalized="a correct answer"),
        asr=ASRMetadata(
            system_id="fixture-asr",
            checkpoint="fixture",
            code_revision="fixture",
            hypotheses=hypotheses,
        ),
        provenance=RecordProvenance(created_utc="2026-08-21T00:00:00Z", host="fixture"),
    )


def test_constrained_lambda_endpoints_and_membership() -> None:
    record = make_record()
    scores = {1: -3.0, 2: -0.5}

    asr_only = constrained_decode(
        record, ec_scores_by_rank=scores, lambda_value=0, ec_system_id="fixture"
    )
    ec_only = constrained_decode(
        record, ec_scores_by_rank=scores, lambda_value=1, ec_system_id="fixture"
    )

    assert asr_only.selected_candidate_rank == 1
    assert ec_only.selected_candidate_rank == 2
    assert asr_only.output.raw in {candidate.text.raw for candidate in record.asr.hypotheses}
    assert ec_only.output.raw in {candidate.text.raw for candidate in record.asr.hypotheses}


def test_closest_projection_uses_word_distance_then_ec_score() -> None:
    result = project_to_closest(
        make_record(),
        unconstrained_text="a correct result",
        ec_scores_by_rank={1: -2.0, 2: -0.1},
        normalizer=IdentityNormalizer(),
        ec_system_id="fixture",
    )

    assert result.selected_candidate_rank == 2
    assert result.output.raw == "a correct answer"


def test_lambda_tuning_uses_validation_wer_and_lowest_tie() -> None:
    record = make_record()
    result = tune_lambda(
        (record,),
        ec_scores_by_utt_id={record.utt_id: {1: -3.0, 2: -0.5}},
        normalizer=IdentityNormalizer(),
        ec_system_id="fixture",
        lambda_values=(0, 0.5, 1),
    )

    assert result.validation_curve[0].error_rate > result.validation_curve[1].error_rate
    assert result.selected_lambda == 0.5


def test_unconstrained_output_is_not_claimed_as_candidate() -> None:
    result = unconstrained_output(
        make_record(),
        generated_text="freely generated correction",
        normalizer=IdentityNormalizer(),
        ec_system_id="fixture",
    )

    assert result.selected_candidate_rank is None
    assert result.output.raw == "freely generated correction"
