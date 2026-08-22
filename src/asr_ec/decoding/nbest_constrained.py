"""Candidate-only decoding with paper-form score interpolation and validation-only tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from asr_ec.contracts.outputs import ECOutput
from asr_ec.contracts.records import TextFields, UtteranceRecord
from asr_ec.data.normalization import TextNormalizer
from asr_ec.evaluation.metrics import CorpusMetrics, corpus_word_error_rate


class ConstrainedDecodingError(ValueError):
    """Raised when candidate scores cannot be reconciled with a stored N-best list."""


@dataclass(frozen=True, slots=True)
class LambdaTuningResult:
    selected_lambda: float
    validation_curve: dict[float, CorpusMetrics]


def constrained_decode(
    record: UtteranceRecord,
    *,
    ec_scores_by_rank: Mapping[int, float],
    lambda_value: float,
    ec_system_id: str,
) -> ECOutput:
    """Select one original N-best candidate using the paper's score interpolation."""

    if not 0 <= lambda_value <= 1:
        raise ConstrainedDecodingError("lambda_value must be within [0, 1]")
    candidates = record.asr.hypotheses
    expected_ranks = {candidate.rank for candidate in candidates}
    if set(ec_scores_by_rank) != expected_ranks:
        raise ConstrainedDecodingError(
            "EC scores must contain exactly one score for every candidate rank"
        )
    if any(candidate.sequence_logscore is None for candidate in candidates):
        raise ConstrainedDecodingError(
            "constrained decoding requires an ASR sequence score per candidate"
        )

    scored_candidates = []
    for candidate in candidates:
        asr_score = candidate.sequence_logscore
        assert asr_score is not None
        ec_score = ec_scores_by_rank[candidate.rank]
        combined_score = lambda_value * ec_score + (1 - lambda_value) * asr_score
        scored_candidates.append((combined_score, ec_score, candidate.rank, candidate))
    _, ec_score, _, selected = max(
        scored_candidates,
        key=lambda item: (item[0], item[1], -item[2], item[3].text.raw),
    )
    asr_score = selected.sequence_logscore
    assert asr_score is not None
    return ECOutput(
        utt_id=record.utt_id,
        ec_system_id=ec_system_id,
        decoding_strategy="nbest_constrained",
        output=TextFields(raw=selected.text.raw, normalized=selected.text.normalized),
        selected_candidate_rank=selected.rank,
        lambda_value=lambda_value,
        ec_sequence_score=ec_score,
        asr_sequence_score=asr_score,
        combined_score=lambda_value * ec_score + (1 - lambda_value) * asr_score,
    )


def tune_lambda(
    validation_records: Sequence[UtteranceRecord],
    *,
    ec_scores_by_utt_id: Mapping[str, Mapping[int, float]],
    normalizer: TextNormalizer,
    ec_system_id: str,
    lambda_values: Sequence[float] | None = None,
) -> LambdaTuningResult:
    """Choose the lowest validation WER lambda, resolving exact ties toward lower lambda."""

    if not validation_records:
        raise ConstrainedDecodingError("validation lambda tuning requires at least one record")
    if lambda_values is None:
        lambda_values = tuple(index / 20 for index in range(21))
    curve: dict[float, CorpusMetrics] = {}
    for lambda_value in lambda_values:
        outputs = [
            constrained_decode(
                record,
                ec_scores_by_rank=ec_scores_by_utt_id[record.utt_id],
                lambda_value=lambda_value,
                ec_system_id=ec_system_id,
            ).output.raw
            for record in validation_records
        ]
        curve[lambda_value] = corpus_word_error_rate(
            [record.reference.raw for record in validation_records],
            outputs,
            normalizer=normalizer,
        )
    selected_lambda = min(
        curve,
        key=lambda value: (
            curve[value].error_rate if curve[value].error_rate is not None else float("inf"),
            value,
        ),
    )
    return LambdaTuningResult(selected_lambda=selected_lambda, validation_curve=curve)
