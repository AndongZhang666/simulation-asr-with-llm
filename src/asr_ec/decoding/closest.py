"""Deterministic closest projection from free EC text onto the original N-best list."""

from __future__ import annotations

from typing import Mapping

from asr_ec.contracts.outputs import ECOutput
from asr_ec.contracts.records import TextFields, UtteranceRecord
from asr_ec.data.normalization import TextNormalizer
from asr_ec.evaluation.metrics import align_tokens


class ClosestProjectionError(ValueError):
    """Raised when a closest projection cannot use the stored candidate set."""


def project_to_closest(
    record: UtteranceRecord,
    *,
    unconstrained_text: str,
    ec_scores_by_rank: Mapping[int, float],
    normalizer: TextNormalizer,
    ec_system_id: str,
) -> ECOutput:
    """Project free text by word distance, then EC score, ASR rank, and lexical order."""

    ranks = {candidate.rank for candidate in record.asr.hypotheses}
    if set(ec_scores_by_rank) != ranks:
        raise ClosestProjectionError(
            "EC scores must contain exactly one score for every candidate rank"
        )
    generated_tokens = normalizer.normalize(unconstrained_text).split()
    scored_candidates = []
    for candidate in record.asr.hypotheses:
        candidate_tokens = normalizer.normalize(candidate.text.raw).split()
        distance = align_tokens(generated_tokens, candidate_tokens).counts.errors
        scored_candidates.append(
            (
                distance,
                -ec_scores_by_rank[candidate.rank],
                candidate.rank,
                candidate.text.raw,
                candidate,
            )
        )
    distance, negative_ec_score, _, _, selected = min(scored_candidates)
    return ECOutput(
        utt_id=record.utt_id,
        ec_system_id=ec_system_id,
        decoding_strategy="closest",
        output=TextFields(raw=selected.text.raw, normalized=selected.text.normalized),
        selected_candidate_rank=selected.rank,
        ec_sequence_score=-negative_ec_score,
        provenance={"word_levenshtein_distance": distance},
    )
