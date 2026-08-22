"""Stable N-best oracle selection with corpus-level aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from asr_ec.contracts.records import Hypothesis, UtteranceRecord
from asr_ec.data.normalization import TextNormalizer

from .metrics import CorpusMetrics, EditCounts, MetricError, align_tokens


@dataclass(frozen=True, slots=True)
class OracleSelection:
    utt_id: str
    selected_rank: int
    selected_text_raw: str
    counts: EditCounts


def select_oracle(
    reference: str,
    hypotheses: Sequence[Hypothesis],
    *,
    normalizer: TextNormalizer,
    utt_id: str = "",
) -> OracleSelection:
    """Choose the candidate with fewest edits, breaking equal-cost ties by ASR rank."""

    if not hypotheses:
        raise MetricError("oracle selection requires at least one hypothesis")
    normalized_reference = normalizer.normalize(reference).split()
    ranked: list[tuple[int, int, Hypothesis, EditCounts]] = []
    for hypothesis in hypotheses:
        counts = align_tokens(
            normalized_reference, normalizer.normalize(hypothesis.text.raw).split()
        ).counts
        ranked.append((counts.errors, hypothesis.rank, hypothesis, counts))
    _, _, selected, selected_counts = min(ranked, key=lambda item: (item[0], item[1]))
    return OracleSelection(
        utt_id=utt_id,
        selected_rank=selected.rank,
        selected_text_raw=selected.text.raw,
        counts=selected_counts,
    )


def oracle_corpus_metrics(
    records: Sequence[UtteranceRecord], *, normalizer: TextNormalizer
) -> tuple[CorpusMetrics, tuple[OracleSelection, ...]]:
    """Aggregate per-utterance N-best oracle choices as one corpus metric."""

    total = EditCounts()
    selections: list[OracleSelection] = []
    for record in records:
        selection = select_oracle(
            record.reference.raw,
            record.asr.hypotheses,
            normalizer=normalizer,
            utt_id=record.utt_id,
        )
        total += selection.counts
        selections.append(selection)
    return (
        CorpusMetrics(
            metric_name="oracle_wer",
            normalizer_id=normalizer.normalizer_id,
            counts=total,
            utterance_count=len(records),
        ),
        tuple(selections),
    )
