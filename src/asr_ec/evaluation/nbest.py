"""Evaluation and diversity diagnostics for immutable ASR N-best artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

from asr_ec.contracts.records import UtteranceRecord
from asr_ec.data.normalization import TextNormalizer

from .metrics import CorpusMetrics, EditCounts, align_tokens, corpus_word_error_rate
from .oracle import oracle_corpus_metrics


class NBestEvaluationError(ValueError):
    """Raised when a stored N-best artifact cannot support a coherent evaluation."""


@dataclass(frozen=True, slots=True)
class NBestDiversity:
    requested_n: int
    average_raw_unique: float
    average_normalized_unique: float
    cross_wer: CorpusMetrics
    empty_candidate_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_n": self.requested_n,
            "average_raw_unique": self.average_raw_unique,
            "average_normalized_unique": self.average_normalized_unique,
            "cross_wer": self.cross_wer.to_dict(),
            "empty_candidate_count": self.empty_candidate_count,
        }


@dataclass(frozen=True, slots=True)
class NBestEvaluation:
    baseline: CorpusMetrics
    oracle_by_n: dict[int, CorpusMetrics]
    diversity: NBestDiversity

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline": self.baseline.to_dict(),
            "oracle_by_n": {
                str(nbest): metrics.to_dict() for nbest, metrics in self.oracle_by_n.items()
            },
            "diversity": self.diversity.to_dict(),
        }


def evaluate_nbest(
    records: Sequence[UtteranceRecord], *, normalizer: TextNormalizer, requested_n: int
) -> NBestEvaluation:
    """Compute corpus baseline/oracle metrics and documented within-list diversity."""

    if not records:
        raise NBestEvaluationError("N-best evaluation requires at least one utterance record")
    if requested_n < 1:
        raise NBestEvaluationError("requested_n must be positive")
    if any(len(record.asr.hypotheses) < requested_n for record in records):
        raise NBestEvaluationError("every record must contain the requested N-best count")

    references = [record.reference.raw for record in records]
    baseline_hypotheses = [record.asr.hypotheses[0].text.raw for record in records]
    baseline = corpus_word_error_rate(references, baseline_hypotheses, normalizer=normalizer)
    oracle_by_n = {}
    for nbest in sorted({1, min(5, requested_n), requested_n}):
        truncated_records = tuple(
            UtteranceRecord(
                utt_id=record.utt_id,
                dataset=record.dataset,
                split=record.split,
                audio_path=record.audio_path,
                reference=record.reference,
                asr=record.asr.__class__(
                    system_id=record.asr.system_id,
                    checkpoint=record.asr.checkpoint,
                    code_revision=record.asr.code_revision,
                    decode_config_sha256=record.asr.decode_config_sha256,
                    hypotheses=record.asr.hypotheses[:nbest],
                ),
                provenance=record.provenance,
                schema_version=record.schema_version,
                lattice_uri=record.lattice_uri,
            )
            for record in records
        )
        oracle_by_n[nbest] = oracle_corpus_metrics(truncated_records, normalizer=normalizer)[0]
    diversity = _diversity(records, normalizer=normalizer, requested_n=requested_n)
    return NBestEvaluation(baseline=baseline, oracle_by_n=oracle_by_n, diversity=diversity)


def _diversity(
    records: Sequence[UtteranceRecord], *, normalizer: TextNormalizer, requested_n: int
) -> NBestDiversity:
    raw_unique_counts = []
    normalized_unique_counts = []
    total_cross_counts = EditCounts()
    empty_candidate_count = 0
    for record in records:
        hypotheses = record.asr.hypotheses[:requested_n]
        raw_unique_counts.append(len({hypothesis.text.raw for hypothesis in hypotheses}))
        normalized_texts = []
        for hypothesis in hypotheses:
            normalized = normalizer.normalize(hypothesis.text.raw)
            empty_candidate_count += not bool(normalized)
            if normalized not in normalized_texts:
                normalized_texts.append(normalized)
        normalized_unique_counts.append(len(normalized_texts))
        for reference_text, hypothesis_text in combinations(normalized_texts, 2):
            total_cross_counts += align_tokens(
                reference_text.split(), hypothesis_text.split()
            ).counts
    return NBestDiversity(
        requested_n=requested_n,
        average_raw_unique=sum(raw_unique_counts) / len(records),
        average_normalized_unique=sum(normalized_unique_counts) / len(records),
        cross_wer=CorpusMetrics(
            metric_name="cross_wer",
            normalizer_id=normalizer.normalizer_id,
            counts=total_cross_counts,
            utterance_count=len(records),
        ),
        empty_candidate_count=empty_candidate_count,
    )
