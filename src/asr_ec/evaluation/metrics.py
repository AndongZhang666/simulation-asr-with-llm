"""Deterministic dynamic-programming alignment and corpus-level error metrics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Sequence

from asr_ec.data.normalization import TextNormalizer


class MetricError(ValueError):
    """Raised when metric inputs do not support a meaningful comparison."""


class EditOperationType(str, Enum):
    CORRECT = "correct"
    SUBSTITUTION = "substitution"
    DELETION = "deletion"
    INSERTION = "insertion"


@dataclass(frozen=True, slots=True)
class EditOperation:
    operation: EditOperationType
    reference_token: str | None
    hypothesis_token: str | None


@dataclass(frozen=True, slots=True)
class EditCounts:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_tokens: int = 0

    def __post_init__(self) -> None:
        values = (self.substitutions, self.deletions, self.insertions, self.reference_tokens)
        if any(not isinstance(value, int) or value < 0 for value in values):
            raise MetricError("edit counts must be non-negative integers")

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def error_rate(self) -> float | None:
        """Return error rate, or None when a non-empty hypothesis has an empty reference."""

        if self.reference_tokens == 0:
            return 0.0 if self.errors == 0 else None
        return self.errors / self.reference_tokens

    def __add__(self, other: EditCounts) -> EditCounts:
        return EditCounts(
            substitutions=self.substitutions + other.substitutions,
            deletions=self.deletions + other.deletions,
            insertions=self.insertions + other.insertions,
            reference_tokens=self.reference_tokens + other.reference_tokens,
        )

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "reference_tokens": self.reference_tokens,
            "errors": self.errors,
            "error_rate": self.error_rate,
        }


@dataclass(frozen=True, slots=True)
class Alignment:
    operations: tuple[EditOperation, ...]
    counts: EditCounts


@dataclass(frozen=True, slots=True)
class CorpusMetrics:
    metric_name: str
    normalizer_id: str
    counts: EditCounts
    utterance_count: int

    @property
    def error_rate(self) -> float | None:
        return self.counts.error_rate

    def to_dict(self) -> dict[str, object]:
        return {
            "metric_name": self.metric_name,
            "normalizer_id": self.normalizer_id,
            "utterance_count": self.utterance_count,
            "counts": self.counts.to_dict(),
            "error_rate": self.error_rate,
        }


@dataclass(frozen=True, slots=True)
class CorrectionDiagnostics:
    utterance_count: int
    exact_copy_count: int
    baseline_correct_ec_wrong: int
    baseline_wrong_ec_correct: int
    baseline_wrong_ec_different_wrong: int

    @property
    def exact_copy_rate(self) -> float:
        return self.exact_copy_count / self.utterance_count if self.utterance_count else 0.0

    def to_dict(self) -> dict[str, int | float]:
        return {
            "utterance_count": self.utterance_count,
            "exact_copy_count": self.exact_copy_count,
            "exact_copy_rate": self.exact_copy_rate,
            "baseline_correct_ec_wrong": self.baseline_correct_ec_wrong,
            "baseline_wrong_ec_correct": self.baseline_wrong_ec_correct,
            "baseline_wrong_ec_different_wrong": self.baseline_wrong_ec_different_wrong,
        }


def align_tokens(reference: Sequence[str], hypothesis: Sequence[str]) -> Alignment:
    """Align token sequences with deterministic tie-breaking and return S/D/I counts."""

    reference_length = len(reference)
    hypothesis_length = len(hypothesis)
    costs = [[0] * (hypothesis_length + 1) for _ in range(reference_length + 1)]
    for reference_index in range(1, reference_length + 1):
        costs[reference_index][0] = reference_index
    for hypothesis_index in range(1, hypothesis_length + 1):
        costs[0][hypothesis_index] = hypothesis_index

    for reference_index in range(1, reference_length + 1):
        for hypothesis_index in range(1, hypothesis_length + 1):
            substitution_cost = costs[reference_index - 1][hypothesis_index - 1] + (
                reference[reference_index - 1] != hypothesis[hypothesis_index - 1]
            )
            deletion_cost = costs[reference_index - 1][hypothesis_index] + 1
            insertion_cost = costs[reference_index][hypothesis_index - 1] + 1
            costs[reference_index][hypothesis_index] = min(
                substitution_cost, deletion_cost, insertion_cost
            )

    operations: list[EditOperation] = []
    reference_index = reference_length
    hypothesis_index = hypothesis_length
    while reference_index or hypothesis_index:
        current_cost = costs[reference_index][hypothesis_index]
        if (
            reference_index
            and hypothesis_index
            and reference[reference_index - 1] == hypothesis[hypothesis_index - 1]
            and costs[reference_index - 1][hypothesis_index - 1] == current_cost
        ):
            operations.append(
                EditOperation(
                    EditOperationType.CORRECT,
                    reference[reference_index - 1],
                    hypothesis[hypothesis_index - 1],
                )
            )
            reference_index -= 1
            hypothesis_index -= 1
        elif (
            reference_index
            and hypothesis_index
            and costs[reference_index - 1][hypothesis_index - 1] + 1 == current_cost
        ):
            operations.append(
                EditOperation(
                    EditOperationType.SUBSTITUTION,
                    reference[reference_index - 1],
                    hypothesis[hypothesis_index - 1],
                )
            )
            reference_index -= 1
            hypothesis_index -= 1
        elif reference_index and costs[reference_index - 1][hypothesis_index] + 1 == current_cost:
            operations.append(
                EditOperation(EditOperationType.DELETION, reference[reference_index - 1], None)
            )
            reference_index -= 1
        else:
            operations.append(
                EditOperation(EditOperationType.INSERTION, None, hypothesis[hypothesis_index - 1])
            )
            hypothesis_index -= 1

    operations.reverse()
    substitutions = sum(
        operation.operation is EditOperationType.SUBSTITUTION for operation in operations
    )
    deletions = sum(operation.operation is EditOperationType.DELETION for operation in operations)
    insertions = sum(operation.operation is EditOperationType.INSERTION for operation in operations)
    return Alignment(
        operations=tuple(operations),
        counts=EditCounts(
            substitutions=substitutions,
            deletions=deletions,
            insertions=insertions,
            reference_tokens=reference_length,
        ),
    )


def _word_tokens(text: str, normalizer: TextNormalizer) -> list[str]:
    return normalizer.normalize(text).split()


def _character_tokens(text: str, normalizer: TextNormalizer) -> list[str]:
    return list(normalizer.normalize(text).replace(" ", ""))


def _score_corpus(
    references: Sequence[str],
    hypotheses: Sequence[str],
    *,
    normalizer: TextNormalizer,
    metric_name: str,
    tokenizer: Callable[[str, TextNormalizer], list[str]],
) -> CorpusMetrics:
    if len(references) != len(hypotheses):
        raise MetricError("references and hypotheses must have the same number of utterances")
    total = EditCounts()
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        total += align_tokens(
            tokenizer(reference, normalizer), tokenizer(hypothesis, normalizer)
        ).counts
    return CorpusMetrics(
        metric_name=metric_name,
        normalizer_id=normalizer.normalizer_id,
        counts=total,
        utterance_count=len(references),
    )


def corpus_word_error_rate(
    references: Sequence[str], hypotheses: Sequence[str], *, normalizer: TextNormalizer
) -> CorpusMetrics:
    """Score word error rates from summed corpus edit counts."""

    return _score_corpus(
        references,
        hypotheses,
        normalizer=normalizer,
        metric_name="wer",
        tokenizer=_word_tokens,
    )


def character_error_rate(
    references: Sequence[str], hypotheses: Sequence[str], *, normalizer: TextNormalizer
) -> CorpusMetrics:
    """Score character error rates after removing whitespace from normalized text."""

    return _score_corpus(
        references,
        hypotheses,
        normalizer=normalizer,
        metric_name="cer",
        tokenizer=_character_tokens,
    )


def word_error_rate_reduction(baseline_wer: float, system_wer: float) -> float | None:
    """Return relative WER reduction, or None when the baseline is already zero."""

    if baseline_wer < 0 or system_wer < 0:
        raise MetricError("WER values must be non-negative")
    if baseline_wer == 0:
        return None
    return (baseline_wer - system_wer) / baseline_wer * 100


def correction_diagnostics(
    references: Iterable[str],
    baseline_hypotheses: Iterable[str],
    corrected_hypotheses: Iterable[str],
    *,
    normalizer: TextNormalizer,
) -> CorrectionDiagnostics:
    """Measure exact copies, beneficial corrections, and overcorrection states."""

    reference_items = list(references)
    baseline_items = list(baseline_hypotheses)
    corrected_items = list(corrected_hypotheses)
    if not (len(reference_items) == len(baseline_items) == len(corrected_items)):
        raise MetricError("reference, baseline, and corrected inputs must have equal lengths")

    exact_copy_count = 0
    baseline_correct_ec_wrong = 0
    baseline_wrong_ec_correct = 0
    baseline_wrong_ec_different_wrong = 0
    for reference, baseline, corrected in zip(
        reference_items, baseline_items, corrected_items, strict=True
    ):
        normalized_reference = normalizer.normalize(reference)
        normalized_baseline = normalizer.normalize(baseline)
        normalized_corrected = normalizer.normalize(corrected)
        exact_copy_count += normalized_baseline == normalized_corrected
        baseline_correct = normalized_baseline == normalized_reference
        corrected_correct = normalized_corrected == normalized_reference
        if baseline_correct and not corrected_correct:
            baseline_correct_ec_wrong += 1
        elif not baseline_correct and corrected_correct:
            baseline_wrong_ec_correct += 1
        elif (
            not baseline_correct
            and not corrected_correct
            and normalized_baseline != normalized_corrected
        ):
            baseline_wrong_ec_different_wrong += 1

    return CorrectionDiagnostics(
        utterance_count=len(reference_items),
        exact_copy_count=exact_copy_count,
        baseline_correct_ec_wrong=baseline_correct_ec_wrong,
        baseline_wrong_ec_correct=baseline_wrong_ec_correct,
        baseline_wrong_ec_different_wrong=baseline_wrong_ec_different_wrong,
    )
