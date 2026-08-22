import pytest

from asr_ec.data.normalization import ConservativeEnglishNormalizer, IdentityNormalizer
from asr_ec.evaluation.metrics import (
    MetricError,
    align_tokens,
    character_error_rate,
    corpus_word_error_rate,
    correction_diagnostics,
    word_error_rate_reduction,
)


def test_alignment_reports_hand_calculated_substitution_deletion_and_insertion() -> None:
    substitution = align_tokens(["a", "b", "c"], ["a", "x", "c"]).counts
    deletion = align_tokens(["a", "b", "c"], ["a", "c"]).counts
    insertion = align_tokens(["a", "c"], ["a", "b", "c"]).counts

    assert substitution.substitutions == 1
    assert substitution.deletions == 0
    assert substitution.insertions == 0
    assert deletion.deletions == 1
    assert insertion.insertions == 1


def test_empty_reference_and_hypothesis_are_explicit() -> None:
    empty = align_tokens([], []).counts
    insertion_only = align_tokens([], ["extra"]).counts
    missing = align_tokens(["missing"], []).counts

    assert empty.error_rate == 0.0
    assert insertion_only.insertions == 1
    assert insertion_only.error_rate is None
    assert missing.deletions == 1
    assert missing.error_rate == 1.0


def test_corpus_wer_uses_summed_counts_not_mean_utterance_wer() -> None:
    normalizer = IdentityNormalizer()
    metrics = corpus_word_error_rate(["a", "a b c d"], ["x", "a b c e"], normalizer=normalizer)
    utterance_average = (1.0 + 0.25) / 2

    assert metrics.counts.errors == 2
    assert metrics.counts.reference_tokens == 5
    assert metrics.error_rate == pytest.approx(0.4)
    assert metrics.error_rate != pytest.approx(utterance_average)


def test_cer_uses_normalized_characters_without_whitespace() -> None:
    metrics = character_error_rate(["A B"], ["A C"], normalizer=ConservativeEnglishNormalizer())

    assert metrics.counts.substitutions == 1
    assert metrics.counts.reference_tokens == 2
    assert metrics.error_rate == pytest.approx(0.5)


def test_werr_and_zero_baseline_handling() -> None:
    assert word_error_rate_reduction(0.2, 0.1) == pytest.approx(50.0)
    assert word_error_rate_reduction(0.0, 0.0) is None
    with pytest.raises(MetricError, match="non-negative"):
        word_error_rate_reduction(-0.1, 0.1)


def test_correction_diagnostics_distinguishes_benefit_and_overcorrection() -> None:
    diagnostics = correction_diagnostics(
        ["correct", "fixed", "target", "reference"],
        ["correct", "wrong", "wrong", "wrong"],
        ["different", "fixed", "other", "wrong"],
        normalizer=IdentityNormalizer(),
    )

    assert diagnostics.exact_copy_count == 1
    assert diagnostics.baseline_correct_ec_wrong == 1
    assert diagnostics.baseline_wrong_ec_correct == 1
    assert diagnostics.baseline_wrong_ec_different_wrong == 1


def test_metrics_reject_mismatched_corpus_lengths() -> None:
    with pytest.raises(MetricError, match="same number"):
        corpus_word_error_rate(["one"], [], normalizer=IdentityNormalizer())
