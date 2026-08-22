"""Transparent corpus-level ASR error-correction evaluation."""

from .metrics import (
    Alignment,
    CorpusMetrics,
    CorrectionDiagnostics,
    EditCounts,
    align_tokens,
    character_error_rate,
    corpus_word_error_rate,
    correction_diagnostics,
    word_error_rate_reduction,
)
from .oracle import OracleSelection, oracle_corpus_metrics, select_oracle

__all__ = [
    "Alignment",
    "CorrectionDiagnostics",
    "CorpusMetrics",
    "EditCounts",
    "OracleSelection",
    "align_tokens",
    "character_error_rate",
    "correction_diagnostics",
    "corpus_word_error_rate",
    "oracle_corpus_metrics",
    "select_oracle",
    "word_error_rate_reduction",
]
