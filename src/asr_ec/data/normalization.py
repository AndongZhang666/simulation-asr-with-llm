"""Versioned normalizers used explicitly by evaluation and data preparation."""

from __future__ import annotations

import re
import unicodedata
from typing import Protocol


class TextNormalizer(Protocol):
    """A named text transformation suitable for reproducible metric calls."""

    @property
    def normalizer_id(self) -> str:
        """Return a stable implementation identifier."""

    def normalize(self, text: str) -> str:
        """Transform text without changing the original artifact field."""


class IdentityNormalizer:
    """Preserve text verbatim for tests and explicitly raw comparisons."""

    normalizer_id = "identity@v1"

    def normalize(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return text


class ConservativeEnglishNormalizer:
    """Normalize case, spacing, and punctuation without unreported number expansion.

    This adapter is intentionally not labeled as Whisper's normalizer. It supplies a
    stable Phase 1 baseline until a pinned Whisper revision is installed and tested.
    """

    normalizer_id = "conservative-english@v1"

    def normalize(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        normalized = unicodedata.normalize("NFKC", text).lower()
        normalized = normalized.replace("\u2018", "'").replace("\u2019", "'")
        normalized = re.sub(r"[^\w\s']|_", " ", normalized)
        return " ".join(normalized.split())


class WhisperEnglishNormalizer:
    """Expose pinned official Whisper normalization through the project protocol."""

    def __init__(self, package_version: str) -> None:
        try:
            from whisper.normalizers import EnglishTextNormalizer  # type: ignore[import-untyped]
        except ImportError as error:
            raise RuntimeError(
                "install the asr extra to use Whisper English normalization"
            ) from error
        self.normalizer_id = f"openai-whisper-english@{package_version}"
        self._normalizer = EnglishTextNormalizer()

    def normalize(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return str(self._normalizer(text))


def normalizer_from_id(normalizer_id: str) -> TextNormalizer:
    """Construct a known normalizer from immutable artifact provenance."""

    if normalizer_id == IdentityNormalizer.normalizer_id:
        return IdentityNormalizer()
    if normalizer_id == ConservativeEnglishNormalizer.normalizer_id:
        return ConservativeEnglishNormalizer()
    prefix = "openai-whisper-english@"
    if normalizer_id.startswith(prefix):
        return WhisperEnglishNormalizer(normalizer_id.removeprefix(prefix))
    raise ValueError(f"unsupported normalizer id: {normalizer_id}")
