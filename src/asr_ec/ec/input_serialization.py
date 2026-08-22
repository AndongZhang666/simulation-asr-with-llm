"""Explicit, fingerprinted ordered N-best serializers for supervised T5 inputs."""

from __future__ import annotations

import hashlib
import json
from typing import Protocol, Sequence

from asr_ec.contracts.records import Hypothesis


class SerializationError(ValueError):
    """Raised when an N-best list cannot be serialized without losing rank meaning."""


class InputSerializer(Protocol):
    """Transform an ASR-score-ordered N-best list into an EC model input string."""

    @property
    def serializer_id(self) -> str:
        """Return a stable configuration-derived identifier."""

    def serialize(self, hypotheses: Sequence[Hypothesis]) -> str:
        """Serialize candidate text in its existing ASR rank order."""

    def fingerprint(self) -> str:
        """Return a SHA-256 hash of all text-shaping settings."""


class _SeparatorSerializer:
    variant: str

    def __init__(self, *, separator: str, prefix: str = "") -> None:
        if not separator:
            raise SerializationError("separator must be non-empty")
        self._separator = separator
        self._prefix = prefix

    @property
    def serializer_id(self) -> str:
        return f"{self.variant}@v1"

    def serialize(self, hypotheses: Sequence[Hypothesis]) -> str:
        _validate_rank_order(hypotheses)
        serialized = f" {self._separator} ".join(hypothesis.text.raw for hypothesis in hypotheses)
        return f"{self._prefix}{serialized}"

    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "serializer_id": self.serializer_id,
                "separator": self._separator,
                "prefix": self._prefix,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LiteralSeparatorSerializer(_SeparatorSerializer):
    """Primary paper-visible representation: ``hyp1 [SEP] hyp2 ...``."""

    variant = "literal-sep"

    def __init__(self, *, prefix: str = "") -> None:
        super().__init__(separator="[SEP]", prefix=prefix)


class AddedSpecialSeparatorSerializer(_SeparatorSerializer):
    """Explicit ablation that requires the T5 tokenizer to add one special token."""

    variant = "added-special-sep"

    def __init__(self, *, special_token: str = "<asr_sep>", prefix: str = "") -> None:
        super().__init__(separator=special_token, prefix=prefix)


class T5SentinelSerializer(_SeparatorSerializer):
    """Explicit ablation using a native T5 sentinel token as the separator."""

    variant = "t5-sentinel"

    def __init__(self, *, sentinel: str = "<extra_id_0>", prefix: str = "") -> None:
        super().__init__(separator=sentinel, prefix=prefix)


def _validate_rank_order(hypotheses: Sequence[Hypothesis]) -> None:
    if not hypotheses:
        raise SerializationError("cannot serialize an empty N-best list")
    ranks = tuple(hypothesis.rank for hypothesis in hypotheses)
    expected_ranks = tuple(range(1, len(hypotheses) + 1))
    if ranks != expected_ranks:
        raise SerializationError("hypotheses must be contiguous and supplied in ASR rank order")
