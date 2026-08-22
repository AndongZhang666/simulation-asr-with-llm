"""Immutable, JSON-serializable records for ASR N-best artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class RecordValidationError(ValueError):
    """Raised when a persisted research record violates its contract."""


class ScoreType(str, Enum):
    """The documented semantics of a stored ASR candidate score."""

    SUM_LOGPROB = "sum_logprob"
    LENGTH_NORMALIZED_LOGPROB = "length_normalized_logprob"
    RNNT_SCORE = "rnnt_score"
    UNKNOWN = "unknown"


def _require_nonempty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise RecordValidationError(f"{field_name} must be a non-empty string")


def _require_finite(value: float, field_name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise RecordValidationError(f"{field_name} must be finite when provided")


@dataclass(frozen=True, slots=True)
class TextFields:
    """Raw text and separately persisted evaluation-normalized text."""

    raw: str
    normalized: str

    def __post_init__(self) -> None:
        if not isinstance(self.raw, str) or not isinstance(self.normalized, str):
            raise RecordValidationError("raw and normalized text must be strings")

    def to_dict(self) -> dict[str, str]:
        return {"raw": self.raw, "normalized": self.normalized}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TextFields:
        return cls(raw=data["raw"], normalized=data["normalized"])


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """One score-ordered ASR candidate before any primary-evaluation deduplication."""

    rank: int
    text: TextFields
    source_system_id: str
    source_original_rank: int
    token_ids: tuple[int, ...] = ()
    token_logprobs: tuple[float, ...] = ()
    sequence_logscore: float | None = None
    score_type: ScoreType = ScoreType.UNKNOWN
    length_penalty: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.rank, int) or isinstance(self.rank, bool) or self.rank < 1:
            raise RecordValidationError("rank must be an integer greater than or equal to 1")
        _require_nonempty(self.source_system_id, "source_system_id")
        if (
            not isinstance(self.source_original_rank, int)
            or isinstance(self.source_original_rank, bool)
            or self.source_original_rank < 1
        ):
            raise RecordValidationError(
                "source_original_rank must be an integer greater than or equal to 1"
            )
        if self.token_logprobs and len(self.token_ids) != len(self.token_logprobs):
            raise RecordValidationError("token_ids and token_logprobs must have matching lengths")
        if any(
            not isinstance(token_id, int) or isinstance(token_id, bool)
            for token_id in self.token_ids
        ):
            raise RecordValidationError("token_ids must contain integers")
        for token_logprob in self.token_logprobs:
            _require_finite(token_logprob, "token_logprobs entries")
        if self.sequence_logscore is not None:
            _require_finite(self.sequence_logscore, "sequence_logscore")
        if self.length_penalty is not None:
            _require_finite(self.length_penalty, "length_penalty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "text_raw": self.text.raw,
            "text_normalized": self.text.normalized,
            "token_ids": list(self.token_ids),
            "token_logprobs": list(self.token_logprobs),
            "sequence_logscore": self.sequence_logscore,
            "score_type": self.score_type.value,
            "length_penalty": self.length_penalty,
            "source_system_id": self.source_system_id,
            "source_original_rank": self.source_original_rank,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Hypothesis:
        return cls(
            rank=data["rank"],
            text=TextFields(raw=data["text_raw"], normalized=data["text_normalized"]),
            source_system_id=data["source_system_id"],
            source_original_rank=data["source_original_rank"],
            token_ids=tuple(data.get("token_ids", ())),
            token_logprobs=tuple(data.get("token_logprobs", ())),
            sequence_logscore=data.get("sequence_logscore"),
            score_type=ScoreType(data.get("score_type", ScoreType.UNKNOWN.value)),
            length_penalty=data.get("length_penalty"),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class ASRMetadata:
    """Pinned backend identity and candidate list for one utterance."""

    system_id: str
    checkpoint: str
    code_revision: str
    hypotheses: tuple[Hypothesis, ...]
    decode_config_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty(self.system_id, "system_id")
        _require_nonempty(self.checkpoint, "checkpoint")
        _require_nonempty(self.code_revision, "code_revision")
        if not self.hypotheses:
            raise RecordValidationError("at least one hypothesis is required")
        expected_ranks = tuple(range(1, len(self.hypotheses) + 1))
        ranks = tuple(hypothesis.rank for hypothesis in self.hypotheses)
        if ranks != expected_ranks:
            raise RecordValidationError("hypothesis ranks must be contiguous and ordered from 1")
        if any(hypothesis.source_system_id != self.system_id for hypothesis in self.hypotheses):
            raise RecordValidationError("each hypothesis source_system_id must match ASR system_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_id": self.system_id,
            "checkpoint": self.checkpoint,
            "code_revision": self.code_revision,
            "decode_config_sha256": self.decode_config_sha256,
            "hypotheses": [hypothesis.to_dict() for hypothesis in self.hypotheses],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ASRMetadata:
        return cls(
            system_id=data["system_id"],
            checkpoint=data["checkpoint"],
            code_revision=data["code_revision"],
            decode_config_sha256=data.get("decode_config_sha256"),
            hypotheses=tuple(Hypothesis.from_dict(item) for item in data["hypotheses"]),
        )


@dataclass(frozen=True, slots=True)
class RecordProvenance:
    """Creation context that travels with immutable ASR artifacts."""

    created_utc: str
    host: str
    software_lock_sha256: str | None = None
    normalizer_id: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty(self.created_utc, "created_utc")
        _require_nonempty(self.host, "host")

    def to_dict(self) -> dict[str, str | None]:
        return {
            "created_utc": self.created_utc,
            "host": self.host,
            "software_lock_sha256": self.software_lock_sha256,
            "normalizer_id": self.normalizer_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RecordProvenance:
        return cls(
            created_utc=data["created_utc"],
            host=data["host"],
            software_lock_sha256=data.get("software_lock_sha256"),
            normalizer_id=data.get("normalizer_id"),
        )


@dataclass(frozen=True, slots=True)
class UtteranceRecord:
    """Versioned interchange record for one evaluated ASR utterance."""

    utt_id: str
    dataset: str
    split: str
    audio_path: str
    reference: TextFields
    asr: ASRMetadata
    provenance: RecordProvenance
    schema_version: str = "1.0"
    lattice_uri: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty(self.utt_id, "utt_id")
        _require_nonempty(self.dataset, "dataset")
        _require_nonempty(self.split, "split")
        _require_nonempty(self.audio_path, "audio_path")
        if self.schema_version != "1.0":
            raise RecordValidationError(f"unsupported schema version: {self.schema_version}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "utt_id": self.utt_id,
            "dataset": self.dataset,
            "split": self.split,
            "audio_path": self.audio_path,
            "reference": self.reference.to_dict(),
            "asr": self.asr.to_dict(),
            "lattice_uri": self.lattice_uri,
            "provenance": self.provenance.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> UtteranceRecord:
        return cls(
            schema_version=data.get("schema_version", "1.0"),
            utt_id=data["utt_id"],
            dataset=data["dataset"],
            split=data["split"],
            audio_path=data["audio_path"],
            reference=TextFields.from_dict(data["reference"]),
            asr=ASRMetadata.from_dict(data["asr"]),
            lattice_uri=data.get("lattice_uri"),
            provenance=RecordProvenance.from_dict(data["provenance"]),
        )

    @classmethod
    def from_json(cls, payload: str) -> UtteranceRecord:
        return cls.from_dict(json.loads(payload))
