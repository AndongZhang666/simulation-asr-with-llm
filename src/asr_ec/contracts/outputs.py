"""Immutable output records for EC decoding and hosted-LLM exchanges."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from .records import RecordValidationError, TextFields


class ParserStatus(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    PARSED = "parsed"
    EMPTY = "empty"
    MALFORMED = "malformed"
    TRUNCATED = "truncated"
    PROVIDER_ERROR = "provider_error"


def _require_nonempty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise RecordValidationError(f"{field_name} must be a non-empty string")


def _require_optional_finite(value: float | None, field_name: str) -> None:
    if value is not None and (
        not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
    ):
        raise RecordValidationError(f"{field_name} must be finite when provided")


@dataclass(frozen=True, slots=True)
class ECOutput:
    """One decoding result, with candidate provenance when output is constrained."""

    utt_id: str
    ec_system_id: str
    decoding_strategy: str
    output: TextFields
    selected_candidate_rank: int | None = None
    lambda_value: float | None = None
    ec_sequence_score: float | None = None
    asr_sequence_score: float | None = None
    combined_score: float | None = None
    parser_status: ParserStatus = ParserStatus.NOT_APPLICABLE
    latency_ms: float | None = None
    truncated: bool = False
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty(self.utt_id, "utt_id")
        _require_nonempty(self.ec_system_id, "ec_system_id")
        _require_nonempty(self.decoding_strategy, "decoding_strategy")
        if self.selected_candidate_rank is not None and self.selected_candidate_rank < 1:
            raise RecordValidationError("selected_candidate_rank must be positive when provided")
        if self.lambda_value is not None and not 0 <= self.lambda_value <= 1:
            raise RecordValidationError("lambda_value must be within [0, 1]")
        for value, field_name in (
            (self.lambda_value, "lambda_value"),
            (self.ec_sequence_score, "ec_sequence_score"),
            (self.asr_sequence_score, "asr_sequence_score"),
            (self.combined_score, "combined_score"),
            (self.latency_ms, "latency_ms"),
        ):
            _require_optional_finite(value, field_name)
        if self.latency_ms is not None and self.latency_ms < 0:
            raise RecordValidationError("latency_ms must not be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "utt_id": self.utt_id,
            "ec_system_id": self.ec_system_id,
            "decoding_strategy": self.decoding_strategy,
            "output_raw": self.output.raw,
            "output_normalized": self.output.normalized,
            "selected_candidate_rank": self.selected_candidate_rank,
            "lambda_value": self.lambda_value,
            "ec_sequence_score": self.ec_sequence_score,
            "asr_sequence_score": self.asr_sequence_score,
            "combined_score": self.combined_score,
            "parser_status": self.parser_status.value,
            "latency_ms": self.latency_ms,
            "truncated": self.truncated,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ECOutput:
        return cls(
            utt_id=data["utt_id"],
            ec_system_id=data["ec_system_id"],
            decoding_strategy=data["decoding_strategy"],
            output=TextFields(raw=data["output_raw"], normalized=data["output_normalized"]),
            selected_candidate_rank=data.get("selected_candidate_rank"),
            lambda_value=data.get("lambda_value"),
            ec_sequence_score=data.get("ec_sequence_score"),
            asr_sequence_score=data.get("asr_sequence_score"),
            combined_score=data.get("combined_score"),
            parser_status=ParserStatus(
                data.get("parser_status", ParserStatus.NOT_APPLICABLE.value)
            ),
            latency_ms=data.get("latency_ms"),
            truncated=data.get("truncated", False),
            provenance=data.get("provenance", {}),
        )


@dataclass(frozen=True, slots=True)
class LLMMessage:
    """One provider-neutral request message retained before any response parsing."""

    role: str
    content: str

    def __post_init__(self) -> None:
        _require_nonempty(self.role, "message role")
        if not isinstance(self.content, str):
            raise RecordValidationError("message content must be a string")

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LLMMessage:
        return cls(role=data["role"], content=data["content"])


@dataclass(frozen=True, slots=True)
class LLMExchange:
    """Cached provider call with raw request/response and parser outcome."""

    request_hash: str
    provider: str
    model_id: str
    messages: tuple[LLMMessage, ...]
    generation_parameters: Mapping[str, Any]
    requested_utc: str
    parser_status: ParserStatus
    raw_response: str | None = None
    parsed_output: TextFields | None = None
    selected_candidate_rank: int | None = None
    response_utc: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None
    retry_count: int = 0
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.request_hash) != 64 or any(
            character not in "0123456789abcdef" for character in self.request_hash
        ):
            raise RecordValidationError("request_hash must be a lowercase SHA-256 digest")
        _require_nonempty(self.provider, "provider")
        _require_nonempty(self.model_id, "model_id")
        _require_nonempty(self.requested_utc, "requested_utc")
        if not self.messages:
            raise RecordValidationError("LLM exchanges require at least one message")
        if self.selected_candidate_rank is not None and self.selected_candidate_rank < 1:
            raise RecordValidationError("selected_candidate_rank must be positive when provided")
        if self.input_tokens is not None and self.input_tokens < 0:
            raise RecordValidationError("input_tokens must not be negative")
        if self.output_tokens is not None and self.output_tokens < 0:
            raise RecordValidationError("output_tokens must not be negative")
        if self.retry_count < 0:
            raise RecordValidationError("retry_count must not be negative")
        _require_optional_finite(self.latency_ms, "latency_ms")
        if self.latency_ms is not None and self.latency_ms < 0:
            raise RecordValidationError("latency_ms must not be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_hash": self.request_hash,
            "provider": self.provider,
            "model_id": self.model_id,
            "messages": [message.to_dict() for message in self.messages],
            "generation_parameters": dict(self.generation_parameters),
            "requested_utc": self.requested_utc,
            "response_utc": self.response_utc,
            "raw_response": self.raw_response,
            "parsed_output": self.parsed_output.to_dict() if self.parsed_output else None,
            "selected_candidate_rank": self.selected_candidate_rank,
            "parser_status": self.parser_status.value,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "retry_count": self.retry_count,
            "errors": list(self.errors),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LLMExchange:
        parsed_output = data.get("parsed_output")
        return cls(
            request_hash=data["request_hash"],
            provider=data["provider"],
            model_id=data["model_id"],
            messages=tuple(LLMMessage.from_dict(item) for item in data["messages"]),
            generation_parameters=data["generation_parameters"],
            requested_utc=data["requested_utc"],
            response_utc=data.get("response_utc"),
            raw_response=data.get("raw_response"),
            parsed_output=TextFields.from_dict(parsed_output) if parsed_output else None,
            selected_candidate_rank=data.get("selected_candidate_rank"),
            parser_status=ParserStatus(data["parser_status"]),
            input_tokens=data.get("input_tokens"),
            output_tokens=data.get("output_tokens"),
            latency_ms=data.get("latency_ms"),
            retry_count=data.get("retry_count", 0),
            errors=tuple(data.get("errors", ())),
        )
