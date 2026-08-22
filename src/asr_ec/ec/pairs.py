"""Immutable supervised N-best-to-reference pair construction."""

from __future__ import annotations

from dataclasses import dataclass

from asr_ec.contracts.records import UtteranceRecord

from .input_serialization import InputSerializer


@dataclass(frozen=True, slots=True)
class ECPair:
    """One inspected source/target training example built from an ASR artifact."""

    utt_id: str
    source_text: str
    target_raw: str
    target_normalized: str
    input_nbest: int
    serializer_id: str
    serializer_sha256: str

    def to_dict(self) -> dict[str, str | int]:
        return {
            "utt_id": self.utt_id,
            "source_text": self.source_text,
            "target_raw": self.target_raw,
            "target_normalized": self.target_normalized,
            "input_nbest": self.input_nbest,
            "serializer_id": self.serializer_id,
            "serializer_sha256": self.serializer_sha256,
        }


def build_ec_pair(
    record: UtteranceRecord, *, serializer: InputSerializer, input_nbest: int
) -> ECPair:
    """Build a pair using only ranked ASR text as source and manual reference as target."""

    if input_nbest < 1 or input_nbest > len(record.asr.hypotheses):
        raise ValueError("input_nbest must be between 1 and the stored candidate count")
    hypotheses = record.asr.hypotheses[:input_nbest]
    return ECPair(
        utt_id=record.utt_id,
        source_text=serializer.serialize(hypotheses),
        target_raw=record.reference.raw,
        target_normalized=record.reference.normalized,
        input_nbest=input_nbest,
        serializer_id=serializer.serializer_id,
        serializer_sha256=serializer.fingerprint(),
    )
