"""Supervised error-correction input construction and model adapters."""

from .input_serialization import (
    AddedSpecialSeparatorSerializer,
    LiteralSeparatorSerializer,
    T5SentinelSerializer,
)
from .pairs import ECPair, build_ec_pair
from .sequence_scoring import SequenceScore, sequence_logprob_from_logits

__all__ = [
    "AddedSpecialSeparatorSerializer",
    "ECPair",
    "LiteralSeparatorSerializer",
    "SequenceScore",
    "T5SentinelSerializer",
    "build_ec_pair",
    "sequence_logprob_from_logits",
]
