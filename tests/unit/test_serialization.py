import pytest

from asr_ec.contracts.records import Hypothesis, TextFields
from asr_ec.ec.input_serialization import (
    AddedSpecialSeparatorSerializer,
    LiteralSeparatorSerializer,
    SerializationError,
    T5SentinelSerializer,
)


def make_hypothesis(rank: int, text: str) -> Hypothesis:
    return Hypothesis(
        rank=rank,
        text=TextFields(raw=text, normalized=text),
        source_system_id="fixture",
        source_original_rank=rank,
    )


def test_literal_separator_preserves_asr_rank_order() -> None:
    serializer = LiteralSeparatorSerializer()

    assert serializer.serialize((make_hypothesis(1, "first"), make_hypothesis(2, "second"))) == (
        "first [SEP] second"
    )


def test_separator_variants_are_explicit_and_fingerprinted() -> None:
    hypotheses = (make_hypothesis(1, "first"), make_hypothesis(2, "second"))

    assert AddedSpecialSeparatorSerializer().serialize(hypotheses) == "first <asr_sep> second"
    assert T5SentinelSerializer().serialize(hypotheses) == "first <extra_id_0> second"
    assert LiteralSeparatorSerializer().fingerprint() != T5SentinelSerializer().fingerprint()


def test_serializer_rejects_empty_or_out_of_order_nbest() -> None:
    serializer = LiteralSeparatorSerializer()

    with pytest.raises(SerializationError, match="empty"):
        serializer.serialize(())
    with pytest.raises(SerializationError, match="rank order"):
        serializer.serialize((make_hypothesis(2, "second"), make_hypothesis(1, "first")))
