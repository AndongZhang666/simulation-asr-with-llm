from asr_ec.contracts.records import (
    ASRMetadata,
    Hypothesis,
    RecordProvenance,
    TextFields,
    UtteranceRecord,
)
from asr_ec.ec.input_serialization import LiteralSeparatorSerializer
from asr_ec.ec.pairs import build_ec_pair


def test_ec_pair_uses_ranked_asr_text_as_source_and_reference_as_target() -> None:
    record = UtteranceRecord(
        utt_id="fixture-1",
        dataset="fixture",
        split="train",
        audio_path="fixture.flac",
        reference=TextFields(raw="manual reference", normalized="manual reference"),
        asr=ASRMetadata(
            system_id="fixture-asr",
            checkpoint="fixture",
            code_revision="fixture",
            hypotheses=(
                Hypothesis(
                    rank=1,
                    text=TextFields(raw="first asr hypothesis", normalized="first asr hypothesis"),
                    source_system_id="fixture-asr",
                    source_original_rank=1,
                ),
                Hypothesis(
                    rank=2,
                    text=TextFields(
                        raw="second asr hypothesis", normalized="second asr hypothesis"
                    ),
                    source_system_id="fixture-asr",
                    source_original_rank=2,
                ),
            ),
        ),
        provenance=RecordProvenance(created_utc="2026-08-21T00:00:00Z", host="fixture"),
    )

    pair = build_ec_pair(record, serializer=LiteralSeparatorSerializer(), input_nbest=2)

    assert pair.source_text == "first asr hypothesis [SEP] second asr hypothesis"
    assert pair.target_raw == "manual reference"
    assert "manual reference" not in pair.source_text
    assert pair.input_nbest == 2
