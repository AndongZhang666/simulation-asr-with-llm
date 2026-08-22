import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from asr_ec.contracts.outputs import ECOutput, LLMExchange, LLMMessage, ParserStatus
from asr_ec.contracts.records import (
    ASRMetadata,
    Hypothesis,
    RecordProvenance,
    TextFields,
    UtteranceRecord,
)

SCHEMA_DIRECTORY = Path("src/asr_ec/contracts/schemas")


def make_record() -> UtteranceRecord:
    hypothesis = Hypothesis(
        rank=1,
        text=TextFields(raw="candidate", normalized="candidate"),
        source_system_id="fixture-asr",
        source_original_rank=1,
    )
    return UtteranceRecord(
        utt_id="fixture-1",
        dataset="fixture",
        split="test",
        audio_path="fixture.wav",
        reference=TextFields(raw="reference", normalized="reference"),
        asr=ASRMetadata(
            system_id="fixture-asr",
            checkpoint="fixture@1",
            code_revision="fixture-code@1",
            hypotheses=(hypothesis,),
        ),
        provenance=RecordProvenance(created_utc="2026-08-21T00:00:00Z", host="fixture"),
    )


@pytest.mark.parametrize(
    ("schema_name", "instance"),
    [
        ("utterance_record.schema.json", lambda: make_record().to_dict()),
        (
            "ec_output.schema.json",
            lambda: ECOutput(
                utt_id="fixture-1",
                ec_system_id="fixture-ec",
                decoding_strategy="unconstrained",
                output=TextFields(raw="output", normalized="output"),
            ).to_dict(),
        ),
        (
            "llm_exchange.schema.json",
            lambda: LLMExchange(
                request_hash="a" * 64,
                provider="fixture-provider",
                model_id="fixture-model@1",
                messages=(LLMMessage(role="user", content="correct this"),),
                generation_parameters={"temperature": 0},
                requested_utc="2026-08-21T00:00:00Z",
                parser_status=ParserStatus.PARSED,
                raw_response="corrected",
                parsed_output=TextFields(raw="corrected", normalized="corrected"),
            ).to_dict(),
        ),
    ],
)
def test_contract_fixture_validates_against_json_schema(schema_name: str, instance: object) -> None:
    schema = json.loads((SCHEMA_DIRECTORY / schema_name).read_text(encoding="utf-8"))

    Draft202012Validator(schema).validate(instance())


def test_ec_output_rejects_invalid_lambda() -> None:
    with pytest.raises(ValueError, match="lambda_value"):
        ECOutput(
            utt_id="fixture-1",
            ec_system_id="fixture-ec",
            decoding_strategy="constrained",
            output=TextFields(raw="output", normalized="output"),
            lambda_value=1.01,
        )
