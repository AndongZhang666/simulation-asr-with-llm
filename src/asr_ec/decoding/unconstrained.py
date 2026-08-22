"""Unconstrained supervised EC output representation."""

from __future__ import annotations

from asr_ec.contracts.outputs import ECOutput
from asr_ec.contracts.records import UtteranceRecord
from asr_ec.data.normalization import TextNormalizer


def unconstrained_output(
    record: UtteranceRecord,
    *,
    generated_text: str,
    normalizer: TextNormalizer,
    ec_system_id: str,
    ec_sequence_score: float | None = None,
    truncated: bool = False,
) -> ECOutput:
    """Represent a free EC generation without falsely claiming candidate membership."""

    from asr_ec.contracts.records import TextFields

    return ECOutput(
        utt_id=record.utt_id,
        ec_system_id=ec_system_id,
        decoding_strategy="unconstrained",
        output=TextFields(raw=generated_text, normalized=normalizer.normalize(generated_text)),
        ec_sequence_score=ec_sequence_score,
        truncated=truncated,
    )
