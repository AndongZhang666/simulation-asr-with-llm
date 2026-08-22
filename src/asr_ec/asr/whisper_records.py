"""Transform finalized Whisper candidates into versioned ASR interchange records."""

from __future__ import annotations

import hashlib
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from asr_ec.contracts.records import (
    ASRMetadata,
    Hypothesis,
    RecordProvenance,
    ScoreType,
    TextFields,
    UtteranceRecord,
)
from asr_ec.data.manifests import DataManifestRecord
from asr_ec.data.normalization import TextNormalizer

from .whisper_nbest import NBestDecodingResult


class NBestArtifactError(ValueError):
    """Raised when an immutable Whisper N-best artifact cannot be built faithfully."""


@dataclass(frozen=True, slots=True)
class NBestArtifact:
    """Content-addressed JSONL record set produced by a fixed ASR configuration."""

    sha256: str
    path: Path
    record_count: int


def build_utterance_record(
    manifest_record: DataManifestRecord,
    decoded: NBestDecodingResult,
    *,
    normalizer: TextNormalizer,
    checkpoint: str,
    code_revision: str,
    decode_config_sha256: str,
    length_penalty: float | None,
    software_lock_sha256: str | None,
) -> UtteranceRecord:
    """Preserve raw candidates and their exact decoder score/ranking semantics."""

    system_id = "whisper-small.en"
    hypotheses = tuple(
        Hypothesis(
            rank=candidate.rank,
            text=TextFields(
                raw=candidate.text,
                normalized=normalizer.normalize(candidate.text),
            ),
            source_system_id=system_id,
            source_original_rank=candidate.rank,
            token_ids=candidate.token_ids,
            token_logprobs=candidate.token_logprobs,
            sequence_logscore=candidate.sequence_logscore,
            score_type=ScoreType.SUM_LOGPROB,
            length_penalty=length_penalty,
            metadata={
                "ranking_score": candidate.ranking_score,
                "ranking_score_semantics": (
                    "sum_logprob divided by Whisper MaximumLikelihoodRanker penalty"
                ),
                "token_logprobs_available": candidate.token_logprobs_available,
                "no_speech_prob": decoded.no_speech_prob,
            },
        )
        for candidate in decoded.candidates
    )
    return UtteranceRecord(
        utt_id=manifest_record.utt_id,
        dataset=manifest_record.dataset,
        split=manifest_record.split,
        audio_path=manifest_record.audio_path,
        reference=TextFields(
            raw=manifest_record.reference_raw,
            normalized=manifest_record.reference_normalized,
        ),
        asr=ASRMetadata(
            system_id=system_id,
            checkpoint=checkpoint,
            code_revision=code_revision,
            decode_config_sha256=decode_config_sha256,
            hypotheses=hypotheses,
        ),
        provenance=RecordProvenance(
            created_utc=datetime.now(timezone.utc).isoformat(),
            host=socket.gethostname(),
            software_lock_sha256=software_lock_sha256,
            normalizer_id=normalizer.normalizer_id,
        ),
    )


def write_nbest_artifact_once(
    records: Sequence[UtteranceRecord], *, output_root: Path
) -> NBestArtifact:
    """Publish records to a stable content-addressed JSONL path without mutation."""

    if not records:
        raise NBestArtifactError("refusing to publish an empty N-best artifact")
    utt_ids = [record.utt_id for record in records]
    if len(utt_ids) != len(set(utt_ids)):
        raise NBestArtifactError("N-best artifact contains duplicate utterance IDs")
    payload = (
        "\n".join(record.to_json() for record in sorted(records, key=lambda item: item.utt_id))
        + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    destination = output_root / "whisper-small-en" / digest / "records.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != payload:
            raise NBestArtifactError("existing N-best path has unexpected content")
    else:
        destination.write_bytes(payload)
    return NBestArtifact(sha256=digest, path=destination, record_count=len(records))


def nbest_artifact_sha256(path: Path) -> str:
    """Hash an already-published JSONL artifact for run completion provenance."""

    return hashlib.sha256(path.read_bytes()).hexdigest()
