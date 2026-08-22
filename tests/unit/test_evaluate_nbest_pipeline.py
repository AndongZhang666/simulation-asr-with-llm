import json
from pathlib import Path

from asr_ec.contracts.records import (
    ASRMetadata,
    Hypothesis,
    RecordProvenance,
    TextFields,
    UtteranceRecord,
)
from asr_ec.pipelines.evaluate_nbest import run_evaluate_nbest


def make_record() -> UtteranceRecord:
    return UtteranceRecord(
        utt_id="fixture-1",
        dataset="fixture",
        split="test",
        audio_path="fixture.flac",
        reference=TextFields(raw="a reference", normalized="a reference"),
        asr=ASRMetadata(
            system_id="fixture-asr",
            checkpoint="fixture",
            code_revision="fixture",
            hypotheses=(
                Hypothesis(
                    rank=1,
                    text=TextFields(raw="a reference", normalized="a reference"),
                    source_system_id="fixture-asr",
                    source_original_rank=1,
                    sequence_logscore=-1.0,
                ),
            ),
        ),
        provenance=RecordProvenance(
            created_utc="2026-08-21T00:00:00Z",
            host="fixture",
            normalizer_id="identity@v1",
        ),
    )


def test_evaluate_nbest_writes_immutable_metrics_and_supports_dry_run(tmp_path: Path) -> None:
    artifact = tmp_path / "records.jsonl"
    artifact.write_text(make_record().to_json() + "\n", encoding="utf-8")

    dry_result = run_evaluate_nbest(artifact, runs_root=tmp_path / "runs", dry_run=True)
    persisted_result = run_evaluate_nbest(artifact, runs_root=tmp_path / "runs")

    assert dry_result.run_directory is None
    assert persisted_result.run_directory is not None
    metrics = json.loads((persisted_result.run_directory / "metrics.json").read_text())
    assert metrics["baseline"]["error_rate"] == 0
    assert (persisted_result.run_directory / "run_manifest.json").is_file()
