"""Evaluate stored N-best artifacts without invoking ASR or model inference."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from asr_ec.contracts.records import UtteranceRecord
from asr_ec.data.normalization import normalizer_from_id
from asr_ec.evaluation.nbest import NBestEvaluation, NBestEvaluationError, evaluate_nbest
from asr_ec.tracking.run_manifest import create_run_manifest


@dataclass(frozen=True, slots=True)
class NBestEvaluationResult:
    evaluation: NBestEvaluation
    run_directory: Path | None
    dry_run: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "dry_run": self.dry_run,
            "run_directory": self.run_directory.as_posix() if self.run_directory else None,
            "metrics": self.evaluation.to_dict(),
        }


def run_evaluate_nbest(
    artifact_path: Path, *, runs_root: Path = Path("runs"), dry_run: bool = False
) -> NBestEvaluationResult:
    """Load an immutable JSONL artifact and score its saved raw text reproducibly."""

    if not artifact_path.is_file():
        raise NBestEvaluationError(f"N-best artifact does not exist: {artifact_path}")
    records = tuple(
        UtteranceRecord.from_json(line)
        for line in artifact_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not records:
        raise NBestEvaluationError("N-best artifact contains no records")
    normalizer_ids = {record.provenance.normalizer_id for record in records}
    if len(normalizer_ids) != 1 or None in normalizer_ids:
        raise NBestEvaluationError("N-best artifact must contain one known normalizer_id")
    normalizer_id = normalizer_ids.pop()
    assert normalizer_id is not None
    requested_n = len(records[0].asr.hypotheses)
    evaluation = evaluate_nbest(
        records,
        normalizer=normalizer_from_id(normalizer_id),
        requested_n=requested_n,
    )
    if dry_run:
        return NBestEvaluationResult(evaluation=evaluation, run_directory=None, dry_run=True)
    artifact_hash = _file_sha256(artifact_path)
    run_directory, _ = create_run_manifest(
        runs_root,
        prefix="evaluate-nbest",
        resolved_config={
            "artifact_path": artifact_path.as_posix(),
            "artifact_sha256": artifact_hash,
            "normalizer_id": normalizer_id,
            "requested_n": requested_n,
        },
        command=("asr-ec", "evaluate-nbest", "--artifact", str(artifact_path)),
        input_artifact_hashes={"nbest": artifact_hash},
        retry_on_collision=True,
    )
    metrics_path = run_directory / "metrics.json"
    metrics_path.write_text(
        json.dumps(evaluation.to_dict(), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return NBestEvaluationResult(evaluation=evaluation, run_directory=run_directory, dry_run=False)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
